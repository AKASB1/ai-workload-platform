"""Two drivers for the controller core.

- `VirtualDriver`: jumps the clock to the earlier of the controller's next deadline and the
  backend's next event (never polls an idle system); ticks once at time 0 so an inventory is stored;
  feeds the submissions of a trace through the same admission function the API calls. At one
  instant: completions (observed in the tick), then submissions, then one scheduling cycle.
- `LiveDriver`: asyncio; the core runs in a worker thread so the event loop is never blocked; it
  ticks at least every `observe_interval_ms` (wall clock) and earlier when a deadline or a backend
  event is due, or when the API kicks it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ai_workload_platform import admission
from ai_workload_platform.clock import SystemClock, VirtualClock
from ai_workload_platform.controller import Controller
from ai_workload_platform.crash import Crash
from ai_workload_platform.models import LeaseLost, PlatformError, StoreUnavailable
from ai_workload_platform.store import ops
from ai_workload_platform.store.sql import TERMINAL_SQL, Store

log = logging.getLogger("awp.driver")


@dataclass(frozen=True)
class Submission:
    """A client action at `at_ms`: a submission, or (with `cancel` = workload id) a cancel request."""

    at_ms: int
    namespace: str
    spec: dict[str, Any] | None
    key: str | None = None
    cancel: str | None = None


@dataclass
class RunStats:
    ticks: int = 0
    end_ms: int = 0
    submitted: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    deadlock: bool = False
    crashes: int = 0
    wall_s: float = 0.0


def live_count(store: Store) -> int:
    with store.read() as tx:
        row = tx.one(f"SELECT COUNT(*) FROM workloads WHERE state NOT IN {TERMINAL_SQL}")
    return int(row[0]) if row else 0


class VirtualDriver:
    def __init__(
        self,
        store: Store,
        backend: Any,
        controller: Controller,
        clock: VirtualClock,
        submissions: list[Submission],
        *,
        max_ms: int | None = None,
        max_ticks: int = 2_000_000,
        metrics: Any = None,
        restart: Callable[[], Controller] | None = None,
    ) -> None:
        self.store = store
        self.backend = backend
        self.controller = controller
        self.clock = clock
        self.pending = deque(sorted(submissions, key=lambda s: s.at_ms))  # stable: trace order at equal times
        self.max_ms = max_ms
        self.max_ticks = max_ticks
        self.metrics = metrics
        self.restart = restart
        self.stats = RunStats()

    def _submit_due(self, now: int) -> None:
        while self.pending and self.pending[0].at_ms <= now:
            s = self.pending.popleft()
            try:
                if s.cancel is not None:
                    ops.request_cancel(self.store, s.namespace, s.cancel, now)
                    continue
                admission.submit(self.store, s.namespace, s.spec, s.key, now, metrics=self.metrics)
                self.stats.submitted += 1
            except PlatformError as e:
                self.stats.rejected[e.code] = self.stats.rejected.get(e.code, 0) + 1

    def next_time(self, now: int) -> int | None:
        cands = [
            t
            for t in (
                self.controller.next_wakeup_ms(),
                self.backend.next_event_ms(),
                self.pending[0].at_ms if self.pending else None,
            )
            if t is not None
        ]
        if not cands:
            return None
        return max(min(cands), now + 1)

    def run(self) -> RunStats:
        t0 = time.perf_counter()
        while True:
            now = self.clock.now_ms()
            try:
                self.controller.tick(now, pre_cycle=lambda now=now: self._submit_due(now))
            except Crash:
                if self.restart is None:
                    raise
                self.stats.crashes += 1
                self.controller.close()
                self.controller = self.restart()  # a new controller with no memory, same store and backend
                continue
            self.stats.ticks += 1
            if not self.pending and live_count(self.store) == 0:
                break
            nxt = self.next_time(now)
            if nxt is None:
                self.stats.deadlock = True
                break
            if (self.max_ms is not None and nxt > self.max_ms) or self.stats.ticks >= self.max_ticks:
                break
            self.clock.advance_to(nxt)
        self.stats.end_ms = self.clock.now_ms()
        self.stats.wall_s = time.perf_counter() - t0
        return self.stats


class LiveDriver:
    def __init__(
        self,
        make_controller: Callable[[], Controller],
        clock: SystemClock,
        backend: Any,
        *,
        observe_interval_ms: int = 1000,
    ) -> None:
        self.make_controller = make_controller
        self.clock = clock
        self.backend = backend
        self.observe_interval_ms = observe_interval_ms
        self.controller: Controller | None = None
        self.kick_event: asyncio.Event | None = None
        self.stop_event: asyncio.Event | None = None
        self.last_error: str | None = None
        self.last_backend_error = False
        self.ticks = 0

    def kick(self) -> None:
        """Wake the loop early (called from the API's worker threads: thread-safe)."""
        loop, ev = getattr(self, "loop", None), self.kick_event
        if loop is not None and ev is not None and not loop.is_closed():
            loop.call_soon_threadsafe(ev.set)

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.kick_event = asyncio.Event()
        self.stop_event = asyncio.Event()
        self.controller = await asyncio.to_thread(self.make_controller)
        while not self.stop_event.is_set():
            ctl = self.controller
            now = self.clock.now_ms()
            try:
                res = await asyncio.to_thread(ctl.tick, now)
                self.last_backend_error = res.leader and not (res.inventory_ok and res.observe_ok)
                await asyncio.to_thread(ctl.update_gauges)
                self.last_error = None
            except LeaseLost as e:
                log.warning("lease lost; starting a fresh controller", extra={"fields": {"error": str(e)}})
                await asyncio.to_thread(ctl.close)
                self.controller = await asyncio.to_thread(self.make_controller)
            except StoreUnavailable as e:
                self.last_error = f"store unavailable: {e}"
                log.warning("store unavailable", extra={"fields": {"error": str(e)}})
            except Exception as e:  # noqa: BLE001 - the loop must survive anything a tick raises
                self.last_error = f"{type(e).__name__}: {e}"
                log.exception("tick failed")
            self.ticks += 1
            wait_s = self.observe_interval_ms / 1000.0
            now = self.clock.now_ms()
            for t in (self.controller.next_wakeup_ms(), self.backend.next_event_ms()):
                if t is not None:
                    wait_s = min(wait_s, max(0.0, (t - now) / 1000.0 / self.clock.scale))
            self.kick_event.clear()
            try:
                await asyncio.wait_for(self.kick_event.wait(), timeout=max(0.005, wait_s))
            except TimeoutError:
                pass

    async def stop(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
            if self.kick_event is not None:
                self.kick_event.set()
        if self.controller is not None:
            await asyncio.to_thread(self.controller.close)
