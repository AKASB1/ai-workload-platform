"""Deterministic local backend: a model of the nodes on the injected clock (docs/contracts.md §7).

An attempt runs `sim.runtime_s` of work at `rate = min(speed) / f` and finishes
`start_latency + ceil(work * 1000 / rate - 1e-6)` ms after its start, or fails as `sim` says.
The backend does not enforce capacity: an over-allocation by the platform shows up in its own
records (`usage()`), where invariant I3 looks for it. Fault hooks are driven by the harness.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ai_workload_platform.clock import Clock
from ai_workload_platform.cluster import ClusterConfig, rate_of
from ai_workload_platform.models import (
    AttemptRequest,
    AttemptStatus,
    BackendError,
    FailReason,
    NodeInfo,
    Phase,
    Snapshot,
)
from ai_workload_platform.models.spec import mem_mb


@dataclass
class _Att:
    id: str
    nodes: tuple[tuple[str, int], ...]
    gpus: int
    cpus: int
    mem_mb: int
    start_ms: int
    run_ms: int  # becomes running
    end_ms: int  # planned end
    outcome_exit: int  # 0 = succeeded
    rate: float
    phase: Phase = Phase.STARTING
    started_ms: int | None = None
    ended_ms: int | None = None
    exit_code: int | None = None
    reason: FailReason | None = None
    stop_at_ms: int | None = None  # stop requested: stopped at this time (unless the node is down)
    frozen: bool = False  # on a down node: no progress, last phase reported
    overhead_ms: int = 0  # restart overhead after a preemption (held, no progress)
    retained_s: float = 0.0


@dataclass
class LocalFaults:
    """Fault switches set by the harness (all off in production)."""

    observe_down: bool = False
    stale_age_ms: int = 0  # > 0: observe returns a snapshot at least this old (within A1)
    fail_start: int = 0  # the next n start calls fail
    fail_start_effect: bool = False  # ... after taking effect (the reply was lost)
    fail_stop: int = 0
    fail_stop_effect: bool = False
    fail_inventory: bool = False
    slow_start_ms: dict[str, int] = field(default_factory=dict)  # attempt id -> extra start latency
    calls: dict[str, int] = field(default_factory=dict)


class LocalBackend:
    name = "local"

    def __init__(
        self, cluster: ClusterConfig, clock: Clock, *, start_latency_ms: int = 0, stop_latency_ms: int = 0
    ) -> None:
        self.cluster = cluster
        self.clock = clock
        self.start_latency_ms = int(start_latency_ms)
        self.stop_latency_ms = int(stop_latency_ms)
        self.nodes: dict[str, NodeInfo] = {n.name: n for n in cluster.nodes}
        self.down_since: dict[str, int] = {}
        self.atts: dict[str, _Att] = {}
        self.faults = LocalFaults()
        self._history: deque[Snapshot] = deque(maxlen=256)

    # --- contract ------------------------------------------------------------------------------
    def inventory(self) -> list[NodeInfo]:
        self._count("inventory")
        if self.faults.fail_inventory:
            raise BackendError("inventory unavailable (injected)")
        return [
            NodeInfo(
                n.name, n.rack, n.gpu_class, n.speed, n.gpus, n.cpus, n.mem_mb, n.name not in self.down_since
            )
            for n in sorted(self.nodes.values(), key=lambda x: (x.rack, x.name))
        ]

    def start(self, req: AttemptRequest) -> None:
        self._count("start")
        if self.faults.fail_start > 0:
            self.faults.fail_start -= 1
            if self.faults.fail_start_effect and req.attempt_id not in self.atts:
                self._create(req)  # the call took effect, but the reply was lost
            raise BackendError("start failed (injected)")
        if req.attempt_id not in self.atts:
            self._create(req)

    def _create(self, req: AttemptRequest) -> None:
        now = self.clock.now_ms()
        spec = req.spec
        nodes = [self.nodes[n] for n, _ in req.placement]
        rate = rate_of(
            nodes, spec["topology"], self.cluster.cross_node_factor, self.cluster.cross_rack_factor
        )
        sim = spec["sim"]
        retained = 0.0
        if req.n <= int(sim.get("fail_attempts") or 0):
            work, code = float(sim["fail_after_s"]), int(sim.get("exit_code") or 1)
        else:
            retained = min(float(req.retained_s), float(sim["runtime_s"]))
            work, code = float(sim["runtime_s"]) - retained, 0
        latency = self.start_latency_ms + self.faults.slow_start_ms.get(req.attempt_id, 0)
        run_ms = now + latency
        overhead_ms = int(round(float(req.restart_overhead_s) * 1000))
        end_ms = run_ms + overhead_ms + int(math.ceil(work * 1000.0 / rate - 1e-6))
        self.atts[req.attempt_id] = _Att(
            req.attempt_id,
            tuple(req.placement),
            int(spec["gpus"]),
            int(spec["cpus"]),
            mem_mb(spec["mem_gb"]),
            now,
            run_ms,
            end_ms,
            code,
            rate,
            overhead_ms=overhead_ms,
            retained_s=retained,
        )
        # an attempt started on a node that is down freezes at once
        if any(n in self.down_since for n, _ in req.placement):
            self.atts[req.attempt_id].frozen = True

    def stop(self, attempt_id: str) -> None:
        self._count("stop")
        if self.faults.fail_stop > 0:
            self.faults.fail_stop -= 1
            if not self.faults.fail_stop_effect:
                raise BackendError("stop failed (injected)")
            self._stop(attempt_id)
            raise BackendError("stop failed after taking effect (injected)")
        self._stop(attempt_id)

    def _stop(self, attempt_id: str) -> None:
        a = self.atts.get(attempt_id)
        if a is None:
            return
        self._advance(self.clock.now_ms())
        if a.phase in (Phase.SUCCEEDED, Phase.FAILED, Phase.STOPPED) or a.stop_at_ms is not None:
            return
        a.stop_at_ms = self.clock.now_ms() + self.stop_latency_ms

    def observe(self) -> Snapshot:
        self._count("observe")
        if self.faults.observe_down:
            raise BackendError("observe unavailable (injected)")
        now = self.clock.now_ms()
        self._advance(now)
        snap = Snapshot(now, tuple(self._status(a) for a in sorted(self.atts.values(), key=lambda x: x.id)))
        self._history.append(snap)
        if self.faults.stale_age_ms > 0:
            old = [s for s in self._history if s.taken_ms <= now - self.faults.stale_age_ms]
            if old:
                return old[-1]
        return snap

    def forget(self, attempt_id: str) -> None:
        self._count("forget")
        self.atts.pop(attempt_id, None)

    def next_event_ms(self) -> int | None:
        now = self.clock.now_ms()
        best: int | None = None
        for a in self.atts.values():
            if a.frozen or a.phase in (Phase.SUCCEEDED, Phase.FAILED, Phase.STOPPED):
                continue
            for t in (a.run_ms if a.phase == Phase.STARTING else None, a.end_ms, a.stop_at_ms):
                if t is not None and t > now and (best is None or t < best):
                    best = t
        return best

    # --- model ---------------------------------------------------------------------------------
    def _advance(self, now: int) -> None:
        for a in self.atts.values():
            if a.frozen or a.phase in (Phase.SUCCEEDED, Phase.FAILED, Phase.STOPPED):
                continue
            # events in time order: running, then the earlier of the planned end and the stop
            if (
                a.phase == Phase.STARTING
                and a.run_ms <= now
                and (a.stop_at_ms is None or a.run_ms < a.stop_at_ms)
            ):
                a.phase = Phase.RUNNING
                a.started_ms = a.run_ms
            if (
                a.stop_at_ms is not None
                and a.stop_at_ms <= now
                and (a.phase == Phase.STARTING or a.stop_at_ms < a.end_ms)
            ):
                a.phase = Phase.STOPPED
                a.ended_ms = a.stop_at_ms
                continue
            if a.phase == Phase.RUNNING and a.end_ms <= now:
                a.ended_ms = a.end_ms
                a.exit_code = a.outcome_exit
                if a.outcome_exit == 0:
                    a.phase = Phase.SUCCEEDED
                else:
                    a.phase = Phase.FAILED
                    a.reason = FailReason.EXIT

    def _status(self, a: _Att) -> AttemptStatus:
        terminal = a.phase in (Phase.SUCCEEDED, Phase.FAILED, Phase.STOPPED)
        return AttemptStatus(
            attempt_id=a.id,
            phase=a.phase,
            nodes=tuple(n for n, _ in a.nodes),
            exit_code=a.exit_code if terminal else None,
            reason=a.reason if a.phase == Phase.FAILED else None,
            started_ms=a.started_ms,
            ended_ms=a.ended_ms if terminal else None,
            rate=a.rate,
            workers_by_node=a.nodes,
            work_done_s=self.work_done(a),
        )

    def work_done(self, a: _Att) -> float:
        """Retained work plus this run's progress (after the restart overhead), up to now or the end."""
        until = a.ended_ms if a.ended_ms is not None else self.clock.now_ms()
        progress_ms = (
            max(0, min(until, a.end_ms) - a.run_ms - a.overhead_ms) if a.started_ms is not None else 0
        )
        return round(a.retained_s + progress_ms / 1000.0 * a.rate, 6)

    def _count(self, op: str) -> None:
        self.faults.calls[op] = self.faults.calls.get(op, 0) + 1

    # --- fault hooks (harness) -----------------------------------------------------------------
    def node_down(self, name: str) -> None:
        now = self.clock.now_ms()
        self._advance(now)
        if name in self.down_since:
            return
        self.down_since[name] = now
        for a in self.atts.values():
            if a.phase in (Phase.STARTING, Phase.RUNNING) and any(n == name for n, _ in a.nodes):
                a.frozen = True

    def node_up(self, name: str) -> None:
        now = self.clock.now_ms()
        if name not in self.down_since:
            return
        del self.down_since[name]
        for a in self.atts.values():
            if a.frozen and not any(n in self.down_since for n, _ in a.nodes):
                a.frozen = False
                if a.stop_at_ms is not None:
                    a.phase, a.ended_ms = Phase.STOPPED, now
                else:
                    a.phase, a.ended_ms, a.exit_code, a.reason = Phase.FAILED, now, None, FailReason.NODE_LOST

    def crash(self, attempt_id: str, exit_code: int = 137) -> bool:
        """The attempt's process dies now (reported failed, reason exit)."""
        now = self.clock.now_ms()
        self._advance(now)
        a = self.atts.get(attempt_id)
        if a is None or a.frozen or a.phase not in (Phase.STARTING, Phase.RUNNING):
            return False
        a.phase, a.ended_ms, a.exit_code, a.reason = Phase.FAILED, now, exit_code, FailReason.EXIT
        return True

    def lose(self, attempt_id: str) -> bool:
        """The backend loses its record of the attempt (and the attempt with it)."""
        return self.atts.pop(attempt_id, None) is not None

    # --- ground truth for the invariants -------------------------------------------------------
    def usage(self) -> dict[str, tuple[int, int, int]]:
        """Resources held per node in the backend's own records (non-terminal attempts)."""
        self._advance(self.clock.now_ms())
        out: dict[str, list[int]] = {}
        for a in self.atts.values():
            if a.phase in (Phase.SUCCEEDED, Phase.FAILED, Phase.STOPPED):
                continue
            for n, k in a.nodes:
                u = out.setdefault(n, [0, 0, 0])
                u[0] += a.gpus * k
                u[1] += a.cpus * k
                u[2] += a.mem_mb * k
        return {n: (v[0], v[1], v[2]) for n, v in out.items()}

    def active_ids(self) -> set[str]:
        self._advance(self.clock.now_ms())
        return {
            a.id for a in self.atts.values() if a.phase not in (Phase.SUCCEEDED, Phase.FAILED, Phase.STOPPED)
        }

    def known_ids(self) -> set[str]:
        return set(self.atts)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "local",
            "start_latency_ms": self.start_latency_ms,
            "stop_latency_ms": self.stop_latency_ms,
        }
