"""Invariants I1-I8, checked after every step of a schedule (docs/failure-injection.md).

The event log is replayed incrementally (the same pure fold as a replay from seq 1, applied to the
new events only) and compared with the tables; the backend's own records are the ground truth for I3.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from ai_workload_platform.lifecycle import retry_at_ms
from ai_workload_platform.models import (
    COUNTED_REASONS,
    TERMINAL_STATES,
    AttemptState,
    EndReason,
    WorkloadState,
)
from ai_workload_platform.store.replay import ReplayError, ReplayState, apply, diff
from ai_workload_platform.store.sql import Store

TERMINAL_VALUES = {s.value for s in TERMINAL_STATES}


@dataclass(frozen=True)
class Violation:
    invariant: str
    at_ms: int
    message: str

    def to_json(self) -> dict[str, Any]:
        return {"invariant": self.invariant, "at_ms": self.at_ms, "message": self.message}


class Checker:
    def __init__(
        self,
        store: Store,
        ground_truth: Any,
        kind: str,
        cluster: Any,
        caps: dict[str, int],
        lost_grace_ms: int,
        seed: int,
    ) -> None:
        self.store = store
        self.gt = ground_truth
        self.kind = kind
        self.cluster = cluster
        self.cap_history: dict[str, list[tuple[int, int]]] = {ns: [(0, c)] for ns, c in caps.items()}
        self.lost_grace_ms = lost_grace_ms
        self.seed = seed
        self.replay = ReplayState()
        self.terminal_events: Counter[str] = Counter()
        self.terminal_state: dict[str, str] = {}
        self.orphan_since: dict[str, int] = {}
        self.keys: Counter[tuple[str, str]] = Counter()
        self.max_epoch = 0
        self.broken = False

    def set_cap(self, ns: str, cap: int, at_ms: int) -> None:
        self.cap_history.setdefault(ns, []).append((at_ms, cap))

    def _cap_at(self, ns: str, at_ms: int) -> int:
        cap = 10**9
        for t, c in self.cap_history.get(ns, []):
            if t <= at_ms:
                cap = c
        return cap

    def check(self, now_ms: int) -> list[Violation]:
        out: list[Violation] = []

        def v(inv: str, msg: str) -> None:
            out.append(Violation(inv, now_ms, msg))

        st = self.replay
        if not self.broken:
            for ev in self.store.events(after=st.last_seq):
                if ev.type == "started":
                    w = st.workloads.get(ev.workload_id)
                    if w is not None:
                        alloc = sum(
                            g for (ns, _n), (g, _c, _m) in st.books_view().items() if ns == w.namespace
                        )
                        cap = self._cap_at(w.namespace, ev.at_ms)
                        if alloc + w.gpus * w.workers > cap:
                            v(
                                "I4",
                                f"seq {ev.seq}: start of {w.id} takes {w.namespace} to "
                                f"{alloc + w.gpus * w.workers} GPUs above its cap {cap}",
                            )
                try:
                    apply(st, ev)
                except ReplayError as e:
                    v("I5", f"replay failed: {e}")
                    self.broken = True
                    break
                if ev.type == "submitted" and ev.data.get("idempotency_key"):
                    k = (ev.namespace, ev.data["idempotency_key"])
                    self.keys[k] += 1
                    if self.keys[k] > 1:
                        v("I8", f"key {k[1]} in {k[0]} created a second workload ({ev.workload_id})")
                ep = ev.data.get("epoch")
                if ep is not None:
                    if ep < self.max_epoch:
                        v(
                            "I5",
                            f"seq {ev.seq}: written under lease epoch {ep} after epoch {self.max_epoch} "
                            f"(a controller that lost the lease wrote)",
                        )
                    self.max_epoch = max(self.max_epoch, ep)
                state = ev.data.get("state")
                if state in TERMINAL_VALUES and ev.type in ("attempt_ended", "cancel_requested"):
                    self.terminal_events[ev.workload_id] += 1
                    if self.terminal_events[ev.workload_id] > 1:
                        v("I1", f"{ev.workload_id} has a second terminal event (seq {ev.seq})")
                if ev.type == "attempt_ended":
                    self._i6(ev, v)
        ws = self.store.all_workloads()
        ats = self.store.all_attempts()
        books = self.store.books()
        if not self.broken:
            d = diff(st, ws, ats, books)
            if d:
                v("I5", "replay differs from the tables: " + "; ".join(d)[:600])
        # I1: a terminal state is never left
        for w in ws:
            prev = self.terminal_state.get(w.id)
            if prev is not None and w.state.value != prev:
                v("I1", f"{w.id} left terminal state {prev} for {w.state.value}")
            elif w.state in TERMINAL_STATES:
                self.terminal_state[w.id] = w.state.value
        # I2: at most one attempt that is not ENDED; none for terminal, QUEUED, RETRY_WAIT
        open_by: Counter[str] = Counter(a.workload_id for a in ats if a.state != AttemptState.ENDED)
        for w in ws:
            n = open_by.get(w.id, 0)
            if n > 1:
                v("I2", f"{w.id} has {n} attempts that are not ENDED")
            if n and (
                w.state in TERMINAL_STATES or w.state in (WorkloadState.QUEUED, WorkloadState.RETRY_WAIT)
            ):
                v("I2", f"{w.id} is {w.state.value} with an attempt that is not ENDED")
        # I3 (a): the backend's own records never exceed a node
        usage = self.gt.usage()
        caps = {n.name: n for n in self.cluster.nodes}
        for node, (g, c, m) in sorted(usage.items()):
            cap = caps[node]
            if g > cap.gpus or (self.kind == "local" and (c > cap.cpus or m > cap.mem_mb)):
                v(
                    "I3",
                    f"backend over-allocates {node}: {g}/{cap.gpus} GPUs, {c}/{cap.cpus} CPUs, {m}/{cap.mem_mb} MB",
                )
        # I3 (b): the books equal the sum over the attempts that are not ENDED
        expect: Counter[tuple[str, str]] = Counter()
        for a in ats:
            if a.state != AttemptState.ENDED:
                for p in a.placement:
                    expect[(a.namespace, p["node"])] += a.gpus * int(p["workers"])
        got = {k: g for k, (g, _c, _m) in books.items() if g}
        if {k: x for k, x in expect.items() if x} != got:
            v("I3", f"books {got} != sum over attempts not ENDED {dict(expect)}")
        # I3 (c): every attempt the backend still runs has a store attempt that is not ENDED. Checked
        # strictly for attempts the store knows (it ends one only on the backend's terminal report or after
        # lost_grace_ms of absence under A1, so an ENDED attempt the backend still runs means the books freed
        # what the backend uses); attempts the store never knew get R3's grace of lost_grace_ms.
        live = {a.id for a in ats if a.state != AttemptState.ENDED}
        ended = {a.id for a in ats if a.state == AttemptState.ENDED}
        running = self.gt.active_ids() if self.kind == "local" else self.gt.active_attempts()
        for aid in sorted(running):
            if aid in live:
                self.orphan_since.pop(aid, None)
                continue
            if aid in ended:
                v("I3", f"the backend still runs {aid}, which the store has ENDED (its resources were freed)")
                continue
            since = self.orphan_since.setdefault(aid, now_ms)
            if now_ms - since > self.lost_grace_ms:
                v("I3", f"the backend still runs {aid}, unknown to the store, for {now_ms - since} ms")
        for aid in list(self.orphan_since):
            if aid not in running:
                del self.orphan_since[aid]
        return out

    def _i6(self, ev: Any, v: Any) -> None:
        w = self.replay.workloads.get(ev.workload_id)
        if w is None:
            return
        retry = w.spec["retry"]
        maxa = int(retry["max_attempts"])
        if w.counted > maxa:
            v("I6", f"{w.id} has {w.counted} counted attempts > max_attempts {maxa}")
        if ev.data.get("state") == "DEAD_LETTER" and (
            w.counted != maxa or EndReason(ev.data["reason"]) not in COUNTED_REASONS
        ):
            v(
                "I6",
                f"{w.id} is DEAD_LETTER with {w.counted} counted attempts (max {maxa}), last {ev.data['reason']}",
            )
        r_at = ev.data.get("retry_at_ms")
        if r_at is not None:
            exp = retry_at_ms(ev.at_ms, w.counted, retry, w.id, self.seed)
            if r_at != exp:
                v("I6", f"{w.id}: retry_at_ms {r_at} != formula {exp}")
            if not 0 <= r_at - ev.at_ms <= int(float(retry["backoff_cap_s"]) * 1000) + 1:
                v("I6", f"{w.id}: retry delay {r_at - ev.at_ms} ms outside [0, backoff_cap_s]")

    def final(self, now_ms: int, backend_known: set[str]) -> list[Violation]:
        """I7 at the end of the settle period: everything terminal and nothing leaked either way."""
        out = []
        ws = self.store.all_workloads()
        stuck = sorted(w.id + ":" + w.state.value for w in ws if w.state not in TERMINAL_STATES)
        if stuck:
            out.append(Violation("I7", now_ms, f"not converged: {stuck[:10]}"))
        open_atts = sorted(a.id for a in self.store.all_attempts() if a.state != AttemptState.ENDED)
        if open_atts:
            out.append(Violation("I7", now_ms, f"store attempts not ENDED: {open_atts[:10]}"))
        if backend_known:
            out.append(Violation("I7", now_ms, f"backend keeps attempts: {sorted(backend_known)[:10]}"))
        return out
