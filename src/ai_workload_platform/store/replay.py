"""Replay: a pure fold over the event log that rebuilds the workloads and attempts tables and the
namespace usage books. It is written independently of the store's write path (store/ops.py), so
comparing the two (invariant I5) checks the write path against the log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_workload_platform.lifecycle import attempt_after, workload_after
from ai_workload_platform.models import (
    COUNTED_REASONS,
    TERMINAL_STATES,
    Attempt,
    AttemptState,
    EndReason,
    Event,
    Workload,
    WorkloadState,
)
from ai_workload_platform.models.spec import mem_mb


class ReplayError(Exception):
    pass


@dataclass
class ReplayState:
    workloads: dict[str, Workload] = field(default_factory=dict)
    attempts: dict[str, Attempt] = field(default_factory=dict)
    books: dict[tuple[str, str], list[int]] = field(default_factory=dict)
    last_seq: int = 0
    last_at_ms: int = 0

    def books_view(self) -> dict[tuple[str, str], tuple[int, int, int]]:
        return {k: (v[0], v[1], v[2]) for k, v in sorted(self.books.items()) if tuple(v) != (0, 0, 0)}


def _book(st: ReplayState, a: Attempt, sign: int) -> None:
    for p in a.placement:
        k = int(p["workers"]) * sign
        cur = st.books.setdefault((a.namespace, p["node"]), [0, 0, 0])
        cur[0] += a.gpus * k
        cur[1] += a.cpus * k
        cur[2] += a.mem_mb * k


def apply(st: ReplayState, ev: Event) -> None:
    """Apply one event; raises ReplayError when the log is inconsistent."""
    if ev.seq != st.last_seq + 1:
        raise ReplayError(f"gap in the log: seq {ev.seq} after {st.last_seq}")
    if ev.at_ms < st.last_at_ms:
        raise ReplayError(f"at_ms decreases at seq {ev.seq}: {ev.at_ms} < {st.last_at_ms}")
    st.last_seq = ev.seq
    st.last_at_ms = ev.at_ms
    d: dict[str, Any] = ev.data
    t = ev.type
    at = ev.at_ms
    try:
        if t == "submitted":
            if ev.workload_id in st.workloads:
                raise ReplayError(f"seq {ev.seq}: workload {ev.workload_id} submitted twice")
            spec = d["spec"]
            st.workloads[ev.workload_id] = Workload(
                id=ev.workload_id,
                namespace=ev.namespace,
                spec=spec,
                spec_hash=d["spec_hash"],
                priority=int(spec["priority"]),
                gpus=int(spec["gpus"]),
                workers=int(spec["workers"]),
                cpus=int(spec["cpus"]),
                mem_mb=mem_mb(spec["mem_gb"]),
                submit_ms=at,
                submit_seq=ev.seq,
                state=workload_after(None, t),
                version=1,
                state_since_ms=at,
            )
            _check_state(ev, st.workloads[ev.workload_id].state)
            return
        w = st.workloads.get(ev.workload_id)
        if w is None:
            raise ReplayError(f"seq {ev.seq}: unknown workload {ev.workload_id}")
        old_state = w.state
        if t == "started":
            n = int(d["attempt"])
            if n != w.attempts + 1 or ev.attempt_id != f"{w.id}-a{n}":
                raise ReplayError(f"seq {ev.seq}: attempt number {n} does not follow {w.attempts}")
            if ev.attempt_id in st.attempts:
                raise ReplayError(f"seq {ev.seq}: attempt {ev.attempt_id} started twice")
            w.state = workload_after(w.state, t)
            w.attempts = n
            w.retry_at_ms = None
            if w.first_started_ms is None:
                w.first_started_ms = at
            a = Attempt(
                id=ev.attempt_id,
                workload_id=w.id,
                namespace=w.namespace,
                n=n,
                state=attempt_after(None, t),
                placement=[dict(p) for p in d["placement"]],
                gpus=w.gpus,
                cpus=w.cpus,
                mem_mb=w.mem_mb,
                version=1,
                started_ms=at,
                state_since_ms=at,
            )
            st.attempts[a.id] = a
            _book(st, a, +1)
        elif t == "running":
            a = _att(st, ev)
            w.state = workload_after(w.state, t)
            a.state = attempt_after(a.state, t)
            a.running_ms = at
            a.observed_started_ms = d.get("observed_started_ms")
            a.observed_nodes = list(d.get("nodes") or [])
            a.state_since_ms = at
            a.version += 1
        elif t == "stop_requested":
            a = _att(st, ev)
            w.state = workload_after(w.state, t)
            a.state = attempt_after(a.state, t)
            a.stop_reason = d["reason"]
            a.stop_requested_ms = at
            a.state_since_ms = at
            a.version += 1
        elif t == "attempt_ended":
            a = _att(st, ev)
            reason = d["reason"]
            counted = EndReason(reason) in COUNTED_REASONS
            if bool(d.get("counted")) != counted:
                raise ReplayError(f"seq {ev.seq}: counted flag disagrees with reason {reason}")
            w.counted += 1 if counted else 0
            w.state = workload_after(
                w.state,
                t,
                reason=reason,
                cancel_requested=w.cancel_requested,
                counted_after=w.counted,
                max_attempts=int(w.spec["retry"]["max_attempts"]),
            )
            a.state = attempt_after(a.state, t, reason)
            a.end_reason = reason
            a.exit_code = d.get("exit_code")
            a.counted = counted
            a.observed_started_ms = d.get("observed_started_ms")
            a.observed_ended_ms = d.get("observed_ended_ms")
            a.observed_nodes = list(d.get("nodes") or [])
            a.ended_ms = at
            a.state_since_ms = at
            a.version += 1
            if reason == "preempted":
                w.preemptions += 1
                w.retained_ms = int(round(float(d.get("retained_s") or 0) * 1000))
            w.retry_at_ms = d.get("retry_at_ms") if w.state == WorkloadState.RETRY_WAIT else None
            if (w.state == WorkloadState.RETRY_WAIT) != (w.retry_at_ms is not None):
                raise ReplayError(f"seq {ev.seq}: retry_at_ms missing or unexpected")
            w.terminal_ms = at if w.state in TERMINAL_STATES else None
            _book(st, a, -1)
        elif t == "requeued":
            w.state = workload_after(w.state, t)
            w.retry_at_ms = None
        elif t == "cancel_requested":
            if w.cancel_requested and w.state not in (WorkloadState.QUEUED, WorkloadState.RETRY_WAIT):
                raise ReplayError(f"seq {ev.seq}: repeated cancel_requested")
            w.state = workload_after(w.state, t)
            w.cancel_requested = True
            immediate = w.state == WorkloadState.CANCELLED
            if bool(d.get("immediate")) != immediate:
                raise ReplayError(f"seq {ev.seq}: immediate flag disagrees")
            if immediate:
                w.retry_at_ms = None
                w.terminal_ms = at
        else:
            raise ReplayError(f"seq {ev.seq}: unknown event type {t}")
    except ReplayError:
        raise
    except Exception as e:  # InvalidTransition and malformed data
        raise ReplayError(f"seq {ev.seq} ({t} {ev.workload_id}): {e}") from e
    if w.state != old_state:
        w.state_since_ms = at
    w.version += 1
    _check_state(ev, w.state)


def _att(st: ReplayState, ev: Event) -> Attempt:
    a = st.attempts.get(ev.attempt_id or "")
    if a is None or a.workload_id != ev.workload_id:
        raise ReplayError(f"seq {ev.seq}: unknown attempt {ev.attempt_id}")
    return a


def _check_state(ev: Event, state: WorkloadState) -> None:
    logged = ev.data.get("state")
    if logged is not None and logged != state.value:
        raise ReplayError(f"seq {ev.seq}: logged state {logged} but the table gives {state.value}")


def replay(events: list[Event]) -> ReplayState:
    st = ReplayState()
    for ev in events:
        apply(st, ev)
    return st


def diff(
    st: ReplayState,
    workloads: list[Workload],
    attempts: list[Attempt],
    books: dict[tuple[str, str], tuple[int, int, int]],
) -> list[str]:
    """Differences between a replayed state and the stored tables (empty = equal)."""
    out: list[str] = []
    tw = {w.id: w for w in workloads}
    if set(tw) != set(st.workloads):
        out.append(
            f"workload ids differ: table-only {sorted(set(tw) - set(st.workloads))}, "
            f"replay-only {sorted(set(st.workloads) - set(tw))}"
        )
    for wid in sorted(set(tw) & set(st.workloads)):
        if tw[wid] != st.workloads[wid]:
            out.append(f"workload {wid}: table {tw[wid]} != replay {st.workloads[wid]}")
    ta = {a.id: a for a in attempts}
    if set(ta) != set(st.attempts):
        out.append(
            f"attempt ids differ: table-only {sorted(set(ta) - set(st.attempts))}, "
            f"replay-only {sorted(set(st.attempts) - set(ta))}"
        )
    for aid in sorted(set(ta) & set(st.attempts)):
        if ta[aid] != st.attempts[aid]:
            out.append(f"attempt {aid}: table {ta[aid]} != replay {st.attempts[aid]}")
    rb = st.books_view()
    if dict(sorted(books.items())) != rb:
        out.append(f"books differ: table {dict(sorted(books.items()))} != replay {rb}")
    return out


__all__ = ["AttemptState", "ReplayError", "ReplayState", "apply", "diff", "replay"]
