"""Mutations: each is one write transaction with compare-and-set and exactly one event.

A mutation takes the caller's snapshot of the row(s) it decided on (read earlier, outside this
transaction), computes the new row from that snapshot with the transition tables, and writes it
with `Tx.cas_update`. If another writer changed the row since, the write fails with
`VersionConflict` and the caller re-reads and re-decides; nothing is retried blindly.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from ai_workload_platform.crash import crashpoint
from ai_workload_platform.lifecycle import attempt_after, is_counted, retry_at_ms, workload_after
from ai_workload_platform.models import (
    TERMINAL_STATES,
    Attempt,
    AttemptState,
    EndReason,
    Event,
    EventType,
    PlatformError,
    VersionConflict,
    Workload,
    WorkloadState,
)
from ai_workload_platform.models.spec import mem_mb
from ai_workload_platform.store.sql import Fence, Store, Tx


class CapExceeded(VersionConflict):
    """A start would take the namespace above its cap as configured now (a stale action)."""

    code = "STALE_CAP"


def books_delta(event_type: str, attempt_state_before: str | None) -> int:
    """+1 when an event reserves the attempt's placement in the books, -1 when it releases it.

    R9: resources are held from `started` until the attempt is ENDED (STOPPING included).
    """
    if event_type == EventType.STARTED:
        return 1
    if event_type == EventType.ATTEMPT_ENDED:
        return -1
    return 0


def retained_after_preemption(w: Workload, work_done_s: float | None) -> int:
    """Work retained after a preemption (ms of reference work): the work done rounded down to a multiple of
    `checkpoint_interval_s` (nothing without checkpoints), never more than the work done. When the backend
    could not say how much was done, the earlier retained work stays."""
    if work_done_s is None:
        return w.retained_ms
    ck = float(w.spec.get("checkpoint_interval_s") or 0)
    if ck <= 0:
        return 0
    done = max(0.0, float(work_done_s))
    kept = min(
        done, math.floor(done / ck * (1 + 1e-9)) * ck
    )  # the backend's figure includes earlier retained work
    return int(round(kept * 1000))


def _audit(data: dict[str, Any], fence: Fence | None) -> dict[str, Any]:
    """Controller writes carry the lease epoch they were made under (I5 checks it never decreases)."""
    if fence is not None:
        data["epoch"] = fence.epoch
    return data


def _apply_books(tx: Tx, event_type: str, a: Attempt, state_before: str | None) -> None:
    d = books_delta(event_type, state_before)
    if d:
        tx.add_books(a.namespace, a.placement, a.gpus, a.cpus, a.mem_mb, d)


# --- submission (called by admission inside its transaction) --------------------------------


def insert_submitted(
    tx: Tx, namespace: str, canon: dict[str, Any], spec_hash: str, idem_key: str | None, now_ms: int
) -> tuple[Workload, Event]:
    spec = dict(canon)
    wid = spec.get("id")
    while wid is None or (not canon.get("id") and tx.exists(wid)):
        wid = f"w{tx.next_workload_number()}"  # skip numbers a client already used as an explicit id
    spec["id"] = wid
    at = tx.event_time(now_ms)
    ev = tx.append_event(
        at,
        EventType.SUBMITTED,
        namespace,
        wid,
        None,
        {
            "spec": spec,
            "spec_hash": spec_hash,
            "idempotency_key": idem_key,
            "state": WorkloadState.QUEUED.value,
        },
    )
    w = Workload(
        id=wid,
        namespace=namespace,
        spec=spec,
        spec_hash=spec_hash,
        priority=int(spec["priority"]),
        gpus=int(spec["gpus"]),
        workers=int(spec["workers"]),
        cpus=int(spec["cpus"]),
        mem_mb=mem_mb(spec["mem_gb"]),
        submit_ms=ev.at_ms,
        submit_seq=ev.seq,
        state=WorkloadState.QUEUED,
        version=1,
        state_since_ms=ev.at_ms,
    )
    tx.insert_workload(w)
    if idem_key is not None:
        tx.put_idempotency(namespace, idem_key, spec_hash, wid)
    return w, ev


# --- controller mutations ----------------------------------------------------------------------


def commit_started(
    store: Store,
    fence: Fence | None,
    w: Workload,
    placement: list[dict[str, Any]],
    now_ms: int,
    overhead_s: float = 0.0,
) -> tuple[Workload, Attempt, Event]:
    with store.transaction(fence) as tx:
        at = tx.event_time(now_ms)
        state = workload_after(w.state, EventType.STARTED)
        ns = tx.namespace(w.namespace)
        if ns is None or tx.ns_allocated_gpus(w.namespace) + w.total_gpus > ns.cap_gpus:
            raise CapExceeded(f"start of {w.id} would exceed the cap of {w.namespace}", {"workload": w.id})
        n = w.attempts + 1
        aid = f"{w.id}-a{n}"
        a = Attempt(
            id=aid,
            workload_id=w.id,
            namespace=w.namespace,
            n=n,
            state=attempt_after(None, EventType.STARTED),
            placement=[{"node": p["node"], "workers": int(p["workers"])} for p in placement],
            gpus=w.gpus,
            cpus=w.cpus,
            mem_mb=w.mem_mb,
            version=1,
            started_ms=at,
            state_since_ms=at,
        )
        nw = replace(
            w,
            state=state,
            attempts=n,
            version=w.version + 1,
            state_since_ms=at,
            retry_at_ms=None,
            first_started_ms=w.first_started_ms if w.first_started_ms is not None else at,
        )
        tx.update_workload(nw, w.version)
        tx.insert_attempt(a)
        _apply_books(tx, EventType.STARTED, a, None)
        ev = tx.append_event(
            at,
            EventType.STARTED,
            w.namespace,
            w.id,
            aid,
            _audit(
                {
                    "attempt": n,
                    "placement": a.placement,
                    "state": state.value,
                    "retained_s": w.retained_ms / 1000.0,
                    "overhead_s": overhead_s,
                },
                fence,
            ),
        )
    return nw, a, ev


def commit_running(
    store: Store,
    fence: Fence | None,
    w: Workload,
    a: Attempt,
    observed_started_ms: int | None,
    nodes: list[str],
    now_ms: int,
    workers_by_node: list[tuple[str, int]] | None = None,
) -> tuple[Workload, Attempt, Event]:
    with store.transaction(fence) as tx:
        at = tx.event_time(now_ms)
        ws = workload_after(w.state, EventType.RUNNING)
        ast = attempt_after(a.state, EventType.RUNNING)
        nw = replace(
            w, state=ws, version=w.version + 1, state_since_ms=at if ws != w.state else w.state_since_ms
        )
        na = replace(
            a,
            state=ast,
            version=a.version + 1,
            running_ms=at,
            observed_started_ms=observed_started_ms,
            observed_nodes=list(nodes),
            state_since_ms=at,
        )
        tx.update_workload(nw, w.version)
        tx.update_attempt(na, a.version)
        ev = tx.append_event(
            at,
            EventType.RUNNING,
            w.namespace,
            w.id,
            a.id,
            _audit(
                {
                    "observed_started_ms": observed_started_ms,
                    "nodes": list(nodes),
                    "state": ws.value,
                    "workers_by_node": [[n, int(k)] for n, k in (workers_by_node or [])],
                },
                fence,
            ),
        )
    return nw, na, ev


def commit_stop_requested(
    store: Store, fence: Fence | None, w: Workload, a: Attempt, reason: str, now_ms: int
) -> tuple[Workload, Attempt, Event]:
    with store.transaction(fence) as tx:
        at = tx.event_time(now_ms)
        ws = workload_after(w.state, EventType.STOP_REQUESTED)
        ast = attempt_after(a.state, EventType.STOP_REQUESTED)
        nw = replace(w, state=ws, version=w.version + 1)
        na = replace(
            a,
            state=ast,
            version=a.version + 1,
            stop_reason=str(reason),
            stop_requested_ms=at,
            state_since_ms=at,
        )
        tx.update_workload(nw, w.version)
        tx.update_attempt(na, a.version)
        _apply_books(tx, EventType.STOP_REQUESTED, a, a.state)
        ev = tx.append_event(
            at,
            EventType.STOP_REQUESTED,
            w.namespace,
            w.id,
            a.id,
            _audit({"reason": str(reason), "state": ws.value}, fence),
        )
    return nw, na, ev


def commit_attempt_ended(
    store: Store,
    fence: Fence | None,
    w: Workload,
    a: Attempt,
    reason: str,
    exit_code: int | None,
    observed_started_ms: int | None,
    observed_ended_ms: int | None,
    nodes: list[str] | None,
    now_ms: int,
    seed: int,
    crash_point: str | None = None,
    work_done_s: float | None = None,
) -> tuple[Workload, Attempt, Event]:
    with store.transaction(fence) as tx:
        at = tx.event_time(now_ms)
        retained_ms, preemptions = w.retained_ms, w.preemptions
        extra: dict[str, Any] = {}
        if reason == EndReason.PREEMPTED:
            retained_ms = retained_after_preemption(w, work_done_s)
            preemptions += 1
            extra = {"work_done_s": work_done_s, "retained_s": retained_ms / 1000.0}
        counted = is_counted(reason)
        counted_after = w.counted + (1 if counted else 0)
        retry = w.spec["retry"]
        ws = workload_after(
            w.state,
            EventType.ATTEMPT_ENDED,
            reason=reason,
            cancel_requested=w.cancel_requested,
            counted_after=counted_after,
            max_attempts=int(retry["max_attempts"]),
        )
        ast = attempt_after(a.state, EventType.ATTEMPT_ENDED, reason)
        r_at = retry_at_ms(at, counted_after, retry, w.id, seed) if ws == WorkloadState.RETRY_WAIT else None
        obs_nodes = list(nodes) if nodes else list(a.observed_nodes)
        na = replace(
            a,
            state=ast,
            version=a.version + 1,
            end_reason=str(reason),
            exit_code=exit_code,
            counted=counted,
            observed_started_ms=observed_started_ms
            if observed_started_ms is not None
            else a.observed_started_ms,
            observed_ended_ms=observed_ended_ms,
            observed_nodes=obs_nodes,
            ended_ms=at,
            state_since_ms=at,
        )
        nw = replace(
            w,
            state=ws,
            version=w.version + 1,
            counted=counted_after,
            retry_at_ms=r_at,
            state_since_ms=at,
            terminal_ms=at if ws in TERMINAL_STATES else None,
            retained_ms=retained_ms,
            preemptions=preemptions,
        )
        tx.update_workload(nw, w.version)
        tx.update_attempt(na, a.version)
        _apply_books(tx, EventType.ATTEMPT_ENDED, a, a.state)
        ev = tx.append_event(
            at,
            EventType.ATTEMPT_ENDED,
            w.namespace,
            w.id,
            a.id,
            _audit(
                {
                    "reason": str(reason),
                    "exit_code": exit_code,
                    "counted": counted,
                    "state": ws.value,
                    "retry_at_ms": r_at,
                    "observed_started_ms": na.observed_started_ms,
                    "observed_ended_ms": observed_ended_ms,
                    "nodes": obs_nodes,
                    **extra,
                },
                fence,
            ),
        )
        if crash_point:
            crashpoint(crash_point)
    return nw, na, ev


def commit_requeued(store: Store, fence: Fence | None, w: Workload, now_ms: int) -> tuple[Workload, Event]:
    with store.transaction(fence) as tx:
        at = tx.event_time(now_ms)
        ws = workload_after(w.state, EventType.REQUEUED)
        nw = replace(w, state=ws, version=w.version + 1, retry_at_ms=None, state_since_ms=at)
        tx.update_workload(nw, w.version)
        ev = tx.append_event(
            at, EventType.REQUEUED, w.namespace, w.id, None, _audit({"state": ws.value}, fence)
        )
        crashpoint("requeue.before_commit")
    return nw, ev


# --- client command ------------------------------------------------------------------------------


def request_cancel(store: Store, namespace: str, wid: str, now_ms: int) -> tuple[Workload, bool]:
    """Cancel a workload. Returns (workload after, changed). A repeat appends no event."""
    with store.transaction() as tx:
        w = tx.workload(wid)
        if w is None or w.namespace != namespace:
            raise PlatformError(
                f"unknown workload {wid}",
                {"namespace": namespace, "id": wid},
                code="UNKNOWN_WORKLOAD",
                status=404,
            )
        if w.state in TERMINAL_STATES or (
            w.cancel_requested and w.state in (WorkloadState.STARTING, WorkloadState.RUNNING)
        ):
            return w, False
        at = tx.event_time(now_ms)
        ws = workload_after(w.state, EventType.CANCEL_REQUESTED)
        immediate = ws == WorkloadState.CANCELLED
        nw = replace(
            w,
            state=ws,
            version=w.version + 1,
            cancel_requested=True,
            retry_at_ms=None if immediate else w.retry_at_ms,
            state_since_ms=at if immediate else w.state_since_ms,
            terminal_ms=at if immediate else None,
        )
        tx.update_workload(nw, w.version)
        tx.append_event(
            at,
            EventType.CANCEL_REQUESTED,
            w.namespace,
            w.id,
            None,
            {"immediate": immediate, "state": ws.value},
        )
    return nw, True


__all__ = [
    "AttemptState",
    "CapExceeded",
    "books_delta",
    "commit_attempt_ended",
    "commit_requeued",
    "commit_running",
    "commit_started",
    "commit_stop_requested",
    "insert_submitted",
    "request_cancel",
]
