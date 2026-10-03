"""Workload and attempt transition tables (as data), and the retry/backoff rule.

The tables are the single definition used by the store's write path, by the replay function,
and by docs/contracts.md (a test checks that the document and these tables agree).
"""

from __future__ import annotations

import math

from ai_workload_platform.models import (
    COUNTED_REASONS,
    AttemptState,
    EndReason,
    EventType,
    InvalidTransition,
    WorkloadState,
)
from ai_workload_platform.rng import stream

W = WorkloadState
A = AttemptState

# Event keys: `attempt_ended` is split by its reason because the result depends on it.
ATTEMPT_ENDED_KEYS = tuple(f"attempt_ended:{r.value}" for r in EndReason)
EVENT_KEYS = (
    "submitted",
    "started",
    "running",
    "stop_requested",
    *ATTEMPT_ENDED_KEYS,
    "requeued",
    "cancel_requested",
)

# (state before or None, event key) -> state after.  Special values:
#   "RETRY_OR_DEAD"  -> RETRY_WAIT, or DEAD_LETTER when the counted attempts reach max_attempts
#   "SAME"           -> unchanged
# A workload with cancel_requested whose attempt ends with any reason except `succeeded`
# becomes CANCELLED (applied on top of this table by `workload_after`).
WORKLOAD_TRANSITIONS: dict[tuple[str | None, str], str] = {
    (None, "submitted"): W.QUEUED,
    (W.QUEUED, "started"): W.STARTING,
    (W.STARTING, "running"): W.RUNNING,
    (W.STARTING, "stop_requested"): "SAME",
    (W.RUNNING, "stop_requested"): "SAME",
    (W.RETRY_WAIT, "requeued"): W.QUEUED,
    (W.QUEUED, "cancel_requested"): W.CANCELLED,
    (W.RETRY_WAIT, "cancel_requested"): W.CANCELLED,
    (W.STARTING, "cancel_requested"): "SAME",
    (W.RUNNING, "cancel_requested"): "SAME",
}
for _s in (W.STARTING, W.RUNNING):
    WORKLOAD_TRANSITIONS[(_s, "attempt_ended:succeeded")] = W.SUCCEEDED
    WORKLOAD_TRANSITIONS[(_s, "attempt_ended:failed_fatal")] = W.FAILED
    for _r in ("failed_retryable", "node_lost", "backend_lost", "start_timeout"):
        WORKLOAD_TRANSITIONS[(_s, f"attempt_ended:{_r}")] = "RETRY_OR_DEAD"
    WORKLOAD_TRANSITIONS[(_s, "attempt_ended:preempted")] = W.QUEUED
    WORKLOAD_TRANSITIONS[(_s, "attempt_ended:cancelled")] = W.CANCELLED

ATTEMPT_TRANSITIONS: dict[tuple[str | None, str], str] = {
    (None, "started"): A.STARTING,
    (A.STARTING, "running"): A.RUNNING,
    (A.STARTING, "stop_requested"): A.STOPPING,
    (A.RUNNING, "stop_requested"): A.STOPPING,
}
for _s in (A.STARTING, A.RUNNING, A.STOPPING):
    for _k in ATTEMPT_ENDED_KEYS:
        ATTEMPT_TRANSITIONS[(_s, _k)] = A.ENDED

# Retry table: counted and uncounted end reasons.
RETRY_TABLE = {
    r.value: {
        "counted": r in COUNTED_REASONS,
        "retried": r in COUNTED_REASONS and r != EndReason.FAILED_FATAL,
    }
    for r in EndReason
}


def event_key(event_type: str, reason: str | None = None) -> str:
    if event_type == EventType.ATTEMPT_ENDED:
        return f"attempt_ended:{reason}"
    return str(event_type)


def is_counted(reason: str) -> bool:
    return EndReason(reason) in COUNTED_REASONS


def workload_after(
    state: str | None,
    event_type: str,
    *,
    reason: str | None = None,
    cancel_requested: bool = False,
    counted_after: int = 0,
    max_attempts: int = 1,
) -> WorkloadState:
    """The workload state after an event; raises InvalidTransition for a pair not in the table."""
    key = (W(state) if state is not None else None, event_key(event_type, reason))
    result = WORKLOAD_TRANSITIONS.get(key)
    if result is None:
        raise InvalidTransition(f"no transition from {state} on {key[1]}", {"state": state, "event": key[1]})
    if result == "SAME":
        return W(state)  # type: ignore[arg-type]
    if event_type == EventType.ATTEMPT_ENDED and cancel_requested and reason != EndReason.SUCCEEDED:
        return W.CANCELLED
    if result == "RETRY_OR_DEAD":
        return W.DEAD_LETTER if counted_after >= max_attempts else W.RETRY_WAIT
    return W(result)


def attempt_after(state: str | None, event_type: str, reason: str | None = None) -> AttemptState:
    key = (A(state) if state is not None else None, event_key(event_type, reason))
    result = ATTEMPT_TRANSITIONS.get(key)
    if result is None:
        raise InvalidTransition(
            f"no attempt transition from {state} on {key[1]}", {"state": state, "event": key[1]}
        )
    return A(result)


def backoff_s(k: int, base_s: float, cap_s: float) -> float:
    """Delay before jitter for the k-th counted failure: min(cap, base * 2^(k-1))."""
    if k < 1:
        raise ValueError("k starts at 1")
    return min(float(cap_s), float(base_s) * (2.0 ** (k - 1)))


def jitter_u(seed: int, workload_id: str, k: int) -> float:
    """First random() of a fresh stream named retry:<workload id>:<k>."""
    return stream(seed, f"retry:{workload_id}:{k}").random()


def retry_at_ms(at_ms: int, k: int, retry: dict, workload_id: str, seed: int) -> int:
    d = backoff_s(k, retry["backoff_base_s"], retry["backoff_cap_s"])
    u = jitter_u(seed, workload_id, k) if retry.get("jitter", "full") == "full" else 1.0
    return int(at_ms) + int(math.ceil(1000.0 * d * u - 1e-6))
