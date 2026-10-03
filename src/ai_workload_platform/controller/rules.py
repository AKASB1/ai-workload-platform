"""Pure decision helpers of the reconciliation rules (docs/contracts.md §8). No I/O, no clock."""

from __future__ import annotations

from dataclasses import dataclass

from ai_workload_platform.models import Attempt, AttemptStatus, EndReason, FailReason, Phase

RULE_ORDER = ("R6", "R5", "R4", "R3", "R2", "R1", "R8", "R7")


@dataclass(frozen=True)
class Thresholds:
    start_retry_ms: int = 5000
    start_timeout_ms: int = 120_000
    node_grace_ms: int = 30_000
    lost_grace_ms: int = 30_000
    stop_retry_ms: int = 10_000
    observe_interval_ms: int = 1000
    lease_ttl_ms: int = 15_000
    policy_timeout_s: float = 60.0
    policy_max_failures: int = 3

    def to_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


def terminal_reason(a: Attempt, s: AttemptStatus, fatal_exit_codes: list[int]) -> str:
    """R6: the end reason for a terminal phase. `succeeded` always wins; otherwise a requested stop
    wins; `failed` maps by its reason; `stopped` without a request is `backend_lost`."""
    if s.phase == Phase.SUCCEEDED:
        return EndReason.SUCCEEDED.value
    if a.stop_reason:
        return a.stop_reason
    if s.phase == Phase.FAILED:
        if s.reason in (FailReason.NODE_LOST, FailReason.EVICTED):
            return EndReason.NODE_LOST.value
        if s.exit_code is not None and s.exit_code in fatal_exit_codes:
            return EndReason.FAILED_FATAL.value
        return EndReason.FAILED_RETRYABLE.value
    return EndReason.BACKEND_LOST.value  # stopped, but no stop was requested


def lost_since(last_seen_ms: int | None, state_since_ms: int, controller_start_ms: int) -> int:
    """R5 counts from the latest of: the last snapshot that showed it, its entry into the state,
    and the start of this controller."""
    return max(last_seen_ms if last_seen_ms is not None else -1, state_since_ms, controller_start_ms)


def r5_due(now_ms: int, since_ms: int, grace_ms: int) -> bool:
    return now_ms - since_ms >= grace_ms


def r1_due(now_ms: int, started_ms: int, last_call_ms: int | None, retry_ms: int) -> bool:
    return now_ms >= max(started_ms, last_call_ms if last_call_ms is not None else started_ms) + retry_ms


def r2_due(now_ms: int, started_ms: int, timeout_ms: int) -> bool:
    return now_ms - started_ms >= timeout_ms


def r4_due(now_ms: int, not_ready_since_ms: int | None, grace_ms: int) -> bool:
    return not_ready_since_ms is not None and now_ms - not_ready_since_ms >= grace_ms


def r8_restop_due(now_ms: int, last_stop_call_ms: int | None, retry_ms: int) -> bool:
    """A STOPPING attempt gets `stop` again `stop_retry_ms` after the last call this controller made,
    or at once when this controller has made none (a fresh controller has no memory)."""
    return last_stop_call_ms is None or now_ms - last_stop_call_ms >= retry_ms
