"""Records, states, reasons, and errors shared by every layer (no framework, no driver)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class WorkloadState(StrEnum):
    QUEUED = "QUEUED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    RETRY_WAIT = "RETRY_WAIT"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    DEAD_LETTER = "DEAD_LETTER"
    CANCELLED = "CANCELLED"


TERMINAL_STATES = frozenset(
    {WorkloadState.SUCCEEDED, WorkloadState.FAILED, WorkloadState.DEAD_LETTER, WorkloadState.CANCELLED}
)
ACTIVE_STATES = frozenset({WorkloadState.STARTING, WorkloadState.RUNNING})


class AttemptState(StrEnum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    ENDED = "ENDED"


class EndReason(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_FATAL = "failed_fatal"
    NODE_LOST = "node_lost"
    BACKEND_LOST = "backend_lost"
    START_TIMEOUT = "start_timeout"
    PREEMPTED = "preempted"
    CANCELLED = "cancelled"


COUNTED_REASONS = frozenset(
    {
        EndReason.FAILED_RETRYABLE,
        EndReason.FAILED_FATAL,
        EndReason.NODE_LOST,
        EndReason.BACKEND_LOST,
        EndReason.START_TIMEOUT,
    }
)
# Reasons a stop request may carry (R2, R4, R8, Tier 2 preemption).
STOP_REASONS = frozenset(
    {EndReason.START_TIMEOUT, EndReason.NODE_LOST, EndReason.CANCELLED, EndReason.PREEMPTED}
)


class EventType(StrEnum):
    SUBMITTED = "submitted"
    STARTED = "started"
    RUNNING = "running"
    STOP_REQUESTED = "stop_requested"
    ATTEMPT_ENDED = "attempt_ended"
    REQUEUED = "requeued"
    CANCEL_REQUESTED = "cancel_requested"


class Phase(StrEnum):
    """Attempt phase as a backend reports it in `observe()`."""

    STARTING = "starting"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"


TERMINAL_PHASES = frozenset({Phase.SUCCEEDED, Phase.FAILED, Phase.STOPPED})


class FailReason(StrEnum):
    """Why a backend reports phase `failed`."""

    EXIT = "exit"
    NODE_LOST = "node_lost"
    EVICTED = "evicted"


# --- errors ---------------------------------------------------------------------------------


class PlatformError(Exception):
    """An error with a stable upper-case code and an HTTP status (API body: section 3)."""

    status = 400
    code = "INTERNAL"

    def __init__(
        self,
        message: str = "",
        details: dict[str, Any] | None = None,
        *,
        code: str | None = None,
        status: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message or (code or self.code))
        self.message = message or (code or self.code)
        self.details = details or {}
        if code is not None:
            self.code = code
        if status is not None:
            self.status = status
        self.headers = headers or {}

    def body(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class VersionConflict(PlatformError):
    status = 409
    code = "VERSION_CONFLICT"


class LeaseLost(PlatformError):
    status = 503
    code = "LEASE_LOST"


class StoreUnavailable(PlatformError):
    status = 503
    code = "STORE_UNAVAILABLE"


class InvalidTransition(PlatformError):
    status = 409
    code = "INVALID_TRANSITION"


class BackendError(Exception):
    """Any backend call may raise it; the platform treats it as transient."""


# --- rows -------------------------------------------------------------------------------------


@dataclass
class Workload:
    id: str
    namespace: str
    spec: dict[str, Any]
    spec_hash: str
    priority: int
    gpus: int  # per worker
    workers: int
    cpus: int  # per worker
    mem_mb: int  # per worker
    submit_ms: int
    state: WorkloadState
    version: int
    cancel_requested: bool = False
    submit_seq: int = 0  # seq of the `submitted` event (pagination order)
    counted: int = 0  # counted attempts so far
    attempts: int = 0  # attempts created so far (n of the last attempt)
    retry_at_ms: int | None = None
    state_since_ms: int = 0
    first_started_ms: int | None = None
    terminal_ms: int | None = None
    retained_ms: int = 0  # work retained by checkpoints (reference-GPU milliseconds), Tier 2 preemption
    preemptions: int = 0

    @property
    def total_gpus(self) -> int:
        return self.gpus * self.workers

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES


@dataclass
class Attempt:
    id: str
    workload_id: str
    namespace: str
    n: int
    state: AttemptState
    placement: list[dict[str, Any]]  # [{"node": name, "workers": k}], nodes in (rack, name) order
    gpus: int  # per worker
    cpus: int
    mem_mb: int
    version: int
    started_ms: int  # time of the `started` event
    state_since_ms: int
    stop_reason: str | None = None
    stop_requested_ms: int | None = None
    running_ms: int | None = None  # time of the `running` event
    observed_started_ms: int | None = None
    observed_ended_ms: int | None = None
    ended_ms: int | None = None
    end_reason: str | None = None
    exit_code: int | None = None
    counted: bool | None = None
    observed_nodes: list[str] = field(default_factory=list)

    @property
    def workers(self) -> int:
        return sum(int(p["workers"]) for p in self.placement)


@dataclass(frozen=True)
class Event:
    seq: int
    at_ms: int
    type: str
    namespace: str
    workload_id: str
    attempt_id: str | None
    data: dict[str, Any]

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "at_ms": self.at_ms,
            "type": self.type,
            "namespace": self.namespace,
            "workload_id": self.workload_id,
            "attempt_id": self.attempt_id,
            "data": self.data,
        }


@dataclass
class Namespace:
    name: str
    quota_gpus: int
    cap_gpus: int
    max_priority: int = 9
    max_queued: int = 1000

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "quota_gpus": self.quota_gpus,
            "cap_gpus": self.cap_gpus,
            "max_priority": self.max_priority,
            "max_queued": self.max_queued,
        }


@dataclass(frozen=True)
class NodeInfo:
    """One node of the inventory as the backend reports it."""

    name: str
    rack: str
    gpu_class: str
    speed: float
    gpus: int
    cpus: int
    mem_mb: int
    ready: bool = True

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rack": self.rack,
            "class": self.gpu_class,
            "speed": self.speed,
            "gpus": self.gpus,
            "cpus": self.cpus,
            "mem_mb": self.mem_mb,
            "ready": self.ready,
        }


@dataclass(frozen=True)
class AttemptStatus:
    """One entry of an `observe()` snapshot."""

    attempt_id: str
    phase: Phase
    nodes: tuple[str, ...] = ()
    exit_code: int | None = None
    reason: FailReason | None = None
    started_ms: int | None = None
    ended_ms: int | None = None
    rate: float | None = None
    workers_by_node: tuple[tuple[str, int], ...] = ()
    work_done_s: float | None = (
        None  # reference-GPU seconds done (retained + progress), when the backend knows
    )
    incomplete: bool = False  # a gang whose start left only part of it behind (R1 starts it again)


@dataclass(frozen=True)
class Snapshot:
    taken_ms: int
    attempts: tuple[AttemptStatus, ...]

    def by_id(self) -> dict[str, AttemptStatus]:
        return {a.attempt_id: a for a in self.attempts}


@dataclass(frozen=True)
class AttemptRequest:
    """What `SchedulerAdapter.start` receives."""

    attempt_id: str
    workload_id: str
    namespace: str
    n: int  # attempt number (1-based)
    spec: dict[str, Any]
    placement: tuple[tuple[str, int], ...]  # (node, workers)
    retained_s: float = 0.0  # work retained by checkpoints from earlier preempted attempts
    restart_overhead_s: float = 0.0  # held without progress before the run (a start after a preemption)
