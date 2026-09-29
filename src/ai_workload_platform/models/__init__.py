"""Workload records shared by the control plane."""
from dataclasses import dataclass
from enum import Enum

class WorkloadState(str, Enum):
    PENDING = "pending"
    ADMITTED = "admitted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass(frozen=True)
class ResourceRequest:
    cpu: float
    memory_mb: int
    gpus: int = 0

    def __post_init__(self) -> None:
        if self.cpu <= 0 or self.memory_mb <= 0 or self.gpus < 0:
            raise ValueError("resource requests must be positive; gpus may be zero")

@dataclass
class Workload:
    id: str
    namespace: str
    resources: ResourceRequest
    state: WorkloadState = WorkloadState.PENDING
