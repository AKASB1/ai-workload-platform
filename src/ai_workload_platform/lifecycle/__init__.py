"""Allowed local workload state transitions."""
from ai_workload_platform.models import Workload, WorkloadState

_ALLOWED = {
    WorkloadState.PENDING: {WorkloadState.ADMITTED, WorkloadState.CANCELLED},
    WorkloadState.ADMITTED: {WorkloadState.RUNNING, WorkloadState.CANCELLED},
    WorkloadState.RUNNING: {WorkloadState.SUCCEEDED, WorkloadState.FAILED, WorkloadState.CANCELLED},
}

def transition(workload: Workload, target: WorkloadState) -> None:
    if target not in _ALLOWED.get(workload.state, set()):
        raise ValueError(f"invalid transition: {workload.state} -> {target}")
    workload.state = target
