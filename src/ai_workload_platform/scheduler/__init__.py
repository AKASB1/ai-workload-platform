"""Scheduler boundary and deterministic local adapter."""
from typing import Protocol
from ai_workload_platform.models import Workload, WorkloadState

class SchedulerAdapter(Protocol):
    async def submit(self, workload: Workload) -> None: ...
    async def cancel(self, workload_id: str) -> None: ...
    async def status(self, workload_id: str) -> WorkloadState: ...

class LocalScheduler:
    def __init__(self) -> None:
        self.jobs: dict[str, WorkloadState] = {}

    async def submit(self, workload: Workload) -> None:
        if workload.id in self.jobs:
            raise ValueError("duplicate workload id")
        self.jobs[workload.id] = WorkloadState.RUNNING

    async def cancel(self, workload_id: str) -> None:
        if workload_id not in self.jobs:
            raise KeyError(workload_id)
        self.jobs[workload_id] = WorkloadState.CANCELLED

    async def status(self, workload_id: str) -> WorkloadState:
        return self.jobs[workload_id]
