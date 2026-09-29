"""Small in-process submission service; HTTP endpoints are planned."""
from ai_workload_platform.admission import admit
from ai_workload_platform.lifecycle import transition
from ai_workload_platform.models import ResourceRequest, Workload, WorkloadState
from ai_workload_platform.scheduler import SchedulerAdapter

class WorkloadService:
    def __init__(self, scheduler: SchedulerAdapter, quota: ResourceRequest) -> None:
        self.scheduler = scheduler
        self.quota = quota
        self.workloads: dict[str, Workload] = {}

    async def submit(self, workload: Workload) -> Workload:
        if workload.id in self.workloads:
            raise ValueError("duplicate workload id")
        if not admit(workload.resources, self.quota):
            raise ValueError("quota exceeded")
        transition(workload, WorkloadState.ADMITTED)
        await self.scheduler.submit(workload)
        transition(workload, WorkloadState.RUNNING)
        self.workloads[workload.id] = workload
        return workload
