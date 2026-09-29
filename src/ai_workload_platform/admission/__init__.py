"""Local quota check; persistent quotas are planned."""
from ai_workload_platform.models import ResourceRequest

def admit(request: ResourceRequest, quota: ResourceRequest) -> bool:
    return (request.cpu <= quota.cpu and request.memory_mb <= quota.memory_mb
            and request.gpus <= quota.gpus)
