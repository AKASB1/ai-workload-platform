"""Read-only local cluster inventory model."""
from dataclasses import dataclass
from ai_workload_platform.models import ResourceRequest

@dataclass(frozen=True)
class Node:
    name: str
    capacity: ResourceRequest

def total_gpus(nodes: list[Node]) -> int:
    return sum(node.capacity.gpus for node in nodes)
