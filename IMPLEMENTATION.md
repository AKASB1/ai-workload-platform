# Implementation plan

## Core abstractions

### Workload

A workload owns a job specification, requested resources, priority, queue, retry policy, and current lifecycle state.

### Resource request

Represent CPU, memory, GPU count/type, and optional topology constraints separately from the container command.

### Scheduler adapter

Keep scheduling behind a narrow interface:

```python
class SchedulerAdapter:
    async def submit(self, workload): ...
    async def cancel(self, workload_id): ...
    async def status(self, workload_id): ...
```

The first adapter targets Kubernetes Jobs. Later adapters can target Kueue, Volcano, or a local simulator.

## Initial modules

1. `api` — workload CRUD and event stream
2. `admission` — quotas, validation, queue assignment
3. `cluster` — node/GPU inventory snapshot
4. `scheduler` — backend adapters
5. `lifecycle` — retries, cancellation, reconciliation
6. `observability` — queue delay, runtime, utilization

## Delivery order

1. local fake scheduler
2. PostgreSQL workload state
3. Kubernetes Job adapter
4. queue priority and quotas
5. failure reconciliation
6. GPU inventory
7. scheduler policy plug-ins
8. benchmark workloads and dashboards
