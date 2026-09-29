# AI Workload Orchestration Platform

A control plane for running AI jobs on shared compute resources. The project separates workload lifecycle management from the underlying scheduler so different placement policies can be compared without changing the API layer.

**Status:** implementation scaffold.

## Scope

- job submission and lifecycle management
- GPU/CPU resource requests and queue priorities
- namespaces, quotas, and admission control
- Kubernetes-backed worker execution
- scheduler adapter interface
- failure recovery and job resubmission
- cluster inventory and workload metrics
- policy experiments for fairness, utilization, and queueing delay

## Proposed stack

Python 3.12 · FastAPI · Pydantic · PostgreSQL · Redis · Kubernetes API · gRPC · Prometheus · Docker

## Control-plane model

```text
Client / SDK
     │
     ▼
 API + Job Store
     │
     ▼
Admission / Quota
     │
     ▼
Scheduler Adapter ──► Kubernetes
     │                    │
     ▼                    ▼
Cluster State ◄──── Worker Pods
     │
     ▼
Metrics / Events
```

## Repository layout

```text
src/ai_workload_platform/
  api/
  models/
  admission/
  scheduler/
  cluster/
  lifecycle/
  observability/
tests/
configs/
deploy/
```

See [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Reference projects

- [kubernetes-sigs/kueue](https://github.com/kubernetes-sigs/kueue) — queueing and admission for batch workloads
- [volcano-sh/volcano](https://github.com/volcano-sh/volcano) — batch scheduling and gang scheduling on Kubernetes
- [ray-project/ray](https://github.com/ray-project/ray) — distributed execution and AI workload management

## License

MIT

## Available now

The local in-process path has a typed workload model, quota validation, lifecycle transitions, and an async scheduler adapter. Run `PYTHONPATH=src python -m unittest discover -s tests`. HTTP, Kubernetes, persistence, and metrics exporters remain planned.
