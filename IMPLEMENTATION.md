# Implementation plan

## Core abstractions

### Workload

A workload is a validated, canonical specification (`docs/contracts.md` §2) in a namespace, with a state (`QUEUED`, `STARTING`, `RUNNING`, `RETRY_WAIT`, `SUCCEEDED`, `FAILED`, `DEAD_LETTER`, `CANCELLED`), a version, and attempts. An attempt is one try on a placement (`STARTING`, `RUNNING`, `STOPPING`, `ENDED` with an end reason). Every change is a versioned write with one event in an append-only log.

### Resource request

GPUs per worker, workers (the gang size), CPUs and memory per worker, an optional GPU class, and a topology sensitivity (`any`, `rack`, `node`), kept separate from the (optional, unused) container command. Simulated workloads carry their true run time in `sim`, hidden from policies.

### Two interfaces instead of one scheduler adapter

The scaffold put scheduling behind one adapter (`submit`, `cancel`, `status`). That mixed two jobs, which are now two interfaces:

```python
class Policy(Protocol):              # who starts where; never executes anything
    def hello(self, msg: dict) -> None: ...
    def schedule(self, view: dict) -> dict: ...   # wire form of the external-policy protocol v1
    def close(self) -> None: ...

class SchedulerAdapter(Protocol):    # the backend: executes and observes
    def inventory(self) -> list[NodeInfo]: ...
    def start(self, attempt: AttemptRequest) -> None: ...   # idempotent on the attempt id
    def stop(self, attempt_id: str) -> None: ...            # idempotent
    def observe(self) -> Snapshot: ...                      # complete, may be stale (A1)
    def forget(self, attempt_id: str) -> None: ...
```

Policies: three built-ins (`fifo+first_fit`, `priority+best_fit`, `quota+first_fit`) and `ExternalPolicy`, a child process that speaks the protocol of `gpu-cluster-scheduler`. Backends: a deterministic local backend on an injected clock, and a Kubernetes backend (Indexed Jobs) on the official client or a deterministic fake.

## Modules

1. `api` — HTTP API v1 (FastAPI), the service runtime
2. `admission` — validation and the admission transaction with its reason codes
3. `cluster` — cluster configuration v1, topology
4. `scheduler` — the backend contract, the local backend, the Kubernetes backend
5. `policy` — the policy interface, built-ins, the external-policy client, failure handling
6. `lifecycle` — transition and retry tables
7. `store` — versioned store, event log, replay, lease
8. `controller` — the reconciliation rules, the tick, the drivers
9. `observability` — Prometheus metrics, JSON logs, evaluation metrics
10. `faults` — the fault-injection harness; `bench` — the evaluation

## Delivery order

Revised: policy plug-ins and benchmark workloads come before the Kubernetes adapter. Mapped to the milestones of this implementation:

- [x] local fake scheduler — the local backend and the virtual driver (M4, M5)
- [x] PostgreSQL workload state — the versioned store on SQLite and PostgreSQL, one code path (M2)
- [x] queue priority and quotas — admission, namespaces with quota and cap, priority-aware built-ins (M3)
- [x] scheduler policy plug-ins backed by `gpu-cluster-scheduler` policies, and benchmark workloads from its traces — the policy interface, the external-policy protocol client with a stub server, the trace schema v1 loader and a generator (M3, M8); running the policy server of `gpu-cluster-scheduler` itself is Tier 2 item 3 (done: it matches that project's simulator job by job, `docs/contracts.md` §11)
- [x] Kubernetes Job adapter — the Kubernetes backend, on a fake client and on a kind cluster (M4)
- [x] failure reconciliation — rules R1–R9 (M5) and the fault-injection harness that checks them (M7)
- [x] GPU inventory — `inventory()` from the backend, stored by the controller, advertised extended resources on Kubernetes (M4)
- [ ] dashboards — Tier 2 item 6 (Prometheus profile, alert rules, a dashboard); `/metrics` exists

## Evaluation and acceptance

- failure-injection tests: crashes at seven named points, a paused controller, store, backend, and policy faults, and races, with invariants I1–I8 checked after every step (`docs/failure-injection.md`); a real-process kill test
- the same trace through the local backend, the Kubernetes fake, and a kind cluster (E2), reported side by side without ranking
- queue delay, run time, utilization, and quota metrics per namespace from the event log
- one command brings up the local stack (`python -m ai_workload_platform up`, or `docker compose up` with PostgreSQL)

No performance numbers are added before they are measured; the results and their commands are in `benchmarks/README.md`.

## Cross-project contracts

No code is shared with the sibling projects; this project follows their documents:

- From `gpu-cluster-scheduler`: the trace schema v1 (16 columns, manifest), the cluster configuration v1, the metric definitions (window, wait, JCT, bounded slowdown, utilization, Jain, SLO attainment, quota metrics), the external-policy protocol v1 (`hello`, `schedule`, `decision`, `bye`, `error`), and the policy semantics of its baselines.
- From `llm-serving-control`: the conventions (injected clock, named random streams, units, determinism, CSV and manifest rules).
- From `distributed-task-platform`: the ideas of versioned compare-and-set transitions, a lease used as a fencing token, retry with backoff, and a reconciler that repairs from the database.
- Defined here, in `docs/contracts.md`: the workload specification, the lifecycle and retry tables, the event log, the store guarantees, the HTTP API, the backend contract, and the reconciliation rules.

## Implementation status

Built and tested (Tier 1): the store with compare and set, the event log and its replay (SQLite and PostgreSQL), admission with reason codes and idempotency, the policy interface with three built-ins and the external-policy client, the local and Kubernetes backends (fake and kind), the controller with rules R1–R9, crash points and a fenced lease, the virtual and asyncio drivers, the HTTP API with its OpenAPI document, the client and CLI, Prometheus metrics and JSON logs, the fault harness with four injected bugs, the evaluation E1–E3, Docker Compose, Kubernetes manifests (validated), and CI.

Built and tested (Tier 2, items 1–4): preemption with checkpoints (`+preempt`, `+reclaim`, retained work, restart overhead, lost-work metrics, the preempt fault in the harness); the platform as a Deployment in the kind cluster with the in-cluster configuration and a least-privilege Role; the policy server of `gpu-cluster-scheduler` through the protocol, matching its simulator job by job; the time-scale sweep, the partial-gang and node-loss demonstrations on kind, and the fake client run with the measured start latency.

Not built: Tier 2 items 5–7 (two controllers on PostgreSQL, monitoring, API keys and retention) and Tier 3.
