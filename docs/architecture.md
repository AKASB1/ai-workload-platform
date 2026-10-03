# Architecture

A control plane for AI workloads on shared GPU compute that keeps its state correct when its parts fail. The GPUs are **simulated**: workloads sleep for their run time; nothing runs a model. Formats and definitions are in `docs/contracts.md`; the Kubernetes mapping is in `docs/kubernetes.md`; the fault harness is in `docs/failure-injection.md`.

![framework](figures/framework.png)

## Modules and boundaries

```text
src/ai_workload_platform/
  clock.py, rng.py      injectable clock (integer ms) and named random streams (SplitMix64 / FNV-1a)
  models/               records, states, reasons, errors; the workload specification (pydantic)
  lifecycle/            transition and retry tables as data; backoff
  store/                Store + Tx (connections, transactions, CAS), ops (one mutation = one tx + one event),
                        replay (pure fold over the log), dialect (the only module that knows sqlite3 / psycopg)
  admission/            the admission transaction and its reason codes
  policy/               view builder, decision validation, built-ins, ExternalPolicy (protocol v1), stub server,
                        PolicyRunner (failure counting, fallback, recovery)
  scheduler/            SchedulerAdapter (the backend contract), local backend, kube/ (KubeClient, fake, backend)
  cluster/              cluster configuration v1, topology factor
  controller/           Controller.tick (lease, inventory, observe, rules, cycle), rules (pure helpers), drivers
  observability/        Prometheus metrics, JSON logs, evaluation metrics from the event log
  api/                  FastAPI application, service runtime (Platform)
  client.py             Python client (one Idempotency-Key per logical submission)
  faults/               harness, schedules, invariants, injected bugs (test-only patches)
  bench/                trace schema v1, generator, E1-E3 runner, statistics, real-cluster runner
  crash.py              named crash points and interleaving points (no-ops in production)
  sim.py                assemble an in-process platform on the virtual clock
```

Dependency direction: `api` and `bench` use everything below them; `controller` uses `store`, `policy`, `scheduler`; `store/ops` uses `lifecycle`; the domain code (`models`, `lifecycle`, `admission`, the policy interface and built-ins, `controller/rules.py`, `observability/evaluation.py`) imports no web framework and no database driver and never reads the wall clock — it receives `now_ms` or a `Clock` and works on the store interface.

The scaffold's `SchedulerAdapter` (submit / cancel / status) mixed two things that are now separate interfaces: the **policy** (`Policy.schedule(view) -> decision`: who starts where) and the **backend** (`SchedulerAdapter`: `inventory`, `start`, `stop`, `observe`, `forget`: execute and observe).

## The control loop

`Controller.tick(now_ms)` is one synchronous pass:

1. **Lease.** Acquire or renew the lease row (`holder`, `epoch`, `expires_ms`). A controller that finds another live holder is a standby; one whose epoch was superseded raises `LeaseLost` and stops. Every later write of the tick checks holder and epoch inside its own transaction (fencing), and every event it writes records the epoch.
2. **Inventory.** `backend.inventory()`; the controller stores the snapshot (and since when each node has been not ready).
3. **Observe.** `backend.observe()`: a complete snapshot of every attempt the backend knows.
4. **Rules** in the order R6, R5, R4, R3, R2, R1, R8, R7 (`docs/contracts.md` §8), on a fresh read of the store. Each repair is one transaction with compare and set on the rows it decided on; a `VersionConflict` means re-read and re-decide on the next tick (the controller then asks to be woken at once).
5. **Scheduling cycle** (only with a fresh inventory and snapshot): build the view from the store's books and the inventory, call the policy (through `PolicyRunner`), validate the decision all or nothing, commit one `started` event per action (write-ahead), then call `backend.start`.

A failing `inventory()` or `observe()` skips the rules that need it and the scheduling cycle (R7, the retry release, still runs); a store error ends the tick and the next one starts from the store again.

## Two drivers

- **Virtual** (`VirtualDriver`; simulations, the harness, the benchmarks): ticks once at time 0 so an inventory is stored, then jumps the clock to the earliest of the controller's next deadline (retry times, timeouts, grace periods, and a poll every `observe_interval_ms` while a start is in flight), the backend's next internal event, and the next client action. It never polls an idle system. At one instant: completions (observed in the tick), then submissions (applied between the rules and the cycle through the same admission function the API calls), then one scheduling cycle. A crash at a crash point restarts a new controller object over the same store and backend.
- **Live** (`LiveDriver`, asyncio, inside the FastAPI lifespan): the tick runs in a worker thread so the event loop is never blocked; it ticks at least every `observe_interval_ms` (wall clock), earlier when a deadline or a backend event is due or when the API kicks it after a submission or a cancel. The clock is the system clock, optionally scaled (`--scale`), with its origin stored in the store so a restarted service continues the same timeline. On `LeaseLost` the driver discards the controller and starts a fresh one (a standby until it can take the lease).

## The store and its guarantees

- The store is the only source of truth; controllers and the API keep only caches (and memory that a fresh controller may lose: the last start and stop calls, the last snapshot that showed an attempt).
- Every mutation is one transaction that checks the expected version of the row(s) it changes, bumps it, and appends exactly one event with the next gap-free `seq`. Writers are serialized (SQLite `BEGIN IMMEDIATE` with WAL and a busy timeout; PostgreSQL a transaction-level advisory lock), and the event time never decreases.
- The event log is complete: `store/replay.py` (written independently of the write path) rebuilds the workloads and attempts tables and the books from it exactly.
- Write-ahead: `started` is committed before the backend is called; a crash in between leaves a `STARTING` attempt that R1 repeats.
- Idempotent submission: the key, the specification hash, and the workload id are written in the submission transaction.
- SQLite is the default (a file for the service, in-memory for the virtual driver); PostgreSQL runs the same code (`AWP_PG_DSN`), and the store contract suite runs on both.

## Failure model

| Failure | Detection | Repair |
|---|---|---|
| controller crash at any point | the next controller (same store) | rules R1-R8 on the stored state; write-ahead and idempotency make every repair safe |
| controller paused beyond the lease TTL | lease epoch | its writes fail with `LeaseLost` (fencing) |
| lost or failed `start` call | attempt `STARTING`, absent from snapshots | R1 repeats `start` (idempotent); R2 stops it after `start_timeout_ms` |
| attempt or node lost | absent from snapshots / node not ready | R5 after `lost_grace_ms`; R4 after `node_grace_ms`; counted retries with backoff |
| stop not confirmed | attempt `STOPPING` not terminal | R8 repeats `stop` |
| leftovers in the backend | snapshot entries the store does not know or has ended | R3 stops and forgets them |
| store outage / failed transaction | `StoreUnavailable` | the API answers 503; the tick ends; the client retries with its key |
| policy failure (invalid, error, timeout, crash, malformed) | `PolicyRunner` | counted; after 3 in a row `fifo+first_fit` decides; retries at `min(60 s, 1 s · 2^j)` |
| stale or unavailable snapshots | backend contract, assumption A1 | rules that need a snapshot are skipped; R5 counts its grace from the last snapshot that showed the attempt |

What is assumed rather than handled: snapshots are complete and at most `lost_grace_ms / 3` old (A1); the store is durable; one controller is active per store (the lease); the clock of one service does not go back.

## Deployment shapes

- One process: `python -m ai_workload_platform up` (API + controller), SQLite file, local backend on a scaled clock.
- `docker compose up`: the same with PostgreSQL.
- In a cluster: the manifests in `deploy/k8s/` (Deployment with the Kubernetes backend and in-cluster configuration, least-privilege RBAC). They are validated with kubeconform; see `deploy/README.md` for what was applied.
