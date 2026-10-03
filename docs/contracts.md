# Contracts, version 1

This document is the binding, language-neutral definition of the formats, definitions, and interfaces of `ai-workload-platform`. Other projects follow this document; they do not import the Python package. Changes are deliberate, versioned, and logged.

Version: **1**. Everything here describes a control plane for **simulated** GPU workloads: workloads sleep for their run time and nothing runs a model.

Sources: the shared conventions follow version 1 of the conventions of `llm-serving-control` (its `docs/contracts.md` §1–2) and of `gpu-cluster-scheduler` (its `docs/contracts.md` §1). From `gpu-cluster-scheduler` this project also takes, as documents (no code is shared), the trace schema v1 (§2 there), the cluster configuration v1 (§3), the metric definitions (§5), the external-policy protocol v1 (§6), and the policy semantics (§7). From `distributed-task-platform` it takes the ideas of versioned transitions, leases as fencing tokens, retry with backoff, and a reconciler. It defines its own workload specification, event log, HTTP API, and backend contract below.

## 1. Conventions

- **Time.** The platform reads time only from an injectable clock: integer milliseconds since the start of the run (a virtual clock for tests, simulations, and the fault harness; the system clock, optionally scaled, for the live service, whose origin is kept in the store so that a restarted service continues the same timeline). Domain code never reads the wall clock. Stored times are integer milliseconds; files, reports, and API fields named `*_s` show decimal seconds with at most three decimals. A duration computed from a rate becomes an event time as `ceil(work * 1000 / rate - 1e-6)` ms, so event order never depends on float accumulation.
- **Randomness.** One stream per component: `random.Random(splitmix64(seed XOR fnv1a64(name)))`, using only `random()` and `getrandbits()` (stable across CPython versions); every distribution is built from `random()` (exponential by inversion, normal by Box–Muller with the second value discarded, categorical by one draw against cumulative weights). `fnv1a64` is 64-bit FNV-1a over the UTF-8 bytes of the name (offset `0xcbf29ce484222325`, prime `0x100000001b3`); `splitmix64` is the standard SplitMix64 output function (add `0x9e3779b97f4a7c15`, xor-shift-multiply by `0xbf58476d1ce4e5b9` and `0x94d049bb133111eb`). Known answers (tested): `fnv1a64("") = 0xcbf29ce484222325`, `fnv1a64("a") = 0xaf63dc4c8601ec8c`, `fnv1a64("foobar") = 0x85944171f73967e8`, `splitmix64(0) = 0xe220a8397b1dcdaf`. The streams are **not** bit-identical to the Go streams of `gpu-cluster-scheduler` and `llm-serving-control` (those feed the mixed seed into PCG); a trace generated here differs from one generated there with the same seed. Stream names: `gen:<part>` (trace generator), `retry:<workload id>:<k>` (retry jitter), `faults` (fault schedule), `backend` (local backend), `policy:<name>` (each policy). The global state of the `random` module and `hash()` never influence a result.
- **Units.** GPUs, CPUs, workers, and workloads are integers; memory is GB, compared in whole MB (`round(mem_gb * 1000)`); times are seconds; the unit of work is the reference GPU second (a second of progress on a GPU of speed 1.0); no currency.
- **Identifiers.** A namespace name, a workload id, and a node name are DNS-1123 labels of at most 40 characters (lowercase letters, digits, `-`; start and end alphanumeric). Workload ids are unique across the platform. The id of an attempt is `<workload_id>-a<n>`, `n` from 1.
- **Determinism.** The same configuration, trace, fault schedule, and seed give identical event logs on the virtual clock. Ties break by an explicit total order, never by dict or set iteration order: workloads by (priority high first, submit time, id), nodes by (rack, name), attempts by id. Wall-clock measurements live in fields and columns whose names start with `wall_`; byte-comparison tests exclude them.
- **Errors.** Every API error has the body `{"error": {"code": "...", "message": "...", "details": {}}}` with a stable upper-case `code`.
- **Thresholds** (configuration; the defaults are part of the contract):

| Name | Default | Meaning |
|---|---|---|
| `start_retry_ms` | 5000 | R1: repeat `start` for a `STARTING` attempt the backend does not show |
| `start_timeout_ms` | 120000 | R2: stop an attempt not observed running in time |
| `node_grace_ms` | 30000 | R4: stop attempts on a node not ready for this long |
| `lost_grace_ms` | 30000 | R5: end an attempt absent from every snapshot for this long |
| `stop_retry_ms` | 10000 | R8: repeat `stop` for a `STOPPING` attempt |
| `observe_interval_ms` | 1000 | live driver: tick at least this often |
| `lease_ttl_ms` | 15000 | controller lease time to live |
| `policy_timeout_s` | 60 | external policy: wall-clock timeout per reply |
| `policy_max_failures` | 3 | consecutive policy failures before the fallback takes over |

The fault harness uses shorter thresholds so that most schedules end within 600 virtual seconds; each schedule records the values it used (`docs/failure-injection.md`).

## 2. Workload specification

The body of a submission (the namespace comes from the URL). It maps one to one onto the trace schema v1 of `gpu-cluster-scheduler` (16 columns: `tenant` is the namespace, `user` is `labels.user`, `runtime_s` is `sim.runtime_s`), so a trace can be replayed through the API.

```json
{
  "id": "train-01", "priority": 4,
  "gpus": 4, "workers": 2, "gpu_class": "a100", "topology": "rack", "cpus": 48, "mem_gb": 384,
  "estimate_s": 1800, "preemptible": false, "checkpoint_interval_s": 600, "max_wait_s": 3600,
  "retry": {"max_attempts": 3, "backoff_base_s": 5, "backoff_cap_s": 300, "jitter": "full", "fatal_exit_codes": [2]},
  "labels": {"user": "u1"},
  "sim": {"runtime_s": 1500, "fail_after_s": 20, "fail_attempts": 1, "exit_code": 1}
}
```

| Field | Rule (default) |
|---|---|
| `id` | optional DNS-1123 label (≤ 40); absent → `w<counter>` from a store counter inside the submission transaction |
| `priority` | integer 0–9 (4); higher is more important |
| `gpus` | integer ≥ 1, per worker |
| `workers` | integer ≥ 1 (1); the gang size: all workers start together or the workload does not start |
| `gpu_class` | optional; any class when absent, otherwise the only class it may use |
| `topology` | `any` (default), `rack`, `node`: the tightest domain the workload is sensitive to |
| `cpus` | integer ≥ 0 (0), per worker |
| `mem_gb` | decimal ≥ 0 (0), per worker |
| `estimate_s` | decimal > 0 (3600): the user's run-time estimate; the only run-time information a policy gets |
| `preemptible`, `checkpoint_interval_s` | boolean (false), decimal ≥ 0 (0, no checkpoint): a policy may preempt a preemptible workload; on preemption the work done is kept rounded down to a multiple of `checkpoint_interval_s` (§3) |
| `max_wait_s` | optional decimal ≥ 0: the target for the wait before the first start (a metric, never a rule) |
| `retry.max_attempts` | integer 1–10 (3) |
| `retry.backoff_base_s`, `retry.backoff_cap_s` | decimals ≥ 0 (5, 300) |
| `retry.jitter` | `full` (default) or `none` |
| `retry.fatal_exit_codes` | list of integers 1–255 (empty) |
| `labels` | at most 8 string pairs; keys DNS-1123 labels (≤ 63), values ≤ 63 characters; `labels.user` is the user for runtime predictors |
| `image`, `command` | optional string and list of strings; stored and returned, ignored by the simulated workloads |
| `sim.runtime_s` | decimal > 0, required: the true run time on a speed 1.0 GPU without topology penalty; hidden from policies |
| `sim.fail_after_s`, `sim.fail_attempts`, `sim.exit_code` | decimal > 0 (required when `fail_attempts` > 0), integer ≥ 0 (0), integer 1–255 (1): the first `fail_attempts` attempts, by attempt number, fail with `exit_code` after `fail_after_s` of their own run time; later attempts run to the end and succeed |

Validation rejects unknown fields, wrong types (strict: `"4"` is not an integer, `true` is not a number), decimals with more than three places, and values out of range with `422 INVALID_SPEC`; `details.fields` lists every failing field path (for example `sim.runtime_s`) and `details.problems` the reason per path. The specification is **canonicalized** before it is hashed or stored: every default filled in, keys sorted, integral decimals written as integers (`1500`, not `1500.0`), `fatal_exit_codes` sorted and unique, an empty `gpu_class` written as `null`. The hash is SHA-256 over the canonical JSON (`sort_keys`, separators `,` and `:`, UTF-8). The stored specification carries the assigned `id`; the hash is taken over the specification as submitted (with `id` `null` when absent).

## 3. Lifecycle

Workload states: `QUEUED`, `STARTING`, `RUNNING`, `RETRY_WAIT`, and the terminal `SUCCEEDED`, `FAILED` (a fatal exit code), `DEAD_LETTER` (retries exhausted), `CANCELLED`. Admission is synchronous: an accepted workload is `QUEUED`.

An **attempt** is one try of a workload on a placement. Attempt states: `STARTING`, `RUNNING`, `STOPPING`, `ENDED`. An ended attempt has an end reason: `succeeded`, `failed_retryable`, `failed_fatal`, `node_lost`, `backend_lost`, `start_timeout`, `preempted`, `cancelled`. An attempt holds its placement's GPUs, CPUs, and memory in the store's books from its `started` event until it is `ENDED`, `STOPPING` included (R9).

### Workload transitions (data)

A pair that is not in this table is invalid (`InvalidTransition`); `tests/test_tables.py` parses this table and enumerates every (state, event) pair against the code. `RETRY_WAIT / DEAD_LETTER` means `RETRY_WAIT` with `retry_at_ms`, or `DEAD_LETTER` when the counted attempts have reached `max_attempts`.

| Workload state | Event | Result |
|---|---|---|
| (none) | `submitted` | QUEUED |
| QUEUED | `started` | STARTING |
| STARTING | `running` | RUNNING |
| STARTING | `stop_requested` | (unchanged) |
| RUNNING | `stop_requested` | (unchanged) |
| RETRY_WAIT | `requeued` | QUEUED |
| QUEUED | `cancel_requested` | CANCELLED |
| RETRY_WAIT | `cancel_requested` | CANCELLED |
| STARTING | `cancel_requested` | (unchanged) |
| RUNNING | `cancel_requested` | (unchanged) |
| STARTING | `attempt_ended:succeeded` | SUCCEEDED |
| STARTING | `attempt_ended:failed_fatal` | FAILED |
| STARTING | `attempt_ended:failed_retryable` | RETRY_WAIT / DEAD_LETTER |
| STARTING | `attempt_ended:node_lost` | RETRY_WAIT / DEAD_LETTER |
| STARTING | `attempt_ended:backend_lost` | RETRY_WAIT / DEAD_LETTER |
| STARTING | `attempt_ended:start_timeout` | RETRY_WAIT / DEAD_LETTER |
| STARTING | `attempt_ended:preempted` | QUEUED |
| STARTING | `attempt_ended:cancelled` | CANCELLED |
| RUNNING | `attempt_ended:succeeded` | SUCCEEDED |
| RUNNING | `attempt_ended:failed_fatal` | FAILED |
| RUNNING | `attempt_ended:failed_retryable` | RETRY_WAIT / DEAD_LETTER |
| RUNNING | `attempt_ended:node_lost` | RETRY_WAIT / DEAD_LETTER |
| RUNNING | `attempt_ended:backend_lost` | RETRY_WAIT / DEAD_LETTER |
| RUNNING | `attempt_ended:start_timeout` | RETRY_WAIT / DEAD_LETTER |
| RUNNING | `attempt_ended:preempted` | QUEUED |
| RUNNING | `attempt_ended:cancelled` | CANCELLED |

Rules on top of the table:

- **Cancel wins over a retry.** When `cancel_requested` is set and the attempt ends for any reason except `succeeded`, the workload becomes `CANCELLED`. **Success wins over a cancel**: the cancel request answered 202 and lost the race; the workload shows `SUCCEEDED`. A repeated cancel changes nothing and appends no event; cancelling a terminal workload changes nothing.
- `started` creates attempt `n` = previous attempts + 1 in `STARTING` and reserves its placement; `stop_requested` moves the attempt to `STOPPING` with a reason; `attempt_ended` moves it to `ENDED`.
- **Preemption** (Tier 2; the semantics of `docs/simulator.md` §4 of `gpu-cluster-scheduler`): a policy's `preempt` action is committed as `stop_requested` with reason `preempted`, and the attempt keeps its resources until it has ended (R9). Its `attempt_ended` (reason `preempted`, not counted) returns the workload to `QUEUED` with `retained_s` = the work done (reported by the backend) rounded down to a multiple of `checkpoint_interval_s` (0 without checkpoints; a 1e-9 relative tolerance; never more than the work done) and `preemptions` + 1. Every later start of a preempted workload first holds its GPUs for the cluster's `restart_overhead_s` without progress and then runs the remaining work `runtime_s − retained_s` (local backend; on Kubernetes the overhead is whatever the restart really costs). Lost work = work done − retained.

### Attempt transitions (data)

| Attempt state | Event | Result |
|---|---|---|
| (none) | `started` | STARTING |
| STARTING | `running` | RUNNING |
| STARTING | `stop_requested` | STOPPING |
| RUNNING | `stop_requested` | STOPPING |
| STARTING | `attempt_ended:*` | ENDED |
| RUNNING | `attempt_ended:*` | ENDED |
| STOPPING | `attempt_ended:*` | ENDED |

### Retry table (data)

| End reason | Counted | Retried (if attempts remain) |
|---|---|---|
| `succeeded` | no | no |
| `failed_retryable` | yes | yes |
| `failed_fatal` | yes | no |
| `node_lost` | yes | yes |
| `backend_lost` | yes | yes |
| `start_timeout` | yes | yes |
| `preempted` | no | (returns to `QUEUED`) |
| `cancelled` | no | no |

An infrastructure failure (`node_lost`, `backend_lost`, `start_timeout`) counts as an attempt: without a bound, a workload that keeps killing nodes (or that can never start) would loop forever.

**Backoff.** For the k-th counted failure, `d = min(backoff_cap_s, backoff_base_s * 2^(k-1))` seconds. With `jitter: full`, `d` is multiplied by `u`, the first `random()` of a fresh stream named `retry:<workload id>:<k>` under the platform seed (stored in the store), so that a repeated computation after a crash gives the same value. `retry_at_ms = at_ms + ceil(1000 * d * u - 1e-6)` where `at_ms` is the time of the `attempt_ended` event (`u` = 1 for `jitter: none`). With the defaults and `jitter: none` the delays are:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| delay (s) | 5 | 10 | 20 | 40 | 80 | 160 | 300 | 300 |

## 4. Store and events

- The store is the only source of truth: tables `namespaces`, `workloads`, `attempts`, `events`, `idempotency`, `lease`, `inventory` (the last snapshot the controller stored), `books` (allocated GPUs, CPUs, and MB per namespace and node), `counters` (next `seq`, last `at_ms`, next workload number), and `meta` (instance id, seed, clock origin). Controllers and the API keep only caches rebuilt from it.
- **Every mutation is one transaction**: it checks the expected `version` of the row(s) it changes (compare and set: `UPDATE ... WHERE id = ? AND version = ?`), bumps it by one, and appends exactly one event with the next `seq` (gap-free from 1; events are never updated or deleted). Every event bumps the version of its workload; an event that names an attempt also bumps that attempt. So a workload's version equals the number of its events, and an attempt's version the number of events that name it. A failed check raises `VersionConflict`; the caller re-reads and re-decides and never retries blindly. Writers are serialized (SQLite `BEGIN IMMEDIATE`; PostgreSQL a transaction-level advisory lock), which makes the gap-free `seq` a counter row read and bumped inside the transaction.
- Event time: `at_ms = max(caller's now, last at_ms)`, so `at_ms` never decreases even when two threads read the clock before taking the writer lock.
- An event has `seq`, `at_ms`, `type`, `namespace`, `workload_id`, `attempt_id` (when it applies), and `data`; the API and the saved logs use JSON lines. Every `data` carries `state`, the workload state after the event.

| Type | Data |
|---|---|
| `submitted` | `spec` (canonical, with `id`), `spec_hash`, `idempotency_key` |
| `started` | `attempt` (n), `placement` (`[{"node","workers"}]`), `retained_s`, `overhead_s` (restart overhead of this start) |
| `running` | `observed_started_ms`, `nodes` (observed), `workers_by_node` (observed `[node, workers]`) |
| `stop_requested` | `reason` |
| `attempt_ended` | `reason`, `exit_code`, `counted`, `retry_at_ms`, `observed_started_ms`, `observed_ended_ms`, `nodes`; for `preempted` also `work_done_s` and `retained_s` |
| `requeued` | — |
| `cancel_requested` | `immediate` (true for a `QUEUED` or `RETRY_WAIT` workload) |

```json
{"seq": 12, "at_ms": 41250, "type": "attempt_ended", "namespace": "team-a", "workload_id": "train-01", "attempt_id": "train-01-a1", "data": {"reason": "failed_retryable", "exit_code": 1, "counted": true, "state": "RETRY_WAIT", "retry_at_ms": 44250, "observed_started_ms": 21250, "observed_ended_ms": 41250, "nodes": ["r0-n00"]}}
```

- Every event written by a controller carries `epoch`, the lease epoch it was written under (an audit trail; the epochs in the log never decrease).
- **The log is complete.** A pure function (`store/replay.py`, written independently of the write path) replays it from `seq` 1 and rebuilds the `workloads` and `attempts` tables and the books exactly; the harness compares the replay with the tables after every step (I5). Namespace configuration, the inventory, the lease, and the idempotency table are not event-sourced.
- **Write-ahead.** `started` is committed before the backend call. A crash in between leaves a `STARTING` attempt that the backend never saw; R1 repeats the call.
- **Lease.** One active controller per store: the `lease` row holds `holder`, `epoch`, `expires_ms`. Taking a free or expired lease (or one held under the controller's own name by an earlier incarnation) bumps `epoch`. Every controller write checks inside its own transaction that the row still has its holder and epoch (fencing), and every backend call that can start, stop, or forget work checks the lease row first; a controller that was paused and resumed after another took over fails with `LeaseLost` and stops. The lease is renewed every tick, and again after the policy decided (a policy may take up to `policy_timeout_s`, longer than the TTL).
- **Admission writes** (submissions, cancels) come from the API and are not fenced: they are client commands, serialized with the controller's writes.
- **The `started` transaction** re-checks the namespace cap against the configuration and the books at that moment (a `PUT` may have lowered it after the view was built); a start that would exceed it is dropped as stale. Node capacity is not re-checked there: the lease makes one controller the only writer of starts, and fencing is what keeps a paused controller's stale view from over-allocating a node.

## 5. Namespaces and admission

A namespace has `quota_gpus` (nominal), `cap_gpus` (hard limit, ≥ `quota_gpus`), `max_priority` (0–9), and `max_queued` (workloads in `QUEUED` or `RETRY_WAIT`). `PUT /v1/namespaces/{ns}` creates or updates it; a change affects later admissions and cycles only.

Admission is one transaction together with the insert. The checks run in this order and the first failure is the answer:

| # | Check | Status | Code |
|---|---|---|---|
| 1 | unknown namespace | 404 | `UNKNOWN_NAMESPACE` |
| 2 | `Idempotency-Key` seen with a different canonical specification (an equal one returns the original workload, 200) | 409 | `IDEMPOTENCY_MISMATCH` |
| 3 | invalid specification | 422 | `INVALID_SPEC` |
| 4 | priority above `max_priority` | 403 | `PRIORITY_NOT_ALLOWED` |
| 5 | no inventory stored yet | 503 | `NO_INVENTORY` |
| 5 | can never run on the stored inventory (class, GPUs, CPUs, or memory per worker, or total workers on the empty cluster) | 422 | `UNSCHEDULABLE` |
| 6 | `gpus * workers` above `cap_gpus` | 422 | `EXCEEDS_NAMESPACE_CAP` |
| 7 | `id` exists | 409 | `DUPLICATE_ID` |
| 8 | `max_queued` reached | 429 | `QUEUE_FULL` (header `Retry-After: 5`) |

- An `Idempotency-Key` header has 1 to 64 characters (otherwise 422 `INVALID_REQUEST`). The key, the specification hash, and the workload id are written in the submission transaction, so a crash cannot leave one without the others.
- Rejections are counted in `awp_admission_rejections_total{namespace,reason}` and logged; they are not events.
- "Can never run": some node of the workload's class (any class when unset) has at least `gpus`, `cpus`, and the memory of one worker, and the sum over those nodes of the workers that fit on the empty node reaches `workers`. Every node of the inventory counts, ready or not.

## 6. Policy interface

The policy decides who starts where and never executes anything. One scheduling cycle: the controller builds a view from the store and the last inventory, calls the policy, validates the decision, commits a `started` event for every action (write-ahead), and then calls the backend.

- **Form.** `Policy.schedule(view) -> decision`, where view and decision are dictionaries in the wire form of the external-policy protocol v1 of `gpu-cluster-scheduler` (`job_id` is the workload id, `tenant` the namespace). Built-in policies are called in process; `ExternalPolicy` starts a child process and speaks the protocol (`hello`, `schedule`, `decision`, `bye`, `error`). The in-process view carries two extra keys, `awp_namespaces` (`[{"tenant","quota_gpus","cap_gpus","allocated_gpus"}]`, allocated from the books, `STOPPING` included) and `awp_stopping` (per node the GPUs, CPUs, and MB held by attempts being stopped for a preemption: soon free), which the built-ins use; `ExternalPolicy` removes every `awp_*` key before sending, so the wire form is exactly that of the protocol. A decision's `wake_at_s` is honoured: the controller ticks at that time.
- **Session.** `hello` carries the cluster: classes, racks, `cross_node_factor`, `cross_rack_factor`, `restart_overhead_s`, `preempt_grace_s`, and every node of the inventory (name, rack, class, speed, gpus, cpus, `mem_mb`) in (rack, name) order. A node added to the inventory restarts the session.
- **View.** `now_s`; `nodes` with free GPUs, CPUs, and memory (whole MB) computed from the books, in which every attempt that is not `ENDED` holds its placement, and the running workloads per node; a node that is not ready shows zero free and no running jobs; `pending` = the `QUEUED` workloads ordered by (priority high first, submit time, id), with `wait_s` = now − submit, `retained_s`, `started` (an attempt has ever started), `preemptions`; `running` = the attempts in `STARTING` or `RUNNING`, with `run_start_s` (the observed start, else the `started` event), `overhead_s` (the restart overhead of a start after a preemption, else 0), `retained_at_start_s`, `rate` as the backend reports it (1 when it reports none), `work_done_s` = retained + `max(0, now − run_start − overhead) * rate`, `est_remaining_work_s` = max(0, estimate − work done), and `est_end_s` by the overrun rule of `gpu-cluster-scheduler` (`run_start + overhead + ceil_ms((estimate − retained) / rate)`, or now when that is past); `tenants` with GPU-seconds so far (allocated, from the attempts' `started` to `ended` or now) and running GPUs, CPUs, and memory per namespace; `history_new` = the attempts that ended `succeeded` since the previous call of the session (the first call of a session carries all of them; the policy runner tracks this per session) with the user, the submit time, and the run time the platform measured (observed end − observed start, else event times).
- **Validation** (all or nothing against the view): a `start` names a `QUEUED` workload once; its nodes exist, are ready, and are listed once; worker counts are ≥ 1 and sum to `workers`; capacities (GPUs, CPUs, whole MB) hold when the earlier actions of the same decision are counted; the class constraint holds; and the namespace stays within `cap_gpus` (books plus the earlier actions). A `preempt` names a workload of `running` once, which must be preemptible and is not started in the same decision; it frees that workload's resources for the later actions of the decision (as in `gpu-cluster-scheduler`). The controller still keeps them held until the stopped attempt has ended (R9), so a start that does not fit the actual books yet is deferred (not stale, not a failure) and decided again in a later cycle. Unknown fields in a reply or an action, a wrong `type` or `seq` (an integer, never a boolean), and wrong types are `malformed`; a line cut off by an exit is a `crash`, and so is any other error of the configured policy (for example a command that cannot start).
- **Stale actions.** Commits are per action with compare and set: an action whose commit fails with `VersionConflict` (for example the workload was cancelled after the view was built, or the cap was lowered) is dropped as stale (`awp_stale_actions_total`), is not a policy failure, and the cycle goes on.
- **Failure.** An invalid decision, an `error` reply, a timeout, a crash, or a malformed line is a policy failure: the decision changes nothing, `awp_policy_failures_total{kind}` increases (`invalid`, `error`, `timeout`, `crash`, `malformed`), and a log line names the policy and the reason (with the tail of the child's stderr). After `policy_max_failures` consecutive failures the controller **degrades**: `fifo+first_fit` decides in its place, `/healthz` reports `degraded`, `awp_policy_degraded` is 1, and the configured policy is tried again at times `min(60 s, 1 s * 2^j)` after the previous try on the platform clock (`j` = 0, 1, …; an external policy in a fresh session with a new `hello`). A valid decision at such a retry is applied and ends the degraded state. The platform never aborts because of a policy.
- **External process.** Started without a shell, with binary pipes, `PYTHONUTF8=1`, and `PYTHONDONTWRITEBYTECODE=1`; stdout carries protocol lines only, stderr goes to a log file; every reply is read in a thread with a timeout (`policy_timeout_s`; Windows pipes cannot be polled); the child is closed and reaped on every path (`bye`, error, timeout, shutdown). The command comes from the configuration (`AWP_POLICY_CMD`, `AWP_POLICY_CWD`).

**Built-in policies** (all ignore `topology`, which only slows a workload that spans more than it allows; placements respect class, CPUs, and memory and are complete):

| Name | Order | Placement | When a workload does not fit |
|---|---|---|---|
| `fifo+first_fit` | submit time, id (priority ignored) | nodes in index order, as many workers per node as fit | stop (it blocks the queue) |
| `priority+best_fit` | priority high first, submit time, id (no aging) | worker by worker on the node left with the fewest free GPUs (ties by node index) | skip and go on (no reservation: large workloads can starve) |
| `quota+first_fit` | first the workloads whose namespace stays within `quota_gpus` after the start (counting the earlier actions of the decision), in base order (priority high first, submit time, id); then the others, which borrow up to `cap_gpus` | first fit | skip |

A start that would take a namespace above `cap_gpus` counts as "does not fit" for all of them.

Preempting variants (Tier 2), as in §7 of `gpu-cluster-scheduler`'s contracts: `priority+best_fit+preempt` — after the base pass, each pending workload that was not started and does not fit (even counting the capacity already being stopped for a preemption) may preempt running preemptible workloads with priority at most its own minus 1; victims in order of the work they would lose (GPUs × (work done − retained by the checkpoint rule)), then lower priority, latest start, id, until it fits (at most 8); victims it does not need are then dropped, last added first. `quota+first_fit+reclaim` — a pending workload within its namespace's quota that does not fit may preempt preemptible workloads of namespaces running above their quota, victims chosen the same way. Both start the waiting workload in a later cycle, once the preempted attempts have ended.

## 7. Backend contract

The backend executes and observes; it knows nothing about queues, quotas, retries, or policies. The platform talks to it only through `SchedulerAdapter`:

| Call | Contract |
|---|---|
| `inventory()` | the nodes: name, rack, class, speed, total GPUs, CPUs, memory (MB), `ready` |
| `start(attempt)` | begin the attempt (specification, attempt id, attempt number, placement = workers per node, `retained_s`, `restart_overhead_s`). Idempotent on the attempt id |
| `stop(attempt_id)` | ask for the attempt to end. Idempotent; an unknown or ended attempt is not an error |
| `observe()` | **every** attempt the backend knows, ended ones included until forgotten: `attempt_id`, `phase` (`starting`, `running`, `succeeded`, `failed`, `stopped`), the nodes it occupies, for terminal phases `exit_code` and `reason` (`exit`, `node_lost`, `evicted`), `started_ms` and `ended_ms` when known, `rate` and `work_done_s` (retained + progress) when known, and `incomplete` for a gang whose start left only part of it behind. Complete, never silently partial (a backend that cannot produce a complete snapshot fails the call), but it may be stale or repeat earlier information |
| `forget(attempt_id)` | the platform has recorded the terminal state; the backend may release what it keeps |

- Every call may raise `BackendError`. The platform treats it as transient: `awp_backend_errors_total{op}` increases, the next tick tries again, and it is never a verdict on a workload.
- A terminal phase means the attempt holds nothing on its nodes any more (R9 relies on it).
- **Assumption A1**: a snapshot is at most `lost_grace_ms / 3` old. The harness respects it; E2 measures `observe_lag_ms` on a real cluster to see whether it holds there.

**Local backend.** A model of the nodes on the injected clock. A started attempt runs `sim.runtime_s` of work at `rate = min(speed of the classes it occupies) / f`, with the topology factor `f` from the cluster file as in `gpu-cluster-scheduler` (`docs/simulator.md` §3 there: `f` = 1 when the span is within the workload's `topology`; `cross_node_factor` for a `node`-sensitive workload over several nodes of one rack; `cross_rack_factor` for a `node`- or `rack`-sensitive workload over several racks), and finishes `start latency + ceil(work * 1000 / rate - 1e-6)` ms after its start, or fails as `sim` says (after `fail_after_s` of its own run time, `ceil(fail_after_s * 1000 / rate - 1e-6)` ms). The start latency (default 0 ms) and the stop latency (default 0 ms) are configuration. It does not enforce capacity: an over-allocation by the platform shows up in its own records, where invariant I3 looks for it. Fault hooks for the harness: node down and up (the attempts on a down node freeze and are reported failed with reason `node_lost` when it comes back, as a crashed kubelet would), an attempt crash (exit code 137), an attempt lost (the backend forgets it), `observe` unavailable or stale (within A1), `start` and `stop` failures (with or without effect), a slow start.

**Kubernetes backend**: see `docs/kubernetes.md`.

## 8. Reconciliation

Level-triggered: each tick the reconciler compares the store with one `observe()` snapshot and repairs the differences. It assumes nothing about whether the last request arrived; every repair is idempotent; timeouts run on the platform clock. Rules, applied in the order R6, R5, R4, R3, R2, R1, R8, R7:

| Rule | Condition | Repair |
|---|---|---|
| **R1** Lost start | an attempt in `STARTING` that the snapshot does not show (or shows as an `incomplete` gang), `start_retry_ms` after its `started` event or after its last start call | `start` again (idempotent) |
| **R2** Start timeout | an attempt in `STARTING` not observed `running` within `start_timeout_ms` of its `started` event | stop request, reason `start_timeout` |
| **R3** Orphans | an attempt in the snapshot that the store does not know, or whose store attempt is `ENDED` | `stop`, and `forget` once its phase is terminal |
| **R4** Node loss | an attempt (not `ENDED`, not `STOPPING`) on a node that the inventory has shown not ready for `node_grace_ms` | stop request, reason `node_lost` |
| **R5** Lost attempt | an attempt in `RUNNING` or `STOPPING` absent from every successful snapshot for `lost_grace_ms`, counted from the latest of: the last snapshot that showed it, its entry into the state, and the start of this controller | end it with its stop reason when a stop was requested, else `backend_lost`. The only rule by which the platform ends an attempt without the backend saying so |
| **R6** Observed progress | a phase `running` or terminal for an attempt in `STARTING` | record `running` first (observed `started_ms`); then a terminal phase ends the attempt: `succeeded` always wins (even after a stop request); `failed` maps by its reason (`exit` → `failed_retryable`, or `failed_fatal` for an exit code in `fatal_exit_codes`; `node_lost` and `evicted` → `node_lost`), but the stop reason wins when a stop was requested; `stopped` ends with the stop reason, or `backend_lost` when none was requested. Then `forget` |
| **R7** Retry release | a `RETRY_WAIT` workload whose `retry_at_ms` has passed | `requeued` → `QUEUED` |
| **R8** Stops | a workload with `cancel_requested` whose attempt is not yet `STOPPING` | stop request, reason `cancelled`; and an attempt in `STOPPING` that is not terminal `stop_retry_ms` after the last `stop` call (or with no `stop` call known to this controller) gets `stop` again. A stop request is committed (`stop_requested`) before `stop` is called |
| **R9** Holding | — | an attempt's resources stay held until it is `ENDED`, `STOPPING` included; they are freed only when the backend said the attempt is terminal (R6), or by R5 |

## 9. Evaluation metrics

One function (`observability/evaluation.py`, CLI `report`) computes them from the event log alone, with the cluster and the namespace configuration, so the numbers can be recomputed from a saved log. A metric that has a name in `gpu-cluster-scheduler` (its `docs/contracts.md` §5) has the same definition:

- **Window**: the workloads that reached a terminal state, in submit order, without the first and the last `floor(n / 10)`. Workloads that did not succeed are left out of `wait`, `jct`, and `bsld` and counted by terminal state.
- `wait` = first `started` event − submit; `jct` = completion (the terminal `attempt_ended`) − submit (event times: the platform's knowledge); `ideal` = `sim.runtime_s` / speed of the fastest class the workload may use; `bsld = max(1, jct / max(ideal, 10 s))`. Percentiles are nearest rank (the `ceil(p n)`-th smallest).
- `utilization` = allocated GPU-seconds / (total GPUs × the window's span), the span running from the submit time of the first to that of the last window workload (when that span is empty: to the last completion). "Allocated" means held by an attempt that is not `ENDED` (from `started` to `attempt_ended`).
- `jain_bsld` = `(Σx)² / (n Σx²)` over the per-namespace mean `bsld`; `slo_attainment` = window successes with `wait <= max_wait_s` / window successes that have a `max_wait_s`.
- Quota metrics over the span, with usage = allocated GPUs of the namespace, demand = usage + GPUs of its `QUEUED` workloads, quota = `quota_gpus`: `quota_satisfaction` = `Σ_ns ∫ min(usage, demand, quota)` / `Σ_ns ∫ min(demand, quota)`; `borrowed_gpu_hours` = `Σ_ns ∫ max(0, usage − quota)` in GPU-hours; `jain_weighted_bsld` = `(Σ w b)² / (Σ w · Σ w b²)` over the per-namespace mean `bsld` `b`, with the shares `w` = `quota_gpus / Σ quota_gpus`.
- Platform metrics (whole run): `running_gpu_seconds` (attempts from `running` to `attempt_ended`) and `allocated_gpu_seconds` (from `started` to `attempt_ended`) — the difference is the cost of starting, stopping, and lost time; `admit_to_running_ms` = `running` − `started` per attempt (mean, P50, P95; on the virtual driver the controller polls every `observe_interval_ms` while a start is in flight, so it includes that observation delay); `observe_lag_ms` = `attempt_ended` − the backend's `ended_ms` (mean, P95, max); `placement_match` = matched workers / placed workers, comparing the intended placement with the per-node worker counts of the `running` event (`pinned` must give 1.0; the local backend always gives 1.0); `makespan_s` = last terminal event − first submission.
- Counters: attempts by end reason, retries (`requeued` events), dead letters, counted failures. Policy failures and stale actions are not in the log; the benchmark reports them from the controller's counters.

### Worked example (asserted by `tests/test_metrics_by_hand.py`)

One node `n0` with 4 GPUs of speed 1.0; namespaces `a`, `b`, `c` with `quota_gpus` 2, 1, 1 (shares 0.5, 0.25, 0.25) and cap 4. Ten single-GPU workloads (times in seconds; `w08` is cancelled while queued at 80 s):

| id | ns | submit | start | end | runtime | max_wait | wait | jct | ideal | bsld |
|---|---|---|---|---|---|---|---|---|---|---|
| w01 | a | 0 | 0 | 100 | 100 | – | (outside the window) | | | |
| w02 | b | 10 | 10 | 60 | 50 | 5 | 0 | 50 | 50 | 1 |
| w03 | c | 20 | 20 | 40 | 20 | – | 0 | 20 | 20 | 1 |
| w04 | a | 30 | 30 | 230 | 200 | 60 | 0 | 200 | 200 | 1 |
| w05 | a | 40 | 40 | 45 | 5 | – | 0 | 5 | 5 | max(1, 5/10) = 1 |
| w06 | b | 50 | 60 | 160 | 100 | 5 | 10 | 110 | 100 | 1.1 |
| w07 | c | 60 | 100 | 150 | 50 | – | 40 | 90 | 50 | 1.8 |
| w08 | c | 70 | – | 80 (cancelled) | 30 | – | (not succeeded) | | | |
| w09 | b | 80 | 150 | 250 | 100 | 100 | 70 | 170 | 100 | 1.7 |
| w10 | a | 90 | 160 | 170 | 10 | – | (outside the window) | | | |

- Window: n = 10 terminal workloads, `floor(10/10)` = 1 dropped at each end → w02…w09 (8 workloads, 7 succeeded, 1 cancelled).
- `wait`: mean 120/7 = 17.143 s; P50 = 4th smallest of (0, 0, 0, 0, 10, 40, 70) = 0; P95 = `ceil(0.95·7)` = 7th = 70 s.
- `jct`: (5, 20, 50, 90, 110, 170, 200): mean 645/7 = 92.143 s, P50 = 90, P95 = 200.
- `bsld`: (1, 1, 1, 1, 1.1, 1.7, 1.8): mean 8.6/7 = 1.2286, P50 = 1, P95 = 1.8.
- Span = [10 s, 80 s] = 70 s. Allocated GPUs: 2 in [10, 20), 3 in [20, 30), 4 in [30, 45), 3 in [45, 80) → 20 + 30 + 60 + 105 = 215 GPU-s; `utilization` = 215 / (4 × 70) = 0.7679.
- Per-namespace mean `bsld`: a = (1 + 1)/2 = 1; b = (1 + 1.1 + 1.7)/3 = 1.2667; c = (1 + 1.8)/2 = 1.4. `jain_bsld` = 3.6667² / (3 × (1 + 1.6044 + 1.96)) = 13.4444 / 13.6933 = 0.9818.
- `slo_attainment`: w02 (0 ≤ 5), w04 (0 ≤ 60), w09 (70 ≤ 100) meet their targets, w06 (10 > 5) does not → 3/4 = 0.75.
- Quota metrics in [10, 80]: usage a = 1, 2, 3, 2 in [10, 30), [30, 40), [40, 45), [45, 80); b = 1 throughout; c = 1 in [20, 40). Queued GPUs: b 1 in [50, 60) (w06); c 1 in [60, 80) (w07) and 1 in [70, 80) (w08). `∫ min(usage, demand, quota)` = a 120 + b 70 + c 20 = 210; `∫ min(demand, quota)` = a 120 + b 70 + c (20 + 10 + 10) = 230; `quota_satisfaction` = 210/230 = 0.9130. a borrows 1 GPU in [40, 45): `borrowed_gpu_hours` = 5/3600 = 0.001389.
- `jain_weighted_bsld` = (0.5·1 + 0.25·1.2667 + 0.25·1.4)² / (1 × (0.5·1 + 0.25·1.6044 + 0.25·1.96)) = 1.3611 / 1.3911 = 0.9784.
- `makespan_s` = 250 − 0 = 250.

## 10. Generator (assumed distributions)

The trace generator (`bench/generator.py`, CLI `gen`) writes trace schema v1 with a manifest. Its distributions are assumptions shaped like those of `gpu-cluster-scheduler` and simplified; every parameter is in `DEFAULTS` there and in `benchmarks/README.md`. Tolerances checked by `tests/test_inputs.py` over 20 seeds of 300 workloads: the mean realized offered load within ±10 % of the target, each namespace share within ±0.03 of its configured share, and each size class within ±0.03 of its probability.

## 11. Conformance with gpu-cluster-scheduler (Tier 2 item 3)

The same trace (generator `balanced`, load 0.8, 200 workloads, trace seeds 1–3, the cluster file `configs/clusters/reference.json`) ran through this platform — local backend, virtual clock, zero start latency — and through the simulator of `gpu-cluster-scheduler` (built from its repository outside this one; nothing copied). On the platform side the policy was its Python policy server behind `ExternalPolicy` (`AWP_POLICY_CMD` / `AWP_POLICY_CWD`, protocol v1, no change to the server), and for `fifo+first_fit+none` also the built-in `fifo+first_fit`. Start times and placements come from the simulator's assignment log; its end times follow from the logged start and placement by its documented model (rate = min speed / topology factor, end = start + `ceil(runtime * 1000 / rate − 1e-6)` ms). `scripts/conformance_07.py` writes `benchmarks/results/conformance/conformance.csv`.

| Policy of gpu-cluster-scheduler | Platform side | Workloads compared | Start differences | End differences | Placement differences |
|---|---|---|---|---|---|
| `fifo+first_fit+none` | its policy server through the protocol | 3 × 200 | 0 | 0 | 0 |
| `fifo+first_fit+none` | built-in `fifo+first_fit` | 3 × 200 | 0 | 0 | 0 |
| `fifo+first_fit+easy` | its policy server through the protocol | 3 × 200 | 0 | 0 | 0 |

No difference had to be explained. What makes them agree: the platform's virtual driver invokes the policy at the same instants (completions, then submissions, then one cycle; and at a requested `wake_at_s`), the view carries the same fields and orders, and the local backend uses the same rate and rounding rules. Not covered: preempting and MILP policies of that project, failures, and traces with class constraints.
