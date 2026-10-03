# Failure injection

The harness (`src/ai_workload_platform/faults/`) runs seeded schedules on the virtual clock: a generated workload list with client actions, the platform on the virtual driver over an in-memory SQLite store and a backend (the local backend or the Kubernetes backend on the fake client), and faults drawn from the schedule seed. After every step it checks the invariants below, with the backend's own records as ground truth where noted. Everything is **simulated**: the workloads sleep on a virtual clock.

```text
python -m ai_workload_platform faults --seed 7 --backend local            # one schedule; prints the violated invariant and the tail of the event log
python -m ai_workload_platform faults --seeds 1-200 --backend kube-fake --workers 4
python -m ai_workload_platform faults --seeds 1-200 --backend local --bug no_version_check --workers 4
```

## Schedules

A schedule is a pure function of its seed and backend (`faults/schedule.py`); the same seed gives a byte-identical event log, in one process and across spawned worker processes (`tests/test_faults.py`, `tests/test_determinism.py`).

- **Workloads and client actions** (stream `harness:workloads`): 20 to 40 workloads submitted within the first 300 virtual seconds; 1, 2, 4, or 8 GPUs per worker, 15 % two-worker gangs, run times 5–120 s, `max_attempts` 1–4, backoff base 1–3 s and cap 10–30 s, full jitter for 70 %, a fatal exit code for half of them, and 25 % that fail on their first one or two attempts (with exit code 1, or the fatal 2). Every submission carries an `Idempotency-Key`; 25 % of the workloads are submitted again with the same key and specification one to three times later, and 15 % are cancelled by the client at a random later time. A client whose request failed with a store error (or whose reply was lost in a crash after the commit) retries with the same key.
- **Policy**: one of the three built-ins or the two preempting variants (`priority+best_fit+preempt`, `quota+first_fit+reclaim`) per schedule; half of the workloads are preemptible, with checkpoint intervals of 0, 5, or 20 s.
- **Backend parameters**: local start latency 0–500 ms and stop latency 0–1000 ms; the fake uses its assumed latencies (`docs/kubernetes.md`).
- **Thresholds** (shorter than the defaults so that most schedules end within 600 virtual seconds — in E1, 75 of 2000 local and 29 of 2000 fake schedules took longer, the longest until 802 s; recorded in every schedule):

| `start_retry_ms` | `start_timeout_ms` | `node_grace_ms` | `lost_grace_ms` | `stop_retry_ms` | `lease_ttl_ms` | `policy_max_failures` |
|---|---|---|---|---|---|---|
| 2000 | 20000 | 5000 | 6000 | 3000 | 5000 | 3 |

- **Faults** (stream `faults`): one to six per schedule, at times in [5 s, 400 s]:

| Fault | Backends | What happens |
|---|---|---|
| `crash` | both | at the first hit of a named crash point (chosen at random) after the fault time, the controller raises; the open transaction rolls back; a new controller object with no memory restarts over the same store and backend |
| `zombie` | both | at an interleaving point (`cycle.after_view`, `rules.before_write`, `tick.after_lease`, or just before a backend `start` call) the controller pauses longer than the lease TTL; another controller takes the lease and works for 0–30 s (while 0–5 urgent priority-9 submissions arrive through the API); then the paused one continues its tick; its writes must fail with `LeaseLost` |
| `store_tx_fail` | both | the next 1–3 write transactions fail before they commit |
| `store_outage` | both | every transaction fails for 1–10 s (API calls answer `STORE_UNAVAILABLE`) |
| `observe_down` | both | `observe` fails for 1–21 s (local: `BackendError`; fake: the list calls answer 503) |
| `observe_stale` | both | for 2–20 s every other `observe` returns a snapshot that is 0.2–2 s old (≤ `lost_grace_ms / 3`, assumption A1): a flapping, differently cached view |
| `node_down` | both | a node goes down for 1–41 s (its attempts freeze and are reported lost when it returns) |
| `attempt_crash`, `attempt_lost` | local | a running attempt exits with 137; the backend loses an attempt's record |
| `pod_evicted`, `job_pending` | fake | a running pod is evicted; the pods of a future attempt's Job are never scheduled |
| `start_fail`, `stop_fail` | local | the next 1–3 `start` (or `stop`) calls fail, with or without having taken effect |
| `slow_start` | local | a future attempt starts 1–41 s late (sometimes beyond `start_timeout_ms`) |
| `api_errors` | fake | the next 1–5 API calls answer 500 |
| `policy_fault` | both | for 1–31 s every policy call fails as `invalid`, `error`, `timeout`, or `crash` (a wrapper around the built-in) |
| `race_cancel` | both | at the next interleaving point after the fault time the client cancels a workload — half of the time exactly the workload the controller is about to write (a cancel against a completion, a retry release, or a start) |
| `cap_change` | both | a namespace's cap drops to 4–16 GPUs for 5–65 s |
| `preempt` | both | the next decision also preempts a random running preemptible workload (with the stop latency, this races the preemption against the attempt's completion) |

- **Settle bound**: after the last fault the schedule runs until every workload is terminal and nothing is left in the backend, or until `settle_ms` = Σ over workloads of run time × workers × `max_attempts` + workloads × max(`max_attempts`) × (`start_timeout_ms` + `node_grace_ms` + `lost_grace_ms` + 3 × `stop_retry_ms` + 30 s) + `lease_ttl_ms` + 120 s has passed. If nothing can change any more (no deadline, no backend event, no client action, no fault) before that, the run stops and the convergence invariant decides.

## Crash points and interleaving points

| Crash point | Where |
|---|---|
| `submit.after_commit` | after the submission transaction committed (the client does not get the reply and retries with its key) |
| `start.after_commit` | after the `started` commit, before the backend `start` call (write-ahead) |
| `start.after_call` | after the backend `start` call |
| `ended.before_forget` | after a terminal observation was committed (`attempt_ended`), before `forget` |
| `stop.before_call` | between a committed stop request (cancel, node loss, start timeout) and the `stop` call |
| `requeue.before_commit` | inside the retry release, before `requeued` is committed |
| `stopped.before_commit` | after the backend confirmed a stop, inside the transaction that commits `attempt_ended` |

The interleaving points (`tick.after_lease`, `rules.before_write`, `cycle.after_view`) sit outside transactions; there the harness runs another actor's step: a client cancel or another controller's ticks. Production code runs all of them as no-ops.

## Invariants

I1–I6 and I8 are checked after every step; I7 when the run ends, after the settle period or as soon as everything is terminal and nothing is left (`faults/invariants.py`). E1 reports the convergence time from the later of the end of the last fault and the end of the 300 s submission window, so it includes the remaining run time of the workloads.

- **I1** A terminal state is never left, and a workload has exactly one terminal event (an `attempt_ended` whose resulting state is terminal, or an immediate `cancel_requested`).
- **I2** A workload has at most one attempt that is not `ENDED`; a terminal, `QUEUED`, or `RETRY_WAIT` workload has none.
- **I3** In the backend's own records no node holds more GPUs, CPUs, or memory than it has (local backend; the fake's scheduler never over-commits, so there only GPUs are checked); the store's books equal the sum over the attempts that are not `ENDED`; and every attempt the backend still runs has a store attempt that is not `ENDED`. This last part is checked **more strictly** than the task text: an attempt the store has `ENDED` while the backend still runs it is a violation at once, because the platform ends an attempt only on the backend's terminal report (which means it holds nothing) or after `lost_grace_ms` of absence from snapshots that are at most `lost_grace_ms / 3` old — so under A1 this cannot happen without a bug. Attempts the store never knew get R3's grace of `lost_grace_ms`.
- **I4** No `started` event takes a namespace above its `cap_gpus` as configured at that moment (the harness records every cap change).
- **I5** The log is gap-free from 1, `at_ms` never decreases, every row's version equals the number of events that changed it, and the replay equals the tables (the replay is applied incrementally — the same fold as a replay from `seq` 1 — and compared with the tables after every step). Every event a controller writes carries the lease `epoch` it was written under; the epochs in the log never decrease (a controller that lost the lease wrote nothing after the takeover).
- **I6** The counted attempts of a workload never exceed `max_attempts`; a `DEAD_LETTER` workload has exactly `max_attempts` counted attempts and its last attempt ended for a counted reason; every `retry_at_ms` equals the documented formula for its seed and lies within `backoff_cap_s` of the end of the attempt.
- **I7** Convergence: after the settle period every workload is terminal, no store attempt is left that is not `ENDED`, and the backend keeps nothing of this instance.
- **I8** Submitting the same key and specification any number of times, with crashes between, gives one workload and one `submitted` event.

## Injected bugs

Each is a test-only patch — a context manager that replaces one function (`faults/bugs.py`), never a flag in production code — and must be caught within 200 schedules (`tests/test_faults.py`; the E1 numbers are in `benchmarks/README.md`):

| Bug | Patch | How it shows |
|---|---|---|
| 1 `release_at_stop` | `ops.books_delta` releases the books at the stop request instead of at the end (breaks R9) | I5 (the replay keeps the books until `attempt_ended`), or I4 when a start counted the released GPUs and took the namespace above its cap first (once per backend in the full evaluation); I3's books check would also see it, but runs later |
| 2 `no_epoch_check` | `Tx.check_fence` does nothing | I5: a paused controller writes under an older epoch after the takeover. Compare-and-set alone already rejects most of its stale writes (the controller that took over has changed the same rows); the fence is what stops writes whose rows the other controller did not touch, such as starts on capacity the other controller gave to urgent work |
| 3 `no_lost_grace` | `rules.r5_due` always answers yes (R5 without its grace) | I3: on a flapping stale snapshot an attempt that was just seen running is missing; it is ended while the backend still runs it |
| 4 `no_version_check` | `Tx.cas_update` drops `AND version = ?` | I5: a stale write overwrites a cancel or another change (the replay then fails or differs) |

## Bugs found in the platform during development

These were found by the harness and the rule tests while the platform was built; each was fixed. Bug 2's schedules are a regression test (`tests/test_faults.py`) and bugs 3 and 4 have rule tests; bug 1's schedule came from an earlier version of the generator, so its fix is covered only by the harness as a whole.

1. **A lost race left no deadline** (harness, schedule 176 on the Kubernetes fake in an early version of the schedule generator). A client cancel landed at `rules.before_write` just before R6 wrote the end of the same workload's attempt; the write failed with `VersionConflict` (correct: re-read and re-decide), but the controller then had no deadline left, so the virtual driver never ticked again and the attempt stayed `RUNNING` while the backend reported it `failed` (I7). A live driver would have re-ticked within `observe_interval_ms`. Fix: after a conflict, and whenever the last snapshot shows work the store has not recorded (a terminal phase, an unrecorded `running`, a pending cancel), the controller asks to be woken at once.
2. **A store error at the end of a tick left no deadline** (harness, full evaluation on commit 012db44: schedules 1487 and 1704 on the local backend and 1704 on the Kubernetes fake, 3 of 4000). A `store_tx_fail` fault hit the transaction that R6 used to record a completion; the tick ended with the store error (correct: the next tick starts from the store again), but the controller kept the deadline computed before, which was empty, so the virtual driver never ticked again and the workload stayed `RUNNING` (I7). A live driver re-ticks within `observe_interval_ms`. Fix: a tick that ends with a store error asks to be woken one `observe_interval_ms` later; the full evaluation was then run again on the fixed commit.
3. **R5 forgot what it ended** (found while validating the harness against injected bug 3). R5 called `forget` after ending a lost attempt; on the local backend `forget` kills the attempt, which hid a premature R5 completely. The task's rules leave such leftovers to R3 (stop, then forget once terminal), so R5 no longer forgets.
4. **Controller start at time 0** (rule test for R5). `start_ms or now` treated a controller started at virtual time 0 as missing, so R5 counted its grace from "now" and never fired. Fixed with an explicit `None` check (the same pattern was removed elsewhere).

## What the harness cannot reach

- **True concurrency.** A schedule is single-threaded; interleavings happen only at the named points. Races inside a transaction, or between two controllers that are both active at the same instant, are not explored (writers are serialized by the store, and the lease makes one controller the writer).
- **The real Kubernetes API and real PostgreSQL.** The harness uses the fake client and in-memory SQLite. PostgreSQL runs the same store code in the store contract suite (check 2), and the Kubernetes backend runs the contract suite on a real kind cluster (check 5); neither runs under the harness.
- **Snapshots staler than A1.** A snapshot older than `lost_grace_ms / 3`, or a partial snapshot, would let R5 end attempts that still run; the backend contract excludes both. On the kind cluster the platform recorded the end of a pod 3.4–3.6 s after it happened on average and 4.8 s at most (E2 c and the time-scale sweep), within A1's 10 s at the default `lost_grace_ms` of 30 s.
- **Durability failures of the store** (a lost committed transaction, a corrupted file) and clock skew between processes.
- **Real external-policy processes.** Policy faults are injected by an in-process wrapper; the subprocess paths (timeouts, crashes, malformed lines, reaping by PID) are tested separately (check 4).
- **Node loss on a real cluster** is covered by one demonstration on kind, outside the harness (`docs/kubernetes.md`, Kubernetes depth): R4 requested the stop (`node_lost`) 30 s after the inventory showed the stopped worker not ready; the attempt ended only after the node came back, and the retry then ran.
