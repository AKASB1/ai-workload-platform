# Kubernetes backend

The Kubernetes backend runs each attempt as Kubernetes Jobs. The GPUs are **simulated**: the worker nodes advertise an extended resource (`nvidia.com/gpu`) through a node status patch, and every workload container runs `sleep` for its run time and then exits with the code the attempt ends with. Nothing runs a model, and no device plugin is involved.

Code: `src/ai_workload_platform/scheduler/kube/` — `client.py` (the `KubeClient` interface and the official-client implementation), `fake.py` (the deterministic fake), `backend.py` (`KubeBackend`).

## Mapping

| Platform | Kubernetes |
|---|---|
| attempt `<workload>-a<n>` with placement `[(node, workers), ...]` | one Job per placement entry, named `<attempt id>-<k>` (k from 0), in one namespace (default `awp-workloads`) |
| workers on a node | `completionMode: Indexed`, `completions` = `parallelism` = the workers on that node, `backoffLimit: 0`, `restartPolicy: Never`, `imagePullPolicy: IfNotPresent`, `terminationGracePeriodSeconds: 2` |
| GPUs per worker | `resources.requests` = `resources.limits` = `nvidia.com/gpu: <gpus>` (configurable), plus `cpu: 10m` and `memory: 16Mi` requests |
| `pinned` mode | `nodeSelector: {awp.local/node: <node>}` puts each Job on the node the policy chose |
| `delegate` mode | no node selector; the kube-scheduler places the pods. The store keeps the intended placement (what the books reserve) and the observed nodes (`placement_match` in the reports); a difference is reported, not repaired |
| simulated run | `sh -c "sleep <work / rate / time_scale>; exit <code>"` with busybox; `code` is 0, or the `sim.exit_code` for the first `sim.fail_attempts` attempts (after `fail_after_s / rate / time_scale`) |
| labels on every Job and pod | `app.kubernetes.io/managed-by=ai-workload-platform`, `awp.local/instance` (an id of this platform instance, kept in the store's `meta` table), `awp.local/workload`, `awp.local/attempt`, `awp.local/namespace`; annotations `awp.local/gang-size` and `awp.local/intended-node` on the Job |

**Calls.** `start` creates the Jobs and treats `409 AlreadyExists` as success only when the existing Job is listed under this instance's selector with this attempt's label (a leftover of another store with the same name is an error). If a create fails for another reason, the Jobs created by that call are deleted again and the call raises `BackendError`, so the attempt is absent from the next snapshot and R1 repeats the start; if only part of a gang is left behind anyway (a crash between two creates, a timed-out create that did succeed), `observe` reports the attempt `starting` with `incomplete`, and R1 starts it again (idempotent). `stop` deletes the attempt's Jobs with background propagation unless the whole gang has already completed or failed (so a success stays visible); a partial gang is never "ended". `forget` deletes whatever is left. Every one of these calls is fenced by the lease (the controller checks the lease row first). For a preempted attempt the next start sleeps only for the remaining work (`runtime_s − retained_s`, divided by the rate and `time_scale`); the restart overhead is whatever the real restart costs, and the backend reports the work done from the pod start times and the Job annotations (`awp.local/rate`, `awp.local/retained-s`, `awp.local/time-scale`). `observe` lists the Jobs and pods of this instance by label — level-triggered, nothing depends on a watch — and builds one entry per attempt:

| Phase | Condition |
|---|---|
| `starting` | Jobs exist and not all pods run yet (a partially started gang is `starting` as a whole) |
| `running` | all Jobs of the gang exist and all their pods run (the observed start is the latest container start) |
| `succeeded` | all Jobs of the gang are `Complete` (this wins over a stop that raced with the completion) |
| `failed` | a Job failed and none of the attempt's pods is still active; the reason comes from the pod that explains the failure (not deleted by the platform or the Job controller; a node or eviction reason first; then the earliest to finish): `evicted` (`status.reason` `Evicted` or a `DisruptionTarget` condition), `node_lost` (`NodeLost`, `NodeShutdown`, `Terminated`, ...), else `exit` with the container's exit code. When one Job of a gang fails, the backend deletes the gang's surviving Jobs (gang semantics) and reports `running` until their pods are gone |
| `stopped` | the attempt was deleted by `stop` of this process and no Job or pod of it is left (a backend process that restarted has forgotten its stops; the attempt is then absent and R5 ends it with its stop reason) |

A terminal phase therefore always means that the attempt holds nothing on its nodes (R9). Labelled objects of this instance that the store does not know come out as unknown attempts; R3 stops and forgets them.

**Inventory.** The nodes labelled `awp.local/node` (by `scripts/kind_setup.py`), with rack and class from the labels `awp.local/rack` and `awp.local/class`, GPUs from the advertised extended resource in `status.capacity` (as the task specifies; the kube-scheduler uses `allocatable`, which equals it on the kind workers), readiness from the `Ready` condition only (a cordoned node keeps running its pods, so it stays ready), and CPU and memory from the cluster configuration (`configs/clusters/kind.json`; a kind node reports the host's, which would be misleading). A cluster whose nodes advertise no GPUs is not usable, and `inventory()` fails with that message.

## The fake client

`FakeKubeClient` implements the same `KubeClient` interface with Jobs and pods in memory on the virtual clock, so the contract suite and the fault harness run without a cluster. It models only what this backend uses:

- Indexed Jobs with `backoffLimit: 0`: one pod per completion; a Job is `Failed` as soon as one pod failed (and, like the Job controller, the fake then deletes the Job's other pods) and `Complete` when all succeeded.
- A stand-in for the kube-scheduler: a pod binds `schedule_ms` after its creation (and whenever capacity frees later); `pinned` pods only to their selected node, `delegate` pods to the ready node with the most free GPUs (ties by name); a pod that fits nowhere stays `Pending`. Capacity counts the GPUs of bound pods that are not terminal, terminating pods included.
- A pod runs `container_start_ms` after binding and ends after the `sleep` of its command with the command's exit code.
- Background deletion: a running pod disappears `delete_ms` after the deletion (the termination grace), others at once.
- Faults for the harness: a node down (its pods freeze in their last state; deletions on it do not complete) and up (its frozen pods fail with reason `NodeLost`, its terminating pods vanish), an evicted pod (`Failed`, reason `Evicted`), a Job whose pods are never scheduled, and API errors (the next n calls answer 500).

Latencies (virtual milliseconds; **assumed defaults**; the measured values and a run with them are in [Kubernetes depth](#kubernetes-depth-tier-2-item-4) — the defaults stay, because the measured start latency changes a 2000-workload run by less than 0.01 %):

| Parameter | Default | Note |
|---|---|---|
| `schedule_ms` | 50 | create → bound |
| `container_start_ms` | 1000 | bound → running. The kind spike measured about 6 s from create to running for its first 2-pod Job; in E2 (c) the platform recorded 974 ms from its `started` commit to its `running` commit (pinned, median of 3 run means), which includes up to one observe interval |
| `delete_ms` | 2000 | deletion of a running pod → gone (`terminationGracePeriodSeconds: 2`) |

The same contract tests (`tests/test_backend_contract.py`) run against the fake and against a real cluster (when `AWP_KUBECONFIG` points at one); that is how the fake's fidelity is checked.

## Not modelled

- **Device plugins.** GPUs are an advertised extended resource; no container sees a GPU.
- **Gang admission** (Kueue, Volcano, JobSet). A gang is atomic only at the platform level: the platform reserves the whole placement in its books, but in `delegate` mode the kube-scheduler places pods one by one and can leave part of a gang `Pending` (the partial-gang test shows it). Such an attempt stays `starting` and R2 ends it after `start_timeout_ms` (demonstrated on kind below).
- Pod restarts, `podFailurePolicy`, preemption by the kube-scheduler, taints, priorities, resource quotas, and the node lifecycle controller's eviction of pods from unreachable nodes (the fake keeps them frozen until the node returns).
- Watches: the backend lists by label on every `observe` (level-triggered), which costs two list calls per tick.

## Kubernetes depth (Tier 2 item 4)

On the local kind cluster (kind v0.33.0, Kubernetes v1.37.0, 4 workers with 8 advertised GPUs each; simulated GPUs: the pods sleep), measured on a shared desktop machine (one other agent run active, one service at a time; CPU load 60 % sampled once, before E2 (c); the sweep and the demonstrations have no sample of their own), with the platform at commit `3b6e03b` (`scripts/kind_partial_gang.py` was added afterwards and changes no platform code). Trace runs: median [min, max] of 3; the demonstrations ran once.

| Experiment | Setting | Result |
|---|---|---|
| Time-scale sweep: trace `balanced`, load 0.8, seed 1, 40 workloads, `fifo+first_fit`, `pinned` (`bench --cluster --sweep`) | longest workload 120 s real (`time_scale` 60) | makespan 11 157 [11 114, 11 165] trace s, mean wait 517 [515, 517] s, running / allocated 0.955, 186 s wall |
| | longest 60 s (`time_scale` 120; E2 c) | 12 395 [12 278, 12 455] s, 1 043 [1 009, 1 058] s, 0.926, 103 s wall |
| | longest 30 s (`time_scale` 240) | 14 108 [13 889, 14 111] s, 1 894 [1 738, 1 913] s, 0.886, 59 s wall |
| | the local backend, same trace (virtual clock) | 10 526 s, 523 s, 0.999 |
| `delegate` against `pinned` (E2 c, `time_scale` 120) | the kube-scheduler places the pods | makespan 12 375 [12 117, 12 500] s, mean wait 1 090 [1 086, 1 173] s, admit → running 1 510 [1 149, 1 586] ms (pinned 974), placement match 0.22 [0.20, 0.37], running / allocated 0.799 [0.796, 0.881] |
| Partial gang (`scripts/kind_partial_gang.py`, `delegate`, `start_timeout_ms` 30 s) | pods of another tenant hold all GPUs of three workers; the platform's books do not see them; a 2 × 8-GPU gang | worker 0 `Running` on the free node, worker 1 `Pending`; R2 requested the stop (`start_timeout`, counted) at 30.0 s, the attempt ended at 33.1 s; the other pods were deleted, and the retry ran both workers from 39.1 s and `SUCCEEDED` |
| Node loss (`scripts/kind_node_loss.py`, `pinned`, `node_grace_ms` 30 s) | `docker stop cvproject-awp-worker4` while it runs an 8-GPU workload | the inventory showed the node not ready 46 s after the stop; R4 requested the stop (`node_lost`) 30 s later; the container was started again, the attempt ended 5 s after that, and the retry ran 91.5 s after the start of the demonstration |
| The fake client with the measured latency (`scripts/fake_measured.py`; 2000 workloads, `balanced` 0.8, virtual clock) | `container_start_ms` 923 (the measured 973.9 ms, truncated, minus 50 ms scheduling) against the assumed 1000 | makespan 266 481 against 266 489 trace s, mean wait 3 361 against 3 363 s; all 2000 `SUCCEEDED`; 8000 events in 19–28 s of wall time |

- **Overhead share.** Running / allocated is the share of the GPU time the platform held for an attempt in which its pod ran. An attempt on kind costs a roughly fixed amount of real time outside the run: about 1 s between the platform's `started` and `running` commits (`pinned`; 1.5 s in `delegate`) and 3.4–3.6 s until the platform records the end (the kubelet and the Job controller report it; the maximum, 4.8 s, is within assumption A1's `lost_grace_ms / 3` = 10 s). In trace time that cost is multiplied by `time_scale`, so the overhead share is 4.5 %, 7.4 %, and 11.4 % at `time_scale` 60, 120, and 240, and the makespan exceeds the local model's by 6 %, 18 %, and 34 %. At `time_scale` 60 the mean wait is within 2 % of the local model's.
- **`delegate`.** The kube-scheduler does not know the policy's placement and spreads the pods by its own scoring: only 20–37 % of the workers ran where the policy had booked them, and the starts took longer. The platform's books then describe a placement that does not exist: the cluster-wide GPU count stays right, but per node it can be wrong, so a workload that fits in the books can stay `Pending` until R2 ends it.
- **Partial gang.** The policy booked `cvproject-awp-worker` and `cvproject-awp-worker2` (free in the platform's books); the kube-scheduler put worker 0 on `cvproject-awp-worker4`, the only node with free GPUs, and left worker 1 pending. A partially started gang is `starting` as a whole, so R2 ended it after `start_timeout_ms` and the stop deleted both Jobs, the running worker included. That is the cost of a gang that is atomic only at the platform level; a gang scheduler (Kueue, Volcano, JobSet) would not start any pod until all fit.
- **Node loss.** Kubernetes marked the node `NotReady` after its node-monitor grace period, and R4 waited `node_grace_ms` from the first snapshot that showed it. The stop could complete only when the node came back: the pod on an unreachable node is not removed until its kubelet answers or the node object is deleted, so the attempt stayed `STOPPING` with its GPUs booked (R9). Had the node not come back, it would have stayed so until an operator deleted the node; that case was not demonstrated.
- **The fake's start latency.** The measured admit → running includes up to one observe interval, so it bounds the real start latency from above; at this trace's run times (30 s to 2 h) the start latency hardly matters. What the fake does not model is the end-detection lag of 3.4–3.6 s, the largest difference between the fake and kind.

## The platform inside the cluster (Tier 2 item 2)

`deploy/k8s/` runs the platform itself as a Deployment in `awp-system` with the in-cluster configuration (`--in-cluster`), a ServiceAccount with a least-privilege Role in `awp-workloads` (Jobs: get, list, create, delete; pods: get, list) and a ClusterRole for nodes (get, list), the kind cluster configuration from a ConfigMap, a non-root user, a read-only root file system, and `/healthz` probes. `scripts/kind_platform.py` applies the manifests, waits for the Deployment, and feeds the platform a 12-workload trace from a Job in the cluster that uses the same image (`replay --time-scale 60 --wait` against the Service). In this repository's run (kind v0.33.0, Kubernetes v1.37.0, one run on a shared desktop machine, on the platform code of commit `5f78c2c`; the script prints its summary and writes no result file) the Deployment became ready, the platform started the workloads as Indexed Jobs in `awp-workloads` under that ServiceAccount, all 12 workloads ended `SUCCEEDED`, `/healthz` answered `ok`, and `/metrics` showed 32 GPUs of capacity and the controller as leader; the script then deleted what it had applied. The store was SQLite on an `emptyDir` (a durable deployment would set `AWP_PG_DSN`).

## Cleanup rules

- Everything the platform creates carries `app.kubernetes.io/managed-by=ai-workload-platform` and `awp.local/instance=<instance id>`; the backend lists, stops, and deletes only objects with both labels, in the workload namespace.
- `forget` deletes the Jobs of an attempt whose terminal state the store has recorded; Jobs and pods of unknown attempts are stopped and forgotten by R3.
- The local kind cluster is deleted by its name: `kind delete cluster --name cvproject-awp` (never by pattern).
