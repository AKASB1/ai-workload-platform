# Benchmarks

Everything here is **simulated**: workloads sleep for their run time (on the virtual clock, or as `sleep` in pods on a local kind cluster with advertised, not real, GPUs). The numbers describe this control plane under the stated assumptions — in this simulation, under these assumptions — and nothing about production clusters or real GPUs.

## Reproduce

```bash
python -m ai_workload_platform bench --quick
python -m ai_workload_platform bench --full
python -m ai_workload_platform bench --cluster --kubeconfig outputs/kind/kubeconfig
python -m ai_workload_platform bench --cluster --sweep --kubeconfig outputs/kind/kubeconfig
python scripts/fake_measured.py
python scripts/kind_partial_gang.py --kubeconfig outputs/kind/kubeconfig
python scripts/kind_node_loss.py --kubeconfig outputs/kind/kubeconfig
python scripts/plot_results.py --suite full
python scripts/plot_timeline.py --suite full
python scripts/plot_framework.py
```

`--quick` needs no Docker; `--cluster` and the `kind_*` scripts need the kind cluster of `deploy/README.md`; `fake_measured.py` reads the kind results. The conformance check with `gpu-cluster-scheduler` (`scripts/conformance_07.py`) needs that project's simulator and policy server (`docs/contracts.md` §11). Each suite writes small CSV/JSON files to `benchmarks/results/<suite>/` with a `manifest.json` (commit, configuration hash, seeds, Python and library versions, hardware class, worker count, the load note, and for the cluster the Docker, kind, and Kubernetes versions). Event logs are not kept (they go to the ignored `outputs/`), except the small run behind the timeline figure (`results/full/timeline_events.jsonl`). The demonstration scripts and the conformance script write one result file each and no manifest. Runs in parallel use `spawn` worker processes; results are sorted by (scenario, policy, seed) before they are written, so the files do not depend on the worker count.

## Protocol (fixed before the first evaluation run)

- **Reference system** (`configs/clusters/reference.json`, assumed): 4 nodes × 8 GPUs of class `a100` (speed 1.0) in racks `r0` and `r1` (2 nodes each), 128 CPUs and 1024 GB per node, `cross_node_factor` 1.1 and `cross_rack_factor` 1.25 (both 1.0 in E2, `reference-e2.json`, because a real cluster has no topology penalty to match), `restart_overhead_s` 120 (used only by the preempting variants of Tier 2). Namespaces `team-a`, `team-b`, `team-c` with shares 0.4, 0.3, 0.3 → `quota_gpus` 12, 9, 9; `cap_gpus` 32; `max_queued` 200.
- **Traces** come from the generator (`bench/generator.py`, CLI `gen`), whose distributions are assumptions shaped like those of `gpu-cluster-scheduler` and simplified:

| Parameter | Value |
|---|---|
| arrivals | Poisson at the offered load: mean rate = `load` × work capacity (32 reference GPU-s per s) / E[GPUs × workers × run time given the users drawn for the seed] (Monte Carlo, 20 000 draws on a fixed stream) |
| variants | `balanced` (namespace shares 0.4/0.3/0.3), `skew` (0.7/0.15/0.15), `bursty` (shares as balanced; a two-state Markov-modulated Poisson process: rate × 2.5 in the on state, × 0.25 off, mean sojourn 1800 s on / 3600 s off, starting in its stationary distribution) |
| users | 4 per namespace; each user's typical run time lognormal(ln 600 s, 0.8) drawn once per seed |
| run time | the user's typical value × lognormal(0, 0.8), truncated to [30 s, 7200 s] |
| sizes (GPUs × workers, probability) | 1×1 0.40, 2×1 0.25, 4×1 0.17, 8×1 0.10, 4×2 0.05, 8×2 0.03 (gangs rack-sensitive with probability 0.7) |
| priority classes | 1 (50 %, preemptible, no wait target), 4 (40 %, half preemptible, `max_wait_s` 3600), 8 (10 %, not preemptible, `max_wait_s` 600) |
| estimates | run time × lognormal(0, 0.5) |
| CPU, memory | 12 CPUs and 96 GB per GPU |

  Realized offered load over 20 seeds of 300 workloads at a target of 0.8 (total work over capacity × total arrival span): balanced 0.82, skew 0.83, bursty 0.87 (`tests/test_inputs.py` checks ±10 %, the namespace shares ±0.03, and the size mix ±0.03).
- **E1 Failure injection (the main result):** the harness of `docs/failure-injection.md`, schedule seeds 1–2000 per backend (local and the Kubernetes fake) in `--full`, 1–200 in the default tests, 1–100 in `--quick`; each injected bug on seeds 1–200 per backend in `--full`.
- **E2 Backends:** trace `balanced`, load 0.8, 40 workloads, `fifo+first_fit`; (a) the local backend and (b) the Kubernetes backend on the fake client, trace seeds 1–10 on the virtual clock; (c) a kind cluster (4 workers with 8 advertised GPUs each) in `pinned` and `delegate` modes, trace seed 1, 3 repetitions each, `time_scale` = longest run time / 60 s so that the longest workload sleeps about one minute (`replay` divides the submit times and the backend the run times by it). The comparison says where the backends differ and why; it does not rank them.
- **E3 Policies on the platform:** `fifo+first_fit`, `priority+best_fit`, `quota+first_fit`, and the stub external policy (which implements `fifo+first_fit` over the protocol and must reproduce it) on `balanced`, `skew`, and `bursty` traces at offered loads 0.8 and 1.0, 300 workloads, trace seeds 101–110, local backend, virtual clock, defaults only (no tuning, so there is no tuning set). It shows that the platform runs the policies and that their differences have the shape the policies imply; it is not a policy benchmark (that is `gpu-cluster-scheduler`).
- **Statistics:** mean and 95 % Student-t interval across seeds; paired differences against `fifo+first_fit` on the common seeds with their intervals; win/tie/loss with a tie band of 5 % of the baseline value.
- **Wall-clock numbers** (the kind runs) are measured on a shared machine: the manifest states the load (other agent runs active, the worker count, a CPU-load sample), each configuration is repeated 3 times, and the median, minimum, and maximum are reported. They are kept out of every deterministic comparison. The virtual-clock results are deterministic (byte-identical event logs for a seed).

## Results

All results come from commit `3b6e03b`: the evaluation suites, the kind demonstrations, `fake_measured.py`, and the conformance check (`scripts/kind_partial_gang.py` was added afterwards and changes no platform code). [`results/full/table.md`](results/full/table.md) has every table with every policy (written by `scripts/plot_results.py`). The manifests record the load: the virtual evaluation (`results/full/manifest.json`) ran with 4 worker processes while one other agent run was active, CPU load 32 % sampled before it, 647 s of wall time; the kind runs (`results/cluster/manifest.json`, `manifest-sweep.json`) ran one service at a time with the same other run active; the CPU load was sampled once, before E2 (c) (60 %), and the sweep and the demonstrations have no sample of their own. The manifests say `dirty: true` because the flag comes from `git status` after a suite has written its own result files, which are tracked, so it is set whenever they differ from the committed ones (here, and in the quick results, also uncommitted figures and documentation); no source file differed from the recorded commit. In this run a 12-worker evaluation of the same commit (repeated with 4 workers when the other run was noticed) wrote the same result files as the 4-worker one in every column except the wall-clock times; only the 4-worker files are committed.

### E1 Failure injection

| Backend | Schedules | Violating | Converged | Faults injected | Crash points hit | Convergence after the last fault P50 / P95 / max (virtual s) |
|---|---|---|---|---|---|---|
| local | 2000 | **0** | 2000 | 6 539 (16 kinds) | 735 (7 of 7) | 124.3 / 261.3 / 494.1 |
| Kubernetes fake | 2000 | **0** | 2000 | 6 480 (14 kinds) | 797 (7 of 7) | 115.7 / 211.8 / 398.3 |

Faults injected counts the faults that took effect: a crash, a controller pause, and a racing cancel count only when their point was reached (local: 735 of 986 scheduled crashes, 564 of 622 pauses), the others when they were scheduled; 24 local and 27 fake schedules had no fault that took effect. Convergence is the virtual time from the later of the end of the last fault and the end of the 300 s submission window until every workload is terminal and the backend holds nothing, so it includes the remaining run time of the workloads (815 local and 786 fake schedules count from the end of the submission window).

| Injected bug | Local: schedules caught of 200 (first seed) | Fake: caught of 200 (first seed) | First invariant violated |
|---|---|---|---|
| 1 `release_at_stop` | 176 (1) | 178 (1) | I5; I4 once per backend |
| 2 `no_epoch_check` | 3 (3) | 1 (126) | I5 (lease-epoch audit) |
| 3 `no_lost_grace` | 12 (32) | 4 (30) | I3 |
| 4 `no_version_check` | 29 (6) | 27 (6) | I5 |

Bug 2 is the hardest to reach: compare-and-set alone rejects most writes of a paused controller, so only a schedule where the paused controller writes rows the new one never touched shows it (`docs/failure-injection.md`). The first full evaluation (commit `012db44`) had 3 of 4000 schedules that did not converge (I7); the cause and the fix are item 2 of "Bugs found in the platform during development" there, and these results are from the fixed commit.

![E1](../docs/figures/e1_convergence.png)

### E2 Backends

| Backend | Runs | Makespan (trace s) | Mean wait (trace s) | Admit → running (ms) | Observe lag mean / max (ms) | Placement match | Running / allocated |
|---|---|---|---|---|---|---|---|
| local | seeds 1–10 | 12 622 ± 2 397 | 763 ± 710 | 993 ± 8 | 0 / 0 | 1.00 | 0.999 |
| Kubernetes fake | seeds 1–10 | 12 624 ± 2 397 | 764 ± 710 | 1 050 | 0 / 0 | 1.00 | 0.999 |
| local (seed 1) | 1 | 10 526 | 523 | 971 | 0 / 0 | 1.00 | 0.999 |
| kind `pinned` (seed 1, `time_scale` 120) | 3 | 12 395 [12 278, 12 455] | 1 043 [1 009, 1 058] | 974 [963, 1 004] | 3 533 / 4 572 | 1.00 | 0.926 [0.923, 0.929] |
| kind `delegate` (seed 1, `time_scale` 120) | 3 | 12 375 [12 117, 12 500] | 1 090 [1 086, 1 173] | 1 510 [1 149, 1 586] | 3 396 / 4 394 | 0.22 [0.20, 0.37] | 0.799 [0.796, 0.881] |

Local and fake: mean ± 95 % Student-t half-width over seeds 1–10; kind: median [min, max] of 3 runs (the observe-lag columns: medians of the run means and of the run maxima); all 240 workloads on kind `SUCCEEDED` at their first attempt. Admit → running runs from the platform's `started` commit to its `running` commit; on kind it includes up to one observe interval (1 s). On the virtual clock the local backend starts at once, and its 993 ms is the driver's polling at the observe interval while a start is in flight; the fake reports the start at its modelled latency (50 ms + 1 s). The utilization column of `table.md` counts allocated GPU time over the submission window; on kind an allocation also holds the GPUs while a pod starts and until its end is recorded (seconds of real time, minutes of trace time), so utilization is higher there (0.86 against 0.73 on the local run of seed 1) without more work being done; read it together with running / allocated.

What differs and why: on kind every attempt costs about 1 s to start and 3.4–3.6 s until its end is recorded, real time that `time_scale` turns into minutes of trace time. The time-scale sweep (`bench --cluster --sweep`, `pinned`, 3 runs each) separates that from the backend: with the longest workload at 120, 60, and 30 s of real time (`time_scale` 60, 120, 240) the makespan is 11 157, 12 395, and 14 108 trace s (local: 10 526), the mean wait 517, 1 043, and 1 894 trace s (local: 523), and the share of allocated GPU time without a running pod 4.5 %, 7.4 %, and 11.4 % (medians). In `delegate` mode the kube-scheduler ignores the policy's placement for most workers.

![E2](../docs/figures/e2_backends.png)

### E3 Policies

P95 bounded slowdown, mean ± 95 % CI over trace seeds 101–110 (300 workloads each), and paired W/T/L against `fifo+first_fit`:

| Variant, load | fifo+first_fit | priority+best_fit | quota+first_fit | stub (external fifo) |
|---|---|---|---|---|
| balanced 0.8 | 32.4 ± 27.4 | 8.9 ± 6.9 (10/0/0) | 6.5 ± 3.8 (10/0/0) | 32.4 ± 27.4 (0/10/0) |
| balanced 1.0 | 78.7 ± 67.3 | 19.1 ± 10.6 (10/0/0) | 17.0 ± 8.2 (10/0/0) | 78.7 ± 67.3 (0/10/0) |
| skew 0.8 | 30.2 ± 17.6 | 9.8 ± 7.5 (10/0/0) | 11.2 ± 11.4 (10/0/0) | 30.2 ± 17.6 (0/10/0) |
| skew 1.0 | 62.8 ± 29.4 | 17.3 ± 11.3 (10/0/0) | 17.2 ± 12.3 (10/0/0) | 62.8 ± 29.4 (0/10/0) |
| bursty 0.8 | 67.0 ± 65.9 | 22.2 ± 11.9 (9/1/0) | 20.9 ± 10.2 (10/0/0) | 67.0 ± 65.9 (0/10/0) |
| bursty 1.0 | 75.5 ± 60.2 | 30.5 ± 19.8 (10/0/0) | 23.8 ± 11.6 (10/0/0) | 75.5 ± 60.2 (0/10/0) |

SLO attainment (among the succeeded workloads in the measurement window, which leaves out the first and the last 10 % of the terminal workloads by submission, the share of those with a wait target that met it) is 0.37–0.73 for `fifo+first_fit` and 0.88–0.97 for the two skipping policies; P95 wait, Jain's index, utilization, and quota satisfaction are in `table.md`. The stub reproduces `fifo+first_fit` exactly on every seed. The intervals are wide because run times are heavy-tailed.

![E3](../docs/figures/e3_policies.png)

### Tier 2

- **Preemption with checkpoints** (`results/full/e3_preemption.csv`; the same traces, paired with the base policy): `priority+best_fit+preempt` preempts 68–132 times per run on average, loses 7.9–13.5 GPU-hours of work and spends 6.6–11.4 GPU-hours in restart overhead, raises SLO attainment in all six settings (for example 0.928 → 0.990 at `balanced` 1.0), and worsens the P95 bounded slowdown on 6–8 of 10 seeds. `quota+first_fit+reclaim` preempts 23–82 times, raises SLO attainment by 0.01–0.02, and splits wins and losses on the slowdown. Figure: `docs/figures/e3_preemption.png`.
- **Conformance with `gpu-cluster-scheduler`** (`results/conformance/conformance.csv`): its policy server through the protocol (`fifo+first_fit+none` and EASY) and the built-in `fifo+first_fit` give the same start times, end times, and placements as its simulator for 3 × 200 workloads each: 0 differences (`docs/contracts.md` §11).
- **Kubernetes depth** (`results/cluster/`): the sweep above, the partial-gang and node-loss demonstrations, and the fake client with the measured start latency, in one table in [`docs/kubernetes.md`](../docs/kubernetes.md#kubernetes-depth-tier-2-item-4).

## TODO

- Repeat the kind runs more often and on a quiet machine; 3 runs on a shared desktop show where the backends differ, not how fast they are.
- Model the end-detection lag in the fake client (it is the largest difference to kind) and run E2 again.
- Run the fault harness against PostgreSQL and against the kind cluster (today: in-memory SQLite and the fake client).
- Run E3 with the preempting and MILP policies of `gpu-cluster-scheduler` through the protocol.
