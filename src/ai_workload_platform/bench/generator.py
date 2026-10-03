"""Seeded trace generator (assumed distributions, shaped like those of gpu-cluster-scheduler and simplified).

Variants: `balanced` (namespace shares 0.4/0.3/0.3), `skew` (70 % of the load from team-a), `bursty`
(a two-state Markov-modulated Poisson process with the balanced shares). The mean arrival rate is
`load * work_capacity / E[gpus * workers * runtime | users]` (offered load as in gpu-cluster-scheduler,
docs/simulator.md §10 there), with the expectation taken given the users drawn for the seed.
All parameters are listed in `DEFAULTS` and in benchmarks/README.md.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from ai_workload_platform.bench.trace import TraceRow
from ai_workload_platform.cluster import ClusterConfig
from ai_workload_platform.rng import categorical, exponential, lognormal, stream

GENERATOR = {"name": "awp-gen", "version": 1}

DEFAULTS: dict[str, Any] = {
    "jobs": 300,
    "load": 0.8,
    "variant": "balanced",
    "namespaces": ["team-a", "team-b", "team-c"],
    "shares": {"balanced": [0.4, 0.3, 0.3], "skew": [0.7, 0.15, 0.15], "bursty": [0.4, 0.3, 0.3]},
    "users_per_namespace": 4,
    # (gpus per worker, workers, probability): mostly small, a few 2-worker gangs
    "sizes": [[1, 1, 0.40], [2, 1, 0.25], [4, 1, 0.17], [8, 1, 0.10], [4, 2, 0.05], [8, 2, 0.03]],
    "gang_rack_sensitive": 0.7,
    "user_runtime_median_s": 600.0,
    "user_runtime_sigma": 0.8,
    "job_runtime_sigma": 0.8,
    "runtime_min_s": 30.0,
    "runtime_max_s": 7200.0,
    "estimate_sigma": 0.5,
    # priority classes: (priority, probability, max_wait_s or None, preemptible probability)
    "priorities": [[1, 0.5, None, 1.0], [4, 0.4, 3600, 0.5], [8, 0.1, 600, 0.0]],
    "cpus_per_gpu": 12,
    "mem_gb_per_gpu": 96,
    "checkpoint_interval_s": 600,
    "bursty": {"on_rate_factor": 2.5, "off_rate_factor": 0.25, "on_mean_s": 1800.0, "off_mean_s": 3600.0},
    "calibration_samples": 20000,
}


def params(**over: Any) -> dict[str, Any]:
    p = copy.deepcopy(DEFAULTS)
    for k, v in over.items():
        if v is not None:
            p[k] = v
    if p["variant"] not in ("balanced", "skew", "bursty"):
        raise ValueError(f"unknown variant {p['variant']!r}")
    return p


def _r3(x: float) -> float:
    return round(x, 3)


def generate(cluster: ClusterConfig, seed: int, **over: Any) -> tuple[list[TraceRow], dict[str, Any]]:
    """Return (rows, generator description for the manifest). Depends only on the parameters and the seed."""
    p = params(**over)
    nss: list[str] = list(p["namespaces"])
    shares: list[float] = list(p["shares"][p["variant"]])
    users = stream(seed, "gen:users")
    typical: dict[str, float] = {}
    for ns in nss:
        for k in range(int(p["users_per_namespace"])):
            typical[f"{ns}-u{k}"] = lognormal(
                users, math.log(p["user_runtime_median_s"]), p["user_runtime_sigma"]
            )
    sizes = p["sizes"]
    size_w = [s[2] for s in sizes]
    mean_gw = sum(s[0] * s[1] * s[2] for s in sizes) / sum(size_w)

    def runtime(r: Any, user: str) -> float:
        x = typical[user] * lognormal(r, 0.0, p["job_runtime_sigma"])
        return min(p["runtime_max_s"], max(p["runtime_min_s"], x))

    # E[runtime | users]: Monte Carlo on a fixed stream, users drawn by namespace share
    cal = stream(seed, "gen:calib")
    total = 0.0
    n_cal = int(p["calibration_samples"])
    for _ in range(n_cal):
        ns = nss[categorical(cal, shares)]
        u = f"{ns}-u{min(int(cal.random() * p['users_per_namespace']), p['users_per_namespace'] - 1)}"
        total += runtime(cal, u)
    mean_rt = total / n_cal
    rate = p["load"] * cluster.work_capacity / (mean_gw * mean_rt)  # arrivals per second

    arr = stream(seed, "gen:arrivals")
    jobs = stream(seed, "gen:jobs")
    b = p["bursty"]
    t = 0.0
    # the bursty process starts in its stationary distribution (on with probability on / (on + off))
    on = (
        arr.random() < b["on_mean_s"] / (b["on_mean_s"] + b["off_mean_s"])
        if p["variant"] == "bursty"
        else True
    )
    state_end = (
        exponential(arr, b["on_mean_s"] if on else b["off_mean_s"]) if p["variant"] == "bursty" else math.inf
    )
    rows: list[TraceRow] = []
    for i in range(int(p["jobs"])):
        if p["variant"] == "bursty":
            # piecewise-constant rate; draw the next arrival across state changes (memoryless)
            while True:
                lam = rate * (b["on_rate_factor"] if on else b["off_rate_factor"])
                dt = exponential(arr, 1.0 / lam)
                if t + dt <= state_end:
                    t += dt
                    break
                t = state_end
                on = not on
                state_end = t + exponential(arr, b["on_mean_s"] if on else b["off_mean_s"])
        else:
            t += exponential(arr, 1.0 / rate)
        ns = nss[categorical(jobs, shares)]
        user = f"{ns}-u{min(int(jobs.random() * p['users_per_namespace']), p['users_per_namespace'] - 1)}"
        g, w, _ = sizes[categorical(jobs, size_w)]
        topo = "rack" if w > 1 and jobs.random() < p["gang_rack_sensitive"] else "any"
        rt = _r3(runtime(jobs, user))
        est = _r3(max(1.0, rt * lognormal(jobs, 0.0, p["estimate_sigma"])))
        pi = categorical(jobs, [c[1] for c in p["priorities"]])
        prio, _pp, max_wait, pre_p = p["priorities"][pi]
        pre = 1 if jobs.random() < pre_p else 0
        rows.append(
            TraceRow(
                job_id=f"j{i:06d}",
                submit_s=_r3(t),
                tenant=ns,
                user=user,
                priority=int(prio),
                gpus=int(g),
                workers=int(w),
                gpu_class="",
                topology=topo,
                cpus=int(g * p["cpus_per_gpu"]),
                mem_gb=float(g * p["mem_gb_per_gpu"]),
                runtime_s=rt,
                estimate_s=est,
                preemptible=pre,
                checkpoint_interval_s=float(p["checkpoint_interval_s"]),
                max_wait_s=None if max_wait is None else float(max_wait),
            )
        )
    # submit times are rounded to ms and stay non-decreasing
    desc = {
        **GENERATOR,
        "params": {
            **p,
            "cluster": cluster.name,
            "mean_rate_per_s": rate,
            "expected_runtime_s": mean_rt,
            "expected_gpus_per_job": mean_gw,
        },
    }
    return rows, desc


def offered_load(rows: list[TraceRow], cluster: ClusterConfig) -> float:
    """Realized offered load: total work over (work capacity x the span of the arrivals)."""
    if len(rows) < 2:
        return 0.0
    span = rows[-1].submit_s - rows[0].submit_s
    work = sum(r.gpus * r.workers * r.runtime_s for r in rows)
    return work / (cluster.work_capacity * span) if span > 0 else 0.0
