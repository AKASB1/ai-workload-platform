"""The evaluation: E1 (failure injection), E2 (backends), E3 (policies on the platform).

python -m ai_workload_platform bench --quick | --full | --cluster
Results (small CSV/JSON files) go to benchmarks/results/<suite>/; event logs are not kept.
Everything is simulated; wall-clock numbers carry a load note.
"""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from ai_workload_platform import __version__

ROOT = Path(__file__).resolve().parents[3]
CLUSTERS = ROOT / "configs" / "clusters"

SEEDS = {
    "e1": {"quick": (1, 100), "full": (1, 2000)},
    "e2": {"quick": (1, 3), "full": (1, 10)},
    "e3": {"quick": (101, 103), "full": (101, 110)},
}
E1_BUG_SEEDS = (1, 200)
E2 = {
    "variant": "balanced",
    "load": 0.8,
    "jobs": 40,
    "policy": "fifo+first_fit",
    "cluster": "reference-e2.json",
}
E3 = {
    "jobs": 300,
    "cluster": "reference.json",
    "policies": ["fifo+first_fit", "priority+best_fit", "quota+first_fit", "stub"],
    "variants": {"quick": ["balanced"], "full": ["balanced", "skew", "bursty"]},
    "loads": {"quick": [0.8], "full": [0.8, 1.0]},
}
METRICS = [
    "makespan_s",
    "wait_mean",
    "wait_p95",
    "jct_mean",
    "bsld_p95",
    "jain_bsld",
    "utilization",
    "slo_attainment",
    "quota_satisfaction",
    "borrowed_gpu_hours",
    "jain_weighted_bsld",
    "running_over_allocated",
    "admit_to_running_ms_mean",
    "admit_to_running_ms_p95",
    "observe_lag_ms_mean",
    "observe_lag_ms_p95",
    "observe_lag_ms_max",
    "placement_match",
    "attempts",
    "retries",
    "dead_letters",
    "preemptions",
    "lost_gpu_seconds",
    "overhead_gpu_seconds",
]
# Tier 2 item 1: the preempting variants on the same traces, compared with their base policies
E3P = {"priority+best_fit+preempt": "priority+best_fit", "quota+first_fit+reclaim": "quota+first_fit"}


# --- unit runs (picklable, run in spawn workers) -------------------------------------------------


def run_trace(job: dict[str, Any]) -> dict[str, Any]:
    from ai_workload_platform.bench.generator import generate
    from ai_workload_platform.bench.trace import to_submissions
    from ai_workload_platform.cluster import load_cluster
    from ai_workload_platform.observability.evaluation import compute
    from ai_workload_platform.sim import simulate

    cluster = load_cluster(CLUSTERS / job["cluster"])
    rows, _ = generate(cluster, job["seed"], jobs=job["jobs"], variant=job["variant"], load=job["load"])
    t0 = time.perf_counter()
    res = simulate(
        to_submissions(rows),
        cluster,
        policy=job["policy"],
        backend=job["backend"],
        seed=job["seed"],
        keep=True,
    )
    wall = time.perf_counter() - t0
    m = compute(res.events, cluster, res.namespaces)
    counters = dict(res.controller.counters)
    runner = res.controller.runner
    res.close()
    out = {k: job[k] for k in ("experiment", "backend", "variant", "load", "policy", "seed")}
    out.update({k: m.get(k) for k in METRICS})
    for ns in ("team-a", "team-b", "team-c"):
        out[f"wait_p95_{ns}"] = m.get(f"ns_{ns}_wait_p95")
    out["attempts_by_reason"] = json.dumps(m["attempts_by_reason"], sort_keys=True)
    out["terminal_states"] = json.dumps(m["terminal_states"], sort_keys=True)
    out["stale_actions"] = counters.get("stale_action", 0)
    out["policy_failures"] = sum(runner.failures.values())
    out["events"] = len(res.events)
    out["wall_s"] = round(wall, 4)
    return out


def run_e1(job: tuple[int, str, str | None]) -> dict[str, Any]:
    from ai_workload_platform.faults.harness import run_schedule

    o = run_schedule(*job)
    return {
        "backend": o.backend,
        "bug": o.bug,
        "seed": o.seed,
        "violations": len(o.violations),
        "invariant": o.violations[0]["invariant"] if o.violations else "",
        "message": o.violations[0]["message"][:160] if o.violations else "",
        "faults": json.dumps(o.faults_injected, sort_keys=True),
        "crash_points": json.dumps(o.crash_points_hit, sort_keys=True),
        "converge_ms": o.converge_ms,
        "events": o.events,
        "ticks": o.ticks,
        "workloads": o.workloads,
        "wall_s": round(o.wall_s, 4),
    }


def _pool_map(fn: Any, jobs: list[Any], workers: int) -> list[Any]:
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(fn, jobs, chunksize=max(1, len(jobs) // (workers * 6))))


# --- experiments ---------------------------------------------------------------------------------


def e1(suite: str, workers: int) -> tuple[list[dict], dict[str, Any]]:
    lo, hi = SEEDS["e1"][suite]
    jobs: list[tuple[int, str, str | None]] = [
        (s, b, None) for b in ("local", "kube-fake") for s in range(lo, hi + 1)
    ]
    if suite == "full":
        from ai_workload_platform.faults.bugs import BUGS

        blo, bhi = E1_BUG_SEEDS
        jobs += [
            (s, b, bug) for b in ("local", "kube-fake") for bug in sorted(BUGS) for s in range(blo, bhi + 1)
        ]
    rows = sorted(_pool_map(run_e1, jobs, workers), key=lambda r: (r["backend"], r["bug"], r["seed"]))
    summary: dict[str, Any] = {}
    for b in ("local", "kube-fake"):
        rs = [r for r in rows if r["backend"] == b and r["bug"] == "none"]
        conv = sorted(r["converge_ms"] for r in rs if r["converge_ms"] is not None)
        faults: dict[str, int] = {}
        points: dict[str, int] = {}
        for r in rs:
            for k, v in json.loads(r["faults"]).items():
                faults[k] = faults.get(k, 0) + v
            for k, v in json.loads(r["crash_points"]).items():
                points[k] = points.get(k, 0) + v

        def nr(p: float, conv: list = conv) -> float | None:
            import math

            return conv[max(0, math.ceil(p * len(conv)) - 1)] / 1000 if conv else None

        summary[b] = {
            "schedules": len(rs),
            "violating_schedules": sum(1 for r in rs if r["violations"]),
            "faults_injected": dict(sorted(faults.items())),
            "crash_points_hit": dict(sorted(points.items())),
            "converge_s_p50": nr(0.5),
            "converge_s_p95": nr(0.95),
            "converge_s_max": conv[-1] / 1000 if conv else None,
            "converged": len(conv),
            "events": sum(r["events"] for r in rs),
            "wall_s_sum": round(sum(r["wall_s"] for r in rs), 2),
        }
        bugs = {}
        for bug in sorted({r["bug"] for r in rows if r["bug"] != "none"}):
            br = sorted((r for r in rows if r["backend"] == b and r["bug"] == bug), key=lambda r: r["seed"])
            caught = [r["seed"] for r in br if r["violations"]]
            bugs[bug] = {
                "schedules": len(br),
                "caught": len(caught),
                "schedules_until_caught": caught[0] - br[0]["seed"] + 1 if caught else None,
                "first_seed": caught[0] if caught else None,
                "invariants": dict(
                    sorted(
                        {
                            inv: sum(1 for r in br if r["invariant"] == inv)
                            for inv in {r["invariant"] for r in br if r["invariant"]}
                        }.items()
                    )
                ),
            }
        if bugs:
            summary[b]["bugs"] = bugs
    return rows, summary


def e2(suite: str, workers: int) -> list[dict]:
    lo, hi = SEEDS["e2"][suite]
    jobs = [
        {
            "experiment": "E2",
            "backend": b,
            "variant": E2["variant"],
            "load": E2["load"],
            "policy": E2["policy"],
            "seed": s,
            "jobs": E2["jobs"],
            "cluster": E2["cluster"],
        }
        for b in ("local", "kube-fake")
        for s in range(lo, hi + 1)
    ]
    return sorted(_pool_map(run_trace, jobs, workers), key=lambda r: (r["backend"], r["seed"]))


def e3(suite: str, workers: int) -> list[dict]:
    lo, hi = SEEDS["e3"][suite]
    jobs = [
        {
            "experiment": "E3",
            "backend": "local",
            "variant": v,
            "load": ld,
            "policy": p,
            "seed": s,
            "jobs": E3["jobs"],
            "cluster": E3["cluster"],
        }
        for v in E3["variants"][suite]
        for ld in E3["loads"][suite]
        for p in [*E3["policies"], *E3P]
        for s in range(lo, hi + 1)
    ]
    order = [*E3["policies"], *E3P]
    return sorted(
        _pool_map(run_trace, jobs, workers),
        key=lambda r: (r["variant"], r["load"], order.index(r["policy"]), r["seed"]),
    )


def aggregate_e3p(rows: list[dict]) -> list[dict]:
    """Tier 2: each preempting variant against its base policy, paired on the same traces."""
    from ai_workload_platform.bench.stats import DIRECTION, mean_ci, paired

    out = []
    metrics = [
        "bsld_p95",
        "wait_p95",
        "slo_attainment",
        "utilization",
        "jain_bsld",
        "quota_satisfaction",
        "preemptions",
        "lost_gpu_seconds",
        "overhead_gpu_seconds",
    ]
    for v, ld in sorted({(r["variant"], r["load"]) for r in rows}):
        for p, base in E3P.items():
            rs = {r["seed"]: r for r in rows if (r["variant"], r["load"], r["policy"]) == (v, ld, p)}
            bs = {r["seed"]: r for r in rows if (r["variant"], r["load"], r["policy"]) == (v, ld, base)}
            if not rs:
                continue
            for m in metrics:
                mean, ci, n = mean_ci([r[m] for r in rs.values()])
                bmean, bci, _ = mean_ci([r[m] for r in bs.values()])
                pr = paired(
                    {s: r[m] for s, r in bs.items()}, {s: r[m] for s, r in rs.items()}, DIRECTION.get(m, 0)
                )
                out.append(
                    {
                        "variant": v,
                        "load": ld,
                        "policy": p,
                        "base": base,
                        "metric": m,
                        "mean": mean,
                        "ci95": ci,
                        "base_mean": bmean,
                        "base_ci95": bci,
                        "n": n,
                        **pr,
                    }
                )
    return out


def aggregate_e3(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    from ai_workload_platform.bench.stats import DIRECTION, mean_ci, paired

    agg, pairs = [], []
    metrics = [*DIRECTION, "wait_p95_team-a", "wait_p95_team-b", "wait_p95_team-c"]
    keys = sorted({(r["variant"], r["load"]) for r in rows})
    for v, ld in keys:
        base = {
            r["seed"]: r for r in rows if (r["variant"], r["load"], r["policy"]) == (v, ld, "fifo+first_fit")
        }
        for p in E3["policies"]:
            rs = {r["seed"]: r for r in rows if (r["variant"], r["load"], r["policy"]) == (v, ld, p)}
            for m in metrics:
                mean, ci, n = mean_ci([r[m] for r in rs.values()])
                agg.append(
                    {"variant": v, "load": ld, "policy": p, "metric": m, "mean": mean, "ci95": ci, "n": n}
                )
                if p != "fifo+first_fit":
                    pr = paired(
                        {s: r[m] for s, r in base.items()},
                        {s: r[m] for s, r in rs.items()},
                        DIRECTION.get(m, -1),
                    )
                    pairs.append({"variant": v, "load": ld, "policy": p, "metric": m, **pr})
    return agg, pairs


# --- the timeline scenario (framework of the timeline figure) --------------------------------------


def timeline(out: Path) -> None:
    """A few workloads with a controller crash and a node loss, on the virtual clock (local backend)."""
    from ai_workload_platform.faults.harness import Harness
    from ai_workload_platform.faults.schedule import HARNESS_THRESHOLDS, Action, Fault, Schedule

    spec = {
        "gpus": 4,
        "cpus": 48,
        "mem_gb": 384,
        "estimate_s": 90,
        "retry": {"max_attempts": 3, "backoff_base_s": 5, "jitter": "none"},
    }
    runtimes = [80, 120, 60, 100, 90, 70, 110, 50]
    actions = [
        Action(
            i * 8000,
            "submit",
            ("team-a", "team-b", "team-c")[i % 3],
            f"t{i}",
            {**spec, "gpus": (4, 8, 4, 8, 2, 4, 8, 4)[i], "sim": {"runtime_s": rt}},
        )
        for i, rt in enumerate(runtimes)
    ]
    faults = [
        Fault(40_000, "crash", {"point": "start.after_commit"}),
        Fault(70_000, "node_down", {"node": "r0-n01"}, 110_000),
    ]
    sched = Schedule(
        0, "local", HARNESS_THRESHOLDS, "fifo+first_fit", 0, 0, actions, faults, 110_000, 600_000
    )
    h = Harness(sched)
    o = h.run()
    (out / "timeline_events.jsonl").write_bytes(h.log_bytes)
    meta = {
        "faults": [f.to_json() for f in faults],
        "outcome": {k: v for k, v in o.to_json().items() if k != "log_tail"},
        "thresholds": HARNESS_THRESHOLDS.to_dict(),
        "note": "simulated, virtual clock, local backend",
    }
    (out / "timeline_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


# --- manifest ------------------------------------------------------------------------------------


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10, check=False
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def hardware() -> dict[str, Any]:
    cpu = platform.processor()
    try:
        if sys.platform == "win32":
            cpu = (
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                ).stdout.strip()
                or cpu
            )
        elif Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except Exception:  # noqa: BLE001
        pass
    ram_gb = None
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            ram_gb = round(ms.ullTotalPhys / 2**30)
        elif Path("/proc/meminfo").exists():
            ram_gb = round(int(Path("/proc/meminfo").read_text().split()[1]) / 2**20)
    except Exception:  # noqa: BLE001
        pass
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "ram_gb": ram_gb, "os": platform.platform(terse=True)}


def versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {"python": platform.python_version(), "ai-workload-platform": __version__}
    for p in (
        "fastapi",
        "pydantic",
        "uvicorn",
        "prometheus-client",
        "kubernetes",
        "psycopg",
        "numpy",
        "scipy",
        "matplotlib",
        "httpx",
    ):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return out


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True).encode() + b"".join(
        (CLUSTERS / f).read_bytes() for f in sorted(os.listdir(CLUSTERS))
    )
    from ai_workload_platform.bench.generator import DEFAULTS
    from ai_workload_platform.faults.schedule import HARNESS_THRESHOLDS

    blob += json.dumps(DEFAULTS, sort_keys=True).encode() + json.dumps(HARNESS_THRESHOLDS.to_dict()).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def manifest(
    suite: str, workers: int, load_note: str, started: float, extra: dict | None = None
) -> dict[str, Any]:
    cfg = {"suite": suite, "seeds": SEEDS, "e1_bug_seeds": E1_BUG_SEEDS, "e2": E2, "e3": E3}
    return {
        "suite": suite,
        "commit": _git("rev-parse", "--short", "HEAD") or "unknown",
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "config_hash": config_hash(cfg),
        "config": cfg,
        "workers": workers,
        "load_note": load_note,
        "hardware": hardware(),
        "versions": versions(),
        "wall_s": round(time.perf_counter() - started, 1),
        "simulated": True,
        **(extra or {}),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    k: ("" if r.get(k) is None else (round(r[k], 6) if isinstance(r[k], float) else r[k]))
                    for k in cols
                }
            )


def main(a: Any) -> int:
    from ai_workload_platform.bench.cluster import run_cluster

    load_note = os.environ.get("AWP_LOAD_NOTE", "not recorded")
    workers = a.workers or 4
    t0 = time.perf_counter()
    if a.cluster_run:
        out = Path(a.out) / "cluster"
        out.mkdir(parents=True, exist_ok=True)
        sweep = getattr(a, "sweep", False)
        configs = (("pinned", 30.0), ("pinned", 120.0)) if sweep else (("pinned", 60.0), ("delegate", 60.0))
        rows, extra = run_cluster(a.kubeconfig or os.environ.get("AWP_KUBECONFIG"), out, configs=configs)
        write_csv(out / ("sweep_runs.csv" if sweep else "e2_cluster_runs.csv"), rows)
        (out / ("manifest-sweep.json" if sweep else "manifest.json")).write_text(
            json.dumps(manifest("cluster", 1, load_note, t0, extra), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(json.dumps({"runs": len(rows), "out": str(out)}, indent=2))
        return 0
    suite = "quick" if a.quick else "full"
    out = Path(a.out) / suite
    out.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}
    t = time.perf_counter()
    e1_rows, e1_summary = e1(suite, workers)
    timing["e1"] = round(time.perf_counter() - t, 1)
    write_csv(out / "e1_runs.csv", e1_rows)
    (out / "e1_summary.json").write_text(
        json.dumps(e1_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    t = time.perf_counter()
    e2_rows = e2(suite, workers)
    timing["e2"] = round(time.perf_counter() - t, 1)
    write_csv(out / "e2_runs.csv", e2_rows)
    t = time.perf_counter()
    e3_rows = e3(suite, workers)
    timing["e3"] = round(time.perf_counter() - t, 1)
    write_csv(out / "e3_runs.csv", e3_rows)
    agg, pairs = aggregate_e3([r for r in e3_rows if r["policy"] in E3["policies"]])
    write_csv(out / "e3_aggregate.csv", agg)
    write_csv(out / "e3_paired.csv", pairs)
    write_csv(out / "e3_preemption.csv", aggregate_e3p(e3_rows))
    timeline(out)
    man = manifest(suite, workers, load_note, t0, {"wall_s_by_experiment": timing})
    (out / "manifest.json").write_text(
        json.dumps(man, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    viol = {b: s["violating_schedules"] for b, s in e1_summary.items()}
    print(
        json.dumps(
            {
                "suite": suite,
                "out": str(out),
                "e1_violating_schedules": viol,
                "wall_s": man["wall_s"],
                "timing_s": timing,
            },
            indent=2,
        )
    )
    return 0 if not any(viol.values()) else 1
