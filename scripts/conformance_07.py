"""Job-by-job conformance with the simulator of gpu-cluster-scheduler (Tier 2 item 3).

The same trace runs (a) through this platform — local backend, virtual clock, zero start latency, the same
cluster file — with the policy server of gpu-cluster-scheduler behind ExternalPolicy (and, for
fifo+first_fit+none, also with the built-in fifo+first_fit), and (b) through that project's simulator.
Start times and placements come from its assignment log; its end times follow from the logged start and
placement by its documented model (docs/simulator.md §3 there: rate = min speed / topology factor, end =
start + ceil(runtime * 1000 / rate - 1e-6) ms, no restart overhead without preemption).

Nothing of gpu-cluster-scheduler is copied or imported: its simulator binary and its repository are named
by environment variables (built outside this repository).

  AWP_GCS_BIN     the simulator binary (go build -o <outside this repo> ./cmd/scheduler)
  AWP_POLICY_CWD  its repository root (the policy server runs as `python -m harness.policy_server` there)

Usage: python scripts/conformance_07.py --seeds 1-3 [--jobs 200] [--out benchmarks/results/conformance]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLUSTER = ROOT / "configs" / "clusters" / "reference.json"
POLICIES = [
    ("fifo+first_fit+none", ["external", "builtin:fifo+first_fit"]),
    ("fifo+first_fit+easy", ["external"]),
]


def theirs(bin_path: str, cwd: str, trace: Path, policy: str, rows: list, cluster) -> dict[str, tuple]:
    from ai_workload_platform.cluster import rate_of

    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "log.csv"
        r = subprocess.run(
            [bin_path, "-trace", str(trace), "-cluster", str(CLUSTER), "-policy", policy, "-log", str(log)],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if r.returncode != 0:
            raise SystemExit(f"simulator failed: {r.stderr[-2000:]}")
        with open(log, newline="", encoding="utf-8") as f:
            entries = [e for e in csv.DictReader(f) if e["op"] == "start"]
    by_id = {r.job_id: r for r in rows}
    nodes = {n.name: n for n in cluster.nodes}
    out = {}
    for e in entries:
        pl = tuple((p.split(":")[0], int(p.split(":")[1])) for p in e["placement"].split(";"))
        job = by_id[e["job_id"]]
        rate = rate_of(
            [nodes[n] for n, _ in pl], job.topology, cluster.cross_node_factor, cluster.cross_rack_factor
        )
        start_ms = round(float(e["t_s"]) * 1000)
        end_ms = start_ms + int(math.ceil(job.runtime_s * 1000 / rate - 1e-6))
        out[e["job_id"]] = (start_ms, end_ms, pl)
    return out


def ours(rows: list, cluster, policy: str, cwd: str, seed: int) -> tuple[dict[str, tuple], int]:
    from ai_workload_platform.bench.trace import to_submissions
    from ai_workload_platform.policy.external import ExternalPolicy
    from ai_workload_platform.sim import simulate

    if policy.startswith("builtin:"):
        pol = policy.split(":", 1)[1]
    else:
        pol = ExternalPolicy(
            [sys.executable, "-m", "harness.policy_server"],
            name=f"external:{policy}",
            wire_name=policy,
            seed=seed,
            cwd=cwd,
            stderr_path=str(Path(tempfile.gettempdir()) / "awp-conformance-policy.log"),
        )
    res = simulate(to_submissions(rows), cluster, policy=pol, backend="local", seed=seed, keep=True)
    failures = sum(res.controller.runner.failures.values())
    start: dict[str, tuple] = {}
    out = {}
    for e in res.events:
        if e.type == "started":
            start[e.workload_id] = (e.at_ms, tuple((p["node"], p["workers"]) for p in e.data["placement"]))
        elif e.type == "attempt_ended" and e.data["reason"] == "succeeded":
            s, pl = start[e.workload_id]
            out[e.workload_id] = (s, e.at_ms, pl)
    res.close()
    return out, failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="1-3")
    ap.add_argument("--jobs", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "conformance"))
    a = ap.parse_args()
    bin_path, cwd = os.environ.get("AWP_GCS_BIN"), os.environ.get("AWP_POLICY_CWD")
    if not bin_path or not cwd:
        print("set AWP_GCS_BIN and AWP_POLICY_CWD (see the docstring)", file=sys.stderr)
        return 2
    from ai_workload_platform.bench.generator import generate
    from ai_workload_platform.bench.trace import write_trace
    from ai_workload_platform.cluster import load_cluster

    cluster = load_cluster(CLUSTER)
    lo, hi = (int(x) for x in a.seeds.split("-"))
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_out = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in range(lo, hi + 1):
            rows, desc = generate(cluster, seed, jobs=a.jobs, variant="balanced", load=0.8)
            trace = Path(tmp) / f"trace-{seed}.csv"
            write_trace(trace, rows, desc, seed)
            for policy, sides in POLICIES:
                ref = theirs(bin_path, cwd, trace, policy, rows, cluster)
                for side in sides:
                    got, failures = ours(rows, cluster, policy if side == "external" else side, cwd, seed)
                    diffs = [j for j in sorted(ref) if got.get(j) != ref[j]]
                    rows_out.append(
                        {
                            "seed": seed,
                            "jobs": len(rows),
                            "policy_07": policy,
                            "platform_policy": f"external:{policy}" if side == "external" else side,
                            "jobs_compared": len(ref),
                            "start_differences": sum(1 for j in ref if got.get(j, (None,))[0] != ref[j][0]),
                            "end_differences": sum(
                                1 for j in ref if got.get(j, (None, None))[1] != ref[j][1]
                            ),
                            "placement_differences": sum(
                                1 for j in ref if got.get(j, (None, None, None))[2] != ref[j][2]
                            ),
                            "policy_failures": failures,
                            "first_difference": diffs[0] if diffs else "",
                        }
                    )
                    print(json.dumps(rows_out[-1]), flush=True)
    with open(out_dir / "conformance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows_out)
    return (
        0
        if all(
            r["start_differences"] == r["end_differences"] == r["placement_differences"] == 0
            for r in rows_out
        )
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
