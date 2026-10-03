"""Tier 2 item 4: the fake client's latencies set from the kind measurements, and a 2000-workload virtual run.

Reads benchmarks/results/cluster/e2_cluster_runs.csv (pinned runs), sets the fake's container start latency to
the median measured `admit_to_running_ms_mean` minus its scheduling latency (real milliseconds; the virtual
run is at real speed, time_scale 1), and runs a 2000-workload trace (balanced, load 0.8, seed 1,
fifo+first_fit) on the Kubernetes backend with the fake client, once with the assumed defaults and once with
the measured latency. Writes benchmarks/results/cluster/fake_measured.json. Simulated.
Usage: python scripts/fake_measured.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    from ai_workload_platform.bench.generator import generate
    from ai_workload_platform.bench.trace import to_submissions
    from ai_workload_platform.cluster import load_cluster
    from ai_workload_platform.observability.evaluation import compute
    from ai_workload_platform.scheduler.kube.fake import DEFAULT_LATENCIES
    from ai_workload_platform.sim import simulate

    src = ROOT / "benchmarks" / "results" / "cluster" / "e2_cluster_runs.csv"
    with open(src, newline="", encoding="utf-8") as f:
        pinned = [r for r in csv.DictReader(f) if r["mode"] == "pinned"]
    measured = statistics.median(float(r["admit_to_running_ms_mean"]) for r in pinned)
    cluster = load_cluster(ROOT / "configs" / "clusters" / "reference-e2.json")
    rows, _ = generate(cluster, 1, jobs=2000, variant="balanced", load=0.8)
    out: dict = {
        "note": "simulated; virtual clock; Kubernetes backend on the fake client; 2000 workloads",
        "measured_admit_to_running_ms_median_kind_pinned": round(measured, 1),
        "runs": [],
    }
    for label, lat in (
        ("assumed defaults", dict(DEFAULT_LATENCIES)),
        (
            "measured on kind",
            {
                **DEFAULT_LATENCIES,
                "container_start_ms": max(0, int(measured) - DEFAULT_LATENCIES["schedule_ms"]),
            },
        ),
    ):
        t0 = time.perf_counter()
        res = simulate(
            to_submissions(rows),
            cluster,
            policy="fifo+first_fit",
            backend="kube-fake",
            seed=1,
            kube_latencies=lat,
        )
        wall = time.perf_counter() - t0
        m = compute(res.events, cluster, res.namespaces)
        out["runs"].append(
            {
                "latencies": label,
                **lat,
                "workloads": len(rows),
                "events": len(res.events),
                "wall_s": round(wall, 2),
                "events_per_wall_s": round(len(res.events) / wall),
                **{
                    k: m[k]
                    for k in (
                        "makespan_s",
                        "wait_mean",
                        "wait_p95",
                        "utilization",
                        "running_over_allocated",
                        "admit_to_running_ms_mean",
                        "terminal_states",
                    )
                },
            }
        )
        print(json.dumps(out["runs"][-1]), flush=True)
    dst = ROOT / "benchmarks" / "results" / "cluster" / "fake_measured.json"
    dst.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
