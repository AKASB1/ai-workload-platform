"""Tier 2 item 4: node loss on the kind cluster. Stops one kind worker container by its name, watches R4 stop
the attempt on it (reason node_lost) and the retry, then starts the container again.

The service runs as a child process (Kubernetes backend, pinned, time_scale 10); four 8-GPU workloads fill the
four workers; the worker under test is stopped with `docker stop <name>`; the timeline comes from the event log.
Simulated GPUs: the pods sleep. Usage: python scripts/kind_node_loss.py --kubeconfig PATH [--node NAME]
Writes benchmarks/results/cluster/node_loss.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kubeconfig", required=True)
    ap.add_argument("--node", default="cvproject-awp-worker4")
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "cluster" / "node_loss.json"))
    a = ap.parse_args()
    from ai_workload_platform.client import Client
    from ai_workload_platform.procutil import child_env, kill_tree, popen_kwargs

    tmp = Path(tempfile.mkdtemp(prefix="awp-nodeloss-"))
    log = open(tmp / "service.log", "wb")  # noqa: SIM115
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ai_workload_platform",
            "up",
            "--backend",
            "kube",
            "--kubeconfig",
            a.kubeconfig,
            "--cluster",
            str(ROOT / "configs" / "clusters" / "kind.json"),
            "--time-scale",
            "10",
            "--kube-mode",
            "pinned",
            "--db",
            str(tmp / "run.db"),
            "--port",
            "0",
            "--port-file",
            str(tmp / "port"),
            "--log-level",
            "WARNING",
        ],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=child_env(drop=("AWP_PG_DSN",)),
        **popen_kwargs(),
    )
    marks: dict[str, float] = {}
    stopped = False
    try:
        while not (tmp / "port").exists():
            if proc.poll() is not None:
                raise RuntimeError((tmp / "service.log").read_text(errors="replace")[-2000:])
            time.sleep(0.1)
        url = f"http://127.0.0.1:{(tmp / 'port').read_text().strip()}"
        with Client(url, timeout_s=30) as c:
            while not c.nodes():
                time.sleep(0.2)
            t0 = time.monotonic()
            for i in range(4):
                c.submit(
                    ("team-a", "team-b", "team-c", "team-a")[i],
                    {
                        "id": f"nl{i}",
                        "gpus": 8,
                        "estimate_s": 1800,
                        "sim": {"runtime_s": 1800},
                        "retry": {"max_attempts": 3, "backoff_base_s": 5, "jitter": "none"},
                    },
                    f"nl{i}",
                )
            victim = None
            while victim is None:
                for i in range(4):
                    w = c.get(("team-a", "team-b", "team-c", "team-a")[i], f"nl{i}")
                    if w["state"] == "RUNNING" and any(
                        p["node"] == a.node for p in w["attempts"][-1]["placement"]
                    ):
                        victim = w
                assert time.monotonic() - t0 < 300, "workloads did not start"
                time.sleep(0.5)
            ns, wid = victim["namespace"], victim["id"]
            marks["running"] = time.monotonic() - t0
            subprocess.run(["docker", "stop", a.node], check=True, capture_output=True, timeout=120)
            stopped = True
            marks["container_stopped"] = time.monotonic() - t0
            while True:
                n = next(x for x in c.nodes() if x["name"] == a.node)
                if not n["ready"] and "node_not_ready_in_inventory" not in marks:
                    marks["node_not_ready_in_inventory"] = time.monotonic() - t0
                att = c.get(ns, wid)["attempts"][0]
                if att["state"] == "STOPPING":
                    marks["stop_requested_node_lost"] = time.monotonic() - t0
                    break
                assert time.monotonic() - t0 < 600, "R4 did not act"
                time.sleep(1)
            subprocess.run(["docker", "start", a.node], check=True, capture_output=True, timeout=120)
            stopped = False
            marks["container_started"] = time.monotonic() - t0
            while True:
                w = c.get(ns, wid)
                if w["attempts"][0]["state"] == "ENDED" and "attempt_ended" not in marks:
                    marks["attempt_ended"] = time.monotonic() - t0
                if len(w["attempts"]) > 1 and w["attempts"][1]["state"] in ("RUNNING", "ENDED"):
                    marks["retry_running"] = time.monotonic() - t0
                    break
                assert time.monotonic() - t0 < 900, f"no retry: {w['state']}"
                time.sleep(1)
            final = c.get(ns, wid)
            events = [e for e in c.events() if e["workload_id"] == wid]
            for i in range(4):  # end the demonstration: cancel everything
                c.cancel(("team-a", "team-b", "team-c", "team-a")[i], f"nl{i}")
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline and any(
                c.get(("team-a", "team-b", "team-c", "team-a")[i], f"nl{i}")["state"]
                not in ("CANCELLED", "SUCCEEDED", "FAILED", "DEAD_LETTER")
                for i in range(4)
            ):
                time.sleep(1)
    finally:
        if stopped:
            subprocess.run(["docker", "start", a.node], capture_output=True, timeout=120)
        kill_tree(proc.pid)
        proc.wait(timeout=60)
        log.close()
    first = final["attempts"][0]
    out = {
        "note": "simulated GPUs on a local kind cluster; wall-clock seconds on a shared machine; one run",
        "node": a.node,
        "workload": wid,
        "thresholds_ms": {"node_grace_ms": 30000, "lost_grace_ms": 30000},
        "marks_s": {k: round(v, 1) for k, v in marks.items()},
        "first_attempt": {
            "end_reason": first["end_reason"],
            "stop_reason": first["stop_reason"],
            "counted": first["counted"],
            "placement": first["placement"],
        },
        "retry_attempt": {
            "placement": final["attempts"][1]["placement"] if len(final["attempts"]) > 1 else None
        },
        "events": [
            {
                "t_s": round((e["at_ms"] - events[0]["at_ms"]) / 1000, 1),
                "type": e["type"],
                "reason": e["data"].get("reason"),
                "state": e["data"].get("state"),
            }
            for e in events
        ],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
