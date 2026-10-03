"""Demo: start the service on the local backend with a scaled clock, feed it a small trace with
`replay`, print a few API answers and /metrics, and stop the service by PID.

Usage: python scripts/demo.py [--scale 120] [--jobs 10] [--keep DIR]
Everything is simulated: workloads sleep on a sped-up clock; no GPU is used.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ai_workload_platform.client import Client
from ai_workload_platform.procutil import child_env, kill_tree, pid_alive, popen_kwargs

ROOT = Path(__file__).resolve().parents[1]


def cli(*args: str) -> list[str]:
    return [sys.executable, "-m", "ai_workload_platform", *args]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scale", type=float, default=120.0, help="platform seconds per wall second")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--keep", default=None, help="keep the database, trace, and logs in this folder")
    a = ap.parse_args()
    work = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="awp-demo-"))
    work.mkdir(parents=True, exist_ok=True)
    trace, port_file, log_path = work / "trace.csv", work / "port", work / "service.log"
    subprocess.run(
        cli("gen", "--seed", "1", "--jobs", str(a.jobs), "--runtime-max-s", "900", "--out", str(trace)),
        check=True,
        cwd=ROOT,
        env=child_env(drop=("AWP_PG_DSN",)),
        timeout=120,
    )
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            cli(
                "up",
                "--db",
                str(work / "awp.db"),
                "--port",
                "0",
                "--port-file",
                str(port_file),
                "--scale",
                str(a.scale),
                "--log-level",
                "WARNING",
            ),
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=child_env(drop=("AWP_PG_DSN",)),
            **popen_kwargs(),
        )
    print(f"service started, PID {proc.pid}")
    try:
        deadline = time.monotonic() + 60
        while not port_file.exists():
            if time.monotonic() > deadline or proc.poll() is not None:
                print(log_path.read_text(encoding="utf-8", errors="replace"), file=sys.stderr)
                raise SystemExit("the service did not start")
            time.sleep(0.1)
        url = f"http://127.0.0.1:{port_file.read_text().strip()}"
        with Client(url) as c:
            while c.healthz()[1].get("reasons") and time.monotonic() < deadline:
                time.sleep(0.1)  # wait for the first controller tick (lease and inventory)
            print(f"service at {url}: health {c.healthz()[1]}")
            r = subprocess.run(
                cli(
                    "replay",
                    str(trace),
                    "--url",
                    url,
                    "--scale",
                    str(a.scale),
                    "--wait",
                    "--wait-timeout",
                    "300",
                ),
                cwd=ROOT,
                env=child_env(drop=("AWP_PG_DSN",)),
                capture_output=True,
                text=True,
                timeout=400,
                check=True,
            )
            print("replay:", r.stdout.strip().replace("\n", " "))
            print("namespaces:", json.dumps(c.namespaces()))
            print(
                "nodes:",
                json.dumps([{k: n[k] for k in ("name", "gpus", "free_gpus", "ready")} for n in c.nodes()]),
            )
            first = next(c.list("team-a"))
            print(
                "workload:",
                json.dumps(
                    {
                        k: first[k]
                        for k in ("id", "state", "version", "submit_s", "first_started_s", "terminal_s")
                    }
                ),
            )
            print("usage team-a:", json.dumps(c.usage("team-a")))
            print("events:", len(c.events()))
            lines = [
                ln
                for ln in c.metrics().splitlines()
                if ln.startswith(("awp_workloads{", "awp_attempts_total", "awp_controller_leader"))
                and not ln.endswith(" 0.0")
            ]
            print("metrics:\n  " + "\n  ".join(lines))
    finally:
        kill_tree(proc.pid)
        proc.wait(timeout=30)
        print(f"service stopped (PID {proc.pid} alive: {pid_alive(proc.pid)})")
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
