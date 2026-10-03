"""E2 (c): the E2 trace through the live service with the Kubernetes backend on a real (kind) cluster,
`pinned` and `delegate`, 3 repetitions each. Simulated GPUs: the pods sleep for run time / time_scale."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from ai_workload_platform.models import Event

ROOT = Path(__file__).resolve().parents[3]
CLUSTERS = ROOT / "configs" / "clusters"
TERMINAL = {"SUCCEEDED", "FAILED", "DEAD_LETTER", "CANCELLED"}


def _scaled(events: list[Event], f: float) -> list[Event]:
    out = []
    for e in events:
        d = dict(e.data)
        for k in ("observed_started_ms", "observed_ended_ms"):
            if d.get(k) is not None:
                d[k] = int(round(d[k] * f))
        out.append(Event(e.seq, int(round(e.at_ms * f)), e.type, e.namespace, e.workload_id, e.attempt_id, d))
    return out


def _tool_versions(kubeconfig: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, cmd in (
        ("docker", ["docker", "version", "--format", "{{.Server.Version}}"]),
        ("kind", [shutil.which("kind") or "kind", "version"]),
    ):
        try:
            out[name] = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, check=False
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            out[name] = "unknown"
    try:
        from ai_workload_platform.scheduler.kube.client import RealKubeClient

        nodes = RealKubeClient(kubeconfig).list_nodes()
        info = nodes[0]["status"]["nodeInfo"]
        out["kubernetes"] = info.get("kubeletVersion")
        out["node_os"] = info.get("osImage")
        out["container_runtime"] = info.get("containerRuntimeVersion")
    except Exception:  # noqa: BLE001
        pass
    return out


def run_cluster(
    kubeconfig: str | None,
    out: Path,
    *,
    reps: int = 3,
    configs: tuple[tuple[str, float], ...] = (("pinned", 60.0), ("delegate", 60.0)),
    seed: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """`configs`: (mode, target length in seconds of the longest workload); the time scale of each is the
    longest run time of the trace divided by the target."""
    from ai_workload_platform.bench.generator import generate
    from ai_workload_platform.client import Client
    from ai_workload_platform.cluster import load_cluster
    from ai_workload_platform.observability.evaluation import compute
    from ai_workload_platform.procutil import child_env, kill_tree, popen_kwargs
    from ai_workload_platform.scheduler.kube.client import ApiError, RealKubeClient
    from ai_workload_platform.sim import reference_namespaces
    from ai_workload_platform.store.dialect import SQLiteDialect
    from ai_workload_platform.store.sql import Store

    if not kubeconfig:
        raise SystemExit("bench --cluster needs --kubeconfig or AWP_KUBECONFIG (see deploy/README.md)")
    kube = RealKubeClient(kubeconfig)
    kube.list_nodes()
    ref = load_cluster(CLUSTERS / "reference-e2.json")
    kind = load_cluster(CLUSTERS / "kind.json")
    rows_trace, _ = generate(ref, seed, jobs=40, variant="balanced", load=0.8)
    longest = max(r.runtime_s for r in rows_trace)
    nss = reference_namespaces(kind.total_gpus)
    results: list[dict[str, Any]] = []
    for mode, target in configs:
        time_scale = round(longest / target, 1)
        for rep in range(1, reps + 1):
            tmp = Path(tempfile.mkdtemp(prefix="awp-e2c-"))
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
                    kubeconfig,
                    "--cluster",
                    str(CLUSTERS / "kind.json"),
                    "--time-scale",
                    str(time_scale),
                    "--kube-mode",
                    mode,
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
            t_wall0 = time.monotonic()
            try:
                while not (tmp / "port").exists():
                    if proc.poll() is not None or time.monotonic() - t_wall0 > 120:
                        raise RuntimeError((tmp / "service.log").read_text(errors="replace")[-2000:])
                    time.sleep(0.1)
                url = f"http://127.0.0.1:{(tmp / 'port').read_text().strip()}"
                with Client(url, timeout_s=30) as c:
                    while not c.nodes():
                        time.sleep(0.2)
                    t0 = time.monotonic()
                    for r in rows_trace:
                        delay = r.submit_s / time_scale - (time.monotonic() - t0)
                        if delay > 0:
                            time.sleep(delay)
                        c.submit(r.tenant, r.to_spec(), f"trace-{r.job_id}")
                    deadline = time.monotonic() + 1800
                    while True:
                        states = [w["state"] for ns in ("team-a", "team-b", "team-c") for w in c.list(ns)]
                        if len(states) == len(rows_trace) and all(s in TERMINAL for s in states):
                            break
                        if time.monotonic() > deadline:
                            raise RuntimeError(f"run did not finish: {states}")
                        time.sleep(1.0)
                    wall = time.monotonic() - t0
                    evs = [
                        Event(
                            e["seq"],
                            e["at_ms"],
                            e["type"],
                            e["namespace"],
                            e["workload_id"],
                            e["attempt_id"],
                            e["data"],
                        )
                        for e in c.events()
                    ]
            finally:
                kill_tree(proc.pid)
                proc.wait(timeout=60)
                log.close()
            s = Store(SQLiteDialect(str(tmp / "run.db")))
            inst = s.instance_id
            s.close()
            sel = f"app.kubernetes.io/managed-by=ai-workload-platform,awp.local/instance={inst}"
            for j in kube.list_jobs("awp-workloads", sel):  # leftovers of this instance only
                try:
                    kube.delete_job("awp-workloads", j["metadata"]["name"])
                except ApiError:
                    pass
            real = compute(evs, kind, nss)
            trace = compute(_scaled(evs, time_scale), kind, nss)
            results.append(
                {
                    "mode": mode,
                    "target_longest_s": target,
                    "rep": rep,
                    "seed": seed,
                    "time_scale": time_scale,
                    "workloads": len(rows_trace),
                    "makespan_s": trace["makespan_s"],
                    "wait_mean": trace["wait_mean"],
                    "wait_p95": trace["wait_p95"],
                    "bsld_p95": trace["bsld_p95"],
                    "utilization": trace["utilization"],
                    "running_over_allocated": trace["running_over_allocated"],
                    "admit_to_running_ms_mean": real["admit_to_running_ms_mean"],
                    "admit_to_running_ms_p95": real["admit_to_running_ms_p95"],
                    "observe_lag_ms_mean": real["observe_lag_ms_mean"],
                    "observe_lag_ms_p95": real["observe_lag_ms_p95"],
                    "observe_lag_ms_max": real["observe_lag_ms_max"],
                    "placement_match": real["placement_match"],
                    "attempts": real["attempts"],
                    "attempts_by_reason": json.dumps(real["attempts_by_reason"], sort_keys=True),
                    "terminal_states": json.dumps(real["terminal_states"], sort_keys=True),
                    "wall_makespan_s": round(real["makespan_s"], 3),
                    "wall_run_s": round(wall, 1),
                }
            )
            print(json.dumps(results[-1]), flush=True)
            shutil.rmtree(tmp, ignore_errors=True)
    extra = {
        "cluster": {
            "trace_seed": seed,
            "configs": [
                {"mode": m, "target_longest_s": t, "time_scale": round(longest / t, 1)} for m, t in configs
            ],
            "repetitions": reps,
            "longest_runtime_s": max(r.runtime_s for r in rows_trace),
            **_tool_versions(kubeconfig),
        }
    }
    return results, extra
