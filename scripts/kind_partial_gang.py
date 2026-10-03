"""Tier 2 item 4: a partial gang on the kind cluster in `delegate` mode, ended by R2.

Pods that the platform does not own (another tenant of the cluster) take all GPUs of three workers; the
platform's books do not see them, so the policy starts a 2 x 8-GPU gang. The kube-scheduler places one worker
on the free node and leaves the other `Pending`; the attempt stays `STARTING` until R2 stops it after
`start_timeout_ms` (reason `start_timeout`, counted). The other tenant's pods are then deleted and the retry
runs the whole gang. The service runs as a child process (Kubernetes backend, delegate, time_scale 10).
Simulated GPUs: the pods sleep. Usage: python scripts/kind_partial_gang.py --kubeconfig PATH
Writes benchmarks/results/cluster/partial_gang.json.
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
NS = "awp-workloads"
OTHER = {"app.kubernetes.io/name": "awp-demo-other-tenant", "cvproject": "ai-workload-platform"}
START_TIMEOUT_MS = 30_000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kubeconfig", required=True)
    ap.add_argument("--image", default="busybox:1.37")
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "cluster" / "partial_gang.json"))
    a = ap.parse_args()
    from kubernetes import client, config

    from ai_workload_platform.client import Client
    from ai_workload_platform.procutil import child_env, kill_tree, popen_kwargs

    config.load_kube_config(a.kubeconfig)
    core = client.CoreV1Api()
    kind_nodes = [
        n["name"] for n in json.loads((ROOT / "configs" / "clusters" / "kind.json").read_text())["nodes"]
    ]
    taken = kind_nodes[:3]

    def other_pods() -> list:
        return core.list_namespaced_pod(
            NS, label_selector="app.kubernetes.io/name=" + OTHER["app.kubernetes.io/name"]
        ).items

    def delete_other() -> None:
        for p in other_pods():
            core.delete_namespaced_pod(p.metadata.name, NS, grace_period_seconds=0)

    def gang_pods(wid: str) -> list[tuple[str, str | None, str]]:
        ps = core.list_namespaced_pod(NS, label_selector=f"awp.local/workload={wid}").items
        return sorted((p.metadata.name, p.spec.node_name, p.status.phase) for p in ps)

    tmp = Path(tempfile.mkdtemp(prefix="awp-gang-"))
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
            "delegate",
            "--thresholds",
            json.dumps({"start_timeout_ms": START_TIMEOUT_MS}),
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
    pods_seen: dict[str, list] = {}
    try:
        for i, node in enumerate(taken):  # the other tenant: 8 GPUs on each of three workers
            core.create_namespaced_pod(
                NS,
                {
                    "apiVersion": "v1",
                    "kind": "Pod",
                    "metadata": {"name": f"awp-demo-other-{i}", "labels": OTHER},
                    "spec": {
                        "restartPolicy": "Never",
                        "terminationGracePeriodSeconds": 0,
                        "nodeSelector": {"awp.local/node": node},
                        "containers": [
                            {
                                "name": "other",
                                "image": a.image,
                                "imagePullPolicy": "IfNotPresent",
                                "command": ["sleep", "3600"],
                                "resources": {
                                    "requests": {"nvidia.com/gpu": "8", "cpu": "10m", "memory": "16Mi"},
                                    "limits": {"nvidia.com/gpu": "8"},
                                },
                            }
                        ],
                    },
                },
            )
        deadline = time.monotonic() + 120
        while not (len(other_pods()) == 3 and all(p.status.phase == "Running" for p in other_pods())):
            assert time.monotonic() < deadline, "the other tenant's pods did not start"
            time.sleep(0.5)
        while not (tmp / "port").exists():
            if proc.poll() is not None:
                raise RuntimeError((tmp / "service.log").read_text(errors="replace")[-2000:])
            time.sleep(0.1)
        url = f"http://127.0.0.1:{(tmp / 'port').read_text().strip()}"
        with Client(url, timeout_s=30) as c:
            while not c.nodes():
                time.sleep(0.2)
            t0 = time.monotonic()
            c.submit(
                "team-a",
                {
                    "id": "gang",
                    "gpus": 8,
                    "workers": 2,
                    "estimate_s": 600,
                    "sim": {"runtime_s": 600},
                    "retry": {"max_attempts": 3, "backoff_base_s": 5, "jitter": "none"},
                },
                "gang",
            )
            while True:
                w = c.get("team-a", "gang")
                ps = gang_pods("gang")
                if w["attempts"] and "started" not in marks:
                    marks["started"] = time.monotonic() - t0
                if [p[2] for p in ps].count("Running") == 1 and "one_worker_running" not in marks:
                    marks["one_worker_running"] = time.monotonic() - t0
                    pods_seen["first_attempt"] = ps
                if w["attempts"] and w["attempts"][0]["state"] == "STOPPING":
                    marks["stop_requested"] = time.monotonic() - t0
                    pods_seen.setdefault("first_attempt", ps)
                    break
                assert time.monotonic() - t0 < 300, f"R2 did not act: {w['state']} {ps}"
                time.sleep(0.5)
            delete_other()
            marks["other_tenant_deleted"] = time.monotonic() - t0
            while True:
                w = c.get("team-a", "gang")
                if w["attempts"][0]["state"] == "ENDED" and "attempt_ended" not in marks:
                    marks["attempt_ended"] = time.monotonic() - t0
                if (
                    len(w["attempts"]) > 1
                    and w["attempts"][1]["state"] == "RUNNING"
                    and "retry_running" not in marks
                ):
                    marks["retry_running"] = time.monotonic() - t0
                    pods_seen["retry"] = gang_pods("gang")
                if w["state"] in ("SUCCEEDED", "FAILED", "DEAD_LETTER", "CANCELLED"):
                    marks["terminal"] = time.monotonic() - t0
                    break
                assert time.monotonic() - t0 < 600, f"no retry: {w['state']}"
                time.sleep(0.5)
            final = c.get("team-a", "gang")
            events = [e for e in c.events() if e["workload_id"] == "gang"]
    finally:
        delete_other()
        kill_tree(proc.pid)
        proc.wait(timeout=60)
        log.close()
    first = final["attempts"][0]
    out = {
        "note": "simulated GPUs on a local kind cluster; delegate mode; wall-clock seconds on a shared machine; one run",
        "thresholds_ms": {"start_timeout_ms": START_TIMEOUT_MS},
        "other_tenant_nodes": taken,
        "marks_s": {k: round(v, 1) for k, v in marks.items()},
        "pods": {k: [{"pod": n, "node": nd, "phase": ph} for n, nd, ph in v] for k, v in pods_seen.items()},
        "final_state": final["state"],
        "first_attempt": {
            "end_reason": first["end_reason"],
            "stop_reason": first["stop_reason"],
            "counted": first["counted"],
            "placement_by_policy": first["placement"],
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
