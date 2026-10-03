"""Tier 2 item 2: run the platform inside the kind cluster and feed it a small trace.

Applies deploy/k8s/*.yaml (namespaces, least-privilege RBAC, ConfigMap, Deployment, Service) with the
Python client, waits for the Deployment, then runs a Job in `awp-system` with the same image that generates a
small trace and replays it against the in-cluster Service (`replay --wait`), and prints its output (the
replay summary, /healthz, and a few /metrics lines). Finally it deletes what it applied (`--keep` keeps it).
The image `ai-workload-platform:dev` must be loaded into the nodes first (`docker save` + `kind load
image-archive`, see deploy/README.md). Simulated GPUs: workloads are busybox pods that sleep.

Usage: python scripts/kind_platform.py --kubeconfig PATH [--jobs 12] [--time-scale 60] [--keep]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPLAY = (
    "python -m ai_workload_platform gen --seed 1 --jobs {jobs} --runtime-max-s 1200 --out /tmp/trace.csv"
    " && python -m ai_workload_platform replay /tmp/trace.csv --url http://awp-platform.awp-system.svc:18400"
    " --time-scale {ts} --wait --wait-timeout 900"
    " && python -c \"import urllib.request as u; b='http://awp-platform.awp-system.svc:18400';"
    " print(u.urlopen(b+'/healthz').read().decode());"
    " print(chr(10).join(l for l in u.urlopen(b+'/metrics').read().decode().splitlines()"
    " if l.startswith(('awp_workloads{{', 'awp_attempts_total', 'awp_controller_leader', 'awp_gpus_capacity'))"
    " and not l.endswith(' 0.0')))\""
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kubeconfig", required=True)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--time-scale", type=float, default=60.0)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    from kubernetes import client, config, utils

    config.load_kube_config(config_file=a.kubeconfig)
    api = client.ApiClient()
    core, apps, batch, rbac = (
        client.CoreV1Api(),
        client.AppsV1Api(),
        client.BatchV1Api(),
        client.RbacAuthorizationV1Api(),
    )
    files = ["namespaces.yaml", "rbac.yaml", "configmap.yaml", "deployment.yaml"]
    for f in files:
        for doc in yaml.safe_load_all((ROOT / "deploy" / "k8s" / f).read_text(encoding="utf-8")):
            try:
                utils.create_from_dict(api, doc)
            except utils.FailToCreateError as e:
                if not all(getattr(x, "status", 0) == 409 for x in e.api_exceptions):
                    raise
    print("applied:", ", ".join(files), flush=True)
    t0 = time.monotonic()
    while True:
        d = apps.read_namespaced_deployment("awp-platform", "awp-system")
        if (d.status.ready_replicas or 0) >= 1:
            break
        if time.monotonic() - t0 > 300:
            print("deployment not ready", file=sys.stderr)
            return 1
        time.sleep(1)
    print(f"deployment ready after {time.monotonic() - t0:.1f} s", flush=True)
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": "awp-replay", "namespace": "awp-system"},
        "spec": {
            "backoffLimit": 0,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "replay",
                            "image": "ai-workload-platform:dev",
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["sh", "-c", REPLAY.format(jobs=a.jobs, ts=a.time_scale)],
                        }
                    ],
                }
            },
        },
    }
    batch.create_namespaced_job("awp-system", job)
    t1 = time.monotonic()
    while True:
        j = batch.read_namespaced_job("awp-replay", "awp-system")
        if (j.status.succeeded or 0) + (j.status.failed or 0) >= 1:
            break
        if time.monotonic() - t1 > 1200:
            print("replay job did not finish", file=sys.stderr)
            return 1
        time.sleep(2)
    pods = core.list_namespaced_pod("awp-system", label_selector="job-name=awp-replay").items
    print(f"replay job {'succeeded' if j.status.succeeded else 'failed'} after {time.monotonic() - t1:.1f} s")
    for p in pods:
        text = core.read_namespaced_pod_log(p.metadata.name, "awp-system")
        print(text.decode() if isinstance(text, bytes) else text)
    if not a.keep:
        batch.delete_namespaced_job("awp-replay", "awp-system", propagation_policy="Background")
        core.delete_namespace("awp-system")
        rbac.delete_cluster_role_binding("awp-nodes-readonly")
        rbac.delete_cluster_role("awp-nodes-readonly")
        print(
            "deleted namespace awp-system and the cluster-scoped RBAC (awp-workloads is kept for the backend)"
        )
    return 0 if j.status.succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
