"""Prepare the local kind cluster for the platform (simulated GPUs).

- labels the workers `awp.local/node` (the node name), `awp.local/rack` (r0, r0, r1, r1, ...), and
  `awp.local/class` (a100);
- advertises GPUs per worker by a JSON patch of `status.capacity` with `nvidia.com~1gpu`
  (Kubernetes docs, "Advertise Extended Resources for a Node");
- loads the workload image into the nodes (`kind load docker-image`; the nodes may not reach Docker Hub);
- creates the workload namespace.

Usage: python scripts/kind_setup.py --kubeconfig PATH [--gpus 8] [--image busybox:1.37] [--kind KIND_BINARY]
The cluster itself is created with `kind create cluster --config deploy/kind/cluster.yaml` (deploy/README.md).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

CLUSTER = "cvproject-awp"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kubeconfig", required=True)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--gpu-resource", default="nvidia.com/gpu")
    ap.add_argument("--image", default="busybox:1.37")
    ap.add_argument("--namespace", default="awp-workloads")
    ap.add_argument("--kind", default=shutil.which("kind") or "kind")
    ap.add_argument("--platform", default="linux/amd64")
    ap.add_argument("--skip-load", action="store_true")
    a = ap.parse_args(argv)

    from kubernetes import client, config

    config.load_kube_config(config_file=a.kubeconfig)
    core = client.CoreV1Api()
    workers = sorted(
        n.metadata.name
        for n in core.list_node().items
        if "node-role.kubernetes.io/control-plane" not in (n.metadata.labels or {})
    )
    if not workers:
        print("no worker nodes found", file=sys.stderr)
        return 1
    path_res = a.gpu_resource.replace("~", "~0").replace("/", "~1")
    for i, name in enumerate(workers):
        rack = f"r{i // 2}"
        core.patch_node(
            name,
            {
                "metadata": {
                    "labels": {"awp.local/node": name, "awp.local/rack": rack, "awp.local/class": "a100"}
                }
            },
        )
        core.patch_node_status(
            name, [{"op": "add", "path": f"/status/capacity/{path_res}", "value": str(a.gpus)}]
        )
        print(f"{name}: rack {rack}, {a.gpus} x {a.gpu_resource}")
    try:
        core.create_namespace(
            client.V1Namespace(
                metadata=client.V1ObjectMeta(
                    name=a.namespace, labels={"app.kubernetes.io/managed-by": "ai-workload-platform"}
                )
            )
        )
        print(f"namespace {a.namespace} created")
    except client.ApiException as e:
        if e.status != 409:
            raise
        print(f"namespace {a.namespace} exists")
    if not a.skip_load:
        # `kind load docker-image` fails on Docker's containerd image store with multi-platform images
        # ("content digest ... not found"); a single-platform archive works.
        with tempfile.TemporaryDirectory() as tmp:
            tar = os.path.join(tmp, "image.tar")
            for cmd in (
                ["docker", "save", "--platform", a.platform, a.image, "-o", tar],
                [a.kind, "load", "image-archive", "--name", CLUSTER, tar],
            ):
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
                if r.returncode != 0:
                    print(r.stdout.strip(), r.stderr.strip(), file=sys.stderr)
                    return r.returncode
        print(f"image {a.image} loaded into the nodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
