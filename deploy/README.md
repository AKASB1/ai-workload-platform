# Deployment

Three shapes, all for **simulated** GPU workloads (they sleep; nothing runs a model).

## Docker Compose (PostgreSQL + the platform)

```bash
docker compose up
```

`docker-compose.yml` (project name `awp`) starts `postgres:17-alpine` (published on `127.0.0.1:18432`) and the platform image built from the `Dockerfile`, with the local backend on a scaled clock (`--scale 60`: one wall second is one platform minute) and the store in PostgreSQL (`AWP_PG_DSN`). The API is published on `127.0.0.1:18400`. Everything carries the label `cvproject=ai-workload-platform`; `docker compose down` removes the containers and the network.

If the build cannot reach PyPI directly, pass a proxy or a package index at build time (nothing is configured in the repository), for example `docker compose build --build-arg HTTPS_PROXY=http://host.docker.internal:<port>` or `--build-arg PIP_INDEX_URL=<mirror>`.

**Applied in this repository's run:** yes — `docker compose up` was started, a workload was submitted through the API and succeeded, and the stack was removed again.

## A local Kubernetes cluster with kind

The Kubernetes backend runs workloads as Indexed Jobs (`docs/kubernetes.md`). For the real-cluster experiments a kind cluster with one control plane and four workers is used:

```bash
kind create cluster --name cvproject-awp --config deploy/kind/cluster.yaml --kubeconfig outputs/kind/kubeconfig
python scripts/kind_setup.py --kubeconfig outputs/kind/kubeconfig
```

`scripts/kind_setup.py` labels the workers (`awp.local/node`, `awp.local/rack` = r0, r0, r1, r1, `awp.local/class` = a100), advertises 8 `nvidia.com/gpu` per worker by a JSON patch of the node status (the Kubernetes documentation page "Advertise Extended Resources for a Node"), creates the namespace `awp-workloads`, and loads `busybox:1.37` into the nodes. `kind load docker-image` fails on Docker's containerd image store for multi-platform images, so the script saves a single-platform archive and uses `kind load image-archive`. The kubeconfig stays outside `~/.kube/config`; the API server listens on `127.0.0.1:18443`. Remove the cluster by its name: `kind delete cluster --name cvproject-awp`.

Then point the platform at it:

```bash
python -m ai_workload_platform up --backend kube --kubeconfig outputs/kind/kubeconfig --cluster configs/clusters/kind.json --time-scale 60
```

**Applied in this repository's run:** the cluster was created (kind v0.33.0, node image `kindest/node:v1.37.0`), the backend contract suite ran against it (`AWP_KUBECONFIG=... python -m pytest tests/test_backend_contract.py`), E2 (c) ran the benchmark trace through the service with the Kubernetes backend (`python -m ai_workload_platform bench --cluster`), and the time-scale sweep (`bench --cluster --sweep`), the partial-gang demonstration (`python scripts/kind_partial_gang.py --kubeconfig ...`), and the node-loss demonstration (`python scripts/kind_node_loss.py --kubeconfig ...`; it stops and restarts the kind worker container `cvproject-awp-worker4` by name) ran afterwards; see `benchmarks/README.md` and `docs/kubernetes.md`.

## Kubernetes manifests for the platform itself

`deploy/k8s/` holds the manifests to run the platform inside a cluster with the in-cluster configuration:

| File | Objects |
|---|---|
| `namespaces.yaml` | `awp-system` (the platform), `awp-workloads` (the Jobs) |
| `rbac.yaml` | ServiceAccount `awp-controller`; Role + RoleBinding in `awp-workloads` (Jobs: get, list, create, delete; pods: get, list); ClusterRole + ClusterRoleBinding for nodes (get, list) — nodes are cluster-scoped, so a namespaced Role cannot grant them |
| `configmap.yaml` | `cluster.json` (the kind cluster configuration) |
| `deployment.yaml` | Deployment (one replica, `Recreate`, non-root, read-only root file system, `/healthz` probes) and a ClusterIP Service on port 18400 |

The Deployment keeps its SQLite file on an `emptyDir`, which is enough for a demonstration but loses the store with the pod; a durable deployment sets `AWP_PG_DSN`.

The manifests pass `kubeconform -strict` (10 resources). **Applied (Tier 2 item 2):** `scripts/kind_platform.py` applies them to the kind cluster (after `docker save` + `kind load image-archive` of `ai-workload-platform:dev`), waits for the Deployment, runs a Job in `awp-system` with the same image that generates a 12-workload trace and replays it against the in-cluster Service with `--time-scale 60`, prints the replay summary, `/healthz`, and `/metrics`, and deletes what it applied:

```bash
python scripts/kind_platform.py --kubeconfig outputs/kind/kubeconfig
```

In this repository's run all 12 workloads succeeded through the in-cluster platform (Kubernetes backend, in-cluster configuration, the RBAC above); see `docs/kubernetes.md`.
