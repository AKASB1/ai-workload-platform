"""Deterministic fake of the Kubernetes API surface the backend uses (docs/kubernetes.md).

Jobs and pods live in memory on the injected (virtual) clock. It models: Indexed Jobs with
backoffLimit 0 (one pod per completion), a stand-in for the kube-scheduler (pinned: the node of
`awp.local/node`; delegate: the ready node with the most free GPUs, ties by name; a pod that fits
nowhere stays Pending), container start and termination latencies (assumed defaults, see
docs/kubernetes.md), `sh -c "sleep X; exit C"` workloads, background deletion, AlreadyExists,
node failures, eviction, a Job that stays pending, and API errors. Nothing else.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from ai_workload_platform.clock import Clock
from ai_workload_platform.cluster import ClusterConfig
from ai_workload_platform.scheduler.kube.client import ApiError, format_ts

EPOCH_MS = 946_684_800_000  # 2000-01-01T00:00:00Z: virtual ms 0 in the fake's timestamps
_CMD = re.compile(r"sleep ([0-9.]+); exit ([0-9]+)")

DEFAULT_LATENCIES = {"schedule_ms": 50, "container_start_ms": 1000, "delete_ms": 2000}


@dataclass
class _Pod:
    ns: str
    name: str
    job: str
    labels: dict[str, str]
    created_ms: int
    gpus: int
    selector: str | None
    sleep_ms: int
    exit_code: int
    seq: int
    node: str | None = None
    bind_at: int = 0
    run_at: int | None = None
    started_ms: int | None = None
    end_at: int | None = None
    finished_ms: int | None = None
    phase: str = "Pending"
    reason: str | None = None
    term_exit: int | None = None
    delete_at: int | None = None  # deletion requested; gone at this time (unless its node is down)
    deleting: bool = False
    frozen: bool = False
    never_schedule: bool = False
    bind_due: bool = True  # the first bind attempt at bind_at is still to come


@dataclass
class FakeFaults:
    api_errors: int = 0  # the next n calls raise ApiError(500)
    pending_jobs: set[str] = field(default_factory=set)  # Job names whose pods are never scheduled
    down_ops: set[str] = field(default_factory=set)  # operations that answer 503 while listed
    calls: dict[str, int] = field(default_factory=dict)


class FakeKubeClient:
    def __init__(
        self,
        cluster: ClusterConfig,
        clock: Clock,
        *,
        gpu_resource: str = "nvidia.com/gpu",
        latencies: dict[str, int] | None = None,
    ) -> None:
        self.cluster = cluster
        self.clock = clock
        self.gpu_resource = gpu_resource
        self.lat = {**DEFAULT_LATENCIES, **(latencies or {})}
        self.nodes: dict[str, dict[str, Any]] = {}
        for n in cluster.nodes:
            self.nodes[n.name] = {
                "name": n.name,
                "rack": n.rack,
                "class": n.gpu_class,
                "gpus": n.gpus,
                "cpus": n.cpus,
                "mem_mb": n.mem_mb,
                "ready": True,
            }
        self.jobs: dict[tuple[str, str], dict[str, Any]] = {}
        self.pods: dict[tuple[str, str], _Pod] = {}
        self.faults = FakeFaults()
        self._seq = 0

    # --- API ---------------------------------------------------------------------------------
    def _api(self, op: str) -> None:
        self.faults.calls[op] = self.faults.calls.get(op, 0) + 1
        if op in self.faults.down_ops:
            raise ApiError(503, "ServiceUnavailable", f"{op} unavailable (injected)")
        if self.faults.api_errors > 0:
            self.faults.api_errors -= 1
            raise ApiError(500, "InternalError", f"{op} failed (injected)")
        self._advance(self.clock.now_ms())

    def list_nodes(self) -> list[dict[str, Any]]:
        self._api("list_nodes")
        out = []
        for name in sorted(self.nodes):
            n = self.nodes[name]
            out.append(
                {
                    "metadata": {
                        "name": name,
                        "labels": {
                            "awp.local/node": name,
                            "awp.local/rack": n["rack"],
                            "awp.local/class": n["class"],
                        },
                    },
                    "spec": {},
                    "status": {
                        "capacity": {
                            self.gpu_resource: str(n["gpus"]),
                            "cpu": str(n["cpus"]),
                            "memory": f"{n['mem_mb']}M",
                        },
                        "conditions": [{"type": "Ready", "status": "True" if n["ready"] else "Unknown"}],
                    },
                }
            )
        return out

    def list_jobs(self, namespace: str, label_selector: str) -> list[dict[str, Any]]:
        self._api("list_jobs")
        sel = _selector(label_selector)
        return [
            self._job_json(k)
            for k in sorted(self.jobs)
            if k[0] == namespace and _match(self.jobs[k]["metadata"]["labels"], sel)
        ]

    def list_pods(self, namespace: str, label_selector: str) -> list[dict[str, Any]]:
        self._api("list_pods")
        sel = _selector(label_selector)
        return [
            self._pod_json(p)
            for k, p in sorted(self.pods.items())
            if k[0] == namespace and _match(p.labels, sel)
        ]

    def create_job(self, namespace: str, body: dict[str, Any]) -> dict[str, Any]:
        self._api("create_job")
        name = body["metadata"]["name"]
        if (namespace, name) in self.jobs:
            raise ApiError(409, "AlreadyExists", f"jobs.batch {name!r} already exists")
        now = self.clock.now_ms()
        job = copy.deepcopy(body)
        job["metadata"]["creationTimestamp"] = format_ts(EPOCH_MS + now)
        self.jobs[(namespace, name)] = job
        spec = job["spec"]
        tmpl = spec["template"]
        cont = tmpl["spec"]["containers"][0]
        gpus = int(cont["resources"]["requests"].get(self.gpu_resource, 0))
        m = _CMD.search(" ".join(cont.get("command") or []))
        sleep_ms = int(round(float(m.group(1)) * 1000)) if m else 0
        code = int(m.group(2)) if m else 0
        sel = (tmpl["spec"].get("nodeSelector") or {}).get("awp.local/node")
        for i in range(int(spec["completions"])):
            self._seq += 1
            labels = dict(tmpl["metadata"].get("labels") or {})
            labels["batch.kubernetes.io/job-completion-index"] = str(i)
            p = _Pod(
                namespace,
                f"{name}-{i}",
                name,
                labels,
                now,
                gpus,
                sel,
                sleep_ms,
                code,
                self._seq,
                bind_at=now + self.lat["schedule_ms"],
                never_schedule=name in self.faults.pending_jobs,
            )
            self.pods[(namespace, p.name)] = p
        return self._job_json((namespace, name))

    def delete_job(self, namespace: str, name: str) -> None:
        self._api("delete_job")
        if (namespace, name) not in self.jobs:
            raise ApiError(404, "NotFound", f"jobs.batch {name!r} not found")
        del self.jobs[(namespace, name)]
        now = self.clock.now_ms()
        for p in self.pods.values():
            if p.ns == namespace and p.job == name and not p.deleting:
                p.deleting = True
                # a pod that never ran, or has finished, goes at once; a running one after the grace period
                p.delete_at = now + (self.lat["delete_ms"] if p.phase == "Running" else 0)
        self._advance(now)

    # --- model -------------------------------------------------------------------------------
    def _free(self) -> dict[str, int]:
        free = {n: v["gpus"] for n, v in self.nodes.items()}
        for p in self.pods.values():
            if p.node is not None and p.phase in ("Pending", "Running"):
                free[p.node] -= p.gpus
        return free

    def _advance(self, now: int) -> None:
        while True:
            t = self._next_due(now)
            if t is None:
                break
            self._step(t)
        self._try_bind(now)

    def _next_due(self, now: int) -> int | None:
        best = None
        for p in self.pods.values():
            for x in self._pod_times(p):
                if x is not None and x <= now and (best is None or x < best):
                    best = x
        return best

    def _pod_times(self, p: _Pod) -> tuple[int | None, ...]:
        if p.frozen:
            return ()
        node_up = p.node is None or self.nodes[p.node]["ready"]
        return (
            p.bind_at
            if p.bind_due
            and p.node is None
            and p.phase == "Pending"
            and not p.deleting
            and not p.never_schedule
            else None,
            p.delete_at if p.deleting and node_up else None,
            p.run_at if p.phase == "Pending" and p.node is not None and not p.deleting else None,
            p.end_at if p.phase == "Running" and not p.deleting else None,
        )

    def _step(self, t: int) -> None:
        # deletions and completions first (they free capacity), then starts, then binds
        for key in sorted(self.pods, key=lambda k: self.pods[k].seq):
            p = self.pods.get(key)
            if p is None or p.frozen:
                continue
            if (
                p.deleting
                and p.delete_at is not None
                and p.delete_at <= t
                and (p.node is None or self.nodes[p.node]["ready"])
            ):
                del self.pods[key]
                continue
            if p.phase == "Running" and not p.deleting and p.end_at is not None and p.end_at <= t:
                p.phase = "Succeeded" if p.exit_code == 0 else "Failed"
                p.finished_ms = p.end_at
                p.term_exit = p.exit_code
                if p.phase == "Failed":
                    self._fail_job(p, t)
        for p in sorted(self.pods.values(), key=lambda x: x.seq):
            if p.frozen or p.deleting:
                continue
            if p.phase == "Pending" and p.node is not None and p.run_at is not None and p.run_at <= t:
                p.phase = "Running"
                p.started_ms = p.run_at
                p.end_at = p.run_at + p.sleep_ms
        self._try_bind(t)

    def _fail_job(self, failed: _Pod, t: int) -> None:
        """backoffLimit 0: the Job fails with its first failed pod, and the Job controller deletes the rest."""
        for q in self.pods.values():
            if (
                q is not failed
                and q.ns == failed.ns
                and q.job == failed.job
                and q.phase in ("Pending", "Running")
                and not q.deleting
            ):
                q.deleting = True
                q.delete_at = t + (self.lat["delete_ms"] if q.phase == "Running" else 0)

    def _try_bind(self, t: int) -> None:
        free = None
        for p in sorted(self.pods.values(), key=lambda x: x.seq):
            if p.node is not None or p.deleting or p.never_schedule or p.bind_at > t or p.phase != "Pending":
                continue
            p.bind_due = False
            if free is None:
                free = self._free()
            ready = [n for n in sorted(self.nodes) if self.nodes[n]["ready"] and free[n] >= p.gpus]
            if p.selector is not None:
                node = p.selector if p.selector in ready else None
            else:
                node = min(ready, key=lambda n: (-free[n], n), default=None)
            if node is None:
                continue
            p.node = node
            p.run_at = t + self.lat["container_start_ms"]
            free[node] -= p.gpus

    def next_event_ms(self) -> int | None:
        now = self.clock.now_ms()
        best = None
        for p in self.pods.values():
            cand = list(self._pod_times(p))
            for x in cand:
                if x is not None and x > now and (best is None or x < best):
                    best = x
        return best

    # --- JSON views --------------------------------------------------------------------------
    def _pod_json(self, p: _Pod) -> dict[str, Any]:
        st: dict[str, Any] = {"phase": p.phase}
        if p.node is not None:
            st["startTime"] = format_ts(EPOCH_MS + p.bind_at)
        if p.reason:
            st["reason"] = p.reason
        if p.phase == "Running":
            cs = {"state": {"running": {"startedAt": format_ts(EPOCH_MS + (p.started_ms or 0))}}}
        elif p.phase in ("Succeeded", "Failed"):
            term: dict[str, Any] = {
                "exitCode": p.term_exit if p.term_exit is not None else 137,
                "finishedAt": format_ts(EPOCH_MS + (p.finished_ms or 0)),
            }
            if p.started_ms is not None:
                term["startedAt"] = format_ts(EPOCH_MS + p.started_ms)
            cs = {"state": {"terminated": term}}
        else:
            cs = {"state": {"waiting": {"reason": "ContainerCreating" if p.node else "Unschedulable"}}}
        st["containerStatuses"] = [{"name": "worker", **cs}]
        meta: dict[str, Any] = {"name": p.name, "namespace": p.ns, "labels": dict(p.labels)}
        if p.deleting:
            meta["deletionTimestamp"] = format_ts(EPOCH_MS + (p.delete_at or 0))
        spec: dict[str, Any] = {}
        if p.node is not None:
            spec["nodeName"] = p.node
        return {"metadata": meta, "spec": spec, "status": st}

    def _job_json(self, key: tuple[str, str]) -> dict[str, Any]:
        job = copy.deepcopy(self.jobs[key])
        pods = [p for p in self.pods.values() if p.ns == key[0] and p.job == key[1]]
        succeeded = sum(1 for p in pods if p.phase == "Succeeded")
        failed = sum(1 for p in pods if p.phase == "Failed")
        active = sum(1 for p in pods if p.phase in ("Pending", "Running") and not p.deleting)
        conds = []
        if failed > 0:
            conds.append({"type": "Failed", "status": "True", "reason": "BackoffLimitExceeded"})
        elif succeeded >= int(job["spec"]["completions"]):
            conds.append({"type": "Complete", "status": "True"})
        job["status"] = {"active": active, "succeeded": succeeded, "failed": failed, "conditions": conds}
        return job

    # --- fault hooks (harness) and ground truth ----------------------------------------------
    def node_down(self, name: str) -> None:
        self._advance(self.clock.now_ms())
        self.nodes[name]["ready"] = False
        for p in self.pods.values():
            if p.node == name and p.phase in ("Pending", "Running"):
                p.frozen = True

    def node_up(self, name: str) -> None:
        now = self.clock.now_ms()
        self.nodes[name]["ready"] = True
        for key, p in list(self.pods.items()):
            if p.node == name and p.frozen:
                p.frozen = False
                if p.deleting:
                    del self.pods[key]
                    continue
                p.phase, p.reason, p.finished_ms, p.term_exit = "Failed", "NodeLost", now, 137
                self._fail_job(p, now)
        self._advance(now)

    def evict(self, pod_name: str) -> bool:
        now = self.clock.now_ms()
        self._advance(now)
        for p in self.pods.values():
            if p.name == pod_name and p.phase == "Running" and not p.frozen and not p.deleting:
                p.phase, p.reason, p.finished_ms, p.term_exit = "Failed", "Evicted", now, 137
                self._fail_job(p, now)
                return True
        return False

    def usage(self) -> dict[str, tuple[int, int, int]]:
        """GPUs held per node by bound pods that are not terminal (terminating pods included)."""
        self._advance(self.clock.now_ms())
        out: dict[str, int] = {}
        for p in self.pods.values():
            if p.node is not None and p.phase in ("Pending", "Running"):
                out[p.node] = out.get(p.node, 0) + p.gpus
        return {n: (g, 0, 0) for n, g in out.items()}

    def active_attempts(self) -> set[str]:
        self._advance(self.clock.now_ms())
        return {
            p.labels.get("awp.local/attempt", "")
            for p in self.pods.values()
            if p.phase in ("Pending", "Running")
        }


def _selector(sel: str) -> dict[str, str]:
    out = {}
    for part in filter(None, (s.strip() for s in sel.split(","))):
        k, _, v = part.partition("=")
        out[k] = v
    return out


def _match(labels: dict[str, str], sel: dict[str, str]) -> bool:
    return all(labels.get(k) == v for k, v in sel.items())
