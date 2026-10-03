"""Kubernetes backend: one Indexed Job per placement entry; level-triggered observation by label.

See docs/kubernetes.md for the mapping, what is not modelled, and the cleanup rules.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ai_workload_platform.cluster import ClusterConfig, rate_of
from ai_workload_platform.models import (
    AttemptRequest,
    AttemptStatus,
    BackendError,
    FailReason,
    NodeInfo,
    Phase,
    Snapshot,
)
from ai_workload_platform.scheduler.kube.client import ApiError, KubeClient, parse_ts_ms

log = logging.getLogger("awp.kube")

MANAGED = "app.kubernetes.io/managed-by"
NODE_REASONS = {
    "NodeLost",
    "NodeShutdown",
    "Terminated",
    "Shutdown",
    "UnexpectedAdmissionError",
    "NodeAffinity",
}


class KubeBackend:
    name = "kube"

    def __init__(
        self,
        client: KubeClient,
        cluster: ClusterConfig,
        *,
        instance_id: str,
        to_platform_ms: Callable[[float], int],
        now_ms: Callable[[], int],
        namespace: str = "awp-workloads",
        mode: str = "pinned",
        gpu_resource: str = "nvidia.com/gpu",
        time_scale: float = 1.0,
        image: str = "busybox:1.37",
    ) -> None:
        if mode not in ("pinned", "delegate"):
            raise ValueError("mode is pinned or delegate")
        self.client = client
        self.cluster = cluster
        self.instance_id = instance_id
        self.to_platform_ms = to_platform_ms
        self.now_ms = now_ms
        self.namespace = namespace
        self.mode = mode
        self.gpu_resource = gpu_resource
        self.time_scale = float(time_scale)
        self.image = image
        self.selector = f"{MANAGED}=ai-workload-platform,awp.local/instance={instance_id}"
        self.stopped: dict[str, int] = {}  # attempt id -> time of the stop call (memory of this process)
        self._meta: dict[str, tuple[float, float, float, int | None]] = {}  # rate, retained, scale, started
        self._cfg_nodes = {n.name: n for n in cluster.nodes}

    # --- contract ------------------------------------------------------------------------------
    def inventory(self) -> list[NodeInfo]:
        try:
            items = self.client.list_nodes()
        except ApiError as e:
            raise BackendError(f"list nodes: {e}") from e
        out = []
        for n in items:
            labels = n["metadata"].get("labels") or {}
            if "awp.local/node" not in labels:
                continue
            name = n["metadata"]["name"]
            cap = n["status"].get("capacity") or {}
            gpus = int(cap.get(self.gpu_resource, "0") or 0)
            # readiness is the Ready condition only: a cordoned node keeps running its pods
            ready = any(
                c.get("type") == "Ready" and c.get("status") == "True"
                for c in n["status"].get("conditions") or []
            )
            cls = labels.get("awp.local/class", "a100")
            cfg = self._cfg_nodes.get(name)
            speed = self.cluster.classes.get(cls, 1.0)
            cpus = cfg.cpus if cfg else _cpu(cap.get("cpu", "0"))
            mem = cfg.mem_mb if cfg else _mem_mb(cap.get("memory", "0"))
            out.append(NodeInfo(name, labels.get("awp.local/rack", "r0"), cls, speed, gpus, cpus, mem, ready))
        if not any(n.gpus > 0 for n in out):
            raise BackendError(
                f"cluster not usable: no node labelled awp.local/node advertises {self.gpu_resource}"
            )
        return sorted(out, key=lambda x: (x.rack, x.name))

    def _job_body(self, req: AttemptRequest, k: int, node: str, workers: int, gang: int) -> dict[str, Any]:
        spec = req.spec
        nodes = [self._cfg_nodes[n] for n, _ in req.placement if n in self._cfg_nodes]
        rate = (
            rate_of(nodes, spec["topology"], self.cluster.cross_node_factor, self.cluster.cross_rack_factor)
            if nodes
            else 1.0
        )
        sim = spec["sim"]
        if req.n <= int(sim.get("fail_attempts") or 0):
            work, code = float(sim["fail_after_s"]), int(sim.get("exit_code") or 1)
        else:
            work, code = float(sim["runtime_s"]) - min(float(req.retained_s), float(sim["runtime_s"])), 0
        secs = work / rate / self.time_scale
        labels = {
            MANAGED: "ai-workload-platform",
            "awp.local/instance": self.instance_id,
            "awp.local/workload": req.workload_id,
            "awp.local/attempt": req.attempt_id,
            "awp.local/namespace": req.namespace,
        }
        pod_spec: dict[str, Any] = {
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 2,
            "containers": [
                {
                    "name": "worker",
                    "image": self.image,
                    "imagePullPolicy": "IfNotPresent",
                    "command": ["sh", "-c", f"sleep {secs:.3f}; exit {code}"],
                    "resources": {
                        "requests": {self.gpu_resource: str(spec["gpus"]), "cpu": "10m", "memory": "16Mi"},
                        "limits": {self.gpu_resource: str(spec["gpus"])},
                    },
                }
            ],
        }
        if self.mode == "pinned":
            pod_spec["nodeSelector"] = {"awp.local/node": node}
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": f"{req.attempt_id}-{k}",
                "labels": labels,
                "annotations": {
                    "awp.local/gang-size": str(gang),
                    "awp.local/intended-node": node,
                    "awp.local/rate": f"{rate:.6f}",
                    "awp.local/retained-s": f"{float(req.retained_s):.3f}",
                    "awp.local/time-scale": f"{self.time_scale:g}",
                },
            },
            "spec": {
                "completionMode": "Indexed",
                "completions": workers,
                "parallelism": workers,
                "backoffLimit": 0,
                "template": {"metadata": {"labels": labels}, "spec": pod_spec},
            },
        }

    def start(self, req: AttemptRequest) -> None:
        created: list[str] = []
        gang = len(req.placement)
        for k, (node, workers) in enumerate(req.placement):
            body = self._job_body(req, k, node, workers, gang)
            try:
                self.client.create_job(self.namespace, body)
                created.append(body["metadata"]["name"])
            except ApiError as e:
                if (
                    e.status == 409
                ):  # AlreadyExists: success only if it is this attempt's Job of this instance
                    if self._owned(body["metadata"]["name"], req.attempt_id):
                        continue
                    raise BackendError(
                        f"a Job named {body['metadata']['name']} exists but does not belong to this "
                        "platform instance"
                    ) from e
                for name in created:  # roll back this call's Jobs so the attempt is absent and R1 repeats
                    try:
                        self.client.delete_job(self.namespace, name)
                    except ApiError:
                        pass
                raise BackendError(f"create job {body['metadata']['name']}: {e}") from e

    def _owned(self, name: str, attempt_id: str) -> bool:
        try:
            jobs = self.client.list_jobs(self.namespace, f"{self.selector},awp.local/attempt={attempt_id}")
        except ApiError as e:
            raise BackendError(f"list jobs: {e}") from e
        return any(j["metadata"]["name"] == name for j in jobs)

    @staticmethod
    def _gang(jobs: list[dict]) -> int:
        return max(
            (int((j["metadata"].get("annotations") or {}).get("awp.local/gang-size", "0")) for j in jobs),
            default=0,
        )

    def stop(self, attempt_id: str) -> None:
        try:
            jobs = [
                j
                for j in self.client.list_jobs(self.namespace, self.selector)
                if j["metadata"]["labels"].get("awp.local/attempt") == attempt_id
            ]
        except ApiError as e:
            raise BackendError(f"list jobs: {e}") from e
        whole = bool(jobs) and len(jobs) >= self._gang(jobs)
        if whole and all(_cond(j, "Complete") for j in jobs):
            return  # ended: a stop of an ended attempt changes nothing (success must stay visible)
        if whole and any(_cond(j, "Failed") for j in jobs) and not any(_active(j) for j in jobs):
            return  # a failed gang that holds nothing any more
        if not jobs:
            try:
                pods = [
                    p
                    for p in self.client.list_pods(self.namespace, self.selector)
                    if p["metadata"]["labels"].get("awp.local/attempt") == attempt_id
                ]
            except ApiError as e:
                raise BackendError(f"list pods: {e}") from e
            if not pods:
                return  # unknown to the cluster: nothing to stop
        self.stopped.setdefault(attempt_id, self.now_ms())
        self._delete_jobs([j["metadata"]["name"] for j in jobs])

    def _delete_jobs(self, names: list[str]) -> None:
        for name in names:
            try:
                self.client.delete_job(self.namespace, name)
            except ApiError as e:
                if e.status != 404:
                    raise BackendError(f"delete job {name}: {e}") from e

    def forget(self, attempt_id: str) -> None:
        try:
            jobs = [
                j["metadata"]["name"]
                for j in self.client.list_jobs(self.namespace, self.selector)
                if j["metadata"]["labels"].get("awp.local/attempt") == attempt_id
            ]
        except ApiError as e:
            raise BackendError(f"list jobs: {e}") from e
        self._delete_jobs(jobs)
        self.stopped.pop(attempt_id, None)
        self._meta.pop(attempt_id, None)

    def observe(self) -> Snapshot:
        try:
            jobs = self.client.list_jobs(self.namespace, self.selector)
            pods = self.client.list_pods(self.namespace, self.selector)
        except ApiError as e:
            raise BackendError(f"list: {e}") from e
        now = self.now_ms()
        by_att: dict[str, tuple[list[dict], list[dict]]] = {}
        for j in jobs:
            by_att.setdefault(j["metadata"]["labels"].get("awp.local/attempt", ""), ([], []))[0].append(j)
        for p in pods:
            by_att.setdefault(p["metadata"]["labels"].get("awp.local/attempt", ""), ([], []))[1].append(p)
        out = []
        for aid in sorted(set(by_att) | set(self.stopped)):
            if not aid:
                continue
            js, ps = by_att.get(aid, ([], []))
            st = self._status(aid, js, ps, now)
            if st is not None:
                out.append(st)
        return Snapshot(now, tuple(out))

    def _work(self, aid: str, js: list[dict], started: int | None, until: int | None) -> float | None:
        """Work done: retained + (until - started) x rate x time_scale, from the Job annotations (kept in
        memory for attempts whose Jobs are gone after a stop)."""
        if js:
            ann = js[0]["metadata"].get("annotations") or {}
            try:
                meta = (
                    float(ann["awp.local/rate"]),
                    float(ann["awp.local/retained-s"]),
                    float(ann["awp.local/time-scale"]),
                    started,
                )
            except (KeyError, ValueError):
                return None
            if started is None and aid in self._meta:
                meta = (*meta[:3], self._meta[aid][3])
            self._meta[aid] = meta
        if aid not in self._meta:
            return None
        rate, retained, scale, st = self._meta[aid]
        if st is None or until is None:
            return retained
        return round(retained + max(0, until - st) / 1000.0 * rate * scale, 6)

    def _status(self, aid: str, js: list[dict], ps: list[dict], now: int) -> AttemptStatus | None:
        st = self._status_inner(aid, js, ps, now)
        if st is None:
            return None
        until = st.ended_ms if st.ended_ms is not None else now
        return AttemptStatus(
            st.attempt_id,
            st.phase,
            st.nodes,
            st.exit_code,
            st.reason,
            st.started_ms,
            st.ended_ms,
            st.rate,
            st.workers_by_node,
            self._work(aid, js, st.started_ms, until),
            st.incomplete,
        )

    def _status_inner(self, aid: str, js: list[dict], ps: list[dict], now: int) -> AttemptStatus | None:
        nodes = tuple(sorted({p["spec"]["nodeName"] for p in ps if p.get("spec", {}).get("nodeName")}))
        wbn: dict[str, int] = {}
        for p in ps:
            n = p.get("spec", {}).get("nodeName")
            if n and p["status"].get("phase") in ("Pending", "Running"):
                wbn[n] = wbn.get(n, 0) + 1
        live = [p for p in ps if p["status"].get("phase") in ("Pending", "Running")]
        running = [
            p
            for p in live
            if p["status"].get("phase") == "Running" and not p["metadata"].get("deletionTimestamp")
        ]
        started = self._max_ts([_started_at(p) for p in ps])
        workers = tuple(sorted(wbn.items()))
        gang = self._gang(js)
        whole_complete = bool(js) and len(js) >= gang and all(_cond(j, "Complete") for j in js)
        if (
            aid in self.stopped and not whole_complete
        ):  # a completion that won the race with the stop stays a success
            if not js and not ps:
                return AttemptStatus(aid, Phase.STOPPED, (), None, None, started, self.stopped[aid])
            return AttemptStatus(
                aid,
                Phase.RUNNING if running else Phase.STARTING,
                nodes,
                workers_by_node=workers,
                started_ms=started,
            )
        failed = [p for p in ps if p["status"].get("phase") == "Failed"]
        if failed or any(_cond(j, "Failed") for j in js):
            survivors = [j["metadata"]["name"] for j in js if not _cond(j, "Failed") and _active(j)]
            if survivors:  # gang semantics: one failed worker fails the gang; end the rest
                self._delete_jobs(survivors)
            if live:
                return AttemptStatus(
                    aid,
                    Phase.RUNNING if running else Phase.STARTING,
                    nodes,
                    workers_by_node=workers,
                    started_ms=started,
                )
            f = min(failed, key=_failure_rank) if failed else None
            reason, code = FailReason.EXIT, None
            if f is not None:
                r = f["status"].get("reason") or ""
                if r == "Evicted" or any(
                    c.get("type") == "DisruptionTarget" and c.get("status") == "True"
                    for c in f["status"].get("conditions") or []
                ):
                    reason = FailReason.EVICTED
                elif r in NODE_REASONS:
                    reason = FailReason.NODE_LOST
                code = _exit_code(f)
            ended = self._max_ts([_finished_at(p) for p in failed]) or now
            return AttemptStatus(aid, Phase.FAILED, nodes, code, reason, started, ended)
        expected = sum(int(j["spec"]["completions"]) for j in js)
        if whole_complete:
            ended = self._max_ts([_finished_at(p) for p in ps]) or now
            return AttemptStatus(aid, Phase.SUCCEEDED, nodes, 0, None, started, ended)
        if js and len(js) >= gang and len(running) >= expected and expected > 0:
            return AttemptStatus(aid, Phase.RUNNING, nodes, started_ms=started, workers_by_node=workers)
        if not js and not live:
            return None  # leftovers of a forgotten attempt (terminated pods being cleaned up)
        # fewer Jobs than the gang: a start that left only part of it behind; R1 starts it again (idempotent)
        return AttemptStatus(
            aid,
            Phase.STARTING,
            nodes,
            workers_by_node=workers,
            started_ms=None,
            incomplete=bool(js) and len(js) < gang,
        )

    def _max_ts(self, ts: list[float | None]) -> int | None:
        vals = [t for t in ts if t is not None]
        return self.to_platform_ms(max(vals)) if vals else None

    def next_event_ms(self) -> int | None:
        f = getattr(self.client, "next_event_ms", None)
        return f() if f is not None else None


def _failure_rank(p: dict[str, Any]) -> tuple:
    """The pod that explains a gang's failure: not deleted by the platform or the Job controller, a node or
    eviction reason first, then the earliest to finish."""
    st = p["status"]
    node_related = (st.get("reason") or "") in NODE_REASONS | {"Evicted"} or any(
        c.get("type") == "DisruptionTarget" and c.get("status") == "True" for c in st.get("conditions") or []
    )
    return (
        bool(p["metadata"].get("deletionTimestamp")),
        not node_related,
        _finished_at(p) or float("inf"),
        p["metadata"]["name"],
    )


def _cond(job: dict[str, Any], kind: str) -> bool:
    return any(
        c.get("type") == kind and c.get("status") == "True"
        for c in (job.get("status") or {}).get("conditions") or []
    )


def _active(job: dict[str, Any]) -> bool:
    return int((job.get("status") or {}).get("active") or 0) > 0


def _state(p: dict[str, Any]) -> dict[str, Any]:
    cs = p["status"].get("containerStatuses") or []
    return (cs[0].get("state") or {}) if cs else {}


def _started_at(p: dict[str, Any]) -> float | None:
    s = _state(p)
    return parse_ts_ms(
        (s.get("running") or {}).get("startedAt") or (s.get("terminated") or {}).get("startedAt")
    )


def _finished_at(p: dict[str, Any]) -> float | None:
    return parse_ts_ms((_state(p).get("terminated") or {}).get("finishedAt"))


def _exit_code(p: dict[str, Any]) -> int | None:
    t = _state(p).get("terminated") or {}
    return int(t["exitCode"]) if "exitCode" in t else None


def _cpu(v: str) -> int:
    return int(float(v[:-1]) / 1000) if v.endswith("m") else int(float(v or 0))


def _mem_mb(v: str) -> int:
    units = {"Ki": 1024 / 1e6, "Mi": 1024**2 / 1e6, "Gi": 1024**3 / 1e6, "K": 1e-3, "M": 1.0, "G": 1000.0}
    for u, f in units.items():
        if v.endswith(u):
            return int(float(v[: -len(u)]) * f)
    return int(float(v or 0) / 1e6)
