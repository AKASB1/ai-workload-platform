"""All-or-nothing validation of a decision against the view (docs/contracts.md §6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_workload_platform.models import NodeInfo
from ai_workload_platform.policy import PolicyFailure


@dataclass(frozen=True)
class StartAction:
    workload_id: str
    placement: tuple[tuple[str, int], ...]  # (node, workers) in (rack, name) order

    def placement_dicts(self) -> list[dict[str, Any]]:
        return [{"node": n, "workers": w} for n, w in self.placement]


@dataclass(frozen=True)
class PreemptAction:
    workload_id: str


def _bad(msg: str) -> PolicyFailure:
    return PolicyFailure("invalid", msg)


def validate_decision(
    decision: Any, view: dict[str, Any], nodes: list[NodeInfo], caps: dict[str, tuple[int, int]]
) -> list[StartAction | PreemptAction]:
    """Return the actions in order, or raise PolicyFailure('invalid') if any action is invalid.

    `caps` maps namespace -> (cap_gpus, allocated_gpus from the books). A `preempt` names a running (STARTING or
    RUNNING) preemptible workload once and frees its resources for the later actions of the same decision (as in
    gpu-cluster-scheduler); the controller still keeps them held until the stopped attempt has ended (R9).
    """
    if not isinstance(decision, dict):
        raise _bad("decision is not an object")
    actions = decision.get("actions")
    if not isinstance(actions, list):
        raise _bad("actions is not a list")
    pending = {p["job_id"]: p for p in view["pending"]}
    free = {n["name"]: [n["free_gpus"], n["free_cpus"], n["free_mem_mb"]] for n in view["nodes"]}
    info = {n.name: n for n in nodes}
    order = {n.name: (n.rack, n.name) for n in nodes}
    alloc = {ns: a for ns, (_cap, a) in caps.items()}
    running = {r["job_id"]: r for r in view.get("running", [])}
    started: set[str] = set()
    preempted: set[str] = set()
    out: list[StartAction | PreemptAction] = []
    for i, act in enumerate(actions):
        where = f"action {i}"
        if not isinstance(act, dict):
            raise _bad(f"{where}: not an object")
        op = act.get("op")
        if op == "preempt":
            if set(act) != {"op", "job_id"}:
                raise _bad(f"{where}: expected {{op: preempt, job_id}}")
            if not isinstance(act["job_id"], str):
                raise _bad(f"{where}: job_id must be a string")
            r = running.get(act["job_id"])
            if r is None:
                raise _bad(f"{where}: {act['job_id']!r} is not a running workload of the view")
            if not r.get("preemptible"):
                raise _bad(f"{where}: {r['job_id']} is not preemptible")
            if r["job_id"] in preempted:
                raise _bad(f"{where}: {r['job_id']} preempted twice")
            preempted.add(r["job_id"])
            for e in r["placement"]:
                f = free.get(e["node"])
                if f is not None and info.get(e["node"]) is not None and info[e["node"]].ready:
                    k = int(e["workers"])
                    f[0] += r["gpus"] * k
                    f[1] += r["cpus"] * k
                    f[2] += r["mem_mb"] * k
            if r["tenant"] in alloc:
                alloc[r["tenant"]] -= r["gpus"] * r["workers"]
            out.append(PreemptAction(r["job_id"]))
            continue
        if op != "start" or set(act) != {"op", "job_id", "placement"}:
            raise _bad(f"{where}: expected {{op: start, job_id, placement}}")
        jid = act["job_id"]
        if not isinstance(jid, str):
            raise _bad(f"{where}: job_id must be a string")
        job = pending.get(jid)
        if job is None:
            raise _bad(f"{where}: {jid!r} is not a QUEUED workload of the view")
        if jid in started:
            raise _bad(f"{where}: {jid} started twice")
        if jid in preempted:
            raise _bad(f"{where}: {jid} is preempted and started in the same decision")
        pl = act["placement"]
        if not isinstance(pl, list) or not pl:
            raise _bad(f"{where}: empty placement")
        seen: set[str] = set()
        total = 0
        entries: list[tuple[str, int]] = []
        for e in pl:
            if not isinstance(e, dict) or set(e) != {"node", "workers"}:
                raise _bad(f"{where}: placement entries are {{node, workers}}")
            node, k = e["node"], e["workers"]
            if not isinstance(node, str):
                raise _bad(f"{where}: node must be a string")
            if not isinstance(k, int) or isinstance(k, bool) or k < 1:
                raise _bad(f"{where}: workers must be an integer >= 1")
            if node not in info or node not in free:
                raise _bad(f"{where}: unknown node {node!r}")
            if node in seen:
                raise _bad(f"{where}: node {node} listed twice")
            if not info[node].ready:
                raise _bad(f"{where}: node {node} is not ready")
            seen.add(node)
            if job["gpu_class"] and info[node].gpu_class != job["gpu_class"]:
                raise _bad(
                    f"{where}: node {node} has class {info[node].gpu_class}, {jid} needs {job['gpu_class']}"
                )
            need = (job["gpus"] * k, job["cpus"] * k, job["mem_mb"] * k)
            f = free[node]
            if need[0] > f[0] or need[1] > f[1] or need[2] > f[2]:
                raise _bad(f"{where}: {jid} does not fit on {node} (needs {need}, free {tuple(f)})")
            f[0] -= need[0]
            f[1] -= need[1]
            f[2] -= need[2]
            total += k
            entries.append((node, k))
        if total != job["workers"]:
            raise _bad(f"{where}: {total} workers placed, {jid} needs {job['workers']}")
        ns = job["tenant"]
        cap = caps.get(ns, (10**9, 0))[0]
        g = job["gpus"] * job["workers"]
        if alloc.get(ns, 0) + g > cap:
            raise _bad(f"{where}: {jid} would take namespace {ns} above its cap {cap}")
        alloc[ns] = alloc.get(ns, 0) + g
        started.add(jid)
        out.append(StartAction(jid, tuple(sorted(entries, key=lambda x: order[x[0]]))))
    return out
