"""The policy view, built from the store and the last inventory (docs/contracts.md §6)."""

from __future__ import annotations

import math
from typing import Any

from ai_workload_platform.clock import ms_to_s
from ai_workload_platform.models import Attempt, AttemptState, Namespace, NodeInfo, Workload, WorkloadState


def user_of(w: Workload) -> str:
    return str(w.spec.get("labels", {}).get("user") or w.namespace)


def pending_key(w: Workload) -> tuple[int, int, str]:
    return (-w.priority, w.submit_ms, w.id)


def build_view(
    now_ms: int,
    nodes: list[NodeInfo],
    used_by_node: dict[str, tuple[int, int, int]],
    workloads: dict[str, Workload],
    open_attempts: list[Attempt],
    namespaces: list[Namespace],
    allocated_by_ns: dict[str, int],
    gpu_ms_by_ns: dict[str, int],
    rates: dict[str, float],
    history_new: list[dict[str, Any]],
    restart_overhead_s: float = 0.0,
) -> dict[str, Any]:
    nodes = sorted(nodes, key=lambda n: (n.rack, n.name))
    running_attempts = sorted(
        (a for a in open_attempts if a.state in (AttemptState.STARTING, AttemptState.RUNNING)),
        key=lambda a: a.id,
    )
    per_node_running: dict[str, list[dict[str, Any]]] = {n.name: [] for n in nodes}
    for a in running_attempts:
        for p in a.placement:
            if p["node"] in per_node_running:
                per_node_running[p["node"]].append({"job_id": a.workload_id, "workers": int(p["workers"])})
    vnodes = []
    for n in nodes:
        g, c, m = used_by_node.get(n.name, (0, 0, 0))
        if n.ready:
            vnodes.append(
                {
                    "name": n.name,
                    "free_gpus": max(0, n.gpus - g),
                    "free_cpus": max(0, n.cpus - c),
                    "free_mem_mb": max(0, n.mem_mb - m),
                    "running": per_node_running[n.name],
                }
            )
        else:
            vnodes.append({"name": n.name, "free_gpus": 0, "free_cpus": 0, "free_mem_mb": 0, "running": []})

    running = []
    run_by_ns: dict[str, list[int]] = {}
    for a in running_attempts:
        w = workloads.get(a.workload_id)
        if w is None:
            continue
        rate = float(rates.get(a.id) or 1.0)
        start_ms = a.observed_started_ms if a.observed_started_ms is not None else a.started_ms
        start_ms = min(start_ms, now_ms)
        est = float(w.spec["estimate_s"])
        retained = w.retained_ms / 1000.0
        overhead_ms = int(round(restart_overhead_s * 1000)) if w.preemptions > 0 else 0
        done = retained + max(0.0, (now_ms - start_ms - overhead_ms) / 1000.0 * rate)
        est_end_ms = start_ms + overhead_ms + int(math.ceil(max(0.0, est - retained) * 1000.0 / rate - 1e-6))
        if est_end_ms < now_ms:
            est_end_ms = now_ms
        running.append(
            {
                "job_id": w.id,
                "tenant": w.namespace,
                "user": user_of(w),
                "priority": w.priority,
                "gpus": w.gpus,
                "workers": w.workers,
                "cpus": w.cpus,
                "mem_mb": w.mem_mb,
                "gpu_class": w.spec.get("gpu_class") or "",
                "topology": w.spec["topology"],
                "submit_s": ms_to_s(w.submit_ms),
                "first_start_s": ms_to_s(
                    w.first_started_ms if w.first_started_ms is not None else a.started_ms
                ),
                "run_start_s": ms_to_s(start_ms),
                "overhead_s": overhead_ms / 1000.0,
                "placement": [{"node": p["node"], "workers": int(p["workers"])} for p in a.placement],
                "rate": rate,
                "estimate_s": est,
                "retained_at_start_s": retained,
                "work_done_s": round(done, 3),
                "est_remaining_work_s": round(max(0.0, est - done), 3),
                "est_end_s": ms_to_s(est_end_ms),
                "preemptible": bool(w.spec["preemptible"]),
                "checkpoint_interval_s": w.spec["checkpoint_interval_s"],
                "preemptions": w.preemptions,
            }
        )
        acc = run_by_ns.setdefault(w.namespace, [0, 0, 0])
        acc[0] += a.gpus * a.workers
        acc[1] += a.cpus * a.workers
        acc[2] += a.mem_mb * a.workers

    pending = []
    for w in sorted((w for w in workloads.values() if w.state == WorkloadState.QUEUED), key=pending_key):
        pending.append(
            {
                "job_id": w.id,
                "tenant": w.namespace,
                "user": user_of(w),
                "submit_s": ms_to_s(w.submit_ms),
                "priority": w.priority,
                "gpus": w.gpus,
                "workers": w.workers,
                "cpus": w.cpus,
                "mem_mb": w.mem_mb,
                "gpu_class": w.spec.get("gpu_class") or "",
                "topology": w.spec["topology"],
                "estimate_s": w.spec["estimate_s"],
                "wait_s": ms_to_s(max(0, now_ms - w.submit_ms)),
                "retained_s": w.retained_ms / 1000.0,
                "max_wait_s": w.spec.get("max_wait_s"),
                "preemptible": bool(w.spec["preemptible"]),
                "checkpoint_interval_s": w.spec["checkpoint_interval_s"],
                "preemptions": w.preemptions,
                "started": w.attempts > 0,
            }
        )

    names = sorted({ns.name for ns in namespaces} | set(gpu_ms_by_ns) | set(run_by_ns))
    tenants = []
    for t in names:
        r = run_by_ns.get(t, [0, 0, 0])
        tenants.append(
            {
                "tenant": t,
                "gpu_seconds": round(gpu_ms_by_ns.get(t, 0) / 1000.0, 3),
                "running_gpus": r[0],
                "running_cpus": r[1],
                "running_mem_mb": r[2],
            }
        )
    return {
        "now_s": ms_to_s(now_ms),
        "nodes": vnodes,
        "running": running,
        "pending": pending,
        "tenants": tenants,
        "history_new": history_new,
        # in-process extension: capacity held by attempts that are being stopped for a preemption (soon free)
        "awp_stopping": _stopping(open_attempts),
        "awp_namespaces": [
            {
                "tenant": ns.name,
                "quota_gpus": ns.quota_gpus,
                "cap_gpus": ns.cap_gpus,
                "allocated_gpus": int(allocated_by_ns.get(ns.name, 0)),
            }
            for ns in sorted(namespaces, key=lambda x: x.name)
        ],
    }


def _stopping(open_attempts: list[Attempt]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for a in sorted(open_attempts, key=lambda x: x.id):
        if a.state == AttemptState.STOPPING and a.stop_reason == "preempted":
            for p in a.placement:
                v = out.setdefault(p["node"], [0, 0, 0])
                k = int(p["workers"])
                v[0] += a.gpus * k
                v[1] += a.cpus * k
                v[2] += a.mem_mb * k
    return out


def wire_view(view: dict[str, Any]) -> dict[str, Any]:
    """The view as sent over the protocol: in-process `awp_*` extensions removed."""
    return {k: v for k, v in view.items() if not k.startswith("awp_")}
