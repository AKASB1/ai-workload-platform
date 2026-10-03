"""Admission: one transaction together with the insert; the first failing check is the answer.

Order: unknown namespace (404) -> idempotency (200 repeat / 409 mismatch) -> invalid spec (422)
-> priority above max (403) -> no inventory (503) / unschedulable (422) -> above the namespace
cap (422) -> duplicate id (409) -> queue full (429 with Retry-After).
"""

from __future__ import annotations

import logging
from typing import Any

from ai_workload_platform.crash import crashpoint
from ai_workload_platform.models import NodeInfo, PlatformError, Workload
from ai_workload_platform.models.spec import mem_mb, spec_hash, validate_spec
from ai_workload_platform.store import ops
from ai_workload_platform.store.sql import Store

log = logging.getLogger("awp.admission")

REASONS = (
    "UNKNOWN_NAMESPACE",
    "IDEMPOTENCY_MISMATCH",
    "INVALID_SPEC",
    "PRIORITY_NOT_ALLOWED",
    "NO_INVENTORY",
    "UNSCHEDULABLE",
    "EXCEEDS_NAMESPACE_CAP",
    "DUPLICATE_ID",
    "QUEUE_FULL",
)


def max_workers_on(node: NodeInfo, gpus: int, cpus: int, mem: int) -> int:
    """How many identical workers fit on an empty node."""
    k = node.gpus // gpus
    if cpus > 0:
        k = min(k, node.cpus // cpus)
    if mem > 0:
        k = min(k, node.mem_mb // mem)
    return max(0, k)


def unschedulable_reason(spec: dict[str, Any], nodes: list[NodeInfo]) -> str | None:
    """Why a request can never run on the inventory (None when it can)."""
    g, c, m = int(spec["gpus"]), int(spec["cpus"]), mem_mb(spec["mem_gb"])
    cls = spec.get("gpu_class")
    eligible = [n for n in nodes if cls is None or n.gpu_class == cls]
    if not eligible:
        return "class"
    if not any(n.gpus >= g for n in eligible):
        return "gpus"
    if not any(n.gpus >= g and n.cpus >= c for n in eligible):
        return "cpus"
    if not any(n.gpus >= g and n.cpus >= c and n.mem_mb >= m for n in eligible):
        return "memory"
    if sum(max_workers_on(n, g, c, m) for n in eligible) < int(spec["workers"]):
        return "workers"
    return None


def _reject(metrics: Any, namespace: str, err: PlatformError) -> PlatformError:
    if metrics is not None:
        metrics.admission_rejections.labels(namespace=namespace, reason=err.code).inc()
    log.info(
        "admission rejected",
        extra={"fields": {"namespace": namespace, "code": err.code, "message": err.message}},
    )
    return err


def submit(
    store: Store,
    namespace: str,
    raw_spec: Any,
    idempotency_key: str | None,
    now_ms: int,
    *,
    metrics: Any = None,
    retry_after_s: int = 5,
) -> tuple[Workload, bool]:
    """Admit a submission. Returns (workload, created); created is False for an idempotent repeat."""
    if idempotency_key is not None and not 1 <= len(idempotency_key) <= 64:
        raise _reject(
            metrics,
            namespace,
            PlatformError(
                "Idempotency-Key must have 1 to 64 characters",
                {"header": "Idempotency-Key"},
                code="INVALID_REQUEST",
                status=422,
            ),
        )
    canon: dict[str, Any] | None = None
    spec_error: PlatformError | None = None
    try:
        canon = validate_spec(raw_spec)
    except PlatformError as e:
        spec_error = e
    h = spec_hash(canon) if canon is not None else None

    with store.transaction() as tx:
        ns = tx.namespace(namespace)
        if ns is None:
            raise _reject(
                metrics,
                namespace,
                PlatformError(
                    f"unknown namespace {namespace}",
                    {"namespace": namespace},
                    code="UNKNOWN_NAMESPACE",
                    status=404,
                ),
            )
        if idempotency_key is not None:
            prior = tx.idempotency(namespace, idempotency_key)
            if prior is not None:
                if h is not None and prior[0] == h:
                    w = tx.workload(prior[1])
                    assert w is not None
                    return w, False
                raise _reject(
                    metrics,
                    namespace,
                    PlatformError(
                        "Idempotency-Key was used with a different specification",
                        {"idempotency_key": idempotency_key, "workload_id": prior[1]},
                        code="IDEMPOTENCY_MISMATCH",
                        status=409,
                    ),
                )
        if spec_error is not None:
            raise _reject(metrics, namespace, spec_error)
        assert canon is not None and h is not None
        if int(canon["priority"]) > ns.max_priority:
            raise _reject(
                metrics,
                namespace,
                PlatformError(
                    f"priority {canon['priority']} is above the namespace maximum {ns.max_priority}",
                    {"priority": canon["priority"], "max_priority": ns.max_priority},
                    code="PRIORITY_NOT_ALLOWED",
                    status=403,
                ),
            )
        inv = tx.inventory()
        if not inv.nodes:
            raise _reject(
                metrics,
                namespace,
                PlatformError("no inventory has been stored yet", {}, code="NO_INVENTORY", status=503),
            )
        why = unschedulable_reason(canon, inv.nodes)
        if why is not None:
            raise _reject(
                metrics,
                namespace,
                PlatformError(
                    f"the request can never run on this cluster ({why})",
                    {"constraint": why},
                    code="UNSCHEDULABLE",
                    status=422,
                ),
            )
        total = int(canon["gpus"]) * int(canon["workers"])
        if total > ns.cap_gpus:
            raise _reject(
                metrics,
                namespace,
                PlatformError(
                    f"{total} GPUs exceed the namespace cap {ns.cap_gpus}",
                    {"gpus": total, "cap_gpus": ns.cap_gpus},
                    code="EXCEEDS_NAMESPACE_CAP",
                    status=422,
                ),
            )
        if canon.get("id") is not None and tx.exists(canon["id"]):
            raise _reject(
                metrics,
                namespace,
                PlatformError(
                    f"workload {canon['id']} exists", {"id": canon["id"]}, code="DUPLICATE_ID", status=409
                ),
            )
        if tx.count_waiting(namespace) >= ns.max_queued:
            raise _reject(
                metrics,
                namespace,
                PlatformError(
                    f"namespace {namespace} has {ns.max_queued} queued workloads",
                    {"max_queued": ns.max_queued},
                    code="QUEUE_FULL",
                    status=429,
                    headers={"Retry-After": str(int(retry_after_s))},
                ),
            )
        w, _ev = ops.insert_submitted(tx, namespace, canon, h, idempotency_key, now_ms)
    crashpoint("submit.after_commit")
    return w, True
