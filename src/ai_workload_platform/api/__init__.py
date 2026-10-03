"""HTTP API v1 (FastAPI). Binds to 127.0.0.1 by default; no authentication in Tier 1.

Every error has the body {"error": {"code", "message", "details"}}; the framework's own error bodies
are replaced (malformed query or body: 422 INVALID_REQUEST; store outage: 503 STORE_UNAVAILABLE).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Body, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from ai_workload_platform import __version__, admission
from ai_workload_platform.clock import ms_to_s
from ai_workload_platform.models import (
    Attempt,
    Namespace,
    PlatformError,
    StoreUnavailable,
    Workload,
    WorkloadState,
)
from ai_workload_platform.models.spec import is_dns_label
from ai_workload_platform.store import ops

ERROR_SCHEMA = {
    "type": "object",
    "properties": {
        "error": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "message": {"type": "string"},
                "details": {"type": "object"},
            },
            "required": ["code", "message", "details"],
        }
    },
    "required": ["error"],
}


def _err(status: int, desc: str) -> dict[int | str, dict[str, Any]]:
    return {status: {"description": desc, "content": {"application/json": {"schema": ERROR_SCHEMA}}}}


class NamespaceBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    quota_gpus: int = Field(ge=0)
    cap_gpus: int = Field(ge=0)
    max_priority: int = Field(9, ge=0, le=9)
    max_queued: int = Field(1000, ge=1)


def _s(ms: int | None) -> float | None:
    return None if ms is None else ms_to_s(ms)


def attempt_json(a: Attempt) -> dict[str, Any]:
    return {
        "id": a.id,
        "n": a.n,
        "state": a.state.value,
        "placement": a.placement,
        "observed_nodes": a.observed_nodes,
        "stop_reason": a.stop_reason,
        "end_reason": a.end_reason,
        "exit_code": a.exit_code,
        "counted": a.counted,
        "started_s": _s(a.started_ms),
        "running_s": _s(a.running_ms),
        "ended_s": _s(a.ended_ms),
        "observed_started_s": _s(a.observed_started_ms),
        "observed_ended_s": _s(a.observed_ended_ms),
        "version": a.version,
    }


def workload_json(w: Workload, attempts: list[Attempt] | None = None) -> dict[str, Any]:
    out = {
        "namespace": w.namespace,
        "id": w.id,
        "state": w.state.value,
        "version": w.version,
        "cancel_requested": w.cancel_requested,
        "priority": w.priority,
        "spec": w.spec,
        "submit_s": _s(w.submit_ms),
        "first_started_s": _s(w.first_started_ms),
        "terminal_s": _s(w.terminal_ms),
        "retry_at_s": _s(w.retry_at_ms),
        "counted_attempts": w.counted,
        "attempt_count": w.attempts,
    }
    if attempts is not None:
        out["attempts"] = [attempt_json(a) for a in attempts]
    return out


def create_app(platform: Any = None, *, run_driver: bool = True) -> FastAPI:
    """`platform` is an api.service.Platform (None only to generate the OpenAPI document)."""

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task = None
        if platform is not None and run_driver:
            task = asyncio.create_task(platform.driver.run())
        try:
            yield
        finally:
            if task is not None:
                await platform.driver.stop()
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    app = FastAPI(
        title="ai-workload-platform",
        version=__version__,
        lifespan=lifespan,
        description="Control plane for AI workloads on shared, simulated GPU compute. No authentication.",
    )
    p = platform

    def now() -> int:
        return p.clock.now_ms()

    def kick() -> None:
        if p is not None and run_driver:
            p.driver.kick()

    @app.exception_handler(PlatformError)
    async def platform_error(_req: Request, e: PlatformError) -> JSONResponse:
        return JSONResponse(e.body(), status_code=e.status, headers=e.headers or None)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_req: Request, e: RequestValidationError) -> JSONResponse:
        fields = [".".join(str(x) for x in err.get("loc", ())) for err in e.errors()]
        body = PlatformError(
            "malformed request", {"fields": fields}, code="INVALID_REQUEST", status=422
        ).body()
        return JSONResponse(body, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_req: Request, e: StarletteHTTPException) -> JSONResponse:
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(e.status_code, "HTTP_ERROR")
        return JSONResponse(PlatformError(str(e.detail), {}, code=code).body(), status_code=e.status_code)

    def _ns_name(ns: str) -> None:
        if not is_dns_label(ns):
            raise PlatformError(
                "namespace names are DNS-1123 labels of at most 40 characters",
                {"namespace": ns},
                code="INVALID_REQUEST",
                status=422,
            )

    def _get_ns(ns: str) -> Namespace:
        n = p.store.get_namespace(ns)
        if n is None:
            raise PlatformError(
                f"unknown namespace {ns}", {"namespace": ns}, code="UNKNOWN_NAMESPACE", status=404
            )
        return n

    @app.put("/v1/namespaces/{ns}", responses={**_err(422, "Invalid body or name")})
    def put_namespace(ns: str, body: NamespaceBody) -> dict[str, Any]:
        """Create or update a namespace; a change affects later admissions and cycles only."""
        _ns_name(ns)
        if body.cap_gpus < body.quota_gpus:
            raise PlatformError(
                "cap_gpus must be at least quota_gpus",
                {"fields": ["cap_gpus"]},
                code="INVALID_REQUEST",
                status=422,
            )
        n = Namespace(ns, body.quota_gpus, body.cap_gpus, body.max_priority, body.max_queued)
        p.put_namespace(n)
        kick()
        return n.to_json_obj()

    @app.get("/v1/namespaces")
    def list_namespaces() -> dict[str, Any]:
        return {"items": [n.to_json_obj() for n in p.store.namespaces()]}

    @app.get("/v1/namespaces/{ns}", responses={**_err(404, "Unknown namespace")})
    def get_namespace(ns: str) -> dict[str, Any]:
        return _get_ns(ns).to_json_obj()

    @app.post(
        "/v1/namespaces/{ns}/workloads",
        status_code=201,
        responses={
            200: {"description": "Idempotent repeat: the original workload"},
            **_err(404, "UNKNOWN_NAMESPACE"),
            **_err(409, "IDEMPOTENCY_MISMATCH or DUPLICATE_ID"),
            **_err(422, "INVALID_SPEC, UNSCHEDULABLE, EXCEEDS_NAMESPACE_CAP, or INVALID_REQUEST"),
            **_err(403, "PRIORITY_NOT_ALLOWED"),
            **_err(429, "QUEUE_FULL (Retry-After)"),
            **_err(503, "NO_INVENTORY"),
        },
    )
    def submit(
        ns: str,
        spec: Any = Body(..., description="Workload specification (docs/contracts.md §2)"),
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ) -> Response:
        w, created = admission.submit(p.store, ns, spec, idempotency_key, now(), metrics=p.metrics)
        kick()
        return JSONResponse(workload_json(w, p.store.attempts_of(w.id)), status_code=201 if created else 200)

    @app.get("/v1/namespaces/{ns}/workloads/{wid}", responses={**_err(404, "UNKNOWN_WORKLOAD")})
    def get_workload(ns: str, wid: str) -> dict[str, Any]:
        w = p.store.get_workload(wid)
        if w is None or w.namespace != ns:
            raise PlatformError(
                f"unknown workload {wid}", {"namespace": ns, "id": wid}, code="UNKNOWN_WORKLOAD", status=404
            )
        return workload_json(w, p.store.attempts_of(wid))

    @app.get("/v1/namespaces/{ns}/workloads", responses={**_err(404, "UNKNOWN_NAMESPACE")})
    def list_workloads(
        ns: str,
        state: WorkloadState | None = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        after: int = Query(0, ge=0, description="cursor: the `next` of the previous page"),
    ) -> dict[str, Any]:
        _get_ns(ns)
        ws = p.store.list_workloads(ns, state.value if state else None, after, limit)
        nxt = ws[-1].submit_seq if len(ws) == limit else None
        return {"items": [workload_json(w) for w in ws], "next": nxt}

    @app.post(
        "/v1/namespaces/{ns}/workloads/{wid}:cancel",
        status_code=202,
        responses={
            200: {"description": "The workload is terminal: nothing changed"},
            **_err(404, "UNKNOWN_WORKLOAD"),
        },
    )
    def cancel(ns: str, wid: str) -> Response:
        w, _changed = ops.request_cancel(p.store, ns, wid, now())
        kick()
        return JSONResponse(
            workload_json(w),
            status_code=200 if w.terminal and not (w.state == WorkloadState.CANCELLED and _changed) else 202,
        )

    @app.get("/v1/events")
    def events(after: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
        evs = p.store.events(after, limit)
        return {"items": [e.to_json_obj() for e in evs], "next": evs[-1].seq if evs else after}

    @app.get("/v1/cluster/nodes")
    def nodes() -> dict[str, Any]:
        inv = p.store.inventory()
        used: dict[str, list[int]] = {}
        for (_ns, node), (g, c, m) in p.store.books().items():
            u = used.setdefault(node, [0, 0, 0])
            u[0] += g
            u[1] += c
            u[2] += m
        out = []
        for n in inv.nodes:
            u = used.get(n.name, [0, 0, 0])
            out.append(
                {
                    **n.to_json_obj(),
                    "free_gpus": n.gpus - u[0] if n.ready else 0,
                    "free_cpus": n.cpus - u[1] if n.ready else 0,
                    "free_mem_mb": n.mem_mb - u[2] if n.ready else 0,
                    "not_ready_since_s": _s(inv.not_ready_since.get(n.name)),
                }
            )
        return {"items": out}

    @app.get("/v1/namespaces/{ns}/usage", responses={**_err(404, "UNKNOWN_NAMESPACE")})
    def usage(ns: str) -> dict[str, Any]:
        n = _get_ns(ns)
        alloc = sum(g for (nsn, _node), (g, _c, _m) in p.store.books().items() if nsn == ns)
        queued = len(p.store.list_workloads(ns, "QUEUED")) + len(p.store.list_workloads(ns, "RETRY_WAIT"))
        return {
            "namespace": ns,
            "allocated_gpus": alloc,
            "quota_gpus": n.quota_gpus,
            "cap_gpus": n.cap_gpus,
            "queued": queued,
        }

    @app.get("/healthz", responses={**_err(503, "STORE_UNAVAILABLE")})
    def healthz() -> dict[str, Any]:
        status, reasons = p.health()
        return {"status": status, "reasons": reasons}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> Response:
        from prometheus_client import CONTENT_TYPE_LATEST

        return Response(p.metrics.exposition(), media_type=CONTENT_TYPE_LATEST)

    @app.middleware("http")
    async def store_errors(request: Request, call_next: Any) -> Response:
        try:
            return await call_next(request)
        except StoreUnavailable as e:
            return JSONResponse(PlatformError(str(e), {}, code="STORE_UNAVAILABLE").body(), status_code=503)

    return app


def openapi_json() -> str:
    """docs/openapi.json, generated from the application."""
    return json.dumps(create_app(None).openapi(), indent=2, sort_keys=True) + "\n"
