"""A small Python client for the HTTP API v1.

`submit` generates one Idempotency-Key per logical submission and reuses it on every retry, so a
retried request after a lost reply or a restarted service gives one workload.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from typing import Any

import httpx


class ClientError(Exception):
    def __init__(self, status: int, body: Any) -> None:
        err = body.get("error", {}) if isinstance(body, dict) else {}
        self.status = status
        self.code = err.get("code", "HTTP_ERROR")
        self.message = err.get("message", str(body))
        self.details = err.get("details", {})
        super().__init__(f"{status} {self.code}: {self.message}")


class Client:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:18400",
        *,
        timeout_s: float = 10.0,
        retries: int = 5,
        backoff_s: float = 0.2,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport, trust_env=False)
        self.retries = retries
        self.backoff_s = backoff_s

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *a: object) -> None:
        self.close()

    def _req(self, method: str, path: str, *, retry: bool = True, **kw: Any) -> httpx.Response:
        attempt = 0
        while True:
            try:
                r = self.http.request(method, path, **kw)
            except httpx.TransportError:
                if not retry or attempt >= self.retries:
                    raise
            else:
                if r.status_code < 500 or not retry or attempt >= self.retries:
                    return r
            attempt += 1
            time.sleep(self.backoff_s * attempt)

    @staticmethod
    def _json(r: httpx.Response) -> Any:
        body = r.json() if r.content else None
        if r.status_code >= 400:
            raise ClientError(r.status_code, body)
        return body

    def put_namespace(
        self, ns: str, quota_gpus: int, cap_gpus: int, max_priority: int = 9, max_queued: int = 1000
    ) -> dict[str, Any]:
        return self._json(
            self._req(
                "PUT",
                f"/v1/namespaces/{ns}",
                json={
                    "quota_gpus": quota_gpus,
                    "cap_gpus": cap_gpus,
                    "max_priority": max_priority,
                    "max_queued": max_queued,
                },
            )
        )

    def namespaces(self) -> list[dict[str, Any]]:
        return self._json(self._req("GET", "/v1/namespaces"))["items"]

    def submit(
        self, ns: str, spec: dict[str, Any], idempotency_key: str | None = None
    ) -> tuple[int, dict[str, Any]]:
        """Returns (HTTP status, workload). The key is reused on every retry of this call."""
        key = idempotency_key or uuid.uuid4().hex
        r = self._req("POST", f"/v1/namespaces/{ns}/workloads", json=spec, headers={"Idempotency-Key": key})
        return r.status_code, self._json(r)

    def get(self, ns: str, wid: str) -> dict[str, Any]:
        return self._json(self._req("GET", f"/v1/namespaces/{ns}/workloads/{wid}"))

    def list(self, ns: str, state: str | None = None, limit: int = 100) -> Iterator[dict[str, Any]]:
        after = 0
        while True:
            params: dict[str, Any] = {"limit": limit, "after": after}
            if state:
                params["state"] = state
            page = self._json(self._req("GET", f"/v1/namespaces/{ns}/workloads", params=params))
            yield from page["items"]
            if page["next"] is None:
                return
            after = page["next"]

    def cancel(self, ns: str, wid: str) -> tuple[int, dict[str, Any]]:
        r = self._req("POST", f"/v1/namespaces/{ns}/workloads/{wid}:cancel")
        return r.status_code, self._json(r)

    def events(self, after: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        while True:
            page = self._json(self._req("GET", "/v1/events", params={"after": after, "limit": limit}))
            out.extend(page["items"])
            if len(page["items"]) < limit:
                return out
            after = page["next"]

    def nodes(self) -> list[dict[str, Any]]:
        return self._json(self._req("GET", "/v1/cluster/nodes"))["items"]

    def usage(self, ns: str) -> dict[str, Any]:
        return self._json(self._req("GET", f"/v1/namespaces/{ns}/usage"))

    def healthz(self) -> tuple[int, dict[str, Any]]:
        r = self._req("GET", "/healthz", retry=False)
        return r.status_code, r.json()

    def metrics(self) -> str:
        r = self._req("GET", "/metrics")
        r.raise_for_status()
        return r.text
