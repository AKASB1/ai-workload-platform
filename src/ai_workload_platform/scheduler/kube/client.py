"""`KubeClient`: the only Kubernetes calls the backend makes (list, create, and delete for Jobs and pods,
list for nodes), on plain JSON dictionaries in the API's own shape. Two implementations: the official
`kubernetes` Python client (`RealKubeClient`) and the deterministic fake (`fake.FakeKubeClient`).
"""

from __future__ import annotations

import calendar
import json
import re
import time
from typing import Any, Protocol


class ApiError(Exception):
    def __init__(self, status: int, reason: str, message: str = "") -> None:
        super().__init__(f"{status} {reason}: {message}")
        self.status = status
        self.reason = reason


class KubeClient(Protocol):
    def list_nodes(self) -> list[dict[str, Any]]: ...

    def list_jobs(self, namespace: str, label_selector: str) -> list[dict[str, Any]]: ...

    def list_pods(self, namespace: str, label_selector: str) -> list[dict[str, Any]]: ...

    def create_job(self, namespace: str, body: dict[str, Any]) -> dict[str, Any]: ...

    def delete_job(self, namespace: str, name: str) -> None:
        """Delete with propagationPolicy=Background; a missing Job raises ApiError(404)."""


_TS = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(\.\d+)?Z$")


def parse_ts_ms(ts: str | None) -> float | None:
    """RFC 3339 UTC timestamp (as Kubernetes writes it) -> epoch milliseconds."""
    if not ts:
        return None
    m = _TS.match(ts)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(m.group(i)) for i in range(1, 7))
    frac = float(m.group(7)) if m.group(7) else 0.0
    return (calendar.timegm((y, mo, d, h, mi, s, 0, 0, 0)) + frac) * 1000.0


def format_ts(epoch_ms: int) -> str:
    sec, ms = divmod(int(epoch_ms), 1000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(sec)) + f".{ms:03d}Z"


class RealKubeClient:
    """The official client; responses are read as raw JSON (no model deserialization)."""

    def __init__(
        self, kubeconfig: str | None = None, *, in_cluster: bool = False, timeout_s: float = 10.0
    ) -> None:
        from kubernetes import client, config

        if in_cluster:
            config.load_incluster_config()
        else:
            config.load_kube_config(config_file=kubeconfig)
        self._client = client
        self.core = client.CoreV1Api()
        self.batch = client.BatchV1Api()
        self.timeout_s = timeout_s

    def _call(self, fn: Any, *args: Any, **kw: Any) -> Any:
        try:
            r = fn(*args, _preload_content=False, _request_timeout=self.timeout_s, **kw)
            data = r.data
            return json.loads(data) if data else {}
        except self._client.ApiException as e:
            reason = ""
            try:
                reason = json.loads(e.body).get("reason", "")
            except Exception:  # noqa: BLE001
                pass
            raise ApiError(int(e.status or 0), reason or str(e.reason), str(e.body)[:300]) from None
        except Exception as e:  # noqa: BLE001 - connection errors
            raise ApiError(0, "Unreachable", str(e)[:300]) from None

    def list_nodes(self) -> list[dict[str, Any]]:
        return self._call(self.core.list_node)["items"]

    def list_jobs(self, namespace: str, label_selector: str) -> list[dict[str, Any]]:
        return self._call(self.batch.list_namespaced_job, namespace, label_selector=label_selector)["items"]

    def list_pods(self, namespace: str, label_selector: str) -> list[dict[str, Any]]:
        return self._call(self.core.list_namespaced_pod, namespace, label_selector=label_selector)["items"]

    def create_job(self, namespace: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._call(self.batch.create_namespaced_job, namespace, body)

    def delete_job(self, namespace: str, name: str) -> None:
        self._call(self.batch.delete_namespaced_job, name, namespace, propagation_policy="Background")
