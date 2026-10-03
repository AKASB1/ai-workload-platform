"""Check 10: API contract. OpenAPI drift, the success status of every endpoint, the status and body of
every error code, pagination, idempotent submission over HTTP, /metrics names, and a real-socket
smoke test on port 0 through the client."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from ai_workload_platform.api import create_app, openapi_json
from ai_workload_platform.api.service import Platform, ServiceConfig
from ai_workload_platform.client import Client, ClientError
from ai_workload_platform.controller.rules import Thresholds
from ai_workload_platform.models import StoreUnavailable
from ai_workload_platform.observability import METRIC_NAMES

ROOT = Path(__file__).resolve().parents[1]
CLUSTER = str(ROOT / "configs" / "clusters" / "reference.json")
SPEC = {"gpus": 2, "sim": {"runtime_s": 30}, "estimate_s": 60}


@pytest.fixture
def plat(tmp_path):
    p = Platform(ServiceConfig(db=str(tmp_path / "api.db"), cluster=CLUSTER, scale=1.0))
    p.make_controller().tick(p.clock.now_ms())  # store an inventory (no live driver in these tests)
    yield p
    p.close()


@pytest.fixture
def api(plat):
    with TestClient(create_app(plat, run_driver=False)) as c:
        yield c


def err(r, status: int, code: str) -> dict:
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == {"error"} and set(body["error"]) == {"code", "message", "details"}
    assert body["error"]["code"] == code
    return body["error"]


def test_openapi_file_has_no_drift() -> None:
    assert (ROOT / "docs" / "openapi.json").read_text(encoding="utf-8") == openapi_json()


def test_namespaces_endpoints(api) -> None:
    r = api.put("/v1/namespaces/team-x", json={"quota_gpus": 4, "cap_gpus": 8})
    assert r.status_code == 200 and r.json()["max_queued"] == 1000
    assert api.get("/v1/namespaces/team-x").status_code == 200
    names = [n["name"] for n in api.get("/v1/namespaces").json()["items"]]
    assert names == ["team-a", "team-b", "team-c", "team-x"]
    err(api.get("/v1/namespaces/nope"), 404, "UNKNOWN_NAMESPACE")
    err(api.put("/v1/namespaces/team-x", json={"quota_gpus": 9, "cap_gpus": 8}), 422, "INVALID_REQUEST")
    err(api.put("/v1/namespaces/Bad_Name", json={"quota_gpus": 1, "cap_gpus": 1}), 422, "INVALID_REQUEST")
    err(api.put("/v1/namespaces/team-x", json={"quota_gpus": "x", "cap_gpus": 1}), 422, "INVALID_REQUEST")
    u = api.get("/v1/namespaces/team-x/usage").json()
    assert u == {"namespace": "team-x", "allocated_gpus": 0, "quota_gpus": 4, "cap_gpus": 8, "queued": 0}


def test_submission_statuses_and_every_admission_code(api, plat) -> None:
    r = api.post(
        "/v1/namespaces/team-a/workloads", json={**SPEC, "id": "w-one"}, headers={"Idempotency-Key": "k1"}
    )
    assert r.status_code == 201 and r.json()["state"] == "QUEUED" and r.json()["attempts"] == []
    r2 = api.post(
        "/v1/namespaces/team-a/workloads", json={**SPEC, "id": "w-one"}, headers={"Idempotency-Key": "k1"}
    )
    assert r2.status_code == 200 and r2.json()["id"] == "w-one"
    err(
        api.post(
            "/v1/namespaces/team-a/workloads", json={**SPEC, "id": "w-two"}, headers={"Idempotency-Key": "k1"}
        ),
        409,
        "IDEMPOTENCY_MISMATCH",
    )
    err(api.post("/v1/namespaces/nope/workloads", json=SPEC), 404, "UNKNOWN_NAMESPACE")
    e = err(api.post("/v1/namespaces/team-a/workloads", json={"gpus": 0}), 422, "INVALID_SPEC")
    assert set(e["details"]["fields"]) == {"gpus", "sim"}
    api.put("/v1/namespaces/low", json={"quota_gpus": 2, "cap_gpus": 4, "max_priority": 1, "max_queued": 1})
    err(api.post("/v1/namespaces/low/workloads", json=SPEC), 403, "PRIORITY_NOT_ALLOWED")
    err(
        api.post("/v1/namespaces/low/workloads", json={**SPEC, "priority": 1, "gpus": 9}),
        422,
        "UNSCHEDULABLE",
    )
    err(
        api.post("/v1/namespaces/low/workloads", json={**SPEC, "priority": 1, "gpus": 8}),
        422,
        "EXCEEDS_NAMESPACE_CAP",
    )
    err(api.post("/v1/namespaces/team-b/workloads", json={**SPEC, "id": "w-one"}), 409, "DUPLICATE_ID")
    assert api.post("/v1/namespaces/low/workloads", json={**SPEC, "priority": 1}).status_code == 201
    r = api.post("/v1/namespaces/low/workloads", json={**SPEC, "priority": 1})
    err(r, 429, "QUEUE_FULL")
    assert r.headers["Retry-After"] == "5"
    err(
        api.post(
            "/v1/namespaces/team-a/workloads",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        ),
        422,
        "INVALID_REQUEST",
    )
    err(
        api.post("/v1/namespaces/team-a/workloads", json=SPEC, headers={"Idempotency-Key": "x" * 65}),
        422,
        "INVALID_REQUEST",
    )


def test_no_inventory_503(tmp_path) -> None:
    p = Platform(ServiceConfig(db=str(tmp_path / "noinv.db"), cluster=CLUSTER))
    with TestClient(create_app(p, run_driver=False)) as c:
        err(c.post("/v1/namespaces/team-a/workloads", json=SPEC), 503, "NO_INVENTORY")
    p.close()


def test_get_cancel_and_unknown_workload(api) -> None:
    api.post("/v1/namespaces/team-a/workloads", json={**SPEC, "id": "c1"})
    assert api.get("/v1/namespaces/team-a/workloads/c1").json()["version"] == 1
    err(api.get("/v1/namespaces/team-a/workloads/zz"), 404, "UNKNOWN_WORKLOAD")
    err(api.get("/v1/namespaces/team-b/workloads/c1"), 404, "UNKNOWN_WORKLOAD")
    r = api.post("/v1/namespaces/team-a/workloads/c1:cancel")
    assert r.status_code == 202 and r.json()["state"] == "CANCELLED"
    r = api.post("/v1/namespaces/team-a/workloads/c1:cancel")
    assert r.status_code == 200 and r.json()["version"] == 2  # terminal: nothing changed
    err(api.post("/v1/namespaces/team-a/workloads/zz:cancel"), 404, "UNKNOWN_WORKLOAD")
    err(api.get("/v1/nothing-here"), 404, "NOT_FOUND")
    err(api.delete("/v1/namespaces"), 405, "METHOD_NOT_ALLOWED")


def test_pagination_events_nodes(api) -> None:
    for i in range(7):
        api.post("/v1/namespaces/team-c/workloads", json={**SPEC, "id": f"p{i}"})
    seen, after = [], 0
    while True:
        page = api.get("/v1/namespaces/team-c/workloads", params={"limit": 3, "after": after}).json()
        seen += [w["id"] for w in page["items"]]
        if page["next"] is None:
            break
        after = page["next"]
    assert seen == [f"p{i}" for i in range(7)]
    assert [
        w["id"]
        for w in api.get("/v1/namespaces/team-c/workloads", params={"state": "QUEUED"}).json()["items"]
    ] == seen
    err(api.get("/v1/namespaces/team-c/workloads", params={"limit": 0}), 422, "INVALID_REQUEST")
    err(api.get("/v1/namespaces/team-c/workloads", params={"state": "BOGUS"}), 422, "INVALID_REQUEST")
    err(api.get("/v1/events", params={"after": "x"}), 422, "INVALID_REQUEST")
    ev = api.get("/v1/events", params={"limit": 4}).json()
    assert [e["seq"] for e in ev["items"]] == [1, 2, 3, 4] and ev["next"] == 4
    ev2 = api.get("/v1/events", params={"after": 4, "limit": 1000}).json()
    assert ev2["items"][0]["seq"] == 5 and ev2["next"] == 7
    nodes = api.get("/v1/cluster/nodes").json()["items"]
    assert [n["name"] for n in nodes] == ["r0-n00", "r0-n01", "r1-n00", "r1-n01"] and nodes[0][
        "free_gpus"
    ] == 8


def test_healthz_metrics_and_store_unavailable(api, plat) -> None:
    r = api.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] in ("ok", "degraded")
    api.post("/v1/namespaces/nope/workloads", json=SPEC)  # a rejection
    text = api.get("/metrics").text
    fams = {f.name for f in text_string_to_metric_families(text)}
    for name in METRIC_NAMES:
        assert name.removesuffix("_total") in fams, name

    class Down:
        def before_begin(self, write: bool) -> None:
            raise StoreUnavailable("injected outage")

        def before_commit(self) -> None:
            pass

    plat.store.injector = Down()
    err(api.get("/healthz"), 503, "STORE_UNAVAILABLE")
    err(api.post("/v1/namespaces/team-a/workloads", json=SPEC), 503, "STORE_UNAVAILABLE")
    plat.store.injector = None


@pytest.mark.slow
def test_real_socket_smoke_on_port_0(tmp_path) -> None:
    import uvicorn

    p = Platform(
        ServiceConfig(
            db=str(tmp_path / "live.db"),
            cluster=CLUSTER,
            scale=100.0,
            thresholds=Thresholds(observe_interval_ms=50),
        )
    )
    server = uvicorn.Server(
        uvicorn.Config(create_app(p), host="127.0.0.1", port=0, log_config=None, access_log=False)
    )
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.monotonic() + 20
    while not server.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        with Client(f"http://127.0.0.1:{port}") as c:
            while True:  # the controller stores the inventory on its first tick
                try:
                    status, w = c.submit("team-a", {**SPEC, "id": "live-1"}, "live-key")
                    break
                except ClientError as e:
                    assert e.code == "NO_INVENTORY" and time.monotonic() < deadline
                    time.sleep(0.05)
            assert status == 201
            assert c.submit("team-a", {**SPEC, "id": "live-1"}, "live-key")[0] == 200
            while c.get("team-a", "live-1")["state"] != "SUCCEEDED":
                assert time.monotonic() < deadline + 20, c.get("team-a", "live-1")
                time.sleep(0.05)
            assert c.healthz()[1]["status"] == "ok"
            assert "awp_workloads" in c.metrics()
            assert [e["type"] for e in c.events()][:1] == ["submitted"]
    finally:
        server.should_exit = True
        t.join(timeout=20)
        p.close()
    assert not t.is_alive()
