"""Check 3: admission reason codes, their order, Retry-After, idempotency, and max_queued under concurrency."""

from __future__ import annotations

import threading

import pytest

from ai_workload_platform import admission
from ai_workload_platform.models import Namespace, NodeInfo, PlatformError
from ai_workload_platform.observability import Metrics
from ai_workload_platform.store.sql import Store
from tests.conftest import setup_reference

OK = {"gpus": 2, "sim": {"runtime_s": 100}}


def _err(store: Store, ns: str, spec, key=None, metrics=None) -> PlatformError:
    with pytest.raises(PlatformError) as e:
        admission.submit(store, ns, spec, key, 1, metrics=metrics)
    return e.value


@pytest.fixture
def store() -> Store:
    s = Store()
    setup_reference(s, max_queued=3)
    s.put_namespace(Namespace("low", 4, 8, 2, 3), 0)
    yield s
    s.close()


def test_unknown_namespace_404(store: Store) -> None:
    e = _err(store, "nope", OK)
    assert (e.status, e.code) == (404, "UNKNOWN_NAMESPACE")
    assert e.body() == {
        "error": {
            "code": "UNKNOWN_NAMESPACE",
            "message": "unknown namespace nope",
            "details": {"namespace": "nope"},
        }
    }


def test_invalid_spec_422_lists_every_field(store: Store) -> None:
    e = _err(
        store, "team-a", {"gpus": 0, "priority": 12, "bogus": 1, "sim": {"runtime_s": 1.2345}, "mem_gb": "x"}
    )
    assert (e.status, e.code) == (422, "INVALID_SPEC")
    assert set(e.details["fields"]) == {"gpus", "priority", "bogus", "sim.runtime_s", "mem_gb"}
    e2 = _err(store, "team-a", {"gpus": 1})
    assert e2.details["fields"] == ["sim"]
    e3 = _err(store, "team-a", ["not", "an", "object"])
    assert e3.code == "INVALID_SPEC"


def test_priority_not_allowed_403(store: Store) -> None:
    e = _err(store, "low", {**OK, "priority": 3})
    assert (e.status, e.code) == (403, "PRIORITY_NOT_ALLOWED")


def test_no_inventory_503_then_unschedulable_422() -> None:
    s = Store()
    s.put_namespace(Namespace("team-a", 8, 32, 9, 10), 0)
    e = _err(s, "team-a", OK)
    assert (e.status, e.code) == (503, "NO_INVENTORY")
    s.save_inventory(
        [
            NodeInfo("n0", "r0", "a100", 1.0, 8, 64, 512_000),
            NodeInfo("n1", "r0", "a100", 1.0, 8, 64, 512_000),
        ],
        0,
        None,
    )
    cases = {
        "class": {**OK, "gpu_class": "v100"},
        "gpus": {**OK, "gpus": 9},
        "cpus": {**OK, "cpus": 65},
        "memory": {**OK, "mem_gb": 512.001},
        "workers": {**OK, "gpus": 8, "workers": 3},
    }
    for why, spec in cases.items():
        e = _err(s, "team-a", spec)
        assert (e.status, e.code, e.details["constraint"]) == (422, "UNSCHEDULABLE", why)
    w, created = admission.submit(s, "team-a", {**OK, "gpus": 8, "workers": 2}, None, 1)
    assert created
    s.close()


def test_exceeds_cap_duplicate_and_queue_full(store: Store) -> None:
    e = _err(store, "low", {**OK, "gpus": 4, "workers": 3, "priority": 1})
    assert (e.status, e.code) == (422, "EXCEEDS_NAMESPACE_CAP")
    admission.submit(store, "team-a", {**OK, "id": "job-1"}, None, 1)
    e = _err(store, "team-a", {**OK, "id": "job-1"})
    assert (e.status, e.code) == (409, "DUPLICATE_ID")
    admission.submit(store, "team-a", OK, None, 2)
    admission.submit(store, "team-a", OK, None, 3)
    m = Metrics()
    e = _err(store, "team-a", OK, metrics=m)
    assert (e.status, e.code) == (429, "QUEUE_FULL")
    assert e.headers == {"Retry-After": "5"}
    assert m.admission_rejections.labels(namespace="team-a", reason="QUEUE_FULL")._value.get() == 1


def test_order_of_checks(store: Store) -> None:
    # every check fails at once: the namespace wins, then idempotency, then the spec, ...
    bad = {"gpus": 99, "priority": 9, "sim": {"runtime_s": -1}}
    assert _err(store, "nope", bad).code == "UNKNOWN_NAMESPACE"
    admission.submit(store, "low", {**OK, "id": "k-owner", "priority": 1}, "key-x", 1)
    assert _err(store, "low", bad, key="key-x").code == "IDEMPOTENCY_MISMATCH"
    assert _err(store, "low", bad).code == "INVALID_SPEC"
    assert (
        _err(store, "low", {"gpus": 99, "priority": 9, "sim": {"runtime_s": 1}}).code
        == "PRIORITY_NOT_ALLOWED"
    )
    assert _err(store, "low", {"gpus": 99, "priority": 1, "sim": {"runtime_s": 1}}).code == "UNSCHEDULABLE"
    assert (
        _err(store, "low", {"gpus": 8, "workers": 2, "sim": {"runtime_s": 1}, "priority": 1}).code
        == "EXCEEDS_NAMESPACE_CAP"
    )
    admission.submit(store, "low", {**OK, "id": "d1", "priority": 1}, None, 1)
    admission.submit(store, "low", {**OK, "priority": 1}, None, 1)
    assert _err(store, "low", {**OK, "id": "d1", "priority": 1}).code == "DUPLICATE_ID"
    assert _err(store, "low", {**OK, "id": "d2", "priority": 1}).code == "QUEUE_FULL"


def test_idempotent_repeat_200_and_mismatch_409(store: Store) -> None:
    w1, c1 = admission.submit(store, "team-b", OK, "abc", 1)
    w2, c2 = admission.submit(
        store, "team-b", {"sim": {"runtime_s": 100.0}, "gpus": 2}, "abc", 2
    )  # same canon
    assert c1 and not c2 and w1.id == w2.id
    e = _err(store, "team-b", {**OK, "gpus": 1}, key="abc")
    assert (e.status, e.code) == (409, "IDEMPOTENCY_MISMATCH")
    e = _err(store, "team-b", OK, key="")
    assert e.code == "INVALID_REQUEST"
    e = _err(store, "team-b", OK, key="x" * 65)
    assert e.code == "INVALID_REQUEST"


def test_max_queued_never_exceeded_under_concurrency(tmp_path) -> None:
    from ai_workload_platform.store.dialect import SQLiteDialect

    s = Store(SQLiteDialect(str(tmp_path / "q.db")))
    setup_reference(s, max_queued=25)
    accepted: list[str] = []
    full = []
    lock = threading.Lock()

    def worker(k: int) -> None:
        for i in range(10):
            try:
                w, _ = admission.submit(s, "team-c", OK, None, k * 100 + i)
                with lock:
                    accepted.append(w.id)
            except PlatformError as e:
                assert e.code == "QUEUE_FULL"
                with lock:
                    full.append(1)

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(16)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=60)
    assert len(accepted) == 25 and len(full) == 160 - 25
    assert len(s.list_workloads("team-c")) == 25
    s.close()
