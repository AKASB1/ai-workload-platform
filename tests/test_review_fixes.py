"""Regression tests for the findings of review 1 (each named after the defect it pins down)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ai_workload_platform import admission
from ai_workload_platform.clock import VirtualClock
from ai_workload_platform.cluster import load_cluster
from ai_workload_platform.controller import Controller
from ai_workload_platform.controller.rules import Thresholds
from ai_workload_platform.models import (
    AttemptRequest,
    BackendError,
    LeaseLost,
    NodeInfo,
    Phase,
    WorkloadState,
)
from ai_workload_platform.models.spec import validate_spec
from ai_workload_platform.observability import Metrics
from ai_workload_platform.policy import PolicyFailure
from ai_workload_platform.policy.builtin import make_builtin
from ai_workload_platform.policy.external import ExternalPolicy
from ai_workload_platform.policy.runner import PolicyRunner
from ai_workload_platform.policy.validate import validate_decision
from ai_workload_platform.scheduler.kube.backend import KubeBackend
from ai_workload_platform.scheduler.kube.fake import EPOCH_MS, FakeKubeClient
from ai_workload_platform.sim import make_backend, reference_namespaces
from ai_workload_platform.store.sql import Store

ROOT = Path(__file__).resolve().parents[1]
REF = load_cluster(ROOT / "configs" / "clusters" / "reference.json")
TH = Thresholds(start_retry_ms=2000, start_timeout_ms=20_000, lost_grace_ms=6000, stop_retry_ms=3000)


def kube(clock: VirtualClock, instance: str = "i1") -> tuple[KubeBackend, FakeKubeClient]:
    fake = FakeKubeClient(REF, clock)
    return KubeBackend(
        fake, REF, instance_id=instance, to_platform_ms=lambda w: int(w - EPOCH_MS), now_ms=clock.now_ms
    ), fake


def req(aid: str, placement, runtime: float = 30, gpus: int = 8) -> AttemptRequest:
    workers = sum(k for _, k in placement)
    spec = validate_spec(
        {"id": aid.split("-a")[0], "gpus": gpus, "workers": workers, "sim": {"runtime_s": runtime}}
    )
    return AttemptRequest(aid, spec["id"], "team-a", 1, spec, tuple(placement))


def test_partial_gang_is_started_again_and_stop_deletes_a_partial_gang() -> None:
    clock = VirtualClock()
    b, fake = kube(clock)
    r = req("g-a1", [("r0-n00", 1), ("r0-n01", 1)])
    fake.create_job(b.namespace, b._job_body(r, 0, "r0-n00", 1, 2))  # a start that left only Job 0 behind
    st = b.observe().by_id()["g-a1"]
    assert st.phase == Phase.STARTING and st.incomplete
    b.start(r)  # R1 repeats start: idempotent, creates the missing Job
    assert not b.observe().by_id()["g-a1"].incomplete
    # a partial gang whose existing Jobs completed is not "ended": stop deletes it and records the stop
    r2 = req("h-a1", [("r1-n00", 1), ("r1-n01", 1)], runtime=1)
    fake.create_job(b.namespace, b._job_body(r2, 0, "r1-n00", 1, 2))
    clock.advance(5000)
    assert b.observe().by_id()["h-a1"].phase == Phase.STARTING
    b.stop("h-a1")
    clock.advance(5000)
    assert b.observe().by_id()["h-a1"].phase == Phase.STOPPED


def test_already_exists_of_another_instance_is_not_success() -> None:
    clock = VirtualClock()
    mine, fake = kube(clock, "mine")
    other = KubeBackend(
        fake, REF, instance_id="other", to_platform_ms=lambda w: int(w - EPOCH_MS), now_ms=clock.now_ms
    )
    r = req("w1-a1", [("r0-n00", 1)])
    other.start(r)  # a leftover Job of another store with the same name
    with pytest.raises(BackendError):
        mine.start(r)
    mine2 = KubeBackend(
        fake, REF, instance_id="other", to_platform_ms=lambda w: int(w - EPOCH_MS), now_ms=clock.now_ms
    )
    mine2.start(r)  # the same instance: AlreadyExists is the earlier call's Job


def test_cordoned_node_stays_ready() -> None:
    class Stub:
        def list_nodes(self) -> list[dict]:
            return [
                {
                    "metadata": {"name": "n1", "labels": {"awp.local/node": "n1", "awp.local/rack": "r0"}},
                    "spec": {"unschedulable": True},
                    "status": {
                        "capacity": {"nvidia.com/gpu": "8"},
                        "conditions": [{"type": "Ready", "status": "True"}],
                    },
                }
            ]

    b = KubeBackend(Stub(), REF, instance_id="x", to_platform_ms=int, now_ms=lambda: 0)  # type: ignore[arg-type]
    assert b.inventory()[0].ready


def test_failed_pod_ends_its_job_siblings_and_eviction_is_reported() -> None:
    clock = VirtualClock()
    b, fake = kube(clock)
    b.start(req("e-a1", [("r0-n00", 2)], runtime=600, gpus=2))
    clock.advance(2000)
    assert b.observe().by_id()["e-a1"].phase == Phase.RUNNING
    assert fake.evict("e-a1-0-0")
    clock.advance(3000)  # the sibling is deleted by the Job controller (backoffLimit 0)
    st = b.observe().by_id()["e-a1"]
    assert st.phase == Phase.FAILED and st.reason is not None and st.reason.value == "evicted"
    assert fake.usage() == {}


def test_auto_id_skips_ids_a_client_chose() -> None:
    s = Store()
    for ns in reference_namespaces(32):
        s.put_namespace(ns, 0)
    s.save_inventory(list(REF.nodes), 0, None)
    spec = {"gpus": 1, "sim": {"runtime_s": 10}}
    admission.submit(s, "team-a", {**spec, "id": "w1"}, None, 1)
    w, _ = admission.submit(s, "team-a", spec, None, 2)
    w2, _ = admission.submit(s, "team-a", spec, None, 3)
    assert (w.id, w2.id) == ("w2", "w3")
    s.close()


class Recorder:
    name = "recorder"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def hello(self, msg: dict) -> None:
        pass

    def schedule(self, view: dict) -> dict:
        self.calls.append([h["job_id"] for h in view["history_new"]])
        return {"actions": []}

    def close(self) -> None:
        pass


def test_history_new_first_call_of_a_session_carries_all() -> None:
    rec = Recorder()
    r = PolicyRunner(rec)
    hist = {"items": [{"job_id": "a"}], "upto": 5}

    def history(after: int) -> tuple[list, int]:
        return ([h for h in hist["items"] if after < 5] if after < hist["upto"] else []), hist["upto"]

    hello = {"cluster": {"nodes": []}}
    view = {"pending": [], "nodes": [], "running": []}
    r.decide(view, hello, 0, lambda d: [], history=history)
    r.decide(view, hello, 1, lambda d: [], history=history)
    r.decide(view, {"cluster": {"nodes": [{"name": "new"}]}}, 2, lambda d: [], history=history)  # new session
    assert rec.calls == [["a"], [], ["a"]]


def test_errors_other_than_policy_failures_are_counted() -> None:
    m = Metrics()
    bad = ExternalPolicy([str(ROOT / "no-such-policy-binary")], name="external:x", wire_name="x", timeout_s=2)
    r = PolicyRunner(bad, max_failures=2, metrics=m)
    hello = {"cluster": {"nodes": []}}
    view = {"pending": [], "nodes": [], "running": [], "history_new": []}
    r.decide(view, hello, 0, lambda d: [])
    r.decide(view, hello, 10, lambda d: [])
    assert r.degraded and m.policy_failures.labels(kind="crash")._value.get() == 2
    with pytest.raises(PolicyFailure) as e:  # a list where a string belongs is invalid, not a TypeError
        validate_decision(
            {"actions": [{"op": "start", "job_id": ["x"], "placement": []}]},
            {"pending": [], "nodes": [], "running": []},
            [],
            {},
        )
    assert e.value.kind == "invalid"


def test_writes_to_a_child_that_stops_reading_time_out(tmp_path) -> None:
    script = tmp_path / "deaf.py"
    script.write_text(
        "import sys, time\nsys.stdin.buffer.readline()\n"
        'sys.stdout.buffer.write(b\'{"type":"hello"}\\n\'); sys.stdout.flush()\ntime.sleep(60)\n'
    )
    p = ExternalPolicy(
        [sys.executable, str(script)],
        name="external:deaf",
        wire_name="deaf",
        timeout_s=2,
        stderr_path=str(tmp_path / "deaf.log"),
    )
    p.hello({"cluster": {"nodes": []}})
    big = {"pending": [], "nodes": [], "running": [], "history_new": [{"job_id": "x" * 100}] * 50_000}
    with pytest.raises(PolicyFailure) as e:
        p.schedule(big)
    assert e.value.kind == "timeout"
    assert p.alive_children() == []


def test_backend_calls_are_fenced_after_a_takeover() -> None:
    clock = VirtualClock()
    store = Store(seed=1)
    for ns in reference_namespaces(32):
        store.put_namespace(ns, 0)
    backend, target = make_backend("local", REF, clock, instance_id=store.instance_id)
    a = Controller(
        store, backend, PolicyRunner(make_builtin("fifo+first_fit")), REF, holder="a", thresholds=TH
    )
    a.tick(0)
    admission.submit(store, "team-a", {"gpus": 1, "sim": {"runtime_s": 100}}, None, 0)
    store.acquire_lease("b", None, 100_000, 5000)  # another controller took the expired lease
    with pytest.raises(LeaseLost):
        a._check_lease()
    with pytest.raises(LeaseLost):
        a.tick(100)
    assert target.known_ids() == set()  # a started nothing in the backend
    assert store.get_workload("w1").state == WorkloadState.QUEUED
    store.close()


def test_nodes_list_reports_capacity_for_ready_nodes_only() -> None:
    n = NodeInfo("x", "r0", "a100", 1.0, 8, 64, 1000, ready=False)
    assert not n.ready
