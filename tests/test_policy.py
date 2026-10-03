"""Check 4 (protocol client, failures, fallback, recovery) and the built-ins on hand-made views."""

from __future__ import annotations

import pytest

from ai_workload_platform.cluster import ClusterConfig
from ai_workload_platform.models import NodeInfo
from ai_workload_platform.observability import Metrics
from ai_workload_platform.policy import PolicyFailure, make_policy, stub_command
from ai_workload_platform.policy.builtin import make_builtin
from ai_workload_platform.policy.external import ExternalPolicy
from ai_workload_platform.policy.runner import PolicyRunner
from ai_workload_platform.policy.validate import validate_decision

NODES = [
    NodeInfo("r0-n00", "r0", "a100", 1.0, 8, 128, 1_024_000),
    NodeInfo("r0-n01", "r0", "a100", 1.0, 8, 128, 1_024_000),
    NodeInfo("r1-n00", "r1", "v100", 0.4, 8, 128, 1_024_000),
]
CLUSTER = ClusterConfig("t", {"a100": 1.0, "v100": 0.4}, tuple(NODES))
HELLO = {"cluster": CLUSTER.hello_cluster()}


def job(
    jid: str,
    gpus: int,
    workers: int = 1,
    prio: int = 4,
    submit: float = 0.0,
    tenant: str = "team-a",
    cls: str = "",
    cpus: int = 0,
    mem: int = 0,
) -> dict:
    return {
        "job_id": jid,
        "tenant": tenant,
        "user": tenant,
        "submit_s": submit,
        "priority": prio,
        "gpus": gpus,
        "workers": workers,
        "cpus": cpus,
        "mem_mb": mem,
        "gpu_class": cls,
        "topology": "any",
        "estimate_s": 100,
        "wait_s": 0,
        "retained_s": 0,
        "max_wait_s": None,
        "preemptible": False,
        "checkpoint_interval_s": 0,
        "preemptions": 0,
        "started": False,
    }


def view(free: dict[str, int], pending: list[dict], namespaces: list[dict] | None = None) -> dict:
    nodes = [
        {
            "name": n.name,
            "free_gpus": free.get(n.name, 0),
            "free_cpus": 128,
            "free_mem_mb": 1_024_000,
            "running": [],
        }
        for n in NODES
    ]
    pend = sorted(pending, key=lambda j: (-j["priority"], j["submit_s"], j["job_id"]))
    v = {"now_s": 10.0, "nodes": nodes, "running": [], "pending": pend, "tenants": [], "history_new": []}
    if namespaces is not None:
        v["awp_namespaces"] = namespaces
    return v


def decide(name: str, v: dict) -> list[tuple[str, list[tuple[str, int]]]]:
    p = make_builtin(name)
    p.hello(HELLO)
    d = p.schedule(v)
    return [(a["job_id"], [(e["node"], e["workers"]) for e in a["placement"]]) for a in d["actions"]]


def test_fifo_first_fit_blocks_and_packs_in_node_order() -> None:
    v = view(
        {"r0-n00": 4, "r0-n01": 8, "r1-n00": 8},
        [job("a", 2, 3, submit=1), job("b", 8, 1, submit=2, prio=9), job("c", 1, submit=3)],
    )
    # a: 2 workers on n00 (4 GPUs), 1 on n01; b (8) does not fit anywhere of a100... it may use v100 r1-n00
    assert decide("fifo+first_fit", v) == [
        ("a", [("r0-n00", 2), ("r0-n01", 1)]),
        ("b", [("r1-n00", 1)]),
        ("c", [("r0-n01", 1)]),
    ]
    v2 = view({"r0-n00": 4}, [job("a", 8, submit=1), job("b", 1, submit=2)])
    assert decide("fifo+first_fit", v2) == []  # a blocks the queue


def test_priority_best_fit_skips_and_picks_fullest_node() -> None:
    v = view(
        {"r0-n00": 3, "r0-n01": 8, "r1-n00": 2},
        [
            job("big", 8, 2, prio=9, submit=0),
            job("small", 2, prio=4, submit=1),
            job("mid", 3, prio=6, submit=2),
        ],
    )
    # big (16) does not fit -> skipped; mid (3) -> node left with fewest free: r0-n00 (3-3=0)
    # small (2) -> r1-n00 (2-2=0)
    assert decide("priority+best_fit", v) == [("mid", [("r0-n00", 1)]), ("small", [("r1-n00", 1)])]


def test_class_cpu_and_memory_constraints() -> None:
    v = view({"r0-n00": 8, "r0-n01": 8, "r1-n00": 8}, [job("v", 4, cls="v100"), job("cpu", 1, cpus=200)])
    assert decide("priority+best_fit", v) == [("v", [("r1-n00", 1)])]


def test_quota_first_fit_within_quota_first_then_borrowers() -> None:
    ns = [
        {"tenant": "team-a", "quota_gpus": 4, "cap_gpus": 8, "allocated_gpus": 2},
        {"tenant": "team-b", "quota_gpus": 8, "cap_gpus": 8, "allocated_gpus": 0},
    ]
    v = view(
        {"r0-n00": 8, "r0-n01": 0, "r1-n00": 0},
        [
            job("a1", 4, prio=9, submit=0, tenant="team-a"),
            job("b1", 4, prio=1, submit=1, tenant="team-b"),
            job("a2", 2, prio=9, submit=2, tenant="team-a"),
        ],
        ns,
    )
    # a1 (2+4 > quota 4) is deferred; a2 (2+2 = 4) and b1 are within quota; then a1 may borrow (4+4 <= cap 8)
    # but only 2 GPUs are left -> a1 skipped
    assert decide("quota+first_fit", v) == [("a2", [("r0-n00", 1)]), ("b1", [("r0-n00", 1)])]
    v2 = view({"r0-n00": 8, "r0-n01": 8}, [job("a1", 8, prio=9, tenant="team-a")], ns)
    assert decide("quota+first_fit", v2) == []  # 2 + 8 > cap 8


def test_validation_rejects_each_kind_of_bad_action() -> None:
    v = view({"r0-n00": 8, "r0-n01": 2, "r1-n00": 8}, [job("a", 4, 2), job("v", 2, cls="v100")])
    caps = {"team-a": (32, 0)}
    good = {"actions": [{"op": "start", "job_id": "a", "placement": [{"node": "r0-n00", "workers": 2}]}]}
    assert validate_decision(good, v, NODES, caps)[0].placement == (("r0-n00", 2),)
    bad = [
        {"op": "start", "job_id": "zz", "placement": [{"node": "r0-n00", "workers": 1}]},
        {"op": "start", "job_id": "a", "placement": [{"node": "nope", "workers": 2}]},
        {"op": "start", "job_id": "a", "placement": [{"node": "r0-n00", "workers": 1}]},
        {"op": "start", "job_id": "a", "placement": [{"node": "r0-n01", "workers": 2}]},
        {
            "op": "start",
            "job_id": "a",
            "placement": [{"node": "r0-n00", "workers": 1}, {"node": "r0-n00", "workers": 1}],
        },
        {"op": "start", "job_id": "v", "placement": [{"node": "r0-n00", "workers": 1}]},
        {"op": "preempt", "job_id": "a"},
        {"op": "start", "job_id": "a", "placement": [{"node": "r0-n00", "workers": 0}]},
    ]
    for act in bad:
        with pytest.raises(PolicyFailure) as e:
            validate_decision({"actions": [act]}, v, NODES, caps)
        assert e.value.kind == "invalid"
    twice = {"actions": [good["actions"][0], good["actions"][0]]}
    with pytest.raises(PolicyFailure):
        validate_decision(twice, v, NODES, caps)
    with pytest.raises(PolicyFailure):
        validate_decision(good, v, NODES, {"team-a": (6, 0)})  # cap
    notready = [
        NodeInfo(n.name, n.rack, n.gpu_class, n.speed, n.gpus, n.cpus, n.mem_mb, n.name != "r0-n00")
        for n in NODES
    ]
    with pytest.raises(PolicyFailure):
        validate_decision(good, v, notready, caps)


@pytest.fixture
def stub(tmp_path):
    made: list[ExternalPolicy] = []

    def factory(fault: str | None = None, at_seq: int = 1, timeout_s: float = 20.0) -> ExternalPolicy:
        params = {"fault": fault, "at_seq": at_seq} if fault else {}
        p = ExternalPolicy(
            stub_command(),
            name="stub",
            wire_name="fifo+first_fit",
            params=params,
            timeout_s=timeout_s,
            stderr_path=str(tmp_path / f"stub-{len(made)}.log"),
        )
        made.append(p)
        return p

    yield factory
    for p in made:
        p.close()
        assert p.alive_children() == [], f"child left: {p.pids}"


def test_stub_hello_schedule_bye_matches_builtin(stub) -> None:
    p = stub()
    p.hello(HELLO)
    v = view(
        {"r0-n00": 4, "r0-n01": 8, "r1-n00": 8},
        [job("a", 2, 3, submit=1), job("b", 8, 1, submit=2, prio=9), job("c", 1, submit=3)],
    )
    d1 = p.schedule(v)
    d2 = p.schedule(v)
    assert d1["seq"] == 1 and d2["seq"] == 2
    b = make_builtin("fifo+first_fit")
    b.hello(HELLO)
    assert d1["actions"] == b.schedule(v)["actions"]
    pid = p.proc.pid
    p.close()
    assert p.proc is None and p.alive_children() == []
    assert pid in p.pids


@pytest.mark.parametrize(
    "fault,kind",
    [
        ("error", "error"),
        ("crash", "crash"),
        ("malformed", "malformed"),
        ("unknown_field", "malformed"),
        ("wrong_seq", "malformed"),
    ],
)
def test_stub_failures_are_classified_and_reaped(stub, fault: str, kind: str) -> None:
    p = stub(fault)
    p.hello(HELLO)
    with pytest.raises(PolicyFailure) as e:
        p.schedule(view({"r0-n00": 8}, [job("a", 1)]))
    assert e.value.kind == kind
    assert p.proc is None and p.alive_children() == []
    if fault != "crash":
        assert "injecting" in e.value.stderr_tail


def test_stub_timeout_kills_the_child(stub) -> None:
    p = stub("hang", timeout_s=1.0)
    p.hello(HELLO)
    with pytest.raises(PolicyFailure) as e:
        p.schedule(view({"r0-n00": 8}, [job("a", 1)]))
    assert e.value.kind == "timeout"
    assert p.alive_children() == []


def test_stub_invalid_action_is_a_policy_failure(stub) -> None:
    p = stub("invalid")
    p.hello(HELLO)
    v = view({"r0-n00": 8}, [job("a", 1)])
    with pytest.raises(PolicyFailure) as e:
        validate_decision(p.schedule(v), v, NODES, {})
    assert e.value.kind == "invalid"


def test_runner_degrades_after_max_failures_and_recovers(stub) -> None:
    m = Metrics()
    p = stub("error", at_seq=1)
    r = PolicyRunner(p, max_failures=3, metrics=m)
    v = view({"r0-n00": 8}, [job("a", 1)])

    def val(d):
        return validate_decision(d, v, NODES, {})

    for t in (0, 10, 20):
        assert r.decide(v, HELLO, t, val) == [] or r.degraded
    assert r.degraded and m.policy_degraded._value.get() == 1
    assert m.policy_failures.labels(kind="error")._value.get() == 3
    # degraded: the fallback decides
    acts = r.decide(v, HELLO, 500, val)
    assert [a.workload_id for a in acts] == ["a"]
    # retry at 20 + 1000 ms fails again (stub still faulty) -> next retry after 2 s
    r.decide(v, HELLO, 1020, val)
    assert r.next_try_ms == 1020 + 2000
    # heal the stub: the next session is started without the fault
    p.params = {}
    acts = r.decide(v, HELLO, 3020, val)
    assert [a.workload_id for a in acts] == ["a"] and not r.degraded
    assert m.policy_degraded._value.get() == 0
    r.close()


def test_make_policy_names() -> None:
    for n in ("fifo+first_fit", "priority+best_fit", "quota+first_fit"):
        assert make_policy(n).name == n
    with pytest.raises(ValueError):
        make_policy("nope")


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_stub_event_logs_equal_builtin_fifo(seed: int) -> None:
    from ai_workload_platform.sim import trace_log

    assert trace_log(seed, "local", "stub") == trace_log(seed, "local", "fifo+first_fit")
