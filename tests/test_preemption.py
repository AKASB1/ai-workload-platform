"""Tier 2 item 1: preemption with checkpoints (policies, validation, and the platform end to end)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_workload_platform.clock import VirtualClock
from ai_workload_platform.cluster import ClusterConfig, load_cluster
from ai_workload_platform.controller import Controller
from ai_workload_platform.controller.drivers import Submission, VirtualDriver
from ai_workload_platform.controller.rules import Thresholds
from ai_workload_platform.models import NodeInfo
from ai_workload_platform.policy import PolicyFailure
from ai_workload_platform.policy.builtin import make_builtin
from ai_workload_platform.policy.runner import PolicyRunner
from ai_workload_platform.policy.validate import PreemptAction, StartAction, validate_decision
from ai_workload_platform.sim import make_backend, reference_namespaces
from ai_workload_platform.store.replay import diff, replay
from ai_workload_platform.store.sql import Store

ROOT = Path(__file__).resolve().parents[1]
REF = load_cluster(ROOT / "configs" / "clusters" / "reference.json")
NODES = [
    NodeInfo("n0", "r0", "a100", 1.0, 8, 128, 1_024_000),
    NodeInfo("n1", "r0", "a100", 1.0, 8, 128, 1_024_000),
]
ONE = ClusterConfig("two", {"a100": 1.0}, tuple(NODES), restart_overhead_s=120.0)
HELLO = {"cluster": ONE.hello_cluster()}


def running(
    jid: str,
    node: str,
    gpus: int,
    prio: int,
    done: float,
    ck: float = 0,
    pre: bool = True,
    tenant: str = "team-a",
    start: float = 0,
) -> dict:
    return {
        "job_id": jid,
        "tenant": tenant,
        "user": tenant,
        "priority": prio,
        "gpus": gpus,
        "workers": 1,
        "cpus": 0,
        "mem_mb": 0,
        "gpu_class": "",
        "topology": "any",
        "submit_s": 0,
        "first_start_s": start,
        "run_start_s": start,
        "overhead_s": 0,
        "placement": [{"node": node, "workers": 1}],
        "rate": 1.0,
        "estimate_s": 1000,
        "retained_at_start_s": 0,
        "work_done_s": done,
        "est_remaining_work_s": 1000 - done,
        "est_end_s": 1000,
        "preemptible": pre,
        "checkpoint_interval_s": ck,
        "preemptions": 0,
    }


def pending(jid: str, gpus: int, prio: int, tenant: str = "team-b") -> dict:
    return {
        "job_id": jid,
        "tenant": tenant,
        "user": tenant,
        "submit_s": 5,
        "priority": prio,
        "gpus": gpus,
        "workers": 1,
        "cpus": 0,
        "mem_mb": 0,
        "gpu_class": "",
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


def view(free: dict[str, int], run: list[dict], pend: list[dict], ns=None, stopping=None) -> dict:
    v = {
        "now_s": 100.0,
        "nodes": [
            {
                "name": n.name,
                "free_gpus": free.get(n.name, 0),
                "free_cpus": 128,
                "free_mem_mb": 1_024_000,
                "running": [],
            }
            for n in NODES
        ],
        "running": run,
        "pending": pend,
        "tenants": [],
        "history_new": [],
    }
    if ns is not None:
        v["awp_namespaces"] = ns
    if stopping is not None:
        v["awp_stopping"] = stopping
    return v


def decide(name: str, v: dict) -> list[dict]:
    p = make_builtin(name)
    p.hello(HELLO)
    return p.schedule(v)["actions"]


def test_preempt_picks_least_lost_work_and_drops_unneeded_victims() -> None:
    run = [
        running("a", "n0", 4, 1, done=500, ck=400),  # loses 100 x 4 = 400
        running("b", "n0", 4, 1, done=300, ck=0),  # loses 300 x 4 = 1200
        running("c", "n1", 8, 1, done=50, ck=0),  # loses 50 x 8 = 400, later start: sorted after a
        running("d", "n1", 8, 9, done=1, ck=0),
    ]  # priority 9: never a victim of a priority-8 job
    acts = decide("priority+best_fit+preempt", view({}, run, [pending("hi", 8, 8)]))
    # a (least lost) does not free 8 GPUs alone; with c the job fits on n1, so a is dropped again
    assert acts == [{"op": "preempt", "job_id": "c"}]
    acts = decide("priority+best_fit+preempt", view({}, run, [pending("hi", 4, 8)]))
    assert acts == [{"op": "preempt", "job_id": "a"}]


def test_no_more_victims_while_a_preemption_is_under_way() -> None:
    run = [running("c", "n1", 4, 1, done=50)]
    v = view({}, run, [pending("hi", 8, 8)], stopping={"n0": [8, 0, 0]})
    assert decide("priority+best_fit+preempt", v) == []


def test_reclaim_preempts_only_namespaces_above_their_quota() -> None:
    ns = [
        {"tenant": "team-a", "quota_gpus": 4, "cap_gpus": 16, "allocated_gpus": 12},
        {"tenant": "team-b", "quota_gpus": 8, "cap_gpus": 16, "allocated_gpus": 4},
    ]
    run = [
        running("a1", "n0", 8, 1, done=100, tenant="team-a"),
        running("b1", "n1", 4, 1, done=10, tenant="team-b"),
        running("a2", "n1", 4, 1, done=20, tenant="team-a"),
    ]
    acts = decide("quota+first_fit+reclaim", view({}, run, [pending("bq", 4, 1, tenant="team-b")], ns))
    assert acts == [{"op": "preempt", "job_id": "a2"}]  # b1 is team-b's own work; team-a is above its quota
    ns[1]["allocated_gpus"] = 6  # team-b would go above its quota with the new workload: no reclaim
    assert decide("quota+first_fit+reclaim", view({}, run, [pending("bq", 4, 1, tenant="team-b")], ns)) == []


def test_validation_of_preempt_actions() -> None:
    run = [running("r", "n0", 8, 1, done=10), running("np", "n1", 8, 1, done=10, pre=False)]
    v = view({}, run, [pending("p", 8, 8)])
    ok = validate_decision(
        {
            "actions": [
                {"op": "preempt", "job_id": "r"},
                {"op": "start", "job_id": "p", "placement": [{"node": "n0", "workers": 1}]},
            ]
        },
        v,
        NODES,
        {},
    )
    assert ok == [PreemptAction("r"), StartAction("p", (("n0", 1),))]
    for acts in (
        [{"op": "preempt", "job_id": "np"}],  # not preemptible
        [{"op": "preempt", "job_id": "r"}, {"op": "preempt", "job_id": "r"}],  # twice
        [{"op": "preempt", "job_id": "p"}],
    ):  # not running
        with pytest.raises(PolicyFailure):
            validate_decision({"actions": acts}, v, NODES, {})


def test_preemption_end_to_end_retained_work_and_restart_overhead() -> None:
    clock = VirtualClock()
    store = Store(seed=1)
    for ns in reference_namespaces(32):
        store.put_namespace(ns, 0)
    backend, _ = make_backend("local", REF, clock, instance_id=store.instance_id)
    th = Thresholds()

    def ctl() -> Controller:
        return Controller(
            store,
            backend,
            PolicyRunner(make_builtin("priority+best_fit+preempt")),
            REF,
            holder="c",
            thresholds=th,
        )

    low = {
        "id": "low",
        "priority": 1,
        "gpus": 8,
        "workers": 4,
        "preemptible": True,
        "checkpoint_interval_s": 100,
        "sim": {"runtime_s": 1000},
    }
    high = {"id": "high", "priority": 8, "gpus": 8, "sim": {"runtime_s": 300}}
    subs = [Submission(0, "team-a", low), Submission(250_000, "team-b", high)]
    drv = VirtualDriver(store, backend, ctl(), clock, subs, max_ms=10_000_000)
    drv.run()
    evs = store.events()
    ended = [e for e in evs if e.type == "attempt_ended" and e.workload_id == "low"]
    assert [e.data["reason"] for e in ended] == ["preempted", "succeeded"]
    pre = ended[0]
    assert pre.data["retained_s"] == 200.0 and 200 <= pre.data["work_done_s"] < 300
    starts = [e for e in evs if e.type == "started" and e.workload_id == "low"]
    assert starts[1].data["overhead_s"] == 120.0 and starts[1].data["retained_s"] == 200.0
    # the restart holds the GPUs 120 s, then runs the remaining 800 s of work
    assert ended[1].at_ms == starts[1].at_ms + 120_000 + 800_000
    w = store.get_workload("low")
    assert (w.preemptions, w.retained_ms, w.counted) == (1, 200_000, 0)
    high_start = next(e for e in evs if e.type == "started" and e.workload_id == "high")
    assert high_start.at_ms >= pre.at_ms  # the GPUs were held until the preempted attempt ended (R9)
    assert diff(replay(evs), store.all_workloads(), store.all_attempts(), store.books()) == []
    store.close()
