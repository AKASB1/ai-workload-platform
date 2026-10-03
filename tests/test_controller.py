"""Check 6: one test per reconciliation rule R1-R9 on hand-built store and backend states, and one per
crash point (crash, restart with a fresh controller over the same store and backend, converge)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from ai_workload_platform import admission, crash
from ai_workload_platform.clock import VirtualClock
from ai_workload_platform.cluster import load_cluster
from ai_workload_platform.controller import Controller
from ai_workload_platform.controller.drivers import Submission, VirtualDriver
from ai_workload_platform.controller.rules import Thresholds
from ai_workload_platform.models import TERMINAL_STATES, AttemptRequest, AttemptState, WorkloadState
from ai_workload_platform.models.spec import validate_spec
from ai_workload_platform.policy.builtin import make_builtin
from ai_workload_platform.policy.runner import PolicyRunner
from ai_workload_platform.sim import make_backend, reference_namespaces
from ai_workload_platform.store import ops
from ai_workload_platform.store.replay import diff, replay
from ai_workload_platform.store.sql import Store

REF = load_cluster(Path(__file__).resolve().parents[1] / "configs" / "clusters" / "reference.json")
TH = Thresholds(
    start_retry_ms=5000,
    start_timeout_ms=60_000,
    node_grace_ms=20_000,
    lost_grace_ms=15_000,
    stop_retry_ms=8000,
    lease_ttl_ms=10_000,
)


class Rig:
    def __init__(self, kind: str = "local", start_latency_ms: int = 0, stop_latency_ms: int = 0) -> None:
        self.clock = VirtualClock(0)
        self.store = Store(seed=1)
        for ns in reference_namespaces(32):
            self.store.put_namespace(ns, 0)
        self.backend, self.target = make_backend(
            kind,
            REF,
            self.clock,
            start_latency_ms=start_latency_ms,
            stop_latency_ms=stop_latency_ms,
            instance_id=self.store.instance_id,
        )
        self.ctl = self.controller()
        self.tick(0)

    def controller(self) -> Controller:
        return Controller(
            self.store,
            self.backend,
            PolicyRunner(make_builtin("fifo+first_fit")),
            REF,
            holder="ctl",
            thresholds=TH,
        )

    def submit(self, spec: dict, ns: str = "team-a"):
        return admission.submit(self.store, ns, spec, None, self.clock.now_ms())[0]

    def tick(self, at: int | None = None):
        if at is not None:
            self.clock.advance_to(at)
        return self.ctl.tick(self.clock.now_ms())

    def w(self, wid: str):
        return self.store.get_workload(wid)

    def a(self, aid: str):
        return self.store.get_attempt(aid)

    def types(self, wid: str) -> list[str]:
        return [
            e.type + (f":{e.data['reason']}" if "reason" in e.data else "")
            for e in self.store.events()
            if e.workload_id == wid
        ]

    def run_until(self, pred, max_ms: int = 3_600_000) -> None:
        while not pred():
            now = self.clock.now_ms()
            cands = [t for t in (self.ctl.next_wakeup_ms(), self.backend.next_event_ms()) if t is not None]
            nxt = max(min(cands), now + 1) if cands else now + 1000
            assert nxt <= max_ms, "condition not reached"
            self.tick(nxt)

    def check_replay(self) -> None:
        assert (
            diff(
                replay(self.store.events()),
                self.store.all_workloads(),
                self.store.all_attempts(),
                self.store.books(),
            )
            == []
        )


def spec(wid: str, gpus: int = 1, runtime: float = 100, **kw) -> dict:
    s = {"id": wid, "gpus": gpus, "sim": {"runtime_s": runtime}, "retry": {"jitter": "none"}}
    for k, v in kw.items():
        if k in ("fail_after_s", "fail_attempts", "exit_code"):
            s["sim"][k] = v
        elif k in ("fatal_exit_codes", "max_attempts"):
            s["retry"][k] = v
        else:
            s[k] = v
    return s


def test_r1_lost_start_is_repeated_after_start_retry() -> None:
    r = Rig()
    r.target.faults.fail_start = 1
    r.submit(spec("r1"))
    r.tick(0)
    assert r.a("r1-a1").state == AttemptState.STARTING and "r1-a1" not in r.target.known_ids()
    r.tick(4999)
    assert "r1-a1" not in r.target.known_ids()
    r.tick(5000)
    assert "r1-a1" in r.target.known_ids()
    r.run_until(lambda: r.w("r1").state in TERMINAL_STATES)
    assert r.w("r1").state == WorkloadState.SUCCEEDED and r.w("r1").attempts == 1
    r.check_replay()


def test_r1_start_that_took_effect_is_not_repeated() -> None:
    r = Rig()
    r.target.faults.fail_start = 1
    r.target.faults.fail_start_effect = True  # the call worked, the reply was lost
    r.submit(spec("r1b"))
    r.tick(0)
    calls = r.target.faults.calls["start"]
    r.run_until(lambda: r.w("r1b").state in TERMINAL_STATES)
    assert r.target.faults.calls["start"] == calls and r.w("r1b").state == WorkloadState.SUCCEEDED


def test_r2_start_timeout_stops_and_retries() -> None:
    r = Rig()
    r.target.faults.slow_start_ms["r2-a1"] = 100_000
    r.submit(spec("r2"))
    r.tick(0)
    r.tick(59_999)
    assert r.a("r2-a1").state == AttemptState.STARTING
    r.tick(60_000)
    assert r.a("r2-a1").state == AttemptState.STOPPING and r.a("r2-a1").stop_reason == "start_timeout"
    r.run_until(lambda: r.w("r2").state in TERMINAL_STATES)
    assert r.a("r2-a1").end_reason == "start_timeout" and r.w("r2").state == WorkloadState.SUCCEEDED
    assert r.w("r2").counted == 1 and r.w("r2").attempts == 2
    r.check_replay()


def test_r3_orphans_are_stopped_and_forgotten() -> None:
    r = Rig()
    ghost = validate_spec(spec("ghost", runtime=1000))
    r.backend.start(AttemptRequest("ghost-a1", "ghost", "team-a", 1, ghost, (("r1-n01", 1),)))
    r.tick(1)
    assert "ghost-a1" not in r.target.active_ids()  # stopped
    r.tick(2)
    assert "ghost-a1" not in r.target.known_ids()  # forgotten once terminal
    assert r.store.events() == []  # the store never knew it


def test_r4_node_loss_stops_after_grace_and_holds_until_the_end() -> None:
    r = Rig()
    r.submit(spec("r4", gpus=8, runtime=1000))
    r.tick(0)
    r.tick(1)
    assert r.a("r4-a1").state == AttemptState.RUNNING
    node = r.a("r4-a1").placement[0]["node"]
    r.target.node_down(node)
    r.tick(100)
    r.tick(20_099)
    assert r.a("r4-a1").state == AttemptState.RUNNING
    r.tick(20_100)
    assert r.a("r4-a1").state == AttemptState.STOPPING and r.a("r4-a1").stop_reason == "node_lost"
    r.tick(25_000)
    assert sum(g for (_ns, n), (g, _c, _m) in r.store.books().items() if n == node) == 8  # R9: still held
    r.target.node_up(node)
    r.run_until(lambda: r.a("r4-a1").state == AttemptState.ENDED)
    assert r.a("r4-a1").end_reason == "node_lost" and r.w("r4").state == WorkloadState.RETRY_WAIT
    r.run_until(lambda: r.w("r4").state in TERMINAL_STATES)
    assert r.w("r4").state == WorkloadState.SUCCEEDED
    r.check_replay()


def test_r5_lost_attempt_ends_after_lost_grace_only() -> None:
    r = Rig()
    r.submit(spec("r5", runtime=1000))
    r.tick(0)
    r.tick(10)
    assert r.a("r5-a1").state == AttemptState.RUNNING
    r.target.lose("r5-a1")
    r.tick(20)  # last snapshot that showed it: t = 10
    r.tick(15_009)
    assert r.a("r5-a1").state == AttemptState.RUNNING
    r.tick(15_010)
    assert r.a("r5-a1").state == AttemptState.ENDED and r.a("r5-a1").end_reason == "backend_lost"
    assert r.w("r5").state == WorkloadState.RETRY_WAIT
    r.check_replay()


def test_r6_observed_progress() -> None:
    r = Rig(start_latency_ms=500)
    r.submit(spec("ok", runtime=10))
    r.submit(spec("fatal", fail_after_s=5, fail_attempts=1, exit_code=2, fatal_exit_codes=[2]))
    r.submit(spec("other", runtime=1000))
    r.tick(0)
    r.tick(500)
    a = r.a("ok-a1")
    assert a.state == AttemptState.RUNNING and a.observed_started_ms == 500
    r.run_until(lambda: r.w("ok").state in TERMINAL_STATES and r.w("fatal").state in TERMINAL_STATES)
    assert r.w("ok").state == WorkloadState.SUCCEEDED and r.a("ok-a1").observed_ended_ms == 10_500
    assert r.w("fatal").state == WorkloadState.FAILED and r.a("fatal-a1").exit_code == 2
    # stopped without a stop request (someone else stopped it): backend_lost
    r.backend.stop("other-a1")
    r.tick(r.clock.now_ms() + 1)
    assert r.a("other-a1").end_reason == "backend_lost"
    r.check_replay()


def test_r6_success_wins_over_a_stop_request() -> None:
    r = Rig(stop_latency_ms=5000)
    r.submit(spec("race", runtime=10))
    r.tick(0)
    r.tick(8000)
    ops.request_cancel(r.store, "team-a", "race", 8000)
    r.tick(8001)  # stop requested, takes 5 s; the attempt ends at 10 s
    assert r.a("race-a1").state == AttemptState.STOPPING
    r.run_until(lambda: r.w("race").state in TERMINAL_STATES)
    assert r.w("race").state == WorkloadState.SUCCEEDED
    r.check_replay()


def test_r7_retry_release_at_retry_at() -> None:
    r = Rig()
    r.submit(spec("r7", fail_after_s=10, fail_attempts=1))
    r.tick(0)
    r.run_until(lambda: r.w("r7").state == WorkloadState.RETRY_WAIT)
    at = r.w("r7").retry_at_ms
    assert at == r.a("r7-a1").ended_ms + 5000  # backoff base 5 s, jitter none
    r.tick(at - 1)
    assert r.w("r7").state == WorkloadState.RETRY_WAIT
    r.tick(at)
    assert r.w("r7").state == WorkloadState.STARTING  # requeued and started in the same tick
    assert r.types("r7")[-2:] == ["requeued", "started"]
    r.run_until(lambda: r.w("r7").state in TERMINAL_STATES)
    assert r.w("r7").state == WorkloadState.SUCCEEDED


def test_r8_cancel_stops_and_repeats_failed_stops() -> None:
    r = Rig()
    r.submit(spec("r8", runtime=1000))
    r.submit(spec("q", gpus=8, runtime=1000), ns="team-b")
    r.submit(spec("q2", gpus=8, runtime=1000), ns="team-b")
    r.submit(spec("q3", gpus=8, runtime=1000), ns="team-b")
    r.submit(spec("waiting", gpus=8), ns="team-c")
    r.tick(0)
    assert r.w("waiting").state == WorkloadState.QUEUED
    _w, changed = ops.request_cancel(r.store, "team-c", "waiting", 1)
    assert changed and r.w("waiting").state == WorkloadState.CANCELLED  # queued: at once
    assert not ops.request_cancel(r.store, "team-c", "waiting", 2)[1]  # repeat: nothing
    r.target.faults.fail_stop = 1
    ops.request_cancel(r.store, "team-a", "r8", 10)
    r.tick(10)
    assert r.a("r8-a1").state == AttemptState.STOPPING and "r8-a1" in r.target.active_ids()
    r.tick(8009)
    assert "r8-a1" in r.target.active_ids()
    r.tick(8010)  # stop_retry_ms after the failed call
    r.tick(8011)
    assert r.w("r8").state == WorkloadState.CANCELLED and r.a("r8-a1").end_reason == "cancelled"
    assert r.types("r8").count("cancel_requested") == 1
    r.check_replay()


def test_r9_resources_held_while_stopping() -> None:
    r = Rig(stop_latency_ms=30_000)
    r.submit(spec("big", gpus=8, workers=4, runtime=1000))
    r.submit(spec("next", gpus=8, runtime=10))
    r.tick(0)
    ops.request_cancel(r.store, "team-a", "big", 5)
    r.tick(5)
    assert r.a("big-a1").state == AttemptState.STOPPING
    r.tick(29_000)
    assert r.w("next").state == WorkloadState.QUEUED  # the cluster is still held by the stopping attempt
    assert sum(g for (g, _c, _m) in r.store.books().values()) == 32
    r.run_until(lambda: r.w("next").state != WorkloadState.QUEUED)
    assert r.a("big-a1").state == AttemptState.ENDED
    assert r.a("next-a1").started_ms >= r.a("big-a1").ended_ms
    r.check_replay()


SCENARIO = [
    Submission(0, "team-a", spec("s-ok", gpus=2, runtime=50)),
    Submission(0, "team-b", spec("s-fail", gpus=4, fail_after_s=10, fail_attempts=1)),
    Submission(1000, "team-c", spec("s-cancel", gpus=8, runtime=500)),
    Submission(2000, "team-a", spec("s-gang", gpus=8, workers=2, runtime=40)),
    Submission(30_000, "team-c", None, cancel="s-cancel"),
    Submission(31_000, "team-b", spec("s-late", gpus=1, runtime=20)),
]


@pytest.mark.parametrize("kind", ["local", "kube-fake"])
@pytest.mark.parametrize("point", crash.CRASH_POINTS)
def test_crash_point_restart_converges(point: str, kind: str) -> None:
    clock = VirtualClock(0)
    store = Store(seed=3)
    for ns in reference_namespaces(32):
        store.put_namespace(ns, 0)
    backend, target = make_backend(kind, REF, clock, instance_id=store.instance_id)

    def new() -> Controller:
        return Controller(
            store, backend, PolicyRunner(make_builtin("fifo+first_fit")), REF, holder="ctl", thresholds=TH
        )

    hits = Counter()

    def hook(name: str, subject: str | None = None) -> None:
        hits[name] += 1
        if name == point and hits[name] == 1:
            raise crash.Crash(name)

    prev = crash.install(hook)
    try:
        drv = VirtualDriver(store, backend, new(), clock, list(SCENARIO), restart=new, max_ms=3_600_000)
        stats = drv.run()
    finally:
        crash.install(prev)
    assert hits[point] >= 1, f"{point} was never reached"
    assert stats.crashes == 1
    ws = {w.id: w for w in store.all_workloads()}
    assert set(ws) == {"s-ok", "s-fail", "s-cancel", "s-gang", "s-late"}
    assert all(w.state in TERMINAL_STATES for w in ws.values())
    assert ws["s-cancel"].state == WorkloadState.CANCELLED
    assert ws["s-ok"].state == ws["s-fail"].state == ws["s-gang"].state == WorkloadState.SUCCEEDED
    assert all(a.state == AttemptState.ENDED for a in store.all_attempts())
    assert store.books() == {}
    assert diff(replay(store.events()), store.all_workloads(), store.all_attempts(), store.books()) == []
    leftover = target.active_ids() if kind == "local" else target.active_attempts()
    assert not leftover
    terminal = Counter(
        e.workload_id
        for e in store.events()
        if e.data.get("state") in {s.value for s in TERMINAL_STATES}
        and e.type in ("attempt_ended", "cancel_requested")
    )
    assert all(v == 1 for v in terminal.values()) and len(terminal) == 5
    store.close()
