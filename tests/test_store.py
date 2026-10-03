"""Check 2: the store contract suite (SQLite always; PostgreSQL when AWP_PG_DSN is reachable)."""

from __future__ import annotations

import random
import threading
from dataclasses import replace

import pytest

from ai_workload_platform import admission
from ai_workload_platform.models import (
    AttemptState,
    InvalidTransition,
    LeaseLost,
    StoreUnavailable,
    VersionConflict,
    WorkloadState,
)
from ai_workload_platform.store import ops
from ai_workload_platform.store.replay import ReplayError, diff, replay
from ai_workload_platform.store.sql import Fence
from tests.conftest import setup_reference

SPEC = {"gpus": 2, "sim": {"runtime_s": 100}, "retry": {"max_attempts": 3, "jitter": "full"}}


def check_replay(store) -> None:
    st = replay(store.events())
    assert diff(st, store.all_workloads(), store.all_attempts(), store.books()) == []


def test_compare_and_set_rejects_a_stale_version(store_factory) -> None:
    s = store_factory()
    setup_reference(s)
    w, _ = admission.submit(s, "team-a", SPEC, None, 10)
    w2, a, _ = ops.commit_started(s, None, w, [{"node": "r0-n00", "workers": 1}], 20)
    before = s.last_seq()
    with pytest.raises(VersionConflict):
        ops.commit_started(s, None, w, [{"node": "r0-n01", "workers": 1}], 21)  # w is the stale snapshot
    assert s.last_seq() == before
    assert s.get_workload(w.id).version == 2
    nw, _ = ops.request_cancel(s, "team-a", w.id, 30)
    with pytest.raises(VersionConflict):
        ops.commit_running(s, None, w2, a, 25, ["r0-n00"], 31)  # cancel changed the workload version
    check_replay(s)


def test_events_are_gap_free_and_monotone_with_rollbacks(store_factory) -> None:
    s = store_factory()
    setup_reference(s)
    ws = [admission.submit(s, "team-b", SPEC, None, t)[0] for t in (5, 6, 7)]
    with pytest.raises(VersionConflict):
        ops.commit_requeued(s, None, replace(ws[0], state=WorkloadState.RETRY_WAIT, version=99), 8)
    with pytest.raises(InvalidTransition):
        ops.commit_requeued(s, None, ws[1], 9)  # QUEUED cannot be requeued
    ops.request_cancel(s, "team-b", ws[2].id, 4)  # a clock reading older than the last event
    evs = s.events()
    assert [e.seq for e in evs] == [1, 2, 3, 4]
    assert [e.at_ms for e in evs] == [5, 6, 7, 7]  # at_ms never decreases
    assert s.get_workload(ws[2].id).state == WorkloadState.CANCELLED
    check_replay(s)


def test_idempotent_submission_and_mismatch(store_factory) -> None:
    s = store_factory()
    setup_reference(s)
    w1, created1 = admission.submit(s, "team-a", SPEC, "key-1", 10)
    w2, created2 = admission.submit(s, "team-a", dict(SPEC), "key-1", 11)
    assert created1 and not created2 and w1.id == w2.id
    w3, created3 = admission.submit(s, "team-b", SPEC, "key-1", 12)  # keys are per namespace
    assert created3 and w3.id != w1.id
    with pytest.raises(Exception) as e:
        admission.submit(s, "team-a", {**SPEC, "gpus": 4}, "key-1", 13)
    assert getattr(e.value, "code", "") == "IDEMPOTENCY_MISMATCH"
    assert sum(1 for e in s.events() if e.type == "submitted") == 2


def test_lease_fencing_stale_holder_cannot_write(store_factory) -> None:
    s = store_factory()
    setup_reference(s)
    w, _ = admission.submit(s, "team-a", SPEC, None, 0)
    ea = s.acquire_lease("ctl-a", None, 0, 1000)
    assert ea == 1
    with pytest.raises(LeaseLost) as standby:
        s.acquire_lease("ctl-b", None, 500, 1000)
    assert standby.value.details.get("standby")
    assert s.acquire_lease("ctl-a", ea, 900, 1000) == ea  # renewal
    eb = s.acquire_lease("ctl-b", None, 1901, 1000)  # expired: b takes over with epoch + 1
    assert eb == 2
    with pytest.raises(LeaseLost):
        ops.commit_started(s, Fence("ctl-a", ea), w, [{"node": "r0-n00", "workers": 1}], 1950)
    with pytest.raises(LeaseLost):
        s.acquire_lease("ctl-a", ea, 1960, 1000)
    w2, a, _ = ops.commit_started(s, Fence("ctl-b", eb), w, [{"node": "r0-n00", "workers": 1}], 1970)
    assert a.state == AttemptState.STARTING and w2.version == 2
    # a fresh incarnation under the same name takes over at once and fences the old one
    eb2 = s.acquire_lease("ctl-b", None, 1980, 1000)
    assert eb2 == 3
    with pytest.raises(LeaseLost):
        ops.commit_running(s, Fence("ctl-b", eb), w2, a, 1990, ["r0-n00"], 1990)
    check_replay(s)


def _random_sequence(s, seed: int, n_ops: int = 40) -> None:
    r = random.Random(seed)
    nodes = ["r0-n00", "r0-n01", "r1-n00", "r1-n01"]
    t = 0
    reasons = [
        "succeeded",
        "failed_retryable",
        "failed_fatal",
        "node_lost",
        "backend_lost",
        "start_timeout",
        "cancelled",
    ]
    for _ in range(n_ops):
        t += r.randint(0, 3)
        ws = s.all_workloads()
        op = r.random()
        try:
            if op < 0.25 or not ws:
                spec = {
                    "gpus": r.choice([1, 2, 4]),
                    "workers": r.choice([1, 1, 2]),
                    "priority": r.randint(0, 9),
                    "sim": {"runtime_s": 50},
                    "retry": {"max_attempts": r.randint(1, 3)},
                }
                key = r.choice([None, f"k{r.randint(0, 5)}"])
                admission.submit(s, r.choice(["team-a", "team-b", "team-c"]), spec, key, t)
                continue
            w = r.choice(ws)
            if r.random() < 0.15:  # a stale snapshot: the CAS must reject it (or it is still current)
                w = replace(w, version=max(1, w.version - 1))
            atts = [a for a in s.attempts_of(w.id) if a.state != AttemptState.ENDED]
            a = atts[0] if atts else None
            if op < 0.4:
                ops.request_cancel(s, w.namespace, w.id, t)
            elif w.state == WorkloadState.QUEUED:
                pl = [{"node": n, "workers": 1} for n in r.sample(nodes, w.workers)]
                ops.commit_started(s, None, w, sorted(pl, key=lambda p: p["node"]), t)
            elif w.state == WorkloadState.RETRY_WAIT:
                ops.commit_requeued(s, None, w, t)
            elif a is not None and a.state == AttemptState.STARTING and r.random() < 0.5:
                ops.commit_running(s, None, w, a, t, [p["node"] for p in a.placement], t)
            elif (
                a is not None
                and a.state in (AttemptState.STARTING, AttemptState.RUNNING)
                and r.random() < 0.3
            ):
                ops.commit_stop_requested(
                    s, None, w, a, r.choice(["cancelled", "node_lost", "start_timeout"]), t
                )
            elif a is not None:
                ops.commit_attempt_ended(
                    s, None, w, a, r.choice(reasons), r.choice([None, 0, 1, 2]), t, t, None, t, s.seed
                )
        except (VersionConflict, InvalidTransition):
            pass
        except Exception as e:  # admission rejections are fine
            if getattr(e, "code", None) is None:
                raise


@pytest.mark.parametrize("batch", range(4))
def test_replay_equals_tables_over_random_sequences(store_factory, batch: int) -> None:
    """200 seeded random operation sequences in total (4 batches x 50), per store kind."""
    for seed in range(batch * 50, batch * 50 + 50):
        s = store_factory(seed=seed)
        setup_reference(s, max_queued=1000)
        _random_sequence(s, seed)
        evs = s.events()
        assert [e.seq for e in evs] == list(range(1, len(evs) + 1))
        st = replay(evs)
        assert diff(st, s.all_workloads(), s.all_attempts(), s.books()) == [], f"seed {seed}"
        for w in s.all_workloads():
            assert w.version == sum(1 for e in evs if e.workload_id == w.id)
        for a in s.all_attempts():
            assert a.version == sum(1 for e in evs if e.attempt_id == a.id)
        s.close()


def test_replay_detects_a_tampered_log(mem_store) -> None:
    setup_reference(mem_store)
    w, _ = admission.submit(mem_store, "team-a", SPEC, None, 1)
    ops.request_cancel(mem_store, "team-a", w.id, 2)
    evs = mem_store.events()
    with pytest.raises(ReplayError):
        replay([evs[1]])  # gap
    bad = evs[1].__class__(2, 2, "requeued", "team-a", w.id, None, {"state": "QUEUED"})
    with pytest.raises(ReplayError):
        replay([evs[0], bad])  # QUEUED cannot be requeued


def test_store_unavailable_rolls_back(mem_store) -> None:
    setup_reference(mem_store)

    class Fail:
        def before_begin(self, write: bool) -> None:
            pass

        def before_commit(self) -> None:
            raise StoreUnavailable("injected")

    mem_store.injector = Fail()
    with pytest.raises(StoreUnavailable):
        admission.submit(mem_store, "team-a", SPEC, None, 1)
    mem_store.injector = None
    assert mem_store.events() == [] and mem_store.all_workloads() == []
    w, _ = admission.submit(mem_store, "team-a", SPEC, None, 2)
    assert w.id == "w1"  # the counter rolled back with the failed transaction


def test_race_16_threads_submit_and_cancel_50_workloads(threaded_store_factory) -> None:
    s = threaded_store_factory()
    setup_reference(s, max_queued=1000)
    ids = [f"race-{i:02d}" for i in range(50)]
    errors: list[BaseException] = []
    barrier = threading.Barrier(16)

    def worker(k: int) -> None:
        r = random.Random(k)
        try:
            barrier.wait(timeout=30)
            for i in r.sample(range(50), 50):
                wid = ids[i]
                if r.random() < 0.5:
                    try:
                        admission.submit(s, "team-a", {**SPEC, "id": wid}, f"key-{wid}", i)
                    except Exception as e:  # noqa: BLE001
                        if getattr(e, "code", None) not in ("DUPLICATE_ID",):
                            raise
                else:
                    try:
                        ops.request_cancel(s, "team-a", wid, i)
                    except Exception as e:  # noqa: BLE001
                        if getattr(e, "code", None) != "UNKNOWN_WORKLOAD":
                            raise
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors, errors
    evs = s.events()
    assert [e.seq for e in evs] == list(range(1, len(evs) + 1))
    by_w: dict[str, list[str]] = {}
    for e in evs:
        by_w.setdefault(e.workload_id, []).append(e.type)
    for wid, types in by_w.items():
        assert types.count("submitted") == 1, wid
        assert types.count("cancel_requested") <= 1, wid
    check_replay(s)
