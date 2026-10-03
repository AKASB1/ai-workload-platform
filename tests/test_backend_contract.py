"""Check 5: one backend contract suite over the local backend, the Kubernetes backend on the fake
client, and the Kubernetes backend on a real cluster (AWP_KUBECONFIG; skipped when unreachable)."""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from ai_workload_platform.clock import SystemClock, VirtualClock
from ai_workload_platform.cluster import load_cluster
from ai_workload_platform.models import AttemptRequest, FailReason, Phase, Snapshot
from ai_workload_platform.models.spec import validate_spec
from ai_workload_platform.scheduler.kube.backend import KubeBackend
from ai_workload_platform.scheduler.kube.client import ApiError
from ai_workload_platform.scheduler.kube.fake import EPOCH_MS, FakeKubeClient
from ai_workload_platform.scheduler.local import LocalBackend

ROOT = Path(__file__).resolve().parents[1]
REF = load_cluster(ROOT / "configs" / "clusters" / "reference.json")
KIND = load_cluster(ROOT / "configs" / "clusters" / "kind.json")
KUBECONFIG = os.environ.get("AWP_KUBECONFIG")


def _real_skip() -> str | None:
    if not KUBECONFIG:
        return "AWP_KUBECONFIG is not set (no real cluster; see deploy/README.md for the kind set-up)"
    try:
        from ai_workload_platform.scheduler.kube.client import RealKubeClient

        RealKubeClient(KUBECONFIG, timeout_s=5).list_nodes()
        return None
    except Exception as e:  # noqa: BLE001
        return f"cluster at AWP_KUBECONFIG is not reachable: {e}"


_REAL_SKIP = _real_skip()


@dataclass
class Ctx:
    kind: str
    backend: object
    nodes: list[str]
    wait: Callable[[Callable[[Snapshot], bool], float], Snapshot]
    node_down: Callable[[str], None] | None = None
    node_up: Callable[[str], None] | None = None
    client: object = None
    cleanup: Callable[[], None] | None = None


def _virtual_wait(clock: VirtualClock, backend) -> Callable:
    def wait(pred: Callable[[Snapshot], bool], max_s: float) -> Snapshot:
        deadline = clock.now_ms() + int(max_s * 1000)
        while True:
            snap = backend.observe()
            if pred(snap):
                return snap
            nxt = backend.next_event_ms()
            if nxt is None or nxt > deadline:
                if clock.now_ms() >= deadline:
                    raise AssertionError(f"condition not reached; last snapshot {snap}")
                clock.advance_to(deadline)
            else:
                clock.advance_to(nxt)

    return wait


def _real_wait(backend) -> Callable:
    def wait(pred: Callable[[Snapshot], bool], max_s: float) -> Snapshot:
        t_end = time.monotonic() + max_s
        while True:
            snap = backend.observe()
            if pred(snap):
                return snap
            if time.monotonic() > t_end:
                raise AssertionError(f"condition not reached within {max_s} s; last snapshot {snap}")
            time.sleep(0.5)  # polling a real cluster with a timeout

    return wait


def make_ctx(kind: str, mode: str = "pinned") -> Ctx:
    if kind == "local":
        clock = VirtualClock()
        b = LocalBackend(REF, clock)
        return Ctx(kind, b, [n.name for n in REF.nodes], _virtual_wait(clock, b), b.node_down, b.node_up)
    if kind == "fake":
        clock = VirtualClock()
        client = FakeKubeClient(REF, clock)
        b = KubeBackend(
            client,
            REF,
            instance_id="test",
            to_platform_ms=lambda w: int(w - EPOCH_MS),
            now_ms=clock.now_ms,
            mode=mode,
        )
        return Ctx(
            kind,
            b,
            [n.name for n in REF.nodes],
            _virtual_wait(clock, b),
            client.node_down,
            client.node_up,
            client,
        )
    from ai_workload_platform.scheduler.kube.client import RealKubeClient

    clock = SystemClock()
    client = RealKubeClient(KUBECONFIG, timeout_s=10)
    inst = f"t{uuid.uuid4().hex[:8]}"
    b = KubeBackend(
        client,
        KIND,
        instance_id=inst,
        to_platform_ms=lambda w: int(w - clock.origin_wall_ms),
        now_ms=clock.now_ms,
        mode=mode,
    )

    def cleanup() -> None:
        for j in client.list_jobs(b.namespace, b.selector):
            try:
                client.delete_job(b.namespace, j["metadata"]["name"])
            except ApiError:
                pass

    return Ctx(kind, b, [n.name for n in KIND.nodes], _real_wait(b), client=client, cleanup=cleanup)


KINDS = [
    "local",
    "fake",
    pytest.param(
        "real", marks=[pytest.mark.kube, pytest.mark.skipif(_REAL_SKIP is not None, reason=_REAL_SKIP or "")]
    ),
]


@pytest.fixture(params=KINDS)
def ctx(request):
    c = make_ctx(request.param)
    yield c
    if c.cleanup:
        c.cleanup()


def req(aid: str, placement: list[tuple[str, int]], runtime: float = 4, **extra) -> AttemptRequest:
    workers = sum(k for _, k in placement)
    spec = validate_spec(
        {
            "id": aid.split("-a")[0],
            "gpus": extra.pop("gpus", 2),
            "workers": workers,
            "sim": {"runtime_s": runtime, **extra},
        }
    )
    return AttemptRequest(aid, spec["id"], "team-a", int(aid.rsplit("-a", 1)[1]), spec, tuple(placement))


def phase(snap: Snapshot, aid: str) -> Phase | None:
    s = snap.by_id().get(aid)
    return s.phase if s else None


def test_inventory(ctx: Ctx) -> None:
    inv = ctx.backend.inventory()
    names = [n.name for n in inv]
    assert [(n.rack, n.name) for n in inv] == sorted((n.rack, n.name) for n in inv)
    assert set(ctx.nodes) <= set(names)
    assert all(n.gpus == 8 and n.ready for n in inv if n.name in ctx.nodes)


def test_start_observe_success(ctx: Ctx) -> None:
    n0 = ctx.nodes[0]
    ctx.backend.start(req("ok1-a1", [(n0, 2)]))
    first = ctx.backend.observe()
    assert phase(first, "ok1-a1") in (Phase.STARTING, Phase.RUNNING)
    snap = ctx.wait(lambda s: phase(s, "ok1-a1") == Phase.SUCCEEDED, 120)
    st = snap.by_id()["ok1-a1"]
    assert st.exit_code == 0 and st.nodes == (n0,)
    assert st.started_ms is not None and st.ended_ms is not None and st.ended_ms >= st.started_ms


def test_failure_with_exit_code(ctx: Ctx) -> None:
    ctx.backend.start(req("bad-a1", [(ctx.nodes[1], 1)], fail_after_s=2, fail_attempts=1, exit_code=3))
    snap = ctx.wait(lambda s: phase(s, "bad-a1") == Phase.FAILED, 120)
    st = snap.by_id()["bad-a1"]
    assert (st.exit_code, st.reason) == (3, FailReason.EXIT)
    # attempt 2 of the same workload runs to the end
    ctx.backend.start(req("bad-a2", [(ctx.nodes[1], 1)], fail_after_s=2, fail_attempts=1, exit_code=3))
    snap = ctx.wait(lambda s: phase(s, "bad-a2") == Phase.SUCCEEDED, 120)


def test_start_twice_is_idempotent(ctx: Ctx) -> None:
    r = req("twice-a1", [(ctx.nodes[0], 1), (ctx.nodes[1], 1)])
    ctx.backend.start(r)
    ctx.backend.start(r)
    snap = ctx.backend.observe()
    assert [a.attempt_id for a in snap.attempts].count("twice-a1") == 1
    if ctx.client is not None:
        jobs = [
            j
            for j in ctx.client.list_jobs(ctx.backend.namespace, ctx.backend.selector)
            if j["metadata"]["labels"]["awp.local/attempt"] == "twice-a1"
        ]
        assert sorted(j["metadata"]["name"] for j in jobs) == ["twice-a1-0", "twice-a1-1"]
    ctx.wait(lambda s: phase(s, "twice-a1") == Phase.SUCCEEDED, 120)


def test_stop_twice_and_unknown(ctx: Ctx) -> None:
    ctx.backend.start(req("long-a1", [(ctx.nodes[2], 1)], runtime=600))
    ctx.wait(lambda s: phase(s, "long-a1") == Phase.RUNNING, 120)
    ctx.backend.stop("long-a1")
    ctx.backend.stop("long-a1")
    ctx.backend.stop("no-such-attempt-a1")
    snap = ctx.wait(lambda s: phase(s, "long-a1") == Phase.STOPPED, 120)
    assert "no-such-attempt-a1" not in snap.by_id()
    ctx.backend.stop("long-a1")  # stop of an ended attempt
    assert phase(ctx.backend.observe(), "long-a1") == Phase.STOPPED


def test_ended_attempts_stay_visible_until_forgotten(ctx: Ctx) -> None:
    ctx.backend.start(req("vis-a1", [(ctx.nodes[3], 1)], runtime=2))
    ctx.wait(lambda s: phase(s, "vis-a1") == Phase.SUCCEEDED, 120)
    for _ in range(3):
        assert phase(ctx.backend.observe(), "vis-a1") == Phase.SUCCEEDED
    ctx.backend.forget("vis-a1")
    ctx.wait(lambda s: "vis-a1" not in s.by_id(), 60)
    ctx.backend.forget("vis-a1")  # forgetting twice is harmless


def test_node_down_and_up(ctx: Ctx) -> None:
    if ctx.node_down is None:
        pytest.skip("on a real cluster the node-loss demonstration is Tier 2 item 4, not part of the suite")
    n = ctx.nodes[1]
    ctx.backend.start(req("nd-a1", [(n, 1)], runtime=600))
    ctx.wait(lambda s: phase(s, "nd-a1") == Phase.RUNNING, 60)
    ctx.node_down(n)
    assert not next(x for x in ctx.backend.inventory() if x.name == n).ready
    snap = ctx.wait(lambda s: True, 1)
    assert phase(snap, "nd-a1") == Phase.RUNNING  # last known state while the node is unreachable
    ctx.node_up(n)
    snap = ctx.wait(lambda s: phase(s, "nd-a1") == Phase.FAILED, 60)
    assert snap.by_id()["nd-a1"].reason == FailReason.NODE_LOST
    assert next(x for x in ctx.backend.inventory() if x.name == n).ready


def test_already_exists_is_success(ctx: Ctx) -> None:
    if ctx.client is None:
        pytest.skip("AlreadyExists is a Kubernetes reply; the local backend's analogue is start-twice")
    r = req("ae-a1", [(ctx.nodes[0], 1)])
    body = ctx.backend._job_body(r, 0, ctx.nodes[0], 1, 1)
    ctx.client.create_job(ctx.backend.namespace, body)
    with pytest.raises(ApiError) as e:
        ctx.client.create_job(ctx.backend.namespace, body)
    assert e.value.status == 409
    ctx.backend.start(r)  # treats 409 AlreadyExists as success
    ctx.wait(lambda s: phase(s, "ae-a1") == Phase.SUCCEEDED, 120)


@pytest.mark.parametrize(
    "kind",
    [
        "fake",
        pytest.param(
            "real",
            marks=[pytest.mark.kube, pytest.mark.skipif(_REAL_SKIP is not None, reason=_REAL_SKIP or "")],
        ),
    ],
)
def test_partial_gang_in_delegate_mode(kind: str) -> None:
    c = make_ctx(kind, mode="delegate")
    try:
        nodes = c.nodes
        # fill three nodes completely, leave one free: a gang of 2 x 8 GPUs can place only one worker
        for i, n in enumerate(nodes[:3]):
            c.backend.start(req(f"fill{i}-a1", [(n, 1)], runtime=600, gpus=8))
        c.wait(lambda s: all(phase(s, f"fill{i}-a1") == Phase.RUNNING for i in range(3)), 120)
        c.backend.start(req("gang-a1", [(nodes[3], 1), (nodes[2], 1)], runtime=60, gpus=8))
        snap = c.wait(
            lambda s: len(s.by_id().get("gang-a1").nodes if "gang-a1" in s.by_id() else ()) == 1, 120
        )
        assert phase(snap, "gang-a1") == Phase.STARTING  # a partially started gang is starting as a whole
        snap = c.wait(lambda s: True, 2)
        assert phase(snap, "gang-a1") == Phase.STARTING
        c.backend.stop("gang-a1")
        c.wait(lambda s: phase(s, "gang-a1") == Phase.STOPPED, 120)
    finally:
        if c.cleanup:
            c.cleanup()
