"""Assemble an in-process platform on the virtual clock (simulations, the harness, the benchmarks)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_workload_platform.clock import VirtualClock
from ai_workload_platform.cluster import ClusterConfig
from ai_workload_platform.controller import Controller
from ai_workload_platform.controller.drivers import RunStats, Submission, VirtualDriver
from ai_workload_platform.controller.rules import Thresholds
from ai_workload_platform.models import Event, Namespace
from ai_workload_platform.policy import Policy, make_policy
from ai_workload_platform.policy.runner import PolicyRunner
from ai_workload_platform.scheduler.kube.backend import KubeBackend
from ai_workload_platform.scheduler.kube.fake import EPOCH_MS, FakeKubeClient
from ai_workload_platform.scheduler.local import LocalBackend
from ai_workload_platform.store.sql import Store

BACKENDS = ("local", "kube-fake")


def reference_namespaces(
    total_gpus: int = 32,
    shares: dict[str, float] | None = None,
    cap: int | None = None,
    max_queued: int = 200,
) -> list[Namespace]:
    """team-a/b/c with shares 0.4/0.3/0.3: quota = floor(share * total GPUs), cap = total GPUs."""
    shares = shares or {"team-a": 0.4, "team-b": 0.3, "team-c": 0.3}
    return [
        Namespace(n, int(s * total_gpus), cap if cap is not None else total_gpus, 9, max_queued)
        for n, s in sorted(shares.items())
    ]


def make_backend(
    kind: str,
    cluster: ClusterConfig,
    clock: VirtualClock,
    *,
    start_latency_ms: int = 0,
    stop_latency_ms: int = 0,
    instance_id: str = "sim",
    kube_mode: str = "pinned",
    kube_latencies: dict[str, int] | None = None,
) -> tuple[Any, Any]:
    """Returns (backend, fault target): the local backend is its own fault target; for the
    Kubernetes backend on the fake client the fault target is the fake."""
    if kind == "local":
        b = LocalBackend(cluster, clock, start_latency_ms=start_latency_ms, stop_latency_ms=stop_latency_ms)
        return b, b
    if kind == "kube-fake":
        fake = FakeKubeClient(cluster, clock, latencies=kube_latencies)
        b = KubeBackend(
            fake,
            cluster,
            instance_id=instance_id,
            to_platform_ms=lambda w: int(round(w - EPOCH_MS)),
            now_ms=clock.now_ms,
            mode=kube_mode,
        )
        return b, fake
    raise ValueError(f"unknown backend {kind!r}; known: {', '.join(BACKENDS)}")


@dataclass
class SimResult:
    events: list[Event]
    stats: RunStats
    store: Store
    controller: Controller
    backend: Any
    namespaces: list[Namespace] = field(default_factory=list)
    cluster: ClusterConfig | None = None

    def close(self) -> None:
        self.controller.close()
        self.store.close()


def simulate(
    submissions: list[Submission],
    cluster: ClusterConfig,
    *,
    policy: str | Policy = "fifo+first_fit",
    backend: str = "local",
    seed: int = 0,
    thresholds: Thresholds | None = None,
    namespaces: list[Namespace] | None = None,
    start_latency_ms: int = 0,
    max_ms: int | None = None,
    metrics: Any = None,
    kube_mode: str = "pinned",
    kube_latencies: dict[str, int] | None = None,
    keep: bool = False,
) -> SimResult:
    clock = VirtualClock(0)
    store = Store(seed=seed, metrics=metrics)
    nss = namespaces if namespaces is not None else reference_namespaces(cluster.total_gpus)
    for ns in nss:
        store.put_namespace(ns, 0)
    b, _target = make_backend(
        backend,
        cluster,
        clock,
        start_latency_ms=start_latency_ms,
        instance_id=store.instance_id,
        kube_mode=kube_mode,
        kube_latencies=kube_latencies,
    )
    th = thresholds or Thresholds()
    pol = make_policy(policy, seed=seed) if isinstance(policy, str) else policy
    runner = PolicyRunner(pol, max_failures=th.policy_max_failures, metrics=metrics)
    ctl = Controller(store, b, runner, cluster, holder="sim", thresholds=th, metrics=metrics)
    drv = VirtualDriver(store, b, ctl, clock, submissions, max_ms=max_ms, metrics=metrics)
    try:
        stats = drv.run()
        events = store.events()
    finally:
        if not keep:
            ctl.close()
    res = SimResult(events, stats, store, ctl, b, nss, cluster)
    if not keep:
        store.close()
    return res


def events_jsonl(events: list[Event]) -> bytes:
    """The event log as JSON lines (sorted keys): the form compared byte for byte."""
    import json

    return "".join(
        json.dumps(e.to_json_obj(), sort_keys=True, separators=(",", ":")) + "\n" for e in events
    ).encode("utf-8")


def trace_log(
    seed: int,
    backend: str = "local",
    policy: str = "fifo+first_fit",
    jobs: int = 40,
    variant: str = "balanced",
    load: float = 0.8,
    cluster_path: str | None = None,
) -> bytes:
    """Generate a trace for `seed`, run it on the virtual driver, return the event log (picklable entry
    point for ProcessPoolExecutor workers)."""
    from pathlib import Path

    from ai_workload_platform.bench.generator import generate
    from ai_workload_platform.bench.trace import to_submissions
    from ai_workload_platform.cluster import load_cluster

    path = cluster_path or str(
        Path(__file__).resolve().parents[2] / "configs" / "clusters" / "reference.json"
    )
    cluster = load_cluster(path)
    rows, _ = generate(cluster, seed, jobs=jobs, variant=variant, load=load)
    res = simulate(to_submissions(rows), cluster, policy=policy, backend=backend, seed=seed)
    return events_jsonl(res.events)
