"""The live service: store + clock + backend + policy + controller on the asyncio driver."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_workload_platform.clock import SystemClock
from ai_workload_platform.cluster import ClusterConfig, load_cluster
from ai_workload_platform.controller import Controller
from ai_workload_platform.controller.drivers import LiveDriver
from ai_workload_platform.controller.rules import Thresholds
from ai_workload_platform.models import Namespace
from ai_workload_platform.observability import Metrics
from ai_workload_platform.policy import make_policy
from ai_workload_platform.policy.runner import PolicyRunner
from ai_workload_platform.sim import reference_namespaces
from ai_workload_platform.store.dialect import PostgresDialect, SQLiteDialect
from ai_workload_platform.store.sql import Store

log = logging.getLogger("awp.service")


@dataclass
class ServiceConfig:
    db: str = "outputs/awp.db"
    pg_dsn: str | None = None
    backend: str = "local"  # local | kube
    cluster: str = "configs/clusters/reference.json"
    policy: str = "fifo+first_fit"
    policy_cmd: list[str] | None = None
    policy_cwd: str | None = None
    scale: float = 1.0  # platform seconds per wall second (local backend)
    seed: int = 0
    holder: str = field(default_factory=lambda: f"ctl-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    kubeconfig: str | None = None
    in_cluster: bool = False
    kube_namespace: str = "awp-workloads"
    kube_mode: str = "pinned"
    time_scale: float = 1.0  # Kubernetes backend: run times are divided by it
    image: str = "busybox:1.37"
    bootstrap_namespaces: bool = True
    start_latency_ms: int = 0
    thresholds: Thresholds = field(default_factory=Thresholds)
    policy_log: str | None = None

    @staticmethod
    def from_env(**over: Any) -> ServiceConfig:
        c = ServiceConfig()
        env = os.environ
        c.db = env.get("AWP_DB", c.db)
        c.pg_dsn = env.get("AWP_PG_DSN") or None
        c.backend = env.get("AWP_BACKEND", c.backend)
        c.cluster = env.get("AWP_CLUSTER", c.cluster)
        c.policy = env.get("AWP_POLICY", c.policy)
        if env.get("AWP_POLICY_CMD"):
            import shlex

            c.policy_cmd = shlex.split(env["AWP_POLICY_CMD"], posix=os.name != "nt")
        c.policy_cwd = env.get("AWP_POLICY_CWD") or None
        c.scale = float(env.get("AWP_SCALE", c.scale))
        c.kubeconfig = env.get("AWP_KUBECONFIG") or None
        c.in_cluster = env.get("AWP_IN_CLUSTER", "") == "1"
        for k, v in over.items():
            if v is not None:
                setattr(c, k, v)
        return c


class Platform:
    """Everything the API needs. `start()`/`stop()` run the live driver (the API tests may skip it)."""

    def __init__(self, cfg: ServiceConfig, *, metrics: Metrics | None = None) -> None:
        self.cfg = cfg
        self.metrics = metrics or Metrics()
        if cfg.pg_dsn:
            dialect: Any = PostgresDialect(cfg.pg_dsn)
        else:
            if cfg.db != ":memory:":
                Path(cfg.db).parent.mkdir(parents=True, exist_ok=True)
            dialect = SQLiteDialect(cfg.db)
        self.store = Store(dialect, seed=cfg.seed, metrics=self.metrics)
        origin = int(self.store.set_meta_if_absent("origin_wall_ms", str(int(SystemClock().origin_wall_ms))))
        self.clock = SystemClock(origin, scale=cfg.scale)
        self.cluster: ClusterConfig = load_cluster(cfg.cluster)
        self.backend = self._make_backend()
        if cfg.bootstrap_namespaces and not self.store.namespaces():
            for ns in reference_namespaces(self.cluster.total_gpus):
                self.store.put_namespace(ns, self.clock.now_ms())
        self.driver = LiveDriver(
            self.make_controller,
            self.clock,
            self.backend,
            observe_interval_ms=cfg.thresholds.observe_interval_ms,
        )

    def _make_backend(self) -> Any:
        if self.cfg.backend == "local":
            from ai_workload_platform.scheduler.local import LocalBackend

            return LocalBackend(self.cluster, self.clock, start_latency_ms=self.cfg.start_latency_ms)
        if self.cfg.backend == "kube":
            from ai_workload_platform.scheduler.kube.backend import KubeBackend
            from ai_workload_platform.scheduler.kube.client import RealKubeClient

            client = RealKubeClient(self.cfg.kubeconfig, in_cluster=self.cfg.in_cluster)
            clock = self.clock
            return KubeBackend(
                client,
                self.cluster,
                instance_id=self.store.instance_id,
                to_platform_ms=lambda w: int((w - clock.origin_wall_ms) * clock.scale),
                now_ms=clock.now_ms,
                namespace=self.cfg.kube_namespace,
                mode=self.cfg.kube_mode,
                time_scale=self.cfg.time_scale,
                image=self.cfg.image,
            )
        raise ValueError(f"unknown backend {self.cfg.backend!r} (local or kube)")

    def make_controller(self) -> Controller:
        th = self.cfg.thresholds
        pol = make_policy(
            self.cfg.policy,
            cmd=self.cfg.policy_cmd,
            cwd=self.cfg.policy_cwd,
            seed=self.cfg.seed,
            timeout_s=th.policy_timeout_s,
            stderr_path=self.cfg.policy_log,
        )
        runner = PolicyRunner(pol, max_failures=th.policy_max_failures, metrics=self.metrics)
        return Controller(
            self.store,
            self.backend,
            runner,
            self.cluster,
            holder=self.cfg.holder,
            thresholds=th,
            metrics=self.metrics,
            clock=self.clock,
        )

    def put_namespace(self, ns: Namespace) -> None:
        self.store.put_namespace(ns, self.clock.now_ms())

    def health(self) -> tuple[str, list[str]]:
        reasons: list[str] = []
        ctl = self.driver.controller
        if ctl is not None and ctl.runner.degraded:
            reasons.append(f"policy degraded ({ctl.runner.last_failure})")
        if self.driver.last_backend_error:
            reasons.append("backend unreachable")
        holder, _epoch, expires = self.store.lease()
        if holder is None or expires < self.clock.now_ms():
            reasons.append("no lease holder")
        if self.driver.last_error:
            reasons.append(self.driver.last_error)
        return ("degraded" if reasons else "ok"), reasons

    def close(self) -> None:
        self.store.close()
