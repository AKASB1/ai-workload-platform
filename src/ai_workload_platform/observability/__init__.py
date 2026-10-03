"""Prometheus metrics (prefix awp_) and JSON-lines logging on stderr."""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

METRIC_NAMES = (
    "awp_workloads",
    "awp_queue_delay_seconds",
    "awp_run_seconds",
    "awp_attempts_total",
    "awp_admission_rejections_total",
    "awp_scheduling_cycle_seconds",
    "awp_policy_failures_total",
    "awp_policy_degraded",
    "awp_stale_actions_total",
    "awp_reconcile_repairs_total",
    "awp_gpus_allocated",
    "awp_gpus_capacity",
    "awp_backend_errors_total",
    "awp_store_errors_total",
    "awp_controller_leader",
)

_DELAY_BUCKETS = (1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600, 7200, 14400, 43200, 86400)


class Metrics:
    """One registry per platform instance (the harness creates many instances in one process)."""

    def __init__(self, wall_timer: Callable[[], float] = time.perf_counter) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.wall_timer = wall_timer
        self.workloads = Gauge(
            "awp_workloads", "Workloads by namespace and state", ["namespace", "state"], registry=r
        )
        self.queue_delay = Histogram(
            "awp_queue_delay_seconds",
            "Submit to first start (platform seconds)",
            ["namespace"],
            buckets=_DELAY_BUCKETS,
            registry=r,
        )
        self.run_seconds = Histogram(
            "awp_run_seconds",
            "Running time of succeeded attempts (platform seconds)",
            ["namespace"],
            buckets=_DELAY_BUCKETS,
            registry=r,
        )
        self.attempts = Counter("awp_attempts", "Ended attempts by end reason", ["outcome"], registry=r)
        self.admission_rejections = Counter(
            "awp_admission_rejections", "Rejected submissions", ["namespace", "reason"], registry=r
        )
        self.cycle_seconds = Histogram(
            "awp_scheduling_cycle_seconds",
            "Wall time of one scheduling cycle",
            buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1, 5),
            registry=r,
        )
        self.policy_failures = Counter("awp_policy_failures", "Policy failures", ["kind"], registry=r)
        self.policy_degraded = Gauge("awp_policy_degraded", "1 while the fallback policy decides", registry=r)
        self.stale_actions = Counter("awp_stale_actions", "Policy actions dropped as stale", registry=r)
        self.repairs = Counter("awp_reconcile_repairs", "Reconciliation repairs", ["rule"], registry=r)
        self.gpus_allocated = Gauge(
            "awp_gpus_allocated", "GPUs held by attempts that are not ENDED", ["namespace"], registry=r
        )
        self.gpus_capacity = Gauge("awp_gpus_capacity", "GPUs of ready nodes in the inventory", registry=r)
        self.backend_errors = Counter("awp_backend_errors", "Backend call errors", ["op"], registry=r)
        self.store_errors = Counter(
            "awp_store_errors", "Store errors (unavailable, failed transactions)", registry=r
        )
        self.leader = Gauge("awp_controller_leader", "1 while this controller holds the lease", registry=r)
        # make label-less series visible from the start
        self.policy_degraded.set(0)
        self.leader.set(0)
        self.gpus_capacity.set(0)

    @contextmanager
    def time_cycle(self) -> Iterator[None]:
        t0 = self.wall_timer()
        try:
            yield
        finally:
            self.cycle_seconds.observe(self.wall_timer() - t0)

    def exposition(self) -> bytes:
        return generate_latest(self.registry)


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, component, msg, and the ids passed in `extra={"fields": {...}}`."""

    def format(self, record: logging.LogRecord) -> str:
        obj: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "component": record.name.removeprefix("awp."),
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            obj.update(fields)
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(obj, default=str, separators=(",", ":"))


def setup_logging(level: str = "INFO", stream: Any = None) -> None:
    root = logging.getLogger("awp")
    root.handlers.clear()
    h = logging.StreamHandler(stream or sys.stderr)
    h.setFormatter(JsonFormatter())
    root.addHandler(h)
    root.setLevel(level.upper())
    root.propagate = False
