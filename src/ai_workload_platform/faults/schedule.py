"""Seeded fault schedules: a workload list, client actions, and faults, all drawn from the schedule seed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_workload_platform.controller.rules import Thresholds
from ai_workload_platform.crash import CRASH_POINTS
from ai_workload_platform.rng import categorical, randbelow, stream

# shorter thresholds so that a schedule fits in 600 virtual seconds (recorded in every schedule)
HARNESS_THRESHOLDS = Thresholds(
    start_retry_ms=2000,
    start_timeout_ms=20_000,
    node_grace_ms=5000,
    lost_grace_ms=6000,
    stop_retry_ms=3000,
    observe_interval_ms=1000,
    lease_ttl_ms=5000,
    policy_timeout_s=60.0,
    policy_max_failures=3,
)

FAULTS = {
    "local": [
        "crash",
        "crash",
        "crash",
        "zombie",
        "zombie",
        "store_tx_fail",
        "store_outage",
        "observe_down",
        "observe_stale",
        "observe_stale",
        "node_down",
        "attempt_crash",
        "attempt_lost",
        "start_fail",
        "stop_fail",
        "slow_start",
        "policy_fault",
        "race_cancel",
        "race_cancel",
        "preempt",
        "cap_change",
    ],
    "kube-fake": [
        "crash",
        "crash",
        "crash",
        "zombie",
        "zombie",
        "store_tx_fail",
        "store_outage",
        "observe_down",
        "observe_stale",
        "observe_stale",
        "node_down",
        "pod_evicted",
        "job_pending",
        "api_errors",
        "policy_fault",
        "race_cancel",
        "race_cancel",
        "preempt",
        "cap_change",
    ],
}
NAMESPACES = ("team-a", "team-b", "team-c")
MAX_WORKLOADS = 40
LAST_SUBMIT_MS = 300_000
LAST_FAULT_MS = 400_000


@dataclass
class Action:
    at_ms: int
    kind: str  # submit | cancel
    namespace: str
    key: str | None = None
    spec: dict[str, Any] | None = None
    target: int | None = None  # cancel: index of the logical submission (the workload is found by its key)


@dataclass
class Fault:
    at_ms: int
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    end_ms: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {"at_ms": self.at_ms, "kind": self.kind, "params": self.params, "end_ms": self.end_ms}


@dataclass
class Schedule:
    seed: int
    backend: str
    thresholds: Thresholds
    policy: str
    start_latency_ms: int
    stop_latency_ms: int
    actions: list[Action]
    faults: list[Fault]
    last_fault_ms: int
    settle_ms: int

    def describe(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "backend": self.backend,
            "policy": self.policy,
            "thresholds": self.thresholds.to_dict(),
            "start_latency_ms": self.start_latency_ms,
            "stop_latency_ms": self.stop_latency_ms,
            "workloads": len({a.key for a in self.actions if a.kind == "submit"}),
            "actions": len(self.actions),
            "faults": [f.to_json() for f in self.faults],
            "last_fault_ms": self.last_fault_ms,
            "settle_ms": self.settle_ms,
        }


def _u(r: Any, lo: float, hi: float) -> float:
    return lo + (hi - lo) * r.random()


def make_schedule(seed: int, backend: str = "local") -> Schedule:
    if backend not in FAULTS:
        raise ValueError(f"unknown backend {backend!r}")
    th = HARNESS_THRESHOLDS
    wr = stream(seed, "harness:workloads")
    fr = stream(seed, "faults")
    n = 20 + randbelow(wr, MAX_WORKLOADS - 20 + 1)
    times = sorted(randbelow(wr, LAST_SUBMIT_MS) for _ in range(n))
    actions: list[Action] = []
    total_runtime_ms = 0
    max_attempts_all = 1
    for i, t in enumerate(times):
        gpus = (1, 2, 4, 8)[categorical(wr, [0.35, 0.3, 0.2, 0.15])]
        workers = 2 if wr.random() < 0.15 else 1
        runtime = round(_u(wr, 5, 120), 3)
        max_attempts = 1 + randbelow(wr, 4)
        max_attempts_all = max(max_attempts_all, max_attempts)
        fatal = [2] if wr.random() < 0.5 else []
        sim: dict[str, Any] = {"runtime_s": runtime}
        if wr.random() < 0.25:
            sim.update(
                fail_attempts=1 + randbelow(wr, 2),
                fail_after_s=round(_u(wr, 1, runtime), 3),
                exit_code=2 if wr.random() < 0.2 else 1,
            )
        spec = {
            "priority": (1, 4, 8)[randbelow(wr, 3)],
            "gpus": gpus,
            "workers": workers,
            "topology": "rack" if workers > 1 and wr.random() < 0.5 else "any",
            "cpus": gpus * 12,
            "mem_gb": gpus * 96,
            "estimate_s": round(runtime * _u(wr, 0.5, 2.0), 3),
            "retry": {
                "max_attempts": max_attempts,
                "backoff_base_s": 1 + randbelow(wr, 3),
                "backoff_cap_s": 10 + randbelow(wr, 21),
                "jitter": "full" if wr.random() < 0.7 else "none",
                "fatal_exit_codes": fatal,
            },
            "labels": {"user": f"u{randbelow(wr, 4)}"},
            "preemptible": wr.random() < 0.5,
            "checkpoint_interval_s": (0, 5, 20)[randbelow(wr, 3)],
            "sim": sim,
        }
        ns = NAMESPACES[randbelow(wr, 3)]
        key = f"k{i}"
        actions.append(Action(t, "submit", ns, key, spec))
        total_runtime_ms += int(runtime * 1000) * max_attempts * workers
        if wr.random() < 0.25:  # the client repeats the same submission with the same key
            for _ in range(1 + randbelow(wr, 3)):
                actions.append(Action(t + randbelow(wr, 200_000), "submit", ns, key, spec))
        if wr.random() < 0.15:  # the owner cancels it later
            actions.append(Action(t + randbelow(wr, 200_000), "cancel", ns, key, None, i))
    faults: list[Fault] = []
    kinds = FAULTS[backend]
    for _ in range(1 + randbelow(fr, 6)):
        at = 5000 + randbelow(fr, LAST_FAULT_MS - 5000)
        kind = kinds[randbelow(fr, len(kinds))]
        p: dict[str, Any] = {}
        end = None
        if kind == "crash":
            p["point"] = CRASH_POINTS[randbelow(fr, len(CRASH_POINTS))]
        elif kind == "zombie":
            p["point"] = (
                "cycle.after_view",
                "cycle.after_view",
                "rules.before_write",
                "tick.after_lease",
                "start.after_commit",
            )[randbelow(fr, 5)]
            p["extra_ms"] = randbelow(fr, 30_001)  # how long the other controller works before the pause ends
            p["burst"] = randbelow(fr, 6)  # urgent submissions that arrive while the controller is paused
        elif kind == "store_tx_fail":
            p["count"] = 1 + randbelow(fr, 3)
        elif kind in ("store_outage", "observe_down", "policy_fault"):
            end = (
                at
                + 1000
                + randbelow(fr, {"store_outage": 9000, "observe_down": 20_000, "policy_fault": 30_000}[kind])
            )
            if kind == "policy_fault":
                p["failure"] = ("invalid", "error", "timeout", "crash")[randbelow(fr, 4)]
        elif kind == "observe_stale":
            end = at + 2000 + randbelow(fr, 18_000)
            p["age_ms"] = 200 + randbelow(fr, th.lost_grace_ms // 3 - 200 + 1)  # assumption A1
        elif kind == "node_down":
            p["node"] = ("r0-n00", "r0-n01", "r1-n00", "r1-n01")[randbelow(fr, 4)]
            end = at + 1000 + randbelow(fr, 40_000)
        elif kind in ("attempt_crash", "attempt_lost", "pod_evicted"):
            p["pick"] = fr.random()
        elif kind in ("start_fail", "stop_fail"):
            p["count"] = 1 + randbelow(fr, 3)
            p["effect"] = fr.random() < 0.5
        elif kind == "slow_start":
            p["workload"] = 1 + randbelow(fr, n)
            p["delay_ms"] = 1000 + randbelow(fr, 2 * th.start_timeout_ms)
        elif kind == "job_pending":
            p["workload"] = 1 + randbelow(fr, n)
        elif kind == "api_errors":
            p["count"] = 1 + randbelow(fr, 5)
        elif kind in ("race_cancel", "preempt"):
            p["pick"] = fr.random()
        elif kind == "cap_change":
            p["namespace"] = NAMESPACES[randbelow(fr, 3)]
            p["cap"] = (4, 8, 12, 16)[randbelow(fr, 4)]
            end = at + 5000 + randbelow(fr, 60_000)
        faults.append(Fault(at, kind, p, end))
    faults.sort(key=lambda f: (f.at_ms, f.kind))
    actions.sort(key=lambda a: a.at_ms)
    last_fault = max([f.end_ms or f.at_ms for f in faults] + [LAST_SUBMIT_MS])
    # settle bound: every attempt of every workload one after the other (a serial worst case), plus per
    # attempt the slowest repair path (start timeout, node and lost grace, stops) and the backoff cap
    per_attempt = th.start_timeout_ms + th.node_grace_ms + th.lost_grace_ms + 3 * th.stop_retry_ms + 30_000
    settle = total_runtime_ms + n * max_attempts_all * per_attempt + th.lease_ttl_ms + 120_000
    policy = (
        "fifo+first_fit",
        "priority+best_fit",
        "quota+first_fit",
        "priority+best_fit+preempt",
        "quota+first_fit+reclaim",
    )[randbelow(fr, 5)]
    return Schedule(
        seed,
        backend,
        th,
        policy,
        randbelow(fr, 501),
        randbelow(fr, 1001),
        actions,
        faults,
        last_fault,
        settle,
    )
