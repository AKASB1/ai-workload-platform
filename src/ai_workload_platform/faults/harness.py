"""The fault-injection harness: one seeded schedule on the virtual clock over an in-memory store and a
backend (local or the Kubernetes fake), with invariants checked after every step.

Reproduce a schedule: python -m ai_workload_platform faults --seed N --backend local
"""

from __future__ import annotations

import heapq
import logging
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ai_workload_platform import admission, crash
from ai_workload_platform.clock import VirtualClock
from ai_workload_platform.cluster import load_cluster
from ai_workload_platform.controller import Controller
from ai_workload_platform.faults.bugs import bug as bug_patch
from ai_workload_platform.faults.invariants import Checker, Violation
from ai_workload_platform.faults.schedule import Action, Schedule, make_schedule
from ai_workload_platform.models import TERMINAL_STATES, LeaseLost, PlatformError, StoreUnavailable
from ai_workload_platform.policy import PolicyFailure
from ai_workload_platform.policy.builtin import make_builtin
from ai_workload_platform.policy.runner import PolicyRunner
from ai_workload_platform.sim import events_jsonl, make_backend, reference_namespaces
from ai_workload_platform.store import ops
from ai_workload_platform.store.sql import Store

REF_PATH = Path(__file__).resolve().parents[3] / "configs" / "clusters" / "reference.json"
_REF = None


def reference_cluster() -> Any:
    global _REF
    if _REF is None:
        _REF = load_cluster(REF_PATH)
    return _REF


class PolicySwitch:
    failure: str | None = None
    force_preempt: float | None = (
        None  # one-shot: the next decision also preempts a running preemptible workload
    )


class FaultyPolicy:
    """Wraps a built-in policy; while a policy fault is active every call fails with its kind."""

    def __init__(self, inner: Any, switch: PolicySwitch) -> None:
        self.inner = inner
        self.name = inner.name
        self.switch = switch

    def hello(self, msg: dict[str, Any]) -> None:
        self.inner.hello(msg)

    def schedule(self, view: dict[str, Any]) -> dict[str, Any]:
        failure = self.switch.failure
        if failure == "invalid":
            return {
                "actions": [
                    {
                        "op": "start",
                        "job_id": "no-such-workload",
                        "placement": [{"node": "nowhere", "workers": 1}],
                    }
                ]
            }
        if failure:
            raise PolicyFailure(failure, f"injected policy {failure}")
        decision = self.inner.schedule(view)
        pick = self.switch.force_preempt
        if pick is not None:
            started = {a["job_id"] for a in decision["actions"]}
            done = {a["job_id"] for a in decision["actions"] if a["op"] == "preempt"}
            cands = sorted(
                r["job_id"]
                for r in view["running"]
                if r["preemptible"] and r["job_id"] not in started and r["job_id"] not in done
            )
            if cands:
                self.switch.force_preempt = None
                decision = {
                    **decision,
                    "actions": [
                        {"op": "preempt", "job_id": cands[int(pick * len(cands))]},
                        *decision["actions"],
                    ],
                }
        return decision

    def close(self) -> None:
        self.inner.close()


class StaleObserve:
    """Backend wrapper: while `age_ms` > 0, every other `observe()` returns a snapshot at least that old
    (a flapping, differently cached view; within assumption A1 when age <= lost_grace_ms / 3)."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.age_ms = 0
        self.calls = 0
        self.history: list[Any] = []

    def observe(self) -> Any:
        snap = self.backend.observe()
        self.history.append(snap)
        self.history = self.history[-64:]
        self.calls += 1
        if self.age_ms > 0 and self.calls % 2 == 1:
            old = [x for x in self.history if x.taken_ms <= snap.taken_ms - self.age_ms]
            if old:
                return old[-1]
        return snap

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)


class Injector:
    """Store faults: transactions that fail before they commit, and an outage window."""

    def __init__(self) -> None:
        self.fail_commits = 0
        self.outage = False

    def before_begin(self, write: bool) -> None:
        if self.outage:
            raise StoreUnavailable("injected store outage")

    def before_commit(self) -> None:
        if self.fail_commits > 0:
            self.fail_commits -= 1
            raise StoreUnavailable("injected transaction failure")


@dataclass
class Outcome:
    seed: int
    backend: str
    bug: str
    violations: list[dict[str, Any]]
    faults_injected: dict[str, int]
    crash_points_hit: dict[str, int]
    converge_ms: int | None
    end_ms: int
    last_fault_ms: int
    events: int
    ticks: int
    workloads: int
    terminal_states: dict[str, int]
    attempts_by_reason: dict[str, int]
    policy: str
    wall_s: float = 0.0
    log_tail: list[str] = field(default_factory=list)
    fault_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_json(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["wall_s"] = round(self.wall_s, 4)
        return d


class Harness:
    def __init__(self, sched: Schedule) -> None:
        self.s = sched
        self.th = sched.thresholds
        self.cluster = reference_cluster()
        self.clock = VirtualClock(0)
        self.store = Store(seed=sched.seed)
        self.injector = Injector()
        nss = reference_namespaces(self.cluster.total_gpus)
        for ns in nss:
            self.store.put_namespace(ns, 0)
        self.store.injector = self.injector
        self.backend, self.target = make_backend(
            sched.backend,
            self.cluster,
            self.clock,
            start_latency_ms=sched.start_latency_ms,
            stop_latency_ms=sched.stop_latency_ms,
            instance_id=self.store.instance_id,
        )
        self.switch = PolicySwitch()
        self.view = StaleObserve(self.backend)  # what the controller talks to
        self.ctl = self._controller("ctl-a")
        self.standby: Controller | None = None
        self.checker = Checker(
            self.store,
            self.target,
            "local" if sched.backend == "local" else "kube",
            self.cluster,
            {ns.name: ns.cap_gpus for ns in nss},
            self.th.lost_grace_ms,
            sched.seed,
        )
        self._seq = 0
        self.actions: list[tuple[int, int, Action]] = []
        for a in sched.actions:
            self._push(a.at_ms, a)
        self.fault_events: list[tuple[int, int, str, Any]] = []
        for i, f in enumerate(sched.faults):
            self.fault_events.append((f.at_ms, i, "start", f))
            if f.end_ms is not None:
                self.fault_events.append((f.end_ms, i, "end", f))
        self.fault_events.sort(key=lambda x: (x[0], x[1], x[2] == "start"))
        self.armed_crash: list[tuple[str, int]] = []
        self.armed_zombie: list[tuple[str, int]] = []
        self.armed_race: list[tuple[int, float]] = []
        self.injected: Counter[str] = Counter()
        self.points_hit: Counter[str] = Counter()
        self.in_zombie = False
        self.ticks = 0
        self.violations: list[Violation] = []
        self.holders = 0
        self.fenced = 0
        self.fault_log: list[dict[str, Any]] = []

    # --- helpers -----------------------------------------------------------------------------
    def _controller(self, holder: str) -> Controller:
        runner = PolicyRunner(
            FaultyPolicy(make_builtin(self.s.policy), self.switch), max_failures=self.th.policy_max_failures
        )
        return Controller(self.store, self.view, runner, self.cluster, holder=holder, thresholds=self.th)

    def _push(self, at_ms: int, a: Action) -> None:
        self._seq += 1
        heapq.heappush(self.actions, (at_ms, self._seq, a))

    @contextmanager
    def _bypass(self) -> Any:
        """Harness bookkeeping and invariant checks read past the injected store faults."""
        saved = self.store.injector
        self.store.injector = None
        try:
            yield
        finally:
            self.store.injector = saved

    def _wid_for_key(self, ns: str, key: str) -> str | None:
        with self._bypass(), self.store.read() as tx:
            got = tx.idempotency(ns, key)
        return got[1] if got else None

    # --- client actions --------------------------------------------------------------------------
    def _client(self, now: int) -> None:
        while self.actions and self.actions[0][0] <= now:
            _t, _s, a = heapq.heappop(self.actions)
            try:
                if a.kind == "submit":
                    admission.submit(self.store, a.namespace, a.spec, a.key, now)
                else:
                    wid = self._wid_for_key(a.namespace, a.key or "")
                    if wid is not None:
                        ops.request_cancel(self.store, a.namespace, wid, now)
            except StoreUnavailable:
                self._push(now + 1000, a)  # the client retries later, with the same key
            except crash.Crash:
                self._push(now + 500, a)  # the reply was lost: the client retries with the same key
                raise
            except PlatformError:
                pass  # a rejection (4xx): the client gives up

    # --- faults ----------------------------------------------------------------------------------
    def _faults(self, now: int) -> None:
        while self.fault_events and self.fault_events[0][0] <= now:
            _t, _i, phase, f = self.fault_events.pop(0)
            p = f.params
            k = f.kind
            t = self.target
            if phase == "start" and k not in ("crash", "zombie", "race_cancel"):
                self.injected[k] += 1
            if k not in ("crash", "zombie", "race_cancel"):
                self.fault_log.append({"at_ms": now, "kind": k, "phase": phase, "params": p})
            if k == "crash" and phase == "start":
                self.armed_crash.append((p["point"], now))
            elif k == "zombie" and phase == "start":
                self.armed_zombie.append((p["point"], p.get("extra_ms", 0), p.get("burst", 0)))
            elif k == "race_cancel" and phase == "start":
                self.armed_race.append((now, p["pick"]))
            elif k == "store_tx_fail":
                self.injector.fail_commits += p["count"]
            elif k == "store_outage":
                self.injector.outage = phase == "start"
            elif k == "observe_down":
                if self.s.backend == "local":
                    t.faults.observe_down = phase == "start"
                else:
                    t.faults.down_ops = {"list_jobs", "list_pods"} if phase == "start" else set()
            elif k == "observe_stale":
                self.view.age_ms = p["age_ms"] if phase == "start" else 0
            elif k == "node_down":
                (t.node_down if phase == "start" else t.node_up)(p["node"])
            elif k == "attempt_crash":
                ids = sorted(t.active_ids())
                if ids:
                    t.crash(ids[int(p["pick"] * len(ids))])
            elif k == "attempt_lost":
                ids = sorted(t.known_ids())
                if ids:
                    t.lose(ids[int(p["pick"] * len(ids))])
            elif k == "pod_evicted":
                pods = sorted(
                    q.name
                    for q in t.pods.values()
                    if q.phase == "Running" and not q.deleting and not q.frozen
                )
                if pods:
                    t.evict(pods[int(p["pick"] * len(pods))])
            elif k == "start_fail":
                t.faults.fail_start += p["count"]
                t.faults.fail_start_effect = p["effect"]
            elif k == "stop_fail":
                t.faults.fail_stop += p["count"]
                t.faults.fail_stop_effect = p["effect"]
            elif k == "slow_start":
                t.faults.slow_start_ms[f"w{p['workload']}-a1"] = p["delay_ms"]
            elif k == "job_pending":
                t.faults.pending_jobs.add(f"w{p['workload']}-a1-0")
            elif k == "api_errors":
                t.faults.api_errors += p["count"]
            elif k == "preempt":
                self.switch.force_preempt = p["pick"]
            elif k == "policy_fault":
                self.switch.failure = p["failure"] if phase == "start" else None
            elif k == "cap_change":
                with self._bypass():
                    ns = self.store.get_namespace(p["namespace"])
                    assert ns is not None
                    cap = p["cap"] if phase == "start" else self.cluster.total_gpus
                    self.store.put_namespace(
                        replace(ns, cap_gpus=cap, quota_gpus=min(ns.quota_gpus, cap)), now
                    )
                self.checker.set_cap(p["namespace"], cap, now)

    def hook(self, name: str, subject: str | None = None) -> None:
        if self.in_zombie:
            return
        now = self.clock.now_ms()
        for i, (point, _at) in enumerate(self.armed_crash):
            if point == name:
                del self.armed_crash[i]
                self.points_hit[name] += 1
                self.injected["crash"] += 1
                self.fault_log.append(
                    {"at_ms": now, "kind": "crash", "phase": "start", "params": {"point": name}}
                )
                raise crash.Crash(name)
        for i, (point, extra, burst) in enumerate(self.armed_zombie):
            if point == name:
                del self.armed_zombie[i]
                self.injected["zombie"] += 1
                self.fault_log.append(
                    {"at_ms": now, "kind": "zombie", "phase": "start", "params": {"point": name}}
                )
                self._zombie(now, extra, burst)
                return
        if name in ("rules.before_write", "cycle.after_view") and self.armed_race:
            _at, pick = self.armed_race.pop(0)
            with self._bypass():
                live = sorted(
                    (w for w in self.store.all_workloads() if w.state not in TERMINAL_STATES),
                    key=lambda x: x.id,
                )
            if live:
                # half of the races cancel exactly the workload being written (the interesting interleaving)
                target = [w for w in live if w.id == subject] if pick < 0.5 else []
                w = target[0] if target else live[int(pick * len(live))]
                self.injected["race_cancel"] += 1
                try:
                    ops.request_cancel(self.store, w.namespace, w.id, now)
                except StoreUnavailable:
                    pass

    def _zombie(self, now: int, extra_ms: int, burst: int = 0) -> None:
        """The current controller pauses beyond the lease TTL; another takes over and works for a while;
        then the paused one continues its tick (its writes must fail with LeaseLost)."""
        self.in_zombie = True
        try:
            self.holders += 1
            b = self._controller(f"ctl-z{self.holders}")
            t = now + self.th.lease_ttl_ms + 1
            until = t + extra_ms
            for j in range(
                burst
            ):  # urgent work keeps arriving through the API while the controller is paused
                self._push(
                    now + 1 + j,
                    Action(
                        now + 1 + j,
                        "submit",
                        ("team-a", "team-b", "team-c")[j % 3],
                        f"burst-{now}-{j}",
                        {
                            "priority": 9,
                            "gpus": 8,
                            "cpus": 96,
                            "mem_gb": 768,
                            "estimate_s": 60,
                            "sim": {"runtime_s": 60},
                        },
                    ),
                )
            while True:
                self.clock.advance_to(max(t, self.clock.now_ms()))
                cur = self.clock.now_ms()
                self._faults(cur)
                try:
                    b.tick(cur, pre_cycle=lambda cur=cur: self._client(cur))
                except (StoreUnavailable, LeaseLost):
                    pass
                self._check()
                cands = [
                    x
                    for x in (
                        b.next_wakeup_ms(),
                        self.backend.next_event_ms(),
                        self.actions[0][0] if self.actions else None,
                        self.fault_events[0][0] if self.fault_events else None,
                    )
                    if x is not None
                ]
                t = max(min(cands), cur + 1) if cands else cur + 1000
                if t > until:
                    break
            self.standby = b
        finally:
            self.in_zombie = False

    def _check(self) -> None:
        with self._bypass():
            self.violations.extend(self.checker.check(self.clock.now_ms()))

    def _backend_known(self) -> set[str]:
        if self.s.backend == "local":
            return set(self.target.known_ids())
        return {j["metadata"]["labels"].get("awp.local/attempt", "") for j in self.target.jobs.values()} | {
            p.labels.get("awp.local/attempt", "") for p in self.target.pods.values()
        }

    def _done(self) -> bool:
        if self.actions or self.fault_events or self.clock.now_ms() < self.s.last_fault_ms:
            return False
        with self._bypass():
            if any(w.state not in TERMINAL_STATES for w in self.store.all_workloads()):
                return False
        return not self._backend_known()

    # --- main loop -------------------------------------------------------------------------------
    def run(self) -> Outcome:
        t0 = time.perf_counter()
        prev = crash.install(self.hook)
        logging.getLogger("awp").setLevel(logging.ERROR)
        converge = None
        try:
            while True:
                now = self.clock.now_ms()
                self._faults(now)
                try:
                    self.ctl.tick(now, pre_cycle=lambda now=now: self._client(now))
                except crash.Crash:
                    self.ctl.close()
                    self.ctl = self._controller(self.ctl.holder)  # a new controller object with no memory
                    continue
                except LeaseLost:
                    self.fenced += 1  # the paused controller was fenced out; the other one continues below
                except StoreUnavailable:
                    pass  # a store error ends the tick; the next one starts from the store again
                try:
                    self._client(self.clock.now_ms())  # the API works even when the tick failed
                except crash.Crash:
                    self.ctl.close()
                    self.ctl = self._controller(self.ctl.holder)
                if self.standby is not None:
                    self.ctl.close()
                    self.ctl, self.standby = self.standby, None
                self.ticks += 1
                self._check()
                if self.violations:
                    break
                now = self.clock.now_ms()
                if self._done():
                    converge = max(0, now - self.s.last_fault_ms)
                    break
                cands = [
                    x
                    for x in (
                        self.ctl.next_wakeup_ms(),
                        self.backend.next_event_ms(),
                        self.actions[0][0] if self.actions else None,
                        self.fault_events[0][0] if self.fault_events else None,
                    )
                    if x is not None
                ]
                if not cands:
                    break  # nothing can change any more: the final check decides
                nxt = max(min(cands), now + 1)
                if nxt > self.s.last_fault_ms + self.s.settle_ms:
                    break
                self.clock.advance_to(nxt)
            if not self.violations and converge is None:
                with self._bypass():
                    self.violations.extend(self.checker.final(self.clock.now_ms(), self._backend_known()))
        finally:
            crash.install(prev)
            self.ctl.close()
        with self._bypass():
            evs = self.store.events()
            ws = self.store.all_workloads()
            ats = self.store.all_attempts()
        tail = []
        if self.violations:
            tail = events_jsonl(evs[-25:]).decode().splitlines()
        out = Outcome(
            seed=self.s.seed,
            backend=self.s.backend,
            bug="none",
            violations=[v.to_json() for v in self.violations],
            faults_injected=dict(sorted(self.injected.items())),
            crash_points_hit=dict(sorted(self.points_hit.items())),
            converge_ms=converge,
            end_ms=self.clock.now_ms(),
            last_fault_ms=self.s.last_fault_ms,
            events=len(evs),
            ticks=self.ticks,
            workloads=len(ws),
            terminal_states=dict(sorted(Counter(w.state.value for w in ws).items())),
            attempts_by_reason=dict(sorted(Counter(a.end_reason for a in ats if a.end_reason).items())),
            policy=self.s.policy,
            wall_s=time.perf_counter() - t0,
            log_tail=tail,
            fault_log=self.fault_log,
        )
        self.log_bytes = events_jsonl(evs)
        self.store.close()
        return out


def run_schedule(seed: int, backend: str = "local", bug: str | None = None) -> Outcome:
    """Run one schedule (optionally under an injected bug). Picklable entry point for worker processes."""
    with bug_patch(bug):
        h = Harness(make_schedule(seed, backend))
        out = h.run()
    out.bug = bug or "none"
    return out


def schedule_log(seed: int, backend: str = "local") -> bytes:
    """The event log of one schedule (determinism checks)."""
    h = Harness(make_schedule(seed, backend))
    h.run()
    return h.log_bytes
