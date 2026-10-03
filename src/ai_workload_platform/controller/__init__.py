"""The controller core: `tick(now_ms)` = lease, inventory, observe, rules R6..R7, one scheduling cycle.

Domain decisions live in `controller/rules.py`, `policy/`, and `store/ops.py`; this module wires them
to the store and the backend. It keeps only memory that a fresh controller may lose (last start and
stop calls, the last snapshot that showed an attempt); everything else is read from the store.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ai_workload_platform.cluster import ClusterConfig
from ai_workload_platform.controller import rules
from ai_workload_platform.controller.rules import Thresholds
from ai_workload_platform.crash import crashpoint
from ai_workload_platform.models import (
    TERMINAL_PHASES,
    Attempt,
    AttemptRequest,
    AttemptState,
    BackendError,
    EndReason,
    InvalidTransition,
    LeaseLost,
    Phase,
    Snapshot,
    StoreUnavailable,
    VersionConflict,
    Workload,
    WorkloadState,
)
from ai_workload_platform.policy.runner import PolicyRunner
from ai_workload_platform.policy.validate import PreemptAction, validate_decision
from ai_workload_platform.policy.view import build_view
from ai_workload_platform.scheduler import SchedulerAdapter
from ai_workload_platform.store import ops
from ai_workload_platform.store.sql import Fence, Inventory, Store

log = logging.getLogger("awp.controller")


@dataclass
class TickResult:
    leader: bool = False
    inventory_ok: bool = False
    observe_ok: bool = False
    repairs: dict[str, int] = field(default_factory=dict)
    started: int = 0
    stale: int = 0


class Controller:
    def __init__(
        self,
        store: Store,
        backend: SchedulerAdapter,
        runner: PolicyRunner,
        cluster: ClusterConfig,
        *,
        holder: str = "controller",
        thresholds: Thresholds | None = None,
        metrics: Any = None,
        clock: Any = None,
    ) -> None:
        self.store = store
        self.backend = backend
        self.clock = clock  # live service only: renew the lease after a slow policy decision
        self.runner = runner
        self.cluster = cluster
        self.holder = holder
        self.th = thresholds or Thresholds()
        self.metrics = metrics
        self.seed = store.seed
        self.epoch: int | None = None
        self.start_ms: int | None = None
        self.last_start_call: dict[str, int] = {}
        self.last_stop_call: dict[str, int] = {}
        self.last_seen: dict[str, int] = {}
        self.last_snapshot: Snapshot | None = None
        self.inventory: Inventory | None = None
        self._wake: int | None = None
        self._redecide = False
        self.counters: dict[str, int] = {}

    @property
    def fence(self) -> Fence:
        assert self.epoch is not None
        return Fence(self.holder, self.epoch)

    def _count(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n

    def _backend_error(self, op: str, e: Exception) -> None:
        self._count(f"backend_error:{op}")
        if self.metrics is not None:
            self.metrics.backend_errors.labels(op=op).inc()
        log.debug("backend error", extra={"fields": {"op": op, "error": str(e)}})

    def _repair(self, rule: str, res: TickResult) -> None:
        res.repairs[rule] = res.repairs.get(rule, 0) + 1
        self._count(f"repair:{rule}")
        if self.metrics is not None:
            self.metrics.repairs.labels(rule=rule).inc()

    # --- the tick ------------------------------------------------------------------------------
    def tick(self, now_ms: int, pre_cycle: Callable[[], None] | None = None) -> TickResult:
        """One pass. Raises LeaseLost when this controller was fenced out (it must stop); a store error ends the
        tick (and is raised), and the controller asks to tick again one observe interval later."""
        try:
            return self._tick(now_ms, pre_cycle)
        except StoreUnavailable:
            self._wake = now_ms + self.th.observe_interval_ms
            raise

    def _tick(self, now_ms: int, pre_cycle: Callable[[], None] | None = None) -> TickResult:
        res = TickResult()
        self._redecide = False
        if self.start_ms is None:
            self.start_ms = now_ms
        try:
            self.epoch = self.store.acquire_lease(self.holder, self.epoch, now_ms, self.th.lease_ttl_ms)
        except LeaseLost as e:
            if self.metrics is not None:
                self.metrics.leader.set(0)
            if self.epoch is None and e.details.get("standby"):
                if pre_cycle:
                    pre_cycle()
                return res  # a standby: another controller holds the lease
            raise
        res.leader = True
        if self.metrics is not None:
            self.metrics.leader.set(1)
        crashpoint("tick.after_lease")
        nodes = None
        try:
            nodes = self.backend.inventory()
            res.inventory_ok = True
        except BackendError as e:
            self._backend_error("inventory", e)
        if nodes is not None:
            self.inventory = self.store.save_inventory(nodes, now_ms, self.fence)
        snap = None
        try:
            snap = self.backend.observe()
            res.observe_ok = True
        except BackendError as e:
            self._backend_error("observe", e)
        if snap is not None:
            self.last_snapshot = snap
        self._rules(now_ms, snap, res)
        if pre_cycle:
            pre_cycle()
        if res.inventory_ok and res.observe_ok:
            self._cycle(now_ms, res)
        self._wake = self._compute_wake(now_ms)
        return res

    # --- reconciliation ------------------------------------------------------------------------
    def _read_state(self) -> tuple[dict[str, Workload], dict[str, Attempt]]:
        with self.store.read() as tx:
            ws = {w.id: w for w in self.store.live_workloads(tx)}
            ats = {a.id: a for a in self.store.open_attempts(tx)}
        return ws, ats

    def _write(self, fn: Callable[[], Any], subject: str | None = None) -> Any:
        """Run one repair transaction; a VersionConflict means re-read and re-decide next tick."""
        crashpoint("rules.before_write", subject)
        try:
            return fn()
        except VersionConflict:
            self._count("version_conflict")
            self._redecide = True
            return None
        except InvalidTransition as e:
            self._count("invalid_transition")
            log.warning("invalid transition in a repair", extra={"fields": {"error": str(e)}})
            return None

    def _overhead_s(self, w: Workload) -> float:
        """Every start after a preemption first holds its GPUs for restart_overhead_s without progress."""
        return float(self.cluster.restart_overhead_s) if w.preemptions > 0 else 0.0

    def _request(self, w: Workload, a: Attempt) -> AttemptRequest:
        return AttemptRequest(
            a.id,
            w.id,
            w.namespace,
            a.n,
            w.spec,
            tuple((p["node"], int(p["workers"])) for p in a.placement),
            retained_s=w.retained_ms / 1000.0,
            restart_overhead_s=self._overhead_s(w),
        )

    def _check_lease(self) -> None:
        """Backend calls are fenced too: a controller that lost the lease (paused beyond the TTL) must not start,
        stop, or forget anything the new leader may already have decided about."""
        holder, epoch, _expires = self.store.lease()
        if holder != self.holder or epoch != self.epoch:
            raise LeaseLost(
                f"{self.holder} lost the lease before a backend call", {"holder": holder, "epoch": epoch}
            )

    def _stop_call(self, aid: str, now_ms: int) -> None:
        try:
            self._check_lease()
            self.backend.stop(aid)
            self.last_stop_call[aid] = now_ms
        except BackendError as e:
            self._backend_error("stop", e)
            self.last_stop_call[aid] = now_ms

    def _forget(self, aid: str) -> None:
        try:
            self._check_lease()
            self.backend.forget(aid)
        except BackendError as e:
            self._backend_error("forget", e)
        self.last_seen.pop(aid, None)
        self.last_start_call.pop(aid, None)
        self.last_stop_call.pop(aid, None)

    def _end(
        self,
        w: Workload,
        a: Attempt,
        reason: str,
        s: Any,
        now_ms: int,
        res: TickResult,
        rule: str,
        crash_point: str | None = None,
    ) -> tuple[Workload, Attempt] | None:
        out = self._write(
            lambda: ops.commit_attempt_ended(
                self.store,
                self.fence,
                w,
                a,
                reason,
                s.exit_code if s is not None else None,
                s.started_ms if s is not None and s.started_ms is not None else a.observed_started_ms,
                s.ended_ms if s is not None else None,
                list(s.nodes) if s is not None and s.nodes else None,
                now_ms,
                self.seed,
                crash_point=crash_point,
                work_done_s=s.work_done_s if s is not None else None,
            ),
            w.id,
        )
        if out is None:
            return None
        nw, na, _ev = out
        self._repair(rule, res)
        if self.metrics is not None:
            self.metrics.attempts.labels(outcome=reason).inc()
            if reason == EndReason.SUCCEEDED and na.running_ms is not None:
                self.metrics.run_seconds.labels(namespace=w.namespace).observe(
                    (na.ended_ms - na.running_ms) / 1000
                )
        return nw, na

    def _rules(self, now_ms: int, snap: Snapshot | None, res: TickResult) -> None:
        ws, ats = self._read_state()
        if snap is None:
            self._r7(ws, now_ms, res)  # the retry release needs no observation
            return
        by_id = snap.by_id()
        for aid in by_id:
            if aid in ats:
                self.last_seen[aid] = max(self.last_seen.get(aid, -1), snap.taken_ms)
        inv = self.inventory
        not_ready = inv.not_ready_since if inv is not None else {}
        ended: set[str] = set()

        # R6 observed progress
        for aid in sorted(ats):
            a, s = ats[aid], by_id.get(aid)
            w = ws.get(a.workload_id)
            if s is None or w is None:
                continue
            if a.state == AttemptState.STARTING and (s.phase == Phase.RUNNING or s.phase in TERMINAL_PHASES):
                out = self._write(
                    lambda w=w, a=a, s=s: ops.commit_running(
                        self.store,
                        self.fence,
                        w,
                        a,
                        s.started_ms,
                        list(s.nodes),
                        now_ms,
                        list(s.workers_by_node),
                    ),
                    w.id,
                )
                if out is None:
                    continue
                w, a, _ = out
                ws[w.id], ats[aid] = w, a
                self._repair("R6", res)
            if s.phase in TERMINAL_PHASES:
                reason = rules.terminal_reason(a, s, list(w.spec["retry"]["fatal_exit_codes"]))
                cp = "stopped.before_commit" if s.phase == Phase.STOPPED else None
                out2 = self._end(w, a, reason, s, now_ms, res, "R6", crash_point=cp)
                if out2 is None:
                    continue
                ws[w.id], ats[aid] = out2
                ended.add(aid)
                crashpoint("ended.before_forget")
                self._forget(aid)

        # R5 lost attempt
        for aid in sorted(ats):
            a = ats[aid]
            if aid in ended or aid in by_id or a.state not in (AttemptState.RUNNING, AttemptState.STOPPING):
                continue
            since = rules.lost_since(
                self.last_seen.get(aid),
                a.state_since_ms,
                self.start_ms if self.start_ms is not None else now_ms,
            )
            if rules.r5_due(now_ms, since, self.th.lost_grace_ms):
                w = ws.get(a.workload_id)
                if w is None:
                    continue
                out2 = self._end(w, a, a.stop_reason or EndReason.BACKEND_LOST.value, None, now_ms, res, "R5")
                if out2 is not None:
                    ws[w.id], ats[aid] = out2
                    ended.add(aid)
                    self.last_seen.pop(aid, None)  # no forget: R3 stops and forgets leftovers

        # R4 node loss
        for aid in sorted(ats):
            a = ats[aid]
            if aid in ended or a.state not in (AttemptState.STARTING, AttemptState.RUNNING):
                continue
            s = by_id.get(aid)
            if s is not None and s.nodes:
                nodes = set(s.nodes)
            else:
                nodes = set(a.observed_nodes) or {p["node"] for p in a.placement}
            if any(rules.r4_due(now_ms, not_ready.get(n), self.th.node_grace_ms) for n in sorted(nodes)):
                self._stop_request(ws, ats, aid, EndReason.NODE_LOST.value, now_ms, res, "R4")

        # R3 orphans: in the snapshot, but unknown to the store or ENDED there
        for aid in sorted(by_id):
            if aid in ended or (aid in ats and ats[aid].state != AttemptState.ENDED):
                continue
            if by_id[aid].phase in TERMINAL_PHASES:
                self._forget(aid)
                self._repair("R3", res)
            elif rules.r8_restop_due(now_ms, self.last_stop_call.get(aid), self.th.stop_retry_ms):
                self._stop_call(aid, now_ms)
                self._repair("R3", res)

        # R2 start timeout
        for aid in sorted(ats):
            a = ats[aid]
            if aid in ended or a.state != AttemptState.STARTING:
                continue
            if rules.r2_due(now_ms, a.started_ms, self.th.start_timeout_ms):
                self._stop_request(ws, ats, aid, EndReason.START_TIMEOUT.value, now_ms, res, "R2")

        # R1 lost start (absent from the snapshot, or a gang whose start left only part of it behind)
        for aid in sorted(ats):
            a = ats[aid]
            if (
                aid in ended
                or a.state != AttemptState.STARTING
                or (aid in by_id and not by_id[aid].incomplete)
            ):
                continue
            if rules.r1_due(now_ms, a.started_ms, self.last_start_call.get(aid), self.th.start_retry_ms):
                w = ws.get(a.workload_id)
                if w is None:
                    continue
                self.last_start_call[aid] = now_ms
                try:
                    self._check_lease()
                    self.backend.start(self._request(w, a))
                except BackendError as e:
                    self._backend_error("start", e)
                self._repair("R1", res)

        # R8 stops: cancel requests, then repeated stops
        for aid in sorted(ats):
            a = ats[aid]
            w = ws.get(a.workload_id)
            if aid in ended or w is None:
                continue
            if w.cancel_requested and a.state in (AttemptState.STARTING, AttemptState.RUNNING):
                self._stop_request(ws, ats, aid, EndReason.CANCELLED.value, now_ms, res, "R8")
            elif a.state == AttemptState.STOPPING:
                s = by_id.get(aid)
                if (s is None or s.phase not in TERMINAL_PHASES) and rules.r8_restop_due(
                    now_ms, self.last_stop_call.get(aid), self.th.stop_retry_ms
                ):
                    self._stop_call(aid, now_ms)
                    self._repair("R8", res)

        self._r7(ws, now_ms, res)

    def _stop_request(
        self,
        ws: dict[str, Workload],
        ats: dict[str, Attempt],
        aid: str,
        reason: str,
        now_ms: int,
        res: TickResult,
        rule: str,
    ) -> None:
        a = ats[aid]
        w = ws.get(a.workload_id)
        if w is None:
            return
        out = self._write(
            lambda: ops.commit_stop_requested(self.store, self.fence, w, a, reason, now_ms), w.id
        )
        if out is None:
            return
        ws[w.id], ats[aid], _ = out
        self._repair(rule, res)
        crashpoint("stop.before_call")
        self._stop_call(aid, now_ms)

    def _r7(self, ws: dict[str, Workload], now_ms: int, res: TickResult) -> None:
        for wid in sorted(ws):
            w = ws[wid]
            if w.state == WorkloadState.RETRY_WAIT and w.retry_at_ms is not None and w.retry_at_ms <= now_ms:
                out = self._write(lambda w=w: ops.commit_requeued(self.store, self.fence, w, now_ms), w.id)
                if out is not None:
                    ws[wid] = out[0]
                    self._repair("R7", res)

    # --- scheduling cycle ----------------------------------------------------------------------
    def _cycle(self, now_ms: int, res: TickResult) -> None:
        inv = self.inventory
        assert inv is not None
        with self.store.read() as tx:
            ws = {w.id: w for w in self.store.live_workloads(tx)}
            if not any(w.state == WorkloadState.QUEUED for w in ws.values()):
                return
            ats = self.store.open_attempts(tx)
            namespaces = self.store.namespaces(tx)
            books = self.store.books(tx)
            gpu_ms = {
                r[0]: int(r[1] or 0)
                for r in tx.q(
                    "SELECT namespace, SUM(gpus * workers * (COALESCE(ended_ms, ?) - started_ms)) FROM attempts "
                    "GROUP BY namespace",
                    (now_ms,),
                )
            }
        history: list[dict[str, Any]] = []  # filled per policy session by the runner (_history)
        used: dict[str, list[int]] = {}
        alloc: dict[str, int] = {}
        for (ns, node), (g, c, m) in books.items():
            u = used.setdefault(node, [0, 0, 0])
            u[0] += g
            u[1] += c
            u[2] += m
            alloc[ns] = alloc.get(ns, 0) + g
        rates = {}
        if self.last_snapshot is not None:
            rates = {s.attempt_id: s.rate for s in self.last_snapshot.attempts if s.rate}
        view = build_view(
            now_ms,
            inv.nodes,
            {n: (v[0], v[1], v[2]) for n, v in used.items()},
            ws,
            ats,
            namespaces,
            alloc,
            gpu_ms,
            rates,
            history,
            restart_overhead_s=float(self.cluster.restart_overhead_s),
        )
        hello = {"cluster": self.cluster.hello_cluster(inv.nodes)}
        caps = {ns.name: (ns.cap_gpus, alloc.get(ns.name, 0)) for ns in namespaces}
        crashpoint("cycle.after_view", view["pending"][0]["job_id"] if view["pending"] else None)
        ctx = self.metrics.time_cycle() if self.metrics is not None else _null()
        with ctx:
            actions = self.runner.decide(
                view,
                hello,
                now_ms,
                lambda d: validate_decision(d, view, inv.nodes, caps),
                history=self._history,
            )
        if self.clock is not None:  # a policy may take up to policy_timeout_s, longer than the lease TTL
            self.epoch = self.store.acquire_lease(
                self.holder, self.epoch, self.clock.now_ms(), self.th.lease_ttl_ms
            )
        # the capacity the books really have now: a preemption frees nothing until its attempt has ended (R9)
        actual = {
            n.name: [
                n.gpus - used.get(n.name, [0, 0, 0])[0],
                n.cpus - used.get(n.name, [0, 0, 0])[1],
                n.mem_mb - used.get(n.name, [0, 0, 0])[2],
            ]
            for n in inv.nodes
            if n.ready
        }
        open_by_w = {
            a.workload_id: a for a in ats if a.state in (AttemptState.STARTING, AttemptState.RUNNING)
        }
        for act in actions:
            w = ws[act.workload_id]
            if isinstance(act, PreemptAction):
                a = open_by_w.get(w.id)
                if a is None:
                    continue
                out = self._write(
                    lambda w=w, a=a: ops.commit_stop_requested(
                        self.store, self.fence, w, a, EndReason.PREEMPTED.value, now_ms
                    ),
                    w.id,
                )
                if out is None:
                    res.stale += 1
                    continue
                ws[w.id], open_by_w[w.id], _ = out
                self._count("preempt")
                crashpoint("stop.before_call")
                self._stop_call(a.id, now_ms)
                continue
            need = [(n, k) for n, k in act.placement]
            if any(
                n not in actual
                or actual[n][0] < w.gpus * k
                or actual[n][1] < w.cpus * k
                or actual[n][2] < w.mem_mb * k
                for n, k in need
            ):
                self._count("deferred_start")  # waits for a preempted attempt to end; re-decided next cycle
                continue
            try:
                nw, a, _ev = ops.commit_started(
                    self.store, self.fence, w, act.placement_dicts(), now_ms, overhead_s=self._overhead_s(w)
                )
            except VersionConflict:
                res.stale += 1
                self._count("stale_action")
                if self.metrics is not None:
                    self.metrics.stale_actions.inc()
                continue
            for n, k in need:
                actual[n][0] -= w.gpus * k
                actual[n][1] -= w.cpus * k
                actual[n][2] -= w.mem_mb * k
            res.started += 1
            if self.metrics is not None and a.n == 1:
                self.metrics.queue_delay.labels(namespace=w.namespace).observe(
                    (a.started_ms - w.submit_ms) / 1000
                )
            crashpoint("start.after_commit")
            self.last_start_call[a.id] = now_ms
            try:
                self._check_lease()
                self.backend.start(self._request(nw, a))
            except BackendError as e:
                self._backend_error("start", e)
            crashpoint("start.after_call")

    def _history(self, after_seq: int) -> tuple[list[dict[str, Any]], int]:
        """Attempts that ended `succeeded` after log position `after_seq`: user, submit time, measured run time."""
        with self.store.read() as tx:
            rows = tx.q(
                "SELECT e.seq, a.workload_id, w.namespace, w.spec, w.submit_ms, a.observed_started_ms, "
                "a.observed_ended_ms, a.running_ms, a.ended_ms FROM events e JOIN attempts a ON a.id = e.attempt_id "
                "JOIN workloads w ON w.id = a.workload_id WHERE e.type = 'attempt_ended' AND e.seq > ? "
                "AND a.end_reason = 'succeeded' ORDER BY e.seq",
                (after_seq,),
            )
        out, upto = [], after_seq
        for r in rows:
            spec = json.loads(r[3])
            if r[5] is not None and r[6] is not None:
                run_ms = r[6] - r[5]
            else:
                run_ms = 0 if r[8] is None else r[8] - (r[7] if r[7] is not None else r[8])
            out.append(
                {
                    "job_id": r[1],
                    "user": spec.get("labels", {}).get("user") or r[2],
                    "submit_s": round(r[4] / 1000, 3),
                    "runtime_s": round(max(0, run_ms) / 1000, 3),
                }
            )
            upto = max(upto, int(r[0]))
        return out, upto

    # --- deadlines -----------------------------------------------------------------------------
    def _compute_wake(self, now_ms: int) -> int | None:
        ws, ats = self._read_state()
        seen = self.last_snapshot.by_id() if self.last_snapshot is not None else {}
        cands: list[int] = []
        for w in ws.values():
            if w.state == WorkloadState.RETRY_WAIT and w.retry_at_ms is not None:
                cands.append(w.retry_at_ms)
        not_ready = self.inventory.not_ready_since if self.inventory is not None else {}
        start = self.start_ms if self.start_ms is not None else now_ms
        for a in ats.values():
            if a.state == AttemptState.STARTING:
                cands.append(a.started_ms + self.th.start_timeout_ms)
                # poll while a start is in flight, as the live driver does every observe_interval_ms
                cands.append(now_ms + self.th.observe_interval_ms)
                if a.id not in seen or seen[a.id].incomplete:
                    last = self.last_start_call.get(a.id)
                    cands.append(
                        max(a.started_ms, last if last is not None else a.started_ms) + self.th.start_retry_ms
                    )
            if a.state in (AttemptState.RUNNING, AttemptState.STOPPING) and a.id not in seen:
                cands.append(
                    rules.lost_since(self.last_seen.get(a.id), a.state_since_ms, start)
                    + self.th.lost_grace_ms
                )
            if a.state == AttemptState.STOPPING:
                last = self.last_stop_call.get(a.id)
                cands.append((last if last is not None else now_ms) + self.th.stop_retry_ms)
            for p in a.placement:
                if p["node"] in not_ready:
                    cands.append(not_ready[p["node"]] + self.th.node_grace_ms)
        for s in seen.values():
            if s.attempt_id not in ats and s.phase not in TERMINAL_PHASES:
                last = self.last_stop_call.get(s.attempt_id)
                cands.append((last if last is not None else now_ms) + self.th.stop_retry_ms)
        # work the last snapshot shows but the store has not recorded yet (a write lost a race and must be
        # re-decided), and cancel requests whose stop is not requested yet: tick again at once
        soon = self._redecide
        for a in ats.values():
            s = seen.get(a.id)
            if s is not None and (
                s.phase in TERMINAL_PHASES or (a.state == AttemptState.STARTING and s.phase == Phase.RUNNING)
            ):
                soon = True
            w = ws.get(a.workload_id)
            if (
                w is not None
                and w.cancel_requested
                and a.state in (AttemptState.STARTING, AttemptState.RUNNING)
            ):
                soon = True
        if soon:
            cands.append(now_ms + 1)
        nr = self.runner.next_wakeup_ms()
        if nr is not None:
            cands.append(nr)
        future = [c for c in cands if c > now_ms]
        return min(future) if future else None

    def next_wakeup_ms(self) -> int | None:
        """The earliest deadline among retry times, timeouts, and grace periods (after the last tick)."""
        return self._wake

    def update_gauges(self) -> None:
        if self.metrics is None:
            return
        with self.store.read() as tx:
            rows = tx.q("SELECT namespace, state, COUNT(*) FROM workloads GROUP BY namespace, state")
            books = self.store.books(tx)
            names = [ns.name for ns in self.store.namespaces(tx)]
        for ns in names:
            for st in WorkloadState:
                self.metrics.workloads.labels(namespace=ns, state=st.value).set(0)
        for ns, st, n in rows:
            self.metrics.workloads.labels(namespace=ns, state=st).set(int(n))
        alloc: dict[str, int] = dict.fromkeys(names, 0)
        for (ns, _node), (g, _c, _m) in books.items():
            alloc[ns] = alloc.get(ns, 0) + g
        for ns, g in alloc.items():
            self.metrics.gpus_allocated.labels(namespace=ns).set(g)
        if self.inventory is not None:
            self.metrics.gpus_capacity.set(sum(n.gpus for n in self.inventory.nodes if n.ready))

    def close(self) -> None:
        self.runner.close()


class _null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *a: object) -> None:
        return None
