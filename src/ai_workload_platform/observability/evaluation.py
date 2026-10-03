"""Evaluation metrics, computed from the event log alone (with the cluster and the namespace
configuration), so that a saved log can be re-reported. Definitions: docs/contracts.md §9; a metric
that has a name in gpu-cluster-scheduler (its docs/contracts.md §5) has the same definition there.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ai_workload_platform.cluster import ClusterConfig
from ai_workload_platform.models import Event, Namespace

TERMINAL = {"SUCCEEDED", "FAILED", "DEAD_LETTER", "CANCELLED"}
TAU_S = 10.0


def nearest_rank(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[max(0, math.ceil(p * len(s)) - 1)]


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def jain(xs: list[float]) -> float | None:
    if not xs:
        return None
    sq = sum(x * x for x in xs)
    return (sum(xs) ** 2) / (len(xs) * sq) if sq > 0 else None


@dataclass
class _W:
    id: str
    ns: str
    spec: dict[str, Any]
    seq: int
    submit_ms: int
    first_start_ms: int | None = None
    terminal: str | None = None
    terminal_ms: int | None = None
    queued_since: int | None = None
    queued: list[tuple[int, int]] = field(default_factory=list)  # QUEUED intervals


@dataclass
class _A:
    id: str
    ns: str
    gw: int
    placement: list[dict[str, Any]]
    started_ms: int
    running_ms: int | None = None
    ended_ms: int | None = None
    reason: str | None = None
    observed_ended_ms: int | None = None
    workers_by_node: list[list[Any]] | None = None


def _parse(
    events: list[Event], acc: dict[str, float] | None = None
) -> tuple[dict[str, _W], dict[str, _A], int]:
    ws: dict[str, _W] = {}
    ats: dict[str, _A] = {}
    last = 0
    acc = acc if acc is not None else {}
    for e in events:
        last = max(last, e.at_ms)
        d = e.data
        if e.type == "submitted":
            ws[e.workload_id] = _W(
                e.workload_id, e.namespace, d["spec"], e.seq, e.at_ms, queued_since=e.at_ms
            )
            continue
        w = ws[e.workload_id]
        if e.type == "started":
            if w.first_start_ms is None:
                w.first_start_ms = e.at_ms
            if w.queued_since is not None:
                w.queued.append((w.queued_since, e.at_ms))
                w.queued_since = None
            if d.get("overhead_s"):
                gw0 = int(w.spec["gpus"]) * int(w.spec["workers"])
                acc["overhead_gpu_seconds"] = acc.get("overhead_gpu_seconds", 0.0) + gw0 * float(
                    d["overhead_s"]
                )
            ats[e.attempt_id] = _A(
                e.attempt_id,
                e.namespace,
                int(w.spec["gpus"]) * int(w.spec["workers"]),
                d["placement"],
                e.at_ms,
            )
        elif e.type == "running":
            a = ats[e.attempt_id]
            a.running_ms = e.at_ms
            a.workers_by_node = d.get("workers_by_node") or None
        elif e.type == "attempt_ended":
            a = ats[e.attempt_id]
            a.ended_ms, a.reason, a.observed_ended_ms = e.at_ms, d["reason"], d.get("observed_ended_ms")
            if d["reason"] == "preempted":
                acc["preemptions"] = acc.get("preemptions", 0) + 1
                if d.get("work_done_s") is not None:
                    lost = max(0.0, float(d["work_done_s"]) - float(d.get("retained_s") or 0))
                    acc["lost_gpu_seconds"] = acc.get("lost_gpu_seconds", 0.0) + a.gw * lost
            if d["state"] == "QUEUED":
                w.queued_since = e.at_ms
        elif e.type == "requeued":
            w.queued_since = e.at_ms
        if d.get("state") in TERMINAL and e.type in ("attempt_ended", "cancel_requested"):
            w.terminal, w.terminal_ms = d["state"], e.at_ms
            if w.queued_since is not None:
                w.queued.append((w.queued_since, e.at_ms))
                w.queued_since = None
    return ws, ats, last


def _integrate(series: list[tuple[int, int, float]], t0: int, t1: int) -> float:
    """Integral over [t0, t1] (ms * value) of a sum of rectangles (start, end, value)."""
    total = 0.0
    for s, e, v in series:
        lo, hi = max(s, t0), min(e, t1)
        if hi > lo:
            total += (hi - lo) * v
    return total


def _joint_steps(
    series: list[list[tuple[int, int, float]]], t0: int, t1: int
) -> list[tuple[int, int, list[float]]]:
    """Several sums of rectangles as piecewise-constant values on common segments within [t0, t1]."""
    pts = {t0, t1}
    deltas: list[Counter[int]] = []
    for ser in series:
        dc: Counter[int] = Counter()
        for s, e, v in ser:
            lo, hi = max(s, t0), min(e, t1)
            if hi > lo:
                dc[lo] += v
                dc[hi] -= v
                pts.update((lo, hi))
        deltas.append(dc)
    ts = sorted(pts)
    cur = [0.0] * len(series)
    out = []
    for a, b in zip(ts, ts[1:], strict=False):
        for i, dc in enumerate(deltas):
            cur[i] += dc.get(a, 0)
        out.append((a, b, list(cur)))
    return out


def compute(events: list[Event], cluster: ClusterConfig, namespaces: list[Namespace]) -> dict[str, Any]:
    acc: dict[str, float] = {}
    ws, ats, last_ms = _parse(events, acc)
    speeds = cluster.classes
    fastest = max(speeds.values()) if speeds else 1.0
    term = sorted((w for w in ws.values() if w.terminal), key=lambda w: w.seq)
    n = len(term)
    k = n // 10
    window = term[k : n - k] if n - 2 * k > 0 else []
    succ = [w for w in window if w.terminal == "SUCCEEDED"]
    out: dict[str, Any] = {
        "workloads": len(ws),
        "terminal": n,
        "window": len(window),
        "window_succeeded": len(succ),
    }
    out["terminal_states"] = dict(sorted(Counter(w.terminal for w in term).items()))
    waits, jcts, bslds = [], [], []
    per_ns: dict[str, list[float]] = {}
    per_ns_wait: dict[str, list[float]] = {}
    slo_met = slo_n = 0
    for w in succ:
        assert w.first_start_ms is not None and w.terminal_ms is not None
        wait = (w.first_start_ms - w.submit_ms) / 1000
        jct = (w.terminal_ms - w.submit_ms) / 1000
        cls = w.spec.get("gpu_class")
        ideal = float(w.spec["sim"]["runtime_s"]) / (speeds.get(cls, fastest) if cls else fastest)
        b = max(1.0, jct / max(ideal, TAU_S))
        waits.append(wait)
        jcts.append(jct)
        bslds.append(b)
        per_ns.setdefault(w.ns, []).append(b)
        per_ns_wait.setdefault(w.ns, []).append(wait)
        mw = w.spec.get("max_wait_s")
        if mw is not None:
            slo_n += 1
            slo_met += 1 if wait <= float(mw) else 0
    for name, xs in (("wait", waits), ("jct", jcts), ("bsld", bslds)):
        out[f"{name}_mean"] = mean(xs)
        out[f"{name}_p50"] = nearest_rank(xs, 0.5)
        out[f"{name}_p95"] = nearest_rank(xs, 0.95)
    out["jain_bsld"] = jain([mean(v) for _ns, v in sorted(per_ns.items())])  # type: ignore[misc]
    out["slo_attainment"] = slo_met / slo_n if slo_n else None
    for ns in sorted({x.name for x in namespaces} | set(per_ns_wait)):
        out[f"ns_{ns}_wait_p95"] = nearest_rank(per_ns_wait.get(ns, []), 0.95)
        out[f"ns_{ns}_bsld_mean"] = mean(per_ns.get(ns, []))

    # the window's time span: first to last window submission (batch: first submission to last completion)
    if window:
        t0, t1 = window[0].submit_ms, window[-1].submit_ms
        if t1 <= t0:
            t1 = max(w.terminal_ms or t0 for w in window)
    else:
        t0 = t1 = 0
    alloc = [
        (a.started_ms, a.ended_ms if a.ended_ms is not None else last_ms, float(a.gw)) for a in ats.values()
    ]
    span_ms = t1 - t0
    total_gpus = cluster.total_gpus
    out["span_s"] = span_ms / 1000
    out["utilization"] = _integrate(alloc, t0, t1) / (total_gpus * span_ms) if span_ms > 0 else None

    # quota metrics over the span
    quotas = {ns.name: ns.quota_gpus for ns in namespaces}
    num = den = borrowed = 0.0
    for ns, q in sorted(quotas.items()):
        usage = [(s, e, v) for (s, e, v), a in zip(alloc, ats.values(), strict=True) if a.ns == ns]
        queued = [
            (s, e, float(int(w.spec["gpus"]) * int(w.spec["workers"])))
            for w in ws.values()
            if w.ns == ns
            for s, e in w.queued + ([(w.queued_since, last_ms)] if w.queued_since is not None else [])
        ]
        for a0, a1, (u, d) in _joint_steps([usage, usage + queued], t0, t1):
            dt = a1 - a0
            num += dt * min(u, d, q)
            den += dt * min(d, q)
            borrowed += dt * max(0.0, u - q)
    out["quota_satisfaction"] = num / den if den > 0 else None
    out["borrowed_gpu_hours"] = borrowed / 3.6e6
    shares = {ns: q / sum(quotas.values()) for ns, q in quotas.items()} if sum(quotas.values()) > 0 else {}
    bw = [(shares.get(ns, 0.0), mean(v)) for ns, v in sorted(per_ns.items())]
    sw = sum(w for w, _b in bw)
    swb2 = sum(w * b * b for w, b in bw)  # type: ignore[operator]
    out["jain_weighted_bsld"] = (
        (sum(w * b for w, b in bw) ** 2) / (sw * swb2) if bw and sw > 0 and swb2 > 0 else None
    )  # type: ignore[operator]

    # platform metrics over the whole run
    run_gs = (
        sum(
            a.gw * ((a.ended_ms if a.ended_ms is not None else last_ms) - a.running_ms)
            for a in ats.values()
            if a.running_ms is not None
        )
        / 1000
    )
    all_gs = (
        sum(a.gw * ((a.ended_ms if a.ended_ms is not None else last_ms) - a.started_ms) for a in ats.values())
        / 1000
    )
    out["running_gpu_seconds"] = run_gs
    out["allocated_gpu_seconds"] = all_gs
    out["running_over_allocated"] = run_gs / all_gs if all_gs > 0 else None
    a2r = [float(a.running_ms - a.started_ms) for a in ats.values() if a.running_ms is not None]
    out["admit_to_running_ms_mean"] = mean(a2r)
    out["admit_to_running_ms_p50"] = nearest_rank(a2r, 0.5)
    out["admit_to_running_ms_p95"] = nearest_rank(a2r, 0.95)
    lag = [
        float(a.ended_ms - a.observed_ended_ms)
        for a in ats.values()
        if a.ended_ms is not None and a.observed_ended_ms is not None
    ]
    out["observe_lag_ms_mean"] = mean(lag)
    out["observe_lag_ms_p95"] = nearest_rank(lag, 0.95)
    out["observe_lag_ms_max"] = max(lag) if lag else None
    matched = placed = 0
    for a in ats.values():
        if not a.workers_by_node:
            continue
        intended = {p["node"]: int(p["workers"]) for p in a.placement}
        observed = {str(nd): int(c) for nd, c in a.workers_by_node}
        placed += sum(intended.values())
        matched += sum(min(c, observed.get(nd, 0)) for nd, c in intended.items())
    out["placement_match"] = matched / placed if placed else None
    starts = [w.submit_ms for w in ws.values()]
    ends = [w.terminal_ms for w in ws.values() if w.terminal_ms is not None]
    out["makespan_s"] = (max(ends) - min(starts)) / 1000 if starts and ends else None
    reasons = Counter(a.reason for a in ats.values() if a.reason)
    out["attempts_by_reason"] = dict(sorted(reasons.items()))
    out["attempts"] = len(ats)
    out["retries"] = sum(1 for e in events if e.type == "requeued")
    out["preemptions"] = int(acc.get("preemptions", 0))
    out["lost_gpu_seconds"] = round(acc.get("lost_gpu_seconds", 0.0), 3)
    out["overhead_gpu_seconds"] = round(acc.get("overhead_gpu_seconds", 0.0), 3)
    out["dead_letters"] = sum(1 for w in ws.values() if w.terminal == "DEAD_LETTER")
    out["counted_failures"] = sum(
        1
        for e in events
        if e.type == "attempt_ended" and e.data.get("counted") and e.data.get("reason") != "succeeded"
    )
    return out
