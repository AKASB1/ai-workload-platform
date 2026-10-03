"""Check 12: the worked example of docs/contracts.md §9 (10 workloads, 3 namespaces, one 4-GPU node),
whose metrics are computed by hand there, against observability.evaluation.compute."""

from __future__ import annotations

import pytest

from ai_workload_platform.cluster import parse_cluster
from ai_workload_platform.models import Event, Namespace

CLUSTER = parse_cluster(
    {
        "schema_version": 1,
        "name": "example",
        "classes": [{"name": "a100", "speed": 1.0}],
        "nodes": [{"name": "n0", "rack": "r0", "class": "a100", "gpus": 4, "cpus": 64, "mem_gb": 512}],
    }
)
NAMESPACES = [Namespace("a", 2, 4), Namespace("b", 1, 4), Namespace("c", 1, 4)]

# id, namespace, submit_s, start_s (None: cancelled while queued at end_s), end_s, runtime_s, max_wait_s
EXAMPLE = [
    ("w01", "a", 0, 0, 100, 100, None),
    ("w02", "b", 10, 10, 60, 50, 5),
    ("w03", "c", 20, 20, 40, 20, None),
    ("w04", "a", 30, 30, 230, 200, 60),
    ("w05", "a", 40, 40, 45, 5, None),
    ("w06", "b", 50, 60, 160, 100, 5),
    ("w07", "c", 60, 100, 150, 50, None),
    ("w08", "c", 70, None, 80, 30, None),
    ("w09", "b", 80, 150, 250, 100, 100),
    ("w10", "a", 90, 160, 170, 10, None),
]


def example_events() -> list[Event]:
    raw: list[tuple[int, int, str, str, str | None, dict]] = []  # (at_ms, order, type, ns, wid, data)
    for wid, ns, sub, start, end, rt, mw in EXAMPLE:
        spec = {
            "id": wid,
            "gpus": 1,
            "workers": 1,
            "priority": 4,
            "gpu_class": None,
            "sim": {"runtime_s": rt},
            "max_wait_s": mw,
            "estimate_s": rt,
        }
        raw.append((sub * 1000, 0, "submitted", ns, wid, {"spec": spec, "state": "QUEUED"}))
        if start is None:
            raw.append(
                (end * 1000, 3, "cancel_requested", ns, wid, {"immediate": True, "state": "CANCELLED"})
            )
            continue
        raw.append(
            (
                start * 1000,
                1,
                "started",
                ns,
                wid,
                {"attempt": 1, "placement": [{"node": "n0", "workers": 1}], "state": "STARTING"},
            )
        )
        raw.append(
            (
                start * 1000,
                2,
                "running",
                ns,
                wid,
                {"nodes": ["n0"], "workers_by_node": [["n0", 1]], "state": "RUNNING"},
            )
        )
        raw.append(
            (
                end * 1000,
                -1,
                "attempt_ended",
                ns,
                wid,
                {
                    "reason": "succeeded",
                    "counted": False,
                    "state": "SUCCEEDED",
                    "observed_ended_ms": end * 1000,
                },
            )
        )
    raw.sort(key=lambda x: (x[0], x[1], x[4]))
    out = []
    for i, (at, _o, typ, ns, wid, data) in enumerate(raw, start=1):
        out.append(
            Event(
                i, at, typ, ns, wid, None if typ in ("submitted", "cancel_requested") else f"{wid}-a1", data
            )
        )
    return out


def test_metrics_match_the_hand_computation() -> None:
    from ai_workload_platform.observability.evaluation import compute

    m = compute(example_events(), CLUSTER, NAMESPACES)
    assert (m["terminal"], m["window"], m["window_succeeded"]) == (10, 8, 7)
    assert m["terminal_states"] == {"CANCELLED": 1, "SUCCEEDED": 9}
    assert m["wait_mean"] == pytest.approx(120 / 7)
    assert (m["wait_p50"], m["wait_p95"]) == (0.0, 70.0)
    assert m["jct_mean"] == pytest.approx(645 / 7)
    assert (m["jct_p50"], m["jct_p95"]) == (90.0, 200.0)
    assert m["bsld_mean"] == pytest.approx(8.6 / 7)
    assert (m["bsld_p50"], m["bsld_p95"]) == (1.0, pytest.approx(1.8))
    assert m["span_s"] == 70.0
    assert m["utilization"] == pytest.approx(215 / 280)
    assert m["jain_bsld"] == pytest.approx((1 + 3.8 / 3 + 1.4) ** 2 / (3 * (1 + (3.8 / 3) ** 2 + 1.4**2)))
    assert m["slo_attainment"] == pytest.approx(0.75)
    assert m["quota_satisfaction"] == pytest.approx(210 / 230)
    assert m["borrowed_gpu_hours"] == pytest.approx(5 / 3600)
    b = [1.0, 3.8 / 3, 1.4]
    w = [0.5, 0.25, 0.25]
    expect = sum(x * y for x, y in zip(w, b, strict=True)) ** 2 / (
        sum(w) * sum(x * y * y for x, y in zip(w, b, strict=True))
    )
    assert m["jain_weighted_bsld"] == pytest.approx(expect)
    assert m["makespan_s"] == 250.0
    assert m["placement_match"] == 1.0
    assert m["observe_lag_ms_max"] == 0.0
