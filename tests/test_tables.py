"""Check 1: the transition and retry tables in docs/contracts.md agree with the code, pair by pair."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ai_workload_platform.lifecycle import (
    ATTEMPT_TRANSITIONS,
    EVENT_KEYS,
    RETRY_TABLE,
    attempt_after,
    backoff_s,
    jitter_u,
    retry_at_ms,
    workload_after,
)
from ai_workload_platform.models import AttemptState, EndReason, InvalidTransition, WorkloadState

DOC = (Path(__file__).resolve().parents[1] / "docs" / "contracts.md").read_text(encoding="utf-8")


def _table(after_heading: str) -> list[list[str]]:
    start = DOC.index(after_heading)
    rows = []
    started = False
    for line in DOC[start:].splitlines()[1:]:
        if line.startswith("|"):
            started = True
            cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
            if set(cells[0]) <= {"-"}:
                continue
            rows.append(cells)
        elif started:
            break
    return rows[1:]  # drop the header


def _state(cell: str) -> str | None:
    return None if cell == "(none)" else cell


def doc_workload_table() -> dict[tuple[str | None, str], str]:
    return {(_state(s), e): r for s, e, r in _table("### Workload transitions (data)")}


def test_workload_table_matches_doc_for_every_pair() -> None:
    doc = doc_workload_table()
    states: list[str | None] = [None, *[s.value for s in WorkloadState]]
    checked = 0
    for s in states:
        for key in EVENT_KEYS:
            etype, _, reason = key.partition(":")
            expected = doc.get((s, key))
            if expected is None:
                with pytest.raises(InvalidTransition):
                    workload_after(s, etype, reason=reason or None, counted_after=1, max_attempts=3)
            elif expected == "(unchanged)":
                assert workload_after(s, etype, reason=reason or None) == s
            elif expected == "RETRY_WAIT / DEAD_LETTER":
                assert (
                    workload_after(s, etype, reason=reason, counted_after=1, max_attempts=3) == "RETRY_WAIT"
                )
                assert (
                    workload_after(s, etype, reason=reason, counted_after=3, max_attempts=3) == "DEAD_LETTER"
                )
            else:
                assert (
                    workload_after(s, etype, reason=reason or None, counted_after=1, max_attempts=3)
                    == expected
                )
            checked += 1
    assert checked == len(states) * len(EVENT_KEYS)
    assert len(doc) == 26


def test_attempt_table_matches_doc_for_every_pair() -> None:
    doc = {(_state(s), e): r for s, e, r in _table("### Attempt transitions (data)")}
    expanded = {}
    for (s, e), r in doc.items():
        if e == "attempt_ended:*":
            for reason in EndReason:
                expanded[(s, f"attempt_ended:{reason.value}")] = r
        else:
            expanded[(s, e)] = r
    assert {(k[0], k[1]): str(v) for k, v in ATTEMPT_TRANSITIONS.items()} == expanded
    for s in [None, *[a.value for a in AttemptState]]:
        for key in ("started", "running", "stop_requested", *[f"attempt_ended:{r.value}" for r in EndReason]):
            etype, _, reason = key.partition(":")
            if (s, key) in expanded:
                assert attempt_after(s, etype, reason or None) == expanded[(s, key)]
            else:
                with pytest.raises(InvalidTransition):
                    attempt_after(s, etype, reason or None)


def test_retry_table_matches_doc() -> None:
    doc = {r: (c, t) for r, c, t in _table("### Retry table (data)")}
    assert set(doc) == {r.value for r in EndReason}
    for reason, (counted, _retried) in doc.items():
        assert RETRY_TABLE[reason]["counted"] == (counted == "yes")


def test_cancel_wins_over_retry_and_success_wins_over_cancel() -> None:
    for reason in (
        "failed_retryable",
        "failed_fatal",
        "node_lost",
        "backend_lost",
        "start_timeout",
        "cancelled",
    ):
        assert (
            workload_after(
                "RUNNING",
                "attempt_ended",
                reason=reason,
                cancel_requested=True,
                counted_after=1,
                max_attempts=3,
            )
            == "CANCELLED"
        )
    assert (
        workload_after("RUNNING", "attempt_ended", reason="succeeded", cancel_requested=True) == "SUCCEEDED"
    )


def test_backoff_by_hand_for_defaults() -> None:
    doc = _table("| k |")
    # the doc's backoff table is one header row (k) and one value row
    values = [float(v) for v in DOC.split("| delay (s) |")[1].splitlines()[0].strip(" |").split("|")]
    assert values == [5, 10, 20, 40, 80, 160, 300, 300]
    assert [backoff_s(k, 5, 300) for k in range(1, 9)] == values
    assert doc is not None
    none = {"backoff_base_s": 5, "backoff_cap_s": 300, "jitter": "none"}
    assert [retry_at_ms(1000, k, none, "w1", 0) - 1000 for k in range(1, 9)] == [
        int(v * 1000) for v in values
    ]


def test_full_jitter_is_a_fresh_stream_per_workload_and_k() -> None:
    full = {"backoff_base_s": 5, "backoff_cap_s": 300, "jitter": "full"}
    a = retry_at_ms(0, 1, full, "w1", 7)
    assert a == retry_at_ms(0, 1, full, "w1", 7)  # repeated computation after a crash gives the same value
    u = jitter_u(7, "w1", 1)
    assert 0 <= u < 1
    assert a == int(-(-5000 * u // 1)) or a == int(5000 * u) + 1
    assert jitter_u(7, "w1", 2) != u and jitter_u(7, "w2", 1) != u and jitter_u(8, "w1", 1) != u
    for k in range(1, 12):
        assert 0 <= retry_at_ms(0, k, full, "w9", 1) <= 300_000


def test_counted_reasons_in_doc_text() -> None:
    assert re.search(r"`node_lost`, `backend_lost`, `start_timeout`\) counts as an attempt", DOC)
