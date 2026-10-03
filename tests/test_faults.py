"""Check 7: the fault harness. 200 seeded schedules per backend with no invariant violated, each of the
four injected bugs caught within 200 schedules and reproduced by its seed, and deterministic logs."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import pytest

from ai_workload_platform.faults.bugs import BUGS
from ai_workload_platform.faults.cli import main as faults_main
from ai_workload_platform.faults.cli import run_many, summarize
from ai_workload_platform.faults.harness import Harness, run_schedule, schedule_log
from ai_workload_platform.faults.schedule import make_schedule

WORKERS = 4  # at most 4 parallel workers on the shared machine (TASK section 0)


@pytest.mark.parametrize("backend", ["local", "kube-fake"])
def test_200_schedules_without_violation(backend: str) -> None:
    outs = run_many(list(range(1, 201)), backend, None, WORKERS)
    s = summarize(outs)
    bad = [(o.seed, o.violations[0]) for o in outs if o.violations]
    assert s["violating_schedules"] == 0, bad[:3]
    assert s["schedules"] == 200
    # the schedules actually exercise the faults
    for kind in (
        "crash",
        "zombie",
        "store_outage",
        "store_tx_fail",
        "node_down",
        "policy_fault",
        "race_cancel",
        "observe_down",
        "observe_stale",
    ):
        assert s["faults_injected"].get(kind, 0) > 0, kind
    assert len(s["crash_points_hit"]) >= 5


@pytest.mark.parametrize("bug", sorted(BUGS))
def test_injected_bug_is_caught_within_200_schedules(bug: str) -> None:
    caught = None
    for seed in range(1, 201):
        o = run_schedule(seed, "local", bug)
        if o.violations:
            caught = o
            break
    assert caught is not None, f"{bug} not caught in 200 schedules"
    again = run_schedule(caught.seed, "local", bug)  # reproduced by its seed
    assert again.violations == caught.violations
    assert not run_schedule(caught.seed, "local", None).violations  # and absent without the bug


def test_cli_prints_the_violation_and_the_log_tail(capsys) -> None:
    seed = next(s for s in range(1, 201) if run_schedule(s, "local", "release_at_stop").violations)
    rc = faults_main(
        argparse.Namespace(seed=seed, seeds=None, backend="local", bug="release_at_stop", workers=1, tail=10)
    )
    out = capsys.readouterr().out
    assert rc == 1 and "VIOLATED I5" in out and '"seq":' in out
    rc = faults_main(
        argparse.Namespace(seed=seed, seeds=None, backend="local", bug="none", workers=1, tail=10)
    )
    assert rc == 0 and "no invariant violated" in capsys.readouterr().out


def test_zombie_controller_is_fenced() -> None:
    """Across schedules with a paused controller nothing is violated, and some paused controllers do try to
    write after the takeover and are fenced (others find every stale write rejected by compare and set)."""
    fenced = runs = 0
    for seed in range(1, 120):
        if not any(f.kind == "zombie" for f in make_schedule(seed, "local").faults):
            continue
        h = Harness(make_schedule(seed, "local"))
        out = h.run()
        assert not out.violations, (seed, out.violations)
        runs += 1
        fenced += h.fenced
        if runs >= 15 and fenced:
            break
    assert runs >= 5 and fenced >= 1


@pytest.mark.parametrize("backend", ["local", "kube-fake"])
def test_schedule_logs_are_deterministic(backend: str) -> None:
    a = schedule_log(12, backend)
    assert a == schedule_log(12, backend) and a != schedule_log(13, backend)
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as ex:
        remote = list(ex.map(schedule_log, [12, 13], [backend, backend]))
    assert remote[0] == a and remote[1] == schedule_log(13, backend)


@pytest.mark.parametrize(("backend", "seed"), [("local", 1487), ("local", 1704), ("kube-fake", 1704)])
def test_store_error_at_a_completion_converges(backend: str, seed: int) -> None:
    # regression: these schedules stalled (I7) before a tick that ends with a store error asked to be woken
    out = run_schedule(seed, backend)
    assert not out.violations, out.violations[:1]
    assert out.converge_ms is not None
