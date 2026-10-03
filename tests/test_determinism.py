"""Check 9: the same seed gives byte-identical event logs, twice in one process and across
ProcessPoolExecutor workers (spawn), for the local backend and the Kubernetes fake; seeds differ."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import pytest

from ai_workload_platform.sim import trace_log


@pytest.mark.parametrize("backend", ["local", "kube-fake"])
def test_same_seed_same_bytes_in_process(backend: str) -> None:
    a = trace_log(7, backend)
    b = trace_log(7, backend)
    assert a == b and len(a) > 1000
    assert trace_log(8, backend) != a


def test_same_seed_same_bytes_across_spawned_workers() -> None:
    jobs = [(s, b) for b in ("local", "kube-fake") for s in (3, 4)]
    local = {j: trace_log(*j) for j in jobs}
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as ex:
        remote = dict(zip(jobs, ex.map(trace_log, *zip(*jobs, strict=True)), strict=True))
    assert remote == local
    assert local[(3, "local")] != local[(4, "local")]


def test_policies_differ_but_each_is_deterministic() -> None:
    logs = {p: trace_log(11, "local", p) for p in ("fifo+first_fit", "priority+best_fit", "quota+first_fit")}
    assert len(set(logs.values())) == 3
    assert trace_log(11, "local", "priority+best_fit") == logs["priority+best_fit"]
