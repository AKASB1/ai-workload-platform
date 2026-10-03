"""Check 11: the trace loader (schema v1 of gpu-cluster-scheduler), the cluster-configuration loader,
the generator's statistics over 20 seeds, and a trace replayed through the API on a scaled clock."""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

from ai_workload_platform.bench.generator import DEFAULTS, generate
from ai_workload_platform.bench.trace import (
    HEADER,
    TraceError,
    load_trace,
    parse_trace,
    to_submissions,
    write_trace,
)
from ai_workload_platform.cluster import ClusterConfigError, load_cluster, parse_cluster

ROOT = Path(__file__).resolve().parents[1]
REF = load_cluster(ROOT / "configs" / "clusters" / "reference.json")
H = ",".join(HEADER)
ROW = "j1,0,team-a,u1,4,2,1,,any,24,192,100.5,120,0,600,3600"


def _bad(text: str, line: int, fragment: str = "") -> None:
    with pytest.raises(TraceError) as e:
        parse_trace(text.encode())
    assert str(e.value).startswith(f"line {line}:"), str(e.value)
    assert fragment in str(e.value)


def test_trace_valid_crlf_quoted_and_empty() -> None:
    rows = parse_trace(f'{H}\r\n{ROW}\r\n"j2",1.25,team-b,,9,8,2,a100,rack,0,0,1,1,1,0,\r\n'.encode())
    assert [r.job_id for r in rows] == ["j1", "j2"]
    assert rows[1].max_wait_s is None and rows[1].to_spec()["labels"] == {"user": "team-b"}
    assert rows[0].to_spec()["sim"] == {"runtime_s": 100.5}
    assert parse_trace(f"{H}\n".encode()) == []  # header only: a valid empty trace


def test_trace_malformed_files_name_the_line() -> None:
    with pytest.raises(TraceError, match="line 1"):
        parse_trace(b"")
    _bad("job_id,submit_s\n", 1, "header")
    _bad(f"{H}\n{ROW},extra\n", 2, "fields")
    _bad(f"{H}\n{ROW}\nj2,0.0001,team-a,u,4,1,1,,any,0,0,1,1,0,0,\n", 3, "three decimals")
    _bad(f"{H}\n{ROW}\n{ROW}\n", 3, "duplicate")
    _bad(f"{H}\nj1,5,team-a,u,4,1,1,,any,0,0,1,1,0,0,\nj2,4,team-a,u,4,1,1,,any,0,0,1,1,0,0,\n", 3, "smaller")
    _bad(f"{H}\nj1,0,,u,4,1,1,,any,0,0,1,1,0,0,\n", 2, "tenant")
    _bad(f"{H}\nj1,0,t,u,4,1,1,,ring,0,0,1,1,0,0,\n", 2, "topology")
    _bad(f"{H}\nj1,0,t,u,10,1,1,,any,0,0,1,1,0,0,\n", 2, "priority")
    _bad(f"{H}\nj1,0,t,u,4,0,1,,any,0,0,1,1,0,0,\n", 2, "gpus")
    _bad(f"{H}\nj1,0,t,u,4,1,1,,any,0,0,0,1,0,0,\n", 2, "runtime_s")
    _bad(f"{H}\nj1,-1,t,u,4,1,1,,any,0,0,1,1,0,0,\n", 2, "submit_s")
    _bad(f"{H}\nj1,0,t,u,4,1,1,,any,0,0,1e3,1,0,0,\n", 2, "runtime_s")


def test_manifest_checks(tmp_path) -> None:
    rows, desc = generate(REF, 5, jobs=20)
    p = tmp_path / "t.csv"
    man = write_trace(p, rows, desc, 5)
    assert load_trace(p) == rows and man["jobs"] == 20 and man["seed"] == 5
    mp = tmp_path / "t.manifest.json"
    good = json.loads(mp.read_text())
    for change, msg in (
        ({"schema_version": 2}, "schema_version"),
        ({"jobs": 19}, "jobs"),
        ({"content_sha256": "0" * 64}, "content_sha256"),
    ):
        mp.write_text(json.dumps({**good, **change}))
        with pytest.raises(TraceError, match=msg):
            load_trace(p)
    mp.unlink()
    with pytest.raises(TraceError, match="manifest missing"):
        load_trace(p)


def test_cluster_loader() -> None:
    assert [n.name for n in REF.nodes] == ["r0-n00", "r0-n01", "r1-n00", "r1-n01"]
    assert (REF.total_gpus, REF.work_capacity, REF.cross_node_factor) == (32, 32.0, 1.1)
    assert REF.nodes[0].mem_mb == 1_024_000
    base = json.loads((ROOT / "configs" / "clusters" / "reference.json").read_text())
    for mutate, msg in (
        (lambda c: c.update(extra=1), "unknown fields"),
        (lambda c: c["node_groups"][0].update(colour="red"), "node_groups"),
        (lambda c: c.update(cross_node_factor=2.0), "cross_node_factor"),
        (lambda c: c["node_groups"][0].update({"class": "h100"}), "unknown class"),
        (
            lambda c: c.update(
                nodes=[{"name": "r0-n00", "rack": "r0", "class": "a100", "gpus": 8, "cpus": 1, "mem_gb": 1}]
            ),
            "duplicate",
        ),
        (lambda c: c.update(schema_version=2), "schema_version"),
        (lambda c: c["node_groups"][0].update(gpus=0), "gpus"),
    ):
        c = json.loads(json.dumps(base))
        mutate(c)
        with pytest.raises(ClusterConfigError, match=msg):
            parse_cluster(c)


@pytest.mark.parametrize("variant", ["balanced", "skew", "bursty"])
def test_generator_statistics_over_20_seeds(variant: str) -> None:
    work = span = 0.0
    ns: Counter[str] = Counter()
    sizes: Counter[tuple[int, int]] = Counter()
    n = 0
    for seed in range(1, 21):
        rows, _ = generate(REF, seed, jobs=300, variant=variant, load=0.8)
        assert [r.submit_s for r in rows] == sorted(r.submit_s for r in rows)
        work += sum(r.gpus * r.workers * r.runtime_s for r in rows)
        span += rows[-1].submit_s - rows[0].submit_s
        ns.update(r.tenant for r in rows)
        sizes.update((r.gpus, r.workers) for r in rows)
        n += len(rows)
    realized = work / (REF.work_capacity * span)
    assert abs(realized - 0.8) <= 0.08, realized
    for name, share in zip(DEFAULTS["namespaces"], DEFAULTS["shares"][variant], strict=True):
        assert abs(ns[name] / n - share) <= 0.03, (name, ns[name] / n)
    total_p = sum(s[2] for s in DEFAULTS["sizes"])
    for g, w, prob in DEFAULTS["sizes"]:
        assert abs(sizes[(g, w)] / n - prob / total_p) <= 0.03, (g, w)
    assert generate(REF, 3, jobs=50, variant=variant)[0] == generate(REF, 3, jobs=50, variant=variant)[0]


@pytest.mark.slow
def test_trace_replayed_through_the_api_matches_the_virtual_driver(tmp_path, capsys) -> None:
    import uvicorn

    from ai_workload_platform.__main__ import cmd_replay
    from ai_workload_platform.api import create_app
    from ai_workload_platform.api.service import Platform, ServiceConfig
    from ai_workload_platform.controller.rules import Thresholds
    from ai_workload_platform.sim import simulate

    rows, desc = generate(REF, 21, jobs=10, runtime_max_s=200.0)
    p = tmp_path / "replay.csv"
    write_trace(p, rows, desc, 21)
    virtual = simulate(to_submissions(rows), REF, keep=True)
    expected = {w.id: w.state.value for w in virtual.store.all_workloads()}
    virtual.close()

    plat = Platform(
        ServiceConfig(
            db=str(tmp_path / "live.db"),
            cluster=str(ROOT / "configs/clusters/reference.json"),
            scale=500.0,
            thresholds=Thresholds(observe_interval_ms=20),
        )
    )
    server = uvicorn.Server(
        uvicorn.Config(create_app(plat), host="127.0.0.1", port=0, log_config=None, access_log=False)
    )
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.monotonic() + 20
    while not server.started:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        while plat.store.inventory().nodes == []:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        rc = cmd_replay(
            argparse.Namespace(
                trace=str(p),
                url=f"http://127.0.0.1:{port}",
                scale=500.0,
                time_scale=1.0,
                wait=True,
                wait_timeout=60.0,
            )
        )
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["submitted"] == 10 and out["rejected"] == 0
        live = {w.id: w.state.value for w in plat.store.all_workloads()}
    finally:
        server.should_exit = True
        t.join(timeout=20)
        plat.close()
    assert live == expected
