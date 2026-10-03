"""Command line: python -m ai_workload_platform <command> (see --help)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_URL = "http://127.0.0.1:18400"
DEFAULT_CLUSTER = "configs/clusters/reference.json"


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def cmd_up(a: argparse.Namespace) -> int:
    import uvicorn

    from ai_workload_platform.api import create_app
    from ai_workload_platform.api.service import Platform, ServiceConfig
    from ai_workload_platform.controller.rules import Thresholds
    from ai_workload_platform.observability import setup_logging

    setup_logging(a.log_level)
    th = Thresholds(**json.loads(a.thresholds)) if a.thresholds else Thresholds()
    cfg = ServiceConfig.from_env(
        db=a.db,
        pg_dsn=a.pg_dsn,
        backend=a.backend,
        cluster=a.cluster,
        policy=a.policy,
        scale=a.scale,
        seed=a.seed,
        kubeconfig=a.kubeconfig,
        kube_mode=a.kube_mode,
        time_scale=a.time_scale,
        thresholds=th,
        start_latency_ms=a.start_latency_ms,
        in_cluster=a.in_cluster or None,
        policy_log=a.policy_log,
    )
    if a.holder:
        cfg.holder = a.holder
    platform = Platform(cfg)
    app = create_app(platform)
    config = uvicorn.Config(app, host=a.host, port=a.port, log_config=None, access_log=False, lifespan="on")
    server = uvicorn.Server(config)

    async def serve() -> None:
        import asyncio

        task = asyncio.create_task(server.serve())
        while not server.started and not task.done():
            await asyncio.sleep(0.05)
        if server.started:
            port = server.servers[0].sockets[0].getsockname()[1]
            msg = {"listening": f"http://{a.host}:{port}", "pid": __import__("os").getpid()}
            print(json.dumps(msg), flush=True)
            if a.port_file:
                Path(a.port_file).write_text(str(port), encoding="utf-8")
        await task

    import asyncio

    try:
        asyncio.run(serve())
    finally:
        platform.close()
    return 0


def _spec_from(a: argparse.Namespace) -> dict[str, Any]:
    if a.file:
        return json.loads(Path(a.file).read_text(encoding="utf-8"))
    return json.loads(a.spec)


def cmd_submit(a: argparse.Namespace) -> int:
    from ai_workload_platform.client import Client, ClientError

    with Client(a.url) as c:
        try:
            status, w = c.submit(a.namespace, _spec_from(a), a.idempotency_key)
        except ClientError as e:
            _print(
                {"status": e.status, "error": {"code": e.code, "message": e.message, "details": e.details}}
            )
            return 1
    _print({"status": status, "workload": w})
    return 0


def cmd_gen(a: argparse.Namespace) -> int:
    from ai_workload_platform.bench.generator import generate, offered_load
    from ai_workload_platform.bench.trace import write_trace
    from ai_workload_platform.cluster import load_cluster

    cluster = load_cluster(a.cluster)
    rows, desc = generate(
        cluster, a.seed, jobs=a.jobs, load=a.load, variant=a.variant, runtime_max_s=a.runtime_max_s
    )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    man = write_trace(out, rows, desc, a.seed)
    _print(
        {
            "trace": str(out),
            "jobs": man["jobs"],
            "duration_s": man["duration_s"],
            "offered_load": round(offered_load(rows, cluster), 3),
            "sha256": man["content_sha256"],
        }
    )
    return 0


def _load_rows(a: argparse.Namespace, cluster: Any) -> list[Any]:
    from ai_workload_platform.bench.generator import generate
    from ai_workload_platform.bench.trace import load_trace

    if a.trace:
        return load_trace(a.trace)
    rows, _ = generate(
        cluster, a.seed, jobs=a.jobs, load=a.load, variant=a.variant, runtime_max_s=a.runtime_max_s
    )
    return rows


def cmd_simulate(a: argparse.Namespace) -> int:
    from ai_workload_platform.bench.trace import to_submissions
    from ai_workload_platform.cluster import load_cluster
    from ai_workload_platform.observability.evaluation import compute
    from ai_workload_platform.sim import events_jsonl, simulate

    cluster = load_cluster(a.cluster)
    rows = _load_rows(a, cluster)
    t0 = time.perf_counter()
    res = simulate(
        to_submissions(rows),
        cluster,
        policy=a.policy,
        backend=a.backend,
        seed=a.seed,
        start_latency_ms=a.start_latency_ms,
    )
    wall = time.perf_counter() - t0
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_bytes(events_jsonl(res.events))
    m = compute(res.events, cluster, res.namespaces)
    _print(
        {
            "workloads": len(rows),
            "events": len(res.events),
            "virtual_end_s": res.stats.end_ms / 1000,
            "wall_s": round(wall, 3),
            "metrics": m,
            "events_file": a.out,
        }
    )
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    from ai_workload_platform.cluster import load_cluster
    from ai_workload_platform.models import Event, Namespace
    from ai_workload_platform.observability.evaluation import compute
    from ai_workload_platform.sim import reference_namespaces

    cluster = load_cluster(a.cluster)
    events = []
    for line in Path(a.events).read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            events.append(
                Event(
                    o["seq"],
                    o["at_ms"],
                    o["type"],
                    o["namespace"],
                    o["workload_id"],
                    o["attempt_id"],
                    o["data"],
                )
            )
    if a.namespaces:
        nss = [Namespace(**n) for n in json.loads(Path(a.namespaces).read_text(encoding="utf-8"))]
    else:
        nss = reference_namespaces(cluster.total_gpus)
    _print(compute(events, cluster, nss))
    return 0


def cmd_replay(a: argparse.Namespace) -> int:
    from ai_workload_platform.bench.trace import load_trace
    from ai_workload_platform.client import Client, ClientError

    rows = load_trace(a.trace)
    factor = a.scale * a.time_scale
    with Client(a.url) as c:
        existing = {n["name"] for n in c.namespaces()}
        total = sum(n["gpus"] for n in c.nodes())
        for t in sorted({r.tenant for r in rows} - existing):
            c.put_namespace(t, total // 3, total, 9, 1000)
        t0 = time.monotonic()
        ok = rejected = 0
        for r in rows:
            delay = r.submit_s / factor - (time.monotonic() - t0)
            if delay > 0:
                time.sleep(delay)
            try:
                c.submit(r.tenant, r.to_spec(), f"trace-{r.job_id}")
                ok += 1
            except ClientError as e:
                rejected += 1
                print(json.dumps({"job_id": r.job_id, "rejected": e.code}), file=sys.stderr)
        result: dict[str, Any] = {
            "submitted": ok,
            "rejected": rejected,
            "submit_wall_s": round(time.monotonic() - t0, 3),
        }
        if a.wait:
            deadline = time.monotonic() + a.wait_timeout
            states: dict[str, str] = {}
            while time.monotonic() < deadline:
                states = {w["id"]: w["state"] for t in sorted({r.tenant for r in rows}) for w in c.list(t)}
                if all(s in ("SUCCEEDED", "FAILED", "DEAD_LETTER", "CANCELLED") for s in states.values()):
                    break
                time.sleep(0.5)
            from collections import Counter

            result["states"] = dict(sorted(Counter(states.values()).items()))
            result["wall_s"] = round(time.monotonic() - t0, 3)
    _print(result)
    return 0


def cmd_openapi(a: argparse.Namespace) -> int:
    from ai_workload_platform.api import openapi_json

    text = openapi_json()
    p = Path(a.out)
    if a.check:
        if not p.exists() or p.read_text(encoding="utf-8") != text:
            print(
                f"{p} differs from the generated OpenAPI document; run: python -m ai_workload_platform openapi",
                file=sys.stderr,
            )
            return 1
        print(f"{p} is up to date")
        return 0
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {p}")
    return 0


def cmd_faults(a: argparse.Namespace) -> int:
    from ai_workload_platform.faults.cli import main as faults_main

    return faults_main(a)


def cmd_bench(a: argparse.Namespace) -> int:
    from ai_workload_platform.bench.run import main as bench_main

    return bench_main(a)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m ai_workload_platform",
        description="AI workload platform (simulated GPUs): control plane, harness, benchmarks",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("up", help="start the service (API + controller)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18400)
    p.add_argument("--port-file", default=None, help="write the bound port here (useful with --port 0)")
    p.add_argument("--db", default="outputs/awp.db", help="SQLite file (ignored with --pg-dsn / AWP_PG_DSN)")
    p.add_argument("--pg-dsn", default=None)
    p.add_argument("--backend", choices=["local", "kube"], default="local")
    p.add_argument("--cluster", default=DEFAULT_CLUSTER)
    p.add_argument("--policy", default="fifo+first_fit")
    p.add_argument("--policy-log", default=None, help="stderr file of an external policy")
    p.add_argument(
        "--scale", type=float, default=1.0, help="platform seconds per wall second (local backend)"
    )
    p.add_argument(
        "--time-scale", type=float, default=1.0, help="Kubernetes backend: run times are divided by it"
    )
    p.add_argument(
        "--start-latency-ms", type=int, default=0, help="local backend start latency (platform ms)"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--holder", default=None, help="controller lease holder name (default: unique per process)"
    )
    p.add_argument("--kubeconfig", default=None)
    p.add_argument("--in-cluster", action="store_true")
    p.add_argument("--kube-mode", choices=["pinned", "delegate"], default="pinned")
    p.add_argument(
        "--thresholds", default=None, help='JSON object overriding thresholds, e.g. {"lost_grace_ms": 5000}'
    )
    p.add_argument("--log-level", default="INFO")
    p.set_defaults(fn=cmd_up)

    p = sub.add_parser("submit", help="submit one workload to a running service")
    p.add_argument("namespace")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", help="JSON specification file")
    g.add_argument("--spec", help="JSON specification")
    p.add_argument("--idempotency-key", default=None)
    p.add_argument("--url", default=DEFAULT_URL)
    p.set_defaults(fn=cmd_submit)

    p = sub.add_parser("replay", help="feed a trace to a running service")
    p.add_argument("trace")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument(
        "--scale", type=float, default=1.0, help="the service's --scale (submit times are divided by it)"
    )
    p.add_argument(
        "--time-scale", type=float, default=1.0, help="the Kubernetes --time-scale (divides submit times)"
    )
    p.add_argument("--wait", action="store_true", help="wait until every workload is terminal")
    p.add_argument("--wait-timeout", type=float, default=600.0)
    p.set_defaults(fn=cmd_replay)

    def trace_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--cluster", default=DEFAULT_CLUSTER)
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--jobs", type=int, default=300)
        p.add_argument("--load", type=float, default=0.8)
        p.add_argument("--variant", choices=["balanced", "skew", "bursty"], default="balanced")
        p.add_argument("--runtime-max-s", type=float, default=None, help="truncate run times (default 7200)")

    p = sub.add_parser("gen", help="write a trace (schema v1) and its manifest")
    trace_args(p)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_gen)

    p = sub.add_parser("simulate", help="run a trace on the virtual driver and write the event log")
    trace_args(p)
    p.add_argument("--trace", default=None, help="trace CSV (default: generate one from the options)")
    p.add_argument("--policy", default="fifo+first_fit")
    p.add_argument("--backend", choices=["local", "kube-fake"], default="local")
    p.add_argument("--start-latency-ms", type=int, default=0)
    p.add_argument("--out", default=None, help="event log (JSON lines)")
    p.set_defaults(fn=cmd_simulate)

    p = sub.add_parser("report", help="metrics from a saved event log")
    p.add_argument("events")
    p.add_argument("--cluster", default=DEFAULT_CLUSTER)
    p.add_argument(
        "--namespaces", default=None, help="JSON list of namespaces (default: the reference three)"
    )
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("faults", help="run seeded fault-injection schedules")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", default=None, help="range A-B (inclusive)")
    p.add_argument("--backend", choices=["local", "kube-fake"], default="local")
    p.add_argument(
        "--bug", default="none", help="an injected bug (test-only patch) to run the schedules with"
    )
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--tail", type=int, default=25, help="event-log lines to print for a failing schedule")
    p.set_defaults(fn=cmd_faults)

    p = sub.add_parser("bench", help="run the evaluation (E1-E3)")
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument("--quick", action="store_true")
    m.add_argument("--full", action="store_true")
    m.add_argument("--cluster", dest="cluster_run", action="store_true", help="E2 (c) on a real cluster")
    p.add_argument(
        "--sweep", action="store_true", help="with --cluster: the time-scale sweep (Tier 2 item 4)"
    )
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", default="benchmarks/results")
    p.add_argument("--kubeconfig", default=None)
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("openapi", help="write (or --check) docs/openapi.json")
    p.add_argument("--out", default="docs/openapi.json")
    p.add_argument("--check", action="store_true")
    p.set_defaults(fn=cmd_openapi)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
