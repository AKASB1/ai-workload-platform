"""`python -m ai_workload_platform faults`: run seeded schedules and report violated invariants."""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from ai_workload_platform.faults.harness import Outcome, run_schedule


def _run(args: tuple[int, str, str | None]) -> Outcome:
    return run_schedule(*args)


def run_many(seeds: list[int], backend: str, bug: str | None = None, workers: int = 1) -> list[Outcome]:
    """Run schedules (in `spawn` worker processes when workers > 1); results sorted by seed."""
    jobs = [(s, backend, bug) for s in seeds]
    if workers <= 1:
        out = [_run(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            out = list(ex.map(_run, jobs, chunksize=max(1, len(jobs) // (workers * 8))))
    return sorted(out, key=lambda o: o.seed)


def summarize(outs: list[Outcome]) -> dict[str, Any]:
    inj: Counter[str] = Counter()
    pts: Counter[str] = Counter()
    conv = sorted(o.converge_ms for o in outs if o.converge_ms is not None)
    for o in outs:
        inj.update(o.faults_injected)
        pts.update(o.crash_points_hit)
    bad = [o for o in outs if o.violations]

    def pct(p: float) -> int | None:
        import math

        return conv[max(0, math.ceil(p * len(conv)) - 1)] if conv else None

    return {
        "schedules": len(outs),
        "violating_schedules": len(bad),
        "violations_by_invariant": dict(sorted(Counter(o.violations[0]["invariant"] for o in bad).items())),
        "first_violating_seed": bad[0].seed if bad else None,
        "faults_injected": dict(sorted(inj.items())),
        "crash_points_hit": dict(sorted(pts.items())),
        "converge_s_p50": None if pct(0.5) is None else pct(0.5) / 1000,
        "converge_s_p95": None if pct(0.95) is None else pct(0.95) / 1000,
        "converge_s_max": conv[-1] / 1000 if conv else None,
        "events": sum(o.events for o in outs),
        "ticks": sum(o.ticks for o in outs),
        "wall_s_sum": round(sum(o.wall_s for o in outs), 2),
    }


def main(a: Any) -> int:
    if a.seed is not None:
        o = run_schedule(a.seed, a.backend, None if a.bug == "none" else a.bug)
        d = o.to_json()
        tail = d.pop("log_tail")
        print(json.dumps(d, indent=2))
        if o.violations:
            v = o.violations[0]
            print(f"\nVIOLATED {v['invariant']} at {v['at_ms']} ms: {v['message']}")
            print(f"\nlast {min(a.tail, len(tail))} events:")
            for line in tail[-a.tail :]:
                print(line)
            return 1
        print("\nno invariant violated")
        return 0
    lo, hi = (int(x) for x in (a.seeds or "1-100").split("-"))
    outs = run_many(list(range(lo, hi + 1)), a.backend, None if a.bug == "none" else a.bug, a.workers)
    s = summarize(outs)
    print(json.dumps({"backend": a.backend, "bug": a.bug, **s}, indent=2))
    for o in outs:
        if o.violations:
            print(
                f"seed {o.seed}: {o.violations[0]['invariant']}: {o.violations[0]['message'][:200]}",
                file=sys.stderr,
            )
    return 1 if s["violating_schedules"] and a.bug == "none" else 0
