"""Timeline figure of one run: a few workloads, with a controller crash and a node loss visible.

Reads benchmarks/results/<suite>/timeline_events.jsonl and timeline_meta.json (written by `bench`).
Usage: python scripts/plot_timeline.py [--suite full]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from _style import GRID, INK, INK2, MUTED, SERIES, save, setup  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REASON_COLOR = {
    "succeeded": SERIES[0],
    "node_lost": SERIES[1],
    "backend_lost": SERIES[7],
    "failed_retryable": SERIES[3],
    "cancelled": SERIES[6],
    "start_timeout": SERIES[4],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="full")
    a = ap.parse_args()
    d = ROOT / "benchmarks" / "results" / a.suite
    if not (d / "timeline_events.jsonl").exists():
        d = ROOT / "benchmarks" / "results" / "quick"
    evs = [json.loads(x) for x in (d / "timeline_events.jsonl").read_text(encoding="utf-8").splitlines() if x]
    meta = json.loads((d / "timeline_meta.json").read_text(encoding="utf-8"))
    setup()
    wids = sorted({e["workload_id"] for e in evs}, key=lambda w: int(w[1:]))
    row = {w: i for i, w in enumerate(wids)}
    submit, start, attempts = {}, {}, []
    for e in evs:
        w, t = e["workload_id"], e["at_ms"] / 1000
        if e["type"] == "submitted":
            submit[w] = t
        elif e["type"] == "started":
            start[e["attempt_id"]] = (w, t, e["data"]["placement"])
        elif e["type"] == "attempt_ended":
            w0, t0, pl = start.pop(e["attempt_id"])
            attempts.append((w0, t0, t, e["data"]["reason"], pl))
    end = max(e["at_ms"] for e in evs) / 1000
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for w, t in submit.items():
        first = min((s for (wx, s, *_r) in attempts if wx == w), default=end)
        ax.plot([t, first], [row[w], row[w]], color=MUTED, linewidth=1.2, linestyle=(0, (2, 2)))
    for w, t0, t1, reason, pl in attempts:
        ax.barh(
            row[w],
            t1 - t0,
            left=t0,
            height=0.5,
            color=REASON_COLOR.get(reason, INK2),
            edgecolor="#fcfcfb",
            linewidth=2,
        )
        ax.text(t1 + 1.5, row[w], ",".join(p["node"] for p in pl), fontsize=7, color=INK2, va="center")
    crash_t = [f["at_ms"] / 1000 for f in meta["outcome"]["fault_log"] if f["kind"] == "crash"]
    nd = [f for f in meta["outcome"]["fault_log"] if f["kind"] == "node_down"]
    if nd:
        t0 = nd[0]["at_ms"] / 1000
        t1 = next((f["at_ms"] / 1000 for f in nd if f["phase"] == "end"), end)
        ax.axvspan(t0, t1, color=GRID, alpha=0.8, zorder=0)
        ax.text(
            (t0 + t1) / 2,
            -0.75,
            f"node {nd[0]['params']['node']} down",
            ha="center",
            va="center",
            fontsize=8,
            color=INK2,
        )
    for t in crash_t:
        ax.axvline(t, color=INK, linewidth=1.2, linestyle="--")
        ax.text(t + 0.8, len(wids) - 0.35, "controller crash\n+ restart", fontsize=8, color=INK, va="top")
    ax.set_yticks(range(len(wids)), wids)
    ax.set_ylim(len(wids) - 0.4, -1.2)
    ax.set_xlabel("virtual time (s)")
    ax.set_title("One run (simulated, virtual clock): attempts by end reason; dashed grey = queued")
    used = sorted({r for *_x, r, _p in attempts})
    ax.legend(
        handles=[Patch(color=REASON_COLOR.get(r, INK2), label=r) for r in used],
        loc="lower right",
        ncol=len(used),
    )
    ax.grid(axis="y", visible=False)
    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    save(fig, str(out / "timeline.png"))
    print(out / "timeline.png")


if __name__ == "__main__":
    main()
