"""Result figures and the result table from the committed results.

Reads benchmarks/results/<suite>/ (and benchmarks/results/cluster/ when present) and writes
docs/figures/e1_convergence.png, e1_faults.png, e2_backends.png, e3_policies.png, e3_quota.png, and
benchmarks/results/<suite>/table.md. Usage: python scripts/plot_results.py [--suite full]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from _style import INK2, SERIES, save, setup  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ["fifo+first_fit", "priority+best_fit", "quota+first_fit", "stub"]
POLICY_LABEL = {
    "fifo+first_fit": "fifo+first_fit",
    "priority+best_fit": "priority+best_fit",
    "quota+first_fit": "quota+first_fit",
    "stub": "stub (external, fifo)",
}


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x: str | None) -> float | None:
    return None if x in (None, "") else float(x)


def fmt(x: float | None, nd: int = 2) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def e1_figures(d: Path, out: Path) -> dict:
    summ = json.loads((d / "e1_summary.json").read_text(encoding="utf-8"))
    runs = rows(d / "e1_runs.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, b in enumerate(("local", "kube-fake")):
        xs = sorted(
            num(r["converge_ms"]) / 1000
            for r in runs
            if r["backend"] == b and r["bug"] == "none" and r["converge_ms"]
        )
        n = len(xs)
        ax.step(
            xs,
            [(k + 1) / n for k in range(n)],
            where="post",
            color=SERIES[i],
            label=f"{b} ({summ[b]['schedules']} schedules, {summ[b]['violating_schedules']} violations)",
        )
    ax.set_xlabel("convergence time after the last fault (virtual s)")
    ax.set_ylabel("share of schedules")
    ax.set_title("E1: every schedule converges (simulated)")
    ax.legend(loc="lower right")
    save(fig, str(out / "e1_convergence.png"))

    kinds = sorted({k for b in ("local", "kube-fake") for k in summ[b]["faults_injected"]})
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    h = 0.38
    for i, b in enumerate(("local", "kube-fake")):
        vals = [summ[b]["faults_injected"].get(k, 0) for k in kinds]
        ax.barh(
            [y + (i - 0.5) * h for y in range(len(kinds))], vals, height=h - 0.04, color=SERIES[i], label=b
        )
    ax.set_yticks(range(len(kinds)), kinds)
    ax.invert_yaxis()
    ax.set_xlabel("faults injected (count over all schedules)")
    ax.set_title("E1: faults injected by kind")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, str(out / "e1_faults.png"))
    return summ


def e2_figure(d: Path, cluster: Path, out: Path) -> list[dict]:
    r2 = rows(d / "e2_runs.csv")
    rc = rows(cluster / "e2_cluster_runs.csv")
    panels = [
        ("makespan_s", "makespan (trace s)"),
        ("wait_mean", "mean wait (trace s)"),
        ("admit_to_running_ms_mean", "admit → running (real ms)"),
        ("observe_lag_ms_mean", "observe lag (real ms)"),
        ("running_over_allocated", "running / allocated GPU-s"),
    ]
    groups = [
        ("local", [r for r in r2 if r["backend"] == "local"], "mean ± 95% CI, 10 seeds"),
        ("kube-fake", [r for r in r2 if r["backend"] == "kube-fake"], "mean ± 95% CI, 10 seeds"),
    ]
    if rc:
        groups += [
            (f"kind {m}", [r for r in rc if r["mode"] == m], "median, min–max, 3 runs, seed 1")
            for m in ("pinned", "delegate")
        ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.4))
    table = []
    for ax, (m, title) in zip(axes, panels, strict=True):
        for i, (name, rs, _how) in enumerate(groups):
            vals = [num(r[m]) for r in rs if num(r[m]) is not None]
            if not vals:
                continue
            if name.startswith("kind"):
                mid, lo, hi = statistics.median(vals), min(vals), max(vals)
                err = [[mid - lo], [hi - mid]]
            else:
                from ai_workload_platform.bench.stats import mean_ci

                mid, ci, _n = mean_ci(vals)
                err = [[ci or 0], [ci or 0]]
            ax.bar(i, mid, color=SERIES[i], width=0.7, yerr=err, ecolor=INK2, capsize=3)
        ax.set_xticks(range(len(groups)), [g[0] for g in groups], rotation=30, ha="right")
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        "E2: the same trace on the local backend, the Kubernetes fake"
        + (", and kind" if rc else "")
        + " (simulated GPUs)",
        fontsize=10.5,
    )
    fig.tight_layout()
    save(fig, str(out / "e2_backends.png"))
    for name, rs, how in groups:
        row = {"backend": name, "how": how, "runs": len(rs)}
        for m in (
            "makespan_s",
            "wait_mean",
            "wait_p95",
            "utilization",
            "running_over_allocated",
            "admit_to_running_ms_mean",
            "observe_lag_ms_mean",
            "observe_lag_ms_max",
            "placement_match",
        ):
            vals = [num(r[m]) for r in rs if num(r[m]) is not None]
            if not vals:
                row[m] = None
            elif name.startswith("kind"):
                row[m] = (statistics.median(vals), min(vals), max(vals))
            else:
                from ai_workload_platform.bench.stats import mean_ci

                mid, ci, _n = mean_ci(vals)
                row[m] = (mid, ci)
        reasons: dict[str, int] = {}
        for r in rs:
            for k, v in json.loads(r["attempts_by_reason"]).items():
                reasons[k] = reasons.get(k, 0) + v
        row["attempts_by_reason"] = reasons
        table.append(row)
    return table


def e3_figures(d: Path, out: Path) -> list[dict]:
    agg = rows(d / "e3_aggregate.csv")
    keys = sorted(
        {(r["variant"], float(r["load"])) for r in agg},
        key=lambda k: (["balanced", "skew", "bursty"].index(k[0]), k[1]),
    )

    def get(v: str, ld: float, p: str, m: str) -> tuple[float | None, float | None]:
        for r in agg:
            if r["variant"] == v and float(r["load"]) == ld and r["policy"] == p and r["metric"] == m:
                return num(r["mean"]), num(r["ci95"])
        return None, None

    def panel_fig(metrics: list[tuple[str, str]], name: str, title: str) -> None:
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.6))
        w = 0.8 / len(POLICIES)
        for ax, (m, mt) in zip(axes, metrics, strict=True):
            for i, p in enumerate(POLICIES):
                xs, ys, es = [], [], []
                for j, (v, ld) in enumerate(keys):
                    mean, ci = get(v, ld, p, m)
                    if mean is None:
                        continue
                    xs.append(j + (i - (len(POLICIES) - 1) / 2) * w)
                    ys.append(mean)
                    es.append(ci or 0)
                ax.bar(
                    xs,
                    ys,
                    width=w * 0.92,
                    color=SERIES[i],
                    yerr=es,
                    ecolor=INK2,
                    capsize=2,
                    label=POLICY_LABEL[p],
                )
            ax.set_xticks(range(len(keys)), [f"{v}\n{ld:g}" for v, ld in keys], fontsize=8)
            ax.set_title(mt, fontsize=9.5)
            ax.grid(axis="x", visible=False)
        axes[0].legend(loc="upper left", fontsize=7.5)
        fig.suptitle(title, fontsize=10.5)
        fig.tight_layout()
        save(fig, str(out / name))

    panel_fig(
        [
            ("bsld_p95", "P95 bounded slowdown"),
            ("wait_p95", "P95 wait (s)"),
            ("jain_bsld", "Jain (bsld)"),
            ("utilization", "utilization"),
        ],
        "e3_policies.png",
        "E3: policies on the platform (simulated; mean ± 95% CI, 10 seeds; x = variant, offered load)",
    )
    panel_fig(
        [
            ("slo_attainment", "SLO attainment"),
            ("quota_satisfaction", "quota satisfaction"),
            ("borrowed_gpu_hours", "borrowed GPU-hours"),
            ("jain_weighted_bsld", "quota-weighted Jain (bsld)"),
        ],
        "e3_quota.png",
        "E3: SLO and quota metrics (simulated; mean ± 95% CI, 10 seeds)",
    )
    return agg


def e3p_figure(d: Path, out: Path) -> list[dict]:
    rs = rows(d / "e3_preemption.csv")
    if not rs:
        return []
    keys = sorted(
        {(r["variant"], float(r["load"])) for r in rs},
        key=lambda k: (["balanced", "skew", "bursty"].index(k[0]), k[1]),
    )
    pols = sorted({(r["base"], r["policy"]) for r in rs})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (m, title) in zip(
        axes, [("bsld_p95", "P95 bounded slowdown"), ("slo_attainment", "SLO attainment")], strict=True
    ):
        series = []
        for base, pol in pols:
            series += [(base, "base_mean", "base_ci95"), (pol, "mean", "ci95")]
        w = 0.8 / len(series)
        for i, (name, mk, ck) in enumerate(series):
            xs, ys, es = [], [], []
            for j, (v, ld) in enumerate(keys):
                r = next(
                    (
                        r
                        for r in rs
                        if r["variant"] == v
                        and float(r["load"]) == ld
                        and r["metric"] == m
                        and (r["base"] == name or r["policy"] == name)
                    ),
                    None,
                )
                if r is None or r[mk] == "":
                    continue
                xs.append(j + (i - (len(series) - 1) / 2) * w)
                ys.append(float(r[mk]))
                es.append(float(r[ck] or 0))
            ax.bar(
                xs,
                ys,
                width=w * 0.92,
                color=SERIES[[1, 4, 2, 5][i % 4]],
                yerr=es,
                ecolor=INK2,
                capsize=2,
                label=name,
            )
        ax.set_xticks(range(len(keys)), [f"{v} {ld:g}" for v, ld in keys], fontsize=8)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Tier 2: preemption with checkpoints (simulated; mean ± 95% CI, 10 seeds)", fontsize=10.5)
    fig.tight_layout()
    save(fig, str(out / "e3_preemption.png"))
    return rs


def table_md(d: Path, e1: dict, e2: list[dict], agg: list[dict], e3p: list[dict] | None = None) -> str:
    pairs = rows(d / "e3_paired.csv")
    lines = [
        "# Result tables (simulated)",
        "",
        "Generated by `scripts/plot_results.py` from the files in this folder.",
        "",
        "## E1 failure injection",
        "",
        "| Backend | Schedules | Violations | Convergence P50 / P95 / max (virtual s) |",
        "|---|---|---|---|",
    ]
    for b, s in e1.items():
        lines.append(
            f"| {b} | {s['schedules']} | {s['violating_schedules']} | {fmt(s['converge_s_p50'], 1)} / "
            f"{fmt(s['converge_s_p95'], 1)} / {fmt(s['converge_s_max'], 1)} |"
        )
    if any("bugs" in s for s in e1.values()):
        lines += [
            "",
            "| Injected bug | Backend | Caught (of 200) | Schedules until caught | Invariant |",
            "|---|---|---|---|---|",
        ]
        for b, s in e1.items():
            for bug, x in s.get("bugs", {}).items():
                lines.append(
                    f"| {bug} | {b} | {x['caught']} | {x['schedules_until_caught']} | "
                    f"{', '.join(x['invariants'])} |"
                )
    lines += [
        "",
        "## E2 backends (trace `balanced`, load 0.8, 40 workloads, fifo+first_fit)",
        "",
        "| Backend | Runs | Makespan (trace s) | Mean wait (trace s) | Utilization | Running / allocated | "
        "Admit → running (ms) | Observe lag mean / max (ms) | Placement match |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def cell(x: tuple | None, nd: int = 1) -> str:
        if x is None:
            return "–"
        if len(x) == 3:
            return f"{x[0]:.{nd}f} [{x[1]:.{nd}f}, {x[2]:.{nd}f}]"
        return f"{x[0]:.{nd}f} ± {x[1] or 0:.{nd}f}"

    for r in e2:
        lag = f"{cell(r['observe_lag_ms_mean'], 0)} / {cell(r['observe_lag_ms_max'], 0)}"
        lines.append(
            f"| {r['backend']} | {r['runs']} | {cell(r['makespan_s'])} | {cell(r['wait_mean'])} | "
            f"{cell(r['utilization'], 3)} | {cell(r['running_over_allocated'], 3)} | "
            f"{cell(r['admit_to_running_ms_mean'], 0)} | {lag} | {cell(r['placement_match'], 2)} |"
        )
    lines += [
        "",
        "Local and fake: mean ± 95 % Student-t half-width over seeds 1–10. kind: median [min, max] over 3 "
        "repetitions of seed 1, wall-clock latencies on a shared machine (see the manifest for the load note).",
        "",
        "## E3 policies (300 workloads, 10 seeds, local backend)",
        "",
        "| Variant | Load | Policy | P95 bsld | P95 wait (s) | Jain (bsld) | Utilization | SLO attainment | "
        "Quota satisfaction | P95 bsld vs fifo (W/T/L) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    by = {(r["variant"], r["load"], r["policy"], r["metric"]): r for r in agg}
    pb = {(r["variant"], r["load"], r["policy"], r["metric"]): r for r in pairs}
    for v, ld in sorted(
        {(r["variant"], r["load"]) for r in agg},
        key=lambda k: (["balanced", "skew", "bursty"].index(k[0]), k[1]),
    ):
        for p in POLICIES:

            def mc(m: str, nd: int = 2, v: str = v, ld: str = ld, p: str = p) -> str:
                r = by.get((v, ld, p, m))
                return (
                    "–"
                    if not r or r["mean"] == ""
                    else f"{float(r['mean']):.{nd}f} ± {float(r['ci95'] or 0):.{nd}f}"
                )

            q = pb.get((v, ld, p, "bsld_p95"))
            wtl = f"{q['win']}/{q['tie']}/{q['loss']}" if q else "(baseline)"
            lines.append(
                f"| {v} | {ld} | {POLICY_LABEL[p]} | {mc('bsld_p95')} | {mc('wait_p95', 0)} | "
                f"{mc('jain_bsld', 3)} | {mc('utilization', 3)} | {mc('slo_attainment', 3)} | "
                f"{mc('quota_satisfaction', 3)} | {wtl} |"
            )
    lines += [
        "",
        "Tie band: a paired difference within 5 % of the baseline value counts as a tie. "
        "The stub external policy implements fifo+first_fit over the protocol and must equal it.",
        "",
    ]
    if e3p:
        lines += [
            "## Tier 2: preemption with checkpoints (same traces, paired with the base policy)",
            "",
            "| Variant | Load | Policy | Preemptions | Lost GPU-h | Overhead GPU-h | P95 bsld (base -> variant) | "
            "SLO attainment (base -> variant) | P95 bsld vs base (W/T/L) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        g = {(r["variant"], r["load"], r["policy"], r["metric"]): r for r in e3p}
        keys = sorted(
            {(r["variant"], r["load"], r["policy"]) for r in e3p},
            key=lambda k: (["balanced", "skew", "bursty"].index(k[0]), k[1], k[2]),
        )
        for v, ld, p in keys:

            def m(name: str, key: str = "mean", nd: int = 2, scale: float = 1.0, v=v, ld=ld, p=p) -> str:
                r = g.get((v, ld, p, name))
                return "–" if r is None or r[key] == "" else f"{float(r[key]) * scale:.{nd}f}"

            b = g.get((v, ld, p, "bsld_p95"))
            wtl = f"{b['win']}/{b['tie']}/{b['loss']}" if b and b.get("win") not in (None, "") else "–"
            lines.append(
                f"| {v} | {ld} | {p} | {m('preemptions', nd=1)} | {m('lost_gpu_seconds', scale=1 / 3600)} | "
                f"{m('overhead_gpu_seconds', scale=1 / 3600)} | {m('bsld_p95', 'base_mean')} -> {m('bsld_p95')} | "
                f"{m('slo_attainment', 'base_mean', 3)} -> {m('slo_attainment', nd=3)} | {wtl} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="full")
    a = ap.parse_args()
    d = ROOT / "benchmarks" / "results" / a.suite
    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    setup()
    e1 = e1_figures(d, out)
    e2 = e2_figure(d, ROOT / "benchmarks" / "results" / "cluster", out)
    agg = e3_figures(d, out)
    e3p = e3p_figure(d, out)
    (d / "table.md").write_text(table_md(d, e1, e2, agg, e3p), encoding="utf-8", newline="\n")
    print("figures in", out, "table in", d / "table.md")


if __name__ == "__main__":
    main()
