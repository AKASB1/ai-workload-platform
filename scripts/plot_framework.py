"""Framework figure: the modules and the control loop (docs/figures/framework.png)."""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from _style import INK, INK2, SERIES, save, setup  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def box(ax, x, y, w, h, title, lines, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4,
            edgecolor=color,
            facecolor="#ffffff",
        )
    )
    ax.text(x + w / 2, y + h - 0.17, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.47 - 0.24 * i, ln, ha="center", va="top", fontsize=7.6, color=INK2)


def arrow(ax, a, b, text="", color=INK2, rad=0.0, dy=0.08):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if text:
        ax.text(
            (a[0] + b[0]) / 2,
            (a[1] + b[1]) / 2 + dy,
            text,
            ha="center",
            va="bottom",
            fontsize=7.3,
            color=INK2,
        )


def main() -> None:
    setup()
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    blue, orange, aqua, yellow = SERIES[:4]
    box(
        ax,
        0.2,
        4.5,
        2.2,
        1.6,
        "Clients",
        ["CLI · Python client", "trace replay", "one key per submission"],
        INK2,
    )
    box(
        ax,
        3.0,
        4.5,
        2.5,
        1.6,
        "HTTP API v1",
        ["FastAPI · error bodies", "admission (8 reason codes)", "/healthz · /metrics"],
        blue,
    )
    box(
        ax,
        6.2,
        4.3,
        2.6,
        1.8,
        "Store (source of truth)",
        ["SQLite / PostgreSQL", "rows with versions (CAS)", "append-only event log", "lease (holder, epoch)"],
        blue,
    )
    box(ax, 9.3, 4.5, 2.0, 1.6, "Replay", ["pure fold over the log", "= tables (I5)"], INK2)
    box(
        ax,
        3.0,
        1.9,
        3.0,
        2.0,
        "Controller tick",
        [
            "1 lease (fencing)",
            "2 inventory · 3 observe",
            "4 rules R6 R5 R4 R3 R2 R1 R8 R7",
            "5 scheduling cycle",
            "crash points (harness)",
        ],
        orange,
    )
    box(
        ax,
        0.2,
        1.9,
        2.3,
        2.0,
        "Policy",
        [
            "schedule(view) -> decision",
            "fifo · priority · quota",
            "ExternalPolicy: protocol v1",
            "child process (stub, 07)",
        ],
        aqua,
    )
    box(
        ax,
        6.6,
        1.9,
        2.2,
        2.0,
        "SchedulerAdapter",
        ["inventory · start · stop", "observe (complete)", "forget"],
        yellow,
    )
    box(ax, 9.2, 2.85, 2.1, 1.05, "Local backend", ["virtual / scaled clock"], yellow)
    box(
        ax,
        9.2,
        1.5,
        2.1,
        1.15,
        "Kubernetes backend",
        ["Indexed Jobs · KubeClient", "real (kind) · fake"],
        yellow,
    )
    box(
        ax,
        3.0,
        0.15,
        5.8,
        1.25,
        "Drivers",
        [
            "virtual: jump to the next deadline or backend event (simulation, harness, bench)",
            "live: asyncio, the core in a worker thread",
        ],
        INK2,
    )
    arrow(ax, (2.4, 5.3), (3.0, 5.3), "HTTP")
    arrow(ax, (5.5, 5.3), (6.2, 5.3), "one tx + event")
    arrow(ax, (8.8, 5.3), (9.3, 5.3))
    arrow(ax, (5.0, 3.9), (6.6, 4.4), "CAS writes", rad=-0.15, dy=0.02)
    arrow(ax, (3.0, 2.9), (2.5, 2.9), "view")
    arrow(ax, (2.5, 2.5), (3.0, 2.5), "decision", dy=-0.28)
    arrow(ax, (6.0, 2.9), (6.6, 2.9), "start/stop")
    arrow(ax, (8.8, 3.2), (9.2, 3.35))
    arrow(ax, (8.8, 2.4), (9.2, 2.1))
    arrow(ax, (4.5, 1.4), (4.5, 1.9))
    ax.text(
        5.75,
        6.32,
        "ai-workload-platform: control plane for simulated GPU workloads",
        ha="center",
        va="top",
        fontsize=11,
        color=INK,
        fontweight="bold",
    )
    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    save(fig, str(out / "framework.png"))
    print(out / "framework.png")


if __name__ == "__main__":
    main()
