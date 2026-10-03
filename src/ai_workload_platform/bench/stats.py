"""Aggregation: mean with a 95 % Student-t interval, paired differences, win/tie/loss with a tie band."""

from __future__ import annotations

import math
from typing import Any

from scipy import stats as st

TIE_BAND = 0.05  # |difference| <= 5 % of |baseline| (absolute 1e-9 for a zero baseline) is a tie

# direction: -1 lower is better, +1 higher is better, 0 descriptive (no win/tie/loss)
DIRECTION = {
    "bsld_p95": -1,
    "wait_p95": -1,
    "wait_mean": -1,
    "jct_mean": -1,
    "jain_bsld": 1,
    "utilization": 1,
    "slo_attainment": 1,
    "quota_satisfaction": 1,
    "borrowed_gpu_hours": 0,
    "jain_weighted_bsld": 1,
}


def mean_ci(xs: list[float]) -> tuple[float | None, float | None, int]:
    """(mean, half-width of the 95 % Student-t interval, n)."""
    vals = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(vals)
    if n == 0:
        return None, None, 0
    m = sum(vals) / n
    if n < 2:
        return m, None, n
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))
    return m, float(st.t.ppf(0.975, n - 1)) * sd / math.sqrt(n), n


def paired(base: dict[int, float], other: dict[int, float], direction: int) -> dict[str, Any]:
    """Paired differences (other - base) on the common seeds, their interval, and win/tie/loss for `other`."""
    seeds = sorted(set(base) & set(other))
    diffs, w, t, lo = [], 0, 0, 0
    for s in seeds:
        b, o = base[s], other[s]
        if b is None or o is None:
            continue
        d = o - b
        diffs.append(d)
        band = TIE_BAND * abs(b) if abs(b) > 1e-12 else 1e-9
        if abs(d) <= band or direction == 0:
            t += 1
        elif (d < 0) == (direction < 0):
            w += 1
        else:
            lo += 1
    m, h, n = mean_ci(diffs)
    out: dict[str, Any] = {"diff_mean": m, "diff_ci95": h, "n": n}
    if direction != 0:
        out.update(win=w, tie=t, loss=lo)
    return out
