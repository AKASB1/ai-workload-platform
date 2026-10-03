"""Injectable clocks. Platform time is integer milliseconds since the start of the run.

Domain code never reads the wall clock; it receives a `Clock` (or a `now_ms` value from the
controller tick). `VirtualClock` is used by tests, simulations, and the fault harness;
`SystemClock` (optionally scaled) by the live service.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Protocol


class Clock(Protocol):
    def now_ms(self) -> int: ...


class VirtualClock:
    """A clock that only moves when told to. Never goes backwards."""

    def __init__(self, start_ms: int = 0) -> None:
        self._now = int(start_ms)

    def now_ms(self) -> int:
        return self._now

    def advance_to(self, t_ms: int) -> None:
        if t_ms < self._now:
            raise ValueError(f"virtual clock cannot go back from {self._now} to {t_ms}")
        self._now = int(t_ms)

    def advance(self, d_ms: int) -> None:
        self.advance_to(self._now + int(d_ms))


class SystemClock:
    """Wall clock as platform milliseconds since `origin_wall_ms`, sped up by `scale`.

    `scale` = platform seconds per wall second (1.0 = real time; 60 = one wall second is one
    platform minute). The origin is kept in the store so that a restarted service continues the
    same timeline. Readings never decrease within one process.
    """

    def __init__(self, origin_wall_ms: int | None = None, scale: float = 1.0) -> None:
        if scale <= 0:
            raise ValueError("scale must be > 0")
        self.origin_wall_ms = int(time.time() * 1000) if origin_wall_ms is None else int(origin_wall_ms)
        self.scale = float(scale)
        self._last = 0
        self._lock = threading.Lock()

    def now_ms(self) -> int:
        wall = time.time() * 1000.0
        t = int((wall - self.origin_wall_ms) * self.scale)
        with self._lock:
            if t < self._last:
                t = self._last
            self._last = t
        return t

    def wall_seconds_for(self, platform_ms: int) -> float:
        """Wall-clock seconds that correspond to a platform duration."""
        return platform_ms / 1000.0 / self.scale


def ceil_ms(work_s: float, rate: float) -> int:
    """Duration in ms of `work_s` reference seconds at `rate`: ceil(work*1000/rate - 1e-6)."""
    if rate <= 0:
        raise ValueError("rate must be > 0")
    return int(math.ceil(work_s * 1000.0 / rate - 1e-6))


def ms_to_s(ms: int) -> float:
    """Decimal seconds with at most three decimals (exact for integer ms)."""
    return round(ms / 1000.0, 3)


def s_to_ms(s: float) -> int:
    return int(round(s * 1000.0))
