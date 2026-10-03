"""Named random streams: random.Random(splitmix64(seed XOR fnv1a64(name))).

Only `random()` and `getrandbits()` are used, because their output is stable across CPython
versions; every distribution below is built from `random()`. The streams are not bit-identical
to the Go streams of gpu-cluster-scheduler (those use PCG); see docs/contracts.md.
"""

from __future__ import annotations

import math
import random

MASK64 = (1 << 64) - 1
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def fnv1a64(name: str) -> int:
    h = FNV_OFFSET
    for b in name.encode("utf-8"):
        h ^= b
        h = (h * FNV_PRIME) & MASK64
    return h


def splitmix64(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def stream(seed: int, name: str) -> random.Random:
    """The random stream of component `name` under run seed `seed`."""
    return random.Random(splitmix64((int(seed) ^ fnv1a64(name)) & MASK64))


# --- distributions built from random() only -------------------------------------------------


def uniform01(r: random.Random) -> float:
    return r.random()


def randbelow(r: random.Random, n: int) -> int:
    """An integer in [0, n) from one random() draw."""
    if n <= 0:
        raise ValueError("n must be > 0")
    return min(n - 1, int(r.random() * n))


def exponential(r: random.Random, mean: float) -> float:
    return -mean * math.log(1.0 - r.random())


def normal(r: random.Random) -> float:
    """Standard normal by Box-Muller (two random() draws, the second value is discarded)."""
    u1 = 1.0 - r.random()  # (0, 1]
    u2 = r.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def lognormal(r: random.Random, mu: float, sigma: float) -> float:
    return math.exp(mu + sigma * normal(r))


def categorical(r: random.Random, weights: list[float]) -> int:
    """Index drawn with probability proportional to `weights` (one random() draw)."""
    total = float(sum(weights))
    u = r.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if u < acc:
            return i
    return len(weights) - 1
