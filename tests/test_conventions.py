"""Clock and random-stream conventions (docs/contracts.md §1): known answers and independence."""

from __future__ import annotations

import pytest

from ai_workload_platform.clock import SystemClock, VirtualClock, ceil_ms, ms_to_s
from ai_workload_platform.rng import categorical, exponential, fnv1a64, randbelow, splitmix64, stream


def test_fnv1a64_known_answers() -> None:
    assert fnv1a64("") == 0xCBF29CE484222325
    assert fnv1a64("a") == 0xAF63DC4C8601EC8C
    assert fnv1a64("foobar") == 0x85944171F73967E8


def test_splitmix64_known_answers() -> None:
    assert splitmix64(0) == 0xE220A8397B1DCDAF
    # second output of the SplitMix64 sequence seeded with 0 (state advanced by the golden gamma)
    assert splitmix64(0x9E3779B97F4A7C15) == 0x6E789E6AA1B965F4


def test_stream_is_reproducible_and_independent() -> None:
    a1 = [stream(7, "gen:arrivals").random() for _ in range(1)]
    a2 = [stream(7, "gen:arrivals").random() for _ in range(1)]
    assert a1 == a2
    names = ["gen:arrivals", "gen:sizes", "faults", "backend", "retry:w1:1", "retry:w1:2", "retry:w2:1"]
    firsts = [stream(7, n).random() for n in names]
    assert len(set(firsts)) == len(names)
    assert stream(7, "faults").random() != stream(8, "faults").random()
    # a long sequence from two streams is uncorrelated enough to differ everywhere
    r1, r2 = stream(1, "x"), stream(1, "y")
    assert sum(r1.random() == r2.random() for _ in range(1000)) == 0


def test_stream_seed_is_documented_mix() -> None:
    import random

    r = stream(42, "backend")
    ref = random.Random(splitmix64(42 ^ fnv1a64("backend")))
    assert [r.random() for _ in range(5)] == [ref.random() for _ in range(5)]


def test_distributions_use_random_only() -> None:
    r = stream(3, "d")
    xs = [exponential(r, 10.0) for _ in range(20000)]
    assert 9.5 < sum(xs) / len(xs) < 10.5
    counts = [0, 0, 0]
    for _ in range(9000):
        counts[categorical(r, [1, 2, 6])] += 1
    assert 800 < counts[0] < 1200 and 1700 < counts[1] < 2300 and 5600 < counts[2] < 6400
    assert all(0 <= randbelow(r, 7) < 7 for _ in range(1000))


def test_ceil_ms_rounds_up_with_tolerance() -> None:
    assert ceil_ms(1000.0, 1 / 1.15) == 1_150_000
    assert ceil_ms(1.0, 1.0) == 1000
    assert ceil_ms(0.0005, 1.0) == 1
    assert ms_to_s(1234) == 1.234


def test_virtual_clock_never_goes_back() -> None:
    c = VirtualClock(5)
    c.advance(10)
    assert c.now_ms() == 15
    with pytest.raises(ValueError):
        c.advance_to(14)


def test_system_clock_scaled_and_monotone() -> None:
    c = SystemClock(scale=60.0)
    a = c.now_ms()
    b = c.now_ms()
    assert b >= a >= 0
    assert c.wall_seconds_for(60_000) == pytest.approx(1.0)
