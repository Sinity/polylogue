"""Proof tests for the autouse host-clock guard (tests/infra/clock_guard.py).

These assert the guard actually makes the host clock unreachable from
guarded test code — the property the old `test-clock-hygiene` lint could
only detect after the fact. If this file is itself removed from the guard
(e.g. via a stray `uses_real_clock` marker), these tests fail loudly because
the `pytest.raises` blocks would no longer see a raise.
"""

from __future__ import annotations

import time
from datetime import datetime

import pytest


def test_time_time_raises_outside_frozen_clock() -> None:
    with pytest.raises(RuntimeError, match="frozen_clock"):
        time.time()


def test_time_monotonic_raises_outside_frozen_clock() -> None:
    with pytest.raises(RuntimeError, match="frozen_clock"):
        time.monotonic()


def test_time_monotonic_ns_raises_outside_frozen_clock() -> None:
    with pytest.raises(RuntimeError, match="frozen_clock"):
        time.monotonic_ns()


def test_time_time_ns_raises_outside_frozen_clock() -> None:
    with pytest.raises(RuntimeError, match="frozen_clock"):
        time.time_ns()


def test_datetime_now_raises_outside_frozen_clock() -> None:
    with pytest.raises(RuntimeError, match="frozen_clock"):
        datetime.now()


def test_datetime_utcnow_raises_outside_frozen_clock() -> None:
    with pytest.raises(RuntimeError, match="frozen_clock"):
        datetime.utcnow()


def test_frozen_clock_fixture_bypasses_the_guard(frozen_clock: object) -> None:
    # Requesting frozen_clock exempts this test from the raising guard;
    # time.time() should resolve to the frozen clock's controlled value
    # instead of raising.
    assert time.time() == pytest.approx(1700000000.0)


@pytest.mark.uses_real_clock("proves the opt-out marker suppresses the guard")
def test_uses_real_clock_marker_bypasses_the_guard() -> None:
    # Must not raise.
    time.time()
    datetime.now()
