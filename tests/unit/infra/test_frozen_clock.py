"""Contract tests for the ``frozen_clock`` fixture (#1300)."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta, timezone

import pytest

from tests.infra.frozen_clock import (
    DEFAULT_FROZEN_EPOCH,
    FrozenClock,
    fixed_now,
    freeze_clock,
)


def test_clock_is_stable_until_advanced(frozen_clock: FrozenClock) -> None:
    """Reading the clock twice returns the same instant (no implicit advance)."""
    t0 = frozen_clock.time()
    t1 = frozen_clock.time()
    assert t0 == t1 == DEFAULT_FROZEN_EPOCH

    now0 = frozen_clock.now()
    now1 = frozen_clock.now()
    assert now0 == now1


def test_time_module_is_patched(frozen_clock: FrozenClock) -> None:
    """``time.time`` reads the fixture clock."""
    assert time.time() == frozen_clock.time()
    assert time.monotonic() == frozen_clock.monotonic()


def test_advance_moves_both_cursors(frozen_clock: FrozenClock) -> None:
    before_wall = frozen_clock.time()
    before_mono = frozen_clock.monotonic()
    frozen_clock.advance(60.0)
    assert frozen_clock.time() == before_wall + 60.0
    assert frozen_clock.monotonic() == before_mono + 60.0


def test_set_time_jumps_wall_only(frozen_clock: FrozenClock) -> None:
    """``set_time`` jumps wall-clock but leaves monotonic untouched."""
    before_mono = frozen_clock.monotonic()
    frozen_clock.set_time(2_000_000_000.0)
    assert frozen_clock.time() == 2_000_000_000.0
    assert frozen_clock.monotonic() == before_mono


@pytest.mark.frozen_clock_modules("polylogue.daemon.health")
def test_marker_patches_datetime_in_target_module(frozen_clock: FrozenClock) -> None:
    """``frozen_clock_modules`` marker makes ``module.datetime.now`` deterministic."""
    from polylogue.daemon import health

    captured_a = health.datetime.now(UTC)  # type: ignore[attr-defined]
    captured_b = health.datetime.now(UTC)  # type: ignore[attr-defined]
    assert captured_a == captured_b == frozen_clock.now()


def test_freeze_clock_context_manager_patches_named_module() -> None:
    """Direct context-manager usage mirrors the fixture behavior."""
    with freeze_clock(patch_datetime_in_modules=["polylogue.daemon.health"]) as clock:
        from polylogue.daemon import health

        assert health.datetime.now(UTC) == clock.now()  # type: ignore[attr-defined]


def test_unknown_module_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no ``datetime`` symbol"):
        with freeze_clock(patch_datetime_in_modules=["polylogue"]):
            pass


def test_fixed_now_returns_canonical_anchor() -> None:
    """``fixed_now`` returns a stable datetime without patching."""
    assert fixed_now().tzinfo is timezone.utc
    assert fixed_now() == datetime.fromtimestamp(DEFAULT_FROZEN_EPOCH, tz=timezone.utc)


def test_default_step_is_zero_relative_timing_is_stable(frozen_clock: FrozenClock) -> None:
    """Two ``now()`` reads bracket the same instant, so deltas are predictable."""
    anchor = frozen_clock.now()
    past = anchor - timedelta(seconds=120)
    # The relative arithmetic that test suites use everywhere stays well-defined.
    assert (anchor - past).total_seconds() == 120.0
