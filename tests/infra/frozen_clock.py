"""Frozen clock context manager for deterministic timestamp testing.

Patches datetime.now(), time.time(), and time.monotonic() to return
controlled values, advancing by a fixed delta on each call.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import patch


class FrozenClock:
    """Deterministic clock that returns controlled time values.

    Advances by a fixed step on each call to time() or monotonic().
    Tracks monotonic counter separately from wall-clock time.
    """

    def __init__(self, start: float = 1700000000.0, step: float = 1.0) -> None:
        """Initialize frozen clock.

        Args:
            start: Initial epoch timestamp (seconds since 1970-01-01 UTC)
            step: Number of seconds to advance on each call to time() or monotonic()
        """
        self.start = start
        self.step = step
        self._current_time = start
        self._monotonic_counter = 0.0

    def time(self) -> float:
        """Return current time and advance by step.

        Returns:
            Current epoch timestamp (seconds since 1970-01-01 UTC)
        """
        result = self._current_time
        self._current_time += self.step
        return result

    def now(self, tz: timezone | None = None) -> datetime:
        """Return current datetime and advance by step.

        Args:
            tz: Timezone for the datetime (default UTC)

        Returns:
            Current datetime at the current time value
        """
        if tz is None:
            tz = timezone.utc
        result = datetime.fromtimestamp(self._current_time, tz=tz)
        self._current_time += self.step
        return result

    def monotonic(self) -> float:
        """Return monotonic counter and advance by step.

        Returns:
            Monotonically increasing counter value (not wall-clock time)
        """
        result = self._monotonic_counter
        self._monotonic_counter += self.step
        return result

    def reset(self) -> None:
        """Reset clock to initial state."""
        self._current_time = self.start
        self._monotonic_counter = 0.0


@contextmanager
def frozen_clock(
    start: float = 1700000000.0, step: float = 1.0
) -> Iterator[FrozenClock]:
    """Context manager that freezes time for deterministic testing.

    Patches time.time() and time.monotonic() to use a controlled clock
    that advances by a fixed step on each call.

    Note: datetime.datetime.now() cannot be directly patched because
    datetime is immutable. Use FrozenClock.now() directly or patch in
    specific modules that use it.

    Args:
        start: Initial epoch timestamp (seconds since 1970-01-01 UTC)
        step: Number of seconds to advance on each call

    Yields:
        FrozenClock instance for fine-grained control

    Example:
        >>> with frozen_clock(start=1700000000.0, step=5.0) as clock:
        ...     t1 = time.time()  # Returns 1700000000.0
        ...     t2 = time.time()  # Returns 1700000005.0 (advanced by 5s)
        ...     dt = clock.now()  # Use clock.now() for datetime
    """
    clock = FrozenClock(start=start, step=step)

    def mock_time() -> float:
        return clock.time()

    def mock_monotonic() -> float:
        return clock.monotonic()

    with patch("time.time", side_effect=mock_time), patch(
        "time.monotonic", side_effect=mock_monotonic
    ):
        yield clock


# For pytest fixtures
import pytest


@pytest.fixture
def clock() -> Iterator[FrozenClock]:
    """Pytest fixture providing a frozen clock instance.

    Usage:
        def test_timing(clock):
            t1 = clock.time()
            t2 = clock.time()
            assert t2 == t1 + 1.0
    """
    with frozen_clock() as c:
        yield c
