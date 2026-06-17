"""Deterministic clock fixtures for timestamp-sensitive tests (#1300).

Two surfaces are exported:

``FrozenClock`` / ``freeze_clock``
    A hand-rolled controlled clock that returns the same instant on every
    call until ``advance()`` is invoked explicitly. ``time.time`` and
    ``time.monotonic`` are patched globally for the lifetime of the
    ``freeze_clock`` context manager.

``frozen_clock`` pytest fixture
    Yields a ``FrozenClock`` and (optionally) patches ``datetime.now`` in
    every module the test asks for via the ``frozen_clock_modules`` marker.
    Tests opt-in by requesting the fixture in their signature; nothing is
    mocked otherwise.

CPython's ``datetime`` is immutable at the C level, so ``datetime.now``
cannot be patched globally the way ``time.time`` can. Instead we patch each
production module's ``datetime`` symbol individually. Tests register the
modules they care about via the ``frozen_clock_modules`` marker.

Usage::

    def test_clock_basics(frozen_clock):
        now = frozen_clock.now()  # always the same instant
        frozen_clock.advance(60)
        assert frozen_clock.now() != now

    @pytest.mark.frozen_clock_modules("polylogue.daemon.health")
    def test_health_alert(frozen_clock):
        # polylogue.daemon.health.datetime.now(UTC) returns frozen_clock.now()
        ...
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from importlib import import_module
from unittest.mock import patch

import pytest

# Canonical anchor: 2023-11-14T22:13:20+00:00. Chosen because it predates
# every realistic test interval and is far enough from epoch that integer
# overflow / negative-duration arithmetic surfaces immediately.
DEFAULT_FROZEN_EPOCH = 1700000000.0


class FrozenClock:
    """Deterministic clock that returns controlled time values.

    The clock holds a single moving cursor that is shared between
    ``time()``, ``monotonic()``, and ``now()``. The cursor does NOT advance
    on each read — the caller advances it explicitly via :meth:`advance`.
    This matches how production code uses ``datetime.now``: a single ``now``
    value captured at the top of a function is reused throughout.
    """

    def __init__(self, start: float = DEFAULT_FROZEN_EPOCH) -> None:
        self.start = start
        self._current = start
        self._monotonic = 0.0

    def time(self) -> float:
        return self._current

    def monotonic(self) -> float:
        return self._monotonic

    def now(self, tz: timezone | None = None) -> datetime:
        if tz is None:
            tz = timezone.utc
        return datetime.fromtimestamp(self._current, tz=tz)

    def advance(self, seconds: float) -> None:
        """Move both wall-clock and monotonic cursors forward by ``seconds``."""
        self._current += seconds
        self._monotonic += seconds

    def set_time(self, epoch: float) -> None:
        """Jump the wall-clock cursor to ``epoch``.

        Monotonic is left alone because it is not allowed to jump backward
        in production code.
        """
        self._current = epoch

    def reset(self) -> None:
        self._current = self.start
        self._monotonic = 0.0


class _FrozenDateTime(datetime):
    """``datetime`` subclass whose ``.now()`` reads from a ``FrozenClock``.

    Installed into a target module's ``datetime`` symbol; production code
    that does ``from datetime import datetime`` then calls
    ``datetime.now(UTC)`` transparently reads the frozen value.
    """

    _clock: FrozenClock

    @classmethod
    def now(cls, tz: timezone | None = None) -> datetime:  # type: ignore[override]
        return cls._clock.now(tz)

    @classmethod
    def utcnow(cls) -> datetime:  # type: ignore[override]
        return cls._clock.now(timezone.utc).replace(tzinfo=None)


def _make_frozen_datetime(clock: FrozenClock) -> type[datetime]:
    # Each clock gets its own subclass so concurrent test sessions don't
    # share the ``_clock`` class attribute.
    return type("FrozenDateTime", (_FrozenDateTime,), {"_clock": clock})


@contextmanager
def freeze_clock(
    start: float = DEFAULT_FROZEN_EPOCH,
    *,
    patch_datetime_in_modules: Sequence[str] = (),
) -> Iterator[FrozenClock]:
    """Context manager that freezes time for deterministic testing.

    Args:
        start: Initial epoch timestamp (seconds since 1970-01-01 UTC).
        patch_datetime_in_modules: Module names whose ``datetime`` symbol
            should be replaced with a frozen subclass. Production modules
            typically do ``from datetime import datetime`` once at import
            time; we patch that binding so all calls in those modules
            resolve to the frozen clock.

    Yields:
        ``FrozenClock`` instance for advance / set_time control.
    """
    clock = FrozenClock(start=start)
    frozen_dt_cls = _make_frozen_datetime(clock)

    with ExitStack() as stack:
        stack.enter_context(patch("time.time", side_effect=clock.time))
        stack.enter_context(patch("time.monotonic", side_effect=clock.monotonic))
        for module_name in patch_datetime_in_modules:
            module = import_module(module_name)
            if not hasattr(module, "datetime"):
                raise AttributeError(
                    f"Module {module_name!r} has no ``datetime`` symbol to patch; "
                    "import it as ``from datetime import datetime`` or drop it "
                    "from the frozen_clock_modules list."
                )
            stack.enter_context(patch.object(module, "datetime", frozen_dt_cls))
        yield clock


def fixed_now(epoch: float = DEFAULT_FROZEN_EPOCH) -> datetime:
    """Return a deterministic ``datetime`` anchor without patching anything.

    Useful for tests that need a stable "now" to derive past timestamps from
    but do not exercise production code that itself reads the clock.
    """
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


def _resolve_module_marker(request: pytest.FixtureRequest) -> tuple[str, ...]:
    """Read ``@pytest.mark.frozen_clock_modules(...)`` markers off the test."""
    modules: list[str] = []
    for marker in request.node.iter_markers(name="frozen_clock_modules"):
        for arg in marker.args:
            if isinstance(arg, str):
                modules.append(arg)
            else:
                modules.extend(arg)
    return tuple(modules)


@pytest.fixture(name="frozen_clock")
def frozen_clock_fixture(request: pytest.FixtureRequest) -> Iterator[FrozenClock]:
    """Pin ``time.time``, ``time.monotonic``, and (optionally) ``datetime.now``.

    Tests opt-in by requesting this fixture. To also pin ``datetime.now``
    inside specific production modules (the common case), decorate the test
    with ``@pytest.mark.frozen_clock_modules("polylogue.x.y", ...)``.

    The clock does not auto-advance — call ``clock.advance(seconds)`` to
    simulate the passage of time.
    """
    modules = _resolve_module_marker(request)
    with freeze_clock(patch_datetime_in_modules=modules) as clock:
        yield clock


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``frozen_clock_modules`` marker.

    Loaded via ``tests/conftest.py`` ``pytest_plugins``.
    """
    config.addinivalue_line(
        "markers",
        "frozen_clock_modules(*module_names): patch ``datetime`` in the named "
        "modules with the test's ``frozen_clock`` instance (#1300).",
    )


__all__ = [
    "DEFAULT_FROZEN_EPOCH",
    "FrozenClock",
    "fixed_now",
    "freeze_clock",
    "frozen_clock_fixture",
    "pytest_configure",
]
