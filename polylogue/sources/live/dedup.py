"""Per-key rate limiter and schema-incompatible handler for live ingest.

Used to dedup high-frequency, structural log events such as schema-version
mismatches: every inotify event under a watched root would otherwise produce
the same warning, flooding the journal and masking other signal.

The limiter is intentionally tiny — a one-shot in-memory map. There is no
need for a new dependency. See #1003 for the IOPS-storm incident this
guards against.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from polylogue.core.degraded import DegradedReason, is_degraded, set_degraded
from polylogue.errors import SchemaIncompatibleError
from polylogue.logging import get_logger

SCHEMA_MISMATCH_DEDUP_WINDOW_S = 60.0
"""Per-(source, signature) suppression window for schema-mismatch warnings."""


class RateLimiter:
    """Admit one event per ``window`` seconds per key.

    Returns ``True`` the first time a key is seen and again after ``window``
    seconds have passed since its last admitted entry.
    """

    __slots__ = ("_window", "_last", "_clock")

    def __init__(self, window_s: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._window = window_s
        self._clock = clock
        self._last: dict[tuple[object, ...], float] = {}

    def admit(self, key: tuple[object, ...]) -> bool:
        now = self._clock()
        last = self._last.get(key)
        if last is None or now - last >= self._window:
            self._last[key] = now
            return True
        return False

    def reset(self) -> None:
        self._last.clear()


schema_warning_limiter = RateLimiter(SCHEMA_MISMATCH_DEDUP_WINDOW_S)


_logger = get_logger("polylogue.sources.live.batch")


def handle_schema_incompatible(source_name: str, exc: SchemaIncompatibleError) -> None:
    """Log once per dedup window and put the daemon in degraded mode.

    Schema-version mismatch is a structural condition that does not change
    across consecutive inotify events. Logging at WARNING for every event
    is what produced the journal flood and IOPS storm in #1003.
    """
    signature = f"schema_incompatible:{exc.current_version}->{exc.expected_version}"
    if schema_warning_limiter.admit((source_name, signature)):
        _logger.warning(
            "live.watcher: %s — refusing further ingest until restart (db schema v%s, runtime expects v%s)",
            source_name,
            exc.current_version,
            exc.expected_version,
        )
    if not is_degraded():
        set_degraded(
            DegradedReason(
                code="schema_incompatible",
                message=str(exc),
                detail={
                    "current_version": exc.current_version,
                    "expected_version": exc.expected_version,
                },
            )
        )


__all__ = [
    "SCHEMA_MISMATCH_DEDUP_WINDOW_S",
    "RateLimiter",
    "handle_schema_incompatible",
    "schema_warning_limiter",
]
