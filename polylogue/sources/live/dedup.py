"""Per-key rate limiter and schema-version-mismatch handler for live ingest.

Used to dedup high-frequency, structural log events such as schema-version
mismatches: every inotify event under a watched root would otherwise produce
the same warning, flooding the journal and masking other signal.

The limiter is intentionally tiny — a one-shot in-memory map. There is no
need for a new dependency. See #1003 for the IOPS-storm incident this
guards against.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

from polylogue.core.degraded import DegradedReason, is_degraded, set_degraded
from polylogue.core.errors import DatabaseError, SchemaVersionMismatchError
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


def handle_structural_database_error(source_name: str, exc: DatabaseError) -> None:
    """Log once per dedup window and put the daemon in degraded mode.

    Schema/layout mismatch is a structural condition that does not change
    across consecutive inotify events. Logging at WARNING for every event is
    what produced the journal flood and IOPS storm in #1003.
    """
    detail: Mapping[str, object]
    if isinstance(exc, SchemaVersionMismatchError):
        code = "schema_version_mismatch"
        signature = f"{code}:{exc.current_version}->{exc.expected_version}"
        detail = {
            "current_version": exc.current_version,
            "expected_version": exc.expected_version,
        }
        version_suffix = f" (db schema v{exc.current_version}, runtime expects v{exc.expected_version})"
    else:
        code = "database_layout_mismatch"
        signature = f"{code}:{type(exc).__name__}:{str(exc)}"
        detail = {"error": str(exc)}
        version_suffix = ""
    if schema_warning_limiter.admit((source_name, signature)):
        _logger.warning(
            "live.watcher: %s — refusing further ingest until restart%s: %s",
            source_name,
            version_suffix,
            exc,
        )
    if not is_degraded():
        set_degraded(
            DegradedReason(
                code=code,
                message=str(exc),
                detail=detail,
            )
        )


def handle_schema_version_mismatch(source_name: str, exc: SchemaVersionMismatchError) -> None:
    """Handle the common version-only structural database error."""
    handle_structural_database_error(source_name, exc)


__all__ = [
    "SCHEMA_MISMATCH_DEDUP_WINDOW_S",
    "RateLimiter",
    "handle_schema_version_mismatch",
    "handle_structural_database_error",
    "schema_warning_limiter",
]
