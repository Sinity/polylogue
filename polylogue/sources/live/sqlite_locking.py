"""SQLite lock handling helpers for live daemon bookkeeping."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable

from polylogue.logging import get_logger

logger = get_logger(__name__)

_SQLITE_LOCK_RETRY_DELAYS_S: tuple[float, ...] = (0.05, 0.2, 0.5)


def is_transient_sqlite_lock(exc: sqlite3.OperationalError) -> bool:
    """Return true for SQLite lock/busy errors worth treating as transient."""
    message = str(exc).lower()
    return "database is locked" in message or "database table is locked" in message or "database is busy" in message


def best_effort_cursor_write(label: str, write: Callable[[], None]) -> bool:
    """Run a non-critical cursor write with bounded lock retries."""
    attempts = len(_SQLITE_LOCK_RETRY_DELAYS_S) + 1
    for attempt in range(attempts):
        try:
            write()
            return True
        except sqlite3.OperationalError as exc:
            if not is_transient_sqlite_lock(exc):
                raise
            if attempt == attempts - 1:
                logger.warning("live.cursor: skipped %s after sqlite lock: %s", label, exc)
                return False
            time.sleep(_SQLITE_LOCK_RETRY_DELAYS_S[attempt])
    return False


__all__ = [
    "best_effort_cursor_write",
    "is_transient_sqlite_lock",
]
