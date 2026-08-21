"""SQLite contention classification shared by runtime layers."""

from __future__ import annotations

import sqlite3

_SQLITE_PRIMARY_RESULT_CODE_MASK = 0xFF
_TRANSIENT_SQLITE_LOCK_CODES = frozenset({sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED})
_TRANSIENT_SQLITE_LOCK_NAMES = frozenset({"SQLITE_BUSY", "SQLITE_LOCKED", "SQLITE_LOCKED_SHAREDCACHE"})


def is_transient_sqlite_lock(exc: BaseException) -> bool:
    """Return whether SQLite reported retryable BUSY or LOCKED contention.

    Extended result codes such as ``SQLITE_LOCKED_SHAREDCACHE`` retain the
    ``SQLITE_LOCKED`` primary code in their low byte.  Prefer SQLite's typed
    result metadata over message text so corruption or I/O failures that happen
    to mention a lock are never mistaken for safe retryable contention.
    """
    if not isinstance(exc, sqlite3.Error):
        return False
    error_code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(error_code, int):
        return error_code & _SQLITE_PRIMARY_RESULT_CODE_MASK in _TRANSIENT_SQLITE_LOCK_CODES
    error_name = getattr(exc, "sqlite_errorname", None)
    if isinstance(error_name, str):
        return error_name in _TRANSIENT_SQLITE_LOCK_NAMES
    message = str(exc).lower()
    return (
        "database is locked" in message
        or "database table is locked" in message
        or "database schema is locked" in message
        or "database is busy" in message
    )


__all__ = ["is_transient_sqlite_lock"]
