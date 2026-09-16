"""SQLite contention classification shared by runtime layers."""

from __future__ import annotations

import sqlite3

_SQLITE_PRIMARY_RESULT_CODE_MASK = 0xFF
_TRANSIENT_SQLITE_LOCK_CODES = frozenset({sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED})
_TRANSIENT_SQLITE_LOCK_NAMES = frozenset({"SQLITE_BUSY", "SQLITE_LOCKED", "SQLITE_LOCKED_SHAREDCACHE"})
_CORRUPT_SQLITE_CODES = frozenset({sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB, sqlite3.SQLITE_IOERR})
_CORRUPT_SQLITE_NAME_PREFIXES = ("SQLITE_CORRUPT", "SQLITE_NOTADB", "SQLITE_IOERR")


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


def is_corrupt_sqlite_database(exc: BaseException) -> bool:
    """Return whether SQLite reported unreadable storage rather than absent content.

    ``SQLITE_CORRUPT`` ("database disk image is malformed"), ``SQLITE_NOTADB``
    and every ``SQLITE_IOERR`` extended code describe a database the caller
    cannot answer *any* content question from.  Publishing such a condition as
    a measured negative -- "not embedded", "no derived models", "index missing"
    -- turns unreadable storage into a clean fact, so callers classify it here
    and report a typed unavailable/degraded condition instead.  Result metadata
    is preferred over message text for the same reason
    ``is_transient_sqlite_lock`` prefers it.
    """
    if not isinstance(exc, sqlite3.Error):
        return False
    error_code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(error_code, int):
        return error_code & _SQLITE_PRIMARY_RESULT_CODE_MASK in _CORRUPT_SQLITE_CODES
    error_name = getattr(exc, "sqlite_errorname", None)
    if isinstance(error_name, str):
        return error_name.startswith(_CORRUPT_SQLITE_NAME_PREFIXES)
    message = str(exc).lower()
    return (
        "database disk image is malformed" in message
        or "file is not a database" in message
        or "disk i/o error" in message
    )


__all__ = ["is_corrupt_sqlite_database", "is_transient_sqlite_lock"]
