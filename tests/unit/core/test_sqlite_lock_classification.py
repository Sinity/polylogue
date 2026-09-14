"""Runtime layers classify SQLite contention by result code, not message text.

Anti-vacuity: reverting any of these predicates to a substring test over the
exception message turns the "no such table: busy_locked_cursors" case green as
transient, and drops the SQLITE_LOCKED cases whose text the old tests missed.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

import pytest

from polylogue.core.sqlite_locking import is_transient_sqlite_lock
from polylogue.daemon.convergence_stages import _is_transient_sqlite_lock
from polylogue.daemon.cursor_lag_baseline import _database_is_locked
from polylogue.daemon.http import _is_sqlite_busy_error

_Predicate = Callable[[sqlite3.OperationalError], bool]

_PREDICATES: tuple[_Predicate, ...] = (
    is_transient_sqlite_lock,
    _is_transient_sqlite_lock,
    _database_is_locked,
    _is_sqlite_busy_error,
)


def _operational_error(message: str, *, errorcode: int | None, errorname: str | None) -> sqlite3.OperationalError:
    exc = sqlite3.OperationalError(message)
    if errorcode is not None:
        exc.sqlite_errorcode = errorcode
    if errorname is not None:
        exc.sqlite_errorname = errorname
    return exc


@pytest.mark.parametrize("predicate", _PREDICATES)
def test_shared_lock_codes_are_transient(predicate: _Predicate) -> None:
    busy = _operational_error("database is locked", errorcode=sqlite3.SQLITE_BUSY, errorname="SQLITE_BUSY")
    locked = _operational_error("database table is locked", errorcode=sqlite3.SQLITE_LOCKED, errorname="SQLITE_LOCKED")
    # Extended code: SQLITE_LOCKED_SHAREDCACHE keeps SQLITE_LOCKED in its low byte.
    shared_cache = _operational_error(
        "database table is locked: cache",
        errorcode=sqlite3.SQLITE_LOCKED | (1 << 8),
        errorname="SQLITE_LOCKED_SHAREDCACHE",
    )
    assert predicate(busy) is True
    assert predicate(locked) is True
    assert predicate(shared_cache) is True


@pytest.mark.parametrize("predicate", _PREDICATES)
def test_permanent_errors_are_never_read_as_contention(predicate: _Predicate) -> None:
    # A relation name carrying "busy" or "locked" must not read as contention:
    # a substring predicate defers this stage forever instead of surfacing it.
    missing_table = _operational_error(
        "no such table: busy_locked_cursors", errorcode=sqlite3.SQLITE_ERROR, errorname="SQLITE_ERROR"
    )
    malformed = _operational_error(
        "database disk image is malformed", errorcode=sqlite3.SQLITE_CORRUPT, errorname="SQLITE_CORRUPT"
    )
    assert predicate(missing_table) is False
    assert predicate(malformed) is False
