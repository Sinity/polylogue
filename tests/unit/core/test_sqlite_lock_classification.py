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


def test_corrupt_storage_is_not_a_transient_lock_and_is_classified_as_corruption() -> None:
    """Unreadable storage is typed as corruption, never as contention or absence.

    Anti-vacuity: an exception carrying SQLITE_CORRUPT but whose text mentions
    neither corruption nor locking is only classified correctly by the result
    code.  A message-text predicate returns False here, reddening the first
    assertion.
    """
    from polylogue.core.sqlite_locking import is_corrupt_sqlite_database

    corrupt = _operational_error(
        "malformed database schema (messages_fts)",
        errorcode=sqlite3.SQLITE_CORRUPT,
        errorname="SQLITE_CORRUPT",
    )
    assert is_corrupt_sqlite_database(corrupt) is True
    assert is_transient_sqlite_lock(corrupt) is False

    busy = _operational_error("database is locked", errorcode=sqlite3.SQLITE_BUSY, errorname="SQLITE_BUSY")
    assert is_corrupt_sqlite_database(busy) is False

    missing = _operational_error(
        "no such table: messages_fts", errorcode=sqlite3.SQLITE_ERROR, errorname="SQLITE_ERROR"
    )
    assert is_corrupt_sqlite_database(missing) is False


def test_search_index_reason_separates_corruption_and_contention_from_a_missing_table() -> None:
    """A corrupt or busy FTS read is not reported as an ordinary missing index.

    Anti-vacuity: all three exceptions below mention ``messages_fts``, so the
    original substring-only reason returns the identical
    "missing or degraded" sentence for every one of them and both inequality
    assertions go red.
    """
    from polylogue.daemon.http import _search_index_degraded_reason

    missing = _operational_error(
        "no such table: messages_fts", errorcode=sqlite3.SQLITE_ERROR, errorname="SQLITE_ERROR"
    )
    corrupt = _operational_error(
        "database disk image is malformed: messages_fts",
        errorcode=sqlite3.SQLITE_CORRUPT,
        errorname="SQLITE_CORRUPT",
    )
    busy = _operational_error(
        "database is locked: messages_fts", errorcode=sqlite3.SQLITE_BUSY, errorname="SQLITE_BUSY"
    )

    missing_reason = _search_index_degraded_reason(missing)
    assert missing_reason is not None and "missing or degraded" in missing_reason
    corrupt_reason = _search_index_degraded_reason(corrupt)
    assert corrupt_reason is not None and "unreadable" in corrupt_reason
    assert corrupt_reason != missing_reason
    busy_reason = _search_index_degraded_reason(busy)
    assert busy_reason is not None and "busy" in busy_reason
    assert busy_reason != missing_reason


def test_readiness_reports_unreadable_derived_models_instead_of_an_empty_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A locked or corrupt derived-model probe is degradation, not absence.

    Anti-vacuity: with the bare ``except sqlite3.OperationalError: return {}``
    restored, both raising cases return ``{}`` and the ``pytest.raises`` blocks
    are red; the missing-table case proves the honest absence path survives.
    """
    import polylogue.readiness as readiness_module
    from polylogue.storage.derived import derived_status as derived_status_module

    def _raising(exc: BaseException) -> Callable[..., dict[str, object]]:
        def _collect(conn: object, *, verify_full: bool) -> dict[str, object]:
            raise exc

        return _collect

    for exc in (
        _operational_error(
            "database disk image is malformed", errorcode=sqlite3.SQLITE_CORRUPT, errorname="SQLITE_CORRUPT"
        ),
        _operational_error("database is locked", errorcode=sqlite3.SQLITE_BUSY, errorname="SQLITE_BUSY"),
    ):
        monkeypatch.setattr(derived_status_module, "collect_derived_model_statuses_sync", _raising(exc))
        with pytest.raises(sqlite3.OperationalError):
            readiness_module._collect_table_status_best_effort(sqlite3.connect(":memory:"), deep=True, probe_only=False)

    monkeypatch.setattr(
        derived_status_module,
        "collect_derived_model_statuses_sync",
        _raising(
            _operational_error(
                "no such table: session_profiles", errorcode=sqlite3.SQLITE_ERROR, errorname="SQLITE_ERROR"
            )
        ),
    )
    assert (
        readiness_module._collect_table_status_best_effort(sqlite3.connect(":memory:"), deep=True, probe_only=False)
        == {}
    )
