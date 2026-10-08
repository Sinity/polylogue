"""Runtime layers classify SQLite contention by result code, not message text.

Anti-vacuity: reverting any of these predicates to a substring test over the
exception message turns the "no such table: busy_locked_cursors" case green as
transient, and drops the SQLITE_LOCKED cases whose text the old tests missed.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Never, cast

import pytest

from polylogue.core.sqlite_locking import is_transient_sqlite_lock
from polylogue.daemon.cursor_lag_baseline import _database_is_locked
from polylogue.daemon.http import _is_sqlite_busy_error

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


_Predicate = Callable[[sqlite3.OperationalError], bool]

_PREDICATES: tuple[_Predicate, ...] = (
    is_transient_sqlite_lock,
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
    protocol = _operational_error("WAL race", errorcode=sqlite3.SQLITE_PROTOCOL, errorname="SQLITE_PROTOCOL")
    assert predicate(protocol) is True
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
    from polylogue.storage.fts.fts_lifecycle import search_index_read_refusal

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

    missing_reason = search_index_read_refusal(missing)
    assert missing_reason is not None and missing_reason.reason == "fts_missing"
    corrupt_reason = search_index_read_refusal(corrupt)
    assert corrupt_reason is not None and corrupt_reason.reason == "archive_unreadable"
    assert corrupt_reason.reason != missing_reason.reason
    busy_reason = search_index_read_refusal(busy)
    assert busy_reason is not None and busy_reason.reason == "archive_busy"
    assert busy_reason.reason != missing_reason.reason


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


@pytest.mark.parametrize("predicate", _PREDICATES)
def test_protocol_name_is_transient_and_corruption_code_takes_precedence(predicate: _Predicate) -> None:
    protocol = _operational_error("WAL race", errorcode=None, errorname="SQLITE_PROTOCOL")
    assert predicate(protocol) is True
    for code in (sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB, sqlite3.SQLITE_IOERR):
        error = _operational_error("locking protocol", errorcode=code, errorname="SQLITE_PROTOCOL")
        assert predicate(error) is False


@pytest.mark.parametrize(
    ("message", "code", "reason"),
    [
        ("no such table: messages_fts", sqlite3.SQLITE_ERROR, "fts_missing"),
        ("malformed messages_fts", sqlite3.SQLITE_CORRUPT, "archive_unreadable"),
        ("locked messages_fts", sqlite3.SQLITE_BUSY, "archive_busy"),
        ("no such table: unrelated", sqlite3.SQLITE_ERROR, None),
        ("syntax error near messages_fts", sqlite3.SQLITE_ERROR, None),
    ],
)
def test_canonical_search_read_classifies_index_failure_and_preserves_unknown_sql(
    tmp_path: Path,
    message: str,
    code: int,
    reason: str | None,
) -> None:
    from polylogue.archive.query.archive_execution import archive_search_hits
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.core.errors import SearchIndexUnavailableError

    original = _operational_error(message, errorcode=code, errorname=None)

    class FailedReader:
        def search_summaries(self, *args: object, **kwargs: object) -> Never:
            raise original

    plan = SessionQuerySpec.from_params({"query": "needle", "limit": 10}).to_plan()
    with pytest.raises(SearchIndexUnavailableError if reason else sqlite3.OperationalError) as raised:
        archive_search_hits(plan, archive_root=tmp_path, config=None, archive=cast("ArchiveStore", FailedReader()))
    if reason:
        assert isinstance(raised.value, SearchIndexUnavailableError)
        assert raised.value.reason == reason
        assert raised.value.__cause__ is original
    else:
        assert raised.value is original


@pytest.mark.parametrize(("exists", "reason"), [(False, "fts_missing"), (True, "fts_incomplete")])
def test_fts_readiness_raises_shared_typed_refusal(exists: bool, reason: str) -> None:
    from polylogue.core.errors import SearchIndexUnavailableError
    from polylogue.storage.fts.fts_lifecycle import check_fts_readiness

    with pytest.raises(SearchIndexUnavailableError) as raised:
        check_fts_readiness({"exists": exists, "ready": False})
    assert raised.value.reason == reason
    check_fts_readiness({"exists": True, "ready": True})
