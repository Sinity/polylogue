"""Pinned-connection twins must carry their path twins' failure contracts.

A "pinned twin" is a reader that takes an already-open ``sqlite3.Connection``
instead of an archive path, copied from an existing path-taking sibling. The
copies were repeatedly landed without the sibling's conversion of a low-level
``sqlite3.Error``/``OSError`` into a typed domain error or a degraded return
value. Both twins agree on the happy path, so only a failure injection
separates them -- which is why the originals went unnoticed.

Each test below drives a real ``sqlite3.OperationalError`` through the pinned
reader by presenting a readable-but-damaged tier (the shape a partially
converged, concurrently written, or damaged archive presents), and asserts the
contracted degraded value emerges instead of the raw driver error.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from polylogue.analysis.schema_drift import schema_drift_status_from_connection
from polylogue.operations.status_workload import raw_failure_status_from_connection
from polylogue.storage.raw_retention import (
    RawRetentionSafetyError,
    _ops_cursor_byte_offsets_from_connection,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def test_pinned_schema_drift_reports_unavailable_when_the_ops_reader_fails() -> None:
    """The pinned drift reader shares ``schema_drift_status``'s contract.

    ``schema_drift_status`` converts ``sqlite3.Error`` from the drift-table
    probe and the summary read into ``{"available": False, "reason": ...}``
    (two separate handlers). The pinned twin ran the same statements bare.
    Its only caller, ``_schema_drift_status`` in
    ``polylogue.operations.daemon_status``, invokes it unguarded, so the raw
    driver error failed the whole status operation instead of degrading one
    component.

    ``schema_drift_status`` now delegates its projection here, so there is one
    body and one contract rather than two that drifted apart.

    Anti-vacuity: delete the ``except (OSError, sqlite3.Error)`` clause in
    ``schema_drift_status_from_connection`` and this goes red with
    ``sqlite3.OperationalError: no such table: ops_tier.sqlite_schema``.
    """
    with closing(sqlite3.connect(":memory:")) as conn:
        # A pinned handle whose ops schema is not attached: the shape a
        # detached/mid-recovery operation reader presents.
        status = schema_drift_status_from_connection(conn, now_ms=1_700_000_000_000, schema="ops_tier")

    assert status["available"] is False
    assert "sqlite_schema" in str(status["reason"])


def test_pinned_schema_drift_still_raises_on_a_caller_bug() -> None:
    """Tier degradation degrades; an unsupported schema stays a raise.

    Anti-vacuity: widen the wrapper to swallow ``ValueError`` and this goes
    red.
    """
    with closing(sqlite3.connect(":memory:")) as conn:
        with pytest.raises(ValueError, match="unsupported schema-drift reader schema"):
            schema_drift_status_from_connection(conn, now_ms=0, schema="index_tier")


def test_pinned_raw_failure_status_degrades_when_the_source_reader_fails(tmp_path: Path) -> None:
    """The pinned raw-failure reader shares its path sibling's contract.

    ``polylogue.daemon.status._archive_raw_failure_info`` converts
    ``(OSError, sqlite3.Error)`` from these same source-tier reads into an
    unavailable projection. The pinned twin ran ``raw_sessions`` sample and
    quarantine queries bare, and ``_raw_failure_status`` in
    ``polylogue.operations.daemon_status`` calls it unguarded.

    The tier is a real, fully readable source database, so the nested
    lifecycle read (which carries its own handler) succeeds and the projection
    proceeds to the quarantine ``COUNT(*)`` -- one of the statements the path
    twin's handler covers and this twin left bare. The failure is injected
    there, on a live handle, as ``SQLITE_BUSY``: the error a concurrent writer
    actually produces during a rebuild.

    The conversion runs through ``capture_sqlite_read`` -- the storage seam
    this module already uses for its two sibling pinned readers -- rather than
    a hand-written handler, which is what ``devtools gate layering`` requires.

    Anti-vacuity: drop the ``capture_sqlite_read`` wrapper in
    ``raw_failure_status_from_connection`` (call the body directly) and this
    goes red with ``sqlite3.OperationalError: database is locked`` escaping.
    """

    class _BusyOnQuarantineCount:
        """Delegates to a real reader, but fails the quarantine count."""

        def __init__(self, conn: sqlite3.Connection) -> None:
            self._conn = conn

        def execute(self, sql: str, *args: Any) -> object:
            if "parsed_at_ms IS NULL" in sql:
                raise sqlite3.OperationalError("database is locked")
            return self._conn.execute(sql, *args)

        def __getattr__(self, name: str) -> object:
            return getattr(self._conn, name)

    source_db = tmp_path / "source.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)

    with closing(sqlite3.connect(source_db)) as conn:
        status = raw_failure_status_from_connection(_BusyOnQuarantineCount(conn))  # type: ignore[arg-type]

    assert status["raw_failure_lifecycle_available"] is False
    assert status["raw_failure_lifecycle_state"] == "unavailable"
    assert "could not read source.db raw failure relations" in str(status["raw_failure_lifecycle_reason"])


def test_pinned_ops_cursor_read_raises_the_typed_retention_refusal() -> None:
    """The pinned cursor reader shares ``_ops_cursor_byte_offsets``'s contract.

    The path twin converts ``(OSError, sqlite3.Error)`` into
    ``RawRetentionSafetyError``. The pinned twin copied across only the typed
    refusal for a missing ``ingest_cursor`` table, leaving both of its
    ``conn.execute`` calls bare. Its sole caller
    ``_check_cursor_ahead_of_accepted`` catches ``RawRetentionSafetyError``,
    so the call site *reads* as protected while a locked or corrupt ops handle
    raised straight through it -- the most dangerous shape of this defect,
    because reviewing the caller alone concludes it is safe.

    Anti-vacuity: delete the ``except (OSError, sqlite3.Error)`` clause in
    ``_ops_cursor_byte_offsets_from_connection`` and this goes red with a bare
    ``sqlite3.OperationalError`` instead of ``RawRetentionSafetyError``.
    """
    with closing(sqlite3.connect(":memory:")) as conn:
        with pytest.raises(RawRetentionSafetyError, match="ops tier raw cursor authority is unreadable"):
            _ops_cursor_byte_offsets_from_connection(conn, schema="ops_tier")


def test_pinned_ops_cursor_read_keeps_its_missing_table_refusal() -> None:
    """The missing-table refusal is not swallowed by the new wrapper.

    Anti-vacuity: if the wrapper caught ``RawRetentionSafetyError`` (or
    ``Exception``) and degraded it, this goes red.
    """
    with closing(sqlite3.connect(":memory:")) as conn:
        with pytest.raises(RawRetentionSafetyError, match="ops tier has no ingest_cursor table"):
            _ops_cursor_byte_offsets_from_connection(conn)
