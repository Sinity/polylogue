"""Source-tier writer handles keep durability mode separate from local policy."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.blob_publication import BlobPublicationReceipt, BlobPublicationReservationStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.connection_profile import WRITE_CONNECTION_PROFILE, write_connection_pragma_statements


def _expected_synchronous() -> int:
    statement = next(
        statement
        for statement in write_connection_pragma_statements(WRITE_CONNECTION_PROFILE)
        if "synchronous" in statement
    )
    value = statement.rsplit("=", maxsplit=1)[1].strip().upper()
    return {"OFF": 0, "NORMAL": 1, "FULL": 2, "EXTRA": 3}[value]


def _writer_pragmas(conn: sqlite3.Connection) -> tuple[str, int, int, int]:
    return (
        str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower(),
        int(conn.execute("PRAGMA synchronous").fetchone()[0]),
        int(conn.execute("PRAGMA busy_timeout").fetchone()[0]),
        int(conn.execute("PRAGMA foreign_keys").fetchone()[0]),
    )


def _assert_source_writer_policy(conn: sqlite3.Connection) -> None:
    """Assert the original writer handle, never a reopened inspection connection."""
    assert _writer_pragmas(conn) == ("wal", _expected_synchronous(), WRITE_CONNECTION_PROFILE.busy_timeout_ms, 1)


def test_fresh_archive_source_writer_handle_uses_durable_mode_and_local_policy(tmp_path: Path) -> None:
    """Fresh bootstrap makes source.db WAL before the persistent writer applies NORMAL.

    Anti-vacuity: deleting source database-mode initialization leaves this
    actual source handle in rollback-journal mode; omitting the local policy
    leaves its synchronous/busy-timeout values at SQLite defaults.
    """
    root = tmp_path / "archive"

    with ArchiveStore(root, initialize=True, read_only=False) as archive:
        _assert_source_writer_policy(archive._ensure_source_conn())


def test_existing_wal_source_writer_reuses_its_handle_without_reconfiguring_database_mode(tmp_path: Path) -> None:
    """A second source writer opens while a current WAL owner holds the write lock.

    The returned handle must receive only connection-local policy while the
    current owner holds the write lock. Database-mode changes belong to fresh
    bootstrap, where no active source-tier transaction can be disturbed.

    Anti-vacuity: bypassing the source-tier factory leaves the actual handle's
    synchronous/busy-timeout policy at SQLite's defaults.
    """
    root = tmp_path / "archive"
    with ArchiveStore(root, initialize=True, read_only=False):
        pass

    holder = sqlite3.connect(root / "source.db")
    holder.execute("BEGIN IMMEDIATE")
    try:
        with ArchiveStore(root, initialize=False, read_only=False) as archive:
            source = archive._ensure_source_conn()
            assert source is archive._ensure_source_conn()
            _assert_source_writer_policy(source)
    finally:
        holder.rollback()
        holder.close()


def test_reservation_path_observes_its_actual_source_writer_handle_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reservation applies the policy before it starts its own short transaction.

    The probe wraps the production connection-opening seam but delegates to it
    unchanged, so the assertions inspect the live handle used by
    ``reserve_many`` rather than a reopened connection with unrelated local
    synchronous state.

    Anti-vacuity: bypassing the source-tier factory in ``reserve_many`` leaves
    the captured handle at SQLite's FULL/5-second defaults.
    """
    root = tmp_path / "archive"
    with ArchiveStore(root, initialize=True, read_only=False):
        pass

    observed: list[tuple[str, int, int, int]] = []
    original = BlobPublicationReservationStore._open_connection

    def observe_open(store: BlobPublicationReservationStore) -> sqlite3.Connection:
        conn = original(store)
        observed.append(_writer_pragmas(conn))
        return conn

    monkeypatch.setattr(BlobPublicationReservationStore, "_open_connection", observe_open)
    BlobPublicationReservationStore(root / "source.db").reserve_many(
        [BlobPublicationReceipt("receipt", "00" * 32, 1, "test-publisher")]
    )

    assert observed == [("wal", _expected_synchronous(), WRITE_CONNECTION_PROFILE.busy_timeout_ms, 1)]
    with sqlite3.connect(root / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone() == (1,)
