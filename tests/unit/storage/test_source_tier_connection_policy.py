"""Source-tier writer handles keep durability mode separate from local policy."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.blob_publication import BlobPublicationReceipt, BlobPublicationReservationStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.connection_profile import WRITE_CONNECTION_PROFILE, write_connection_pragma_statements

# The reviewed durability contract for the DURABLE source tier, written as
# literals on purpose. ``docs/durability-by-tier.md`` publishes source.db as
# "WAL, synchronous=NORMAL" with a named, bounded power-loss window, and the
# performance/logging addendum on polylogue-rk0it requires that no performance
# receipt silently change journal/synchronous policy. Deriving these from
# WRITE_CONNECTION_PROFILE -- the object under test -- made that unenforceable:
# flipping the profile to synchronous=OFF kept this whole module green while
# the durable tier lost the guarantee the document promises. Changing either
# literal is a deliberate edit of the durability contract and must move the
# document with it.
_DURABLE_SOURCE_JOURNAL_MODE = "wal"
_DURABLE_SOURCE_SYNCHRONOUS = 1  # NORMAL


def test_durable_source_tier_profile_declares_its_reviewed_durability_contract() -> None:
    """The source tier's write profile is WAL + NORMAL, as the document publishes.

    Anti-vacuity: setting WRITE_CONNECTION_PROFILE.synchronous to OFF (or
    journal_mode away from WAL) turns this red. That mutation is exactly the
    one every other test in this module tolerated before these literals
    existed, because they compared the live handle against the same profile.
    """
    pragmas = " ".join(write_connection_pragma_statements(WRITE_CONNECTION_PROFILE)).upper()
    assert WRITE_CONNECTION_PROFILE.journal_mode == "WAL"
    assert WRITE_CONNECTION_PROFILE.synchronous == "NORMAL"
    # The declared fields are what actually reach a connection.
    assert "SYNCHRONOUS=NORMAL" in pragmas.replace(" ", "")


def _writer_pragmas(conn: sqlite3.Connection) -> tuple[str, int, int, int]:
    return (
        str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower(),
        int(conn.execute("PRAGMA synchronous").fetchone()[0]),
        int(conn.execute("PRAGMA busy_timeout").fetchone()[0]),
        int(conn.execute("PRAGMA foreign_keys").fetchone()[0]),
    )


def _assert_source_writer_policy(conn: sqlite3.Connection) -> None:
    """Assert the original writer handle, never a reopened inspection connection."""
    assert _writer_pragmas(conn) == (
        _DURABLE_SOURCE_JOURNAL_MODE,
        _DURABLE_SOURCE_SYNCHRONOUS,
        WRITE_CONNECTION_PROFILE.busy_timeout_ms,
        1,
    )


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

    assert observed == [
        (
            _DURABLE_SOURCE_JOURNAL_MODE,
            _DURABLE_SOURCE_SYNCHRONOUS,
            WRITE_CONNECTION_PROFILE.busy_timeout_ms,
            1,
        )
    ]
    with sqlite3.connect(root / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone() == (1,)
