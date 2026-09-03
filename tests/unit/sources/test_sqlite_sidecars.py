"""Rollback journals are SQLite sidecars everywhere a sidecar set is declared."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.daemon.backup import _SQLITE_SIDECAR_SUFFIXES as BACKUP_SUFFIXES
from polylogue.sources.sqlite_snapshot import (
    _SQLITE_SIDECAR_SUFFIXES,
    sqlite_database_for_sidecar,
)
from polylogue.storage.sqlite.audit_leaf import _SQLITE_SIDECAR_SUFFIXES as AUDIT_SUFFIXES
from polylogue.storage.sqlite.migration_runner import _SQLITE_SIDECAR_SUFFIXES as MIGRATION_SUFFIXES


def test_a_rollback_journal_maps_back_to_its_database(tmp_path: Path) -> None:
    """Anti-vacuity: removing "-journal" from _SQLITE_SIDECAR_SUFFIXES returns None
    here, and the watcher then treats a journal event as an unknown path
    instead of work on the database beside it.

    A database in the default journal mode writes ``-journal``, not ``-wal``.
    """
    database = tmp_path / "state.db"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("CREATE TABLE turns (text TEXT)")

    assert sqlite_database_for_sidecar(database.with_name("state.db-journal")) == database
    assert sqlite_database_for_sidecar(database.with_name("state.db-wal")) == database
    assert sqlite_database_for_sidecar(database.with_name("state.db-shm")) == database
    assert sqlite_database_for_sidecar(database.with_name("state.notes")) is None


def test_every_declared_sidecar_set_agrees() -> None:
    """Anti-vacuity: reverting any one of the four constants makes this red.

    A staging discard that misses a suffix strands that sidecar beside a
    published blob, where it reads as damaged state.
    """
    expected = ("-wal", "-shm", "-journal")
    assert expected == _SQLITE_SIDECAR_SUFFIXES
    assert expected == BACKUP_SUFFIXES
    assert expected == AUDIT_SUFFIXES
    assert expected == MIGRATION_SUFFIXES
