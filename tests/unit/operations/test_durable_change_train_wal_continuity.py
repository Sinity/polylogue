"""polylogue-yfc4f: the live source.db continuity check reads the WAL."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.operations.durable_change_train import _validate_audit_adoption_continuity
from polylogue.storage.sqlite.migration_runner import MigrationError


def test_uncheckpointed_wal_version_is_not_ignored(tmp_path: Path) -> None:
    """Commit user_version 32 into the WAL without a checkpoint. With
    ``immutable=1`` the reader saw the pre-migration image (version 0) and
    returned silently; the WAL-aware read sees 32 and, with no control table,
    raises. Anti-vacuity: a fixture that checkpoints makes both readers agree.
    """
    source = tmp_path / "source.db"
    holder = sqlite3.connect(source)
    holder.execute("PRAGMA journal_mode = WAL")
    holder.execute("PRAGMA wal_autocheckpoint = 0")
    holder.execute("CREATE TABLE placeholder (id INTEGER PRIMARY KEY)")
    holder.execute("PRAGMA user_version = 32")
    holder.commit()
    assert (tmp_path / "source.db-wal").stat().st_size > 0
    try:
        with pytest.raises(MigrationError, match="audit adoption continuity requires source audit_continuity_control"):
            _validate_audit_adoption_continuity(tmp_path, receipt_payload={}, expected_initial_file_identity=None)
    finally:
        holder.close()
