"""Off-gate metadata custody preserves an existing SQLite reader's locks."""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.sqlite.audit_leaf import AuditLeafError, VerifiedAuditLeaf
from tests.infra.sqlite_lock_probe import sqlite_lock_state

pytestmark = pytest.mark.uses_real_clock


def test_observer_metadata_custody_preserves_main_and_shm_locks(tmp_path: Path) -> None:
    if not hasattr(os, "O_PATH"):
        with pytest.raises(AuditLeafError):
            VerifiedAuditLeaf(tmp_path, identity_access="lock-preserving")
        return
    path = tmp_path / "audit.db"
    with closing(sqlite3.connect(path)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE evidence(value TEXT)")
        writer.execute("INSERT INTO evidence VALUES ('neutral')")
        writer.commit()
        writer.execute("BEGIN")
        assert writer.execute("SELECT value FROM evidence").fetchone() == ("neutral",)
        expected = {"main": "protected", "shm": "protected"}
        assert sqlite_lock_state(path) == expected
        with VerifiedAuditLeaf(tmp_path, identity_access="lock-preserving") as leaf:
            leaf.assert_unchanged()
            assert leaf.identity_metadata().st_ino == path.stat().st_ino
            assert sqlite_lock_state(path) == expected
            leaf.assert_unchanged()
            assert sqlite_lock_state(path) == expected
        assert sqlite_lock_state(path) == expected


def test_observer_metadata_custody_refuses_missing_platform_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(os, "O_PATH", raising=False)
    with pytest.raises(AuditLeafError):
        VerifiedAuditLeaf(tmp_path, identity_access="lock-preserving")
