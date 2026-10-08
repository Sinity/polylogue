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
            VerifiedAuditLeaf(tmp_path)
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
        with VerifiedAuditLeaf(tmp_path) as leaf:
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
        VerifiedAuditLeaf(tmp_path)


@pytest.mark.parametrize("constructor_fault", [False, True])
def test_verified_leaf_failed_descriptor_cleanup_retains_existing_native_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, constructor_fault: bool
) -> None:
    from builtins import BaseExceptionGroup

    from polylogue.storage.sqlite.connection_profile import (
        NativeConnectionSettlementError,
        retained_native_sql_owners_on_current_thread,
    )

    path = tmp_path / "audit.db"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE evidence(value TEXT)")
        connection.commit()
    leaf = VerifiedAuditLeaf(tmp_path)
    real_close = os.close
    attempts: list[int] = []
    refused: list[int] = []
    primary = ValueError("synthetic audit leaf constructor failure")
    original_open = leaf._open_leaf

    def open_leaf() -> int:
        descriptor = original_open()
        refused.append(descriptor)
        return descriptor

    def close_before_effect(descriptor: int) -> None:
        attempts.append(descriptor)
        if descriptor in refused:
            raise OSError("synthetic leaf close before effect")
        real_close(descriptor)

    monkeypatch.setattr(leaf, "_open_leaf", open_leaf)
    monkeypatch.setattr(os, "close", close_before_effect)
    try:
        if constructor_fault:

            def fail_after_assignment() -> Path:
                raise primary

            monkeypatch.setattr(leaf, "_resolve_portable_child_path", fail_after_assignment)
            with pytest.raises(BaseExceptionGroup) as caught:
                leaf.__enter__()
            assert caught.value.exceptions[0] is primary
            failure = caught.value.exceptions[1]
            assert isinstance(failure, NativeConnectionSettlementError)
        else:
            leaf.__enter__()
            with pytest.raises(NativeConnectionSettlementError) as caught_native:
                leaf.close()
            failure = caught_native.value
        owner = failure.owner
        assert owner in retained_native_sql_owners_on_current_thread()
        assert owner.anchored_descriptors == tuple(refused)
        assert os.fstat(refused[0]).st_ino == path.stat().st_ino
        recorded = list(attempts)
        with pytest.raises(NativeConnectionSettlementError):
            leaf.close()
        assert attempts == recorded
    finally:
        monkeypatch.setattr(os, "close", real_close)
        for descriptor in refused:
            try:
                os.fstat(descriptor)
            except OSError:
                continue
            real_close(descriptor)
        leaf.close()
    assert owner not in retained_native_sql_owners_on_current_thread()
    leaf.close()
    assert attempts == recorded
