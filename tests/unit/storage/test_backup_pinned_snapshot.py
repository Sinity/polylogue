"""Archive backup retains the pinned SQLite cut without draining old readers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import stat
import threading
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.compute_cancel import compute_cancel
from polylogue.operations.archive_backup import backup_archive
from polylogue.storage import backup_package as backup
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    open_scratch_connection,
    retained_native_sql_owners_on_current_thread,
)
from polylogue.storage.sqlite.migration_runner import MigrationError, validate_migration_backup_manifest
from tests.infra.storage_records import db_setup


@pytest.mark.parametrize("later_commit", [False, True])
def test_public_backup_preserves_pinned_cut_with_old_reader_and_later_commit(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, later_commit: bool
) -> None:
    db_setup(workspace_env)
    root = workspace_env["archive_root"]
    source = root / "user.db"
    source.chmod(0o600)
    with closing(sqlite3.connect(source)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE backup_events(value TEXT NOT NULL)")
        writer.execute("INSERT INTO backup_events VALUES ('old')")
        writer.commit()
        with closing(sqlite3.connect(source)) as old_reader:
            old_reader.execute("BEGIN")
            assert old_reader.execute("SELECT value FROM backup_events").fetchall() == [("old",)]
            writer.execute("INSERT INTO backup_events VALUES ('at-cut')")
            writer.commit()
            original_open = backup._open_backup_readonly_connection
            reached = False

            def open_snapshot(path: Path, **kwargs: Any) -> sqlite3.Connection:
                connection = original_open(path, **kwargs)
                if path == source.resolve():
                    original_backup = connection.backup

                    def backup_after_later_commit(target: sqlite3.Connection, **options: Any) -> None:
                        nonlocal reached
                        reached = True
                        assert connection.in_transaction
                        assert connection.execute("SELECT value FROM backup_events ORDER BY rowid").fetchall() == [
                            ("old",),
                            ("at-cut",),
                        ]
                        if later_commit:
                            writer.execute("INSERT INTO backup_events VALUES ('after-cut')")
                            writer.commit()
                        original_backup(target, **options)

                    monkeypatch.setattr(connection, "backup", backup_after_later_commit)
                return connection

            monkeypatch.setattr(backup, "_open_backup_readonly_connection", open_snapshot)
            result = backup_archive(output_dir=tmp_path / "backups", profile="user_overlays", verify=True)
            assert result.ok and result.verified, result.error
            assert reached
            assert old_reader.execute("SELECT value FROM backup_events").fetchall() == [("old",)]
            expected = [("old",), ("at-cut",)] + ([("after-cut",)] if later_commit else [])
            assert writer.execute("SELECT value FROM backup_events ORDER BY rowid").fetchall() == expected
            manifest_path = Path(str(result.output_path)) / "manifest.json"
            if later_commit:
                with pytest.raises(MigrationError):
                    validate_migration_backup_manifest(manifest_path, ArchiveTier.USER, connection=writer)
            else:
                validate_migration_backup_manifest(manifest_path, ArchiveTier.USER, connection=writer)
                writer.execute("BEGIN IMMEDIATE")
                validate_migration_backup_manifest(manifest_path, ArchiveTier.USER, connection=writer)
                writer.rollback()
                writer.execute("INSERT INTO backup_events VALUES ('after-backup')")
                writer.commit()
                with pytest.raises(MigrationError):
                    validate_migration_backup_manifest(manifest_path, ArchiveTier.USER, connection=writer)
    package = Path(str(result.output_path))
    copied = package / "user.db"
    with closing(sqlite3.connect(copied)) as restored:
        assert restored.execute("SELECT value FROM backup_events ORDER BY rowid").fetchall() == [("old",), ("at-cut",)]
    manifest = json.loads((package / "manifest.json").read_text())
    snapshot = manifest["tier_source_fingerprints"]["user.db"]["snapshot"]
    assert manifest["tier_source_fingerprints"]["user.db"]["live_cut_stable"] is (not later_commit)
    assert snapshot["sha256"] == hashlib.sha256(copied.read_bytes()).hexdigest()
    assert snapshot["size_bytes"] == copied.stat().st_size
    assert stat.S_IMODE(copied.stat().st_mode) == 0o600
    assert (package / "verification-receipt.json").is_file()
    assert not list(package.glob("*.db-wal"))


def test_public_backup_cancellation_settles_both_snapshot_handles_before_publication(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_setup(workspace_env)
    cancelled = threading.Event()
    readers: list[sqlite3.Connection] = []
    destinations: list[sqlite3.Connection] = []
    original_open = backup._open_backup_readonly_connection
    original_destination = open_scratch_connection

    def open_snapshot(path: Path, **kwargs: Any) -> sqlite3.Connection:
        conn = original_open(path, **kwargs)
        readers.append(conn)
        original_backup = conn.backup

        def cancel_during_copy(target: sqlite3.Connection, **options: Any) -> None:
            progress = options["progress"]

            def cancel_at_page(status: int, remaining: int, total: int) -> None:
                cancelled.set()
                progress(status, remaining, total)

            options["progress"] = cancel_at_page
            original_backup(target, **options)

        monkeypatch.setattr(conn, "backup", cancel_during_copy)
        return conn

    def open_destination(path: Path, **kwargs: Any) -> Any:
        owner = original_destination(path, **kwargs)
        destinations.append(owner.require_connection())
        return owner

    monkeypatch.setattr(backup, "_open_backup_readonly_connection", open_snapshot)
    monkeypatch.setattr(backup, "open_scratch_connection", open_destination)
    token = compute_cancel.set(cancelled)
    try:
        with pytest.raises(DaemonOperationCancelled):
            backup_archive(output_dir=tmp_path / "backups", profile="user_overlays", verify=True)
    finally:
        compute_cancel.reset(token)
    assert readers and destinations
    for connection in [*readers, *destinations]:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
    assert not list((tmp_path / "backups").rglob("verification-receipt.json"))
    assert not retained_native_sql_owners_on_current_thread()


def test_public_backup_failed_destination_close_retains_its_exact_creator_custody(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_setup(workspace_env)
    original_destination = open_scratch_connection
    original_error = OSError("synthetic backup destination close failure")
    reached = []

    def open_destination(path: Path, **kwargs: Any) -> Any:
        owner = original_destination(path, **kwargs)
        conn = owner.require_connection()
        close = conn.close
        failed = False

        def fail_once() -> None:
            nonlocal failed
            if not failed:
                failed = True
                raise original_error
            close()

        monkeypatch.setattr(conn, "close", fail_once)
        reached.append(owner)
        return owner

    monkeypatch.setattr(backup, "open_scratch_connection", open_destination)
    with pytest.raises(NativeConnectionSettlementError) as failure:
        backup_archive(output_dir=tmp_path / "backups", profile="user_overlays", verify=True)
    assert len(reached) == 1 and failure.value.owner is reached[0]
    owner = reached[0]
    assert failure.value.failure is original_error
    assert owner in retained_native_sql_owners_on_current_thread()
    assert not list((tmp_path / "backups").rglob("verification-receipt.json"))
    owner.close()
    assert owner not in retained_native_sql_owners_on_current_thread()
