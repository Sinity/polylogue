from __future__ import annotations

from pathlib import Path
from typing import cast

from polylogue.storage.blob_store import BlobStore
from polylogue.storage.io_phase_metrics import io_phase_process_snapshot, io_phase_snapshot
from polylogue.storage.sqlite.connection_profile import (
    open_isolated_write_connection,
    open_readonly_connection,
    open_source_tier_write_connection,
)
from polylogue.storage.sqlite.wal_checkpoint import checkpoint_connection
from polylogue.storage.sqlite.write_lease import write_lease


def _count(tier: str, phase: str, inside: bool) -> int:
    return sum(
        sample.count
        for sample in io_phase_snapshot()
        if sample.tier == tier and sample.phase == phase and sample.inside_writer_lease is inside and sample.succeeded
    )


def test_returned_source_handle_records_actual_transaction_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "source.db"
    before = {phase: _count("source", phase, True) for phase in ("connection_create", "begin", "commit", "rollback")}
    with write_lease("io-phase-test", archive_root=tmp_path):
        conn = open_source_tier_write_connection(path, archive_root=tmp_path)
        try:
            conn.execute("CREATE TABLE sample (value INTEGER)")
            conn.execute("INSERT INTO sample VALUES (1)")
            conn.commit()
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO sample VALUES (2)")
            conn.commit()
            conn.execute("BEGIN")
            conn.execute("INSERT INTO sample VALUES (3)")
            conn.rollback()
            assert conn.execute("SELECT value FROM sample ORDER BY value").fetchall() == [(1,), (2,)]
        finally:
            conn.close()
    assert _count("source", "connection_create", True) - before["connection_create"] == 1
    assert _count("source", "begin", True) - before["begin"] == 2
    assert _count("source", "commit", True) - before["commit"] >= 2
    assert _count("source", "rollback", True) - before["rollback"] == 1
    outside_before = _count("source", "connection_create", False)
    reader = open_readonly_connection(path, validate_schema=False)
    reader.close()
    assert _count("source", "connection_create", False) - outside_before == 1
    payload = io_phase_process_snapshot()
    assert payload["scope"] == "process"
    assert isinstance(payload["pid"], int)
    samples = cast(list[dict[str, object]], payload["samples"])
    assert any(sample["tier"] == "source" and sample["phase"] == "connection_create" for sample in samples)


def test_context_manager_commit_is_counted_once(tmp_path: Path) -> None:
    path = tmp_path / "source.db"
    before = _count("source", "commit", True)
    with write_lease("io-context-test", archive_root=tmp_path):
        conn = open_source_tier_write_connection(path, archive_root=tmp_path)
        try:
            with conn:
                conn.execute("CREATE TABLE sample (value INTEGER)")
                conn.execute("INSERT INTO sample VALUES (1)")
        finally:
            conn.close()
    assert _count("source", "commit", True) - before == 1


def test_checkpoint_and_blob_syncs_are_counted_at_the_syscalls(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    before_checkpoint = _count("index", "checkpoint", True)
    with write_lease("io-checkpoint-test", archive_root=tmp_path):
        conn = open_isolated_write_connection(db, purpose="io phase checkpoint test", archive_root=tmp_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE sample (value INTEGER)")
            conn.commit()
            checkpoint_connection(conn, "PASSIVE", boundary="recurring")
        finally:
            conn.close()
    assert _count("index", "checkpoint", True) - before_checkpoint == 1

    before_file = _count("source", "blob_file_fsync", False)
    before_directory = _count("source", "blob_directory_fsync", False)
    BlobStore(tmp_path / "blobs").write_from_bytes(b"synthetic blob")
    assert _count("source", "blob_file_fsync", False) - before_file == 1
    assert _count("source", "blob_directory_fsync", False) - before_directory >= 1
