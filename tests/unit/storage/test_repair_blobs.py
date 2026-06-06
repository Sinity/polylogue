"""Blob cleanup repair contracts."""

from __future__ import annotations

from pathlib import Path

from polylogue.config import Config
from polylogue.paths import blob_store_root
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.repair import repair_orphaned_blobs
from polylogue.storage.sqlite.connection import connection_context
from polylogue.storage.sqlite.connection_profile import open_connection
from tests.infra.storage_records import db_setup


def _config(workspace_env: dict[str, Path], db_path: Path) -> Config:
    return Config(
        archive_root=workspace_env["archive_root"],
        render_root=workspace_env["archive_root"] / "render",
        sources=[],
        db_path=db_path,
    )


def _reference_blob_in_source_db(source_db_path: Path, blob_hash: str, blob_size: int) -> None:
    with open_connection(source_db_path) as conn:
        conn.execute(
            """CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                blob_hash BLOB NOT NULL,
                blob_size INTEGER NOT NULL DEFAULT 0
            ) STRICT"""
        )
        conn.execute(
            "INSERT INTO raw_sessions (raw_id, blob_hash, blob_size) VALUES (?, ?, ?)",
            ("raw-v1", bytes.fromhex(blob_hash), blob_size),
        )
        conn.commit()


def test_repair_orphaned_blobs_dry_run_accounts_full_orphan_set(
    workspace_env: dict[str, Path],
) -> None:
    db_path = db_setup(workspace_env)
    with connection_context(db_path):
        pass
    store = BlobStore(blob_store_root())
    referenced_hash, referenced_size = store.write_from_bytes(b"referenced")
    orphan_a, orphan_a_size = store.write_from_bytes(b"orphan-a")
    orphan_b, orphan_b_size = store.write_from_bytes(b"orphan-b")
    _reference_blob_in_source_db(db_path.with_name("source.db"), referenced_hash, referenced_size)

    result = repair_orphaned_blobs(_config(workspace_env, db_path), dry_run=True)

    assert result.success is True
    assert result.repaired_count == 2
    assert str(orphan_a_size + orphan_b_size) in result.detail
    assert store.exists(orphan_a)
    assert store.exists(orphan_b)
    assert store.exists(referenced_hash)


def test_repair_orphaned_blobs_deletes_only_unreferenced_blobs(
    workspace_env: dict[str, Path],
) -> None:
    db_path = db_setup(workspace_env)
    with connection_context(db_path):
        pass
    store = BlobStore(blob_store_root())
    referenced_hash, referenced_size = store.write_from_bytes(b"referenced")
    orphan_hash, _ = store.write_from_bytes(b"orphan")
    _reference_blob_in_source_db(db_path.with_name("source.db"), referenced_hash, referenced_size)

    result = repair_orphaned_blobs(_config(workspace_env, db_path), dry_run=False)

    assert result.success is True
    assert result.repaired_count == 1
    assert store.exists(referenced_hash)
    assert not store.exists(orphan_hash)


def test_repair_orphaned_blobs_preserves_archive_source_references(
    workspace_env: dict[str, Path],
) -> None:
    db_path = db_setup(workspace_env)
    with connection_context(db_path):
        pass
    store = BlobStore(blob_store_root())
    referenced_hash, referenced_size = store.write_from_bytes(b"archive referenced")
    orphan_hash, _ = store.write_from_bytes(b"archive orphan")
    _reference_blob_in_source_db(db_path.with_name("source.db"), referenced_hash, referenced_size)

    result = repair_orphaned_blobs(_config(workspace_env, db_path), dry_run=False)

    assert result.success is True
    assert result.repaired_count == 1
    assert "source.db.raw_sessions" in result.detail
    assert store.exists(referenced_hash)
    assert not store.exists(orphan_hash)
