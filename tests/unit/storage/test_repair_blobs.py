"""Blob cleanup repair contracts."""

from __future__ import annotations

import os
import time
from pathlib import Path

from polylogue.config import Config
from polylogue.paths import blob_store_root
from polylogue.storage.blob_gc import MIN_AGE_S
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
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index,
                blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "raw-v1",
                "codex-session",
                "raw-v1",
                "/tmp/raw-v1.jsonl",
                0,
                bytes.fromhex(blob_hash),
                blob_size,
                1_770_000_000_000,
            ),
        )
        conn.commit()


def _make_gc_eligible(store: BlobStore, *blob_hashes: str) -> None:
    eligible_mtime = time.time() - MIN_AGE_S - 5
    for blob_hash in blob_hashes:
        os.utime(store.blob_path(blob_hash), (eligible_mtime, eligible_mtime))


def test_repair_orphaned_blobs_dry_run_accounts_full_orphan_set(
    workspace_env: dict[str, Path],
) -> None:
    db_path = db_setup(workspace_env)
    with connection_context(db_path):
        pass
    store = BlobStore(blob_store_root())
    referenced_hash, referenced_size = store.write_from_bytes(b"referenced")
    orphan_a, _ = store.write_from_bytes(b"orphan-a")
    orphan_b, _ = store.write_from_bytes(b"orphan-b")
    _reference_blob_in_source_db(db_path.with_name("source.db"), referenced_hash, referenced_size)
    _make_gc_eligible(store, referenced_hash, orphan_a, orphan_b)

    result = repair_orphaned_blobs(_config(workspace_env, db_path), dry_run=True)

    assert result.success is True
    assert result.repaired_count == 2
    assert result.detail == "Would: delete 2 orphaned blobs; skipped 1 referenced"
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
    _make_gc_eligible(store, referenced_hash, orphan_hash)

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
    _make_gc_eligible(store, referenced_hash, orphan_hash)

    result = repair_orphaned_blobs(_config(workspace_env, db_path), dry_run=False)

    assert result.success is True
    assert result.repaired_count == 1
    assert "skipped 1 referenced" in result.detail
    assert store.exists(referenced_hash)
    assert not store.exists(orphan_hash)
