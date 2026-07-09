"""Coverage for the ops-doctor orphan-blob repair path (polylogue-8jg9.4).

``repair_orphaned_blobs_data`` previously called ``BlobStore.detect_orphans``/
``cleanup_orphans`` directly, comparing disk hashes only against committed
``raw_sessions``/``blob_refs`` rows with no generation-age floor. The fix
routes the doctor path through ``run_blob_gc_report`` instead, the same safe
planner ``run_blob_gc`` already uses, so both surfaces share one safety
implementation (committed reference + generation-age floor;
``docs/internals.md`` "GC concurrency model").

This file used to also cover a blob-GC lease mechanism
(``acquire_blob_leases``/``release_operation_leases``); that mechanism was
removed as unreachable dead code (polylogue-v7e0 — no production ingest
caller ever populated the lease payload keys) and its concurrency tests were
removed with it. What remains here is the still-live reference-check and
doctor-repair-path coverage.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from polylogue.storage.blob_gc import MIN_AGE_S, run_blob_gc
from polylogue.storage.blob_store import BlobStore


def _make_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE raw_sessions (
            raw_id TEXT PRIMARY KEY,
            source_name TEXT NOT NULL DEFAULT '',
            source_path TEXT NOT NULL DEFAULT '',
            blob_hash BLOB,
            blob_size INTEGER NOT NULL DEFAULT 0,
            acquired_at TEXT NOT NULL DEFAULT ''
        )"""
    )
    conn.execute(
        """CREATE TABLE blob_refs (
            blob_hash BLOB NOT NULL CHECK(length(blob_hash) = 32),
            ref_id TEXT NOT NULL,
            ref_type TEXT NOT NULL CHECK(ref_type IN ('raw_payload', 'attachment', 'sidecar')),
            source_path TEXT,
            size_bytes INTEGER NOT NULL DEFAULT 0 CHECK(size_bytes >= 0),
            acquired_at_ms INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (blob_hash, ref_type, ref_id)
        )"""
    )
    # gc_generations matches the split-file source.db DDL: typed reclaim
    # counters keyed by a TEXT generation_id (#1743).
    conn.execute(
        """CREATE TABLE gc_generations (
            generation_id   TEXT PRIMARY KEY,
            started_at_ms   INTEGER NOT NULL,
            completed_at_ms INTEGER,
            reclaimed_count INTEGER NOT NULL DEFAULT 0,
            reclaimed_bytes INTEGER NOT NULL DEFAULT 0
        )"""
    )
    conn.commit()
    return conn


def _age_blob(blob_store: BlobStore, blob_hash: str, *, seconds: float) -> None:
    """Backdate a blob's mtime so it is GC-eligible without sleeping."""
    path = blob_store.blob_path(blob_hash)
    past = time.time() - seconds
    import os

    os.utime(path, (past, past))


def test_db_reference_alone_protects_blob(tmp_path: Path) -> None:
    """A blob with a raw_sessions row survives GC without any lease."""
    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)
    conn = _make_db(db_path)

    blob_hash, _ = blob_store.write_from_bytes(b"durable payload")
    _age_blob(blob_store, blob_hash, seconds=MIN_AGE_S + 5)

    conn.execute(
        "INSERT INTO raw_sessions (raw_id, source_name, source_path, blob_size, acquired_at) "
        "VALUES (?, 'claude', 'x.json', 0, '2024-01-01')",
        (blob_hash,),
    )
    conn.commit()
    conn.close()

    deleted = run_blob_gc(db_path, blob_root, max_batch=10)
    assert deleted == 0
    assert blob_store.exists(blob_hash)


def test_doctor_repair_path_deletes_unreferenced_old_blob(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An old, unreferenced blob is collectable via the ops-doctor repair path."""
    from polylogue.config import Config
    from polylogue.storage.blob_repair import repair_orphaned_blobs_data

    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)
    _make_db(db_path).close()

    blob_hash, _ = blob_store.write_from_bytes(b"truly abandoned payload")
    _age_blob(blob_store, blob_hash, seconds=MIN_AGE_S + 5)

    monkeypatch.setattr("polylogue.paths.blob_store_root", lambda: blob_root)
    config = Config(archive_root=tmp_path, render_root=tmp_path, sources=[], db_path=db_path)

    outcome = repair_orphaned_blobs_data(config, dry_run=False)

    assert not blob_store.exists(blob_hash)
    assert outcome.repaired_count == 1
    assert outcome.success is True


def test_doctor_repair_path_dry_run_never_deletes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """dry_run=True reports the count without touching disk, on the doctor path too."""
    from polylogue.config import Config
    from polylogue.storage.blob_repair import repair_orphaned_blobs_data

    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)
    _make_db(db_path).close()

    blob_hash, _ = blob_store.write_from_bytes(b"dry run candidate")
    _age_blob(blob_store, blob_hash, seconds=MIN_AGE_S + 5)

    monkeypatch.setattr("polylogue.paths.blob_store_root", lambda: blob_root)
    config = Config(archive_root=tmp_path, render_root=tmp_path, sources=[], db_path=db_path)

    outcome = repair_orphaned_blobs_data(config, dry_run=True)

    assert blob_store.exists(blob_hash)
    assert outcome.repaired_count == 1
    assert outcome.success is True
