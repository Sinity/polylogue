"""Daemon-side wiring for the periodic blob-GC drain (automagic-invariants gap).

``polylogue ops maintenance blob-gc`` was, before this module existed, the
*only* path that reclaimed unreferenced blobs — every other mechanical,
non-judgment maintenance operation (FTS merge, WAL checkpoint, embedding
orphan reconcile) already has a daemon-owned periodic equivalent. These
tests exercise ``run_blob_gc_once`` — the bounded sync helper the periodic
daemon loop (``periodic_blob_gc_check``) invokes through the write
coordinator — against real on-disk blob-store fixtures, and prove the
production route (``DaemonWriteCoordinator.run_sync``) actually performs
the reclaim rather than a bypassed direct call.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path

from polylogue.daemon.blob_gc_periodic import run_blob_gc_once
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
from polylogue.storage.blob_store import BlobStore


def _make_source_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                blob_hash BLOB NOT NULL,
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
    finally:
        conn.close()


def _backdate(blob_store: BlobStore, blob_hash: str) -> None:
    """Backdate a blob's mtime well past MIN_AGE_S (60s) so it is GC-eligible.

    A fixed epoch (not a `time.time()`-relative offset) avoids any host-clock
    read in the test itself -- GC eligibility only needs "old enough", not a
    real elapsed duration.
    """
    os.utime(blob_store.blob_path(blob_hash), (0, 0))


def test_run_blob_gc_once_returns_none_when_blob_dir_absent(tmp_path: Path) -> None:
    db_path = tmp_path / "source.db"
    _make_source_db(db_path)
    result = run_blob_gc_once(db_path, tmp_path / "blob")
    assert result is None


def test_run_blob_gc_once_returns_none_when_source_db_absent(tmp_path: Path) -> None:
    blob_dir = tmp_path / "blob"
    blob_dir.mkdir()
    result = run_blob_gc_once(tmp_path / "source.db", blob_dir)
    assert result is None


def test_run_blob_gc_once_reclaims_unreferenced_aged_blob(tmp_path: Path) -> None:
    db_path = tmp_path / "source.db"
    _make_source_db(db_path)
    blob_dir = tmp_path / "blob"
    blob_store = BlobStore(blob_dir)

    blob_hash, _ = blob_store.write_from_bytes(b"orphan blob for daemon gc")
    _backdate(blob_store, blob_hash)

    result = run_blob_gc_once(db_path, blob_dir)

    assert result is not None
    assert result.deleted_count == 1
    assert not blob_store.blob_path(blob_hash).exists()


def test_daemon_coordinator_owns_real_blob_gc_mutation(tmp_path: Path) -> None:
    """Production-route proof; bypassing run_sync or its authority token makes this fail."""
    db_path = tmp_path / "source.db"
    _make_source_db(db_path)
    blob_dir = tmp_path / "blob"
    blob_store = BlobStore(blob_dir)

    blob_hash, _ = blob_store.write_from_bytes(b"orphan blob via coordinator")
    _backdate(blob_store, blob_hash)

    coordinator = DaemonWriteCoordinator()

    async def run() -> object:
        return await coordinator.run_sync("maintenance.blob_gc", run_blob_gc_once, db_path, blob_dir)

    result = asyncio.run(run())

    assert result is not None
    assert result.deleted_count == 1  # type: ignore[attr-defined]
    assert coordinator.snapshot().active_actor is None
    assert not blob_store.blob_path(blob_hash).exists()
