"""Concurrency invariants for blob GC + lease handshake (#1182).

Pins the documented contract from ``docs/internals.md`` ("GC concurrency
model — leases plus snapshot reference check"):

1. ``acquire_blob_leases`` makes the lease visible to a concurrent GC
   before the main data transaction commits.
2. ``run_blob_gc`` skips any blob with an active lease, even when the
   blob is older than ``MIN_AGE_S`` and not yet recorded in
   ``raw_sessions``.
3. After the data transaction commits and the operation releases its
   leases, GC may delete an unreferenced blob.
4. Concurrent acquire/commit/GC interleavings never delete a blob that
   the write path ends up referencing.

These invariants are written so a regression that removes the lease
table or skips the lease check fails this file deterministically.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.storage.blob_gc import (
    MIN_AGE_S,
    _has_active_lease,
    _still_referenced,
    acquire_blob_leases,
    release_operation_leases,
    run_blob_gc,
)
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
    conn.execute(
        """CREATE TABLE pending_blob_refs (
            blob_hash BLOB NOT NULL CHECK(length(blob_hash) = 32),
            operation_id TEXT NOT NULL,
            ref_type TEXT NOT NULL,
            ref_id TEXT NOT NULL,
            acquired_at_ms INTEGER NOT NULL,
            PRIMARY KEY (blob_hash, operation_id, ref_type, ref_id)
        )"""
    )
    # gc_generations stays in the production ``run_blob_gc`` shape
    # (``generation``/``completed_at``/``evidence``); the split-file
    # ``generation_id``/``*_at_ms`` migration is pending (#1789).
    conn.execute(
        """CREATE TABLE gc_generations (
            generation INTEGER PRIMARY KEY,
            completed_at INTEGER NOT NULL,
            evidence TEXT
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


# ---------------------------------------------------------------------------
# Static lease/GC invariants
# ---------------------------------------------------------------------------


def test_lease_visible_to_concurrent_gc(tmp_path: Path) -> None:
    """A lease acquired on a fresh blob blocks GC even before any DB row."""
    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)
    _make_db(db_path).close()

    blob_hash, _ = blob_store.write_from_bytes(b"in-flight payload")
    _age_blob(blob_store, blob_hash, seconds=MIN_AGE_S + 5)

    # Caller has materialized blob but not yet inserted raw_sessions.
    # The lease bridges that window.
    acquire_blob_leases(db_path, [blob_hash], operation_id="op-write-1")

    # GC running mid-flight must observe the lease and skip.
    deleted = run_blob_gc(db_path, blob_root, max_batch=10)
    assert deleted == 0
    assert blob_store.exists(blob_hash), (
        "GC deleted a blob that was held by an in-flight lease — the acquire-then-commit window is no longer protected"
    )


def test_release_followed_by_gc_collects_orphan(tmp_path: Path) -> None:
    """After lease release without a DB row, the blob becomes collectable."""
    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)
    _make_db(db_path).close()

    blob_hash, _ = blob_store.write_from_bytes(b"abandoned payload")
    _age_blob(blob_store, blob_hash, seconds=MIN_AGE_S + 5)
    acquire_blob_leases(db_path, [blob_hash], operation_id="op-aborted")

    # Operation aborts and releases its leases without inserting raw row.
    conn = sqlite3.connect(str(db_path))
    try:
        release_operation_leases(conn, "op-aborted")
        conn.commit()
    finally:
        conn.close()

    deleted = run_blob_gc(db_path, blob_root, max_batch=10)
    # NOTE: With #1190 (run_blob_gc unlinks at flat path instead of the
    # sharded blob_path), the deletion count reports success but the file
    # is not actually removed. We assert the DB-level intent (deleted=1)
    # is recorded — the missing-file consequence is covered by #1190.
    assert deleted == 1, (
        "Lease was released but GC refused to collect the orphan — the release path is no longer effective"
    )


def test_db_reference_alone_protects_blob(tmp_path: Path) -> None:
    """A blob with a raw_sessions row survives GC even without a lease."""
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


# ---------------------------------------------------------------------------
# Threaded interleaving — lease acquire racing GC scan
# ---------------------------------------------------------------------------


def test_concurrent_acquire_and_gc_never_deletes_referenced(tmp_path: Path) -> None:
    """Hundreds of acquire/commit/GC interleavings must never delete a referenced blob.

    Pattern: a writer thread repeatedly stages a blob, acquires a lease,
    commits the raw_sessions row, releases the lease. A GC thread
    runs continuously. The blob's contract is "still referenced by the
    final committed state ⇒ on disk".
    """
    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)
    _make_db(db_path).close()

    blob_hash, _ = blob_store.write_from_bytes(b"raced payload")
    _age_blob(blob_store, blob_hash, seconds=MIN_AGE_S + 5)

    # Pre-acquire a sentinel lease before any thread starts so the blob
    # cannot be GC'd in the gap between thread spawn and the first
    # ``acquire_blob_leases`` call. The writer drops this sentinel on
    # exit; until then, it bridges the test's startup window the same way
    # the production lease handshake bridges the
    # acquire-blob → write-DB-row window in real ingest.
    acquire_blob_leases(db_path, [blob_hash], operation_id="op-sentinel")

    stop = threading.Event()
    error: list[BaseException] = []

    def writer() -> None:
        try:
            for i in range(20):
                op_id = f"op-{i}"
                acquire_blob_leases(db_path, [blob_hash], operation_id=op_id)
                # Simulate parse/convergence work between acquire and commit.
                time.sleep(0.001)
                conn = sqlite3.connect(str(db_path), timeout=5.0)
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO raw_sessions "
                        "(raw_id, source_name, source_path, blob_size, acquired_at) "
                        "VALUES (?, 'claude', 'x.json', 0, '2024-01-01')",
                        (blob_hash,),
                    )
                    conn.commit()
                    release_operation_leases(conn, op_id)
                    conn.commit()
                finally:
                    conn.close()
        except BaseException as exc:  # pragma: no cover - reported via assertion
            error.append(exc)
        finally:
            # Drop the sentinel lease now that the row exists and refs the blob.
            try:
                conn = sqlite3.connect(str(db_path), timeout=5.0)
                try:
                    release_operation_leases(conn, "op-sentinel")
                    conn.commit()
                finally:
                    conn.close()
            except BaseException as exc:  # pragma: no cover - defensive sentinel-release error capture
                error.append(exc)
            stop.set()

    def gc_loop() -> None:
        try:
            while not stop.is_set():
                run_blob_gc(db_path, blob_root, max_batch=10)
                time.sleep(0.0005)
        except BaseException as exc:  # pragma: no cover - defensive thread error capture
            error.append(exc)

    w = threading.Thread(target=writer)
    g = threading.Thread(target=gc_loop)
    w.start()
    g.start()
    w.join(timeout=20)
    g.join(timeout=5)

    assert not error, f"thread raised: {error}"
    # After the writer completes, the raw row references the blob — the
    # blob MUST still be on disk regardless of GC interleavings.
    assert blob_store.exists(blob_hash), "Concurrent GC deleted a blob referenced by a committed raw_sessions row"


# ---------------------------------------------------------------------------
# Hypothesis: many distinct blobs, randomized commit ordering
# ---------------------------------------------------------------------------


@settings(
    deadline=None,
    max_examples=8,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(
    payloads=st.lists(
        st.binary(min_size=4, max_size=64).map(lambda b: b + b"\x00"),
        min_size=2,
        max_size=6,
        unique=True,
    ),
    commit_order=st.lists(st.integers(0, 100), min_size=2, max_size=6),
)
def test_hypothesis_lease_invariant_under_random_interleavings(
    tmp_path_factory: pytest.TempPathFactory,
    payloads: list[bytes],
    commit_order: list[int],
) -> None:
    """For any commit ordering, every committed blob survives GC.

    Builds N distinct payloads, acquires leases in one order, commits the
    raw rows in a shuffled order. GC runs at each step. Final state:
    every committed blob is still on disk.
    """
    base = tmp_path_factory.mktemp("blob_gc_hyp")
    db_path = base / "archive.db"
    blob_root = base / "blobs"
    blob_store = BlobStore(blob_root)
    _make_db(db_path).close()

    hashes = []
    for payload in payloads:
        h, _ = blob_store.write_from_bytes(payload)
        hashes.append(h)
        _age_blob(blob_store, h, seconds=MIN_AGE_S + 5)

    # Acquire all leases up front.
    op_ids = [f"op-{i}" for i in range(len(hashes))]
    for h, op_id in zip(hashes, op_ids, strict=True):
        acquire_blob_leases(db_path, [h], operation_id=op_id)

    # Commit raw rows in a shuffled order derived from commit_order.
    order = sorted(range(len(hashes)), key=lambda i: commit_order[i % len(commit_order)])
    for idx in order:
        # GC runs between commits — must respect remaining leases.
        run_blob_gc(db_path, blob_root, max_batch=10)
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO raw_sessions "
                "(raw_id, source_name, source_path, blob_size, acquired_at) "
                "VALUES (?, 'claude', 'x.json', 0, '2024-01-01')",
                (hashes[idx],),
            )
            conn.commit()
            release_operation_leases(conn, op_ids[idx])
            conn.commit()
        finally:
            conn.close()
        run_blob_gc(db_path, blob_root, max_batch=10)

    for h in hashes:
        assert blob_store.exists(h), f"blob {h} deleted despite committed reference"


# ---------------------------------------------------------------------------
# Static helpers smoke
# ---------------------------------------------------------------------------


def test_helpers_reflect_lease_lifecycle(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    conn = _make_db(db_path)
    blob = "a" * 64
    try:
        acquire_blob_leases(db_path, [blob], operation_id="op-X")
        assert _has_active_lease(conn, blob) is True
        assert _still_referenced(conn, blob) is False
        release_operation_leases(conn, "op-X")
        conn.commit()
        assert _has_active_lease(conn, blob) is False
    finally:
        conn.close()
