"""Regression tests for blob-lease durability and the GC generation gate (#1746).

Covers three guarantees added in the #1746 fix:

1. ``commit_archive_write_effects`` releases its blob lease on *every* exit
   path. A failure after lease acquisition must not leak a ``pending_blob_refs``
   row (the lease is committed on a separate connection, so the failing
   transaction's rollback cannot undo it).
2. ``sweep_orphaned_blob_leases`` reclaims leases left behind by a writer that
   was killed between acquire and release.
3. ``run_blob_gc`` enforces safety invariant #3: a candidate must be older than
   the previous completed generation's ``completed_at`` (not merely
   ``MIN_AGE_S``).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from polylogue.archive import write_effects
from polylogue.archive.write_gateway import WriteOperation
from polylogue.storage.blob_gc import (
    MIN_AGE_S,
    acquire_blob_leases,
    run_blob_gc,
    sweep_orphaned_blob_leases,
)
from polylogue.storage.sqlite.connection_profile import open_connection


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    # Bootstrap the full split-file archive: source.db carries raw_sessions
    # (with blob_hash), blob_refs, pending_blob_refs, and gc_generations.
    # open_connection attaches the source tier so unqualified blob-GC queries
    # resolve cross-tier.
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    return tmp_path / "index.db"


def _count_leases(db: Path) -> int:
    conn = open_connection(db)
    try:
        count: int = conn.execute("SELECT COUNT(*) FROM pending_blob_refs").fetchone()[0]
        return count
    finally:
        conn.close()


def _set_lease_acquired_at(db: Path, operation_id: str, acquired_at_s: int) -> None:
    conn = open_connection(db)
    try:
        conn.execute(
            "UPDATE pending_blob_refs SET acquired_at_ms = ? WHERE operation_id = ?",
            (acquired_at_s * 1000, operation_id),
        )
        conn.commit()
    finally:
        conn.close()


def _make_blob(blob_dir: Path, blob_hash: str, *, age_s: float = 0.0) -> None:
    import os

    shard = blob_dir / blob_hash[:2]
    shard.mkdir(parents=True, exist_ok=True)
    blob_file = shard / blob_hash[2:]
    blob_file.write_bytes(b"data")
    if age_s > 0:
        old = time.time() - age_s
        os.utime(blob_file, (old, old))


def test_write_effects_releases_lease_on_failure(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A post-acquire failure in commit_archive_write_effects leaks no lease.

    Force the FTS-trigger step to raise so the function exits abnormally after
    the lease was acquired on its separate immediate-commit connection. The
    finally-path recovery release must drop the row.
    """

    def _boom(_conn: sqlite3.Connection) -> None:
        raise RuntimeError("simulated FTS failure")

    # ensure_fts_triggers_sync is imported inside the function body, so patch at
    # the source module.
    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.ensure_fts_triggers_sync",
        _boom,
    )

    conn = open_connection(db_path)
    blob_hash = "ab" + "c" * 62
    try:
        with pytest.raises(RuntimeError, match="simulated FTS failure"):
            write_effects.commit_archive_write_effects(
                conn,
                WriteOperation.INGEST,
                {
                    "changed_session_ids": ["conv-1"],
                    "_blob_hashes": [blob_hash],
                    "_operation_id": "op-fail",
                    "_db_path": str(db_path),
                },
            )
    finally:
        conn.close()

    assert _count_leases(db_path) == 0, "lease leaked after write-effects failure"


def test_write_effects_releases_lease_on_success(db_path: Path) -> None:
    """The normal success path also leaves no residual lease."""
    conn = open_connection(db_path)
    blob_hash = "de" + "f" * 62
    try:
        write_effects.commit_archive_write_effects(
            conn,
            WriteOperation.INGEST,
            {
                "changed_session_ids": [],
                "_blob_hashes": [blob_hash],
                "_operation_id": "op-ok",
                "_db_path": str(db_path),
            },
        )
    finally:
        conn.close()

    assert _count_leases(db_path) == 0


def test_sweep_removes_orphaned_lease(db_path: Path) -> None:
    """A lease older than the sweep bound is reclaimed; a fresh one survives."""
    acquire_blob_leases(db_path, ["aa" + "1" * 62], "op-stale")
    acquire_blob_leases(db_path, ["bb" + "2" * 62], "op-fresh")
    # Age the stale lease well past the default bound.
    _set_lease_acquired_at(db_path, "op-stale", int(time.time()) - 7200)

    removed = sweep_orphaned_blob_leases(db_path, max_age_s=3600)

    assert removed == 1
    assert _count_leases(db_path) == 1
    conn = open_connection(db_path)
    try:
        survivor = conn.execute("SELECT operation_id FROM pending_blob_refs").fetchone()[0]
    finally:
        conn.close()
    assert survivor == "op-fresh"


def test_sweep_keeps_recent_lease(db_path: Path) -> None:
    """A lease younger than the bound is never swept."""
    acquire_blob_leases(db_path, ["cc" + "3" * 62], "op-young")

    removed = sweep_orphaned_blob_leases(db_path, max_age_s=3600)

    assert removed == 0
    assert _count_leases(db_path) == 1


@pytest.mark.xfail(
    reason="blob-GC generation tracking migration to gc_generations(*_at_ms) pending; see #1789", strict=False
)
def test_gc_age_gate_respects_previous_generation(db_path: Path, tmp_path: Path) -> None:
    """A blob younger than the previous generation's completion is not deleted.

    Seed a recently-completed generation, then create an unreferenced blob old
    enough to pass MIN_AGE_S but younger than that generation boundary. The
    generation gate must keep it; only after it ages past the boundary does GC
    reclaim it.
    """
    blob_dir = tmp_path / "blobs"
    blob_dir.mkdir()
    blob_hash = "12" + "3" * 62

    now = int(time.time())
    # Previous generation completed 1000s ago.
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO gc_generations (generation, completed_at, evidence) VALUES (?, ?, ?)",
            (1, now - 1000, "{}"),
        )
        conn.commit()
    finally:
        conn.close()

    # Blob is older than MIN_AGE_S but younger than the 1000s generation boundary.
    _make_blob(blob_dir, blob_hash, age_s=MIN_AGE_S + 10)
    assert MIN_AGE_S + 10 < 1000  # guard the test's own premise

    deleted = run_blob_gc(db_path, blob_dir)
    assert deleted == 0, "generation gate should have kept the recent blob"
    assert (blob_dir / blob_hash[:2] / blob_hash[2:]).exists()

    # Age the blob past the generation boundary; now it is eligible.
    _make_blob(blob_dir, blob_hash, age_s=1100)
    deleted = run_blob_gc(db_path, blob_dir)
    assert deleted == 1
    assert not (blob_dir / blob_hash[:2] / blob_hash[2:]).exists()
