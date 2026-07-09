"""Regression tests for the GC generation-age gate (#1746, #1830).

Covers the generation-age safety invariant that survives after
polylogue-v7e0 removed the never-reachable blob-GC lease mechanism (a
prior revision of this file also covered ``acquire_blob_leases``/
``release_operation_leases``/``sweep_orphaned_blob_leases`` durability; those
tests were removed with the mechanism itself — no production ingest caller
ever populated the lease payload keys, so they exercised code that could
never run outside a synthetic test payload).

``run_blob_gc`` enforces safety invariant #2: a candidate must be older than
the previous completed generation's ``completed_at`` (not merely
``MIN_AGE_S``) — see ``docs/internals.md`` "GC concurrency model" for the
current, lease-free contract and why ``MIN_AGE_S`` alone is judged
sufficient.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from polylogue.storage.blob_gc import MIN_AGE_S, run_blob_gc
from polylogue.storage.sqlite.connection_profile import open_connection
from tests.infra.frozen_clock import FrozenClock


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    # Bootstrap the full split-file archive: source.db carries raw_sessions
    # (with blob_hash), blob_refs, and gc_generations. open_connection
    # attaches the source tier so unqualified blob-GC queries resolve
    # cross-tier.
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    return tmp_path / "index.db"


def _make_blob(blob_dir: Path, blob_hash: str, *, age_s: float = 0.0) -> None:
    import os

    shard = blob_dir / blob_hash[:2]
    shard.mkdir(parents=True, exist_ok=True)
    blob_file = shard / blob_hash[2:]
    blob_file.write_bytes(b"data")
    if age_s > 0:
        old = time.time() - age_s
        os.utime(blob_file, (old, old))


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
    # Previous generation completed 1000s ago. gc_generations lives in the
    # attached source tier; open_connection resolves the unqualified name.
    completed_at_ms = (now - 1000) * 1000
    conn = open_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO gc_generations "
            "(generation_id, started_at_ms, completed_at_ms, reclaimed_count, reclaimed_bytes) "
            "VALUES (?, ?, ?, 0, 0)",
            ("gen-seed", completed_at_ms, completed_at_ms),
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


def test_gc_combines_reference_and_generation_guards(
    db_path: Path,
    tmp_path: Path,
    workspace_env: dict[str, Path],
    frozen_clock: FrozenClock,
) -> None:
    """One GC pass applies both safety gates before reclaiming.

    A generation-safe GC pass must keep referenced blobs and blobs newer than
    the previous completed generation, while still reclaiming old
    unreferenced blobs in the same run. This is the executable evidence for
    the combined backup/GC safety story rather than independent unit checks
    for each predicate.
    """
    blob_dir = tmp_path / "blobs"
    blob_dir.mkdir()
    referenced_hash = "aa" + "1" * 62
    generation_young_hash = "cc" + "3" * 62
    orphan_hash = "dd" + "4" * 62

    now = int(frozen_clock.time())
    completed_at_ms = (now - 1000) * 1000
    conn = open_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO gc_generations "
            "(generation_id, started_at_ms, completed_at_ms, reclaimed_count, reclaimed_bytes) "
            "VALUES (?, ?, ?, 0, 0)",
            ("gen-boundary", completed_at_ms, completed_at_ms),
        )
        conn.execute(
            "INSERT INTO blob_refs "
            "(blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms) "
            "VALUES (?, 'ref-1', 'raw_payload', 'source.jsonl', 4, ?)",
            (bytes.fromhex(referenced_hash), completed_at_ms),
        )
        conn.commit()
    finally:
        conn.close()

    _make_blob(blob_dir, referenced_hash, age_s=1200)
    assert MIN_AGE_S + 10 < 1000  # guard the generation-young premise
    _make_blob(blob_dir, generation_young_hash, age_s=MIN_AGE_S + 10)
    _make_blob(blob_dir, orphan_hash, age_s=1200)

    deleted = run_blob_gc(db_path, blob_dir, max_batch=10)

    assert deleted == 1
    assert (blob_dir / referenced_hash[:2] / referenced_hash[2:]).exists()
    assert (blob_dir / generation_young_hash[:2] / generation_young_hash[2:]).exists()
    assert not (blob_dir / orphan_hash[:2] / orphan_hash[2:]).exists()

    conn = open_connection(db_path)
    try:
        rows = [
            tuple(row)
            for row in conn.execute(
                "SELECT reclaimed_count, reclaimed_bytes FROM gc_generations ORDER BY started_at_ms DESC LIMIT 1"
            ).fetchall()
        ]
    finally:
        conn.close()

    assert rows == [(1, 4)]
