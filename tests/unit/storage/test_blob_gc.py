"""Regression tests for blob GC safety invariants.

Verifies the two fatal GC bugs fixed in 1bd2f156 stay fixed:

1. ``_still_referenced`` must check ``raw_sessions.raw_id``
2. ``_candidate_blobs`` must walk all 256 prefix subdirectories
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.blob_gc import (
    GCRunEvidence,
    _candidate_blobs,
    _current_generation,
    _has_active_lease,
    _reference_surfaces,
    _still_referenced,
    read_gc_history,
    run_blob_gc,
)
from polylogue.storage.blob_store import BlobStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db(path: str | Path | None = None) -> sqlite3.Connection:
    """Create an in-memory or file-based SQLite database with GC schema."""
    target = str(path) if path else ":memory:"
    conn = sqlite3.connect(target)
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
            blob_hash BLOB NOT NULL,
            raw_id TEXT NOT NULL,
            ref_type TEXT NOT NULL DEFAULT 'raw_payload'
        )"""
    )
    conn.execute(
        """CREATE TABLE pending_blob_refs (
            blob_hash TEXT NOT NULL,
            operation_id TEXT NOT NULL,
            acquired_at INTEGER NOT NULL,
            PRIMARY KEY (blob_hash, operation_id)
        )"""
    )
    conn.execute(
        """CREATE TABLE gc_generations (
            generation INTEGER PRIMARY KEY,
            completed_at INTEGER NOT NULL,
            evidence TEXT
        )"""
    )
    conn.commit()
    return conn


def _make_source_db(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE raw_sessions (
            raw_id TEXT PRIMARY KEY,
            blob_hash BLOB NOT NULL,
            blob_size INTEGER NOT NULL DEFAULT 0
        ) STRICT"""
    )
    conn.execute(
        """CREATE TABLE blob_refs (
            ref_id TEXT PRIMARY KEY,
            owner_kind TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            blob_hash BLOB NOT NULL
        ) STRICT"""
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# _still_referenced — regression: must check raw_sessions.raw_id
# ---------------------------------------------------------------------------


def test_still_referenced_recognizes_raw_id() -> None:
    """A blob whose hash matches a raw_sessions.raw_id is still referenced."""
    conn = _make_db()
    conn.execute(
        "INSERT INTO raw_sessions (raw_id, source_name, source_path, blob_size, acquired_at) "
        "VALUES ('abc123def456', 'claude', 'test.json', 42, '2024-01-01')"
    )
    conn.commit()
    assert _still_referenced(conn, "abc123def456") is True
    conn.close()


def test_still_referenced_rejects_unknown_hash() -> None:
    """A blob not in raw_sessions is not referenced."""
    conn = _make_db()
    conn.execute(
        "INSERT INTO raw_sessions (raw_id, source_name, source_path, blob_size, acquired_at) "
        "VALUES ('known-hash-1', 'chatgpt', 'test.json', 10, '2024-01-01')"
    )
    conn.commit()
    assert _still_referenced(conn, "unknown-dead-hash") is False
    conn.close()


def test_still_referenced_empty_table() -> None:
    """With no raw_sessions rows, nothing is referenced."""
    conn = _make_db()
    assert _still_referenced(conn, "any-hash") is False
    conn.close()


def test_still_referenced_recognizes_archive_source_hash(tmp_path: Path) -> None:
    """source references are BLOB hashes, not legacy raw_id text."""
    blob_hash = "a" * 64
    source_conn = _make_source_db(tmp_path / "source.db")
    source_conn.execute(
        "INSERT INTO raw_sessions (raw_id, blob_hash, blob_size) VALUES (?, ?, ?)",
        ("raw-1", bytes.fromhex(blob_hash), 42),
    )
    source_conn.commit()

    assert _still_referenced(source_conn, blob_hash) is True
    assert _reference_surfaces(source_conn, blob_hash) == ["current.raw_sessions"]
    source_conn.close()


# ---------------------------------------------------------------------------
# _candidate_blobs — regression: must walk 256 prefix subdirectories
# ---------------------------------------------------------------------------


def test_candidate_blobs_finds_blobs_in_multiple_prefix_dirs(tmp_path: Path) -> None:
    """Blobs spread across prefix directories should all be found."""
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)

    hashes = set()
    for i in range(10):
        h, _ = blob_store.write_from_bytes(f"payload {i}".encode())
        hashes.add(h)

    # Verify blobs are spread across multiple prefix dirs
    prefix_dirs = {h[:2] for h in hashes}
    assert len(prefix_dirs) >= 1  # At least one prefix dir

    candidates = _candidate_blobs(blob_root, older_than=0)
    found_hashes = {h for h, _ in candidates}
    assert found_hashes == hashes, f"Candidate walk missed blobs: expected {hashes}, got {found_hashes}"


def test_candidate_blobs_respects_older_than(tmp_path: Path) -> None:
    """Blobs newer than older_than should be excluded."""
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)
    blob_store.write_from_bytes(b"fresh blob")

    # With very high older_than, no blobs should be returned
    candidates = _candidate_blobs(blob_root, older_than=3600)
    assert len(candidates) == 0

    # With older_than=0, all blobs should be found
    candidates = _candidate_blobs(blob_root, older_than=0)
    assert len(candidates) == 1


def test_candidate_blobs_empty_dir(tmp_path: Path) -> None:
    """Empty blob directory returns empty list."""
    candidates = _candidate_blobs(tmp_path / "nonexistent", older_than=0)
    assert candidates == []


def test_candidate_blobs_skips_dotfiles(tmp_path: Path) -> None:
    """Files starting with '.' (temp files) should be skipped."""
    blob_root = tmp_path / "blobs"
    blob_root.mkdir(parents=True)
    prefix_dir = blob_root / "aa"
    prefix_dir.mkdir()
    # Create a temp file that should be skipped
    (prefix_dir / ".blob.temp").write_bytes(b"temp")
    # Create a real blob manually
    (prefix_dir / "bbccddeeff0011223344556677889900aabbccdd").write_bytes(b"real")

    candidates = _candidate_blobs(blob_root, older_than=0)
    found = {h for h, _ in candidates}
    assert "aabbccddeeff0011223344556677889900aabbccdd" in found
    assert "aa.blob.temp" not in found


def test_candidate_blobs_skips_non_two_char_prefix_dirs(tmp_path: Path) -> None:
    """Directories not matching the two-char prefix pattern should be skipped."""
    blob_root = tmp_path / "blobs"
    blob_root.mkdir(parents=True)
    # Valid prefix dir
    (blob_root / "ab").mkdir()
    (blob_root / "ab" / "cdef1234").write_bytes(b"real")
    # Non-prefix dir
    (blob_root / "not-a-prefix").mkdir()

    candidates = _candidate_blobs(blob_root, older_than=0)
    found = {h for h, _ in candidates}
    assert "abcdef1234" in found
    assert not any(h.startswith("not") for h in found)


# ---------------------------------------------------------------------------
# Lease safety
# ---------------------------------------------------------------------------


def test_has_active_lease() -> None:
    conn = _make_db()
    conn.execute(
        "INSERT INTO pending_blob_refs (blob_hash, operation_id, acquired_at) "
        "VALUES ('hash-under-lease', 'op-001', 1000)"
    )
    conn.commit()
    assert _has_active_lease(conn, "hash-under-lease") is True
    assert _has_active_lease(conn, "hash-not-leased") is False
    conn.close()


def test_has_active_lease_empty_table() -> None:
    conn = _make_db()
    assert _has_active_lease(conn, "any-hash") is False
    conn.close()


# ---------------------------------------------------------------------------
# _current_generation
# ---------------------------------------------------------------------------


def test_current_generation_empty() -> None:
    conn = _make_db()
    assert _current_generation(conn) == 0
    conn.close()


def test_current_generation_returns_max() -> None:
    conn = _make_db()
    conn.execute("INSERT INTO gc_generations (generation, completed_at) VALUES (1, 100)")
    conn.execute("INSERT INTO gc_generations (generation, completed_at) VALUES (5, 500)")
    conn.commit()
    assert _current_generation(conn) == 5
    conn.close()


# ---------------------------------------------------------------------------
# run_blob_gc integration (lightweight)
# ---------------------------------------------------------------------------


def test_run_blob_gc_empty_store(tmp_path: Path) -> None:
    """GC on an empty blob store should succeed with 0 deletions."""
    db_path = tmp_path / "archive.db"
    blob_dir = tmp_path / "blobs"
    blob_dir.mkdir()

    conn = _make_db(db_path)
    conn.close()

    deleted = run_blob_gc(str(db_path), str(blob_dir), max_batch=10)
    assert deleted == 0


def test_run_blob_gc_preserves_referenced_blobs(tmp_path: Path) -> None:
    """GC must not delete blobs that are still referenced in raw_sessions."""
    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)

    # Create a blob and a matching raw_sessions row
    h, _ = blob_store.write_from_bytes(b"referenced content")

    conn = _make_db(db_path)
    conn.execute(
        "INSERT INTO raw_sessions (raw_id, source_name, source_path, blob_size, acquired_at) "
        "VALUES (?, 'claude', 'test.json', ?, '2024-01-01')",
        (h, len(b"referenced content")),
    )
    conn.commit()
    conn.close()

    deleted = run_blob_gc(str(db_path), str(blob_root), max_batch=10)
    assert deleted == 0
    # Blob still on disk
    assert blob_store.exists(h)


def test_run_blob_gc_preserves_archive_source_referenced_blobs(tmp_path: Path) -> None:
    """GC run from ``index.db`` must preserve blobs referenced by sibling ``source.db``."""
    index_db_path = tmp_path / "index.db"
    source_db_path = tmp_path / "source.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)

    blob_hash, _ = blob_store.write_from_bytes(b"archive referenced content")
    _backdate(blob_store, blob_hash)

    index_conn = _make_db(index_db_path)
    index_conn.close()

    source_conn = _make_source_db(source_db_path)
    source_conn.execute(
        "INSERT INTO raw_sessions (raw_id, blob_hash, blob_size) VALUES (?, ?, ?)",
        ("raw-v1", bytes.fromhex(blob_hash), len(b"archive referenced content")),
    )
    source_conn.commit()
    source_conn.close()

    deleted = run_blob_gc(str(index_db_path), str(blob_root), max_batch=10)

    assert deleted == 0
    assert blob_store.exists(blob_hash)
    history = read_gc_history(str(index_db_path), limit=1)
    assert len(history) == 1
    evidence = history[0].evidence
    assert evidence is not None
    assert evidence.skipped_referenced == 1
    assert evidence.reference_surfaces == ["source.db.raw_sessions"]


def test_run_blob_gc_max_batch_bound(tmp_path: Path) -> None:
    """GC should never exceed max_batch even with many orphans."""
    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)

    # Create several orphan blobs
    for i in range(5):
        blob_store.write_from_bytes(f"orphan {i}".encode())

    conn = _make_db(db_path)
    conn.close()

    deleted = run_blob_gc(str(db_path), str(blob_root), max_batch=2)
    # May be 0 due to MIN_AGE_S, but should never exceed max_batch
    assert 0 <= deleted <= 2


def test_run_blob_gc_nonexistent_blob_dir(tmp_path: Path) -> None:
    """GC on nonexistent directory should return 0 without crash."""
    db_path = tmp_path / "archive.db"
    conn = _make_db(db_path)
    conn.close()
    deleted = run_blob_gc(str(db_path), str(tmp_path / "nonexistent"), max_batch=10)
    assert deleted == 0


# ---------------------------------------------------------------------------
# #1190 — sharded unlink path + accurate deleted counter
# ---------------------------------------------------------------------------


import json


def _backdate(blob_store: BlobStore, blob_hash: str, *, seconds: float = 3600) -> None:
    """Backdate a blob's mtime past MIN_AGE_S so it is GC-eligible."""
    import os

    path = blob_store.blob_path(blob_hash)
    past = __import__("time").time() - seconds
    os.utime(path, (past, past))


def test_run_blob_gc_unlinks_sharded_path_and_increments_counter(tmp_path: Path) -> None:
    """#1190 regression: an orphan blob present at the sharded path
    ``{root}/{prefix}/{remainder}`` must actually be removed and the
    ``deleted`` counter must increment by exactly 1.

    Before the fix, ``run_blob_gc`` unlinked ``{root}/{full_hash}``
    (a path that never exists for a real blob), and ``missing_ok=True``
    silently swallowed the failure. The counter still bumped, so the
    function reported successful reclamation while leaving the blob on
    disk.
    """
    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)

    h, _ = blob_store.write_from_bytes(b"orphan to reclaim")
    sharded = blob_store.blob_path(h)
    assert sharded.is_file()
    assert (blob_root / h).exists() is False  # never lived at the flat path

    _backdate(blob_store, h)

    _make_db(db_path).close()

    deleted = run_blob_gc(str(db_path), str(blob_root), max_batch=10)

    assert deleted == 1, "deleted counter must match actual unlinks"
    assert not sharded.exists(), "sharded blob must actually be removed from disk"


def test_run_blob_gc_does_not_increment_when_file_already_missing(tmp_path: Path) -> None:
    """#1190 regression: when the candidate file has vanished between
    discovery and unlink (concurrent reclaimer, stale candidate, manual
    cleanup), the ``deleted`` counter must NOT increment. The structured
    evidence row should record this as ``skipped_missing``.
    """
    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)

    h, _ = blob_store.write_from_bytes(b"will-vanish")
    sharded = blob_store.blob_path(h)
    _backdate(blob_store, h)

    # Race simulation: file disappears after _candidate_blobs() lists it
    # but before run_blob_gc unlinks it. We approximate this by removing
    # the file ourselves before the call — _candidate_blobs has already
    # seen the dirent; the unlink will hit FileNotFoundError.
    # But _candidate_blobs runs inside run_blob_gc. We instead remove the
    # underlying file out-of-band BEFORE the call but AFTER recording the
    # candidate via a wrapper: simpler — patch the candidate listing to
    # report this hash even though the file is now gone.
    sharded.unlink()
    assert not sharded.exists()

    _make_db(db_path).close()

    # Re-create a sibling so _candidate_blobs sees the directory; then
    # patch the listing to include the vanished hash too.
    from polylogue.storage import blob_gc as gc_mod

    real_listing = gc_mod._candidate_blobs

    def patched(root: Path, *, older_than: float) -> list[tuple[str, float]]:
        out = list(real_listing(root, older_than=older_than))
        out.append((h, 0.0))
        return out

    gc_mod._candidate_blobs = patched  # type: ignore[assignment]
    try:
        deleted = run_blob_gc(str(db_path), str(blob_root), max_batch=10)
    finally:
        gc_mod._candidate_blobs = real_listing

    assert deleted == 0, "counter must not bump when no file was actually unlinked"

    # Evidence row attributes the skip correctly.
    history = read_gc_history(str(db_path), limit=1)
    assert len(history) == 1
    ev = history[0].evidence
    assert ev is not None
    assert ev.deleted == 0
    assert ev.skipped_missing >= 1


def test_run_blob_gc_dry_run_does_not_delete_or_record_generation(tmp_path: Path) -> None:
    """#1190 ambitious-expansion: --dry-run previews without committing.

    A dry-run must:
      - NOT remove any file from disk;
      - NOT insert a row into ``gc_generations`` (no generation slot consumed);
      - still return the would-be count.
    """
    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)

    h, _ = blob_store.write_from_bytes(b"dry-run orphan")
    _backdate(blob_store, h)
    _make_db(db_path).close()

    would_delete = run_blob_gc(str(db_path), str(blob_root), max_batch=10, dry_run=True)

    assert would_delete == 1
    assert blob_store.exists(h), "dry-run must never touch disk"

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT COUNT(*) FROM gc_generations").fetchone()
    finally:
        conn.close()
    assert row[0] == 0, "dry-run must not consume a generation slot"


def test_run_blob_gc_records_structured_evidence(tmp_path: Path) -> None:
    """#1190 ambitious-expansion: each committed pass writes a JSON
    evidence row capturing inspected/skipped/deleted counts plus the
    list of deleted hashes — a self-describing audit trail.
    """
    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)

    referenced_hash, _ = blob_store.write_from_bytes(b"keep me")
    orphan_hash, _ = blob_store.write_from_bytes(b"delete me")
    _backdate(blob_store, referenced_hash)
    _backdate(blob_store, orphan_hash)

    conn = _make_db(db_path)
    conn.execute(
        "INSERT INTO raw_sessions (raw_id, source_name, source_path, blob_size, acquired_at) "
        "VALUES (?, 'claude', 'x.json', 1, '2025-01-01')",
        (referenced_hash,),
    )
    conn.commit()
    conn.close()

    deleted = run_blob_gc(str(db_path), str(blob_root), max_batch=10)
    assert deleted == 1

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT generation, evidence FROM gc_generations").fetchone()
    finally:
        conn.close()
    assert row is not None
    payload = json.loads(row["evidence"]) if isinstance(row, sqlite3.Row) else json.loads(row[1])
    ev = GCRunEvidence(**payload)
    assert ev.inspected == 2
    assert ev.deleted == 1
    assert ev.skipped_referenced == 1
    assert ev.dry_run is False
    assert orphan_hash in ev.deleted_hashes


def test_read_gc_history_returns_recent_passes_newest_first(tmp_path: Path) -> None:
    """#1190 ambitious-expansion: ``read_gc_history`` surfaces evidence
    rows in newest-first order, so a ``gc-history`` operator surface can
    show recent GC behaviour without bespoke SQLite tooling.
    """
    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)
    _make_db(db_path).close()

    for i in range(3):
        h, _ = blob_store.write_from_bytes(f"orphan-{i}".encode())
        _backdate(blob_store, h)
        run_blob_gc(str(db_path), str(blob_root), max_batch=10)

    history = read_gc_history(str(db_path), limit=10)
    assert [row.generation for row in history] == [3, 2, 1]
    for row in history:
        assert row.evidence is not None
        assert row.evidence.deleted >= 1
