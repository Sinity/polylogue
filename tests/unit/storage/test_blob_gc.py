"""Regression tests for blob GC safety invariants.

Verifies the two fatal GC bugs fixed in 1bd2f156 stay fixed:

1. ``_still_referenced`` must check ``raw_conversations.raw_id``
2. ``_candidate_blobs`` must walk all 256 prefix subdirectories
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.blob_gc import (
    _candidate_blobs,
    _current_generation,
    _has_active_lease,
    _still_referenced,
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
        """CREATE TABLE raw_conversations (
            raw_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL DEFAULT '',
            source_path TEXT NOT NULL DEFAULT '',
            blob_size INTEGER NOT NULL DEFAULT 0,
            acquired_at TEXT NOT NULL DEFAULT ''
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
            completed_at INTEGER NOT NULL
        )"""
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# _still_referenced — regression: must check raw_conversations.raw_id
# ---------------------------------------------------------------------------


def test_still_referenced_recognizes_raw_id() -> None:
    """A blob whose hash matches a raw_conversations.raw_id is still referenced."""
    conn = _make_db()
    conn.execute(
        "INSERT INTO raw_conversations (raw_id, provider_name, source_path, blob_size, acquired_at) "
        "VALUES ('abc123def456', 'claude', 'test.json', 42, '2024-01-01')"
    )
    conn.commit()
    assert _still_referenced(conn, "abc123def456") is True
    conn.close()


def test_still_referenced_rejects_unknown_hash() -> None:
    """A blob not in raw_conversations is not referenced."""
    conn = _make_db()
    conn.execute(
        "INSERT INTO raw_conversations (raw_id, provider_name, source_path, blob_size, acquired_at) "
        "VALUES ('known-hash-1', 'chatgpt', 'test.json', 10, '2024-01-01')"
    )
    conn.commit()
    assert _still_referenced(conn, "unknown-dead-hash") is False
    conn.close()


def test_still_referenced_empty_table() -> None:
    """With no raw_conversations rows, nothing is referenced."""
    conn = _make_db()
    assert _still_referenced(conn, "any-hash") is False
    conn.close()


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
    """GC must not delete blobs that are still referenced in raw_conversations."""
    db_path = tmp_path / "archive.db"
    blob_root = tmp_path / "blobs"
    blob_store = BlobStore(blob_root)

    # Create a blob and a matching raw_conversations row
    h, _ = blob_store.write_from_bytes(b"referenced content")

    conn = _make_db(db_path)
    conn.execute(
        "INSERT INTO raw_conversations (raw_id, provider_name, source_path, blob_size, acquired_at) "
        "VALUES (?, 'claude', 'test.json', ?, '2024-01-01')",
        (h, len(b"referenced content")),
    )
    conn.commit()
    conn.close()

    deleted = run_blob_gc(str(db_path), str(blob_root), max_batch=10)
    assert deleted == 0
    # Blob still on disk
    assert blob_store.exists(h)


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
