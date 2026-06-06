"""Superseded live raw snapshot cleanup contracts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.blob_store import BlobStore
from polylogue.storage.raw_retention import (
    cleanup_superseded_raw_snapshots,
    superseded_raw_snapshot_candidates,
)
from polylogue.storage.sqlite.schema import _ensure_schema


def _write_blob(store: BlobStore, payload: bytes) -> tuple[str, int]:
    return store.write_from_bytes(payload)


def _ensure_archive_source_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE raw_sessions (
            raw_id TEXT PRIMARY KEY,
            origin TEXT NOT NULL,
            native_id TEXT,
            source_path TEXT NOT NULL,
            source_index INTEGER NOT NULL DEFAULT 0,
            blob_hash BLOB NOT NULL CHECK(length(blob_hash) = 32),
            blob_size INTEGER NOT NULL CHECK(blob_size >= 0),
            acquired_at_ms INTEGER NOT NULL
        ) STRICT"""
    )
    conn.execute(
        """CREATE TABLE blob_refs (
            blob_hash BLOB NOT NULL CHECK(length(blob_hash) = 32),
            raw_id TEXT NOT NULL REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
            ref_type TEXT NOT NULL CHECK(ref_type IN ('raw_payload', 'attachment', 'sidecar')),
            source_path TEXT,
            size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
            acquired_at_ms INTEGER NOT NULL,
            PRIMARY KEY(blob_hash, raw_id, ref_type)
        ) STRICT"""
    )


def _insert_archive_raw_session(
    conn: sqlite3.Connection,
    *,
    raw_id: str,
    source_path: Path,
    source_index: int,
    blob_hash: str,
    blob_size: int,
    acquired_at_ms: int,
) -> None:
    conn.execute(
        """
        INSERT INTO raw_sessions (
            raw_id, origin, native_id, source_path, source_index,
            blob_hash, blob_size, acquired_at_ms
        ) VALUES (?, 'codex', ?, ?, ?, ?, ?, ?)
        """,
        (raw_id, raw_id, str(source_path), source_index, bytes.fromhex(blob_hash), blob_size, acquired_at_ms),
    )
    conn.execute(
        """
        INSERT INTO blob_refs (
            blob_hash, raw_id, ref_type, source_path, size_bytes, acquired_at_ms
        ) VALUES (?, ?, 'raw_payload', ?, ?, ?)
        """,
        (bytes.fromhex(blob_hash), raw_id, str(source_path), blob_size, acquired_at_ms),
    )


def test_superseded_raw_snapshot_cleanup_keeps_newest_per_source(tmp_path: Path) -> None:
    db_path = tmp_path / "source.db"
    source = tmp_path / "rollout.jsonl"
    missing_source = tmp_path / "missing.jsonl"
    source.write_text('{"type":"message"}\n', encoding="utf-8")
    blob_store = BlobStore(tmp_path / "blob")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_archive_source_schema(conn)

    full_old, full_old_size = _write_blob(blob_store, b"full-old")
    full_new, full_new_size = _write_blob(blob_store, b"full-new")
    append_old, append_old_size = _write_blob(blob_store, b"append-old")
    append_current, append_current_size = _write_blob(blob_store, b"append-current")
    leased_old, leased_old_size = _write_blob(blob_store, b"leased-old")
    missing_old, missing_old_size = _write_blob(blob_store, b"missing-old")
    missing_new, missing_new_size = _write_blob(blob_store, b"missing-new")

    # Archive file-set retention is recency-only: it keeps the newest snapshot per
    # (source_path, source_index) and supersedes the rest. There is no legacy
    # live-reference / lease preservation — sessions point at current raws, and
    # provider_events is a retired single-file table absent from archive — so any
    # superseded snapshot whose source file still exists is a candidate.
    def _seed(raw_id: str, source_path: Path, source_index: int, blob_size: int, acquired_at_ms: int) -> None:
        _insert_archive_raw_session(
            conn,
            raw_id=raw_id,
            source_path=source_path,
            source_index=source_index,
            blob_hash=raw_id,
            blob_size=blob_size,
            acquired_at_ms=acquired_at_ms,
        )

    _seed(full_old, source, 0, full_old_size, 1_000)
    _seed(full_new, source, 0, full_new_size, 2_000)
    _seed(append_old, source, -1, append_old_size, 3_000)
    _seed(append_current, source, -1, append_current_size, 4_000)
    _seed(leased_old, source, -1, leased_old_size, 2_500)
    _seed(missing_old, missing_source, 0, missing_old_size, 1_000)
    _seed(missing_new, missing_source, 0, missing_new_size, 2_000)
    conn.commit()

    # full_old (superseded by full_new) and append_old + leased_old (superseded
    # by append_current). missing_old is superseded too, but its source file is
    # gone, so it is excluded from candidates.
    candidates = superseded_raw_snapshot_candidates(conn, limit=100)
    assert {candidate.raw_id for candidate in candidates} == {full_old, append_old, leased_old}

    dry_run = cleanup_superseded_raw_snapshots(conn, dry_run=True, blob_store=blob_store)
    assert dry_run.candidate_count == 3
    assert blob_store.exists(full_old)
    assert blob_store.exists(append_old)
    assert blob_store.exists(leased_old)

    result = cleanup_superseded_raw_snapshots(conn, dry_run=False, blob_store=blob_store)
    assert result.deleted_raw_count == 3
    assert result.deleted_blob_count == 3
    assert not blob_store.exists(full_old)
    assert not blob_store.exists(append_old)
    assert not blob_store.exists(leased_old)
    assert blob_store.exists(full_new)
    assert blob_store.exists(append_current)
    assert blob_store.exists(missing_old)
    assert blob_store.exists(missing_new)

    remaining_raw_ids = {
        str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions ORDER BY raw_id").fetchall()
    }
    assert remaining_raw_ids == {full_new, append_current, missing_old, missing_new}


def test_provider_event_raw_index_is_ensured_on_existing_archive(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "archive.db")
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)

    indexes = {str(row[1]) for row in conn.execute("PRAGMA index_list(provider_events)").fetchall()}
    assert "idx_provider_events_raw_id" in indexes


def test_archive_cleanup_compacts_append_snapshot_without_provider_events(tmp_path: Path) -> None:
    # Archive file-set storage has no ``provider_events`` table (a retired single-file construct), so
    # cleanup never manages a provider-event index or clears provider-event
    # links; it simply compacts the superseded append snapshot.
    db_path = tmp_path / "source.db"
    source = tmp_path / "rollout.jsonl"
    source.write_text('{"type":"message"}\n', encoding="utf-8")
    blob_store = BlobStore(tmp_path / "blob")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_archive_source_schema(conn)

    old_raw, old_size = _write_blob(blob_store, b"old")
    current_raw, current_size = _write_blob(blob_store, b"current")
    _insert_archive_raw_session(
        conn,
        raw_id=old_raw,
        source_path=source,
        source_index=-1,
        blob_hash=old_raw,
        blob_size=old_size,
        acquired_at_ms=1_000,
    )
    _insert_archive_raw_session(
        conn,
        raw_id=current_raw,
        source_path=source,
        source_index=-1,
        blob_hash=current_raw,
        blob_size=current_size,
        acquired_at_ms=2_000,
    )
    conn.commit()

    dry_run = cleanup_superseded_raw_snapshots(conn, dry_run=True, blob_store=blob_store)
    assert dry_run.candidate_count == 1

    result = cleanup_superseded_raw_snapshots(conn, dry_run=False, blob_store=blob_store)
    assert result.deleted_raw_count == 1
    assert result.provider_event_links_cleared == 0
    assert not blob_store.exists(old_raw)
    assert blob_store.exists(current_raw)
    assert conn.execute("SELECT 1 FROM raw_sessions WHERE raw_id = ?", (old_raw,)).fetchone() is None
    assert conn.execute("SELECT 1 FROM raw_sessions WHERE raw_id = ?", (current_raw,)).fetchone() is not None


def test_superseded_raw_snapshot_cleanup_uses_archive_blob_hashes(tmp_path: Path) -> None:
    db_path = tmp_path / "source.db"
    source = tmp_path / "rollout.jsonl"
    source.write_text('{"type":"message"}\n', encoding="utf-8")
    blob_store = BlobStore(tmp_path / "blob")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_archive_source_schema(conn)

    old_blob, old_size = _write_blob(blob_store, b"v1-old")
    current_blob, current_size = _write_blob(blob_store, b"v1-current")
    _insert_archive_raw_session(
        conn,
        raw_id="raw-old-not-a-blob-hash",
        source_path=source,
        source_index=0,
        blob_hash=old_blob,
        blob_size=old_size,
        acquired_at_ms=1_790_000_000_000,
    )
    _insert_archive_raw_session(
        conn,
        raw_id="raw-current-not-a-blob-hash",
        source_path=source,
        source_index=0,
        blob_hash=current_blob,
        blob_size=current_size,
        acquired_at_ms=1_790_000_060_000,
    )
    conn.commit()

    candidates = superseded_raw_snapshot_candidates(conn, limit=100)
    assert [(candidate.raw_id, candidate.blob_store_hash) for candidate in candidates] == [
        ("raw-old-not-a-blob-hash", old_blob)
    ]

    result = cleanup_superseded_raw_snapshots(conn, dry_run=False, blob_store=blob_store)

    assert result.deleted_raw_count == 1
    assert result.deleted_blob_count == 1
    assert result.provider_event_links_cleared == 0
    assert not blob_store.exists(old_blob)
    assert blob_store.exists(current_blob)
    assert conn.execute("SELECT 1 FROM raw_sessions WHERE raw_id = 'raw-old-not-a-blob-hash'").fetchone() is None
    assert conn.execute("SELECT 1 FROM blob_refs WHERE raw_id = 'raw-old-not-a-blob-hash'").fetchone() is None
