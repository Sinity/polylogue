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


def _insert_raw(
    conn: sqlite3.Connection,
    *,
    raw_id: str,
    source_path: Path,
    source_index: int,
    blob_size: int,
    acquired_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO raw_conversations (
            raw_id, source_name, source_path, source_index,
            blob_size, acquired_at, file_mtime
        ) VALUES (?, 'codex', ?, ?, ?, ?, ?)
        """,
        (raw_id, str(source_path), source_index, blob_size, acquired_at, acquired_at),
    )


def _write_blob(store: BlobStore, payload: bytes) -> tuple[str, int]:
    return store.write_from_bytes(payload)


def test_superseded_raw_snapshot_cleanup_preserves_live_references(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    source = tmp_path / "rollout.jsonl"
    missing_source = tmp_path / "missing.jsonl"
    source.write_text('{"type":"message"}\n', encoding="utf-8")
    blob_store = BlobStore(tmp_path / "blob")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_schema(conn)

    full_old, full_old_size = _write_blob(blob_store, b"full-old")
    full_new, full_new_size = _write_blob(blob_store, b"full-new")
    append_old, append_old_size = _write_blob(blob_store, b"append-old")
    append_current, append_current_size = _write_blob(blob_store, b"append-current")
    leased_old, leased_old_size = _write_blob(blob_store, b"leased-old")
    missing_old, missing_old_size = _write_blob(blob_store, b"missing-old")
    missing_new, missing_new_size = _write_blob(blob_store, b"missing-new")

    _insert_raw(
        conn,
        raw_id=full_old,
        source_path=source,
        source_index=0,
        blob_size=full_old_size,
        acquired_at="2026-05-24T00:00:00+00:00",
    )
    _insert_raw(
        conn,
        raw_id=full_new,
        source_path=source,
        source_index=0,
        blob_size=full_new_size,
        acquired_at="2026-05-24T00:01:00+00:00",
    )
    _insert_raw(
        conn,
        raw_id=append_old,
        source_path=source,
        source_index=-1,
        blob_size=append_old_size,
        acquired_at="2026-05-24T00:02:00+00:00",
    )
    _insert_raw(
        conn,
        raw_id=append_current,
        source_path=source,
        source_index=-1,
        blob_size=append_current_size,
        acquired_at="2026-05-24T00:03:00+00:00",
    )
    _insert_raw(
        conn,
        raw_id=leased_old,
        source_path=source,
        source_index=-1,
        blob_size=leased_old_size,
        acquired_at="2026-05-24T00:01:30+00:00",
    )
    _insert_raw(
        conn,
        raw_id=missing_old,
        source_path=missing_source,
        source_index=0,
        blob_size=missing_old_size,
        acquired_at="2026-05-24T00:00:00+00:00",
    )
    _insert_raw(
        conn,
        raw_id=missing_new,
        source_path=missing_source,
        source_index=0,
        blob_size=missing_new_size,
        acquired_at="2026-05-24T00:01:00+00:00",
    )
    conn.execute(
        """
        INSERT INTO conversations (
            conversation_id, source_name, provider_conversation_id,
            content_hash, source_name, version, raw_id
        ) VALUES ('conv-1', 'codex', 'conv-1', 'hash', 'codex-session', 1, ?)
        """,
        (append_current,),
    )
    conn.execute(
        """
        INSERT INTO provider_events (
            event_id, conversation_id, source_name, event_index,
            event_type, raw_id
        ) VALUES ('event-1', 'conv-1', 'codex', 1, 'message', ?)
        """,
        (append_old,),
    )
    conn.execute(
        "INSERT INTO pending_blob_refs (blob_hash, operation_id, acquired_at) VALUES (?, 'op-1', 1)",
        (leased_old,),
    )
    conn.commit()

    candidates = superseded_raw_snapshot_candidates(conn, limit=100)
    assert {candidate.raw_id for candidate in candidates} == {full_old, append_old}

    dry_run = cleanup_superseded_raw_snapshots(conn, dry_run=True, blob_store=blob_store)
    assert dry_run.candidate_count == 2
    assert blob_store.exists(full_old)
    assert blob_store.exists(append_old)

    result = cleanup_superseded_raw_snapshots(conn, dry_run=False, blob_store=blob_store)
    assert result.deleted_raw_count == 2
    assert result.deleted_blob_count == 2
    assert result.provider_event_links_cleared == 1
    assert not blob_store.exists(full_old)
    assert not blob_store.exists(append_old)
    assert blob_store.exists(full_new)
    assert blob_store.exists(append_current)
    assert blob_store.exists(leased_old)
    assert blob_store.exists(missing_old)

    remaining_raw_ids = {
        str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_conversations ORDER BY raw_id").fetchall()
    }
    assert full_old not in remaining_raw_ids
    assert append_old not in remaining_raw_ids
    assert append_current in remaining_raw_ids
    assert conn.execute("SELECT raw_id FROM provider_events WHERE event_id = 'event-1'").fetchone()[0] is None


def test_provider_event_raw_index_is_ensured_on_existing_archive(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "archive.db")
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)

    indexes = {str(row[1]) for row in conn.execute("PRAGMA index_list(provider_events)").fetchall()}
    assert "idx_provider_events_raw_id" in indexes


def test_destructive_cleanup_adds_provider_event_raw_index_when_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    source = tmp_path / "rollout.jsonl"
    source.write_text('{"type":"message"}\n', encoding="utf-8")
    blob_store = BlobStore(tmp_path / "blob")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_schema(conn)
    conn.execute("DROP INDEX idx_provider_events_raw_id")

    old_raw, old_size = _write_blob(blob_store, b"old")
    current_raw, current_size = _write_blob(blob_store, b"current")
    _insert_raw(
        conn,
        raw_id=old_raw,
        source_path=source,
        source_index=-1,
        blob_size=old_size,
        acquired_at="2026-05-24T00:00:00+00:00",
    )
    _insert_raw(
        conn,
        raw_id=current_raw,
        source_path=source,
        source_index=-1,
        blob_size=current_size,
        acquired_at="2026-05-24T00:01:00+00:00",
    )
    conn.execute(
        """
        INSERT INTO conversations (
            conversation_id, source_name, provider_conversation_id,
            content_hash, source_name, version, raw_id
        ) VALUES ('conv-1', 'codex', 'conv-1', 'hash', 'codex-session', 1, ?)
        """,
        (current_raw,),
    )
    conn.execute(
        """
        INSERT INTO provider_events (
            event_id, conversation_id, source_name, event_index,
            event_type, raw_id
        ) VALUES ('event-1', 'conv-1', 'codex', 1, 'message', ?)
        """,
        (old_raw,),
    )
    conn.commit()

    dry_run = cleanup_superseded_raw_snapshots(conn, dry_run=True, blob_store=blob_store)
    assert dry_run.candidate_count == 1
    indexes_after_dry_run = {str(row[1]) for row in conn.execute("PRAGMA index_list(provider_events)").fetchall()}
    assert "idx_provider_events_raw_id" not in indexes_after_dry_run

    result = cleanup_superseded_raw_snapshots(conn, dry_run=False, blob_store=blob_store)
    assert result.deleted_raw_count == 1
    assert result.provider_event_links_cleared == 1
    assert conn.execute("SELECT raw_id FROM provider_events WHERE event_id = 'event-1'").fetchone()[0] is None

    indexes_after_cleanup = {str(row[1]) for row in conn.execute("PRAGMA index_list(provider_events)").fetchall()}
    assert "idx_provider_events_raw_id" in indexes_after_cleanup
