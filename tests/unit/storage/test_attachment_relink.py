"""polylogue-w06b: relinking orphaned ``attachments`` rows via raw re-parse.

Exercises the real production entry points end to end -- ``ingest_record``
(the live ingest worker's per-raw parse entry point) parses genuine raw bytes
out of a real ``source.db``/blob store, and ``relink_orphaned_attachments``
writes the recovered ``attachment_refs`` row back into a real ``index.db`` --
no mocking of the parse or matching logic under test.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.pipeline.services.ingest_worker import ingest_record
from polylogue.storage.attachment_relink import (
    plan_orphaned_attachment_relink,
    relink_orphaned_attachments,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.runtime.raw.records import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive

_CLAUDE_AI_PAYLOAD = {
    "uuid": "relink-session-1",
    "name": "Relink test session",
    "chat_messages": [
        {
            "uuid": "m0",
            "sender": "human",
            "text": "here is a file",
            "attachments": [
                {
                    "file_name": "notes.md",
                    "file_type": "text/markdown",
                    "file_size": 11,
                    "extracted_content": "hello notes",
                }
            ],
        }
    ],
}


def _index_conn(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _source_conn(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.SOURCE)
    return conn


def _write_raw_row(
    source_conn: sqlite3.Connection, blob_store: BlobStore, payload: object, *, raw_id: str, source_path: str
) -> RawSessionRecord:
    content = json.dumps(payload).encode("utf-8")
    blob_hash, blob_size = blob_store.write_from_bytes(content)
    source_conn.execute(
        """
        INSERT INTO raw_sessions (raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms)
        VALUES (?, 'claude-ai-export', ?, 0, ?, ?, 0)
        """,
        (raw_id, source_path, bytes.fromhex(blob_hash), blob_size),
    )
    source_conn.commit()
    return RawSessionRecord(
        raw_id=raw_id,
        source_name=Provider.CLAUDE_AI.value,
        payload_provider=Provider.CLAUDE_AI,
        source_path=source_path,
        source_index=0,
        blob_size=blob_size,
        blob_hash=blob_hash,
        acquired_at="2026-01-01T00:00:00+00:00",
    )


def _preacquired_blobs(store: BlobStore, session: object) -> dict[int, tuple[bytes | None, int, str]]:
    acquired: dict[int, tuple[bytes | None, int, str]] = {}
    for attachment in session.attachments:  # type: ignore[attr-defined]
        if attachment.inline_bytes is None:
            continue
        blob_hash, size = store.write_from_bytes(attachment.inline_bytes)
        acquired[id(attachment)] = (bytes.fromhex(blob_hash), size, "acquired")
    return acquired


def test_orphaned_attachment_is_relinked_from_raw_reparse(tmp_path: Path) -> None:
    """The exact live-archive symptom (ref_count 0, real acquired bytes,
    unreachable from any session) is repaired when the owning raw session
    and its owning message are still present.
    """
    blob_store = BlobStore(tmp_path / "blob")
    index_conn = _index_conn(tmp_path / "index.db")
    source_conn = _source_conn(tmp_path / "source.db")

    raw_record = _write_raw_row(
        source_conn, blob_store, _CLAUDE_AI_PAYLOAD, raw_id="raw-1", source_path="conversations.json"
    )

    result = ingest_record(raw_record, str(tmp_path), "advisory", blob_root_str=str(blob_store.root))
    assert result.error is None, result.error
    assert len(result.sessions) == 1
    write_payload = result.sessions[0]

    # Establish the "before the bug" good state: attachment written with a
    # live ref, exactly as first ingest would have produced.
    write_parsed_session_to_archive(
        index_conn,
        write_payload.parsed_session,
        raw_id=raw_record.raw_id,
        preacquired_attachment_blobs=_preacquired_blobs(blob_store, write_payload.parsed_session),
    )
    row = index_conn.execute(
        "SELECT attachment_id, ref_count, acquisition_status FROM attachments WHERE display_name = 'notes.md'"
    ).fetchone()
    assert row is not None
    attachment_id = str(row["attachment_id"])
    assert row["ref_count"] == 1
    assert row["acquisition_status"] == "acquired"

    # Simulate the historical polylogue-w06b orphan bug: the ref got dropped
    # (e.g. by a pre-#3514 full-replace pass) but the attachments row itself
    # survived with ref_count now stale-0 -- the exact shape the bead's
    # forensics found live (acquisition_status='acquired', ref_count 0).
    index_conn.execute("DELETE FROM attachment_refs WHERE attachment_id = ?", (attachment_id,))
    index_conn.execute("UPDATE attachments SET ref_count = 0 WHERE attachment_id = ?", (attachment_id,))
    index_conn.commit()

    orphan_check = index_conn.execute(
        "SELECT 1 FROM attachments a WHERE a.attachment_id = ? "
        "AND NOT EXISTS (SELECT 1 FROM attachment_refs r WHERE r.attachment_id = a.attachment_id)",
        (attachment_id,),
    ).fetchone()
    assert orphan_check is not None, "test setup failed to reproduce an orphaned attachment row"

    plan = plan_orphaned_attachment_relink(index_conn, source_conn, archive_root=tmp_path, blob_root=blob_store.root)
    assert plan.orphan_count == 1
    assert len(plan.eligible) == 1
    assert plan.eligible[0].attachment_id == attachment_id
    assert plan.eligible[0].session_id == write_payload.session_id
    assert plan.unrecoverable_reason_counts == {}

    exec_result = relink_orphaned_attachments(
        index_conn, source_conn, archive_root=tmp_path, blob_root=blob_store.root, dry_run=False
    )
    index_conn.commit()
    assert exec_result.relinked_count == 1
    assert exec_result.errors == ()

    relinked_row = index_conn.execute(
        "SELECT ref_count FROM attachments WHERE attachment_id = ?", (attachment_id,)
    ).fetchone()
    assert relinked_row is not None
    assert relinked_row["ref_count"] == 1

    ref_row = index_conn.execute(
        "SELECT session_id, message_id FROM attachment_refs WHERE attachment_id = ?", (attachment_id,)
    ).fetchone()
    assert ref_row is not None
    assert ref_row["session_id"] == write_payload.session_id


@pytest.mark.asyncio
async def test_orphaned_attachment_is_reachable_via_production_read_path_after_relink(tmp_path: Path) -> None:
    """Anti-vacuity: after relink, the production ``get_attachments`` read
    path -- the one every session/message attachment surface (MCP, CLI
    ``read --view``, transcript view) goes through -- can see it. Reverting
    the ``INSERT OR REPLACE INTO attachment_refs`` write in
    ``relink_orphaned_attachments`` makes this assertion fail: the row would
    still be present in ``attachments`` (visible via a direct SELECT) but
    ``get_attachments`` INNER JOINs ``attachment_refs``, so a still-ref-less
    row can never be returned.
    """
    import aiosqlite

    from polylogue.storage.sqlite.queries.attachment_records import get_attachments

    blob_store = BlobStore(tmp_path / "blob")
    index_db_path = tmp_path / "index.db"
    index_conn = _index_conn(index_db_path)
    source_conn = _source_conn(tmp_path / "source.db")

    raw_record = _write_raw_row(
        source_conn, blob_store, _CLAUDE_AI_PAYLOAD, raw_id="raw-1", source_path="conversations.json"
    )
    result = ingest_record(raw_record, str(tmp_path), "advisory", blob_root_str=str(blob_store.root))
    assert result.error is None, result.error
    write_payload = result.sessions[0]

    write_parsed_session_to_archive(
        index_conn,
        write_payload.parsed_session,
        raw_id=raw_record.raw_id,
        preacquired_attachment_blobs=_preacquired_blobs(blob_store, write_payload.parsed_session),
    )
    attachment_id = str(
        index_conn.execute("SELECT attachment_id FROM attachments WHERE display_name = 'notes.md'").fetchone()[0]
    )
    index_conn.execute("DELETE FROM attachment_refs WHERE attachment_id = ?", (attachment_id,))
    index_conn.execute("UPDATE attachments SET ref_count = 0 WHERE attachment_id = ?", (attachment_id,))
    index_conn.commit()

    async with aiosqlite.connect(index_db_path) as aconn:
        aconn.row_factory = aiosqlite.Row
        before = await get_attachments(aconn, write_payload.session_id)
    assert all(record.attachment_id != attachment_id for record in before)

    exec_result = relink_orphaned_attachments(
        index_conn, source_conn, archive_root=tmp_path, blob_root=blob_store.root, dry_run=False
    )
    index_conn.commit()
    assert exec_result.relinked_count == 1

    async with aiosqlite.connect(index_db_path) as aconn:
        aconn.row_factory = aiosqlite.Row
        after = await get_attachments(aconn, write_payload.session_id)
    assert any(record.attachment_id == attachment_id for record in after)


def test_orphan_with_no_matching_raw_is_reported_unrecoverable_not_guessed(tmp_path: Path) -> None:
    """An orphan whose identity no raw session in source.db reproduces must
    be reported unrecoverable, never silently linked to an unrelated
    message.
    """
    blob_store = BlobStore(tmp_path / "blob")
    index_conn = _index_conn(tmp_path / "index.db")
    source_conn = _source_conn(tmp_path / "source.db")

    # A raw session exists in source.db, but it shares no attachment
    # identity with the orphan row below.
    _write_raw_row(source_conn, blob_store, _CLAUDE_AI_PAYLOAD, raw_id="raw-1", source_path="conversations.json")

    index_conn.execute(
        "INSERT INTO attachments (attachment_id, display_name, media_type, byte_count, acquisition_status, ref_count) "
        "VALUES ('ghost-attachment-id', 'ghost.bin', 'application/octet-stream', 42, 'acquired', 0)"
    )
    index_conn.commit()

    plan = plan_orphaned_attachment_relink(index_conn, source_conn, archive_root=tmp_path, blob_root=blob_store.root)
    assert plan.orphan_count == 1
    assert plan.eligible == ()
    assert sum(plan.unrecoverable_reason_counts.values()) == 1
    assert plan.unrecoverable_samples[0].attachment_id == "ghost-attachment-id"
    assert "no raw session" in plan.unrecoverable_samples[0].reason

    exec_result = relink_orphaned_attachments(
        index_conn, source_conn, archive_root=tmp_path, blob_root=blob_store.root, dry_run=False
    )
    index_conn.commit()
    assert exec_result.relinked_count == 0

    surviving = index_conn.execute(
        "SELECT ref_count FROM attachments WHERE attachment_id = 'ghost-attachment-id'"
    ).fetchone()
    assert surviving is not None
    assert surviving["ref_count"] == 0


def test_plan_is_dry_run_by_default_makes_no_writes(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blob")
    index_conn = _index_conn(tmp_path / "index.db")
    source_conn = _source_conn(tmp_path / "source.db")

    raw_record = _write_raw_row(
        source_conn, blob_store, _CLAUDE_AI_PAYLOAD, raw_id="raw-1", source_path="conversations.json"
    )
    result = ingest_record(raw_record, str(tmp_path), "advisory", blob_root_str=str(blob_store.root))
    write_payload = result.sessions[0]
    write_parsed_session_to_archive(
        index_conn,
        write_payload.parsed_session,
        raw_id=raw_record.raw_id,
        preacquired_attachment_blobs=_preacquired_blobs(blob_store, write_payload.parsed_session),
    )
    attachment_id = str(
        index_conn.execute("SELECT attachment_id FROM attachments WHERE display_name = 'notes.md'").fetchone()[0]
    )
    index_conn.execute("DELETE FROM attachment_refs WHERE attachment_id = ?", (attachment_id,))
    index_conn.execute("UPDATE attachments SET ref_count = 0 WHERE attachment_id = ?", (attachment_id,))
    index_conn.commit()

    exec_result = relink_orphaned_attachments(index_conn, source_conn, archive_root=tmp_path, blob_root=blob_store.root)
    assert exec_result.eligible_count == 1
    assert exec_result.relinked_count == 0  # dry_run=True default: plans, never writes

    still_orphaned = index_conn.execute(
        "SELECT 1 FROM attachment_refs WHERE attachment_id = ?", (attachment_id,)
    ).fetchone()
    assert still_orphaned is None
