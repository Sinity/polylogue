"""FTS repair contracts for ingest-batch unchanged-content paths."""

from __future__ import annotations

from pathlib import Path

import pytest

import polylogue.pipeline.services.ingest_batch._core as ingest_batch_core
from polylogue.pipeline.services.ingest_batch import _process_ingest_batch_sync, _write_session
from polylogue.pipeline.services.ingest_worker import IngestRecordResult
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.connection import open_connection
from tests.unit.pipeline.test_ingest_batch import _action_event_tuple, _message_tuple, _session_data


def test_process_ingest_batch_repairs_fts_for_unchanged_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "polylogue.db"
    archive_root = tmp_path / "archive"
    blob_root = tmp_path / "blob"
    source_path = tmp_path / "raw.jsonl"
    source_path.write_text("{}", encoding="utf-8")
    raw_record = RawSessionRecord(
        raw_id="raw-unchanged-fts",
        source_name="codex",
        source_path=str(source_path),
        blob_size=source_path.stat().st_size,
        acquired_at="2026-04-02T00:00:00Z",
    )
    session_id = "codex:unchanged-fts"
    message_id = "msg-unchanged-fts"
    session = _session_data(
        session_id,
        content_hash="hash-unchanged-fts",
        message_tuples=[
            _message_tuple(
                message_id,
                session_id,
                role="user",
                text="unchanged content still needs stale FTS repair",
                content_hash="hash-unchanged-message",
                sort_key=0.0,
            )
        ],
        action_event_tuples=[
            _action_event_tuple(
                event_id="event-unchanged-fts",
                session_id=session_id,
                message_id=message_id,
                search_text="unchanged action event needs stale FTS repair",
            )
        ],
    )

    with open_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions
                (raw_id, payload_provider, source_name, source_path, blob_size, acquired_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                raw_record.raw_id,
                raw_record.source_name,
                raw_record.source_name,
                raw_record.source_path,
                raw_record.blob_size,
                raw_record.acquired_at,
            ),
        )
        changed, _counts = _write_session(conn, session)
        assert changed is True
        from polylogue.storage.fts.fts_lifecycle import repair_fts_index_sync

        repair_fts_index_sync(conn, [session_id])
        conn.commit()

        rowid = conn.execute("SELECT rowid FROM messages WHERE message_id = ?", (message_id,)).fetchone()[0]
        conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (rowid,))
        conn.execute("DELETE FROM action_events_fts WHERE session_id = ?", (session_id,))
        conn.commit()

    def fake_ingest_record(
        record: RawSessionRecord,
        archive_root_str: str,
        validation_mode: str,
        measure_ingest_result_size: bool,
        *,
        blob_root_str: str | None,
    ) -> IngestRecordResult:
        del archive_root_str, validation_mode, measure_ingest_result_size, blob_root_str
        assert record.raw_id == raw_record.raw_id
        return IngestRecordResult(raw_id=record.raw_id, sessions=[session])

    monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest_record)

    summary = _process_ingest_batch_sync(
        [raw_record],
        db_path=db_path,
        archive_root_str=str(archive_root),
        blob_root_str=str(blob_root),
        validation_mode="off",
        ingest_workers=1,
        measure_ingest_result_size=False,
    )

    assert summary.changed_session_ids == []
    assert summary.fts_repair_session_ids == [session_id]

    with open_connection(db_path) as conn:
        message_fts_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM messages_fts_docsize
            WHERE id = (SELECT rowid FROM messages WHERE message_id = ?)
            """,
            (message_id,),
        ).fetchone()[0]
        action_fts_count = conn.execute(
            "SELECT COUNT(*) FROM action_events_fts_docsize WHERE id = (SELECT rowid FROM action_events WHERE event_id = ?)",
            ("event-unchanged-fts",),
        ).fetchone()[0]

    assert message_fts_count == 1
    assert action_fts_count == 1
