"""FTS repair contracts for ingest-batch unchanged-content paths."""

from __future__ import annotations

from pathlib import Path

import pytest

import polylogue.pipeline.services.ingest_batch._core as ingest_batch_core
from polylogue.pipeline.services.ingest_batch import _process_ingest_batch_sync
from polylogue.pipeline.services.ingest_worker import IngestRecordResult
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.connection import open_connection
from tests.unit.pipeline.test_ingest_batch import _message_tuple, _session_data

_write_session = ingest_batch_core._write_session


@pytest.mark.parametrize("content_changed", [False, True])
def test_process_ingest_batch_repairs_fts_without_invalidating_unchanged_insights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content_changed: bool,
) -> None:
    """Passing repair IDs as content changes nulls the unchanged row's fresh insight stamp."""
    db_path = tmp_path / "index.db"
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
    session_id = "codex-session:unchanged-fts"
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
    )

    with open_connection(db_path) as conn:
        changed, _counts = _write_session(conn, session)
        assert changed is True
        from polylogue.storage.fts.fts_lifecycle import repair_fts_index_sync

        repair_fts_index_sync(conn, [session_id])
        conn.execute(
            "INSERT INTO session_profiles (session_id, source_sort_key, source_updated_at) VALUES (?, ?, ?)",
            (session_id, 17.0, "2026-04-02T00:00:00Z"),
        )
        conn.commit()

        block_rowid = conn.execute(
            """
            SELECT b.rowid
            FROM blocks b
            JOIN messages m ON m.message_id = b.message_id
            WHERE m.session_id = ? AND m.native_id = ?
            """,
            (session_id, message_id),
        ).fetchone()[0]
        conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (block_rowid,))
        conn.commit()

    incoming_session = session
    if content_changed:
        incoming_session = _session_data(
            session_id,
            content_hash="changed-content-hash",
            message_tuples=[
                _message_tuple(
                    message_id,
                    session_id,
                    role="user",
                    text="changed content requires new insights",
                    content_hash="changed-message-hash",
                    sort_key=0.0,
                )
            ],
        )

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
        return IngestRecordResult(raw_id=record.raw_id, sessions=[incoming_session])

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

    assert summary.changed_session_ids == ([session_id] if content_changed else [])
    assert summary.fts_repair_session_ids == [session_id]

    with open_connection(db_path) as conn:
        stamp = conn.execute(
            "SELECT source_sort_key, source_updated_at FROM session_profiles WHERE session_id = ?", (session_id,)
        ).fetchone()
        assert stamp is not None
        assert tuple(stamp) == ((None, None) if content_changed else (17.0, "2026-04-02T00:00:00Z"))
        message_fts_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM messages_fts_docsize
            WHERE id = (
                SELECT b.rowid
                FROM blocks b
                JOIN messages m ON m.message_id = b.message_id
                WHERE m.session_id = ? AND m.native_id = ?
            )
            """,
            (session_id, message_id),
        ).fetchone()[0]

    assert message_fts_count == 1


def test_bulk_repair_targets_include_unchanged_sessions_missing_fts() -> None:
    """A batched run's bulk FTS repair covers sessions queued only for repair.

    Anti-vacuity: drop the ``fts_repair_session_ids`` merge in
    ``apply_ingest_batch_summary`` and ``unchanged-missing-fts`` disappears
    from the repair targets, so its search rows are never rebuilt.
    """
    from polylogue.pipeline.services.ingest_batch._models import _IngestBatchSummary
    from polylogue.pipeline.services.ingest_batch._summary import apply_ingest_batch_summary
    from polylogue.pipeline.services.parsing_models import ParseResult

    result = ParseResult()
    summary = _IngestBatchSummary(
        changed_session_ids=["changed"],
        fts_repair_session_ids=["unchanged-missing-fts", "changed"],
    )
    apply_ingest_batch_summary(result, summary)

    assert result.changed_session_ids == ("changed",)
    assert result.fts_repair_session_ids == ("changed", "unchanged-missing-fts")
