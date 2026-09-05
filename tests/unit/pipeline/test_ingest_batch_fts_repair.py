"""FTS repair contracts for ingest-batch unchanged-content paths."""

from __future__ import annotations

from pathlib import Path

import pytest

import polylogue.pipeline.services.ingest_batch._core as ingest_batch_core
from polylogue.pipeline.services.ingest_batch import _process_ingest_batch_sync
from polylogue.pipeline.services.ingest_worker import IngestRecordResult
from polylogue.storage.fts.freshness import (
    message_fts_recorded_exact_stale_sync,
    record_fts_invariant_snapshot_sync,
)
from polylogue.storage.fts.fts_lifecycle import fts_invariant_snapshot_sync
from polylogue.storage.fts.session_repair import session_fts_needs_repair_sync
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.connection import open_connection
from tests.unit.pipeline.test_ingest_batch import _message_tuple, _session_data

_write_session = ingest_batch_core._write_session


def test_process_ingest_batch_repairs_fts_for_unchanged_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_process_ingest_batch_keeps_repair_when_recorded_state_is_exact_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exact stale verdict keeps materialized sessions queued for repair.

    The session-local probe (``session_fts_needs_repair_sync``) only sees its
    own session. When the archive carries an archive-wide exact stale verdict,
    a drained session whose own rows happen to balance must still be scheduled
    -- otherwise cleanup silently drops the repair that would clear the
    verdict and the surface stays unsearchable.

    Mutation that fails this: remove the ``message_fts_recorded_exact_stale_sync``
    branch from ``_drain_ready_session_entries``. Nothing is missing for this
    session, so ``fts_repair_session_ids`` comes back empty.
    """
    db_path = tmp_path / "index.db"
    archive_root = tmp_path / "archive"
    blob_root = tmp_path / "blob"
    source_path = tmp_path / "raw.jsonl"
    source_path.write_text("{}", encoding="utf-8")
    raw_record = RawSessionRecord(
        raw_id="raw-exact-stale-fts",
        source_name="codex",
        source_path=str(source_path),
        blob_size=source_path.stat().st_size,
        acquired_at="2026-04-02T00:00:00Z",
    )
    session_id = "codex-session:exact-stale-fts"
    session = _session_data(
        session_id,
        content_hash="hash-exact-stale-fts",
        message_tuples=[
            _message_tuple(
                "msg-exact-stale-fts",
                session_id,
                role="user",
                text="content whose own FTS rows are complete",
                content_hash="hash-exact-stale-message",
                sort_key=0.0,
            )
        ],
    )

    with open_connection(db_path) as conn:
        changed, _counts = _write_session(conn, session)
        assert changed is True
        from polylogue.storage.fts.fts_lifecycle import repair_fts_index_sync

        repair_fts_index_sync(conn, [session_id])
        conn.commit()

        # This session is internally complete: the session-local probe alone
        # would drop it from the repair queue.
        assert session_fts_needs_repair_sync(conn, session_id) is False

        # Record an archive-wide EXACT stale verdict without touching this
        # session's rows, the shape a deferred repair elsewhere leaves behind.
        snapshot = fts_invariant_snapshot_sync(conn)
        record_fts_invariant_snapshot_sync(conn, snapshot)
        conn.execute(
            "UPDATE fts_freshness_state SET state = 'stale', identity_mismatch_rows = 1 WHERE surface = 'messages_fts'"
        )
        conn.commit()
        assert message_fts_recorded_exact_stale_sync(conn) is True

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

    assert session_id in summary.fts_repair_session_ids
