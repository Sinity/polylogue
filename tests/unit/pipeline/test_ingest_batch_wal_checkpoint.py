"""WAL checkpoint observation tests for sync ingest batches."""

from __future__ import annotations

from pathlib import Path

import pytest

import polylogue.pipeline.services.ingest_batch._core as ingest_batch_core
from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.pipeline.services.ingest_batch import _process_ingest_batch_sync
from polylogue.pipeline.services.ingest_worker import IngestRecordResult
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.connection import open_connection
from polylogue.storage.sqlite.wal_checkpoint import WalCheckpointObservation, maybe_checkpoint_wal


def test_process_ingest_batch_sync_records_wal_checkpoint_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.db"
    archive_root = tmp_path / "archive"
    blob_root = tmp_path / "blob"
    source_path = tmp_path / "raw.jsonl"
    source_path.write_text("{}", encoding="utf-8")
    raw_record = RawSessionRecord(
        raw_id="raw-wal",
        source_name="codex",
        source_path=str(source_path),
        blob_size=source_path.stat().st_size,
        acquired_at="2026-04-02T00:00:00Z",
    )

    origin = origin_from_provider(Provider.from_string(raw_record.source_name)).value
    with open_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions
                (raw_id, origin, native_id, source_path, source_index,
                 blob_hash, blob_size, acquired_at_ms)
            VALUES (?, ?, ?, ?, 0, ?, ?, ?)
            """,
            (
                raw_record.raw_id,
                origin,
                raw_record.raw_id,
                raw_record.source_path,
                b"\x00" * 32,
                raw_record.blob_size,
                1_775_433_600_000,
            ),
        )
        conn.commit()

    def fake_ingest_record(
        record: RawSessionRecord,
        archive_root_str: str,
        validation_mode: str,
        measure_ingest_result_size: bool,
        *,
        blob_root_str: str | None,
    ) -> IngestRecordResult:
        del record, archive_root_str, validation_mode, measure_ingest_result_size, blob_root_str
        return IngestRecordResult(raw_id=raw_record.raw_id, sessions=[])

    def fake_checkpoint(db: Path, *, reason: str, **_: object) -> WalCheckpointObservation:
        assert db == db_path
        assert reason == "ingest_batch_commit"
        return WalCheckpointObservation(
            reason=reason,
            mode="truncate",
            wal_bytes_before=900,
            wal_bytes_after=0,
            busy_pages=0,
            log_pages=7,
            checkpointed_pages=7,
            elapsed_s=0.25,
        )

    monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest_record)
    monkeypatch.setattr("polylogue.storage.sqlite.wal_checkpoint.maybe_checkpoint_wal", fake_checkpoint)

    summary = _process_ingest_batch_sync(
        [raw_record],
        db_path=db_path,
        archive_root_str=str(archive_root),
        blob_root_str=str(blob_root),
        validation_mode="off",
        ingest_workers=1,
        measure_ingest_result_size=False,
    )

    assert summary.wal_checkpoint_mode == "truncate"
    assert summary.wal_bytes_before_checkpoint == 900
    assert summary.wal_bytes_after_checkpoint == 0
    assert summary.wal_checkpointed_pages == 7
    assert summary.wal_checkpoint_elapsed_s == 0.25


def test_maybe_checkpoint_wal_reports_blocking_processes_when_busy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.db"

    class FakeConnection:
        def execute(self, sql: str) -> object:
            assert sql == "PRAGMA wal_checkpoint(PASSIVE)"
            return self

        def fetchone(self) -> tuple[int, int, int]:
            return (1, 25, 12)

        def close(self) -> None:
            return None

    monkeypatch.setattr("polylogue.storage.sqlite.wal_checkpoint._wal_size", lambda db: 1024)
    monkeypatch.setattr(
        "polylogue.storage.sqlite.wal_checkpoint.open_connection", lambda *_args, **_kwargs: FakeConnection()
    )
    monkeypatch.setattr(
        "polylogue.storage.sqlite.wal_checkpoint._sqlite_file_holders",
        lambda db: ("1234:polylogue-mcp",) if db == db_path else (),
    )

    observation = maybe_checkpoint_wal(db_path, reason="test", warn_bytes=0, truncate_bytes=0)

    assert observation.busy_pages == 1
    assert observation.blocking_processes == ("1234:polylogue-mcp",)
