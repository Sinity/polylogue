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


def test_format_foreign_key_violations_renders_tuple_rows() -> None:
    rendered = ingest_batch_core._format_foreign_key_violations(
        [
            ("messages", 42, "sessions", 0),
            ("blocks", 7, "messages", 1),
        ]
    )

    assert "sqlite3.Row object" not in rendered
    assert "'table': 'messages'" in rendered
    assert "'rowid': 42" in rendered
    assert "'parent': 'sessions'" in rendered
    assert "'fkid': 0" in rendered


def test_format_foreign_key_violations_renders_sqlite_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "fk.db"
    with open_connection(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("CREATE TABLE parent (id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE child (id TEXT PRIMARY KEY, parent_id TEXT REFERENCES parent(id))")
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("INSERT INTO child (id, parent_id) VALUES ('child-1', 'missing-parent')")
        rows = conn.execute("PRAGMA foreign_key_check").fetchall()

    rendered = ingest_batch_core._format_foreign_key_violations(rows)

    assert "sqlite3.Row object" not in rendered
    assert "'table': 'child'" in rendered
    assert "'parent': 'parent'" in rendered
    assert "'fkid': 0" in rendered


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

    def fake_checkpoint(db: Path, *, reason: str, allow_truncate: bool = True, **_: object) -> WalCheckpointObservation:
        assert db == db_path
        assert reason == "ingest_batch_commit"
        assert allow_truncate is False
        return WalCheckpointObservation(
            reason=reason,
            mode="passive",
            wal_bytes_before=900,
            wal_bytes_after=900,
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

    assert summary.wal_checkpoint_mode == "passive"
    assert summary.wal_bytes_before_checkpoint == 900
    assert summary.wal_bytes_after_checkpoint == 900
    assert summary.wal_checkpointed_pages == 7
    assert summary.wal_checkpoint_elapsed_s == 0.25


def test_process_ingest_batch_sync_does_not_force_memory_release_before_returning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.db"
    archive_root = tmp_path / "archive"
    blob_root = tmp_path / "blob"
    source_path = tmp_path / "large-raw.jsonl"
    source_path.write_text("{}", encoding="utf-8")
    raw_record = RawSessionRecord(
        raw_id="raw-large",
        source_name="codex",
        source_path=str(source_path),
        blob_size=2 * 1024 * 1024 * 1024,
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

    def fail_if_sync_releases_memory() -> None:
        raise AssertionError("sync ingest finalization must not run memory release before returning")

    monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest_record)
    monkeypatch.setattr(ingest_batch_core, "release_process_memory", fail_if_sync_releases_memory)

    summary = _process_ingest_batch_sync(
        [raw_record],
        db_path=db_path,
        archive_root_str=str(archive_root),
        blob_root_str=str(blob_root),
        validation_mode="off",
        ingest_workers=1,
        measure_ingest_result_size=False,
    )

    assert summary.raw_record_count == 1
    assert summary.total_blob_mb >= 1024.0


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
