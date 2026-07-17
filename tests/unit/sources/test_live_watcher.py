"""Tests for the live filesystem watcher: cursor, ingest skip logic,
debounce, bootstrap scan, and end-to-end via the watchfiles event loop."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
import time
import zipfile
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue import Polylogue
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.batch import (
    _SMALL_FULL_PARSE_PROGRESS_MAX_BYTES,
    _SMALL_FULL_PARSE_PROGRESS_MAX_FILES,
    _STREAMING_FULL_INGEST_BYTES,
    LiveBatchProcessor,
    _full_ingest_worker_count,
    _full_parse_progress_groups,
    _FullIngestResult,
    last_complete_newline_from_tail,
)
from polylogue.sources.live.batch_support import encode_cursor_hash_authority
from polylogue.sources.live.cursor import CursorRecord, CursorStore
from polylogue.sources.live.metrics import LiveBatchMetrics
from polylogue.sources.sqlite_snapshot import sqlite_source_revision
from polylogue.storage.blob_store import BlobStore, PreparedBlob
from polylogue.storage.runtime import RawSessionRecord
from tests.infra.frozen_clock import FrozenClock


class _FullIngestMock:
    def __init__(self) -> None:
        self.await_count = 0
        self.side_effect: BaseException | type[BaseException] | None = None

    def reset_mock(self) -> None:
        self.await_count = 0

    async def __call__(
        self, paths: list[Path], *, source_name: str, heartbeat: object = None, attempt_id: str | None = None
    ) -> _FullIngestResult:
        del source_name, heartbeat, attempt_id
        self.await_count += 1
        if self.side_effect is not None:
            if isinstance(self.side_effect, BaseException):
                raise self.side_effect
            if isinstance(self.side_effect, type):
                raise self.side_effect()
        return _FullIngestResult(
            succeeded=list(paths),
            failed=[],
            source_payload_read_bytes=sum(path.stat().st_size for path in paths),
            raw_fingerprints={path: f"raw:{path.name}" for path in paths},
            ingested_session_count=1,
            ingested_message_count=7,
            changed_session_count=1,
            wal_bytes_before_checkpoint=8192,
            wal_bytes_after_checkpoint=1024,
            wal_checkpointed_pages=4,
            wal_busy_pages=2,
            wal_checkpoint_elapsed_s=0.125,
            wal_checkpoint_mode="truncate",
            stage_timings_s={"full.provider_parse": 0.01, "full.index_parsed_write": 0.02},
        )


class _FailingSecondPathConverger:
    def __init__(self, failing_path: Path) -> None:
        self.failing_path = failing_path

    def converge_batch(self, paths: tuple[Path, ...]) -> tuple[dict[Path, object], dict[str, float]]:
        if self.failing_path in paths:
            return (
                {path: SimpleNamespace(converged=path != self.failing_path) for path in paths},
                {"fake": 0.001},
            )
        return ({path: SimpleNamespace(converged=True) for path in paths}, {"fake": 0.001})


def test_live_ingest_metrics_log_separates_read_bytes_from_candidate_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = MagicMock()
    monkeypatch.setattr(live_watcher, "logger", logger)
    metrics = LiveBatchMetrics(
        queued_file_count=2,
        needed_file_count=2,
        skipped_file_count=0,
        succeeded_file_count=2,
        failed_file_count=0,
        source_group_count=1,
        input_bytes=400_000_000,
        source_payload_read_bytes=40_000,
        cursor_fingerprint_read_bytes=0,
        ingest_worker_count_max=1,
        append_file_count=2,
        full_file_count=0,
        archive_bytes_before=0,
        archive_bytes_after=0,
        archive_write_bytes_delta=0,
        parse_time_s=0.5,
        convergence_time_s=0.25,
        total_time_s=1.0,
        wal_bytes_before_checkpoint_max=8_000_000,
        wal_bytes_after_checkpoint_max=1_000_000,
        wal_busy_pages_total=3,
        stage_timings_s={"full_parse": 0.45, "fts": 0.05, "insights": 0.2},
    )

    live_watcher._log_ingest_metrics("live.watcher: changed-file batch", metrics)

    message, *args = logger.info.call_args.args
    assert "read=%.1f MB input=%.1f MB read_amp=%.6fx" in message
    assert "stages=%s" in message
    assert "wal_before_checkpoint=%.1f MB" in message
    assert args[:6] == [
        "live.watcher: changed-file batch",
        0.04,
        400.0,
        0.0001,
        2,
        0,
    ]
    assert args[10:] == ["full_parse:0.450,insights:0.200,fts:0.050", 8.0, 1.0, 3]


def test_live_ingest_stage_timing_summary_is_bounded_and_sorted() -> None:
    assert live_watcher._stage_timing_summary({}) == "none"
    assert (
        live_watcher._stage_timing_summary(
            {
                "parse": 1.25,
                "fts": 0.01,
                "insights": 0.4,
                "usage": 0.2,
                "checkpoint": 0.8,
            },
            limit=3,
        )
        == "parse:1.250,checkpoint:0.800,insights:0.400,+2 more"
    )


# --- CursorStore ---------------------------------------------------------------


def test_cursor_default_is_zero(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    assert store.get(tmp_path / "missing.jsonl") == 0


def test_cursor_round_trip(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    p = tmp_path / "session.jsonl"
    store.set(p, 42, record_count=3)
    assert store.get(p) == 42
    record = store.get_record(p)
    assert isinstance(record, CursorRecord)
    assert record.byte_size == 42
    assert record.byte_offset == 42
    assert record.last_complete_newline == 42
    assert record.record_count == 3


def test_cursor_upsert_overwrites(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    p = tmp_path / "session.jsonl"
    store.set(p, 100)
    store.set(p, 250, record_count=99)
    assert store.get(p) == 250


def test_cursor_isolated_per_path(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    store.set(a, 10)
    store.set(b, 20)
    assert store.get(a) == 10
    assert store.get(b) == 20


def test_cursor_fetches_records_in_bulk(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    missing = tmp_path / "missing.jsonl"

    store.set(first, 11, parser_fingerprint="parser", content_fingerprint="first-hash")
    store.set(second, 22, parser_fingerprint="parser", content_fingerprint="second-hash", excluded=True)

    records = store.get_records([first, second, missing, first])

    assert set(records) == {first, second}
    assert records[first].content_fingerprint == "first-hash"
    assert records[second].excluded is True


def test_cursor_persists_across_instances(tmp_path: Path) -> None:
    db = tmp_path / "live.sqlite"
    store_a = CursorStore(db)
    p = tmp_path / "session.jsonl"
    store_a.set(p, 555)
    store_b = CursorStore(db)
    assert store_b.get(p) == 555


def test_cursor_creates_table_if_missing(tmp_path: Path) -> None:
    db = tmp_path / "live.sqlite"
    CursorStore(db)
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ingest_cursor'").fetchall()
    assert rows == [("ingest_cursor",)]


def test_cursor_writes_updated_at(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    p = tmp_path / "s.jsonl"
    store.set(p, 1)
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row = conn.execute("SELECT updated_at_ms FROM ingest_cursor WHERE source_path=?", (str(p),)).fetchone()
    assert row[0]
    assert isinstance(row[0], int)


def test_cursor_records_live_ingest_attempt_progress(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')

    attempt_id = store.begin_ingest_attempt(
        paths=[source],
        input_bytes=source.stat().st_size,
        queued_file_count=1,
    )
    store.update_ingest_attempt(
        attempt_id,
        phase="full_parse",
        succeeded_file_count=0,
        failed_file_count=0,
        source_payload_read_bytes=0,
        cursor_fingerprint_read_bytes=0,
        parse_time_s=0.25,
        current_source="codex",
        current_path=source,
        rss_current_mb=123.0,
    )
    store.record_ingest_stage_event(
        attempt_id,
        phase="full_parse",
        status="running",
        queued_file_count=1,
        needed_file_count=1,
        skipped_file_count=0,
        succeeded_file_count=0,
        failed_file_count=0,
        input_bytes=source.stat().st_size,
        source_payload_read_bytes=0,
        cursor_fingerprint_read_bytes=0,
        parse_time_s=0.25,
        current_source="codex",
        current_path=source,
    )
    running = store.recent_ingest_attempts(limit=1)[0]
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        events = conn.execute(
            """
            SELECT attempt_id, stage, payload_json
            FROM daemon_stage_events
            ORDER BY observed_at_ms DESC, event_id DESC
            """
        ).fetchall()

    assert running.status == "running"
    assert running.phase == "full_parse"
    assert running.current_path == str(source)
    matching_events = [event for event in events if event[0:2] == (attempt_id, "full_parse")]
    assert matching_events
    assert any(str(source) in event[2] for event in matching_events)

    store.finish_ingest_attempt(attempt_id, status="completed", phase="completed")
    completed = store.recent_ingest_attempts(limit=1)[0]
    assert completed.status == "completed"
    assert completed.completed_at is not None

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        ops_attempt = conn.execute(
            """
            SELECT status, phase, source_path, parsed_raw_count, materialized_count
            FROM ingest_attempts
            WHERE attempt_id = ?
            """,
            (attempt_id,),
        ).fetchone()
        ops_events = conn.execute(
            """
            SELECT attempt_id, stage, status, payload_json
            FROM daemon_stage_events
            WHERE attempt_id = ?
            """,
            (attempt_id,),
        ).fetchall()

    assert ops_attempt == ("completed", "completed", str(source), 0, 0)
    full_parse_events = [event for event in ops_events if event[1] == "full_parse"]
    assert full_parse_events
    assert any(
        event[0] == attempt_id and event[2] == "running" and str(source) in event[3] for event in full_parse_events
    )


def test_cursor_archives_partial_success_as_completed_with_error(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')

    attempt_id = store.begin_ingest_attempt(
        paths=[source],
        input_bytes=source.stat().st_size,
        queued_file_count=1,
    )
    store.update_ingest_attempt(
        attempt_id,
        phase="completed",
        status="completed",
        succeeded_file_count=1,
        failed_file_count=1,
    )
    store.finish_ingest_attempt(
        attempt_id,
        status="completed_with_failures",
        phase="completed",
        error="/tmp/skipped-or-failed.jsonl",
    )

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row = conn.execute(
            """
            SELECT status, phase, parsed_raw_count, materialized_count, error_message
            FROM ingest_attempts
            WHERE attempt_id = ?
            """,
            (attempt_id,),
        ).fetchone()

    assert row == ("completed", "completed", 1, 1, "/tmp/skipped-or-failed.jsonl")


def test_cursor_syncs_positions_to_archive_ops_db(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')

    store.set(
        source,
        42,
        byte_offset=41,
        last_complete_newline=40,
        parser_fingerprint="parser-v1",
        content_fingerprint="content-v1",
        tail_hash="tail-v1",
        st_dev=1,
        st_ino=2,
        mtime_ns=3,
        failure_count=2,
        next_retry_at="2026-05-24T00:01:00+00:00",
        excluded=True,
    )

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row = conn.execute(
            """
            SELECT source_path, stat_size, byte_offset, last_complete_newline,
                   parser_fingerprint, content_fingerprint, tail_hash, st_dev, st_ino, mtime_ns,
                   failure_count, next_retry_at, excluded
            FROM ingest_cursor
            WHERE source_path = ?
            """,
            (str(source),),
        ).fetchone()

    assert row == (
        str(source),
        42,
        41,
        40,
        "parser-v1",
        "content-v1",
        "tail-v1",
        1,
        2,
        3,
        2,
        "2026-05-24T00:01:00+00:00",
        1,
    )


def test_cursor_syncs_convergence_debt_to_archive_ops_db(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")

    store.record_convergence_debt(
        stage="session_profile",
        subject_type="session",
        subject_id="conv-1",
        error="boom",
    )

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row = conn.execute(
            """
            SELECT stage, target_type, target_id, attempts, last_error
            FROM convergence_debt
            WHERE stage = 'session_profile'
            """,
        ).fetchone()
    assert row == ("session_profile", "session", "conv-1", 1, "boom")

    store.clear_convergence_debt(subject_type="session", subject_id="conv-1", stage="session_profile")

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        count = conn.execute("SELECT COUNT(*) FROM convergence_debt").fetchone()[0]
    assert count == 0


@pytest.mark.frozen_clock_modules("polylogue.sources.live.cursor", "polylogue.sources.live.convergence_debt_retry")
def test_cursor_records_messages_fts_surface_debt_as_immediately_due(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    store = CursorStore(tmp_path / "live.sqlite")

    store.record_convergence_debt(
        stage="insights",
        subject_type="session_id",
        subject_id="conv-1",
        error="profile stale",
    )
    frozen_clock.advance(1)
    store.record_convergence_debt(
        stage="fts",
        subject_type="fts_surface",
        subject_id="messages_fts",
        error="startup found stale messages_fts freshness ledger",
    )

    debt = store.list_convergence_debt(limit=2)
    assert [item.subject_id for item in debt] == ["messages_fts", "conv-1"]
    retry_at = datetime.fromisoformat(debt[0].next_retry_at or "")
    failed_at = datetime.fromisoformat(debt[0].last_failed_at)
    assert debt[0].stage == "fts"
    assert retry_at == failed_at

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        priority = conn.execute(
            """
            SELECT priority
            FROM convergence_debt
            WHERE stage = 'fts' AND target_type = 'fts_surface' AND target_id = 'messages_fts'
            """,
        ).fetchone()[0]
    assert priority == 100


def test_cursor_marks_running_attempts_abandoned_on_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "live.sqlite"
    store = CursorStore(db_path)
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')
    attempt_id = store.begin_ingest_attempt(
        paths=[source],
        input_bytes=source.stat().st_size,
        queued_file_count=1,
    )
    store.update_ingest_attempt(
        attempt_id,
        phase="full_parse",
        succeeded_file_count=0,
        failed_file_count=0,
    )

    restarted = CursorStore(db_path)
    attempt = restarted.recent_ingest_attempts(limit=1)[0]

    assert attempt.attempt_id == attempt_id
    assert attempt.status == "interrupted"
    assert attempt.phase == "interrupted"
    assert attempt.completed_at is not None
    assert attempt.error == "daemon stopped before completing this ingest attempt"
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row = conn.execute("SELECT status, phase FROM ingest_attempts WHERE attempt_id = ?", (attempt_id,)).fetchone()
    assert row == ("interrupted", "interrupted")


def test_cursor_mark_failed_creates_record_for_new_path(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    p = tmp_path / "new.jsonl"
    p.write_text('{"a":1}\n')

    store.mark_failed(p)

    record = store.get_record(p)
    assert record is not None
    assert record.byte_size == p.stat().st_size
    assert record.failure_count == 1
    assert record.next_retry_at is not None


def test_cursor_mark_failed_quarantines_repeated_failures(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    p = tmp_path / "poison.jsonl"
    p.write_text('{"a":1}\n')

    for _ in range(5):
        store.mark_failed(p)

    record = store.get_record(p)
    assert record is not None
    assert record.failure_count == 5
    assert record.next_retry_at is None
    assert record.excluded is True
    assert store.list_failed_with_retry() == []


def test_cursor_quarantine_binds_to_failed_replacement_observation(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    path = tmp_path / "capture.json"
    path.write_text('{"accepted":true}', encoding="utf-8")
    accepted = path.stat()
    store.set(
        path,
        accepted.st_size,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="accepted",
        st_dev=accepted.st_dev,
        st_ino=accepted.st_ino,
        mtime_ns=accepted.st_mtime_ns,
    )

    replacement = tmp_path / "replacement.json"
    replacement.write_text('{"malformed":', encoding="utf-8")
    replacement.replace(path)
    failed = path.stat()
    for _ in range(5):
        store.mark_failed(path, failed_stat=failed)

    record = store.get_record(path)
    assert record is not None
    assert record.excluded is True
    assert (record.byte_size, record.st_dev, record.st_ino, record.mtime_ns) == (
        failed.st_size,
        failed.st_dev,
        failed.st_ino,
        failed.st_mtime_ns,
    )


def test_cursor_round_trips_freshness_metadata(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    p = tmp_path / "session.jsonl"
    store.set(
        p,
        42,
        byte_offset=40,
        last_complete_newline=37,
        last_record_ts="2026-05-01T12:00:00+00:00",
        parser_fingerprint="parser-v1",
        content_fingerprint="abc123",
        source_name="codex",
    )

    record = store.get_record(p)

    assert record is not None
    assert record.byte_size == 42
    assert record.byte_offset == 40
    assert record.last_complete_newline == 37
    assert record.last_record_ts == "2026-05-01T12:00:00+00:00"
    assert record.parser_fingerprint == "parser-v1"
    assert record.content_fingerprint == "abc123"
    assert record.source_name == "codex"


def test_cursor_does_not_import_legacy_live_cursor_rows(tmp_path: Path) -> None:
    db = tmp_path / "live.sqlite"
    legacy_path = tmp_path / "legacy.jsonl"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE live_cursor (
                source_path TEXT PRIMARY KEY,
                byte_size INTEGER NOT NULL,
                record_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO live_cursor (source_path, byte_size, record_count, updated_at) VALUES (?, ?, ?, ?)",
            (str(legacy_path), 12, 2, "2026-05-01T00:00:00+00:00"),
        )
        conn.commit()

    store = CursorStore(db)
    record = store.get_record(legacy_path)

    assert record is None
    with sqlite3.connect(db) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(live_cursor)")}
    assert columns == {"source_path", "byte_size", "record_count", "updated_at"}
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row_count = conn.execute("SELECT COUNT(*) FROM ingest_cursor").fetchone()[0]
    assert row_count == 0


def test_live_full_ingest_caps_workers_below_batch_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polylogue.pipeline.services.ingest_batch._core.os.cpu_count", lambda: 16)
    monkeypatch.delenv("POLYLOGUE_LIVE_FULL_INGEST_WORKERS", raising=False)
    records = [
        RawSessionRecord(
            raw_id=f"raw-{index}",
            source_name="claude-code",
            source_path=f"/tmp/session-{index}.jsonl",
            blob_size=2 * 1024 * 1024,
            acquired_at="2026-05-01T00:00:00+00:00",
        )
        for index in range(300)
    ]
    giant = RawSessionRecord(
        raw_id="raw-giant",
        source_name="codex",
        source_path="/tmp/giant.jsonl",
        blob_size=600 * 1024 * 1024,
        acquired_at="2026-05-01T00:00:00+00:00",
    )

    assert _full_ingest_worker_count(records) == 1
    assert _full_ingest_worker_count([giant]) == 1


def test_live_full_ingest_worker_cap_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polylogue.pipeline.services.ingest_batch._core.os.cpu_count", lambda: 16)
    monkeypatch.setenv("POLYLOGUE_LIVE_FULL_INGEST_WORKERS", "4")
    records = [
        RawSessionRecord(
            raw_id=f"raw-{index}",
            source_name="claude-code",
            source_path=f"/tmp/session-{index}.jsonl",
            blob_size=2 * 1024 * 1024,
            acquired_at="2026-05-01T00:00:00+00:00",
        )
        for index in range(300)
    ]

    assert _full_ingest_worker_count(records) == 4


@pytest.mark.asyncio
async def test_live_full_ingest_streams_large_paths_before_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "projects"
    project = root / "project"
    project.mkdir(parents=True)
    source_path = project / "large-session.jsonl"
    records = [
        _claude_code_message(
            session_id="large-live-session",
            uuid=f"msg-{index}",
            role="user",
            text=f"message {index}",
            timestamp="2026-05-01T00:00:00Z",
        )
        for index in range(33)
    ]
    _write_jsonl(source_path, records)
    with source_path.open("r+b") as handle:
        handle.seek(_STREAMING_FULL_INGEST_BYTES + 128)
        handle.write(b"\n")

    db_path = tmp_path / "archive.sqlite"
    polylogue = MagicMock()
    polylogue.archive_root = tmp_path
    polylogue.backend.db_path = db_path
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        polylogue,
        (WatchSource(name="projects", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    calls: list[str] = []
    original_prepare_from_path = BlobStore.prepare_from_path

    def spy_prepare_from_path(
        store: BlobStore, path: Path, *, heartbeat: Callable[[], None] | None = None
    ) -> PreparedBlob:
        calls.append(f"path:{path.name}")
        return original_prepare_from_path(store, path, heartbeat=heartbeat)

    def fail_prepare_from_bytes(_store: object, _payload: bytes) -> PreparedBlob:
        raise AssertionError("large live full ingest should stream from path")

    monkeypatch.setattr("polylogue.sources.live.batch.BlobStore.prepare_from_path", spy_prepare_from_path)
    monkeypatch.setattr("polylogue.sources.live.batch.BlobStore.prepare_from_bytes", fail_prepare_from_bytes)

    result = await processor._ingest_full_paths([source_path], source_name="projects")

    assert result.succeeded == [source_path]
    assert result.failed == []
    assert calls == ["path:large-session.jsonl"]


# --- LiveWatcher: needs_work + ingest_files (batched) --------------------------


def _make_watcher(
    tmp_path: Path,
    root: Path,
    *,
    debounce_s: float = 0.01,
    event_emitter: MagicMock | None = None,
    sources: tuple[WatchSource, ...] | None = None,
) -> tuple[LiveWatcher, _FullIngestMock]:
    polylogue = MagicMock()
    polylogue.archive_root = tmp_path
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    sources = sources or (WatchSource(name="test", root=root),)
    watcher = LiveWatcher(polylogue, sources, debounce_s=debounce_s, cursor=cursor, event_emitter=event_emitter)
    full_ingest = _FullIngestMock()
    watcher._batch_processor._ingest_full_paths = full_ingest  # type: ignore[method-assign]
    return watcher, full_ingest


def test_watcher_default_cursor_uses_archive_database(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    db_path = tmp_path / "ops.db"
    polylogue = cast(
        Any,
        SimpleNamespace(
            archive_root=tmp_path,
            backend=SimpleNamespace(db_path=db_path),
        ),
    )

    watcher = LiveWatcher(polylogue, (WatchSource(name="test", root=root),))

    assert watcher._cursor._db_path == db_path


def test_catch_up_uses_bulk_cursor_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "src"
    root.mkdir()
    files = [root / f"session-{index}.jsonl" for index in range(3)]
    for path in files:
        path.write_text('{"role":"user","content":"a"}\n')
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    bulk_calls = 0
    original_get_records = watcher._cursor.get_records

    def counted_get_records(paths: Iterable[Path]) -> dict[Path, CursorRecord]:
        nonlocal bulk_calls
        bulk_calls += 1
        return original_get_records(paths)

    def fail_get_record(path: Path) -> CursorRecord | None:
        raise AssertionError(f"catch-up should use bulk cursor reads, not per-file reads: {path}")

    captured: dict[str, object] = {}

    async def fake_ingest_files(
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
    ) -> None:
        captured["paths"] = paths
        captured["queued_file_count"] = queued_file_count
        captured["skipped_file_count"] = skipped_file_count

    monkeypatch.setattr(watcher._cursor, "get_records", counted_get_records)
    monkeypatch.setattr(watcher._cursor, "get_record", fail_get_record)
    watcher._ingest_files = fake_ingest_files  # type: ignore[assignment,method-assign]

    asyncio.run(watcher._catch_up([root]))

    assert bulk_calls == 1
    assert captured["paths"] == files
    assert captured["queued_file_count"] == 3
    assert captured["skipped_file_count"] == 0


async def _ingest_one(watcher: LiveWatcher, path: Path) -> None:
    """Helper: check and ingest a single file (mimics old _ingest_if_grown)."""
    if watcher._needs_work(path):
        await watcher._ingest_files([path])


def test_skip_when_file_not_grown(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"role":"user","content":"a"}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)

    asyncio.run(_ingest_one(watcher, f))
    assert parse_sources.await_count == 1

    asyncio.run(_ingest_one(watcher, f))
    assert parse_sources.await_count == 1  # cursor matches size


def test_reingest_when_file_grows(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"role":"user","content":"a"}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)

    asyncio.run(_ingest_one(watcher, f))
    f.write_text('{"a":1}\n{"b":2}\n{"c":3}\n')
    asyncio.run(_ingest_one(watcher, f))
    f.write_text('{"a":1}\n{"b":2}\n{"c":3}\n{"d":4}\n')
    asyncio.run(_ingest_one(watcher, f))

    assert parse_sources.await_count == 3


def test_size_only_cursor_reingests_to_populate_fingerprint(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)
    watcher._cursor.set(f, f.stat().st_size)

    asyncio.run(_ingest_one(watcher, f))
    asyncio.run(_ingest_one(watcher, f))

    assert parse_sources.await_count == 1
    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.content_fingerprint
    assert record.parser_fingerprint


def test_unchanged_file_uses_stat_fast_path_without_fingerprint_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    stat = f.stat()
    watcher._cursor.set(
        f,
        stat.st_size,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="already-known",
        tail_hash=encode_cursor_hash_authority(
            sha256(f.read_bytes()).hexdigest(),
            sha256(f.read_bytes()).hexdigest(),
            ctime_ns=stat.st_ctime_ns,
        ),
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )

    def fail_fingerprint(path: Path) -> tuple[str, int]:
        raise AssertionError(f"unchanged file should not be fingerprinted: {path}")

    def fail_tail_hash(path: Path, size: int) -> tuple[str, int]:
        raise AssertionError(f"stat-stable unchanged file should not read tail hash: {path} ({size})")

    monkeypatch.setattr(live_watcher, "fingerprint_file", fail_fingerprint)
    monkeypatch.setattr(live_watcher, "tail_hash_from_path", fail_tail_hash)

    assert watcher._needs_work(f) is False

    watcher._cursor.set(
        f,
        stat.st_size,
        byte_offset=stat.st_size,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="already-known",
        tail_hash=encode_cursor_hash_authority(
            sha256(f.read_bytes()).hexdigest(),
            sha256(f.read_bytes()).hexdigest(),
            ctime_ns=stat.st_ctime_ns,
        ),
        st_dev=stat.st_dev + 1,
        st_ino=stat.st_ino + 1,
        mtime_ns=stat.st_mtime_ns,
    )
    assert watcher._needs_work(f) is False


def test_new_file_needs_work_without_prefingerprint_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, _parse_sources = _make_watcher(tmp_path, root)

    def fail_fingerprint(path: Path) -> tuple[str, int]:
        raise AssertionError(f"new file should not be fingerprinted before ingest: {path}")

    monkeypatch.setattr(live_watcher, "fingerprint_file", fail_fingerprint)

    assert watcher._needs_work(f) is True


def test_parser_version_change_needs_work_without_prefingerprint_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    stat = f.stat()
    watcher._cursor.set(
        f,
        stat.st_size,
        parser_fingerprint="older-parser",
        content_fingerprint="already-known",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )

    def fail_fingerprint(path: Path) -> tuple[str, int]:
        raise AssertionError(f"parser changes should not prefingerprint before ingest: {path}")

    monkeypatch.setattr(live_watcher, "fingerprint_file", fail_fingerprint)

    assert watcher._needs_work(f) is True


def test_replaced_excluded_file_is_revived_without_retrying_unchanged_poison(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    path = root / "capture.json"
    path.write_text('{"broken":true}', encoding="utf-8")
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    stat = path.stat()
    watcher._cursor.set(
        path,
        stat.st_size,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="broken",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        failure_count=5,
        excluded=True,
    )

    assert watcher._needs_work(path) is False

    replacement = root / "replacement.json"
    replacement.write_text('{"valid":"new capture"}', encoding="utf-8")
    replacement.replace(path)

    assert watcher._needs_work(path) is True
    revived = watcher._cursor.get_record(path)
    assert revived is not None
    assert revived.excluded is False
    assert revived.failure_count == 0
    assert revived.next_retry_at is None


def test_full_cursor_uses_batch_raw_fingerprint_without_db_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, _parse_sources = _make_watcher(tmp_path, root)

    def fail_latest_raw_fingerprint(path: Path) -> str | None:
        raise AssertionError(f"raw fingerprint should already be carried by the batch result: {path}")

    monkeypatch.setattr(watcher._batch_processor, "_latest_raw_fingerprint", fail_latest_raw_fingerprint)

    bytes_read = watcher._batch_processor._record_full_cursor(f, raw_fingerprint="raw-sha256")
    record = watcher._cursor.get_record(f)

    # The cursor stores both a bounded tail and a complete accepted-prefix
    # hash; this one-record fixture makes both reads span the whole file.
    assert bytes_read == 2 * f.stat().st_size
    assert record is not None
    assert record.content_fingerprint == "raw-sha256"


def test_hermes_wal_revision_triggers_resnapshot_and_maps_sidecar_event(tmp_path: Path) -> None:
    root = tmp_path / "hermes"
    root.mkdir()
    state_db = root / "state.db"
    writer = sqlite3.connect(state_db)
    try:
        writer.execute("CREATE TABLE turns(id INTEGER PRIMARY KEY, text TEXT)")
        writer.commit()
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        watcher, _full_ingest = _make_watcher(
            tmp_path,
            root,
            sources=(WatchSource(name="hermes", root=root, suffixes=(".db",)),),
        )
        initial_revision = sqlite_source_revision(state_db)
        stat = state_db.stat()
        watcher._cursor.set(
            state_db,
            stat.st_size,
            parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
            content_fingerprint="snapshot-hash",
            tail_hash=initial_revision,
            st_dev=stat.st_dev,
            st_ino=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
        )
        assert watcher._needs_work(state_db) is False

        writer.execute("INSERT INTO turns(text) VALUES ('WAL-only turn')")
        writer.commit()
        wal_path = state_db.with_name("state.db-wal")

        assert wal_path.stat().st_size > 0
        assert watcher._watch_filter(object(), str(wal_path)) is True
        assert watcher._canonical_watch_path(wal_path) == state_db
        assert watcher._needs_work(state_db) is True
    finally:
        writer.close()


def test_hermes_cursor_records_acquisition_revision_not_live_tail(tmp_path: Path) -> None:
    root = tmp_path / "hermes"
    root.mkdir()
    state_db = root / "state.db"
    with sqlite3.connect(state_db) as conn:
        conn.execute("CREATE TABLE turns(id INTEGER PRIMARY KEY)")
    watcher, _full_ingest = _make_watcher(
        tmp_path,
        root,
        sources=(WatchSource(name="hermes", root=root, suffixes=(".db",)),),
    )

    bytes_read = watcher._batch_processor._record_full_cursor(
        state_db,
        raw_fingerprint="snapshot-hash",
        raw_byte_size=999_999,
        source_revision="acquisition-revision",
    )
    record = watcher._cursor.get_record(state_db)

    assert bytes_read == 0
    assert record is not None
    assert record.byte_size == state_db.stat().st_size
    assert record.content_fingerprint == "acquisition-revision"
    assert record.tail_hash == "acquisition-revision"


def test_append_plan_reads_only_completed_tail(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    original = b'{"a":1}\n'
    appended = b'{"b":2}\n{"c":'
    f.write_bytes(original + appended)
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    stat = f.stat()
    watcher._cursor.set(
        f,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="base",
        tail_hash=encode_cursor_hash_authority(
            sha256(original).hexdigest(),
            sha256(original).hexdigest(),
            ctime_ns=stat.st_ctime_ns,
        ),
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )

    plan = watcher._batch_processor._append_plan(f)

    assert plan is not None
    append_plan = cast(Any, plan)
    assert append_plan.start_offset == len(original)
    assert append_plan.payload == b'{"b":2}\n'
    assert append_plan.bytes_read == len(appended)
    assert append_plan.last_complete_newline == len(original) + len(b'{"b":2}\n')


def test_large_incomplete_jsonl_append_defers_until_the_file_changes(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    original = b'{"a":1}\n'
    f.write_bytes(original)
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    stat = f.stat()
    watcher._cursor.set(
        f,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="base",
        tail_hash=encode_cursor_hash_authority(
            sha256(original).hexdigest(),
            sha256(original).hexdigest(),
            ctime_ns=stat.st_ctime_ns,
        ),
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )
    f.write_bytes(original + (b"x" * (live_watcher._INCOMPLETE_APPEND_PROBE_BYTES + 1)))

    # The first bounded probe observes no newline and records the unfinished
    # tail.  Repeating the same periodic scan must then be a stat-only skip.
    assert watcher._needs_work(f) is False
    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.byte_size == f.stat().st_size
    assert record.byte_offset == len(original)
    assert watcher._needs_work(f) is False


def test_last_complete_newline_from_tail_reads_only_final_chunk(tmp_path: Path) -> None:
    path = tmp_path / "large.jsonl"
    complete_prefix = b'{"a":"' + (b"x" * 200_000) + b'"}\n'
    path.write_bytes(complete_prefix + b'{"b":2}')

    offset, bytes_read = last_complete_newline_from_tail(path, path.stat().st_size)

    assert offset == len(complete_prefix)
    assert bytes_read < path.stat().st_size


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _claude_code_message(
    *,
    session_id: str,
    uuid: str,
    role: str,
    text: str,
    timestamp: str,
    parent_uuid: str | None = None,
) -> dict[str, object]:
    return {
        "type": role,
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "sessionId": session_id,
        "timestamp": timestamp,
        "message": {"role": role, "content": text if role == "user" else [{"type": "text", "text": text}]},
    }


def _codex_session_meta(session_id: str) -> dict[str, object]:
    return {"type": "session_meta", "payload": {"id": session_id, "timestamp": "2026-05-01T00:00:00Z"}}


def _codex_message(*, message_id: str, role: str, text: str, timestamp: str) -> dict[str, object]:
    block_type = "input_text" if role == "user" else "output_text"
    return {
        "type": "response_item",
        "payload": {
            "id": message_id,
            "role": role,
            "type": "message",
            "timestamp": timestamp,
            "content": [{"type": block_type, "text": text}],
        },
    }


@pytest.mark.asyncio
async def test_live_batch_processor_records_durable_attempt(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source_path = root / "session.jsonl"
    source_path.write_text('{"type":"session_meta","payload":{"id":"s"}}\n', encoding="utf-8")
    db_path = tmp_path / "live.sqlite"
    cursor = CursorStore(db_path)
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    metrics = await processor.ingest_files([source_path], emit_event=False)
    attempts = cursor.recent_ingest_attempts(limit=1)

    assert metrics.succeeded_file_count == 1
    assert len(attempts) == 1
    assert attempts[0].status == "completed"
    assert attempts[0].phase == "completed"
    assert attempts[0].needed_file_count == 1
    assert attempts[0].succeeded_file_count == 1
    assert attempts[0].source_payload_read_bytes == source_path.stat().st_size
    assert {
        "full.provider_parse",
        "full.source_raw_write",
        "full.index_parsed_write",
    }.issubset(metrics.stage_timings_s)


@pytest.mark.asyncio
async def test_live_batch_processor_records_cursor_after_each_converged_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    first_path = root / "first.jsonl"
    second_path = root / "second.jsonl"
    _write_jsonl(
        first_path,
        [
            _codex_session_meta("first-session"),
            _codex_message(
                message_id="first-message",
                role="user",
                text="first converged group",
                timestamp="2026-05-01T00:00:01Z",
            ),
        ],
    )
    _write_jsonl(
        second_path,
        [
            _codex_session_meta("second-session"),
            _codex_message(
                message_id="second-message",
                role="user",
                text="second unconverged group",
                timestamp="2026-05-01T00:00:01Z",
            ),
        ],
    )
    db_path = tmp_path / "live.sqlite"
    cursor = CursorStore(db_path)
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
        converger=_FailingSecondPathConverger(second_path),
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch._full_parse_progress_groups",
        lambda paths: ([path] for path in paths),
    )

    metrics = await processor.ingest_files([first_path, second_path], emit_event=False)

    first_cursor = cursor.get_record(first_path)
    second_cursor = cursor.get_record(second_path)
    assert metrics.succeeded_file_count == 2
    assert metrics.failed_file_count == 0
    assert first_cursor is not None
    assert first_cursor.parser_fingerprint == "test-parser"
    assert second_cursor is not None
    assert second_cursor.parser_fingerprint == "test-parser"
    debt = cursor.list_convergence_debt()
    assert len(debt) == 1
    assert debt[0].stage == "convergence"
    assert debt[0].subject_type == "session_id"
    assert "second-session" in debt[0].subject_id


def test_full_parse_progress_groups_bounds_small_files_by_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [tmp_path / f"{index}.jsonl" for index in range(_SMALL_FULL_PARSE_PROGRESS_MAX_FILES + 1)]
    monkeypatch.setattr("polylogue.sources.live.batch_support._path_size", lambda path: 1)

    groups = list(_full_parse_progress_groups(paths))

    assert groups == [paths[:_SMALL_FULL_PARSE_PROGRESS_MAX_FILES], paths[_SMALL_FULL_PARSE_PROGRESS_MAX_FILES:]]


def test_full_parse_progress_groups_bounds_small_files_by_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [tmp_path / f"{index}.jsonl" for index in range(5)]
    byte_size = (_SMALL_FULL_PARSE_PROGRESS_MAX_BYTES // 3) + 1
    monkeypatch.setattr("polylogue.sources.live.batch_support._path_size", lambda path: byte_size)

    groups = list(_full_parse_progress_groups(paths))

    assert sum(byte_size for _ in groups[0]) <= _SMALL_FULL_PARSE_PROGRESS_MAX_BYTES
    assert groups == [paths[:2], paths[2:4], paths[4:]]


@pytest.mark.asyncio
async def test_live_full_ingest_offloads_sync_work_to_keep_loop_responsive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source_path = root / "session.jsonl"
    source_path.write_text('{"type":"session_meta","payload":{"id":"responsive"}}\n', encoding="utf-8")
    cursor = CursorStore(tmp_path / "live.sqlite")
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    def slow_full_ingest(
        paths: list[Path], *, source_name: str, heartbeat: object = None, attempt_id: str | None = None
    ) -> _FullIngestResult:
        del source_name, heartbeat, attempt_id
        time.sleep(0.2)
        return _FullIngestResult(
            succeeded=list(paths),
            failed=[],
            source_payload_read_bytes=sum(path.stat().st_size for path in paths),
            raw_fingerprints={path: f"raw:{path.name}" for path in paths},
        )

    monkeypatch.setattr(processor, "_ingest_full_paths_sync", slow_full_ingest)

    ingest_task = asyncio.create_task(processor.ingest_files([source_path], emit_event=False))
    await asyncio.sleep(0.02)

    assert not ingest_task.done()
    metrics = await ingest_task
    assert metrics.succeeded_file_count == 1


@pytest.mark.asyncio
async def test_live_append_merges_tail_visible_through_public_archive_read(workspace_env: dict[str, Path]) -> None:
    root = workspace_env["data_root"] / "claude-projects"
    project = root / "project"
    project.mkdir(parents=True)
    source_path = project / "session.jsonl"
    db_path = workspace_env["data_root"] / "append-public-read.db"
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="claude-code", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    try:
        _write_jsonl(
            source_path,
            [
                _claude_code_message(
                    session_id="session-public-read",
                    uuid="msg-1",
                    role="user",
                    text="first live message",
                    timestamp="2026-05-01T00:00:00Z",
                ),
                _claude_code_message(
                    session_id="session-public-read",
                    uuid="msg-2",
                    parent_uuid="msg-1",
                    role="assistant",
                    text="second live reply",
                    timestamp="2026-05-01T00:00:01Z",
                ),
                _claude_code_message(
                    session_id="session-public-read",
                    uuid="msg-3",
                    parent_uuid="msg-2",
                    role="user",
                    text="third live followup",
                    timestamp="2026-05-01T00:00:02Z",
                ),
            ],
        )
        initial_metrics = await processor.ingest_files([source_path], emit_event=False)

        with source_path.open("a", encoding="utf-8") as handle:
            for record in (
                _claude_code_message(
                    session_id="session-public-read",
                    uuid="msg-4",
                    parent_uuid="msg-3",
                    role="assistant",
                    text="fourth appended reply",
                    timestamp="2026-05-01T00:00:03Z",
                ),
                _claude_code_message(
                    session_id="session-public-read",
                    uuid="msg-5",
                    parent_uuid="msg-4",
                    role="user",
                    text="fifth appended followup",
                    timestamp="2026-05-01T00:00:04Z",
                ),
            ):
                handle.write(json.dumps(record) + "\n")
        append_metrics = await processor.ingest_files([source_path], emit_event=False)

        session = await archive.get_session("claude-code:session-public-read")
        assert initial_metrics.full_file_count == 1
        assert append_metrics.append_file_count == 1
        assert append_metrics.full_file_count == 0
        assert append_metrics.source_payload_read_bytes < append_metrics.input_bytes
        assert session is not None
        assert [message.text for message in session.messages] == [
            "first live message",
            "second live reply",
            "third live followup",
            "fourth appended reply",
            "fifth appended followup",
        ]
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_live_full_ingest_expands_inbox_zip_members(
    workspace_env: dict[str, Path],
) -> None:
    """A ZIP dropped into a watched root ingests every session member (#1683).

    Regression: ZIP paths previously fell through to the byte-level detection
    branch, where ``orjson.loads`` over the ZIP container raised and the whole
    archive was silently marked excluded. The fix expands ZIP members through
    the maintenance acquisition path, so each member becomes a session.
    """
    root = workspace_env["data_root"] / "claude-projects"
    root.mkdir(parents=True)
    member_a = "\n".join(
        json.dumps(record)
        for record in (
            _claude_code_message(
                session_id="zip-session-a",
                uuid="a-1",
                role="user",
                text="first zip member message",
                timestamp="2026-05-01T00:00:00Z",
            ),
        )
    )
    member_b = "\n".join(
        json.dumps(record)
        for record in (
            _claude_code_message(
                session_id="zip-session-b",
                uuid="b-1",
                role="user",
                text="second zip member message",
                timestamp="2026-05-01T00:00:01Z",
            ),
        )
    )
    zip_path = root / "claude-export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("session-a.jsonl", member_a + "\n")
        zf.writestr("session-b.jsonl", member_b + "\n")

    db_path = workspace_env["data_root"] / "inbox-zip-live.db"
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="claude-code", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    try:
        metrics = await processor.ingest_files([zip_path], emit_event=False)
        record = cursor.get_record(zip_path)

        session_a = await archive.get_session("claude-code:zip-session-a")
        session_b = await archive.get_session("claude-code:zip-session-b")

        assert metrics.succeeded_file_count == 1
        assert metrics.failed_file_count == 0
        assert metrics.source_payload_read_bytes > 0
        assert record is not None
        assert record.excluded is False
        assert session_a is not None
        assert session_b is not None
        assert [message.text for message in session_a.messages] == ["first zip member message"]
        assert [message.text for message in session_b.messages] == ["second zip member message"]
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_live_full_ingest_detects_provider_when_source_name_is_not_provider(
    workspace_env: dict[str, Path],
) -> None:
    root = workspace_env["data_root"] / "projects"
    project = root / "project"
    project.mkdir(parents=True)
    source_path = project / "session.jsonl"
    db_path = workspace_env["data_root"] / "detect-provider-live.db"
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="projects", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    try:
        _write_jsonl(
            source_path,
            [
                _claude_code_message(
                    session_id="session-detected-provider",
                    uuid="msg-1",
                    role="user",
                    text="detected provider message",
                    timestamp="2026-05-01T00:00:00Z",
                ),
            ],
        )
        metrics = await processor.ingest_files([source_path], emit_event=False)

        session = await archive.get_session("claude-code:session-detected-provider")
        assert metrics.succeeded_file_count == 1
        assert metrics.failed_file_count == 0
        assert session is not None
        assert [message.text for message in session.messages] == ["detected provider message"]
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_live_full_ingest_excludes_non_session_sidecars_before_raw_storage(
    workspace_env: dict[str, Path],
) -> None:
    root = workspace_env["data_root"] / "projects"
    project = root / "project"
    project.mkdir(parents=True)
    source_path = project / "sessions-index.json"
    source_path.write_text(json.dumps({"sessions": [{"id": "metadata-only"}]}), encoding="utf-8")
    db_path = workspace_env["data_root"] / "exclude-sidecar-live.db"
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="projects", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    try:
        metrics = await processor.ingest_files([source_path], emit_event=False)
        record = cursor.get_record(source_path)

        with sqlite3.connect(db_path) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            raw_count = (
                conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] if "raw_sessions" in tables else 0
            )
            session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] if "sessions" in tables else 0

        assert metrics.succeeded_file_count == 0
        assert metrics.failed_file_count == 0
        assert metrics.source_payload_read_bytes == 0
        assert raw_count == 0
        assert session_count == 0
        assert record is not None
        assert record.excluded is True
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_live_full_ingest_excludes_invalid_jsonl_sidecars_before_raw_storage(
    workspace_env: dict[str, Path],
) -> None:
    root = workspace_env["data_root"] / "projects"
    project = root / "project" / "analysis"
    project.mkdir(parents=True)
    source_path = project / "architecture_discussions.jsonl"
    source_path.write_text("not json\nstill not json\n", encoding="utf-8")
    db_path = workspace_env["data_root"] / "exclude-invalid-jsonl-live.db"
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="projects", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    try:
        metrics = await processor.ingest_files([source_path], emit_event=False)
        record = cursor.get_record(source_path)

        with sqlite3.connect(db_path) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            raw_count = (
                conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] if "raw_sessions" in tables else 0
            )

        assert metrics.succeeded_file_count == 0
        assert metrics.failed_file_count == 0
        assert metrics.source_payload_read_bytes == 0
        assert raw_count == 0
        assert record is not None
        assert record.excluded is True
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_codex_append_uses_existing_session_identity_when_tail_lacks_session_meta(
    workspace_env: dict[str, Path],
) -> None:
    root = workspace_env["data_root"] / "codex-sessions"
    project = root / "project"
    project.mkdir(parents=True)
    source_path = project / "codex-session.jsonl"
    db_path = workspace_env["data_root"] / "append-codex.db"
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    try:
        _write_jsonl(
            source_path,
            [
                _codex_session_meta("codex-real-session"),
                _codex_message(
                    message_id="msg-1",
                    role="user",
                    text="codex first",
                    timestamp="2026-05-01T00:00:00Z",
                ),
            ],
        )
        await processor.ingest_files([source_path], emit_event=False)

        with source_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    _codex_message(
                        message_id="msg-2",
                        role="assistant",
                        text="codex appended",
                        timestamp="2026-05-01T00:00:01Z",
                    )
                )
                + "\n"
            )
        append_metrics = await processor.ingest_files([source_path], emit_event=False)

        existing = await archive.get_session("codex:codex-real-session")
        fallback = await archive.get_session("codex:codex-session")
        assert append_metrics.append_file_count == 1
        assert append_metrics.full_file_count == 0
        assert {
            "append.archive_open",
            "append.index.blocks",
            "append.index_parsed_write",
            "append.index.messages",
            "append.index.session_events",
            "append.index.session_upsert",
            "append.json_stream",
            "append.provider_parse",
            "append.raw_and_index_write",
            "append.source_raw_write",
        }.issubset(append_metrics.stage_timings_s)
        assert existing is not None
        assert [message.text for message in existing.messages] == ["codex first", "codex appended"]
        assert fallback is None
    finally:
        await archive.close()


def test_parser_fingerprint_change_triggers_reingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)

    asyncio.run(_ingest_one(watcher, f))
    monkeypatch.setattr(live_watcher, "_PARSER_FINGERPRINT", "live-batched-v3")
    asyncio.run(_ingest_one(watcher, f))

    assert parse_sources.await_count == 2
    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.parser_fingerprint == "live-batched-v3"


def test_truncate_rewrite_triggers_reingest(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n{"b":2}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)

    asyncio.run(_ingest_one(watcher, f))
    f.write_text('{"c":3}\n')
    asyncio.run(_ingest_one(watcher, f))

    assert parse_sources.await_count == 2


def test_partial_trailing_line_keeps_cursor_at_last_newline(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    complete = b'{"a":1}\n'
    partial = b'{"b":'
    f.write_bytes(complete + partial)
    watcher, parse_sources = _make_watcher(tmp_path, root)

    asyncio.run(_ingest_one(watcher, f))
    asyncio.run(_ingest_one(watcher, f))

    record = watcher._cursor.get_record(f)
    assert parse_sources.await_count == 1
    assert record is not None
    assert record.byte_size == len(complete + partial)
    assert record.byte_offset == len(complete)
    assert record.last_complete_newline == len(complete)


def test_incomplete_append_event_defers_without_ingest_until_newline(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    complete = b'{"a":1}\n'
    partial = b'{"b":'
    f.write_bytes(complete)
    watcher, parse_sources = _make_watcher(tmp_path, root)
    stat = f.stat()
    watcher._cursor.set(
        f,
        len(complete),
        byte_offset=len(complete),
        last_complete_newline=len(complete),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="base",
        tail_hash=encode_cursor_hash_authority(
            sha256(complete).hexdigest(),
            sha256(complete).hexdigest(),
            ctime_ns=stat.st_ctime_ns,
        ),
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )

    f.write_bytes(complete + partial)
    asyncio.run(_ingest_one(watcher, f))
    asyncio.run(_ingest_one(watcher, f))

    record = watcher._cursor.get_record(f)
    assert parse_sources.await_count == 0
    assert record is not None
    assert record.byte_size == len(complete + partial)
    assert record.byte_offset == len(complete)

    f.write_bytes(complete + b'{"b":2}\n')
    asyncio.run(_ingest_one(watcher, f))

    assert parse_sources.await_count == 1


def test_append_after_partial_line_reingests_completed_record(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n{"b":')
    watcher, parse_sources = _make_watcher(tmp_path, root)

    asyncio.run(_ingest_one(watcher, f))
    f.write_text('{"a":1}\n{"b":2}\n')
    asyncio.run(_ingest_one(watcher, f))

    assert parse_sources.await_count == 2
    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.byte_offset == f.stat().st_size


def test_year_old_file_resumed_triggers_reingest(tmp_path: Path) -> None:
    """A year-old session that gets new lines must be picked up — there
    is no 'live session' concept."""
    root = tmp_path / "src"
    root.mkdir()
    old = root / "old-session.jsonl"
    old.write_text('{"role":"user","content":"original"}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)
    asyncio.run(_ingest_one(watcher, old))
    parse_sources.reset_mock()

    old.write_text('{"role":"user","content":"original"}\n{"role":"user","content":"resumed"}\n')
    asyncio.run(_ingest_one(watcher, old))
    assert parse_sources.await_count == 1


def test_missing_file_is_silent(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    watcher, parse_sources = _make_watcher(tmp_path, root)
    assert not watcher._needs_work(root / "ghost.jsonl")
    assert parse_sources.await_count == 0


def test_parse_failure_is_recorded_and_backed_off(tmp_path: Path) -> None:
    """After a batch failure, cursor state records failure and backs off."""
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"role":"user","content":"a"}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)
    parse_sources.side_effect = RuntimeError("parser sad")

    # First attempt: fails, cursor is set
    asyncio.run(_ingest_one(watcher, f))
    assert parse_sources.await_count == 1
    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.failure_count == 1
    assert record.next_retry_at is not None

    # Second immediate attempt: file is skipped during backoff.
    parse_sources.reset_mock()
    asyncio.run(_ingest_one(watcher, f))
    assert parse_sources.await_count == 0


def test_ingest_files_emits_observable_batch_metrics(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"role":"user","content":"a"}\n')
    emit = MagicMock()
    watcher, _parse_sources = _make_watcher(tmp_path, root, event_emitter=emit)

    asyncio.run(watcher._ingest_files([f], queued_file_count=3, skipped_file_count=2))

    emit.assert_called_once()
    kind = emit.call_args.args[0]
    payload = emit.call_args.args[1]
    assert kind == "ingestion_batch"
    assert payload["queued_file_count"] == 3
    assert payload["needed_file_count"] == 1
    assert payload["skipped_file_count"] == 2
    assert payload["succeeded_file_count"] == 1
    assert payload["failed_file_count"] == 0
    assert payload["source_group_count"] == 1
    assert payload["input_bytes"] == f.stat().st_size
    assert payload["source_payload_read_bytes"] == f.stat().st_size
    assert payload["cursor_fingerprint_read_bytes"] == 2 * f.stat().st_size
    assert payload["read_amplification"] == 1.0
    assert payload["files_per_second"] >= 0
    assert payload["source_mb_per_second"] >= 0
    assert payload["append_file_count"] == 0
    assert payload["full_file_count"] == 1
    assert payload["archive_write_bytes_delta"] >= 0
    assert payload["ingested_session_count"] == 1
    assert payload["ingested_message_count"] == 7
    assert payload["changed_session_count"] == 1
    assert payload["wal_bytes_before_checkpoint_max"] == 8192
    assert payload["wal_bytes_after_checkpoint_max"] == 1024
    assert payload["wal_checkpointed_pages_total"] == 4
    assert payload["wal_busy_pages_total"] == 2
    assert payload["wal_checkpoint_elapsed_s"] == 0.125
    assert payload["wal_checkpoint_modes"] == {"truncate": 1}
    assert payload["wal_checkpoint_errors"] == []
    assert payload["parse_time_s"] >= 0
    assert payload["total_time_s"] >= 0
    assert payload["stage_timings_s"] == {"full.index_parsed_write": 0.02, "full.provider_parse": 0.01}
    assert payload["failed_paths"] == []
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        events = conn.execute(
            "SELECT stage FROM daemon_stage_events ORDER BY observed_at_ms DESC, event_id DESC LIMIT 10"
        ).fetchall()
    assert events[0] == ("completed",)
    assert ("planning",) in events


@pytest.mark.frozen_clock_modules("polylogue.sources.live.watcher", "polylogue.sources.live.cursor")
def test_parse_failure_retries_after_backoff(tmp_path: Path, frozen_clock: FrozenClock) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"role":"user","content":"a"}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)
    parse_sources.side_effect = RuntimeError("parser sad")

    asyncio.run(_ingest_one(watcher, f))
    parse_sources.reset_mock()
    parse_sources.side_effect = None
    past = (frozen_clock.now() - timedelta(seconds=1)).isoformat()
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute("UPDATE ingest_cursor SET next_retry_at = ? WHERE source_path = ?", (past, str(f)))
        conn.commit()

    asyncio.run(_ingest_one(watcher, f))

    assert parse_sources.await_count == 1
    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.failure_count == 0
    assert record.next_retry_at is None


# --- catch_up bootstrap --------------------------------------------------------


def test_catch_up_processes_pre_existing_files(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    # Session files at {project}/{uuid}.jsonl
    proj = root / "my-project"
    proj.mkdir()
    files = [proj / f"s{i}.jsonl" for i in range(3)]
    for f in files:
        f.write_text('{"a":1}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)

    asyncio.run(watcher._catch_up([root]))
    assert parse_sources.await_count == 1


def test_catch_up_skips_already_processed(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    proj = root / "my-project"
    proj.mkdir()
    f = proj / "s.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)
    asyncio.run(_ingest_one(watcher, f))
    parse_sources.reset_mock()

    asyncio.run(watcher._catch_up([root]))
    assert parse_sources.await_count == 0


def test_catch_up_finds_subagent_files(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    proj = root / "my-project"
    session_dir = proj / "some-uuid"
    subagents = session_dir / "subagents"
    subagents.mkdir(parents=True)
    f = subagents / "agent-abc123.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)

    asyncio.run(watcher._catch_up([root]))
    assert parse_sources.await_count == 1


def test_catch_up_ignores_non_jsonl(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    proj = root / "my-project"
    proj.mkdir()
    (proj / "session.jsonl").write_text('{"a":1}\n')
    (proj / "config.toml").write_text("x=1")
    (proj / "README.md").write_text("# hi")
    watcher, parse_sources = _make_watcher(tmp_path, root)

    asyncio.run(watcher._catch_up([root]))
    assert parse_sources.await_count == 1


def test_catch_up_uses_source_suffix_contract_for_json_sessions(tmp_path: Path) -> None:
    root = tmp_path / "gemini"
    root.mkdir()
    (root / "session.json").write_text('{"sessionId":"s1","messages":[]}\n')
    (root / "notes.md").write_text("# no")
    watcher, parse_sources = _make_watcher(
        tmp_path,
        root,
        sources=(WatchSource(name="gemini-cli", root=root, suffixes=(".json", ".jsonl")),),
    )

    asyncio.run(watcher._catch_up([root]))

    assert parse_sources.await_count == 1


def test_catch_up_recurses_like_live_watch(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    (root / "orphan.jsonl").write_text('{"a":1}\n')
    deep = root / "p" / "u" / "extra" / "deep.jsonl"
    deep.parent.mkdir(parents=True)
    deep.write_text('{"a":1}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)

    asyncio.run(watcher._catch_up([root]))
    assert parse_sources.await_count == 1


def test_catch_up_handles_empty_roots(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    watcher, parse_sources = _make_watcher(tmp_path, root)
    asyncio.run(watcher._catch_up([root]))
    assert parse_sources.await_count == 0


# --- debounce ------------------------------------------------------------------


def test_debounce_coalesces_rapid_changes(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root, debounce_s=0.05)

    async def _drive() -> None:
        for _ in range(5):
            watcher._enqueue(f)
            await asyncio.sleep(0.01)
        # Wait for debounce to flush the batch
        while watcher._pending_scheduled or watcher._pending_paths:
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.1)  # let the flush task complete

    asyncio.run(_drive())
    # All 5 enqueues should coalesce into 1 batch
    assert parse_sources.await_count == 1


def test_debounce_waits_for_same_path_quiet_window(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, _parse_sources = _make_watcher(tmp_path, root, debounce_s=0.05)
    producer_done = asyncio.Event()
    batches: list[bool] = []

    async def fake_ingest(
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
    ) -> None:
        del paths, queued_file_count, skipped_file_count
        batches.append(producer_done.is_set())

    async def _drive() -> None:
        watcher._ingest_files = fake_ingest  # type: ignore[assignment,method-assign]
        for _ in range(4):
            watcher._enqueue(f)
            await asyncio.sleep(0.03)
        producer_done.set()
        while watcher._pending_scheduled or watcher._pending_paths:
            await asyncio.sleep(0.01)

    asyncio.run(_drive())

    assert batches == [True]


# --- WatchSource ---------------------------------------------------------------


def test_watch_source_exists_true(tmp_path: Path) -> None:
    src = WatchSource(name="x", root=tmp_path)
    assert src.exists() is True


def test_watch_source_exists_false(tmp_path: Path) -> None:
    src = WatchSource(name="x", root=tmp_path / "nope")
    assert src.exists() is False


def test_watch_source_accepts_configured_suffixes(tmp_path: Path) -> None:
    src = WatchSource(name="x", root=tmp_path, suffixes=(".json", ".jsonl"))
    assert src.accepts(tmp_path / "session.json") is True
    assert src.accepts(tmp_path / "session.jsonl") is True
    assert src.accepts(tmp_path / "README.md") is False


def test_inbox_source_accepts_zip_and_archive_formats() -> None:
    """#1683: inbox must accept .zip (GDPR exports), .json, .jsonl, .ndjson."""
    from polylogue.sources.live.watcher import default_sources

    inbox = next(s for s in default_sources() if s.name == "inbox")
    assert ".zip" in inbox.suffixes
    assert ".json" in inbox.suffixes
    assert ".jsonl" in inbox.suffixes
    assert ".ndjson" in inbox.suffixes


def test_browser_capture_spool_is_default_json_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import polylogue.paths as polylogue_paths
    from polylogue.sources.live.watcher import default_sources

    spool = tmp_path / "browser-capture"
    monkeypatch.setattr(
        polylogue_paths,
        "browser_capture_spool_root",
        lambda: spool,
    )

    browser_capture = next(s for s in default_sources() if s.name == "browser-capture")

    assert browser_capture.root == spool
    assert browser_capture.accepts(spool / "chatgpt" / "capture.json") is True
    assert browser_capture.accepts(spool / "chatgpt" / "capture.jsonl") is False


# --- end-to-end via watchfiles -------------------------------------------------


def test_end_to_end_modify_triggers_ingest(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root, debounce_s=0.05)

    async def _drive() -> None:
        run_task = asyncio.create_task(watcher.run())
        await asyncio.wait_for(watcher.catch_up_complete.wait(), timeout=5.0)
        baseline = parse_sources.await_count
        # Append immediately at the old catch-up/watch handoff boundary.
        with open(f, "a") as fh:
            fh.write('{"b":2}\n')
        for _ in range(60):
            if parse_sources.await_count > baseline:
                break
            await asyncio.sleep(0.1)
        watcher.stop()
        try:
            await asyncio.wait_for(run_task, timeout=5.0)
        except asyncio.TimeoutError:
            run_task.cancel()
        assert parse_sources.await_count > baseline

    asyncio.run(_drive())


def test_end_to_end_new_file_creation_triggers_ingest(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    watcher, parse_sources = _make_watcher(tmp_path, root, debounce_s=0.05)

    async def _drive() -> None:
        run_task = asyncio.create_task(watcher.run())
        await asyncio.sleep(0.2)  # ensure awatch is up
        f = root / "fresh.jsonl"
        f.write_text('{"new":true}\n')
        for _ in range(60):
            if parse_sources.await_count >= 1:
                break
            await asyncio.sleep(0.1)
        watcher.stop()
        try:
            await asyncio.wait_for(run_task, timeout=5.0)
        except asyncio.TimeoutError:
            run_task.cancel()
        assert parse_sources.await_count >= 1

    asyncio.run(_drive())


def test_end_to_end_hidden_root_file_creation_triggers_ingest(tmp_path: Path) -> None:
    root = tmp_path / ".hidden" / "browser-capture"
    root.mkdir(parents=True)
    watcher, parse_sources = _make_watcher(
        tmp_path,
        root,
        debounce_s=0.05,
        sources=(WatchSource(name="browser-capture", root=root, suffixes=(".json",)),),
    )

    async def _drive() -> None:
        run_task = asyncio.create_task(watcher.run())
        await asyncio.sleep(0.2)  # ensure awatch is up
        f = root / "chatgpt" / "fresh.json"
        f.parent.mkdir()
        f.write_text('{"polylogue_capture_kind":"browser_llm_session"}\n')
        for _ in range(60):
            if parse_sources.await_count >= 1:
                break
            await asyncio.sleep(0.1)
        watcher.stop()
        try:
            await asyncio.wait_for(run_task, timeout=5.0)
        except asyncio.TimeoutError:
            run_task.cancel()
        assert parse_sources.await_count >= 1

    asyncio.run(_drive())


def test_periodic_catch_up_drains_missed_browser_capture_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / ".hidden" / "browser-capture"
    root.mkdir(parents=True)
    watcher, parse_sources = _make_watcher(
        tmp_path,
        root,
        debounce_s=0.05,
        sources=(WatchSource(name="browser-capture", root=root, suffixes=(".json",)),),
    )
    monkeypatch.setattr(live_watcher, "_PERIODIC_CATCH_UP_INTERVAL_S", 0.05)

    async def _drive() -> None:
        task = asyncio.create_task(watcher._periodic_catch_up([root]))
        f = root / "chatgpt" / "missed.json"
        f.parent.mkdir()
        f.write_text('{"polylogue_capture_kind":"browser_llm_session"}\n')
        for _ in range(60):
            if parse_sources.await_count >= 1:
                break
            await asyncio.sleep(0.05)
        watcher.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert parse_sources.await_count >= 1

    asyncio.run(_drive())


def test_periodic_catch_up_backs_off_after_each_reconciliation_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    monkeypatch.setattr(live_watcher, "_PERIODIC_CATCH_UP_INTERVAL_S", 0.01)
    monkeypatch.setattr(live_watcher, "_PERIODIC_CATCH_UP_MAX_INTERVAL_S", 0.04)
    delays: list[float] = []
    passes = 0

    async def fake_sleep(delay_s: float) -> None:
        delays.append(delay_s)

    async def fake_catch_up(_roots: list[Path]) -> None:
        nonlocal passes
        passes += 1
        if passes == 3:
            watcher.stop()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    watcher._catch_up = fake_catch_up  # type: ignore[assignment,method-assign]

    asyncio.run(watcher._periodic_catch_up([root]))

    assert delays == [0.01, 0.02, 0.04]


def test_end_to_end_deletion_does_not_ingest(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "doomed.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root, debounce_s=0.05)

    async def _drive() -> None:
        run_task = asyncio.create_task(watcher.run())
        # Wait for catch_up
        for _ in range(50):
            if parse_sources.await_count >= 1:
                break
            await asyncio.sleep(0.05)
        baseline = parse_sources.await_count
        f.unlink()
        await asyncio.sleep(0.5)
        watcher.stop()
        try:
            await asyncio.wait_for(run_task, timeout=5.0)
        except asyncio.TimeoutError:
            run_task.cancel()
        assert parse_sources.await_count == baseline  # deletion did not trigger ingest

    asyncio.run(_drive())
