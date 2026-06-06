from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.batch_observability import record_attempt_progress
from polylogue.sources.live.cursor import CursorStore


def test_cursor_progress_writes_do_not_raise_on_transient_sqlite_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')
    attempt_id = store.begin_ingest_attempt(paths=[source], input_bytes=source.stat().st_size, queued_file_count=1)

    def locked_connect() -> sqlite3.Connection:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("polylogue.sources.live.sqlite_locking.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(store, "_connect_ops", locked_connect)

    assert (
        store.update_ingest_attempt(
            attempt_id,
            phase="full_parse",
            succeeded_file_count=0,
            failed_file_count=0,
        )
        is False
    )
    assert (
        store.record_ingest_stage_event(
            attempt_id,
            phase="full_parse",
            succeeded_file_count=0,
            failed_file_count=0,
        )
        is False
    )
    assert store.finish_ingest_attempt(attempt_id, status="failed", phase="failed", error="boom") is False


def test_cursor_progress_writes_still_raise_non_lock_sqlite_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')
    attempt_id = store.begin_ingest_attempt(paths=[source], input_bytes=source.stat().st_size, queued_file_count=1)

    def broken_connect() -> sqlite3.Connection:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(store, "_connect_ops", broken_connect)

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        store.update_ingest_attempt(attempt_id, phase="full_parse")


def test_record_attempt_progress_skips_stage_event_when_attempt_update_is_locked(tmp_path: Path) -> None:
    class LockedProgressCursor:
        def update_ingest_attempt(self, *args: object, **kwargs: object) -> bool:
            return False

        def record_ingest_stage_event(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("stage event should not be attempted after locked progress update")

    record_attempt_progress(
        LockedProgressCursor(),
        "attempt-1",
        phase="full_parse",
        succeeded_file_count=0,
        failed_file_count=0,
        source_payload_read_bytes=0,
        cursor_fingerprint_read_bytes=0,
        parse_time_s=0.0,
        current_path=tmp_path / "session.jsonl",
    )


def test_failed_cursor_bookkeeping_does_not_raise_on_transient_sqlite_lock(tmp_path: Path) -> None:
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')

    class LockedCursor:
        def get_record(self, path: Path) -> None:
            del path
            raise sqlite3.OperationalError("database is locked")

    processor = LiveBatchProcessor(
        cast(Any, object()),
        [],
        cursor=cast(CursorStore, LockedCursor()),
        parser_fingerprint="fp:test",
    )

    assert processor._record_failed_cursor(source) == source.stat().st_size
