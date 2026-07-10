from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.batch_observability import record_attempt_progress
from polylogue.sources.live.batch_support import _AppendPlan
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


def test_failed_persistence_preserves_last_committed_cursor_offset(tmp_path: Path) -> None:
    source = tmp_path / "session.jsonl"
    committed = b'{"role":"user","content":"committed"}\n'
    appended = b'{"role":"assistant","content":"not persisted"}\n'
    source.write_bytes(committed + appended)
    store = CursorStore(tmp_path / "live.sqlite")
    store.set(
        source,
        len(committed),
        byte_offset=len(committed),
        last_complete_newline=len(committed),
        parser_fingerprint="fp:test",
    )
    processor = LiveBatchProcessor(
        cast(Any, object()),
        [],
        cursor=store,
        parser_fingerprint="fp:test",
    )

    processor._record_failed_cursor(source)

    record = store.get_record(source)
    assert record is not None
    assert record.byte_size == len(committed + appended)
    assert record.byte_offset == len(committed)
    assert record.last_complete_newline == len(committed)
    assert record.failure_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["append", "full"])
async def test_archive_lock_never_advances_or_excludes_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "session.jsonl"
    payload = b'{"type":"session_meta","payload":{"id":"retry-me"}}\n'
    source.write_bytes(payload)
    cursor = CursorStore(tmp_path / "ops.db")
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="fp:test",
    )

    async def locked_full(*_args: object, **_kwargs: object) -> object:
        raise sqlite3.OperationalError("database is locked")

    if route == "append":
        stat = source.stat()
        plan = _AppendPlan(
            path=source,
            source_name="codex",
            start_offset=0,
            last_complete_newline=stat.st_size,
            stat_size=stat.st_size,
            st_dev=stat.st_dev,
            st_ino=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
            payload=payload,
            payload_hash="retry",
            cursor_fingerprint=None,
            bytes_read=len(payload),
        )
        monkeypatch.setattr(processor, "_append_plan", lambda *_args, **_kwargs: plan)
        monkeypatch.setattr(
            processor,
            "_ingest_append_plans",
            lambda _plans: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
        )
    else:
        monkeypatch.setattr(processor, "_append_plan", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(processor, "_ingest_full_paths", locked_full)

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        await processor.ingest_files([source], emit_event=False)

    assert cursor.get_record(source) is None
