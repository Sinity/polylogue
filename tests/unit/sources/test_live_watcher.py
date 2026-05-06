"""Tests for the live filesystem watcher: cursor, ingest skip logic,
debounce, bootstrap scan, and end-to-end via the watchfiles event loop."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue import Polylogue
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorRecord, CursorStore

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
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='live_cursor'").fetchall()
    assert rows == [("live_cursor",)]


def test_cursor_writes_updated_at(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    p = tmp_path / "s.jsonl"
    store.set(p, 1)
    with sqlite3.connect(tmp_path / "live.sqlite") as conn:
        row = conn.execute("SELECT updated_at FROM live_cursor WHERE source_path=?", (str(p),)).fetchone()
    assert row[0]
    assert "T" in row[0]  # ISO 8601


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


def test_cursor_migrates_size_only_rows(tmp_path: Path) -> None:
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

    assert record is not None
    assert record.byte_size == 12
    assert record.byte_offset == 0
    assert record.content_fingerprint is None
    with sqlite3.connect(db) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(live_cursor)")}
    assert {"byte_offset", "content_fingerprint", "source_name"}.issubset(columns)


# --- LiveWatcher: needs_work + ingest_files (batched) --------------------------


def _make_watcher(tmp_path: Path, root: Path, *, debounce_s: float = 0.01) -> tuple[LiveWatcher, AsyncMock]:
    polylogue = MagicMock()
    polylogue.archive_root = tmp_path
    polylogue.parse_sources = AsyncMock()
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    sources = (WatchSource(name="test", root=root),)
    watcher = LiveWatcher(polylogue, sources, debounce_s=debounce_s, cursor=cursor)
    return watcher, polylogue.parse_sources


def test_watcher_default_cursor_uses_archive_database(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    db_path = tmp_path / "archive.db"
    polylogue = cast(
        Any,
        SimpleNamespace(
            archive_root=tmp_path,
            backend=SimpleNamespace(db_path=db_path),
            parse_sources=AsyncMock(),
        ),
    )

    watcher = LiveWatcher(polylogue, (WatchSource(name="test", root=root),))

    assert watcher._cursor._db_path == db_path


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


def test_same_size_rewrite_triggers_reingest(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)

    asyncio.run(_ingest_one(watcher, f))
    assert parse_sources.await_count == 1

    f.write_text('{"b":2}\n')
    assert f.stat().st_size == len('{"a":1}\n')
    asyncio.run(_ingest_one(watcher, f))

    assert parse_sources.await_count == 2


def test_legacy_size_only_cursor_reingests_to_populate_fingerprint(tmp_path: Path) -> None:
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
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )

    def fail_fingerprint(path: Path) -> tuple[str, int]:
        raise AssertionError(f"unchanged file should not be fingerprinted: {path}")

    monkeypatch.setattr(live_watcher, "fingerprint_file", fail_fingerprint)

    assert watcher._needs_work(f) is False


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
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )

    plan = watcher._batch_processor._append_plan(f)

    assert plan is not None
    assert plan.start_offset == len(original)
    assert plan.payload == b'{"b":2}\n'
    assert plan.bytes_read == len(appended)
    assert plan.last_complete_newline == len(original) + len(b'{"b":2}\n')


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
                )
            ],
        )
        initial_metrics = await processor.ingest_files([source_path], emit_event=False)

        with source_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    _claude_code_message(
                        session_id="session-public-read",
                        uuid="msg-2",
                        parent_uuid="msg-1",
                        role="assistant",
                        text="second live reply",
                        timestamp="2026-05-01T00:00:01Z",
                    )
                )
                + "\n"
            )
        append_metrics = await processor.ingest_files([source_path], emit_event=False)

        conversation = await archive.get_conversation("claude-code:session-public-read")
        assert initial_metrics.full_file_count == 1
        assert append_metrics.append_file_count == 1
        assert append_metrics.full_file_count == 0
        assert append_metrics.source_payload_read_bytes < append_metrics.input_bytes
        assert conversation is not None
        assert [message.text for message in conversation.messages] == ["first live message", "second live reply"]
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

        existing = await archive.get_conversation("codex:codex-real-session")
        fallback = await archive.get_conversation("codex:codex-session")
        assert append_metrics.append_file_count == 1
        assert append_metrics.full_file_count == 0
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

    record = watcher._cursor.get_record(f)
    assert parse_sources.await_count == 1
    assert record is not None
    assert record.byte_size == len(complete + partial)
    assert record.byte_offset == len(complete)
    assert record.last_complete_newline == len(complete)


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
    """A year-old conversation that gets new lines must be picked up — there
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
    watcher, _parse_sources = _make_watcher(tmp_path, root)

    with patch("polylogue.daemon.events.emit_daemon_event") as emit:
        asyncio.run(watcher._ingest_files([f], queued_file_count=3, skipped_file_count=2))

    emit.assert_called_once()
    kind = emit.call_args.args[0]
    payload = emit.call_args.kwargs["payload"]
    assert kind == "ingestion_batch"
    assert payload["queued_file_count"] == 3
    assert payload["needed_file_count"] == 1
    assert payload["skipped_file_count"] == 2
    assert payload["succeeded_file_count"] == 1
    assert payload["failed_file_count"] == 0
    assert payload["source_group_count"] == 1
    assert payload["input_bytes"] == f.stat().st_size
    assert payload["source_payload_read_bytes"] == f.stat().st_size
    assert payload["cursor_fingerprint_read_bytes"] == f.stat().st_size
    assert payload["append_file_count"] == 0
    assert payload["full_file_count"] == 1
    assert payload["archive_write_bytes_delta"] >= 0
    assert payload["parse_time_s"] >= 0
    assert payload["total_time_s"] >= 0
    assert payload["stage_timings_s"] == {}
    assert payload["failed_paths"] == []


def test_parse_failure_retries_after_backoff(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"role":"user","content":"a"}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)
    parse_sources.side_effect = RuntimeError("parser sad")

    asyncio.run(_ingest_one(watcher, f))
    parse_sources.reset_mock()
    parse_sources.side_effect = None
    past = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    with sqlite3.connect(tmp_path / "cursor.sqlite") as conn:
        conn.execute("UPDATE live_cursor SET next_retry_at = ? WHERE source_path = ?", (past, str(f)))
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


def test_catch_up_ignores_junk_at_wrong_depth(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    # Files at root level are ignored (not {project}/{uuid}.jsonl)
    (root / "orphan.jsonl").write_text('{"a":1}\n')
    # Files nested too deep without subagent structure are ignored
    deep = root / "p" / "u" / "extra" / "deep.jsonl"
    deep.parent.mkdir(parents=True)
    deep.write_text('{"a":1}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)

    asyncio.run(watcher._catch_up([root]))
    assert parse_sources.await_count == 0


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


# --- WatchSource ---------------------------------------------------------------


def test_watch_source_exists_true(tmp_path: Path) -> None:
    src = WatchSource(name="x", root=tmp_path)
    assert src.exists() is True


def test_watch_source_exists_false(tmp_path: Path) -> None:
    src = WatchSource(name="x", root=tmp_path / "nope")
    assert src.exists() is False


# --- end-to-end via watchfiles -------------------------------------------------


def test_end_to_end_modify_triggers_ingest(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root, debounce_s=0.05)

    async def _drive() -> None:
        run_task = asyncio.create_task(watcher.run())
        # Wait for catch_up to finish ingesting the pre-existing file.
        for _ in range(50):
            if parse_sources.await_count >= 1:
                break
            await asyncio.sleep(0.05)
        baseline = parse_sources.await_count
        # Append more content; expect a second ingest after debounce.
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
