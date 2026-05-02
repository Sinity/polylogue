"""Tests for the live filesystem watcher: cursor, ingest skip logic,
debounce, bootstrap scan, and end-to-end via the watchfiles event loop."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue.sources.live import LiveWatcher, WatchSource
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


# --- LiveWatcher: ingest_if_grown ---------------------------------------------


def _make_watcher(tmp_path: Path, root: Path, *, debounce_s: float = 0.01) -> tuple[LiveWatcher, AsyncMock]:
    polylogue = MagicMock()
    polylogue.archive_root = tmp_path
    polylogue.parse_file = AsyncMock()
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    sources = (WatchSource(name="test", root=root),)
    watcher = LiveWatcher(polylogue, sources, debounce_s=debounce_s, cursor=cursor)
    return watcher, polylogue.parse_file


def test_skip_when_file_not_grown(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"role":"user","content":"a"}\n')
    watcher, parse_file = _make_watcher(tmp_path, root)

    asyncio.run(watcher._ingest_if_grown(f))
    assert parse_file.await_count == 1

    asyncio.run(watcher._ingest_if_grown(f))
    assert parse_file.await_count == 1  # cursor matches size


def test_reingest_when_file_grows(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"role":"user","content":"a"}\n')
    watcher, parse_file = _make_watcher(tmp_path, root)

    asyncio.run(watcher._ingest_if_grown(f))
    f.write_text('{"a":1}\n{"b":2}\n{"c":3}\n')
    asyncio.run(watcher._ingest_if_grown(f))
    f.write_text('{"a":1}\n{"b":2}\n{"c":3}\n{"d":4}\n')
    asyncio.run(watcher._ingest_if_grown(f))

    assert parse_file.await_count == 3


def test_same_size_rewrite_triggers_reingest(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_file = _make_watcher(tmp_path, root)

    asyncio.run(watcher._ingest_if_grown(f))
    assert parse_file.await_count == 1

    f.write_text('{"b":2}\n')
    assert f.stat().st_size == len('{"a":1}\n')
    asyncio.run(watcher._ingest_if_grown(f))

    assert parse_file.await_count == 2


def test_legacy_size_only_cursor_reingests_to_populate_fingerprint(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_file = _make_watcher(tmp_path, root)
    watcher._cursor.set(f, f.stat().st_size)

    asyncio.run(watcher._ingest_if_grown(f))
    asyncio.run(watcher._ingest_if_grown(f))

    assert parse_file.await_count == 1
    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.content_fingerprint
    assert record.parser_fingerprint


def test_parser_fingerprint_change_triggers_reingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_file = _make_watcher(tmp_path, root)

    asyncio.run(watcher._ingest_if_grown(f))
    monkeypatch.setattr(live_watcher, "_PARSER_FINGERPRINT", "live-jsonl-full-file-v2")
    asyncio.run(watcher._ingest_if_grown(f))

    assert parse_file.await_count == 2
    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.parser_fingerprint == "live-jsonl-full-file-v2"


def test_truncate_rewrite_triggers_reingest(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n{"b":2}\n')
    watcher, parse_file = _make_watcher(tmp_path, root)

    asyncio.run(watcher._ingest_if_grown(f))
    f.write_text('{"c":3}\n')
    asyncio.run(watcher._ingest_if_grown(f))

    assert parse_file.await_count == 2


def test_partial_trailing_line_keeps_cursor_at_last_newline(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    complete = b'{"a":1}\n'
    partial = b'{"b":'
    f.write_bytes(complete + partial)
    watcher, parse_file = _make_watcher(tmp_path, root)

    asyncio.run(watcher._ingest_if_grown(f))

    record = watcher._cursor.get_record(f)
    assert parse_file.await_count == 1
    assert record is not None
    assert record.byte_size == len(complete + partial)
    assert record.byte_offset == len(complete)
    assert record.last_complete_newline == len(complete)


def test_append_after_partial_line_reingests_completed_record(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n{"b":')
    watcher, parse_file = _make_watcher(tmp_path, root)

    asyncio.run(watcher._ingest_if_grown(f))
    f.write_text('{"a":1}\n{"b":2}\n')
    asyncio.run(watcher._ingest_if_grown(f))

    assert parse_file.await_count == 2
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
    watcher, parse_file = _make_watcher(tmp_path, root)
    asyncio.run(watcher._ingest_if_grown(old))
    parse_file.reset_mock()

    old.write_text('{"role":"user","content":"original"}\n{"role":"user","content":"resumed"}\n')
    asyncio.run(watcher._ingest_if_grown(old))
    assert parse_file.await_count == 1


def test_missing_file_is_silent(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    watcher, parse_file = _make_watcher(tmp_path, root)
    asyncio.run(watcher._ingest_if_grown(root / "ghost.jsonl"))
    assert parse_file.await_count == 0


def test_parse_failure_does_not_advance_cursor(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"role":"user","content":"a"}\n')
    watcher, parse_file = _make_watcher(tmp_path, root)
    parse_file.side_effect = RuntimeError("parser sad")

    asyncio.run(watcher._ingest_if_grown(f))
    assert watcher._cursor.get(f) == 0  # cursor not advanced

    parse_file.side_effect = None
    asyncio.run(watcher._ingest_if_grown(f))
    assert watcher._cursor.get(f) > 0  # advanced after success


def test_parse_failure_retries_on_next_event(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_file = _make_watcher(tmp_path, root)
    parse_file.side_effect = [RuntimeError("flaky"), None]

    asyncio.run(watcher._ingest_if_grown(f))
    asyncio.run(watcher._ingest_if_grown(f))

    assert parse_file.await_count == 2


def test_parse_file_receives_watch_source_name(tmp_path: Path) -> None:
    root = tmp_path / "src"
    project_dir = root / "my-project-slug"
    project_dir.mkdir(parents=True)
    f = project_dir / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_file = _make_watcher(tmp_path, root)

    asyncio.run(watcher._ingest_if_grown(f))
    parse_file.assert_awaited_once()
    call = parse_file.await_args
    assert call is not None
    assert call.kwargs["source_name"] == "test"


def test_cursor_records_watch_source_name(tmp_path: Path) -> None:
    root = tmp_path / "src"
    nested = root / "my-project-slug"
    nested.mkdir(parents=True)
    f = nested / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, _parse_file = _make_watcher(tmp_path, root)

    asyncio.run(watcher._ingest_if_grown(f))

    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.source_name == "test"


# --- catch_up bootstrap --------------------------------------------------------


def test_catch_up_processes_pre_existing_files(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    files = [root / f"s{i}.jsonl" for i in range(3)]
    for f in files:
        f.write_text('{"a":1}\n')
    watcher, parse_file = _make_watcher(tmp_path, root)

    asyncio.run(watcher._catch_up([root]))
    assert parse_file.await_count == 3


def test_catch_up_skips_already_processed(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "s.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_file = _make_watcher(tmp_path, root)
    asyncio.run(watcher._ingest_if_grown(f))
    parse_file.reset_mock()

    asyncio.run(watcher._catch_up([root]))
    assert parse_file.await_count == 0


def test_catch_up_walks_recursively(tmp_path: Path) -> None:
    root = tmp_path / "src"
    nested = root / "deep" / "nested"
    nested.mkdir(parents=True)
    f = nested / "deep.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_file = _make_watcher(tmp_path, root)

    asyncio.run(watcher._catch_up([root]))
    assert parse_file.await_count == 1


def test_catch_up_ignores_non_jsonl(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    (root / "session.jsonl").write_text('{"a":1}\n')
    (root / "config.toml").write_text("x=1")
    (root / "README.md").write_text("# hi")
    watcher, parse_file = _make_watcher(tmp_path, root)

    asyncio.run(watcher._catch_up([root]))
    assert parse_file.await_count == 1


def test_catch_up_handles_empty_roots(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    watcher, parse_file = _make_watcher(tmp_path, root)
    asyncio.run(watcher._catch_up([root]))
    assert parse_file.await_count == 0


# --- debounce ------------------------------------------------------------------


def test_debounce_coalesces_rapid_changes(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_file = _make_watcher(tmp_path, root, debounce_s=0.05)

    async def _drive() -> None:
        for _ in range(5):
            watcher._schedule(f)
            await asyncio.sleep(0.01)
        # Wait long enough for one debounce window to elapse
        entry = watcher._pending[f]
        task = entry.pending_task
        assert task is not None
        await task

    asyncio.run(_drive())
    assert parse_file.await_count == 1


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
    watcher, parse_file = _make_watcher(tmp_path, root, debounce_s=0.05)

    async def _drive() -> None:
        run_task = asyncio.create_task(watcher.run())
        # Wait for catch_up to finish ingesting the pre-existing file.
        for _ in range(50):
            if parse_file.await_count >= 1:
                break
            await asyncio.sleep(0.05)
        baseline = parse_file.await_count
        # Append more content; expect a second ingest after debounce.
        with open(f, "a") as fh:
            fh.write('{"b":2}\n')
        for _ in range(60):
            if parse_file.await_count > baseline:
                break
            await asyncio.sleep(0.1)
        watcher.stop()
        try:
            await asyncio.wait_for(run_task, timeout=5.0)
        except asyncio.TimeoutError:
            run_task.cancel()
        assert parse_file.await_count > baseline

    asyncio.run(_drive())


def test_end_to_end_new_file_creation_triggers_ingest(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    watcher, parse_file = _make_watcher(tmp_path, root, debounce_s=0.05)

    async def _drive() -> None:
        run_task = asyncio.create_task(watcher.run())
        await asyncio.sleep(0.2)  # ensure awatch is up
        f = root / "fresh.jsonl"
        f.write_text('{"new":true}\n')
        for _ in range(60):
            if parse_file.await_count >= 1:
                break
            await asyncio.sleep(0.1)
        watcher.stop()
        try:
            await asyncio.wait_for(run_task, timeout=5.0)
        except asyncio.TimeoutError:
            run_task.cancel()
        assert parse_file.await_count >= 1

    asyncio.run(_drive())


def test_end_to_end_deletion_does_not_ingest(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "doomed.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_file = _make_watcher(tmp_path, root, debounce_s=0.05)

    async def _drive() -> None:
        run_task = asyncio.create_task(watcher.run())
        # Wait for catch_up
        for _ in range(50):
            if parse_file.await_count >= 1:
                break
            await asyncio.sleep(0.05)
        baseline = parse_file.await_count
        f.unlink()
        await asyncio.sleep(0.5)
        watcher.stop()
        try:
            await asyncio.wait_for(run_task, timeout=5.0)
        except asyncio.TimeoutError:
            run_task.cancel()
        assert parse_file.await_count == baseline  # deletion did not trigger ingest

    asyncio.run(_drive())
