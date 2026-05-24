from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.sources.live import WatchSource
from polylogue.sources.live.append_ingest import ingest_append_plans
from polylogue.sources.live.batch import _MAX_APPEND_PLAN_PAYLOAD_BYTES, LiveBatchProcessor
from polylogue.sources.live.batch_support import (
    _DEFER_APPEND,
    _AppendPlan,
    _AppendResult,
    _detect_provider_from_path_sample,
    _parse_path_as_conversation_artifact,
)
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.runtime import RawConversationRecord
from polylogue.types import Provider


def test_full_ingest_heartbeats_small_file_groups_with_current_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    first = root / "first.jsonl"
    second = root / "second.jsonl"
    first.write_text('{"type":"session_meta","payload":{"id":"first"}}\n', encoding="utf-8")
    second.write_text('{"type":"session_meta","payload":{"id":"second"}}\n', encoding="utf-8")
    db_path = tmp_path / "archive.sqlite"
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    events: list[tuple[str, Path | None, int | None]] = []

    def heartbeat(
        phase: str,
        *,
        current_path: Path | None = None,
        source_payload_read_bytes: int | None = None,
        force: bool = False,
    ) -> None:
        del force
        events.append((phase, current_path, source_payload_read_bytes))

    def fake_write_from_bytes(_store: object, payload: bytes) -> tuple[str, int]:
        suffix = f"{len(events):064x}"[-64:]
        return suffix, len(payload)

    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_conversation_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )
    monkeypatch.setattr("polylogue.sources.live.batch.BlobStore.write_from_bytes", fake_write_from_bytes)
    monkeypatch.setattr(
        "polylogue.sources.live.batch._process_ingest_batch_sync",
        lambda *args, **kwargs: SimpleNamespace(failed_raw_ids=set(), parse_failures=0, worker_count=1),
    )

    result = processor._ingest_full_paths_sync([first, second], source_name="codex", heartbeat=heartbeat)

    assert result.succeeded == [first, second]
    assert ("full_file_scan", first, 0) in events
    assert ("full_file_scan", second, first.stat().st_size) in events
    assert any(event == ("full_worker_wait", second, first.stat().st_size + second.stat().st_size) for event in events)


def test_large_full_ingest_suspends_fts_triggers_and_repairs_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "large.jsonl"
    source.write_text('{"type":"session_meta","payload":{"id":"large"}}\n', encoding="utf-8")
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    captured_kwargs: dict[str, object] = {}

    monkeypatch.setattr("polylogue.sources.live.batch._FULL_INGEST_SUSPEND_FTS_TRIGGER_BYTES", 1)
    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_conversation_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch.BlobStore.write_from_bytes",
        lambda _store, payload: ("a" * 64, len(payload)),
    )

    def fake_process_ingest_batch_sync(*args: object, **kwargs: object) -> SimpleNamespace:
        del args
        captured_kwargs.update(kwargs)
        return SimpleNamespace(failed_raw_ids=set(), parse_failures=0, worker_count=1)

    monkeypatch.setattr("polylogue.sources.live.batch._process_ingest_batch_sync", fake_process_ingest_batch_sync)

    result = processor._ingest_full_paths_sync([source], source_name="codex")

    assert result.succeeded == [source]
    assert captured_kwargs["suspend_fts_triggers"] is True
    assert captured_kwargs["repair_message_fts"] is True
    assert captured_kwargs["repair_action_fts"] is True


def test_streaming_sized_full_ingest_suspends_fts_triggers_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "large.jsonl"
    source.write_bytes(b'{"type":"session_meta","payload":{"id":"large"}}\n' + (b" " * (9 * 1024 * 1024)))
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    captured_kwargs: dict[str, object] = {}

    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_conversation_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch.BlobStore.write_from_path",
        lambda _store, path, heartbeat=None: ("a" * 64, path.stat().st_size),
    )

    def fake_process_ingest_batch_sync(*args: object, **kwargs: object) -> SimpleNamespace:
        del args
        captured_kwargs.update(kwargs)
        return SimpleNamespace(failed_raw_ids=set(), parse_failures=0, worker_count=1)

    monkeypatch.setattr("polylogue.sources.live.batch._process_ingest_batch_sync", fake_process_ingest_batch_sync)

    result = processor._ingest_full_paths_sync([source], source_name="codex")

    assert result.succeeded == [source]
    assert captured_kwargs["suspend_fts_triggers"] is True
    assert captured_kwargs["repair_message_fts"] is True
    assert captured_kwargs["repair_action_fts"] is True


def test_fingerprint_file_streams_in_bounded_memory(tmp_path: Path) -> None:
    """``fingerprint_file`` must not load the whole file into memory.

    Regression: the previous implementation read the entire file via
    ``Path.read_bytes()``, producing an RSS peak proportional to file size.
    This test exercises the streaming path on a multi-megabyte synthetic
    file and asserts that the working set stays bounded by ``chunk_size``.
    """
    import hashlib
    import tracemalloc

    from polylogue.sources.live.batch_support import fingerprint_file

    payload = (b"x" * 4095 + b"\n") * 4096  # ~16 MiB, all lines newline-terminated
    target = tmp_path / "huge.jsonl"
    target.write_bytes(payload)
    expected_hash = hashlib.sha256(payload).hexdigest()
    expected_last_nl = len(payload)  # ends in newline

    tracemalloc.start()
    try:
        fp, last_nl = fingerprint_file(target, chunk_size=64 * 1024)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert fp == expected_hash
    assert last_nl == expected_last_nl
    # Peak Python-allocated memory must stay well under the file size. The
    # 1 MiB budget is generous (chunk_size is 64 KiB) but leaves room for
    # hasher state and chunk overhead without admitting a full-file read.
    assert peak < 1 * 1024 * 1024, f"fingerprint_file peak {peak} bytes is not bounded for a {len(payload)}-byte file"


def test_fingerprint_file_tracks_last_newline_across_chunk_boundary(tmp_path: Path) -> None:
    """The streaming fingerprint must locate the last newline even when it
    sits in an earlier chunk than the file tail."""
    from polylogue.sources.live.batch_support import fingerprint_file

    # 4 KiB of newline-terminated lines, followed by 4 KiB without any \n.
    head = (b"line\n") * 1000  # 5_000 bytes, ends with \n
    tail = b"y" * 5000  # no newline anywhere
    payload = head + tail
    target = tmp_path / "no-trailing-newline.jsonl"
    target.write_bytes(payload)

    _fp, last_nl = fingerprint_file(target, chunk_size=1024)
    assert last_nl == len(head), f"last_complete_newline should be at end-of-head ({len(head)}), got {last_nl}"


def test_fingerprint_file_empty_file(tmp_path: Path) -> None:
    import hashlib

    from polylogue.sources.live.batch_support import fingerprint_file

    target = tmp_path / "empty.jsonl"
    target.write_bytes(b"")

    fp, last_nl = fingerprint_file(target)
    assert fp == hashlib.sha256(b"").hexdigest()
    assert last_nl == 0


def test_large_non_jsonl_full_ingest_planning_does_not_read_whole_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "large.json"
    target.write_text('{"mapping": {}}\n', encoding="utf-8")
    monkeypatch.setattr("polylogue.sources.live.batch_support._path_size", lambda path: 32 * 1024 * 1024)

    def fail_read_bytes(_path: Path) -> bytes:
        raise AssertionError("large full-ingest planning must not materialize the whole file")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    assert _detect_provider_from_path_sample(target, Provider.CHATGPT) is Provider.CHATGPT
    assert _parse_path_as_conversation_artifact(target, provider=Provider.CHATGPT) is True


def test_unclassified_large_non_jsonl_is_not_streamed_as_conversation_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "unknown.large"
    target.write_bytes(b"not-json")
    monkeypatch.setattr("polylogue.sources.live.batch_support._path_size", lambda path: 32 * 1024 * 1024)

    def fail_read_bytes(_path: Path) -> bytes:
        raise AssertionError("unclassified large files must not be materialized during planning")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    assert _parse_path_as_conversation_artifact(target, provider=Provider.UNKNOWN) is False


def test_append_plan_chunks_large_tail_without_full_ingest(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    path = root / "session.jsonl"
    original = b'{"a":1}\n'
    first_chunk = b'{"b":"' + (b"x" * (_MAX_APPEND_PLAN_PAYLOAD_BYTES - 128)) + b'"}\n'
    second_chunk = b'{"c":"' + (b"y" * 512) + b'"}\n'
    appended = first_chunk + second_chunk
    path.write_bytes(original + appended)
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="chatgpt", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    stat = path.stat()
    processor._cursor.set(
        path,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint="test-parser",
        content_fingerprint="base",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )

    plan = processor._append_plan(path)

    assert isinstance(plan, _AppendPlan)
    assert plan.start_offset == len(original)
    assert plan.last_complete_newline == len(original) + len(first_chunk)
    assert plan.stat_size == len(original) + len(appended)
    assert plan.bytes_read == _MAX_APPEND_PLAN_PAYLOAD_BYTES
    assert plan.payload == first_chunk

    assert processor._record_append_cursor(plan) is True
    next_plan = processor._append_plan(path)
    assert isinstance(next_plan, _AppendPlan)
    assert next_plan.start_offset == len(original) + len(first_chunk)
    assert next_plan.last_complete_newline == len(original) + len(appended)
    assert next_plan.payload == second_chunk


def test_append_plan_defers_when_tail_has_no_complete_line(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "session.jsonl"
    original = b'{"a":1}\n'
    path.write_bytes(original + b'{"b":')
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="chatgpt", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    stat = path.stat()
    processor._cursor.set(
        path,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint="test-parser",
        content_fingerprint="base",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )

    assert processor._append_plan(path) is _DEFER_APPEND


def test_incomplete_append_is_requeued_not_full_ingested(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "session.jsonl"
    original = b'{"a":1}\n'
    path.write_bytes(original + b'{"b":')
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="chatgpt", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    stat = path.stat()
    processor._cursor.set(
        path,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint="test-parser",
        content_fingerprint="base",
        source_name="chatgpt",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )

    metrics = asyncio.run(processor.ingest_files([path], emit_event=False))

    assert metrics.full_file_count == 0
    assert metrics.append_file_count == 0
    assert metrics.failed_paths == [str(path)]


def test_codex_append_plan_uses_append_only_conversation_identity(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.schema import _ensure_schema

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "rollout-2026-05-16T13-50-17-019e309f-7614-7381-a8ac-f9080f304ee6.jsonl"
    original = b'{"type":"session_meta","payload":{"id":"019e309f-7614-7381-a8ac-f9080f304ee6"}}\n'
    appended = b'{"type":"event_msg","payload":{"message":"new"}}\n'
    path.write_bytes(original + appended)
    db_path = tmp_path / "archive.sqlite"
    cursor = CursorStore(db_path)
    with cursor._connect() as conn:
        _ensure_schema(conn)
        conn.execute(
            """
            INSERT INTO raw_conversations (
                raw_id, provider_name, payload_provider, source_name, source_path,
                source_index, blob_size, acquired_at, file_mtime
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "append-raw",
                "codex",
                "codex",
                "codex",
                str(path),
                -1,
                len(original),
                "2026-05-24T00:00:00+00:00",
                "2026-05-24T00:00:00+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO conversations (
                conversation_id, provider_name, provider_conversation_id, title,
                created_at, updated_at, sort_key, content_hash, provider_meta,
                metadata, source_name, working_directories_json, git_branch,
                git_repository_url, version, parent_conversation_id, branch_type, raw_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "codex:019e309f-7614-7381-a8ac-f9080f304ee6",
                "codex",
                "019e309f-7614-7381-a8ac-f9080f304ee6",
                "hot session",
                "2026-05-24T00:00:00+00:00",
                "2026-05-24T00:00:00+00:00",
                0.0,
                "base-hash",
                '{"source":"codex"}',
                "{}",
                "codex",
                None,
                None,
                None,
                1,
                None,
                None,
                "append-raw",
            ),
        )
        conn.commit()
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    stat = path.stat()
    cursor.set(
        path,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint="test-parser",
        content_fingerprint="base-cursor",
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )

    plan = processor._append_plan(path)

    assert isinstance(plan, _AppendPlan)
    assert plan.start_offset == len(original)
    assert plan.payload.startswith(b'{"type":"session_meta","payload":{"id":"019e309f-7614-7381-a8ac-f9080f304ee6"}}\n')
    assert plan.payload.endswith(appended)


def test_append_ingest_preserves_successes_when_other_plan_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Owner:
        def __init__(self) -> None:
            self._cursor = CursorStore(tmp_path / "append.sqlite")
            self._polylogue = SimpleNamespace(
                archive_root=tmp_path,
                backend=SimpleNamespace(db_path=self._cursor._db_path),
            )
            self.persisted_count = 0

        def _persist_raw_records(self, records: list[RawConversationRecord]) -> None:
            self.persisted_count = len(records)

    plans = [
        _AppendPlan(
            path=tmp_path / "ok.jsonl",
            source_name="codex",
            start_offset=0,
            last_complete_newline=8,
            stat_size=8,
            st_dev=1,
            st_ino=1,
            mtime_ns=1,
            payload=b'{"ok":1}\n',
            payload_hash="ok",
            cursor_fingerprint="base",
            bytes_read=8,
        ),
        _AppendPlan(
            path=tmp_path / "bad.jsonl",
            source_name="codex",
            start_offset=0,
            last_complete_newline=9,
            stat_size=9,
            st_dev=1,
            st_ino=2,
            mtime_ns=1,
            payload=b'{"bad":1}\n',
            payload_hash="bad",
            cursor_fingerprint="base",
            bytes_read=9,
        ),
    ]
    raw_ids = iter(("raw-ok", "raw-bad"))
    captured_kwargs: dict[str, object] = {}
    monkeypatch.setattr(
        "polylogue.sources.live.append_ingest.BlobStore.write_from_bytes",
        lambda _store, payload: (next(raw_ids), len(payload)),
    )

    def fake_process_ingest_batch_sync(*args: object, **kwargs: object) -> SimpleNamespace:
        del args
        captured_kwargs.update(kwargs)
        return SimpleNamespace(failed_raw_ids={"raw-bad"}, parse_failures=1, worker_count=1)

    monkeypatch.setattr(
        "polylogue.sources.live.append_ingest._process_ingest_batch_sync", fake_process_ingest_batch_sync
    )

    owner = Owner()
    result = ingest_append_plans(owner, plans)

    assert owner.persisted_count == 2
    assert result.succeeded == [plans[0]]
    assert result.failed == [plans[1]]
    assert result.worker_count == 1
    assert captured_kwargs["repair_message_fts"] is True
    assert captured_kwargs["repair_action_fts"] is True


@pytest.mark.asyncio
async def test_live_append_plans_flush_in_bounded_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    paths = [root / f"{index}.jsonl" for index in range(5)]
    for path in paths:
        path.write_text('{"type":"session_meta","payload":{"id":"bounded"}}\n', encoding="utf-8")
    cursor = CursorStore(tmp_path / "live.sqlite")
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=cursor._db_path))
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    groups: list[list[Path]] = []

    def fake_append_plan(path: Path, *, cursor: object | None = None) -> _AppendPlan:
        del cursor
        return _AppendPlan(
            path=path,
            source_name="codex",
            start_offset=0,
            last_complete_newline=10,
            stat_size=10,
            st_dev=1,
            st_ino=1,
            mtime_ns=1,
            payload=b"payload\n",
            payload_hash="tail",
            cursor_fingerprint="base",
            bytes_read=10,
        )

    def fake_ingest_append_plans(plans: list[_AppendPlan]) -> _AppendResult:
        groups.append([plan.path for plan in plans])
        return _AppendResult(succeeded=list(plans), failed=[], worker_count=1)

    monkeypatch.setattr(processor, "_append_plan", fake_append_plan)
    monkeypatch.setattr(processor, "_ingest_append_plans", fake_ingest_append_plans)
    monkeypatch.setattr(processor, "_converge_paths", lambda paths: (paths, 0.0, {}, []))
    monkeypatch.setattr(processor, "_record_append_cursor", lambda plan: True)
    monkeypatch.setattr(processor, "_record_convergence_outcome", lambda path, debts: None)
    monkeypatch.setattr("polylogue.sources.live.batch._append_plan_group_ready", lambda plans: len(plans) >= 2)

    metrics = await processor.ingest_files(paths, emit_event=False)

    assert groups == [paths[:2], paths[2:4], paths[4:]]
    assert metrics.append_file_count == 5
    assert metrics.full_file_count == 0
