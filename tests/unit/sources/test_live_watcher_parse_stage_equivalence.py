"""Equivalence: watcher parse-stage prefetch (flag on) vs. in-hold parse (flag off).

polylogue-wf8a. ``LiveParseStage`` pre-parses small JSONL full-ingest
candidates off the writer hold (mirrors ``DaemonParseStage``, polylogue-m6tp
phase (a), for the watcher's catch-up/live-batch route instead of the
raw-materialization census route). This proves two end-to-end claims against
a real archive:

1. Running ``LiveBatchProcessor.ingest_files`` over the SAME fixture files,
   once with a ``LiveParseStage`` warming candidates ahead of the writer
   hold and once with no parse stage at all, produces byte-identical durable
   archive content.
2. The writer-held archive-write ORDER (and therefore every order-dependent
   decision downstream, e.g. raw-revision-chain classification) is
   unaffected by out-of-order parallel parse completion -- a deliberately
   reversed completion order still yields the exact same archive content and
   the exact same raw_sessions insertion order as flag-off.

Production dependencies exercised: ``LiveParseStage.warm`` (the actual
off-writer-hold pre-parse path) feeding ``LiveBatchProcessor._ingest_full_paths``
`/`_ingest_full_records_archive`` (the actual production plumbing the watcher
uses), not a reimplementation of either.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, NoReturn

import pytest

from polylogue import Polylogue
from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.daemon.intake import FairIntakeDispatcher, IntakeClassSpec
from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter
from polylogue.operations.operation_context import PinnedOperationRead, open_operation_read
from polylogue.sources.live.batch import LiveBatchProcessor, _live_parse_stage_candidates
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.parse_prefetch import LiveParseStage
from polylogue.sources.live.watcher import _PARSER_FINGERPRINT, LiveWatcher, WatchSource
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.raw_failure_lifecycle import read_raw_failure_lifecycle

_VOLATILE_COLUMNS: dict[str, frozenset[str]] = {
    "raw_sessions": frozenset({"acquired_at_ms", "parsed_at_ms"}),
}


def _stalled_process_worker(marker: str) -> None:
    Path(marker).write_text("started")
    time.sleep(30)


def _dead_process_worker() -> None:
    import os

    os._exit(7)


def _delayed_path_worker(
    provider_value: str,
    source_path: str,
    fallback_id: str,
    *,
    is_stream: bool,
    shard_directory: str,
) -> object:
    from polylogue.sources.live.parse_prefetch import live_parse_path_worker

    time.sleep(0.25)
    return live_parse_path_worker(
        provider_value,
        source_path,
        fallback_id,
        is_stream=is_stream,
        shard_directory=shard_directory,
    )


def _codex_session_bytes(native_id: str, messages: tuple[tuple[str, str], ...]) -> bytes:
    rows: list[dict[str, object]] = [
        {"type": "session_meta", "payload": {"id": native_id, "timestamp": "2026-07-19T00:00:00Z"}}
    ]
    for position, (role, text) in enumerate(messages):
        rows.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"{native_id}-m{position}",
                    "role": role,
                    "content": [
                        {
                            "type": "input_text" if role == "user" else "output_text",
                            "text": text,
                        }
                    ],
                },
            }
        )
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def _write_fixture_corpus(root: Path, *, count: int) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = root / f"session-{index}.jsonl"
        path.write_bytes(
            _codex_session_bytes(
                f"session-{index}",
                (("user", f"question {index}"), ("assistant", f"answer {index}")),
            )
        )
        paths.append(path)
    return paths


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_rows(conn: sqlite3.Connection, table: str, *, order_by: str | None = None) -> tuple[tuple[Any, ...], ...]:
    excluded = _VOLATILE_COLUMNS.get(table, frozenset())
    columns = tuple(
        row["name"] for row in conn.execute(f'PRAGMA table_xinfo("{table}")') if row["name"] not in excluded
    )
    quoted = ", ".join(f'"{column}"' for column in columns)
    query = f'SELECT {quoted} FROM "{table}"'
    if order_by is not None:
        query += f" ORDER BY {order_by}"
    cursor_rows = conn.execute(query).fetchall()
    rows = tuple(
        tuple(bytes(value).hex() if isinstance(value, bytes) else value for value in row) for row in cursor_rows
    )
    if order_by is None:
        rows = tuple(sorted(rows, key=repr))
    return rows


def _canonical_snapshot(archive_root: Path) -> dict[str, tuple[tuple[Any, ...], ...]]:
    snapshot: dict[str, tuple[tuple[Any, ...], ...]] = {}
    with _connect(archive_root / "index.db") as conn:
        for table in ("sessions", "messages", "blocks"):
            snapshot[f"index.{table}"] = _table_rows(conn, table)
    with _connect(archive_root / "source.db") as conn:
        snapshot["source.raw_sessions"] = _table_rows(conn, "raw_sessions")
    return snapshot


def _raw_sessions_source_path_order(archive_root: Path) -> tuple[str, ...]:
    with _connect(archive_root / "source.db") as conn:
        rows = conn.execute("SELECT source_path FROM raw_sessions ORDER BY rowid").fetchall()
    return tuple(str(row["source_path"]) for row in rows)


async def _ingest(archive_root: Path, paths: list[Path], *, parse_stage: LiveParseStage | None) -> None:
    archive_root.mkdir(parents=True, exist_ok=True)
    db_path = archive_root / "index.db"
    polylogue = Polylogue(archive_root=archive_root, db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        polylogue,
        (WatchSource(name="codex", root=paths[0].parent),),
        cursor=cursor,
        parser_fingerprint=_PARSER_FINGERPRINT,
        parse_stage=parse_stage,
        read_snapshot=open_operation_read,
    )
    metrics = await processor.ingest_files(paths, emit_event=False)
    assert metrics.failed_file_count == 0
    assert metrics.succeeded_file_count == len(paths)


@pytest.mark.asyncio
async def test_parse_stage_flag_on_and_off_produce_identical_archive_content(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baseline"
    prefetch_root = tmp_path / "prefetch"
    paths = _write_fixture_corpus(tmp_path / "sessions", count=6)

    await _ingest(baseline_root, paths, parse_stage=None)

    stage = LiveParseStage(max_workers=3, max_inflight_bytes=10_000_000)
    try:
        await _ingest(prefetch_root, paths, parse_stage=stage)
    finally:
        stage.shutdown()
    # Every warmed entry was consumed by the writer-held pass, not left
    # stranded -- proves the prefetch path was actually exercised (not a
    # silent no-op equivalence).
    assert len(stage.cache) == 0

    assert _canonical_snapshot(baseline_root) == _canonical_snapshot(prefetch_root)
    with _connect(baseline_root / "index.db") as conn:
        assert int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]) == 6


@pytest.mark.asyncio
async def test_shard_building_parse_stage_produces_identical_archive_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-bp12n.6: a stage that also writes shards changes nothing durable.

    The writer copies each raw's message and block rows out of the shard its
    parse worker sealed instead of binding them one at a time. The archive
    that comes out must be the one the inline path writes, to the row.

    Anti-vacuity is the copy counter: with the shard path removed (or the
    binding rejected) ``copy_shard_session_rows`` never runs and the
    ``copies`` assertion is red while the equivalence assertion stays green.
    """
    import polylogue.storage.sqlite.archive_tiers.write as archive_tier_write

    baseline_root = tmp_path / "baseline"
    shard_root = tmp_path / "sharded"
    paths = _write_fixture_corpus(tmp_path / "sessions", count=6)

    await _ingest(baseline_root, paths, parse_stage=None)

    copies = 0
    real_copy = archive_tier_write.copy_shard_session_rows

    def counting_copy(*args: object, **kwargs: object) -> object:
        nonlocal copies
        copies += 1
        return real_copy(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(archive_tier_write, "copy_shard_session_rows", counting_copy)

    shard_directory = tmp_path / "parse-shards"
    stage = LiveParseStage(max_workers=3, max_inflight_bytes=10_000_000, shard_directory=shard_directory)
    try:
        await _ingest(shard_root, paths, parse_stage=stage)
    finally:
        stage.shutdown()

    assert len(stage.cache) == 0
    assert copies == len(paths), f"the writer copied {copies} shard sessions, expected {len(paths)}"
    assert _canonical_snapshot(baseline_root) == _canonical_snapshot(shard_root)
    # Shards are scratch with a named end: none may outlive the pass.
    assert list(shard_directory.glob("shard-*")) == []


@pytest.mark.asyncio
async def test_process_parse_stage_produces_identical_archive_content(tmp_path: Path) -> None:
    """The ordinary GIL-safe process route keeps the same durable result."""
    baseline_root = tmp_path / "baseline"
    process_root = tmp_path / "process"
    paths = _write_fixture_corpus(tmp_path / "sessions", count=4)

    await _ingest(baseline_root, paths, parse_stage=None)

    stage = LiveParseStage(
        max_workers=2,
        max_inflight_bytes=10_000_000,
        shard_directory=tmp_path / "parse-shards",
        use_processes=True,
    )
    try:
        assert isinstance(stage._executor, ProcessPoolExecutor)
        await _ingest(process_root, paths, parse_stage=stage)
    finally:
        stage.shutdown()

    assert len(stage.cache) == 0
    assert _canonical_snapshot(baseline_root) == _canonical_snapshot(process_root)


@pytest.mark.asyncio
async def test_path_worker_publishes_prepared_rows_from_captured_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import polylogue.sources.live.batch as batch
    import polylogue.storage.sqlite.archive_tiers.write as archive_tier_write

    paths = _write_fixture_corpus(tmp_path / "sessions", count=1)
    await _ingest(tmp_path / "baseline", paths, parse_stage=None)
    monkeypatch.setattr(batch, "_STREAMING_FULL_INGEST_BYTES", 1)
    copied = 0
    original_copy = archive_tier_write.copy_shard_session_rows

    def count_copy(*args: object, **kwargs: object) -> object:
        nonlocal copied
        copied += 1
        return original_copy(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(archive_tier_write, "copy_shard_session_rows", count_copy)
    directory = tmp_path / "parse-shards"
    stage = LiveParseStage(max_workers=1, shard_directory=directory, use_processes=True)
    try:
        await _ingest(tmp_path / "prepared", paths, parse_stage=stage)
    finally:
        stage.shutdown()
    assert copied == 1
    assert _canonical_snapshot(tmp_path / "baseline") == _canonical_snapshot(tmp_path / "prepared")
    assert list(directory.iterdir()) == []


@pytest.mark.asyncio
async def test_json_document_uses_prepared_rows_and_preserves_detected_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import polylogue.storage.sqlite.archive_tiers.write as archive_tier_write

    source = tmp_path / "inbox" / "session.json"
    source.parent.mkdir()
    source.write_text(
        json.dumps(
            {
                "sessionId": "gemini-prepared-json",
                "startTime": "2026-03-16T09:40:00.000Z",
                "lastUpdated": "2026-03-16T09:41:00.000Z",
                "kind": "chat",
                "messages": [
                    {"id": "u1", "timestamp": "2026-03-16T09:40:01.000Z", "type": "user", "content": ["hello"]},
                    {
                        "id": "a1",
                        "timestamp": "2026-03-16T09:40:02.000Z",
                        "type": "gemini",
                        "content": "world",
                        "model": "gemini-test",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    copies = 0
    real_copy = archive_tier_write.copy_shard_session_rows

    def count_copy(*args: object, **kwargs: object) -> object:
        nonlocal copies
        copies += 1
        return real_copy(*args, **kwargs)  # type: ignore[arg-type]

    async def ingest(root: Path, stage: LiveParseStage | None) -> None:
        root.mkdir()
        processor = LiveBatchProcessor(
            Polylogue(archive_root=root, db_path=root / "index.db"),
            (WatchSource(name="inbox", root=source.parent),),
            cursor=CursorStore(root / "index.db"),
            parser_fingerprint=_PARSER_FINGERPRINT,
            parse_stage=stage,
            read_snapshot=open_operation_read,
        )
        result = await processor.ingest_files([source], emit_event=False)
        assert result.succeeded_file_count == 1
        assert result.ingested_session_count == 1

    await ingest(tmp_path / "baseline", None)
    monkeypatch.setattr(archive_tier_write, "copy_shard_session_rows", count_copy)
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "shards")
    try:
        await ingest(tmp_path / "prepared", stage)
    finally:
        stage.shutdown()
    assert copies == 1
    assert _canonical_snapshot(tmp_path / "baseline") == _canonical_snapshot(tmp_path / "prepared")
    with _connect(tmp_path / "prepared" / "source.db") as conn:
        assert conn.execute("SELECT origin FROM raw_sessions").fetchone()[0] == "gemini-cli-session"


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed_initial", [False, True])
async def test_changed_json_after_preparation_uses_captured_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformed_initial: bool
) -> None:
    """A stale worker's provider or parse error cannot label captured bytes."""
    import polylogue.sources.live.cursor as cursor_module

    monkeypatch.setattr(cursor_module, "_FULL_CURSOR_RECONCILIATION_RETRY_DELAY_S", 0)
    source = tmp_path / "inbox" / "session.json"
    source.parent.mkdir()
    source.write_text(
        json.dumps(
            {
                "sessionId": "gemini-before-copy",
                "startTime": "2026-03-16T09:40:00.000Z",
                "lastUpdated": "2026-03-16T09:41:00.000Z",
                "kind": "chat",
                "messages": [
                    {"id": "u1", "timestamp": "2026-03-16T09:40:01.000Z", "type": "user", "content": ["hello"]}
                ],
            }
        ),
        encoding="utf-8",
    )
    if malformed_initial:
        source.write_bytes(b'{"broken":')
    chatgpt = json.dumps(
        [
            {
                "id": "chatgpt-after-copy",
                "title": "captured chat",
                "create_time": 1,
                "current_node": "m",
                "mapping": {
                    "m": {
                        "id": "m",
                        "parent": None,
                        "children": [],
                        "message": {
                            "id": "m",
                            "author": {"role": "user"},
                            "create_time": 1,
                            "content": {"content_type": "text", "parts": ["captured content"]},
                        },
                    }
                },
            }
        ]
    ).encode()
    original_copy = ArchiveBlobPublisher.write_from_path
    changed = False

    def change_before_copy(store: ArchiveBlobPublisher, path: Path, **kwargs: object) -> tuple[str, int]:
        nonlocal changed
        if path == source and not changed:
            if malformed_initial:
                assert stage._path_results[str(source)].error is not None
            else:
                assert stage.resolved_path_provider(str(source)) is Provider.GEMINI_CLI
            source.write_bytes(chatgpt)
            changed = True
        return original_copy(store, path, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ArchiveBlobPublisher, "write_from_path", change_before_copy)
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "shards")
    processor = LiveBatchProcessor(
        Polylogue(archive_root=archive_root, db_path=archive_root / "index.db"),
        (WatchSource(name="inbox", root=source.parent),),
        cursor=CursorStore(archive_root / "index.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
        parse_stage=stage,
        read_snapshot=open_operation_read,
    )
    try:
        first = await processor.ingest_files([source], emit_event=False)
        assert changed and str(source) in first.deferred_paths
        with _connect(archive_root / "source.db") as conn:
            assert [row["origin"] for row in conn.execute("SELECT origin FROM raw_sessions")] == [
                origin_from_provider(Provider.CHATGPT).value
            ]
        second = await processor.ingest_files([source], emit_event=False)
        assert second.ingested_session_count == 1
        with _connect(archive_root / "source.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 1
    finally:
        stage.shutdown()


@pytest.mark.asyncio
async def test_identical_json_paths_keep_distinct_prepared_fallback_ids(tmp_path: Path) -> None:
    """Equal blob hashes cannot exchange path-bound prepared sessions."""
    source_root = tmp_path / "inbox"
    source_root.mkdir()
    paths = [source_root / "a.json", source_root / "b.json"]
    payload = json.dumps(
        [
            {
                "title": "same bytes",
                "create_time": 1,
                "current_node": "m",
                "mapping": {
                    "m": {
                        "id": "m",
                        "parent": None,
                        "children": [],
                        "message": {
                            "id": "m",
                            "author": {"role": "user"},
                            "create_time": 1,
                            "content": {"content_type": "text", "parts": ["same content"]},
                        },
                    }
                },
            }
        ]
    ).encode()
    for path in paths:
        path.write_bytes(payload)

    stage = LiveParseStage(max_workers=2, shard_directory=tmp_path / "shards")
    try:
        await _ingest(tmp_path / "prepared", paths, parse_stage=stage)
    finally:
        stage.shutdown()
    with _connect(tmp_path / "prepared" / "index.db") as conn:
        assert {row["native_id"] for row in conn.execute("SELECT native_id FROM sessions")} == {"a-0", "b-0"}
    with _connect(tmp_path / "prepared" / "source.db") as conn:
        assert {row["source_path"] for row in conn.execute("SELECT source_path FROM raw_sessions")} == {
            str(path) for path in paths
        }
    cursors = CursorStore(tmp_path / "prepared" / "index.db")
    assert all((record := cursors.get_record(path)) is not None and record.content_fingerprint for path in paths)


@pytest.mark.asyncio
async def test_identical_json_paths_keep_independent_pending_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One pending worker cannot make its equal-byte sibling's cursor settle."""
    import threading

    import polylogue.sources.live.parse_prefetch as parse_prefetch

    source_root = tmp_path / "inbox"
    source_root.mkdir()
    pending_path, ready_path = source_root / "a.json", source_root / "b.json"
    payload = json.dumps(
        [
            {
                "title": "same bytes",
                "create_time": 1,
                "current_node": "m",
                "mapping": {
                    "m": {
                        "id": "m",
                        "parent": None,
                        "children": [],
                        "message": {
                            "id": "m",
                            "author": {"role": "user"},
                            "create_time": 1,
                            "content": {"content_type": "text", "parts": ["same content"]},
                        },
                    }
                },
            }
        ]
    ).encode()
    pending_path.write_bytes(payload)
    ready_path.write_bytes(payload)
    released = threading.Event()
    original_worker = parse_prefetch.live_parse_path_worker

    def delayed_worker(*args: object, **kwargs: object) -> object:
        if str(args[1]) == str(pending_path):
            released.wait(timeout=30)
        return original_worker(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", delayed_worker)
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    stage = LiveParseStage(max_workers=2, warm_timeout_seconds=2, shard_directory=tmp_path / "shards")
    processor = LiveBatchProcessor(
        Polylogue(archive_root=archive_root, db_path=archive_root / "index.db"),
        (WatchSource(name="inbox", root=source_root),),
        cursor=CursorStore(archive_root / "index.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
        parse_stage=stage,
        read_snapshot=open_operation_read,
    )
    try:
        result = await processor.ingest_files([pending_path, ready_path], emit_event=False)
        assert str(pending_path) in result.deferred_paths
        assert result.succeeded_file_count == 1
        with _connect(archive_root / "index.db") as conn:
            assert [row["native_id"] for row in conn.execute("SELECT native_id FROM sessions")] == ["b-0"]
        cursors = CursorStore(archive_root / "index.db")
        pending_cursor = cursors.get_record(pending_path)
        ready_cursor = cursors.get_record(ready_path)
        assert pending_cursor is None or pending_cursor.content_fingerprint is None
        assert ready_cursor is not None and ready_cursor.content_fingerprint is not None
        with _connect(archive_root / "source.db") as conn:
            assert {row["source_path"] for row in conn.execute("SELECT source_path FROM raw_sessions")} == {
                str(pending_path),
                str(ready_path),
            }
    finally:
        released.set()
        stage.shutdown()


@pytest.mark.asyncio
async def test_pending_json_worker_retains_bytes_before_source_disappears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow worker defers parsing after durable acquisition of the observed JSON."""
    import threading

    import polylogue.sources.live.parse_prefetch as parse_prefetch
    from polylogue.storage.blob_store import BlobStore

    source = tmp_path / "inbox" / "session.json"
    source.parent.mkdir()
    payload = json.dumps(
        [
            {
                "id": "retained-while-pending",
                "title": "retained while pending",
                "create_time": 1,
                "current_node": "m",
                "mapping": {
                    "m": {
                        "id": "m",
                        "parent": None,
                        "children": [],
                        "message": {
                            "id": "m",
                            "author": {"role": "user"},
                            "create_time": 1,
                            "content": {"content_type": "text", "parts": ["preserved content"]},
                        },
                    }
                },
            }
        ]
    ).encode()
    source.write_bytes(payload)
    released = threading.Event()
    original_worker = parse_prefetch.live_parse_path_worker

    def delayed_worker(*args: object, **kwargs: object) -> object:
        released.wait(timeout=30)
        return original_worker(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", delayed_worker)
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    stage = LiveParseStage(max_workers=1, warm_timeout_seconds=0.01, shard_directory=tmp_path / "shards")
    processor = LiveBatchProcessor(
        Polylogue(archive_root=archive_root, db_path=archive_root / "index.db"),
        (WatchSource(name="inbox", root=source.parent),),
        cursor=CursorStore(archive_root / "index.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
        parse_stage=stage,
        read_snapshot=open_operation_read,
    )
    try:
        result = await processor.ingest_files([source], emit_event=False)
        assert str(source) in result.deferred_paths
        source.unlink()
        blob_hash = hashlib.sha256(payload).hexdigest()
        assert BlobStore(archive_root / "blob").read_all(blob_hash) == payload
        with _connect(archive_root / "source.db") as conn:
            assert [row["origin"] for row in conn.execute("SELECT origin FROM raw_sessions")] == [
                origin_from_provider(Provider.CHATGPT).value
            ]
    finally:
        released.set()
        stage.shutdown()


@pytest.mark.asyncio
async def test_path_worker_failure_retains_raw_for_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import polylogue.sources.live.batch as batch
    import polylogue.sources.live.parse_prefetch as parse_prefetch

    paths = _write_fixture_corpus(tmp_path / "sessions", count=1)
    monkeypatch.setattr(batch, "_STREAMING_FULL_INGEST_BYTES", 1)

    def failed_worker(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("synthetic worker death")

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", failed_worker)
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    polylogue = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    cursor = CursorStore(archive_root / "index.db")
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards")
    processor = LiveBatchProcessor(
        polylogue,
        (WatchSource(name="codex", root=paths[0].parent),),
        cursor=cursor,
        parser_fingerprint=_PARSER_FINGERPRINT,
        parse_stage=stage,
        read_snapshot=open_operation_read,
    )
    try:
        metrics = await processor.ingest_files(paths, emit_event=False)
    finally:
        stage.shutdown()
    assert metrics.failed_file_count == 0
    assert metrics.deferred_file_count == 1
    with _connect(archive_root / "source.db") as conn:
        row = conn.execute("SELECT parse_error FROM raw_sessions").fetchone()
        assert row is not None
        assert row[0] is None
    with _connect(archive_root / "ops.db") as conn:
        row = conn.execute("SELECT stage, status FROM convergence_debt WHERE target_type = 'source_path'").fetchone()
        assert row is not None
        assert tuple(row) == ("live_ingest_deferred", "deferred")
    with _connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_incomplete_jsonl_uses_sealed_prefix_without_large_inline_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import polylogue.sources.live.batch as batch

    path = _write_fixture_corpus(tmp_path / "sessions", count=1)[0]
    complete_prefix = path.read_bytes()
    path.write_bytes(complete_prefix + b'{"type":"response_item","payload":{"type":"message"')
    monkeypatch.setattr(batch, "_STREAMING_FULL_INGEST_BYTES", 1)
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards")
    processor = LiveBatchProcessor(
        Polylogue(archive_root=archive_root, db_path=archive_root / "index.db"),
        (WatchSource(name="codex", root=path.parent),),
        cursor=CursorStore(archive_root / "index.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
        parse_stage=stage,
        read_snapshot=open_operation_read,
    )
    try:
        result = await processor.ingest_files([path], emit_event=False)
    finally:
        stage.shutdown()
    assert result.succeeded_file_count == 1
    cursor = processor._cursor.get_record(path)
    assert cursor is not None and cursor.byte_offset == len(complete_prefix)
    assert cursor.deferred_end_offset == path.stat().st_size
    with _connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] > 0


@pytest.mark.asyncio
async def test_pending_preparation_does_not_spend_cursor_failure_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    import polylogue.sources.live.batch as batch
    import polylogue.sources.live.cursor as cursor_module
    import polylogue.sources.live.parse_prefetch as parse_prefetch

    path = _write_fixture_corpus(tmp_path / "sessions", count=1)[0]
    monkeypatch.setattr(batch, "_STREAMING_FULL_INGEST_BYTES", 1)
    monkeypatch.setattr(cursor_module, "_FULL_CURSOR_RECONCILIATION_RETRY_DELAY_S", 0)
    released = threading.Event()
    original_worker = parse_prefetch.live_parse_path_worker

    def delayed_worker(*args: object, **kwargs: object) -> object:
        released.wait(timeout=30)
        return original_worker(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", delayed_worker)
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    polylogue = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    cursor = CursorStore(archive_root / "index.db")
    stage = LiveParseStage(max_workers=1, warm_timeout_seconds=0.01, shard_directory=tmp_path / "parse-shards")
    processor = LiveBatchProcessor(
        polylogue,
        (WatchSource(name="codex", root=path.parent),),
        cursor=cursor,
        parser_fingerprint=_PARSER_FINGERPRINT,
        parse_stage=stage,
        read_snapshot=open_operation_read,
    )
    try:
        for _ in range(6):
            metrics = await processor.ingest_files([path], emit_event=False)
            assert str(path) in metrics.deferred_paths
            state = cursor.get_record(path)
            assert state is not None and state.failure_count == 0 and not state.excluded
        with _connect(archive_root / "source.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] >= 1
        released.set()
        stage._warm_timeout_seconds = 5
        metrics = await processor.ingest_files([path], emit_event=False)
        assert metrics.succeeded_file_count == 1
    finally:
        released.set()
        stage.shutdown()


@pytest.mark.asyncio
async def test_unchanged_preparation_defer_retries_through_fair_intake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A static source retries after worker preparation defers behind the walk cursor."""
    import threading

    import polylogue.sources.live.cursor as cursor_module
    import polylogue.sources.live.parse_prefetch as parse_prefetch

    deferred_path, later_path = _write_fixture_corpus(tmp_path / "sessions", count=2)
    monkeypatch.setattr(cursor_module, "_FULL_CURSOR_RECONCILIATION_RETRY_DELAY_S", 0)
    released = threading.Event()
    original_worker = parse_prefetch.live_parse_path_worker

    def delayed_worker(*args: object, **kwargs: object) -> object:
        if str(args[1]) == str(deferred_path):
            released.wait(timeout=30)
        return original_worker(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", delayed_worker)
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    source = WatchSource(name="codex", root=deferred_path.parent)
    polylogue = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    cursor = CursorStore(archive_root / "index.db")
    stage = LiveParseStage(max_workers=2, warm_timeout_seconds=0.01, shard_directory=tmp_path / "parse-shards")
    watcher = LiveWatcher(polylogue, (source,), cursor=cursor, parse_stage=stage, read_snapshot=open_operation_read)
    adapter = FileIntakeAdapter(DaemonIntakeContext(archive_root, watcher, (source,)), source)
    dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="codex", adapter=adapter, page_size=1)])
    source_mtime = source.root.stat().st_mtime_ns
    try:
        first = await dispatcher.run_once()
        assert first.require_report("codex").deferred == 1
        assert adapter._after == str(deferred_path)
        pending = cursor.get_record(deferred_path)
        assert pending is not None and pending.next_retry_at is not None
        assert pending.content_fingerprint is None and pending.failure_count == 0

        released.set()
        stage._warm_timeout_seconds = 5
        for _ in range(6):
            await dispatcher.run_once()
            with _connect(archive_root / "index.db") as conn:
                if conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2:
                    break
        else:
            pytest.fail("deferred source and later file did not both reach the archive")
        assert source.root.stat().st_mtime_ns == source_mtime
        settled = cursor.get_record(deferred_path)
        assert settled is not None and settled.next_retry_at is None
        assert settled.content_fingerprint is not None
        later = cursor.get_record(later_path)
        assert later is not None and later.content_fingerprint is not None
    finally:
        released.set()
        stage.shutdown()


@pytest.mark.uses_real_clock("measures concurrent worker wait against the configured timeout")
def test_path_workers_share_one_wait_and_reuse_late_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    import threading
    import time

    import polylogue.sources.live.parse_prefetch as parse_prefetch

    paths = _write_fixture_corpus(tmp_path / "sessions", count=3)
    released = threading.Event()
    original_worker = parse_prefetch.live_parse_path_worker
    calls: list[object] = []

    def delayed_worker(*args: object, **kwargs: object) -> object:
        calls.append(object())
        released.wait(timeout=5)
        return original_worker(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", delayed_worker)
    stage = LiveParseStage(
        max_workers=2,
        warm_timeout_seconds=0.05,
        shard_directory=tmp_path / "parse-shards",
    )
    candidates = [(str(path), Provider.CODEX, True) for path in paths]
    try:
        started = time.monotonic()
        assert stage.warm_paths(candidates) == 3
        assert time.monotonic() - started < 0.5
        for path in paths[:2]:
            pending = stage.pop_path(str(path), blob_hash=hashlib.sha256(path.read_bytes()).hexdigest())
            assert pending is not None and pending.error == "worker preparation pending"
        capacity = stage.pop_path(str(paths[2]), blob_hash=hashlib.sha256(paths[2].read_bytes()).hexdigest())
        assert capacity is not None and capacity.error == "worker preparation capacity is busy"
        released.set()
        stage._warm_timeout_seconds = 5
        assert stage.warm_paths(candidates[:2]) == 2
        assert len(calls) == 2
        for path in paths[:2]:
            result = stage.pop_path(str(path), blob_hash=hashlib.sha256(path.read_bytes()).hexdigest())
            assert result is not None and result.error is None
            result.discard()
        assert stage.warm_paths(candidates[2:]) == 1
        assert len(calls) == 3
        result = stage.pop_path(str(paths[2]), blob_hash=hashlib.sha256(paths[2].read_bytes()).hexdigest())
        assert result is not None and result.error is None
        result.discard()
    finally:
        released.set()
        stage.shutdown()


@pytest.mark.uses_real_clock("checks that a stalled process cannot block watcher shutdown")
def test_path_stage_shutdown_terminates_stalled_process(tmp_path: Path) -> None:
    directory = tmp_path / "parse-shards"
    marker = tmp_path / "worker-started"
    stage = LiveParseStage(max_workers=1, shard_directory=directory, use_processes=True)
    future = stage._executor.submit(_stalled_process_worker, str(marker))
    stage._path_futures["stalled"] = future  # type: ignore[assignment]
    try:
        for _ in range(500):
            if marker.exists():
                break
            time.sleep(0.01)
        assert marker.exists(), "process worker did not start"
        started = time.monotonic()
        stage.shutdown()
        assert time.monotonic() - started < 5
        assert future.done()
        assert list(directory.iterdir()) == []
    finally:
        if not future.done():
            stage.shutdown()


@pytest.mark.uses_real_clock("proves a healthy process result survives the warm deadline")
def test_path_worker_timeout_preserves_late_process_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib

    import polylogue.sources.live.parse_prefetch as parse_prefetch
    from polylogue.sources.prepared_jsonl import PreparedJsonl

    path = _write_fixture_corpus(tmp_path / "sessions", count=1)[0]
    directory = tmp_path / "parse-shards"
    stage = LiveParseStage(max_workers=1, warm_timeout_seconds=0.02, shard_directory=directory, use_processes=True)
    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", _delayed_path_worker)
    candidate = (str(path), Provider.CODEX, True)
    original_verify = PreparedJsonl.verify_files
    inside_writer = False
    full_verifications = 0

    def verify_outside_writer(self: PreparedJsonl, *, full: bool) -> None:
        nonlocal full_verifications
        if full:
            assert not inside_writer, "full artifact digest ran under writer admission"
            full_verifications += 1
        original_verify(self, full=full)

    monkeypatch.setattr(PreparedJsonl, "verify_files", verify_outside_writer)
    try:
        assert stage.warm_paths([candidate]) == 1
        future = stage._path_futures[str(path)]
        future.result(timeout=10)
        inside_writer = True
        pending = stage.pop_path(str(path), blob_hash=hashlib.sha256(path.read_bytes()).hexdigest())
        inside_writer = False
        assert pending is not None and pending.deferred and pending.error == "worker preparation pending"
        assert full_verifications == 0
        stage._warm_timeout_seconds = 10
        assert stage.warm_paths([candidate]) == 1
        assert future.done()
        assert full_verifications == 1
        result = stage.pop_path(str(path), blob_hash=hashlib.sha256(path.read_bytes()).hexdigest())
        assert result is not None and result.error is None
        assert len(list(result.iter_sessions())) == 1
        result.discard()
    finally:
        stage.shutdown()
    assert list(directory.iterdir()) == []


def test_path_worker_death_is_retryable_and_shutdown_cleans_only_owned_files(tmp_path: Path) -> None:
    import hashlib

    path = _write_fixture_corpus(tmp_path / "sessions", count=1)[0]
    directory = tmp_path / "parse-shards"
    stage = LiveParseStage(max_workers=1, shard_directory=directory, use_processes=True)
    unrelated = directory / "operator-note.txt"
    unrelated.write_text("keep")
    (directory / "prepared-orphan.db").write_bytes(b"partial")
    (directory / "shard-orphan.db").write_bytes(b"partial")
    try:
        future = stage._executor.submit(_dead_process_worker)
        stage._path_futures["dead"] = future  # type: ignore[assignment]
        try:
            future.result(timeout=5)
        except Exception:
            pass
        stage.warm_paths([])
        failed = stage.pop_path("dead", blob_hash="0" * 64)
        assert failed is not None and failed.deferred
        assert "worker process died" in str(failed.error)

        candidate = (str(path), Provider.CODEX, True)
        assert stage.warm_paths([candidate]) == 1
        retry = stage.pop_path(str(path), blob_hash=hashlib.sha256(path.read_bytes()).hexdigest())
        assert retry is not None and retry.error is None
        retry.discard()
    finally:
        stage.shutdown()
    assert unrelated.read_text() == "keep"
    assert sorted(item.name for item in directory.iterdir()) == ["operator-note.txt"]


def test_path_preparation_refuses_source_changed_after_worker_seal(tmp_path: Path) -> None:
    import hashlib

    path = _write_fixture_corpus(tmp_path / "sessions", count=1)[0]
    directory = tmp_path / "parse-shards"
    stage = LiveParseStage(max_workers=1, shard_directory=directory)
    try:
        assert stage.warm_paths([(str(path), Provider.CODEX, True)]) == 1
        path.write_bytes(path.read_bytes() + b' {"type":"event_msg","payload":{}}\n')
        result = stage.pop_path(str(path), blob_hash=hashlib.sha256(path.read_bytes()).hexdigest())
        assert result is not None and result.deferred
        assert result.error == "captured source changed after preparation"
    finally:
        stage.shutdown()
    assert list(directory.iterdir()) == []


@pytest.mark.asyncio
async def test_existing_session_preparation_uses_controlled_pinned_snapshot(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    path = _write_fixture_corpus(tmp_path / "sessions", count=1)[0]
    await _ingest(archive_root, [path], parse_stage=None)
    path.write_bytes(_codex_session_bytes("session-0", (("user", "revised question"), ("assistant", "revised answer"))))
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards")
    snapshots = 0

    def controlled_read(root: Path) -> AbstractContextManager[PinnedOperationRead]:
        nonlocal snapshots
        snapshots += 1
        return open_operation_read(root)

    try:
        assert (
            stage.warm_paths(
                [(str(path), Provider.CODEX, True)], archive_root=archive_root, read_snapshot=controlled_read
            )
            == 1
        )
        prepared = stage.pop_path(str(path), blob_hash=hashlib.sha256(path.read_bytes()).hexdigest())
        assert snapshots == 1
        assert prepared is not None and not prepared.deferred
        assert len(prepared.prepared_writes) == 1
        assert prepared.prepared_writes[0].session_id == "codex-session:session-0"
        prepared.discard()
    finally:
        stage.shutdown()


def test_existing_session_preparation_defers_when_controlled_snapshot_unavailable(tmp_path: Path) -> None:
    path = _write_fixture_corpus(tmp_path / "sessions", count=1)[0]
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards")

    def unavailable(_root: Path) -> NoReturn:
        raise RuntimeError("snapshot unavailable")

    try:
        stage.warm_paths([(str(path), Provider.CODEX, True)], archive_root=tmp_path, read_snapshot=unavailable)
        result = stage.pop_path(str(path), blob_hash=hashlib.sha256(path.read_bytes()).hexdigest())
        assert result is not None and result.deferred
        assert result.error == "read-only preparation snapshot unavailable: RuntimeError"
    finally:
        stage.shutdown()


@pytest.mark.asyncio
async def test_a_shard_the_writer_refuses_still_writes_the_session(tmp_path: Path) -> None:
    """A truncated shard is a miss, not a failure: the rows get built inline."""
    baseline_root = tmp_path / "baseline"
    corrupt_root = tmp_path / "corrupt"
    paths = _write_fixture_corpus(tmp_path / "sessions", count=3)

    await _ingest(baseline_root, paths, parse_stage=None)

    shard_directory = tmp_path / "parse-shards"
    stage = LiveParseStage(max_workers=2, max_inflight_bytes=10_000_000, shard_directory=shard_directory)
    try:
        candidates = _live_parse_stage_candidates(paths, fallback_provider=Provider.CODEX)
        assert stage.warm(candidates) == len(paths)
        for shard_path in shard_directory.glob("shard-*"):
            shard_path.write_bytes(b"not a database at all")
        await _ingest(corrupt_root, paths, parse_stage=stage)
    finally:
        stage.shutdown()

    assert _canonical_snapshot(baseline_root) == _canonical_snapshot(corrupt_root)


@pytest.mark.asyncio
async def test_parse_stage_out_of_order_completion_preserves_archive_write_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parallel parse completion order must never leak into archive write order.

    The writer-held loop in ``_ingest_full_paths_sync``/
    ``_ingest_full_records_archive`` iterates the ORIGINAL candidate list in
    submission order and looks up each path's cache entry synchronously --
    it never iterates in whatever order ``ThreadPoolExecutor.as_completed``
    happened to finish. This test forces the SLOWEST-to-parse file to be the
    FIRST one submitted (so completion order is the exact reverse of
    submission order) and asserts the resulting archive is still identical
    to, and inserted in the same order as, a flag-off baseline.
    """
    import polylogue.sources.live.parse_prefetch as parse_prefetch_module

    baseline_root = tmp_path / "baseline"
    prefetch_root = tmp_path / "prefetch"
    paths = _write_fixture_corpus(tmp_path / "sessions", count=5)

    await _ingest(baseline_root, paths, parse_stage=None)

    real_worker = parse_prefetch_module.live_parse_worker
    # Reverse the completion order relative to submission: the FIRST
    # candidate (session-0) sleeps longest, the LAST (session-4) returns
    # immediately.
    sleep_by_source_path = {str(path): 0.05 * (len(paths) - index) for index, path in enumerate(paths)}

    def delayed_worker(*args: object, **kwargs: object) -> object:
        source_path = str(args[3])
        time.sleep(sleep_by_source_path.get(source_path, 0.0))
        return real_worker(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(parse_prefetch_module, "live_parse_worker", delayed_worker)

    stage = LiveParseStage(max_workers=len(paths), max_inflight_bytes=10_000_000)
    try:
        await _ingest(prefetch_root, paths, parse_stage=stage)
    finally:
        stage.shutdown()
    assert len(stage.cache) == 0

    assert _canonical_snapshot(baseline_root) == _canonical_snapshot(prefetch_root)
    assert _raw_sessions_source_path_order(prefetch_root) == _raw_sessions_source_path_order(baseline_root)
    assert _raw_sessions_source_path_order(prefetch_root) == tuple(str(path) for path in paths)


@pytest.mark.asyncio
async def test_unknown_mixed_jsonl_prefetch_falls_back_to_strict_decode(tmp_path: Path) -> None:
    """Strict prefetch refuses a partial unknown-provider conversation."""
    root = tmp_path / "unknown"
    root.mkdir()
    path = root / "mixed.jsonl"
    path.write_bytes(b'{"id":"unknown-1","messages":[{"id":"m1","role":"user","content":"hello"}]}\n{"broken":}\n')
    candidates = _live_parse_stage_candidates([path], fallback_provider=Provider.UNKNOWN)
    assert len(candidates) == 1
    assert candidates[0].provider is Provider.UNKNOWN
    assert candidates[0].is_stream is False
    polylogue = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    stage = LiveParseStage(max_workers=1, max_inflight_bytes=10_000_000)
    processor = LiveBatchProcessor(
        polylogue,
        (WatchSource(name="unknown", root=root),),
        cursor=CursorStore(tmp_path / "index.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
        parse_stage=stage,
    )
    try:
        result = await processor.ingest_files([path], emit_event=False)
    finally:
        stage.shutdown()

    assert result.failed_file_count == 0
    assert result.succeeded_file_count == 1
    assert len(stage.cache) == 0
    with _connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    with _connect(tmp_path / "source.db") as conn:
        artifact = conn.execute("SELECT artifact_kind, support_status FROM raw_artifacts").fetchone()
        # A malformed record anywhere in an unknown JSONL payload is a
        # terminal decode verdict (same contract as the fully-malformed
        # cases in test_live_batch_support): the strict fallback refuses to
        # parse a partial conversation out of the valid prefix.
        assert (artifact[0], artifact[1]) == ("terminal_unknown_json_decode", "decode_failed")
    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.terminal == 1
    assert lifecycle.unexplained == 0


@pytest.mark.uses_real_clock("a real ThreadPoolExecutor worker must outlast warm()'s real timeout")
def test_a_timed_out_prefetch_worker_discards_its_own_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """polylogue-nfr2u: a worker abandoned by warm()'s timeout cleans up its shard.

    Anti-vacuity: the sealed shard path is captured from the worker itself, so
    the assertion cannot pass by observing the directory before the worker got
    there. A shard file surviving a forced timeout is the red condition --
    remove the done-callback the timeout path attaches and this shard stays on
    disk until ``shutdown()`` sweeps it, which runs only after the assertion.
    """
    from polylogue.sources.live import parse_prefetch

    paths = _write_fixture_corpus(tmp_path / "sessions", count=1)
    shard_directory = tmp_path / "parse-shards"
    real_worker = parse_prefetch.live_parse_and_shard_worker
    sealed: list[Path] = []

    def slow_worker(*args: Any, **kwargs: Any) -> Any:
        # Outlast warm()'s timeout, then seal a real shard exactly as production does.
        time.sleep(0.5)
        result = real_worker(*args, **kwargs)
        shard_name = result[3]
        assert shard_name is not None
        sealed.append(Path(shard_name))
        return result

    monkeypatch.setattr(parse_prefetch, "live_parse_and_shard_worker", slow_worker)
    stage = LiveParseStage(
        max_workers=1, max_inflight_bytes=10_000_000, shard_directory=shard_directory, warm_timeout_seconds=0.05
    )
    try:
        candidates = _live_parse_stage_candidates(paths, fallback_provider=Provider.CODEX)
        # The timeout fires before the worker finishes: nothing is cached.
        assert stage.warm(candidates) == 0

        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and not sealed:
            time.sleep(0.02)
        assert sealed, "the abandoned worker never sealed a shard; the test would be vacuous"
        orphan = sealed[0]
        while time.monotonic() < deadline and orphan.exists():
            time.sleep(0.02)
        assert not orphan.exists(), f"timed-out worker orphaned its shard: {orphan}"
    finally:
        stage.shutdown()


def test_shard_build_failure_is_counted_not_only_logged_per_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A systematic shard-build failure must be countable, not per-file noise.

    Every failed shard build falls back to per-row binding, which is correct
    but silently removes the polylogue-bp12n.6 benefit; only a counter makes a
    systematic failure distinguishable from an occasional one
    (polylogue-3r36h). Restoring the count-free fallback (dropping
    ``shard_build_failure_count``) turns this red.
    """
    import polylogue.sources.live.parse_prefetch as parse_prefetch

    def refuse_shard(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("shard build refused")

    monkeypatch.setattr(parse_prefetch, "prepare_session_shard", refuse_shard)

    paths = _write_fixture_corpus(tmp_path / "sessions", count=3)
    stage = LiveParseStage(max_workers=2, max_inflight_bytes=10_000_000, shard_directory=tmp_path / "parse-shards")
    try:
        candidates = _live_parse_stage_candidates(paths, fallback_provider=Provider.CODEX)
        assert stage.warm(candidates) == len(paths)
    finally:
        stage.shutdown()

    assert stage.shard_build_failure_count == len(paths)


def test_warm_timeout_cancels_the_workers_it_stopped_waiting_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A timed-out warm() must not leave its unstarted work queued.

    ``warm()`` used to return on timeout without cancelling or draining its
    futures, so unstarted workers still ran later against a cache nobody was
    waiting for (polylogue-3r36h). Removing the cancellation turns this red:
    the log reports ``cancelled 0 unstarted worker(s)``.
    """
    import threading

    import polylogue.sources.live.parse_prefetch as parse_prefetch

    release = threading.Event()
    real_worker = parse_prefetch.live_parse_and_shard_worker

    def blocking_worker(*args: object, **kwargs: object) -> object:
        release.wait(30.0)
        return real_worker(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(parse_prefetch, "live_parse_and_shard_worker", blocking_worker)

    paths = _write_fixture_corpus(tmp_path / "sessions", count=3)
    stage = LiveParseStage(max_workers=1, max_inflight_bytes=10_000_000, warm_timeout_seconds=0.2)
    try:
        candidates = _live_parse_stage_candidates(paths, fallback_provider=Provider.CODEX)
        with caplog.at_level("WARNING"):
            assert stage.warm(candidates) == 0
    finally:
        release.set()
        stage.shutdown()

    assert "cancelled 2 unstarted worker(s)" in caplog.text
