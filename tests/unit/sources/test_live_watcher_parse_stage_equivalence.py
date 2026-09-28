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
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
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
from polylogue.storage.sqlite.archive_tiers.source_write import deterministic_raw_session_id

_VOLATILE_COLUMNS: dict[str, frozenset[str]] = {
    "raw_sessions": frozenset({"acquired_at_ms", "parsed_at_ms"}),
}


def test_live_path_worker_detects_grok_without_whole_object_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.sources.live.parse_prefetch import live_parse_path_worker

    source = tmp_path / "prod-grok-backend.json"
    source.write_text(
        json.dumps(
            {
                "conversations": [
                    {
                        "conversation": {"title": "Neutral"},
                        "responses": [{"sender": "human", "message": f"Prompt {index}"} for index in range(200)],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def refuse_whole_document(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("live Grok provider detection decoded the whole object")

    monkeypatch.setattr("polylogue.sources.decoders._iter_json_stream", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl._iter_json_stream", refuse_whole_document)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.parse_payload", refuse_whole_document)
    artifact = live_parse_path_worker(
        Provider.UNKNOWN.value,
        str(source),
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    assert artifact.resolved_provider is Provider.GROK
    [session] = artifact.iter_sessions()
    assert len(session.messages) == 200
    artifact.discard()


def _stalled_process_worker(marker: str) -> None:
    Path(marker).write_text("started")
    time.sleep(30)


def _dead_process_worker() -> None:
    import os

    os._exit(7)


def _attempt_death_path_worker(
    provider_value: str,
    source_path: str,
    fallback_id: str,
    *,
    is_stream: bool,
    shard_directory: str,
    attempt_directory: str | None = None,
) -> object:
    if Path(source_path).name.startswith("killed-"):
        import os

        assert attempt_directory is not None
        attempt = Path(attempt_directory)
        (attempt / "prepared-partial.db").write_bytes(b"partial prepared database")
        (attempt / "prepared-partial.db-journal").write_bytes(b"partial journal")
        (attempt / "shard-partial.db").write_bytes(b"partial shard")
        os._exit(7)
    from polylogue.sources.live.parse_prefetch import live_parse_path_worker

    return live_parse_path_worker(
        provider_value,
        source_path,
        fallback_id,
        is_stream=is_stream,
        shard_directory=shard_directory,
        attempt_directory=attempt_directory,
    )


def _delayed_path_worker(
    provider_value: str,
    source_path: str,
    fallback_id: str,
    *,
    is_stream: bool,
    shard_directory: str,
    attempt_directory: str | None = None,
) -> object:
    from polylogue.sources.live.parse_prefetch import live_parse_path_worker

    time.sleep(0.25)
    return live_parse_path_worker(
        provider_value,
        source_path,
        fallback_id,
        is_stream=is_stream,
        shard_directory=shard_directory,
        attempt_directory=attempt_directory,
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


def _chatgpt_bundle_bytes(*texts: str) -> bytes:
    return json.dumps(
        [
            {
                "id": f"bundle-{index}",
                "title": f"session {index}",
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
                            "content": {"content_type": "text", "parts": [text]},
                        },
                    }
                },
            }
            for index, text in enumerate(texts)
        ]
    ).encode()


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


async def _ingest(
    archive_root: Path,
    paths: list[Path],
    *,
    parse_stage: LiveParseStage | None,
    source_name: str = "codex",
) -> None:
    archive_root.mkdir(parents=True, exist_ok=True)
    db_path = archive_root / "index.db"
    polylogue = Polylogue(archive_root=archive_root, db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        polylogue,
        (WatchSource(name=source_name, root=paths[0].parent),),
        cursor=cursor,
        parser_fingerprint=_PARSER_FINGERPRINT,
        parse_stage=parse_stage,
        read_snapshot=open_operation_read,
    )
    metrics = await processor.ingest_files(paths, emit_event=False)
    assert metrics.failed_file_count == 0
    assert metrics.succeeded_file_count == len(paths)


@pytest.mark.asyncio
async def test_large_hermes_snapshot_uses_bounded_live_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A supported Hermes snapshot stays admitted and prepared above 64 MiB.

    The padding is an irrelevant single scalar: the snapshot envelope reader
    must skip it without materializing the containing JSON document.
    """
    from polylogue.sources.live import batch as live_batch

    root = tmp_path / "hermes"
    root.mkdir()
    source = root / "session.json"

    def write_snapshot(*, large: bool, prompt: str = "A neutral prompt") -> None:
        with source.open("wb") as handle:
            handle.write(
                (
                    '{"session_id":"large-hermes","platform":"linux",'
                    f'"messages":[{{"role":"user","content":{json.dumps(prompt)}}}]'
                ).encode()
            )
            if large:
                handle.write(b',"padding":"')
                chunk = b"x" * (1024 * 1024)
                for _ in range(65):
                    handle.write(chunk)
                handle.write(b'"')
            handle.write(b"}")

    write_snapshot(large=False)
    archive_root = tmp_path / "archive"
    db_path = archive_root / "index.db"
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards")
    processor = LiveBatchProcessor(
        Polylogue(archive_root=archive_root, db_path=db_path),
        (WatchSource(name="hermes", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint=_PARSER_FINGERPRINT,
        parse_stage=stage,
        read_snapshot=open_operation_read,
    )
    try:
        initial = await processor.ingest_files([source], emit_event=False, defer_convergence=True)
        assert initial.succeeded_file_count == 1

        def refuse_whole_document(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("large Hermes detection decoded the whole JSON document")

        monkeypatch.setattr("polylogue.sources.decoders._iter_json_stream", refuse_whole_document)
        monkeypatch.setattr(live_batch, "_iter_json_stream", refuse_whole_document)
        write_snapshot(large=True)
        assert source.stat().st_size > 64 * 1024 * 1024
        revised = await processor.ingest_files([source], emit_event=False, defer_convergence=True)
        assert revised.succeeded_file_count == 1
    finally:
        stage.shutdown()


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
async def test_a_slow_json_preparation_is_awaited_not_deferred(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    # The slow preparation finishes well after the stall window; the warm
    # waits for it instead of deferring the file (polylogue-slc55).
    threading.Timer(0.5, released.set).start()

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", delayed_worker)
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    stage = LiveParseStage(max_workers=2, stall_report_seconds=0.05, shard_directory=tmp_path / "shards")
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
        assert not result.deferred_paths
        assert result.succeeded_file_count == 2
        cursors = CursorStore(archive_root / "index.db")
        for path in (pending_path, ready_path):
            cursor = cursors.get_record(path)
            assert cursor is not None and cursor.content_fingerprint is not None
        with _connect(archive_root / "source.db") as conn:
            assert {row["source_path"] for row in conn.execute("SELECT source_path FROM raw_sessions")} == {
                str(pending_path),
                str(ready_path),
            }
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
async def test_deferred_preparation_does_not_spend_cursor_failure_budget(
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

    def failing_worker(*args: object, **kwargs: object) -> object:
        # A retryable worker failure (e.g. worker death) defers the file.
        if not released.is_set():
            raise RuntimeError("synthetic worker failure")
        return original_worker(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", failing_worker)
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    polylogue = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    cursor = CursorStore(archive_root / "index.db")
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards")
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

    def failing_worker(*args: object, **kwargs: object) -> object:
        if str(args[1]) == str(deferred_path) and not released.is_set():
            raise RuntimeError("synthetic worker failure")
        return original_worker(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", failing_worker)
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    source = WatchSource(name="codex", root=deferred_path.parent)
    polylogue = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    cursor = CursorStore(archive_root / "index.db")
    stage = LiveParseStage(max_workers=2, shard_directory=tmp_path / "parse-shards")
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


@pytest.mark.asyncio
async def test_fresh_discovery_keeps_a_page_of_lookahead_for_prefetch(tmp_path: Path) -> None:
    """Discovery keeps the next page pending so its parsing can overlap this one.

    Anti-vacuity: stop the walk once ``limit`` paths are pending and the
    pending list is exactly the offered page, so the intake prefetch has no
    path beyond the batch it is about to warm; drop the refill on the
    carried-over branch and the second page has no lookahead either.
    """
    paths = _write_fixture_corpus(tmp_path / "sessions", count=6)
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    source = WatchSource(name="codex", root=paths[0].parent)
    polylogue = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    watcher = LiveWatcher(polylogue, (source,), cursor=CursorStore(archive_root / "index.db"))
    adapter = FileIntakeAdapter(DaemonIntakeContext(archive_root, watcher, (source,)), source)

    first = [Path(str(item.payload)) for item in await adapter.discover(limit=2)]
    assert len(first) == 2
    assert len(adapter._fresh_pending) == 4
    assert set(adapter._fresh_pending) - set(first)

    # The first page was attempted; the next discovery offers the lookahead
    # and refills a page behind it.
    adapter._fresh_attempted_paths = set(first)
    second = [Path(str(item.payload)) for item in await adapter.discover(limit=2)]
    assert len(second) == 2 and not set(second) & set(first)
    assert len(adapter._fresh_pending) == 4
    assert set(adapter._fresh_pending) - set(second)


@pytest.mark.asyncio
async def test_lookahead_sampling_never_blocks_the_batch_in_flight(tmp_path: Path) -> None:
    """A lookahead whose source read is stuck is skipped, not waited on or stacked.

    Anti-vacuity: await the lookahead sample inside ``_submit_ready_lookahead``
    (or sample on the admission path) and the first submission blocks on the
    held selection instead of returning with nothing submitted; start a
    sampler per offer and the second offer runs the selection again while the
    first is still stuck.
    """
    (path,) = _write_fixture_corpus(tmp_path / "sessions", count=1)
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    submitted: list[list[tuple[str, Provider, bool]]] = []

    class RecordingStage:
        def prefetch_paths(self, candidates: list[tuple[str, Provider, bool]]) -> int:
            submitted.append(list(candidates))
            return len(candidates)

    processor = LiveBatchProcessor(
        Polylogue(archive_root=archive_root, db_path=archive_root / "index.db"),
        (WatchSource(name="codex", root=path.parent),),
        cursor=CursorStore(archive_root / "index.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
        parse_stage=RecordingStage(),  # type: ignore[arg-type]
        read_snapshot=open_operation_read,
    )
    released = threading.Event()
    selections: list[int] = []

    def held_selection() -> list[Path]:
        selections.append(1)
        released.wait(timeout=30)
        return [path]

    processor.offer_parse_lookahead(held_selection, source_name="codex")
    stuck = processor._parse_lookahead
    try:
        await processor._submit_ready_lookahead()
        assert submitted == []
        processor.offer_parse_lookahead(held_selection, source_name="codex")
        assert processor._parse_lookahead is None
    finally:
        released.set()
    assert stuck is not None
    stuck.result(timeout=30)
    assert len(selections) == 1
    processor.offer_parse_lookahead(lambda: [path], source_name="codex")
    lookahead = processor._parse_lookahead
    assert lookahead is not None
    lookahead.result(timeout=30)
    await processor._submit_ready_lookahead()
    assert [[candidate[0] for candidate in batch] for batch in submitted] == [[str(path)]]
    assert processor._parse_lookahead is None


def test_a_cancelled_warm_stops_waiting_and_installs_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancellation ends an unbounded warm promptly and leaves no stale install.

    Anti-vacuity: ignore ``cancelled`` in ``_warm_until`` and the warm waits
    for the held worker, so the thread is still alive after the join; skip
    the post-wait check and the warm installs prepared writes from its
    abandoned snapshot.
    """
    import polylogue.sources.live.parse_prefetch as parse_prefetch

    (path,) = _write_fixture_corpus(tmp_path / "sessions", count=1)
    released = threading.Event()
    started = threading.Event()
    original_worker = parse_prefetch.live_parse_path_worker

    def held_worker(provider_value: str, source_path: str, *args: object, **kwargs: object) -> object:
        started.set()
        released.wait(timeout=30)
        return original_worker(provider_value, source_path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", held_worker)
    monkeypatch.setattr(parse_prefetch, "_PROGRESS_POLL_SECONDS", 0.05)
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards")
    installs: list[object] = []
    monkeypatch.setattr(stage, "_prepare_existing_session_writes", lambda *a, **k: installs.append(k))
    cancelled = threading.Event()
    warm = threading.Thread(
        target=stage.warm_paths,
        args=([(str(path), Provider.CODEX, True)],),
        kwargs={"archive_root": tmp_path / "archive", "cancelled": cancelled},
    )
    try:
        warm.start()
        assert started.wait(timeout=10)
        assert str(path) in stage._path_futures
        cancelled.set()
        warm.join(timeout=5)
        assert not warm.is_alive()
        assert installs == []
        assert str(path) not in stage._path_results
    finally:
        released.set()
        warm.join(timeout=30)
        stage.shutdown()


def test_prefetch_submits_without_waiting_and_warm_claims_the_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prefetched path is parsed once: the later warm waits on the same work.

    Anti-vacuity: if ``prefetch_paths`` blocked, the call would not return
    while the worker is held; if ``warm_paths`` resubmitted, the worker would
    run twice.
    """
    import hashlib
    import threading

    import polylogue.sources.live.parse_prefetch as parse_prefetch

    paths = _write_fixture_corpus(tmp_path / "sessions", count=3)
    released = threading.Event()
    original_worker = parse_prefetch.live_parse_path_worker
    calls: list[str] = []

    def held_worker(provider_value: str, source_path: str, *args: object, **kwargs: object) -> object:
        calls.append(source_path)
        released.wait(timeout=5)
        return original_worker(provider_value, source_path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", held_worker)
    stage = LiveParseStage(max_workers=3, shard_directory=tmp_path / "parse-shards")
    candidates = [(str(path), Provider.CODEX, True) for path in paths]
    try:
        # Read-ahead leaves the last worker for required work.
        assert stage.prefetch_paths(candidates) == 2
        # Returned while both workers are still held: nothing waited on them.
        assert len(stage._path_futures) == 2
        assert not any(future.done() for future in stage._path_futures.values())
        released.set()
        assert stage.warm_paths(candidates) == 3
        assert sorted(calls) == sorted(str(path) for path in paths)
        for path in paths:
            result = stage.pop_path(str(path), blob_hash=hashlib.sha256(path.read_bytes()).hexdigest())
            assert result is not None and result.error is None
    finally:
        released.set()
        stage.shutdown()


def test_unclaimed_prefetch_is_dropped_with_its_scratch(tmp_path: Path) -> None:
    """A prefetched path no warm claims is discarded after its lifetime.

    Anti-vacuity: without the lifetime, the unclaimed result and its sealed
    attempt directory stay held until shutdown, so the result is still in
    ``_path_results`` and an attempt directory still exists.
    """
    import polylogue.sources.live.parse_prefetch as parse_prefetch

    skipped, claimed = _write_fixture_corpus(tmp_path / "sessions", count=2)
    stage = LiveParseStage(max_workers=2, shard_directory=tmp_path / "parse-shards")
    try:
        assert stage.prefetch_paths([(str(skipped), Provider.CODEX, True)]) == 1
        for _ in range(parse_prefetch._SPECULATIVE_LIFETIME_CALLS):
            stage.warm_paths([(str(claimed), Provider.CODEX, True)])
            stage.pop_path(str(claimed), blob_hash="")
        assert str(skipped) not in stage._path_results
        assert str(skipped) not in stage._path_futures
        attempts = tmp_path / "parse-shards" / ".live-parse-attempts"
        assert [path for path in attempts.iterdir() if path.name.startswith("attempt-")] == []
    finally:
        stage.shutdown()


def test_a_prefetch_only_walk_does_not_accumulate_results(tmp_path: Path) -> None:
    """Every file of a walk may be skipped after cursor reconciliation, so no
    warm ever runs. Anti-vacuity: aging speculation only on warms keeps every
    prefetched result (and its sealed scratch) until shutdown."""
    import polylogue.sources.live.parse_prefetch as parse_prefetch

    paths = _write_fixture_corpus(tmp_path / "sessions", count=parse_prefetch._SPECULATIVE_LIFETIME_CALLS + 4)
    stage = LiveParseStage(max_workers=2, shard_directory=tmp_path / "parse-shards")
    try:
        for path in paths:
            stage.prefetch_paths([(str(path), Provider.CODEX, True)])
            for future in tuple(stage._path_futures.values()):
                future.result(timeout=30)
        held = len(stage._path_results) + len(stage._path_futures)
        assert held <= parse_prefetch._SPECULATIVE_LIFETIME_CALLS
    finally:
        stage.shutdown()


def test_a_finished_unclaimed_prefetch_expires_without_a_warm(tmp_path: Path) -> None:
    """A read-ahead that finished but was never claimed is dropped on expiry.

    Anti-vacuity: keep skipping every path still in ``_path_futures`` in
    ``_drop_stale_speculation`` and the finished future, with its sealed
    attempt directory, stays held because read-ahead calls never collect.
    """
    import polylogue.sources.live.parse_prefetch as parse_prefetch

    (skipped,) = _write_fixture_corpus(tmp_path / "sessions", count=1)
    stage = LiveParseStage(max_workers=2, shard_directory=tmp_path / "parse-shards")
    try:
        assert stage.prefetch_paths([(str(skipped), Provider.CODEX, True)]) == 1
        stage._path_futures[str(skipped)].result(timeout=30)
        for _ in range(parse_prefetch._SPECULATIVE_LIFETIME_CALLS):
            stage.prefetch_paths([])
        assert str(skipped) not in stage._path_futures
        assert str(skipped) not in stage._path_results
        assert str(skipped) not in stage._speculative
        attempts = tmp_path / "parse-shards" / ".live-parse-attempts"
        assert [path for path in attempts.iterdir() if path.name.startswith("attempt-")] == []
    finally:
        stage.shutdown()


def test_a_speculative_failure_finishing_during_the_warm_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: dropping the speculative marker before the running
    read-ahead finishes hands the warm its retryable failure."""
    import hashlib
    import threading

    import polylogue.sources.live.parse_prefetch as parse_prefetch
    from polylogue.sources.prepared_jsonl import PreparedJsonl

    [path] = _write_fixture_corpus(tmp_path / "sessions", count=1)
    released = threading.Event()
    original_worker = parse_prefetch.live_parse_path_worker
    calls = 0

    def flaky_worker(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            released.wait(timeout=5)
            return PreparedJsonl(None, None, None, "source changed", deferred=True)
        return original_worker(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", flaky_worker)
    stage = LiveParseStage(max_workers=2, shard_directory=tmp_path / "parse-shards")
    candidate = [(str(path), Provider.CODEX, True)]
    try:
        assert stage.prefetch_paths(candidate) == 1
        threading.Timer(0.05, released.set).start()
        stage.warm_paths(candidate)
        result = stage.pop_path(str(path), blob_hash=hashlib.sha256(path.read_bytes()).hexdigest())
        assert result is not None and result.error is None
        assert calls == 2
    finally:
        released.set()
        stage.shutdown()


def test_a_single_worker_stage_reads_nothing_ahead(tmp_path: Path) -> None:
    """Anti-vacuity: read-ahead on the only worker leaves required work
    queued behind a parse nobody may claim."""
    [path] = _write_fixture_corpus(tmp_path / "sessions", count=1)
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards")
    try:
        assert stage.prefetch_paths([(str(path), Provider.CODEX, True)]) == 0
        assert stage._path_futures == {}
    finally:
        stage.shutdown()


def test_a_failed_prefetch_stat_is_retried_by_the_warm(tmp_path: Path) -> None:
    """Anti-vacuity: caching the speculative stat failure makes the warm
    return that failure for a file that exists by the time it is needed."""
    import hashlib

    [path] = _write_fixture_corpus(tmp_path / "sessions", count=1)
    payload = path.read_bytes()
    path.unlink()
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards")
    try:
        assert stage.prefetch_paths([(str(path), Provider.CODEX, True)]) == 0
        path.write_bytes(payload)
        stage.warm_paths([(str(path), Provider.CODEX, True)])
        result = stage.pop_path(str(path), blob_hash=hashlib.sha256(payload).hexdigest())
        assert result is not None and result.error is None
    finally:
        stage.shutdown()


def test_default_process_pool_is_sized_for_the_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: an uncapped default gives the process pool every core."""
    import polylogue.sources.live.parse_prefetch as parse_prefetch

    monkeypatch.setattr(parse_prefetch, "live_watcher_parse_stage_worker_count", lambda: 23)
    monkeypatch.setattr(parse_prefetch, "_parse_stage_workers_configured", lambda: False)
    stage = LiveParseStage(shard_directory=tmp_path / "parse-shards", use_processes=True)
    try:
        assert stage._worker_count == parse_prefetch._DEFAULT_PROCESS_WORKER_CAP
        assert stage._max_path_pending == stage._worker_count
    finally:
        stage.shutdown()


@pytest.mark.uses_real_clock("workers outlast the stall window, which only reports")
def test_a_warm_waits_for_every_path_through_limited_capacity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Three paths through two workers all come back prepared from one warm.

    Anti-vacuity: a warm that stops at the stall window returns "pending" for
    the running paths and "capacity is busy" for the third (the livelock of
    polylogue-slc55: each later pass re-acquired and deferred the file).
    """
    import hashlib
    import threading

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
    monkeypatch.setattr(parse_prefetch, "_PROGRESS_POLL_SECONDS", 0.02)
    stage = LiveParseStage(max_workers=2, stall_report_seconds=0.05, shard_directory=tmp_path / "parse-shards")
    candidates = [(str(path), Provider.CODEX, True) for path in paths]
    try:
        threading.Timer(0.3, released.set).start()
        assert stage.warm_paths(candidates) == 3
        assert len(calls) == 3
        for path in paths:
            result = stage.pop_path(str(path), blob_hash=hashlib.sha256(path.read_bytes()).hexdigest())
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


@pytest.mark.uses_real_clock("a process worker outlasts the stall window, which only reports")
def test_a_slow_process_worker_is_awaited_and_verified_off_the_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The warm waits for a slow process worker and does its full digest there.

    Anti-vacuity: a deadline returns "worker preparation pending" to the
    writer, which then defers the file and verifies nothing.
    """
    import hashlib

    import polylogue.sources.live.parse_prefetch as parse_prefetch
    from polylogue.sources.prepared_jsonl import PreparedJsonl

    path = _write_fixture_corpus(tmp_path / "sessions", count=1)[0]
    directory = tmp_path / "parse-shards"
    monkeypatch.setattr(parse_prefetch, "_PROGRESS_POLL_SECONDS", 0.02)
    stage = LiveParseStage(max_workers=1, stall_report_seconds=0.02, shard_directory=directory, use_processes=True)
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
        assert stage._path_futures == {}
        assert full_verifications == 1
        inside_writer = True
        result = stage.pop_path(str(path), blob_hash=hashlib.sha256(path.read_bytes()).hexdigest())
        inside_writer = False
        assert result is not None and result.error is None
        assert len(list(result.iter_sessions())) == 1
        result.discard()
    finally:
        stage.shutdown()
    assert list(directory.iterdir()) == []


def test_path_stage_shutdown_is_idempotent(tmp_path: Path) -> None:
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards")

    stage.shutdown()
    stage.shutdown()

    assert not (tmp_path / "parse-shards" / ".live-parse-attempts").exists()


def test_path_worker_death_reclaims_only_its_attempt_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    import polylogue.sources.live.parse_prefetch as parse_prefetch

    sibling_path, retry_path = _write_fixture_corpus(tmp_path / "sessions", count=2)
    directory = tmp_path / "parse-shards"
    attempt_root = directory / ".live-parse-attempts"
    stale_attempt = attempt_root / "attempt-left-by-previous-process"
    stale_attempt.mkdir(parents=True)
    (stale_attempt / "prepared-partial.db").write_bytes(b"stale partial")
    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", _attempt_death_path_worker)
    stage = LiveParseStage(max_workers=1, shard_directory=directory, use_processes=True)
    unrelated = directory / "operator-note.txt"
    unrelated.write_text("keep")
    assert not stale_attempt.exists(), "startup did not reclaim a stale parent-owned attempt directory"
    try:
        assert stage.warm_paths([(str(sibling_path), Provider.CODEX, True)]) == 1
        sibling = stage._path_results[str(sibling_path)]
        assert sibling.attempt_directory is not None and sibling.attempt_directory.exists()

        for index in range(3):
            killed = tmp_path / f"killed-{index}.json"
            killed.write_text("worker exits before parsing")
            candidate = (str(killed), Provider.CODEX, True)
            assert stage.warm_paths([candidate]) == 1
            failed = stage.pop_path(str(killed), blob_hash="0" * 64)
            assert failed is not None and failed.deferred
            assert "worker process died" in str(failed.error)
            attempts = tuple(attempt_root.iterdir())
            assert attempts == (sibling.attempt_directory,)
            assert not any(any(attempt.glob("*partial*")) for attempt in attempts)

        assert stage.warm_paths([(str(retry_path), Provider.CODEX, True)]) == 1
        retry = stage.pop_path(str(retry_path), blob_hash=hashlib.sha256(retry_path.read_bytes()).hexdigest())
        assert retry is not None and retry.error is None
        assert sibling.attempt_directory.exists(), "another attempt failure swept the pending sibling carrier"
        retry.discard()
        published_sibling = stage.pop_path(
            str(sibling_path), blob_hash=hashlib.sha256(sibling_path.read_bytes()).hexdigest()
        )
        assert published_sibling is sibling
        published_sibling.discard()
        assert tuple(attempt_root.iterdir()) == ()
    finally:
        stage.shutdown()
    assert unrelated.read_text() == "keep"
    assert sorted(item.name for item in directory.iterdir()) == ["operator-note.txt"]


def test_path_worker_stop_unknown_blocks_cleanup_and_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import polylogue.pipeline.services.process_pool as process_pool
    import polylogue.sources.live.parse_prefetch as parse_prefetch

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", _attempt_death_path_worker)
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards", use_processes=True)
    attempt_root = tmp_path / "parse-shards" / ".live-parse-attempts"
    killed = tmp_path / "killed-unknown-stop.json"
    killed.write_text("worker exits before parsing")
    try:
        with monkeypatch.context() as context:
            context.setattr(process_pool, "terminate_process_pool", lambda _executor: False)
            assert stage.warm_paths([(str(killed), Provider.CODEX, True)]) == 1
            assert stage._cleanup_blocked
            assert stage.cleanup_failure_count == 1
            failed = stage.pop_path(str(killed), blob_hash="0" * 64)
            assert failed is not None and failed.deferred
            residue = tuple(attempt_root.iterdir())
            assert len(residue) == 1
            assert sorted(path.name for path in residue[0].iterdir()) == [
                "prepared-partial.db",
                "prepared-partial.db-journal",
                "shard-partial.db",
            ]

            retry = tmp_path / "killed-retry-must-not-start.json"
            retry.write_text("must remain unsubmitted")
            assert stage.warm_paths([(str(retry), Provider.CODEX, True)]) == 1
            deferred = stage.pop_path(str(retry), blob_hash="0" * 64)
            assert deferred is not None and deferred.deferred
            assert not stage._path_futures
            assert tuple(attempt_root.iterdir()) == residue

            stage.shutdown()
            assert tuple(attempt_root.iterdir()) == residue
    finally:
        stage.shutdown()


@pytest.mark.asyncio
async def test_killed_path_worker_retains_one_raw_and_retries_through_intake(tmp_path: Path) -> None:
    path = _write_fixture_corpus(tmp_path / "sessions", count=1)[0]
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards", use_processes=True)
    processor = LiveBatchProcessor(
        Polylogue(archive_root=archive_root, db_path=archive_root / "index.db"),
        (WatchSource(name="codex", root=path.parent),),
        cursor=CursorStore(archive_root / "index.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
        parse_stage=stage,
        read_snapshot=open_operation_read,
    )
    try:
        future = stage._executor.submit(_dead_process_worker)
        stage._path_futures[str(path)] = future  # type: ignore[assignment]
        stage._path_sizes[str(path)] = path.stat().st_size
        stage._path_inflight_bytes = path.stat().st_size
        with pytest.raises(BrokenProcessPool):
            future.result(timeout=15)

        first = await processor.ingest_files([path], emit_event=False)
        assert first.deferred_file_count == 1
        assert first.failed_file_count == 0
        assert stage._path_inflight_bytes == 0
        assert not stage._path_futures
        with _connect(archive_root / "source.db") as conn:
            raw = conn.execute("SELECT raw_id, parse_error FROM raw_sessions").fetchall()
            assert len(raw) == 1 and raw[0]["parse_error"] is None
            retained_raw_id = str(raw[0]["raw_id"])
        with _connect(archive_root / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0

        second = await processor.ingest_files([path], emit_event=False)
        assert second.succeeded_file_count == 1
        with _connect(archive_root / "source.db") as conn:
            assert [str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions")] == [retained_raw_id]
        with _connect(archive_root / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
    finally:
        stage.shutdown()


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
async def test_a_warm_cancelled_during_reconciliation_installs_no_prepared_write(tmp_path: Path) -> None:
    """A reconciliation that finishes after cancellation is closed, not installed.

    Anti-vacuity: install every finished reconciliation regardless of the
    event and the result carries the prepared write built from the snapshot
    the cancelled warm pinned.
    """
    archive_root = tmp_path / "archive"
    path = _write_fixture_corpus(tmp_path / "sessions", count=1)[0]
    await _ingest(archive_root, [path], parse_stage=None)
    path.write_bytes(_codex_session_bytes("session-0", (("user", "revised question"), ("assistant", "revised answer"))))
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards")
    cancelled = threading.Event()

    def cancelling_read(root: Path) -> AbstractContextManager[PinnedOperationRead]:
        # The caller is cancelled while this reconciliation reads.
        cancelled.set()
        return open_operation_read(root)

    try:
        stage.warm_paths(
            [(str(path), Provider.CODEX, True)],
            archive_root=archive_root,
            read_snapshot=cancelling_read,
            cancelled=cancelled,
        )
        kept = stage._path_results[str(path)]
        assert kept.error is None
        assert kept.prepared_writes == ()
    finally:
        stage.shutdown()


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


@pytest.mark.asyncio
async def test_existing_session_preparation_overlaps_paths_and_seals_selected_rows(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    paths = _write_fixture_corpus(tmp_path / "sessions", count=2)
    await _ingest(archive_root, paths, parse_stage=None)
    for index, path in enumerate(paths):
        path.write_bytes(
            _codex_session_bytes(
                f"session-{index}",
                (("user", f"revised question {index}"), ("assistant", f"revised answer {index}")),
            ).replace(b"2026-07-19T00:00:00Z", b"2026-07-20T00:00:00Z")
        )
    rendezvous = threading.Barrier(2, timeout=10)

    def concurrent_read(root: Path) -> AbstractContextManager[PinnedOperationRead]:
        rendezvous.wait()
        return open_operation_read(root)

    stage = LiveParseStage(max_workers=2, shard_directory=tmp_path / "parse-shards")
    try:
        candidates = [(str(path), Provider.CODEX, True) for path in paths]
        assert stage.warm_paths(candidates, archive_root=archive_root, read_snapshot=concurrent_read) == 2
        for index, path in enumerate(paths):
            prepared = stage.pop_path(str(path), blob_hash=hashlib.sha256(path.read_bytes()).hexdigest())
            assert prepared is not None and not prepared.deferred
            assert len(prepared.prepared_writes) == 1
            write = prepared.prepared_writes[0]
            assert write.session_id == f"codex-session:session-{index}"
            assert any(f"revised question {index}" in row for row in write.rows.block_rows)
            assert any(f"revised answer {index}" in row for row in write.rows.block_rows)
            prepared.discard()
    finally:
        stage.shutdown()


@pytest.mark.asyncio
async def test_existing_session_preparation_skips_replay_and_prepares_changed_append(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    path = _write_fixture_corpus(tmp_path / "sessions", count=1)[0]
    await _ingest(archive_root, [path], parse_stage=None)
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards")
    candidate = [(str(path), Provider.CODEX, True)]
    try:
        stage.warm_paths(candidate, archive_root=archive_root, read_snapshot=open_operation_read)
        unchanged = stage.pop_path(str(path), blob_hash=hashlib.sha256(path.read_bytes()).hexdigest())
        assert unchanged is not None and unchanged.prepared_writes == ()
        unchanged.discard()

        path.write_bytes(
            _codex_session_bytes(
                "session-0",
                (("user", "question 0"), ("assistant", "answer 0"), ("user", "follow up")),
            )
        )
        stage.warm_paths(candidate, archive_root=archive_root, read_snapshot=open_operation_read)
        changed = stage.pop_path(str(path), blob_hash=hashlib.sha256(path.read_bytes()).hexdigest())
        assert changed is not None and len(changed.prepared_writes) == 1
        changed.discard()
    finally:
        stage.shutdown()


@pytest.mark.asyncio
async def test_existing_session_preparation_binds_capture_mode_instead_of_parser_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import polylogue.storage.sqlite.archive_tiers.write as archive_write

    archive_root = tmp_path / "archive"
    path = _write_fixture_corpus(tmp_path / "sessions", count=1)[0]
    await _ingest(archive_root, [path], parse_stage=None)
    path.write_bytes(_codex_session_bytes("session-0", (("user", "changed question"),)))
    observed_raw_ids: list[str | None] = []
    original_prepare = archive_write.prepare_session_write

    def observe_prepare(*args: Any, **kwargs: Any) -> Any:
        observed_raw_ids.append(kwargs.get("raw_id"))
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(archive_write, "prepare_session_write", observe_prepare)
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards")
    try:
        stage.warm_paths(
            [(str(path), Provider.CODEX, True)],
            archive_root=archive_root,
            read_snapshot=open_operation_read,
            capture_mode=Provider.CLAUDE_CODE,
        )
        prepared = stage.pop_path(str(path), blob_hash=hashlib.sha256(path.read_bytes()).hexdigest())
        assert prepared is not None and len(prepared.prepared_writes) == 1
        prepared.discard()
    finally:
        stage.shutdown()
    assert observed_raw_ids == [
        deterministic_raw_session_id(
            origin_from_provider(Provider.CLAUDE_CODE), str(path), 0, hashlib.sha256(path.read_bytes()).digest()
        )
    ]


@pytest.mark.asyncio
async def test_existing_bundle_preparation_binds_later_acquisition_index_for_each_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import polylogue.storage.sqlite.archive_tiers.write as archive_write

    archive_root = tmp_path / "archive"
    path = tmp_path / "sessions" / "conversations.json"
    path.parent.mkdir()
    path.write_bytes(_chatgpt_bundle_bytes("first old", "second old"))
    await _ingest(archive_root, [path], parse_stage=None, source_name="chatgpt")
    path.write_bytes(_chatgpt_bundle_bytes("first new", "second new"))
    observed_raw_ids: list[str | None] = []
    original_prepare = archive_write.prepare_session_write

    def observe_prepare(*args: Any, **kwargs: Any) -> Any:
        observed_raw_ids.append(kwargs.get("raw_id"))
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(archive_write, "prepare_session_write", observe_prepare)
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards")
    try:
        stage.warm_paths(
            [(str(path), Provider.CHATGPT, False)],
            archive_root=archive_root,
            read_snapshot=open_operation_read,
            capture_mode=Provider.CHATGPT,
            source_index=3,
        )
        prepared = stage.pop_path(str(path), blob_hash=hashlib.sha256(path.read_bytes()).hexdigest())
        assert prepared is not None and len(prepared.prepared_writes) == 2
        prepared.discard()
    finally:
        stage.shutdown()
    expected_raw_id = deterministic_raw_session_id(
        origin_from_provider(Provider.CHATGPT), str(path), 3, hashlib.sha256(path.read_bytes()).digest()
    )
    assert observed_raw_ids == [expected_raw_id, expected_raw_id]


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


@pytest.mark.uses_real_clock("a real worker outlasts the stall window, which only reports")
def test_a_slow_in_memory_prefetch_is_awaited_and_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker slower than the stall window is waited for and its shard used.

    Anti-vacuity: a warm that gives up at the window caches nothing (and
    the writer reparses the file under its lease); no stall is reported.
    """
    from polylogue.sources.live import parse_prefetch

    paths = _write_fixture_corpus(tmp_path / "sessions", count=1)
    real_worker = parse_prefetch.live_parse_and_shard_worker
    stalls: list[dict[str, object]] = []

    def slow_worker(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.3)
        return real_worker(*args, **kwargs)

    def record(event: str, /, **fields: object) -> None:
        if event == "live.parse_prefetch.preparation_stalled":
            stalls.append(fields)

    monkeypatch.setattr(parse_prefetch, "live_parse_and_shard_worker", slow_worker)
    monkeypatch.setattr(parse_prefetch, "emit", record)
    stage = LiveParseStage(
        max_workers=1,
        max_inflight_bytes=10_000_000,
        shard_directory=tmp_path / "parse-shards",
        stall_report_seconds=0.05,
    )
    try:
        candidates = _live_parse_stage_candidates(paths, fallback_provider=Provider.CODEX)
        assert stage.warm(candidates) == 1
        assert stalls
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


@pytest.mark.asyncio
async def test_prepared_session_with_lowered_sink_keeps_identity_and_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker-lowered session is stored exactly as the inline route stores it.

    The worker derives each tool call's outcome in its sealed sink, a
    lowering the inline route applies only while building rows. Both routes
    must store the parse-bound digest and identical rows, and the writer must
    copy the worker's prepared rows rather than rebuild them. Anti-vacuity:
    re-hashing the lowered sink on the writer declines the shard as
    ``content_hash_mismatch`` (``copies`` reads 0) and stores a digest the
    inline route never produces (``sessions.content_hash`` differs).
    """
    import polylogue.storage.sqlite.archive_tiers.write as archive_tier_write

    rows: list[dict[str, object]] = [
        {"type": "session_meta", "payload": {"id": "lowered", "timestamp": "2026-07-19T00:00:00Z"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "lowered-m0",
                "role": "user",
                "content": [{"type": "input_text", "text": "list the files"}],
            },
        },
        {
            "type": "response_item",
            "payload": {"type": "function_call", "name": "shell", "arguments": '{"command":["ls"]}', "call_id": "c1"},
        },
        {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "c1", "output": "a.txt"}},
    ]
    source = tmp_path / "sessions" / "lowered.jsonl"
    source.parent.mkdir()
    source.write_bytes(b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows))
    await _ingest(tmp_path / "baseline", [source], parse_stage=None)

    copies = 0
    real_copy = archive_tier_write.copy_shard_session_rows

    def counting_copy(*args: object, **kwargs: object) -> object:
        nonlocal copies
        copies += 1
        return real_copy(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(archive_tier_write, "copy_shard_session_rows", counting_copy)
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards")
    try:
        await _ingest(tmp_path / "prepared", [source], parse_stage=stage)
    finally:
        stage.shutdown()

    baseline = _canonical_snapshot(tmp_path / "baseline")
    with _connect(tmp_path / "baseline" / "index.db") as conn:
        outcomes = {row[0] for row in conn.execute("SELECT tool_outcome FROM blocks WHERE tool_id = 'c1'")}
    assert outcomes and None not in outcomes
    assert copies == len(baseline["index.sessions"]) == 1
    assert baseline == _canonical_snapshot(tmp_path / "prepared")
