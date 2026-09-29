"""Equivalence: watcher parse-stage prefetch (flag on) vs. in-hold parse (flag off).

polylogue-wf8a. ``LiveParseStage`` pre-parses small JSONL full-ingest
candidates off the writer hold for the watcher's catch-up/live-batch route.
This proves two end-to-end claims against a real archive:

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

import asyncio
import hashlib
import json
import sqlite3
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import AbstractContextManager
from pathlib import Path
from typing import IO, Any, NoReturn

import pytest

from polylogue import Polylogue
from polylogue import logging as plog
from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.daemon.intake import FairIntakeDispatcher, IntakeClassSpec
from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter
from polylogue.operations.operation_context import PinnedOperationRead, open_operation_read
from polylogue.sources.live.batch import LiveBatchProcessor
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


def _fifo_blocked_lookahead_worker(
    fallback_provider_value: str, source_path: str, fallback_id: str, **_kwargs: object
) -> NoReturn:
    # Opening a FIFO with no writer blocks in the kernel, like a read on a
    # stalled mount: nothing in the worker can observe a cancellation.
    with open(Path(source_path).with_suffix(".fifo"), "rb"):
        pass
    raise AssertionError("the FIFO gained a writer")


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
    from polylogue.sources.live.parse_prefetch import live_parse_path_worker

    if Path(source_path).name.startswith("hung-"):
        time.sleep(3600)
    if Path(source_path).name.startswith("slow-"):
        # Holds its worker long enough for a sibling's worker to die first.
        time.sleep(2.5)
    if Path(source_path).name.startswith("growing-"):
        # Slow but advancing: its attempt directory grows the whole time.
        assert attempt_directory is not None
        stop = threading.Event()

        def grow() -> None:
            with (Path(attempt_directory) / "progress.bin").open("ab") as handle:
                while not stop.wait(0.05):
                    handle.write(b".")
                    handle.flush()

        grower = threading.Thread(target=grow)
        grower.start()
        try:
            time.sleep(3.0)
            return live_parse_path_worker(
                provider_value,
                source_path,
                fallback_id,
                is_stream=is_stream,
                shard_directory=shard_directory,
                attempt_directory=attempt_directory,
            )
        finally:
            stop.set()
            grower.join()
    if Path(source_path).name.startswith("killed-"):
        import os

        assert attempt_directory is not None
        attempt = Path(attempt_directory)
        if Path(source_path).name.startswith("killed-late-"):
            time.sleep(1.0)
        (attempt / "prepared-partial.db").write_bytes(b"partial prepared database")
        (attempt / "prepared-partial.db-journal").write_bytes(b"partial journal")
        (attempt / "shard-partial.db").write_bytes(b"partial shard")
        os._exit(7)
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


def _gemini_cli_document(session_id: str) -> bytes:
    return json.dumps(
        {
            "sessionId": session_id,
            "startTime": "2026-03-16T09:40:00.000Z",
            "lastUpdated": "2026-03-16T09:41:00.000Z",
            "kind": "chat",
            "messages": [{"id": "u1", "timestamp": "2026-03-16T09:40:01.000Z", "type": "user", "content": ["hello"]}],
        }
    ).encode()


def _chatgpt_document(conversation_id: str) -> bytes:
    return json.dumps(
        [
            {
                "id": conversation_id,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed_initial", [False, True])
async def test_changed_json_after_preparation_defers_without_writer_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformed_initial: bool
) -> None:
    """A stale worker's provider or parse error cannot label captured bytes.

    Without a carrier for the captured bytes, the writer does not decode the
    document to classify it: the capture is released, the path is deferred
    with its debt recorded, and the next pass retains the bytes once, under
    the provider its own preparation detected. Anti-vacuity: classifying the
    captured blob in the writer calls the patched detector and fails here;
    retaining the capture under the source's provider admits a second raw
    identity for the same bytes on the next pass.
    """
    import polylogue.sources.live.batch as batch
    import polylogue.sources.live.cursor as cursor_module

    monkeypatch.setattr(cursor_module, "_FULL_CURSOR_RECONCILIATION_RETRY_DELAY_S", 0)

    def writer_decode(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("the writer decoded a stage-owned JSON document to classify it")

    monkeypatch.setattr(batch, "detect_provider_from_path_sample_evidence", writer_decode)
    source = tmp_path / "inbox" / "session.json"
    source.parent.mkdir()
    source.write_bytes(b'{"broken":' if malformed_initial else _gemini_cli_document("gemini-before-copy"))
    chatgpt = _chatgpt_document("chatgpt-after-copy")
    original_copy = ArchiveBlobPublisher.write_from_fileobj
    changed = False

    def change_before_copy(store: ArchiveBlobPublisher, stream: IO[bytes], **kwargs: object) -> tuple[str, int]:
        nonlocal changed
        # The capture streams the file through the acquisition boundary,
        # whose raw reader names the source path; nothing is read yet.
        if Path(stream.raw.name) == source and not changed:  # type: ignore[attr-defined]
            if malformed_initial:
                assert stage._path_results[str(source)].error is not None
            else:
                assert stage.resolved_path_provider(str(source)) is Provider.GEMINI_CLI
            source.write_bytes(chatgpt)
            changed = True
        return original_copy(store, stream, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ArchiveBlobPublisher, "write_from_fileobj", change_before_copy)
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
        with plog.capture() as records:
            first = await processor.ingest_files([source], emit_event=False)
        assert changed
        assert str(source) in first.deferred_paths
        assert first.failed_file_count == 0
        assert [record for record in records if record["event"] == "live.ingest.json_capture_deferred"]
        with _connect(archive_root / "source.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 0
        with _connect(archive_root / "ops.db") as conn:
            debt = conn.execute(
                "SELECT stage, status FROM convergence_debt WHERE target_type = 'source_path'"
            ).fetchone()
        assert debt is not None and tuple(debt) == ("live_ingest_deferred", "deferred")

        second = await processor.ingest_files([source], emit_event=False)
        assert second.ingested_session_count == 1
    finally:
        stage.shutdown()
    with _connect(archive_root / "source.db") as conn:
        rows = conn.execute("SELECT origin, hex(blob_hash) FROM raw_sessions").fetchall()
    assert [tuple(row) for row in rows] == [
        (origin_from_provider(Provider.CHATGPT).value, hashlib.sha256(chatgpt).hexdigest().upper())
    ]


def test_path_worker_binds_provider_parse_and_seal_to_one_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source replaced mid-preparation cannot mix two revisions in one carrier.

    Anti-vacuity: sample the provider from the live path and then parse and
    hash it again, and the carrier holds the replacement's digest with the
    original's provider, which the writer accepts for a capture of the
    replacement.
    """
    import polylogue.sources.live.batch_support as batch_support

    source = tmp_path / "inbox" / "session.json"
    source.parent.mkdir()
    original = _gemini_cli_document("gemini-original")
    replacement = _chatgpt_document("chatgpt-replacement")
    source.write_bytes(original)
    real_detect = batch_support._detect_provider_from_path_sample
    replaced = False

    def detect_then_replace(path: Path, fallback_provider: Provider, **kwargs: bool) -> Provider:
        nonlocal replaced
        provider = real_detect(path, fallback_provider, **kwargs)
        staged = source.with_name("replacement.json")
        staged.write_bytes(replacement)
        staged.replace(source)
        replaced = True
        return provider

    monkeypatch.setattr(batch_support, "_detect_provider_from_path_sample", detect_then_replace)
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "shards")
    try:
        assert stage.warm_paths([(str(source), Provider.UNKNOWN, False)]) == frozenset()
        assert replaced
        carrier = stage._path_results[str(source)]
        assert carrier.error is None
        assert carrier.blob_hash == hashlib.sha256(original).hexdigest()
        assert carrier.resolved_provider is Provider.GEMINI_CLI
        refused = stage.pop_path(str(source), blob_hash=hashlib.sha256(replacement).hexdigest())
        assert refused is not None and refused.deferred
        assert refused.error == "captured source changed after preparation"
    finally:
        stage.shutdown()
    assert list((tmp_path / "shards").iterdir()) == []


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
        # A ChatGPT export is imported through the inbox, which classifies;
        # a bound location (``codex``) refuses it as foreign.
        await _ingest(tmp_path / "prepared", paths, parse_stage=stage, source_name="inbox")
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
        with plog.capture() as records:
            result = await processor.ingest_files([pending_path, ready_path], emit_event=False)
        assert not result.deferred_paths
        assert result.succeeded_file_count == 2
        # The stall is reported with its measurements, each registered in
        # the log field allowlist (an unregistered field is dropped with a
        # ``log.field_rejected`` record).
        stalls = [record for record in records if record["event"] == "live.parse_prefetch.preparation_stalled"]
        assert stalls and stalls[0]["paths"] in (1, 2) and "wait_ms" in stalls[0] and "attempt_bytes" in stalls[0]
        assert not [record for record in records if record["event"] == "log.field_rejected"]
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
async def test_lookahead_reads_no_source_on_the_admission_path(tmp_path: Path) -> None:
    """The lookahead hands the stage cursorless paths without reading them.

    Anti-vacuity: sample candidates (open the file) on the parent, whether in
    the offer or the submission, and the FIFO blocks this coroutine until the
    test's writer times out; submit a path that has a cursor row and the
    stage receives it.
    """
    import os

    (ingested,) = _write_fixture_corpus(tmp_path / "sessions", count=1)
    stuck = ingested.parent / "stuck.jsonl"
    os.mkfifo(stuck)
    archive_root = tmp_path / "archive"
    await _ingest(archive_root, [ingested], parse_stage=None)
    submitted: list[tuple[list[str], Provider]] = []

    class RecordingStage:
        def prefetch_paths(
            self, paths: list[str], *, fallback_provider: Provider, archive_root: Path | None = None
        ) -> int:
            submitted.append((list(paths), fallback_provider))
            return len(paths)

    processor = LiveBatchProcessor(
        Polylogue(archive_root=archive_root, db_path=archive_root / "index.db"),
        (WatchSource(name="codex", root=ingested.parent),),
        cursor=CursorStore(archive_root / "index.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
        parse_stage=RecordingStage(),  # type: ignore[arg-type]
        read_snapshot=open_operation_read,
    )
    processor.offer_parse_lookahead([ingested, stuck], source_name="codex")
    await asyncio.wait_for(processor._submit_ready_lookahead(), timeout=10)
    assert submitted == [([str(stuck)], Provider.CODEX)]
    assert processor._parse_lookahead is None


def test_read_ahead_refuses_a_path_swapped_for_a_symlink(tmp_path: Path) -> None:
    """Anti-vacuity: sample without the ``lstat`` check and the worker reads
    and seals the symlink's target outside the source."""
    from polylogue.sources.live.parse_prefetch import live_lookahead_path_worker

    (outside,) = _write_fixture_corpus(tmp_path / "outside", count=1)
    link = tmp_path / "sessions" / "swapped.jsonl"
    link.parent.mkdir()
    link.symlink_to(outside)
    shards = tmp_path / "parse-shards"
    result = live_lookahead_path_worker(Provider.CODEX.value, str(link), "swapped", shard_directory=str(shards))
    assert result.error == "read-ahead path is not a regular file"
    assert result.deferred
    assert not shards.exists() or not any(shards.iterdir())


@pytest.mark.asyncio
async def test_a_held_path_still_claims_its_sessions(tmp_path: Path) -> None:
    """Session overlap closes transitively over held paths.

    Files carry sessions {A}, {A, B} and {B}. The middle one is held for A,
    and the third must wait behind it for B. Anti-vacuity: let a held path
    claim nothing and the third path is admitted, publishing B ahead of the
    held middle file.
    """
    archive_root = tmp_path / "archive"
    (original,) = _write_fixture_corpus(tmp_path / "sessions", count=1)
    await _ingest(archive_root, [original], parse_stage=None)
    bundle = json.loads(_chatgpt_bundle_bytes("alpha", "beta"))
    root = tmp_path / "bundles"
    root.mkdir()
    only_a, both, only_b = root / "a.json", root / "ab.json", root / "b.json"
    only_a.write_text(json.dumps(bundle[:1]), encoding="utf-8")
    both.write_text(json.dumps(bundle), encoding="utf-8")
    only_b.write_text(json.dumps(bundle[1:]), encoding="utf-8")
    stage = LiveParseStage(max_workers=3, shard_directory=tmp_path / "parse-shards")
    candidates = [(str(path), Provider.CHATGPT, False) for path in (only_a, both, only_b)]
    try:
        held = stage.warm_paths(candidates, archive_root=archive_root, read_snapshot=open_operation_read)
        assert all(stage._path_results[str(path)].error is None for path in (only_a, both, only_b))
        assert held == frozenset({str(both), str(only_b)})
    finally:
        stage.shutdown()


@pytest.mark.uses_real_clock("waits on a real worker process")
def test_a_cancelled_retained_preparation_stops_its_worker(tmp_path: Path) -> None:
    """Owner cancellation stops a retained preparation that never finishes.

    Anti-vacuity: wait only for the result, as a deadline-free wait does, and
    the call never returns while the worker sleeps.
    """
    import multiprocessing

    from polylogue.core.compute_cancel import compute_cancel
    from polylogue.sources.revision_backfill import RetainedPreparationRetryableError
    from polylogue.storage.derived import raw as raw_derivation

    marker = tmp_path / "worker-started"
    cancelled = threading.Event()
    pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
    token = compute_cancel.set(cancelled)
    try:
        future = pool.submit(_stalled_process_worker, str(marker))
        deadline = time.monotonic() + 30
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        workers = tuple(pool._processes.values())
        assert marker.exists() and workers
        threading.Timer(0.5, cancelled.set).start()
        started = time.monotonic()
        with pytest.raises(RetainedPreparationRetryableError, match="cancelled"):
            raw_derivation._await_reporting_stalls(future, subject="raw example", pool=pool)
        assert time.monotonic() - started < 10
        assert not any(worker.is_alive() for worker in workers)
    finally:
        compute_cancel.reset(token)
        pool.shutdown(wait=False, cancel_futures=True)


def test_a_cancelled_pass_refuses_an_already_finished_preparation(tmp_path: Path) -> None:
    """Anti-vacuity: check cancellation only after a timed-out wait and a
    preparation that finishes within the poll is accepted after cancel."""
    from concurrent.futures import Future

    from polylogue.core.compute_cancel import compute_cancel
    from polylogue.sources.revision_backfill import RetainedPreparationRetryableError
    from polylogue.storage.derived import raw as raw_derivation

    finished: Future[str] = Future()
    finished.set_result("prepared")
    cancelled = threading.Event()
    cancelled.set()
    pool = ProcessPoolExecutor(max_workers=1)
    token = compute_cancel.set(cancelled)
    try:
        with pytest.raises(RetainedPreparationRetryableError, match="cancelled"):
            raw_derivation._await_reporting_stalls(finished, subject="raw example", pool=pool)
    finally:
        compute_cancel.reset(token)
        pool.shutdown(wait=False)
    assert raw_derivation._await_reporting_stalls(finished, subject="raw example", pool=pool) == "prepared"


@pytest.mark.asyncio
async def test_a_refused_batch_drops_its_lookahead_offer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: drop the offer only inside the locked ingest and a batch
    the authority gate refuses leaves its lookahead for the next batch."""
    (path,) = _write_fixture_corpus(tmp_path / "sessions", count=1)
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    stage = LiveParseStage(max_workers=1)
    watcher = LiveWatcher(
        Polylogue(archive_root=archive_root, db_path=archive_root / "index.db"),
        (WatchSource(name="codex", root=path.parent),),
        cursor=CursorStore(archive_root / "index.db"),
        parse_stage=stage,
    )

    def refuse(_paths: object) -> None:
        raise RuntimeError("cursor authority refused")

    monkeypatch.setattr(watcher._batch_processor, "require_cursor_authority", refuse)
    try:
        watcher.offer_parse_lookahead([path], source_name="codex")
        with pytest.raises(RuntimeError, match="refused"):
            await watcher._ingest_files([path])
        assert watcher._batch_processor._parse_lookahead is None
    finally:
        stage.shutdown()


def test_a_preparation_finishing_as_its_wait_expires_is_accepted(tmp_path: Path) -> None:
    """Anti-vacuity: re-raise the wait's own TimeoutError when the future is
    found done and a completed preparation is discarded as a failure."""
    from concurrent.futures import Future

    from polylogue.storage.derived import raw as raw_derivation

    class FinishesAtTheDeadline(Future[str]):
        def result(self, timeout: float | None = None) -> str:
            if timeout is not None and not self.done():
                self.set_result("prepared")
                raise TimeoutError
            return super().result(timeout)

    pool = ProcessPoolExecutor(max_workers=1)
    try:
        assert (
            raw_derivation._await_reporting_stalls(FinishesAtTheDeadline(), subject="raw example", pool=pool)
            == "prepared"
        )
    finally:
        pool.shutdown(wait=False)


def test_a_cancelled_warm_stops_verifying_a_claimed_read_ahead(tmp_path: Path) -> None:
    """Anti-vacuity: scan a claimed read-ahead without polling the warm's
    event and the cancelled warm still pays the full digest and keeps the
    result."""
    (path,) = _write_fixture_corpus(tmp_path / "sessions", count=1)
    stage = LiveParseStage(max_workers=2, shard_directory=tmp_path / "parse-shards")
    try:
        assert stage.prefetch_paths([str(path)], fallback_provider=Provider.CODEX) == 1
        stage._path_futures[str(path)].result(timeout=30)
        stage._collect_finished()
        assert str(path) in stage._unverified
        cancelled = threading.Event()
        cancelled.set()
        assert stage.warm_paths([(str(path), Provider.CODEX, True)], cancelled=cancelled) == frozenset()
        assert str(path) not in stage._path_results
        attempts = tmp_path / "parse-shards" / ".live-parse-attempts"
        assert [entry for entry in attempts.iterdir() if entry.name.startswith("attempt-")] == []
    finally:
        stage.shutdown()


def test_stage_shutdown_settles_an_active_warm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Daemon stop can shut the stage down while a warm is still running.

    Anti-vacuity: tear the stage down without cancelling and awaiting the
    active warm and the warm keeps waiting on its held worker (or mutates
    bookkeeping shutdown is clearing), so it has not returned before release.
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
    errors: list[BaseException] = []
    outcome: list[frozenset[str]] = []

    def run_warm() -> None:
        try:
            outcome.append(stage.warm_paths([(str(path), Provider.CODEX, True)]))
        except BaseException as exc:
            errors.append(exc)

    warm = threading.Thread(target=run_warm)
    stopper = threading.Thread(target=stage.shutdown)
    try:
        warm.start()
        assert started.wait(timeout=10)
        stopper.start()
        warm.join(timeout=10)
        assert not warm.is_alive(), "shutdown did not settle the active warm"
        assert errors == [] and outcome == [frozenset()]
    finally:
        released.set()
        warm.join(timeout=30)
        stopper.join(timeout=30)
    assert not stopper.is_alive()
    attempts = tmp_path / "parse-shards" / ".live-parse-attempts"
    assert not attempts.exists() or [entry for entry in attempts.iterdir() if entry.name.startswith("attempt-")] == []


def test_shutdown_cannot_miss_a_warm_that_is_publishing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown racing a warm's publication still cancels that warm.

    The instrumented lock starts shutdown while the warm is inside its
    publication step. Anti-vacuity: read the active warm without the lock
    that guards publication and shutdown sees no warm, then blocks behind a
    warm nobody cancelled, so the warm is still waiting on its held worker.
    """
    import polylogue.sources.live.parse_prefetch as parse_prefetch

    (path,) = _write_fixture_corpus(tmp_path / "sessions", count=1)
    released = threading.Event()
    original_worker = parse_prefetch.live_parse_path_worker

    def held_worker(provider_value: str, source_path: str, *args: object, **kwargs: object) -> object:
        released.wait(timeout=30)
        return original_worker(provider_value, source_path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", held_worker)
    monkeypatch.setattr(parse_prefetch, "_PROGRESS_POLL_SECONDS", 0.05)
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards")
    stopper = threading.Thread(target=stage.shutdown)
    real_lock = stage._publish_lock

    class RacingLock:
        entered = 0

        def __enter__(self) -> None:
            real_lock.acquire()
            if threading.current_thread() is not stopper and RacingLock.entered == 0:
                RacingLock.entered += 1
                stopper.start()
                time.sleep(0.2)  # shutdown now runs inside the publication window

        def __exit__(self, *exc: object) -> None:
            real_lock.release()

    stage._publish_lock = RacingLock()  # type: ignore[assignment]
    warm = threading.Thread(target=stage.warm_paths, args=([(str(path), Provider.CODEX, True)],))
    try:
        warm.start()
        warm.join(timeout=10)
        assert not warm.is_alive(), "shutdown missed the publishing warm"
    finally:
        released.set()
        warm.join(timeout=30)
        stopper.join(timeout=30)
    assert not stopper.is_alive()


@pytest.mark.uses_real_clock("waits for a real process worker to finish")
def test_reaped_read_ahead_is_dropped_without_a_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: salvage a finished reaped read-ahead through the pool
    restart and its artifact is fully re-hashed by a warm that never claims it."""
    from polylogue.sources.prepared_jsonl import PreparedJsonl

    (path,) = _write_fixture_corpus(tmp_path / "sessions", count=1)
    full_scans: list[bool] = []
    original_verify = PreparedJsonl.verify_files

    def recording_verify(self: PreparedJsonl, *, full: bool, stop: object = None) -> None:
        full_scans.append(full)
        original_verify(self, full=full, stop=stop)  # type: ignore[arg-type]

    monkeypatch.setattr(PreparedJsonl, "verify_files", recording_verify)
    stage = LiveParseStage(max_workers=2, shard_directory=tmp_path / "parse-shards", use_processes=True)
    try:
        assert stage.prefetch_paths([str(path)], fallback_provider=Provider.CODEX) == 1
        stage._path_futures[str(path)].result(timeout=60)
        stage._reap_speculation([str(path)], reason="test")
        assert True not in full_scans
        assert str(path) not in stage._path_results and str(path) not in stage._speculative
        attempts = tmp_path / "parse-shards" / ".live-parse-attempts"
        assert [entry for entry in attempts.iterdir() if entry.name.startswith("attempt-")] == []
    finally:
        stage.shutdown()


def test_a_claimed_read_ahead_finishing_after_cancel_is_not_scanned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-ahead still running at claim is verified under the warm's cancellation.

    Anti-vacuity: verify its completion without the warm's stop predicate and
    the cancelled warm pays the full digest and records the result.
    """
    import polylogue.sources.live.parse_prefetch as parse_prefetch

    (path,) = _write_fixture_corpus(tmp_path / "sessions", count=1)
    cancelled = threading.Event()
    claimed = threading.Event()
    original_worker = parse_prefetch.live_parse_path_worker

    def cancelling_worker(provider_value: str, source_path: str, *args: object, **kwargs: object) -> object:
        claimed.wait(timeout=30)
        result = original_worker(provider_value, source_path, *args, **kwargs)  # type: ignore[arg-type]
        cancelled.set()  # the warm is cancelled as this preparation completes
        return result

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", cancelling_worker)
    # A long poll: the warm wakes only when the preparation completes, so it
    # collects that completion (under cancellation) before its next check.
    monkeypatch.setattr(parse_prefetch, "_PROGRESS_POLL_SECONDS", 30.0)
    stage = LiveParseStage(max_workers=2, shard_directory=tmp_path / "parse-shards")
    try:
        assert stage.prefetch_paths([str(path)], fallback_provider=Provider.CODEX) == 1
        threading.Timer(0.1, claimed.set).start()
        assert stage.warm_paths([(str(path), Provider.CODEX, True)], cancelled=cancelled) == frozenset()
        assert str(path) not in stage._path_futures
        assert str(path) not in stage._path_results
    finally:
        claimed.set()
        stage.shutdown()


def test_a_stalled_read_ahead_cannot_wedge_required_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unclaimed read-ahead holding the byte budget is preempted, then reaped.

    Anti-vacuity: let a required path wait for budget held by unclaimed
    read-ahead (the stalled worker never finishes and no stage call advances
    its lifetime) and the warm thread is still alive after the join.
    """
    import polylogue.sources.live.parse_prefetch as parse_prefetch

    stalled_path, required_path = _write_fixture_corpus(tmp_path / "sessions", count=2)
    released = threading.Event()
    original_worker = parse_prefetch.live_parse_path_worker

    def stalled_worker(provider_value: str, source_path: str, *args: object, **kwargs: object) -> object:
        if source_path == str(stalled_path):
            released.wait(timeout=60)  # a source read that never returns
        return original_worker(provider_value, source_path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", stalled_worker)
    monkeypatch.setattr(parse_prefetch, "_PROGRESS_POLL_SECONDS", 0.05)
    budget = stalled_path.stat().st_size + required_path.stat().st_size - 1
    stage = LiveParseStage(
        max_workers=2,
        max_inflight_bytes=budget,
        shard_directory=tmp_path / "parse-shards",
        preempt_grace_seconds=0.1,
    )
    warm = threading.Thread(target=stage.warm_paths, args=([(str(required_path), Provider.CODEX, True)],))
    try:
        assert stage.prefetch_paths([str(stalled_path)], fallback_provider=Provider.CODEX) == 1
        with plog.capture() as records:
            warm.start()
            warm.join(timeout=10)
        assert not warm.is_alive()
        events = {record["event"]: record for record in records}
        assert events["live.parse_prefetch.speculation_preempted"]["budget_ms"] == 100
        assert events["live.parse_prefetch.speculation_reaped"]["paths"] == 1
        assert "log.field_rejected" not in events
        prepared = stage.pop_path(str(required_path), blob_hash=hashlib.sha256(required_path.read_bytes()).hexdigest())
        assert prepared is not None and prepared.error is None
        prepared.discard()
        assert str(stalled_path) not in stage._path_futures
        assert str(stalled_path) not in stage._speculative
    finally:
        released.set()
        warm.join(timeout=30)
        stage.shutdown()


@pytest.mark.uses_real_clock("waits for a real worker process to block in a source read")
def test_an_abandoned_read_ahead_stuck_in_a_source_read_is_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read-ahead blocked in a source read is stopped with its worker process.

    The worker blocks in ``open`` on a FIFO with no writer, as a read on a
    stalled mount would. Anti-vacuity: drop an expired read-ahead without
    reaping it and the worker process is still alive after expiry.
    """
    import os

    import polylogue.sources.live.parse_prefetch as parse_prefetch

    (stuck,) = _write_fixture_corpus(tmp_path / "sessions", count=1)
    os.mkfifo(stuck.with_suffix(".fifo"))
    monkeypatch.setattr(parse_prefetch, "live_lookahead_path_worker", _fifo_blocked_lookahead_worker)
    stage = LiveParseStage(max_workers=2, shard_directory=tmp_path / "parse-shards", use_processes=True)
    try:
        assert stage.prefetch_paths([str(stuck)], fallback_provider=Provider.CODEX) == 1
        executor = stage._executor
        assert isinstance(executor, ProcessPoolExecutor)
        deadline = time.monotonic() + 30
        while not getattr(executor, "_processes", None) and time.monotonic() < deadline:
            time.sleep(0.05)
        workers = tuple(executor._processes.values())
        assert workers
        time.sleep(0.5)
        assert not stage._path_futures[str(stuck)].done()
        for _ in range(parse_prefetch._SPECULATIVE_LIFETIME_CALLS):
            stage.prefetch_paths([], fallback_provider=Provider.CODEX)
        assert stage._path_futures == {}
        assert stage._path_inflight_bytes == 0
        assert not any(worker.is_alive() for worker in workers)
        attempts = tmp_path / "parse-shards" / ".live-parse-attempts"
        assert [path for path in attempts.iterdir() if path.name.startswith("attempt-")] == []
    finally:
        stage.shutdown()


@pytest.mark.asyncio
async def test_a_warm_reconciles_one_path_per_session_in_intake_order(tmp_path: Path) -> None:
    """A later revision of a session in the same warm is held, not pinned.

    Both paths revise an archived session. Only the first is reconciled
    against this snapshot; the second is returned held, carrying no prepared
    write, and the next warm reconciles it afresh. Anti-vacuity: reconcile
    every path of the warm against one snapshot and the second path carries
    a prepared write pinned to predecessor state the first path's
    publication replaces, and nothing is held.
    """
    archive_root = tmp_path / "archive"
    (original,) = _write_fixture_corpus(tmp_path / "sessions", count=1)
    await _ingest(archive_root, [original], parse_stage=None)
    first = original.parent / "revision-1.jsonl"
    second = original.parent / "revision-2.jsonl"
    turns: tuple[tuple[str, str], ...] = (("user", "question 0"), ("assistant", "answer 0"), ("user", "more"))
    first.write_bytes(_codex_session_bytes("session-0", turns))
    second.write_bytes(_codex_session_bytes("session-0", (*turns, ("assistant", "latest"))))
    stage = LiveParseStage(max_workers=2, shard_directory=tmp_path / "parse-shards")
    candidates = [(str(first), Provider.CODEX, True), (str(second), Provider.CODEX, True)]
    try:
        held = stage.warm_paths(candidates, archive_root=archive_root, read_snapshot=open_operation_read)
        assert held == frozenset({str(second)})
        assert len(stage._path_results[str(first)].prepared_writes) == 1
        assert stage._path_results[str(second)].prepared_writes == ()
        # The next warm (after the first path publishes) reconciles the held
        # path; a reconciliation never outlives the warm that made it.
        stage._path_results.pop(str(first)).discard()
        assert stage.warm_paths([candidates[1]], archive_root=archive_root, read_snapshot=open_operation_read) == (
            frozenset()
        )
        assert len(stage._path_results[str(second)].prepared_writes) == 1
    finally:
        stage.shutdown()


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
        assert stage.prefetch_paths([str(path) for path in paths], fallback_provider=Provider.CODEX) == 2
        # Returned while both workers are still held: nothing waited on them.
        assert len(stage._path_futures) == 2
        assert not any(future.done() for future in stage._path_futures.values())
        released.set()
        assert stage.warm_paths(candidates) == frozenset()
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
        assert stage.prefetch_paths([str(skipped)], fallback_provider=Provider.CODEX) == 1
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
            stage.prefetch_paths([str(path)], fallback_provider=Provider.CODEX)
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
        assert stage.prefetch_paths([str(skipped)], fallback_provider=Provider.CODEX) == 1
        stage._path_futures[str(skipped)].result(timeout=30)
        for _ in range(parse_prefetch._SPECULATIVE_LIFETIME_CALLS):
            stage.prefetch_paths([], fallback_provider=Provider.CODEX)
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
        assert stage.prefetch_paths([str(path)], fallback_provider=Provider.CODEX) == 1
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
        assert stage.prefetch_paths([str(path)], fallback_provider=Provider.CODEX) == 0
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
        assert stage.prefetch_paths([str(path)], fallback_provider=Provider.CODEX) == 0
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
        assert stage.warm_paths(candidates) == frozenset()
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
        # Spawned-worker start (interpreter plus this module's imports) is
        # host-load dependent; only the shutdown below carries the bound.
        start_deadline = time.monotonic() + 60
        while not marker.exists() and time.monotonic() < start_deadline:
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

    def verify_outside_writer(self: PreparedJsonl, *, full: bool, stop: object = None) -> None:
        nonlocal full_verifications
        if full:
            assert not inside_writer, "full artifact digest ran under writer admission"
            full_verifications += 1
        original_verify(self, full=full, stop=stop)  # type: ignore[arg-type]

    monkeypatch.setattr(PreparedJsonl, "verify_files", verify_outside_writer)
    try:
        assert stage.warm_paths([candidate]) == frozenset()
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
        assert stage.warm_paths([(str(sibling_path), Provider.CODEX, True)]) == frozenset()
        sibling = stage._path_results[str(sibling_path)]
        assert sibling.attempt_directory is not None and sibling.attempt_directory.exists()

        for index in range(3):
            killed = tmp_path / f"killed-{index}.json"
            killed.write_text("worker exits before parsing")
            candidate = (str(killed), Provider.CODEX, True)
            assert stage.warm_paths([candidate]) == frozenset()
            failed = stage.pop_path(str(killed), blob_hash="0" * 64)
            assert failed is not None and failed.deferred
            assert "worker process died" in str(failed.error)
            attempts = tuple(attempt_root.iterdir())
            assert attempts == (sibling.attempt_directory,)
            assert not any(any(attempt.glob("*partial*")) for attempt in attempts)

        assert stage.warm_paths([(str(retry_path), Provider.CODEX, True)]) == frozenset()
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
            assert stage.warm_paths([(str(killed), Provider.CODEX, True)]) == frozenset()
            assert stage._cleanup_blocked
            assert stage.cleanup_failure_count == 1
            failed = stage.pop_path(str(killed), blob_hash="0" * 64)
            assert failed is not None and failed.deferred
            residue = tuple(attempt_root.iterdir())
            assert len(residue) == 1
            assert sorted(path.name for path in residue[0].iterdir()) == [
                ".worker-started",
                "prepared-partial.db",
                "prepared-partial.db-journal",
                "shard-partial.db",
            ]

            retry = tmp_path / "killed-retry-must-not-start.json"
            retry.write_text("must remain unsubmitted")
            assert stage.warm_paths([(str(retry), Provider.CODEX, True)]) == frozenset()
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
        stage._path_observations[str(path)] = (path.stat().st_size, 0, 0)
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


@pytest.mark.uses_real_clock("a real worker process dies while the pool is idle")
@pytest.mark.parametrize("pending_completion", [False, True])
def test_a_pool_broken_while_idle_is_replaced_at_the_next_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pending_completion: bool,
) -> None:
    """A worker that dies with no preparation outstanding cannot poison the stage.

    Anti-vacuity: record the failed submission as a deferred result and keep
    the broken executor, and every later preparation reports
    ``worker submission failed: BrokenProcessPool`` until the daemon restarts.
    """
    first, second = _write_fixture_corpus(tmp_path / "sessions", count=2)
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards", use_processes=True)
    try:
        broken = stage._executor
        with pytest.raises(BrokenProcessPool):
            broken.submit(_dead_process_worker).result(timeout=15)
        assert not stage._path_futures
        if pending_completion:
            from concurrent.futures import Future

            import polylogue.sources.live.parse_prefetch as parse_prefetch

            # The process manager can report a broken submit before all its
            # sibling futures have been signalled. Never await them here.
            pending: Future[Any] = Future()
            pending.set_running_or_notify_cancel()
            stage._path_futures[str(first)] = pending
            try:
                with monkeypatch.context() as guard:

                    def refuse_wait(*_args: Any, **_kwargs: Any) -> None:
                        raise AssertionError("broken submit waited for an unsignalled sibling")

                    guard.setattr(parse_prefetch, "wait", refuse_wait)
                    assert stage._replace_pool_broken_before_submit()
                assert not stage._path_futures
                stage._path_results.clear()
            finally:
                # A deliberately unsignalled test future must not hang cleanup
                # if a regression raises before the stage forgets it.
                if not pending.done():
                    pending.set_exception(BrokenProcessPool("synthetic pending worker completion"))
        for path in (first, second):
            assert stage.warm_paths([(str(path), Provider.CODEX, True)]) == frozenset()
            result = stage.pop_path(str(path), blob_hash=hashlib.sha256(path.read_bytes()).hexdigest())
            assert result is not None and result.error is None
            result.discard()
        assert stage._executor is not broken
    finally:
        stage.shutdown()


_UNKNOWN_RECORD = b'{"id":"unknown-1","messages":[{"id":"m1","role":"user","content":"hello"}]}\n'


@pytest.mark.asyncio
async def test_path_worker_decode_failure_reaches_terminal_unknown_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker's decode failure keeps its identity through the writer.

    Anti-vacuity: re-raise the carrier's failure as an untyped error, and the
    unknown-provider raw is marked failed and retried with no
    ``terminal_unknown_json_decode`` evidence.
    """
    import polylogue.sources.live.parse_prefetch as parse_prefetch

    root = tmp_path / "unknown"
    root.mkdir()
    path = root / "mixed.jsonl"
    path.write_bytes(_UNKNOWN_RECORD + b'{"broken":}\n' + _UNKNOWN_RECORD.replace(b"unknown-1", b"unknown-2"))
    prepared = 0
    real_worker = parse_prefetch.live_parse_path_worker

    def counted_worker(*args: Any, **kwargs: Any) -> object:
        nonlocal prepared
        prepared += 1
        return real_worker(*args, **kwargs)

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", counted_worker)
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "shards")
    processor = LiveBatchProcessor(
        Polylogue(archive_root=archive_root, db_path=archive_root / "index.db"),
        (WatchSource(name="unknown", root=root),),
        cursor=CursorStore(archive_root / "index.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
        parse_stage=stage,
        read_snapshot=open_operation_read,
    )
    try:
        first = await processor.ingest_files([path], emit_event=False)
        assert prepared == 1
        second = await processor.ingest_files([path], emit_event=False)
    finally:
        stage.shutdown()
    # Settled terminal corrupt input: excluded, never admitted (polylogue-xf8qp).
    assert first.failed_file_count == 0 and first.succeeded_file_count == 0
    assert first.excluded_paths == {str(path): "corrupt_input"}
    assert second.failed_file_count == 0
    with _connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 1
        artifact = conn.execute("SELECT artifact_kind, support_status FROM raw_artifacts").fetchone()
    assert (artifact[0], artifact[1]) == ("terminal_unknown_json_decode", "decode_failed")
    lifecycle = read_raw_failure_lifecycle(archive_root / "source.db")
    assert lifecycle.terminal == 1
    assert lifecycle.unexplained == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("prepared_off_writer", [False, True])
async def test_known_provider_jsonl_with_a_malformed_middle_record_is_terminal(
    tmp_path: Path, prepared_off_writer: bool
) -> None:
    """A complete malformed record cannot be skipped on the way to the frontier.

    The valid records around it do not make the capture an ordinary success:
    the raw is retained with terminal corrupt-input evidence, as a malformed
    final record already is, and its unchanged observation is not retried.
    Anti-vacuity: decode known-provider JSONL leniently, in the writer or in
    the path worker, and the malformed line is skipped with a warning while
    the session publishes and the cursor reaches EOF with no evidence.
    """
    rows = _codex_session_bytes("codex-torn", (("user", "before"), ("assistant", "after"))).splitlines(keepends=True)
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "torn.jsonl"
    path.write_bytes(b"".join(rows[:2]) + b'{"type":"response_item","payload":\n' + b"".join(rows[2:]))
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "shards") if prepared_off_writer else None
    processor = LiveBatchProcessor(
        Polylogue(archive_root=archive_root, db_path=archive_root / "index.db"),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(archive_root / "index.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
        parse_stage=stage,
        read_snapshot=open_operation_read,
    )
    try:
        first = await processor.ingest_files([path], emit_event=False)
        second = await processor.ingest_files([path], emit_event=False)
    finally:
        if stage is not None:
            stage.shutdown()
    # Settled terminal corrupt input: excluded, never admitted (polylogue-xf8qp).
    assert first.failed_file_count == 0 and first.succeeded_file_count == 0
    assert first.excluded_paths == {str(path): "corrupt_input"}
    assert second.failed_file_count == 0
    with _connect(archive_root / "source.db") as conn:
        raws = conn.execute("SELECT blob_size, parse_error FROM raw_sessions").fetchall()
        artifact = conn.execute("SELECT artifact_kind, support_status FROM raw_artifacts").fetchone()
    assert len(raws) == 1
    assert raws[0][0] == path.stat().st_size
    assert raws[0][1] is not None
    assert (artifact[0], artifact[1]) == ("terminal_corrupt_input", "decode_failed")
    with _connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    lifecycle = read_raw_failure_lifecycle(archive_root / "source.db")
    assert lifecycle.terminal == 1
    assert lifecycle.unexplained == 0


def test_path_preparation_refuses_source_changed_after_worker_seal(tmp_path: Path) -> None:
    import hashlib

    path = _write_fixture_corpus(tmp_path / "sessions", count=1)[0]
    directory = tmp_path / "parse-shards"
    stage = LiveParseStage(max_workers=1, shard_directory=directory)
    try:
        assert stage.warm_paths([(str(path), Provider.CODEX, True)]) == frozenset()
        path.write_bytes(path.read_bytes() + b' {"type":"event_msg","payload":{}}\n')
        result = stage.pop_path(str(path), blob_hash=hashlib.sha256(path.read_bytes()).hexdigest())
        assert result is not None and result.deferred
        assert result.error == "captured source changed after preparation"
    finally:
        stage.shutdown()
    assert list(directory.iterdir()) == []


@pytest.mark.asyncio
async def test_a_warm_cancelled_during_reconciliation_installs_no_prepared_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reconciliation cancelled mid-carrier stops and installs nothing.

    Anti-vacuity: install every finished reconciliation regardless of the
    event and the result carries the prepared write built from the snapshot
    the cancelled warm pinned; check the event only before the carrier and
    the replacement write is still built.
    """
    import polylogue.storage.sqlite.archive_tiers.write as archive_write

    built: list[object] = []
    original_prepare = archive_write.prepare_session_write

    def recording_prepare(*args: object, **kwargs: object) -> object:
        built.append(args)
        return original_prepare(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(archive_write, "prepare_session_write", recording_prepare)
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
        assert built == []
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
            == frozenset()
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
        assert stage.warm_paths(candidates, archive_root=archive_root, read_snapshot=concurrent_read) == frozenset()
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
async def test_unknown_mixed_jsonl_prefetch_falls_back_to_strict_decode(tmp_path: Path) -> None:
    """Strict prefetch refuses a partial unknown-provider conversation."""
    root = tmp_path / "unknown"
    root.mkdir()
    path = root / "mixed.jsonl"
    path.write_bytes(b'{"id":"unknown-1","messages":[{"id":"m1","role":"user","content":"hello"}]}\n{"broken":}\n')
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
    assert result.succeeded_file_count == 0
    assert result.excluded_paths == {str(path): "corrupt_input"}
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


def _warm_within(stage: LiveParseStage, candidates: list[tuple[str, Provider, bool]], *, timeout: float = 60.0) -> None:
    """Run one warm on a thread, failing instead of hanging when it never returns."""
    warm = threading.Thread(target=stage.warm_paths, args=(candidates,))
    warm.start()
    warm.join(timeout=timeout)
    assert not warm.is_alive(), "the warm never returned"


@pytest.mark.uses_real_clock("real worker processes die under the warm")
def test_a_file_that_kills_its_worker_every_time_escalates_to_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated worker death on one unchanged file stops being deferred.

    Anti-vacuity (polylogue-7c5k6): with every death reported as deferred the
    third attempt is still ``deferred`` and the file is retried every pass for
    the whole build; escalation makes it a non-deferred failure the cursor's
    failure budget can quarantine.
    """
    import polylogue.sources.live.parse_prefetch as parse_prefetch

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", _attempt_death_path_worker)
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards", use_processes=True)
    killer = tmp_path / "killed-every-time.json"
    killer.write_text("worker exits before parsing")
    outcomes = []
    try:
        for _ in range(parse_prefetch._MAX_WORKER_LOSSES_PER_OBSERVATION):
            assert stage.warm_paths([(str(killer), Provider.CODEX, True)]) == frozenset()
            result = stage.pop_path(str(killer), blob_hash="0" * 64)
            assert result is not None and result.error is not None
            outcomes.append(result.deferred)
    finally:
        stage.shutdown()
    assert outcomes[:-1] == [True] * (len(outcomes) - 1)
    assert outcomes[-1] is False


def _broken_future_in_flight(
    stage: LiveParseStage, source_path: str, *, started: bool, finished: bool = False
) -> Future[Any]:
    from polylogue.sources.live.parse_prefetch import _WORKER_FINISHED_MARKER, _WORKER_STARTED_MARKER

    attempt_directory = stage._new_attempt_directory()
    if started:
        (attempt_directory / _WORKER_STARTED_MARKER).touch()
    if finished:
        (attempt_directory / _WORKER_FINISHED_MARKER).touch()
    future: Future[Any] = Future()
    future.set_exception(BrokenProcessPool("a worker died"))
    stage._path_futures[source_path] = future
    stage._path_attempt_dirs[source_path] = attempt_directory
    stage._path_observations[source_path] = (0, 0, 0)
    stage._path_progress[source_path] = (0, 0.0)
    return future


def test_a_broken_pool_does_not_charge_a_never_dispatched_sibling(tmp_path: Path) -> None:
    """A pool break completes every pending future, not only the culprit's.

    Anti-vacuity (Codex, polylogue-b8of0 follow-up): charge every future that
    raises ``BrokenProcessPool`` regardless of whether its worker ever
    started, and a healthy file merely queued behind the one that actually
    killed its worker accumulates worker-loss charges for a death it had
    nothing to do with.
    """
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards", use_processes=True)
    source_path = str(tmp_path / "queued-never-started.json")
    try:
        future = _broken_future_in_flight(stage, source_path, started=False)

        stage._collect_path_future(source_path, future)

        result = stage._path_results.pop(source_path)
        assert result.deferred is True
        assert source_path not in stage._path_worker_losses
        assert source_path not in stage._path_solo_suspects
    finally:
        stage.shutdown()


@pytest.mark.parametrize("collect_first", ["healthy", "culprit"])
def test_a_pool_break_charges_only_the_worker_still_holding_its_task(tmp_path: Path, collect_first: str) -> None:
    """The one task started and not finished at a break is the only charge.

    Anti-vacuity (Codex): with only a started marker, a healthy sibling that
    had also been dispatched is charged when it is collected first, and the
    real culprit is deferred uncharged by the restart; either collection
    order must charge the culprit and only it.
    """
    stage = LiveParseStage(max_workers=2, shard_directory=tmp_path / "parse-shards", use_processes=True)
    healthy, culprit = str(tmp_path / "healthy.json"), str(tmp_path / "culprit.json")
    try:
        futures = {
            healthy: _broken_future_in_flight(stage, healthy, started=True, finished=True),
            culprit: _broken_future_in_flight(stage, culprit, started=True),
        }
        first = healthy if collect_first == "healthy" else culprit
        stage._collect_path_future(first, futures[first])
        assert set(stage._path_worker_losses) == {culprit}
        assert stage._path_results[healthy].deferred and stage._path_results[culprit].deferred
        assert stage._path_solo_suspects == {culprit}
        assert stage._path_futures == {} and stage._path_observations == {} and stage._path_progress == {}
    finally:
        stage.shutdown()


def test_a_success_preserved_by_a_pool_restart_ends_its_loss_streak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sibling success kept by the restart clears its worker-loss count.

    Anti-vacuity (Codex): only a collected success reset the streak, so a
    path with two earlier losses whose success was installed by the restart
    escalated to a terminal failure on its next single loss.
    """
    from polylogue.sources.prepared_jsonl import PreparedJsonl

    monkeypatch.setattr(PreparedJsonl, "verify_files", lambda self, *, full, stop=None: None)
    stage = LiveParseStage(max_workers=2, shard_directory=tmp_path / "parse-shards", use_processes=True)
    monkeypatch.setattr(stage, "_validate_attempt_result", lambda result, attempt_directory: None)
    healthy, culprit = str(tmp_path / "healthy.json"), str(tmp_path / "culprit.json")
    try:
        sealed: Future[Any] = Future()
        sealed.set_result(PreparedJsonl("0" * 64, None, None))
        stage._path_futures[healthy] = sealed
        stage._path_attempt_dirs[healthy] = stage._new_attempt_directory()
        stage._path_observations[healthy] = (0, 0, 0)
        stage._path_worker_losses[healthy] = ((0, 0, 0), 2)
        stage._path_solo_suspects.add(healthy)
        broken = _broken_future_in_flight(stage, culprit, started=True)

        stage._collect_path_future(culprit, broken)

        assert stage._path_results[healthy].error is None
        assert healthy not in stage._path_worker_losses
        assert healthy not in stage._path_solo_suspects
        assert set(stage._path_worker_losses) == {culprit}
    finally:
        stage.shutdown()


def test_a_worker_that_returns_an_error_ends_the_loss_streak(tmp_path: Path) -> None:
    """A preparation that raises, without losing its worker, resets the streak.

    Anti-vacuity (Codex): only an error-free result reset the count, so two
    losses, a transient ``OSError`` and one more loss escalated a file whose
    losses were never consecutive.
    """
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards", use_processes=True)
    source_path = str(tmp_path / "flaky.json")
    try:
        future: Future[Any] = Future()
        future.set_exception(OSError("transient read"))
        stage._path_futures[source_path] = future
        stage._path_attempt_dirs[source_path] = stage._new_attempt_directory()
        stage._path_observations[source_path] = (0, 0, 0)
        stage._path_worker_losses[source_path] = ((0, 0, 0), 2)
        stage._path_solo_suspects.add(source_path)

        stage._collect_path_future(source_path, future)

        assert stage._path_results[source_path].deferred
        assert source_path not in stage._path_worker_losses
        assert source_path not in stage._path_solo_suspects
    finally:
        stage.shutdown()


def test_a_break_with_two_held_tasks_charges_neither(tmp_path: Path) -> None:
    """Two tasks held at one break are both deferred uncharged, as suspects.

    Anti-vacuity: charging whichever future is collected first quarantines a
    healthy file after three such breaks.
    """
    stage = LiveParseStage(max_workers=2, shard_directory=tmp_path / "parse-shards", use_processes=True)
    first, second = str(tmp_path / "first.json"), str(tmp_path / "second.json")
    try:
        future = _broken_future_in_flight(stage, first, started=True)
        _broken_future_in_flight(stage, second, started=True)
        stage._collect_path_future(first, future)
        assert not stage._path_worker_losses
        assert stage._path_solo_suspects == {first, second}
        assert stage._path_results[first].deferred and stage._path_results[second].deferred
    finally:
        stage.shutdown()


def test_a_suspect_is_never_read_ahead(tmp_path: Path) -> None:
    """Read-ahead cannot run a suspect beside other work, nor join a solo run.

    Anti-vacuity: submit read-ahead without the solo check and the suspect
    runs concurrently with whatever else is read ahead, so its next worker
    loss is unattributable again.
    """
    suspect, other = _write_fixture_corpus(tmp_path / "sessions", count=2)
    stage = LiveParseStage(max_workers=3, shard_directory=tmp_path / "parse-shards")
    try:
        stage._path_solo_suspects.add(str(suspect))
        assert stage.prefetch_paths([str(suspect)], fallback_provider=Provider.CODEX) == 0
        assert stage.warm_paths([(str(suspect), Provider.CODEX, True)]) == frozenset()
        # The suspect ran alone and its worker came back: no longer suspect.
        assert str(suspect) not in stage._path_solo_suspects
        assert stage.prefetch_paths([str(other)], fallback_provider=Provider.CODEX) == 1
    finally:
        stage.shutdown()


@pytest.mark.uses_real_clock("two real worker processes run concurrently until one dies")
def test_concurrent_worker_death_is_charged_to_the_killer_after_solo_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A death during concurrent preparations is resolved by running each alone.

    Anti-vacuity (Codex): with a healthy long preparation and a killer
    running at once, charging every started future escalates the healthy file
    after three breaks. Here the healthy file is never charged and succeeds,
    and the killer alone escalates.
    """
    import polylogue.sources.live.parse_prefetch as parse_prefetch

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", _attempt_death_path_worker)
    (fixture,) = _write_fixture_corpus(tmp_path / "fixture", count=1)
    slow = tmp_path / "slow-healthy.jsonl"
    slow.write_bytes(fixture.read_bytes())
    killer = tmp_path / "killed-late-concurrent.json"
    killer.write_text("worker exits before parsing")
    stage = LiveParseStage(max_workers=2, shard_directory=tmp_path / "parse-shards", use_processes=True)
    slow_hash = hashlib.sha256(slow.read_bytes()).hexdigest()
    candidates = [(str(slow), Provider.CODEX, True), (str(killer), Provider.CODEX, True)]
    killer_outcomes: list[bool] = []
    slow_prepared = False
    try:
        assert stage.warm_paths(candidates) == frozenset()
        assert not stage._path_worker_losses
        assert stage._path_solo_suspects == {str(slow), str(killer)}
        for preparation in (stage.pop_path(str(slow), blob_hash=slow_hash), stage.pop_path(str(killer), blob_hash="0")):
            assert preparation is not None and preparation.deferred
        for _ in range(8):
            if killer_outcomes[-1:] == [False] and slow_prepared:
                break
            stage.warm_paths(candidates)
            assert str(slow) not in stage._path_worker_losses
            prepared = stage.pop_path(str(slow), blob_hash=slow_hash)
            if prepared is not None and prepared.error is None:
                slow_prepared = True
                prepared.discard()
            lost = stage.pop_path(str(killer), blob_hash="0")
            if lost is not None and str(killer) in stage._path_worker_losses:
                killer_outcomes.append(lost.deferred)
    finally:
        stage.shutdown()
    assert slow_prepared
    assert killer_outcomes == [True, True, False]


def test_an_escalated_worker_loss_is_not_applied_to_a_rewritten_capture(tmp_path: Path) -> None:
    """A lost-worker failure binds to the revision it failed on.

    Anti-vacuity (Codex): the escalated failure carries no blob hash, so a
    file rewritten after the failure and then captured received the old
    failure and could be quarantined without ever being parsed.
    """
    import os

    from polylogue.sources.prepared_jsonl import PreparedJsonl as LivePathPreparation

    source = tmp_path / "rewritten-after-failure.json"
    source.write_text("revision A")
    stat = source.stat()
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards")
    try:
        failure = LivePathPreparation(
            None,
            None,
            None,
            "worker lost",
            deferred=False,
            failed_observation=(stat.st_size, stat.st_mtime_ns, stat.st_ino),
        )
        stage._path_results[str(source)] = failure
        same = stage.pop_path(str(source), blob_hash="0" * 64)
        assert same is not None and same.deferred is False

        stage._path_results[str(source)] = failure
        source.write_text("revision B")
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        rewritten = stage.pop_path(str(source), blob_hash="0" * 64)
        assert rewritten is not None and rewritten.deferred is True
    finally:
        stage.shutdown()


@pytest.mark.uses_real_clock("real worker processes die under the warm")
def test_rewritten_content_starts_a_new_worker_loss_streak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Losses on earlier bytes never count against rewritten content.

    Anti-vacuity: keyed by path and size alone, two losses on the old bytes
    plus one on a same-length rewrite escalated the new content on its first
    loss.
    """
    import os

    import polylogue.sources.live.parse_prefetch as parse_prefetch

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", _attempt_death_path_worker)
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards", use_processes=True)
    killer = tmp_path / "killed-rewritten.json"
    killer.write_text("worker exits before parsing: A")
    try:
        for _ in range(parse_prefetch._MAX_WORKER_LOSSES_PER_OBSERVATION - 1):
            assert stage.warm_paths([(str(killer), Provider.CODEX, True)]) == frozenset()
            lost = stage.pop_path(str(killer), blob_hash="0" * 64)
            assert lost is not None and lost.deferred
        killer.write_text("worker exits before parsing: B")
        stat = killer.stat()
        os.utime(killer, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        assert stage.warm_paths([(str(killer), Provider.CODEX, True)]) == frozenset()
        rewritten = stage.pop_path(str(killer), blob_hash="0" * 64)
        assert rewritten is not None and rewritten.deferred
        assert stage._path_worker_losses[str(killer)][1] == 1
    finally:
        stage.shutdown()


@pytest.mark.uses_real_clock("the hang bound is measured against real worker processes")
def test_a_hung_preparation_is_stopped_at_its_hang_bound_and_charged_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A required preparation that stops advancing frees its slot and is charged.

    The hang bound scales with source size: the one-byte hung file gets about
    a second and a half, the real sibling minutes, so only the hang is
    stopped. Anti-vacuity (polylogue-7c5k6): with no hang bound the hung
    worker holds the only slot, the warm never returns and ``_warm_within``
    fails; charging collateral work instead records a loss on the sibling.
    """
    import polylogue.sources.live.parse_prefetch as parse_prefetch

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", _attempt_death_path_worker)
    monkeypatch.setattr(parse_prefetch, "_PROGRESS_POLL_SECONDS", 0.05)
    (sibling,) = _write_fixture_corpus(tmp_path / "sessions", count=1)
    assert sibling.stat().st_size > 100
    hung = tmp_path / "hung-forever.json"
    hung.write_text("x")
    stage = LiveParseStage(
        max_workers=1,
        shard_directory=tmp_path / "parse-shards",
        use_processes=True,
        hang_base_seconds=0.5,
        hang_floor_bytes_per_second=1.0,
    )
    sibling_hash = hashlib.sha256(sibling.read_bytes()).hexdigest()
    outcomes = []
    try:
        with plog.capture() as records:
            for _ in range(parse_prefetch._MAX_WORKER_LOSSES_PER_OBSERVATION):
                _warm_within(stage, [(str(hung), Provider.CODEX, True), (str(sibling), Provider.CODEX, True)])
                charged = stage.pop_path(str(hung), blob_hash="0" * 64)
                assert charged is not None and charged.error is not None
                outcomes.append(charged.deferred)
                prepared = stage.pop_path(str(sibling), blob_hash=sibling_hash)
                assert prepared is not None and prepared.error is None, prepared.error if prepared else None
                prepared.discard()
        assert outcomes == [True, True, False]
        assert set(stage._path_worker_losses) == {str(hung)}
        hung_events = [record for record in records if record["event"] == "live.parse_prefetch.preparation_hung"]
        assert [record["path"] for record in hung_events] == [str(hung)] * len(outcomes)
        assert "log.field_rejected" not in {record["event"] for record in records}
    finally:
        stage.shutdown()


@pytest.mark.uses_real_clock("a real worker process advances slowly past the hang base")
def test_a_slow_preparation_that_keeps_advancing_is_never_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hang bound measures time without progress, not time since submission.

    The worker runs three times the hang base while its attempt directory
    keeps growing. Anti-vacuity: measure from submission and it is stopped at
    the base, charged, and returned as a deferred loss instead of a
    preparation.
    """
    import polylogue.sources.live.parse_prefetch as parse_prefetch

    monkeypatch.setattr(parse_prefetch, "live_parse_path_worker", _attempt_death_path_worker)
    monkeypatch.setattr(parse_prefetch, "_PROGRESS_POLL_SECONDS", 0.05)
    (fixture,) = _write_fixture_corpus(tmp_path / "fixture", count=1)
    growing = tmp_path / "growing-slow.jsonl"
    growing.write_bytes(fixture.read_bytes())
    stage = LiveParseStage(
        max_workers=1,
        shard_directory=tmp_path / "parse-shards",
        use_processes=True,
        hang_floor_bytes_per_second=1e12,
    )
    try:
        # Start the worker process under the default bound first, so the
        # short bound below measures the slow preparation, not start-up.
        _warm_within(stage, [(str(fixture), Provider.CODEX, True)])
        assert not stage._path_worker_losses
        stage._hang_base_seconds = 1.0
        _warm_within(stage, [(str(growing), Provider.CODEX, True)])
        prepared = stage.pop_path(str(growing), blob_hash=hashlib.sha256(growing.read_bytes()).hexdigest())
        assert prepared is not None and prepared.error is None, prepared.error if prepared else None
        prepared.discard()
        assert not stage._path_worker_losses
    finally:
        stage.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_site", ["discovery", "preparation"])
async def test_preparation_exception_does_not_restore_whole_json_writer_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    import polylogue.sources.live.batch as batch
    import polylogue.sources.live.cursor as cursor_module

    monkeypatch.setattr(cursor_module, "_FULL_CURSOR_RECONCILIATION_RETRY_DELAY_S", 0)
    source = tmp_path / "inbox" / "chatgpt.json"
    source.parent.mkdir()
    payload = _chatgpt_document("preparation-retry")
    source.write_bytes(payload)
    archive_root = tmp_path / "archive"
    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "shards")
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="inbox", root=source.parent),),
        cursor=CursorStore(archive_root / "index.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
        parse_stage=stage,
        read_snapshot=open_operation_read,
    )

    def forbid_writer_decode(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise AssertionError("whole JSON decoding re-entered the writer after preparation failed")

    monkeypatch.setattr(batch, "detect_provider_from_path_sample_evidence", forbid_writer_decode)
    owner = batch if failure_site == "discovery" else stage
    method = "_live_parse_stage_path_candidates" if failure_site == "discovery" else "warm_paths"
    original = getattr(owner, method)
    failed_once = False

    def fail_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise RuntimeError("synthetic pre-writer preparation exception")
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, method, fail_once)
    try:
        first = await processor.ingest_files([source], emit_event=False)
        assert failed_once and first.failed_file_count == 0
        assert str(source) in first.deferred_paths
        with _connect(archive_root / "source.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 0
        second = await processor.ingest_files([source], emit_event=False)
        assert second.ingested_session_count == 1
        with _connect(archive_root / "source.db") as conn:
            assert [tuple(row) for row in conn.execute("SELECT origin, hex(blob_hash) FROM raw_sessions")] == [
                (origin_from_provider(Provider.CHATGPT).value, hashlib.sha256(payload).hexdigest().upper()),
            ]
    finally:
        stage.shutdown()
        await archive.close()
