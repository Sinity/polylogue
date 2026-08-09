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

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from polylogue import Polylogue
from polylogue.core.enums import Provider
from polylogue.sources.live.batch import LiveBatchProcessor, _live_parse_stage_candidates
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.parse_prefetch import LiveParseStage
from polylogue.sources.live.watcher import _PARSER_FINGERPRINT, WatchSource
from polylogue.storage.raw_failure_lifecycle import read_raw_failure_lifecycle

_VOLATILE_COLUMNS: dict[str, frozenset[str]] = {
    "raw_sessions": frozenset({"acquired_at_ms", "parsed_at_ms"}),
}


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
        assert (artifact[0], artifact[1]) == ("terminal_unknown_json_decode", "decode_failed")
    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.terminal == 1
    assert lifecycle.unexplained == 0
