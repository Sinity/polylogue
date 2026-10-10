"""Complete Boolean-bearing records must not become deferred EOF frontiers."""

from __future__ import annotations

import io
import json
import sqlite3
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue.core.json_envelope import _PrefixStringReader
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.batch_support import (
    claude_semantic_frontier_for_prefix,
    decode_claude_semantic_frontier,
    jsonl_complete_prefix,
    jsonl_complete_prefix_path,
    jsonl_frontier_of_handle,
    jsonl_parse_prefix_size,
)
from polylogue.sources.live.cursor import CursorPathAuthority, CursorStore
from polylogue.sources.source_layout import export_drop_layout
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.raw_owner_routes import run_ingest_files


@pytest.mark.parametrize("tail", [b'{"flag":true}', b'{"flag":false}', b'{"flags":[true,{"done":false}]}'])
@pytest.mark.parametrize("ending", [b"", b"\n", b"\n \r\n"])
def test_complete_boolean_tail_reaches_physical_eof(tmp_path: Path, tail: bytes, ending: bytes) -> None:
    payload = b'{"first":1}\n' + tail + ending
    path = tmp_path / "session.jsonl"
    path.write_bytes(payload)
    boundary = jsonl_complete_prefix(payload)
    file_boundary = jsonl_complete_prefix_path(path)
    with path.open("rb") as handle:
        opened_boundary = jsonl_frontier_of_handle(handle, len(payload))
    for observed in (boundary, file_boundary, opened_boundary):
        assert observed.prefix_size == len(payload)
        assert not observed.incomplete_tail
        assert not observed.malformed_record
        assert jsonl_parse_prefix_size(observed, len(payload)) is None


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 19])
def test_boolean_literals_and_number_continuations_survive_lexical_chunks(chunk_size: int) -> None:
    payload = b'{"flags":[true,false,{"nested":true}],"numbers":[-1,1e-3,0.2,4E+5]}'
    reader = _PrefixStringReader(io.BytesIO(), syntax_only=True)
    transported = b"".join(reader.feed(payload[at : at + chunk_size]) for at in range(0, len(payload), chunk_size))
    transported += reader.finish()
    decoded = json.loads(transported)
    assert decoded["flags"] == [True, False, {"nested": True}]
    assert len(decoded["numbers"]) == 4


@pytest.mark.parametrize("number", [b"01", b"+1", b"-", b"1e", b"1e+", b"1e--2", b"1.2.3", b"-.1"])
@pytest.mark.parametrize("ending", [b"", b"\n"])
def test_boolean_fix_keeps_invalid_numeric_tails_unaccepted(tmp_path: Path, number: bytes, ending: bytes) -> None:
    prefix = b'{"first":1}\n'
    payload = prefix + b'{"flag":false,"number":' + number + b"}" + ending
    path = tmp_path / "invalid.jsonl"
    path.write_bytes(payload)
    boundary = jsonl_complete_prefix(payload)
    file_boundary = jsonl_complete_prefix_path(path)
    for observed in (boundary, file_boundary):
        assert observed.prefix_size == len(prefix)
        assert observed.incomplete_tail
        assert observed.malformed_record == bool(ending)
        # Finished malformed records stay visible to the strict parser;
        # an unfinished tail exposes only the already-complete prefix.
        assert jsonl_parse_prefix_size(observed, len(payload)) == (None if ending else len(prefix))


def test_parser_upgrade_reinspects_old_boolean_deferred_cursor_through_raw_owner(tmp_path: Path) -> None:
    source_root = tmp_path / "claude"
    source_root.mkdir()
    path = source_root / "old-deferred.jsonl"
    header = (
        b'{"type":"user","message":{"role":"user","content":"first"},"uuid":"m0",'
        b'"timestamp":"2026-06-02T00:00:00Z","sessionId":"old-deferred"}\n'
    )
    tail = (
        b'{"type":"assistant","message":{"role":"assistant","content":"last"},"uuid":"m1",'
        b'"timestamp":"2026-06-02T00:00:01Z","isSidechain":false}\n'
    )
    path.write_bytes(header + tail)
    initialize_active_archive_root(tmp_path)
    cursor = CursorStore(tmp_path / "ops.db")
    app = cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db")))
    sources = (WatchSource("claude-code", source_root, layout=export_drop_layout((".jsonl",))),)
    processor = LiveBatchProcessor(
        app, sources, cursor=cursor, parser_fingerprint=lambda: live_watcher._PARSER_FINGERPRINT
    )
    seeded = run_ingest_files(processor, [path], emit_event=False)
    assert seeded.succeeded_file_count == 1
    accepted = cursor.get_record(path)
    assert accepted is not None and accepted.byte_offset == path.stat().st_size
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM messages ORDER BY position").fetchall() == [("m0",), ("m1",)]
    # Represent v6's false-malformed final line without changing its retained
    # raw or the already-complete canonical messages. Only fixture state is set.
    stat = path.stat()
    authority = CursorPathAuthority.of_record(accepted)
    assert authority is not None
    assert cursor.set(
        path,
        stat.st_size,
        byte_offset=len(header),
        last_complete_newline=len(header),
        parser_fingerprint="live-batched-v6",
        content_fingerprint=accepted.content_fingerprint,
        tail_hash=claude_semantic_frontier_for_prefix(path, len(header)),
        source_name="claude-code",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=authority,
        deferred_end_offset=stat.st_size,
        allow_backward=True,
    )
    old = cursor.get_record(path)
    assert old is not None and old.deferred_end_offset == stat.st_size
    assert old.byte_offset < old.deferred_end_offset
    assert old.parser_fingerprint != live_watcher._PARSER_FINGERPRINT
    watcher = LiveWatcher(app, sources, cursor=cursor)
    try:
        assert watcher.classify_ingest_candidates([path]) == ((path,), ())
        # A version mismatch routes through normal full capture, not a cursor
        # reconciliation or an append that assumes the old body's authority.
        assert processor._append_plan(path, cursor=old, cursor_is_known=True) is None
        assert cursor.get_record(path) == old
        recovered = run_ingest_files(processor, [path], emit_event=False)
        assert recovered.full_file_count == recovered.succeeded_file_count == 1
        assert recovered.append_file_count == recovered.failed_file_count == 0
        current = cursor.get_record(path)
        assert current is not None
        assert current.parser_fingerprint == live_watcher._PARSER_FINGERPRINT
        assert current.byte_offset == current.last_complete_newline == stat.st_size
        assert current.deferred_end_offset is None
        assert current.canonical_source_path == accepted.canonical_source_path
        frontier = decode_claude_semantic_frontier(current.tail_hash)
        assert frontier is not None
        assert frontier.body_bytes == len(tail)
        assert frontier.body_sha256 == sha256(tail).hexdigest()
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
            assert conn.execute(
                "SELECT revision_authority, blob_size, parsed_at_ms IS NOT NULL FROM raw_sessions"
            ).fetchall() == [("byte_proven", stat.st_size, 1)]
            assert conn.execute("SELECT status FROM raw_authority_parser_census").fetchall() == [("complete",)]
        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute("SELECT native_id FROM messages ORDER BY position").fetchall() == [("m0",), ("m1",)]
            # A full byte-proven baseline owns the accepted Index head; it
            # need not create a semantic-membership row in Source.
            assert conn.execute(
                "SELECT accepted_frontier_kind, accepted_frontier FROM raw_revision_heads"
            ).fetchall() == [("byte", stat.st_size)]
            assert conn.execute("SELECT decision FROM raw_revision_applications").fetchall() == [("selected_baseline",)]
        assert watcher.classify_ingest_candidates([path]) == ((), ())
    finally:
        watcher.stop()
