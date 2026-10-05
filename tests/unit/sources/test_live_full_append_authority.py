"""A live single-session record stream keeps its byte-revision chain.

Live intake acquires a Codex rollout before parsing it, so the full raw has no
acquired native ID. Its declared shape (one session's record stream) still
makes it that session's byte revision, and the next append extends the chain
in one pass.

Anti-vacuity: treating every native-ID-less pending envelope as a grouped
export leaves the full raw quarantined under its ``pending-raw:`` key, and the
append is acquired quarantined and deferred. Publishing the append once,
without continuing past its census phase, leaves it unapplied and deferred.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.raw_owner_routes import run_ingest_files

_BASELINE = (
    b'{"type":"session_meta","payload":{"id":"live-chain","timestamp":"2026-06-02T00:00:00Z"}}\n'
    b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
    b'"content":[{"type":"input_text","text":"zero"}]}}\n'
)
_APPEND = (
    b'{"type":"response_item","payload":{"type":"message","id":"message-1","role":"assistant",'
    b'"content":[{"type":"output_text","text":"one"}]}}\n'
)


def test_live_full_rollout_is_byte_governed_and_its_append_applies(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "live-chain.jsonl"
    path.write_bytes(_BASELINE)
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    assert run_ingest_files(processor, [path], emit_event=False).succeeded_file_count == 1
    with sqlite3.connect(tmp_path / "source.db") as source:
        full = source.execute(
            "SELECT logical_source_key, revision_kind, revision_authority FROM raw_sessions"
        ).fetchall()
    assert full == [("codex-session:live-chain", "full", "byte_proven")]

    with path.open("ab") as handle:
        handle.write(_APPEND)
    appended = run_ingest_files(processor, [path], emit_event=False)

    assert (appended.succeeded_file_count, appended.failed_file_count, appended.deferred_file_count) == (1, 0, 0)
    with sqlite3.connect(tmp_path / "index.db") as index:
        head = index.execute(
            "SELECT accepted_frontier_kind, accepted_frontier FROM raw_revision_heads "
            "WHERE logical_source_key = 'codex-session:live-chain'"
        ).fetchone()
        messages = index.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = 'codex-session:live-chain'"
        ).fetchone()[0]
    assert head == ("byte", len(_BASELINE) + len(_APPEND))
    assert messages == 2
