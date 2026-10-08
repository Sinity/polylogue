"""A live batch without the daemon writer refuses its Source bodies up front.

Source SQL refuses every write without a held writer lease. A processor built
without the writer runner used to run its full-ingest body on a bare worker
thread, stage blobs, and then fail at its first durable statement with an
opaque "not authorized", counting each offered file as a failure.

Anti-vacuity: restore the ``asyncio.to_thread`` fallback for Source bodies and
the batch reports the file failed (and writes a failed cursor) instead of
raising the typed refusal.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.write_lease import UnleasedWriteError
from tests.infra.archive_templates import bootstrap_archive_root

_ROLLOUT = (
    b'{"type":"session_meta","payload":{"id":"unleased","timestamp":"2026-06-02T00:00:00Z"}}\n'
    b'{"type":"response_item","payload":{"type":"message","id":"m0","role":"user",'
    b'"content":[{"type":"input_text","text":"zero"}]}}\n'
)


@pytest.mark.asyncio
async def test_full_ingest_without_writer_runner_is_refused_before_any_source_write(tmp_path: Path) -> None:
    # Bootstrap takes the synchronous lease, which refuses to block the loop.
    await asyncio.to_thread(bootstrap_archive_root, tmp_path)
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    path = sessions / "rollout.jsonl"
    path.write_bytes(_ROLLOUT)
    cursor = await asyncio.to_thread(CursorStore, tmp_path / "index.db")
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="codex", root=sessions),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    with pytest.raises(UnleasedWriteError, match="requires the daemon writer runner"):
        await processor.ingest_files([path], emit_event=False)

    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)
        assert source.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone() == (0,)
    record = cursor.get_record(path)
    assert record is None or record.failure_count == 0
