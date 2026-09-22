"""One ingested file costs a bounded number of ``ops.db`` connections.

Live ingest publishes its ops-tier bookkeeping through
``LiveBatchProcessor._run_ops_write``. Each publication is several ops
operations -- a full-cursor commit reads the record, upserts it and resets its
failure counters; a convergence outcome clears stale debt for the source path
and for every session the file touched before recording the new rows -- and
every one of those used to open, commit and close its own ``ops.db``
connection. Closing a WAL connection checkpoints it, so the connection churn,
not the statements, dominated: measured at ~10 ops.db write connections per
ingested file, ~60% of a warm chunk's wall clock.

The publication now runs inside one ``ops_write_scope``, which shares a
connection *without* merging transactions.

Anti-vacuity, executed against this file:

- ``test_ops_connections_per_ingested_file_stay_bounded`` measures 2.0
  connections per file here. Removing the scope from ``_run_ops_write`` (or
  removing the one in ``_record_raw_retention_outcome``) puts it back above
  the 4.0 budget -- the pre-fix value is ~10.
- ``test_batch_cursor_snapshot_is_not_re_read_per_path`` fails the moment
  ``_append_plan`` falls back to ``CursorStore.get_record`` for a path whose
  absence the batch's own ``get_records`` already established.
Two further assertions were tried here and are deliberately absent, because
each stayed green under the mutation it was meant to catch:

- A floor on ops.db commits per file, against folding the publications into
  one transaction. Deferring ``_connect_ops``'s commit to scope exit changed
  the count by zero (73 commits for six files either way): every writer
  commits explicitly inside its own block, so no commit here depends on the
  scope. The durable boundary is structural, not a number worth pinning.
- An append-route assertion against a wrong cursor snapshot. Passing
  ``cursor=None, cursor_is_known=True`` still left all six files on the
  append route, because ``_append_plan`` then resynthesizes an equivalent
  cursor from source.db. The snapshot is backstopped, so the route cannot
  witness it.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.convergence_stages import make_default_convergence_stages
from polylogue.sources.live import cursor as live_cursor
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.watcher import WatchSource

_FILES = 6
_MESSAGES_PER_SESSION = 4

# One ingested file's ops publications are the full cursor commit and the
# convergence outcome. Two connections is what sharing one per publication
# costs; the budget leaves room for a third without admitting the per-operation
# shape, which costs about ten.
_MAX_OPS_CONNECTIONS_PER_FILE = 4.0


@dataclass
class _OpsWork:
    """Connections opened and commits issued against ``ops.db`` by the store."""

    connections: int = 0
    commits: int = 0
    statements: list[str] = field(default_factory=list)


@contextmanager
def _ops_work(monkeypatch: pytest.MonkeyPatch) -> Iterator[_OpsWork]:
    """Count ``CursorStore``'s own ops-tier write connections and commits."""
    work = _OpsWork()
    real_open = live_cursor.open_connection

    def counting_open(path: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = cast(sqlite3.Connection, real_open(path, **kwargs))
        if "ops.db" in str(path):
            work.connections += 1

            def trace(statement: str) -> None:
                text = statement.strip()
                if text.startswith("--"):
                    return
                work.statements.append(text)
                if text.upper().startswith("COMMIT"):
                    work.commits += 1

            conn.set_trace_callback(trace)
        return conn

    monkeypatch.setattr(live_cursor, "open_connection", counting_open)
    yield work


def _session_records(uuid: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(_MESSAGES_PER_SESSION):
        is_user = index % 2 == 0
        records.append(
            {
                "parentUuid": None if index == 0 else f"{uuid}-msg-{index - 1:04d}",
                "sessionId": uuid,
                "type": "user" if is_user else "assistant",
                "message": {
                    "role": "user" if is_user else "assistant",
                    "content": f"Synthetic message {index} of {uuid}.",
                },
                "uuid": f"{uuid}-msg-{index:04d}",
                "timestamp": f"2026-05-05T00:00:{index:02d}.000Z",
                "cwd": "/synthetic/project",
                "version": "1.0.6",
                "isSidechain": False,
                "userType": "external",
            }
        )
    return records


def _write_session(corpus_root: Path, ordinal: int) -> Path:
    uuid = f"deadbeef-0000-0000-0000-{ordinal:012x}"
    path = corpus_root / "synthetic-project" / f"{uuid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in _session_records(uuid)),
        encoding="utf-8",
    )
    return path


class _Polylogue:
    def __init__(self, archive_root: Path, db_path: Path) -> None:
        self.archive_root = archive_root
        self.backend = SimpleNamespace(db_path=db_path)
        self.config = None


def _processor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[LiveBatchProcessor, list[Path]]:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    corpus_root = tmp_path / "corpus"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(archive_root / "polylogue.toml"))
    db_path = archive_root / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, _Polylogue(archive_root, db_path)),
        (WatchSource(name="claude-code", root=corpus_root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="ops-work-v1",
        converger=DaemonConverger(stages=make_default_convergence_stages(db_path)),
    )
    return processor, [_write_session(corpus_root, ordinal) for ordinal in range(_FILES)]


def _ingest(processor: LiveBatchProcessor, paths: list[Path]) -> None:
    metrics = asyncio.run(processor.ingest_files(paths, emit_event=False, whole_archive_convergence=False))
    assert metrics.succeeded_file_count == _FILES, metrics.succeeded_file_count


def test_ops_connections_per_ingested_file_stay_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processor, paths = _processor(tmp_path, monkeypatch)
    with _ops_work(monkeypatch) as work:
        _ingest(processor, paths)

    per_file = work.connections / _FILES
    assert per_file <= _MAX_OPS_CONNECTIONS_PER_FILE, (
        f"ingest opened {work.connections} ops.db write connections for {_FILES} files "
        f"({per_file:.2f} per file); an ops publication must share one connection"
    )


def test_batch_cursor_snapshot_is_not_re_read_per_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch that read every cursor once must not ask again path by path."""
    processor, paths = _processor(tmp_path, monkeypatch)
    single_reads: list[str] = []
    real_get_record = CursorStore.get_record

    def counting_get_record(self: CursorStore, path: Path) -> Any:
        single_reads.append(str(path))
        return real_get_record(self, path)

    monkeypatch.setattr(CursorStore, "get_record", counting_get_record)
    _ingest(processor, paths)

    assert single_reads == [], (
        "the batch already read every offered path's cursor in one query; "
        f"these per-path re-reads repeat it: {single_reads}"
    )
