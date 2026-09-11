from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


class _SessionFirstConverger:
    def __init__(self) -> None:
        self.session_calls: list[tuple[str, ...]] = []
        self.batch_calls: list[tuple[Path, ...]] = []

    def converge_sessions(self, session_ids: tuple[str, ...]) -> tuple[dict[str, object], dict[str, float]]:
        self.session_calls.append(session_ids)
        return (
            {session_id: SimpleNamespace(converged=True, stages={"fts": "done"}) for session_id in session_ids},
            {"fts": 0.5},
        )

    def converge_batch(self, paths: tuple[Path, ...]) -> tuple[dict[Path, object], dict[str, float]]:
        self.batch_calls.append(paths)
        return ({path: SimpleNamespace(converged=True) for path in paths}, {"batch": 1.0})


def test_live_batch_converges_known_paths_by_source_path(tmp_path: Path) -> None:
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    source = tmp_path / "session.jsonl"
    source.write_text('{"a": 1}\n')
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    with sqlite3.connect(source_db) as conn:
        raw_id = write_source_raw_session(
            conn,
            origin="codex-session",
            source_path=str(source),
            source_index=0,
            payload=b'{"a": 1}\n',
            acquired_at_ms=1_767_225_600_000,
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, title, content_hash,
                created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "provider-1",
                "codex-session",
                raw_id,
                "hot session",
                bytes([7]) * 32,
                1_767_225_600_000,
                1_767_225_600_000,
            ),
        )
        conn.commit()

    converger = _SessionFirstConverger()
    processor = LiveBatchProcessor(
        MagicMock(archive_root=tmp_path),
        (WatchSource(name="projects", root=tmp_path),),
        cursor=CursorStore(index_db),
        converger=converger,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    completed, elapsed, timings, debts = processor._converge_paths([source])

    assert completed == {source}
    assert elapsed >= 0.0
    assert timings["batch"] == 1.0
    assert timings["hook_paste_enrichment"] >= 0.0
    assert debts == []
    assert converger.session_calls == []
    assert converger.batch_calls == [(source,)]


class _GateTrackingCoordinator:
    """A write coordinator that reports whether its gate is currently held."""

    def __init__(self) -> None:
        self.depth = 0

    @property
    def held(self) -> bool:
        return self.depth > 0

    async def run(self, actor: str, operation: object) -> object:
        self.depth += 1
        try:
            return await cast(Any, operation)()
        finally:
            self.depth -= 1

    async def run_sync(self, actor: str, function: object, /, *args: object, **kwargs: object) -> object:
        self.depth += 1
        try:
            return cast(Any, function)(*args, **kwargs)
        finally:
            self.depth -= 1


@pytest.mark.asyncio
async def test_live_flush_runs_the_embedding_owner_after_releasing_the_writer_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding convergence for an ingest batch runs outside the coordinated region.

    The embed stage inside the coordinated ingest defers rather than calling a
    provider under the gate, so the owner call that follows is where the work
    actually happens -- and it must observe the gate released, or an unrelated
    archive writer would still be queued behind one network round trip.

    Anti-vacuity: move the owner call back inside ``flush_batch`` (or into
    ``_ingest_files``) and ``observed`` records ``gate_held=True``.
    """
    index_db = tmp_path / "index.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    source = tmp_path / "session.jsonl"
    source.write_text('{"a": 1}\n')

    coordinator = _GateTrackingCoordinator()
    observed: list[tuple[str, bool, tuple[Path, ...]]] = []

    async def owner(index_db_path: Path, paths: Sequence[Path], /) -> bool:
        observed.append(("embedding_owner", coordinator.held, tuple(paths)))
        return True

    watcher = live_watcher.LiveWatcher(
        MagicMock(archive_root=tmp_path),
        (WatchSource(name="projects", root=tmp_path),),
        cursor=CursorStore(index_db),
        write_coordinator=cast(Any, coordinator),
        embedding_owner=owner,
    )

    async def fake_ingest_files(paths: list[Path], **_kwargs: object) -> None:
        observed.append(("ingest", coordinator.held, tuple(paths)))
        return None

    monkeypatch.setattr(watcher, "_ingest_files", fake_ingest_files)
    monkeypatch.setattr(watcher, "_needs_work_from_state", lambda *_a, **_k: True)
    monkeypatch.setattr(watcher, "_schedule_failed_retry_scan", lambda: None)
    monkeypatch.setattr(watcher._batch_processor, "admit_paths", lambda paths: list(paths))
    monkeypatch.setattr(watcher, "_archived_cursor_reconciliation_scope", contextlib.nullcontext)
    watcher._pending_paths.add(source)

    assert await watcher._flush_pending() is True

    assert observed == [
        ("ingest", True, (source,)),
        ("embedding_owner", False, (source,)),
    ]
