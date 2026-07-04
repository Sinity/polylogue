from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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
    assert timings == {"batch": 1.0}
    assert debts == []
    assert converger.session_calls == []
    assert converger.batch_calls == [(source,)]
