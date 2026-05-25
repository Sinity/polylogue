from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import polylogue.sources.live.watcher as live_watcher
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.schema import _ensure_schema


class _ConversationFirstConverger:
    def __init__(self) -> None:
        self.conversation_calls: list[tuple[str, ...]] = []
        self.batch_calls: list[tuple[Path, ...]] = []

    def converge_conversations(self, conversation_ids: tuple[str, ...]) -> tuple[dict[str, object], dict[str, float]]:
        self.conversation_calls.append(conversation_ids)
        return (
            {
                conversation_id: SimpleNamespace(converged=True, stages={"fts": "done"})
                for conversation_id in conversation_ids
            },
            {"fts": 0.5},
        )

    def converge_batch(self, paths: tuple[Path, ...]) -> tuple[dict[Path, object], dict[str, float]]:
        self.batch_calls.append(paths)
        return ({path: SimpleNamespace(converged=True) for path in paths}, {"batch": 1.0})


def test_live_batch_converges_known_paths_by_conversation_id(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.sqlite"
    source = tmp_path / "session.jsonl"
    source.write_text('{"a": 1}\n')
    conn = sqlite3.connect(db_path)
    _ensure_schema(conn)
    conn.execute(
        """
        INSERT INTO raw_conversations (
            raw_id, source_name, source_path, blob_size, acquired_at
        ) VALUES ('raw-1', 'codex', ?, 0, '2026-01-01T00:00:00Z')
        """,
        (str(source),),
    )
    conn.execute(
        """
        INSERT INTO conversations (
            conversation_id, source_name, provider_conversation_id,
            source_name, content_hash, version, raw_id
        ) VALUES ('conv-1', 'codex', 'provider-1', 'codex', 'hash-1', 1, 'raw-1')
        """
    )
    conn.commit()
    conn.close()

    converger = _ConversationFirstConverger()
    processor = LiveBatchProcessor(
        MagicMock(),
        (WatchSource(name="projects", root=tmp_path),),
        cursor=CursorStore(db_path),
        converger=converger,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    completed, elapsed, timings, debts = processor._converge_paths([source])

    assert completed == {source}
    assert elapsed >= 0.0
    assert timings == {"fts": 0.5}
    assert debts == []
    assert converger.conversation_calls == [("conv-1",)]
    assert converger.batch_calls == []
