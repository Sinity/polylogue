from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from polylogue.sources.live import hook_paste_enrichment
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def test_hook_paste_enrichment_updates_archive_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_db = tmp_path / "index.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    hook_time_ms = int(datetime(2026, 5, 7, 12, 0, tzinfo=UTC).timestamp() * 1000)
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, content_hash, created_at_ms, updated_at_ms
            ) VALUES ('codex-native-1', 'codex-session', ?, ?, ?)
            """,
            (b"s" * 32, hook_time_ms, hook_time_ms),
        )
        conn.execute(
            """
            INSERT INTO messages (
                session_id, native_id, position, role, content_hash, occurred_at_ms
            ) VALUES ('codex-session:codex-native-1', 'm1', 0, 'user', ?, ?)
            """,
            (b"m" * 32, hook_time_ms + 100),
        )
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "event_type": "UserPromptSubmit",
                "timestamp": "2026-05-07T12:00:00Z",
                "payload": {
                    "session_id": "codex-native-1",
                    "prompt": "Inspect [Pasted text #1]",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(hook_paste_enrichment, "hooks_sidecar_dir", lambda: hooks_dir)

    updated = hook_paste_enrichment.enrich_paste_from_hooks(tmp_path / "ops.db")

    assert updated == 1
    with sqlite3.connect(index_db) as conn:
        message = conn.execute(
            """
            SELECT has_paste, paste_boundary
            FROM messages
            WHERE session_id = 'codex-session:codex-native-1'
            """
        ).fetchone()
        assert message == (1, "hash_only")
        session = conn.execute(
            "SELECT paste_count FROM sessions WHERE session_id = 'codex-session:codex-native-1'"
        ).fetchone()
        assert session == (1,)
        span = conn.execute(
            """
            SELECT start_offset, end_offset, boundary
            FROM paste_spans
            WHERE session_id = 'codex-session:codex-native-1'
            """
        ).fetchone()
        assert span == (0, 0, "hash_only")
