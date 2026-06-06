from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.daemon.convergence_stages import make_fts_stage
from polylogue.storage.fts.fts_lifecycle import rebuild_fts_index_sync
from polylogue.storage.sqlite.schema import _ensure_schema


def _seed_archive_with_orphan_fts(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        conn.execute(
            "INSERT INTO sessions(session_id, source_name, provider_session_id, title, "
            "created_at, updated_at, content_hash, source_name, version) "
            "VALUES('c1', 'codex', 'provider-c1', 'one', '2026-01-01T00:00:00+00:00', "
            "'2026-01-01T00:00:00+00:00', 'h1', 'codex', 1)"
        )
        conn.execute(
            "INSERT INTO messages(rowid, message_id, session_id, role, text, source_name, version) "
            "VALUES(1, 'm1', 'c1', 'user', 'hello', 'codex', 1)"
        )
        rebuild_fts_index_sync(conn)
        conn.execute("DROP TRIGGER messages_fts_ad")
        conn.execute("DELETE FROM messages WHERE rowid = 1")
        conn.commit()
    finally:
        conn.close()


def test_fts_stage_detects_orphan_docsize_overcount(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.sqlite"
    _seed_archive_with_orphan_fts(db_path)

    stage = make_fts_stage(db_path)

    assert stage.check(tmp_path / "unknown-source.jsonl") is True
    assert stage.execute(tmp_path / "unknown-source.jsonl") is True

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 0
    finally:
        conn.close()
