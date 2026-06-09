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
        content_hash = b"\x00" * 32
        # session_id is GENERATED as origin || ':' || native_id; insert the
        # source columns instead of the generated id.
        conn.execute(
            "INSERT INTO sessions(native_id, origin, title, content_hash, "
            "created_at_ms, updated_at_ms) "
            "VALUES('provider-c1', 'codex-session', 'one', ?, 1700000000000, 1700000000000)",
            (content_hash,),
        )
        session_id = "codex-session:provider-c1"
        conn.execute(
            "INSERT INTO messages(session_id, native_id, position, role, content_hash) VALUES(?, 'm1', 0, 'user', ?)",
            (session_id, content_hash),
        )
        # The messages_fts index is populated from blocks.search_text, so the
        # orphan docsize entry is created by inserting a block, rebuilding, then
        # deleting the block with the delete trigger dropped.
        conn.execute(
            "INSERT INTO blocks(message_id, session_id, position, block_type, text) "
            "VALUES((SELECT message_id FROM messages WHERE session_id = ? AND position = 0 AND variant_index = 0), "
            "?, 0, 'text', 'hello')",
            (session_id, session_id),
        )
        rebuild_fts_index_sync(conn)
        conn.execute("DROP TRIGGER messages_fts_ad")
        conn.execute("DELETE FROM blocks WHERE session_id = ?", (session_id,))
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
