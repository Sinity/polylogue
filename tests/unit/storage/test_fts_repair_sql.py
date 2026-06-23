"""Targeted SQL contracts for incremental FTS repair."""

from __future__ import annotations

import sqlite3

from polylogue.storage.fts.fts_lifecycle import (
    rebuild_fts_index_sync,
    repair_message_fts_index_sync,
    reset_message_fts_index_sync,
    restore_fts_triggers_sync,
)
from polylogue.storage.fts.sql import (
    delete_session_rows_sql,
    insert_session_rows_sql,
)


def _seed_text_block(conn: sqlite3.Connection, *, native_session_id: str, native_message_id: str, text: str) -> str:
    origin = "unknown-export"
    session_id = f"{origin}:{native_session_id}"
    message_id = f"{session_id}:{native_message_id}"
    content_hash = b"x" * 32
    conn.execute(
        "INSERT INTO sessions (native_id, origin, title, content_hash) VALUES (?, ?, ?, ?)",
        (native_session_id, origin, "Message repair", content_hash),
    )
    conn.execute(
        """
        INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash)
        VALUES (?, ?, 0, 'user', 'message', ?)
        """,
        (session_id, native_message_id, content_hash),
    )
    conn.execute(
        """
        INSERT INTO blocks (message_id, session_id, position, block_type, text)
        VALUES (?, ?, 0, 'text', ?)
        """,
        (message_id, session_id, text),
    )
    return message_id


def test_incremental_fts_repair_deletes_via_block_rowid(test_conn: sqlite3.Connection) -> None:
    """Incremental FTS repair is keyed by canonical block rowids."""
    message_delete_sql = " ".join(delete_session_rows_sql(1).split())

    assert "DELETE FROM messages_fts WHERE rowid IN" in message_delete_sql
    assert "DELETE FROM messages_fts WHERE session_id" not in message_delete_sql
    assert "FROM blocks" in message_delete_sql
    message_insert_sql = " ".join(insert_session_rows_sql(1).split())
    assert "SELECT DISTINCT session_id FROM raw_target_sessions" in message_insert_sql
    assert "INSERT INTO messages_fts (rowid, block_id, message_id, session_id, block_type, text)" in message_insert_sql

    plan = "\n".join(
        row[3]
        for row in test_conn.execute(
            f"EXPLAIN QUERY PLAN {delete_session_rows_sql(1)}",
            ("test:conv1",),
        )
    )
    assert "SEARCH blocks USING" in plan


def test_message_fts_repair_dedupes_duplicate_session_ids(test_conn: sqlite3.Connection) -> None:
    restore_fts_triggers_sync(test_conn)
    message_id = _seed_text_block(
        test_conn,
        native_session_id="conv-message-repair-dupe",
        native_message_id="msg-message-repair-dupe",
        text="repair duplicate needle",
    )
    session_id = "unknown-export:conv-message-repair-dupe"

    repair_message_fts_index_sync(
        test_conn,
        [
            session_id,
            session_id,
        ],
    )

    row = test_conn.execute(
        """
        SELECT COUNT(*)
        FROM messages_fts_docsize
        WHERE id = (SELECT rowid FROM blocks WHERE message_id = ?)
        """,
        (message_id,),
    ).fetchone()
    assert row[0] == 1


def test_message_fts_trigger_rowids_track_block_rowids(test_conn: sqlite3.Connection) -> None:
    """Message FTS triggers use block rowids so rowid deletes are targeted."""
    restore_fts_triggers_sync(test_conn)
    message_id = _seed_text_block(
        test_conn,
        native_session_id="conv-action-rowid",
        native_message_id="msg-action-rowid",
        text="Ran command",
    )

    # ``messages_fts`` is a contentless FTS5 table (content=''), so its stored
    # columns are not retrievable via plain SELECT — only the rowid and MATCH
    # are. The trigger keys each FTS row by the block rowid; prove the tracking
    # by matching the indexed text and comparing the matched FTS rowid to the
    # canonical block rowid.
    block_rowid = test_conn.execute(
        "SELECT rowid FROM blocks WHERE message_id = ?",
        (message_id,),
    ).fetchone()["rowid"]
    fts_rowid = test_conn.execute(
        "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?",
        ("command",),
    ).fetchone()["rowid"]
    assert fts_rowid == block_rowid

    test_conn.execute("DELETE FROM blocks WHERE message_id = ?", (message_id,))
    remaining = test_conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?",
        ("command",),
    ).fetchone()[0]
    assert remaining == 0


def test_message_fts_reset_drops_orphan_docsize_rows(test_conn: sqlite3.Connection) -> None:
    restore_fts_triggers_sync(test_conn)
    message_id = _seed_text_block(
        test_conn,
        native_session_id="conv-reset-orphan",
        native_message_id="msg-reset-orphan",
        text="reset orphan needle",
    )
    block_rowid = test_conn.execute(
        "SELECT rowid FROM blocks WHERE message_id = ?",
        (message_id,),
    ).fetchone()["rowid"]
    rebuild_fts_index_sync(test_conn)
    test_conn.execute("DROP TRIGGER messages_fts_ad")
    test_conn.execute("DELETE FROM blocks WHERE rowid = ?", (block_rowid,))

    assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 1

    reset_message_fts_index_sync(test_conn)

    assert test_conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 0
    state = test_conn.execute(
        """
        SELECT state, source_rows, indexed_rows, excess_rows, detail
        FROM fts_freshness_state
        WHERE surface = 'messages_fts'
        """
    ).fetchone()
    assert dict(state) == {
        "state": "ready",
        "source_rows": 0,
        "indexed_rows": 0,
        "excess_rows": 0,
        "detail": None,
    }
