"""Targeted SQL contracts for incremental FTS repair."""

from __future__ import annotations

import sqlite3

from polylogue.storage.fts.fts_lifecycle import restore_fts_triggers_sync
from polylogue.storage.fts.sql import (
    ACTION_FTS_REBUILD_SQL,
    delete_action_rows_sql,
    delete_conversation_rows_sql,
    insert_action_rows_sql,
)
from tests.infra.storage_records import make_conversation, make_message, store_records


def test_incremental_fts_repair_deletes_via_base_rowid(test_conn: sqlite3.Connection) -> None:
    """Incremental FTS repair must not filter FTS5 virtual tables by conversation_id."""
    message_delete_sql = " ".join(delete_conversation_rows_sql(1).split())
    action_delete_sql = " ".join(delete_action_rows_sql(1).split())

    assert "DELETE FROM messages_fts WHERE rowid IN" in message_delete_sql
    assert "DELETE FROM messages_fts WHERE conversation_id" not in message_delete_sql
    assert "DELETE FROM action_events_fts WHERE rowid IN" in action_delete_sql
    assert "DELETE FROM action_events_fts WHERE conversation_id" not in action_delete_sql
    assert "INSERT INTO action_events_fts (rowid," in " ".join(insert_action_rows_sql(1).split())
    assert "INSERT INTO action_events_fts (rowid," in " ".join(ACTION_FTS_REBUILD_SQL.split())

    plan = "\n".join(
        row[3] for row in test_conn.execute(f"EXPLAIN QUERY PLAN {delete_conversation_rows_sql(1)}", ("conv1",))
    )
    assert "SEARCH messages USING" in plan


def test_action_fts_trigger_rowids_track_action_event_rowids(test_conn: sqlite3.Connection) -> None:
    """Action-event FTS triggers use base-table rowids so rowid deletes are targeted."""
    restore_fts_triggers_sync(test_conn)
    conv = make_conversation("conv-action-rowid", title="Action rowid")
    msg = make_message("msg-action-rowid", "conv-action-rowid", text="Ran command")
    store_records(conversation=conv, messages=[msg], attachments=[], conn=test_conn)

    test_conn.execute(
        """
        INSERT INTO action_events (
            event_id, conversation_id, message_id, sequence_index,
            action_kind, normalized_tool_name, search_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "event-action-rowid",
            "conv-action-rowid",
            "msg-action-rowid",
            0,
            "shell",
            "bash",
            "pytest rowid needle",
        ),
    )
    row = test_conn.execute(
        """
        SELECT ae.rowid AS action_rowid, f.rowid AS fts_rowid
        FROM action_events ae
        JOIN action_events_fts f ON f.event_id = ae.event_id
        WHERE ae.event_id = ?
        """,
        ("event-action-rowid",),
    ).fetchone()
    assert row is not None
    assert row["action_rowid"] == row["fts_rowid"]

    test_conn.execute("DELETE FROM action_events WHERE event_id = ?", ("event-action-rowid",))
    remaining = test_conn.execute(
        "SELECT COUNT(*) FROM action_events_fts WHERE action_events_fts MATCH ?",
        ("rowid",),
    ).fetchone()[0]
    assert remaining == 0
