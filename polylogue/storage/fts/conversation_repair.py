"""Conversation-scoped FTS repair probes."""

from __future__ import annotations

import sqlite3
from typing import Any, cast


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return bool(
        conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table_name,)).fetchone()
    )


def _row_int(row: sqlite3.Row | tuple[object, ...] | None, key: int | str) -> int:
    if row is None:
        return 0
    try:
        return int(cast(Any, row)[key])
    except (TypeError, ValueError):
        return 0


def conversation_fts_needs_repair_sync(conn: sqlite3.Connection, conversation_id: str) -> bool:
    """Return whether one conversation has missing message/action FTS rows."""
    if not conversation_id:
        return False
    if not _table_exists(conn, "messages_fts_docsize"):
        return True
    missing_messages = _row_int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM messages AS m
            LEFT JOIN messages_fts_docsize AS d ON d.id = m.rowid
            WHERE m.conversation_id = ?
              AND d.id IS NULL
              AND (
                  NULLIF(m.text, '') IS NOT NULL
                  OR EXISTS (
                      SELECT 1
                      FROM content_blocks AS cb
                      WHERE cb.message_id = m.message_id
                        AND (
                            NULLIF(cb.text, '') IS NOT NULL
                            OR NULLIF(cb.tool_input, '') IS NOT NULL
                            OR NULLIF(cb.metadata, '') IS NOT NULL
                        )
                  )
              )
            """,
            (conversation_id,),
        ).fetchone(),
        0,
    )
    if missing_messages:
        return True
    if not (
        _table_exists(conn, "action_events")
        and _table_exists(conn, "action_events_fts")
        and _table_exists(conn, "action_events_fts_docsize")
    ):
        return _table_exists(conn, "action_events")
    missing_actions = _row_int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM action_events AS ae
            LEFT JOIN action_events_fts_docsize AS d ON d.id = ae.rowid
            WHERE ae.conversation_id = ?
              AND d.id IS NULL
            """,
            (conversation_id,),
        ).fetchone(),
        0,
    )
    return missing_actions > 0


__all__ = ["conversation_fts_needs_repair_sync"]
