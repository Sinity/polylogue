"""Shared archive write precedence helpers."""

from __future__ import annotations

import sqlite3


def stored_message_count(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return int(row[0] or 0) if row is not None else 0


def session_has_parser_ingest_flag(conn: sqlite3.Connection, session_id: str, flag: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM session_tags
        WHERE session_id = ?
          AND tag = ?
          AND tag_source = 'auto'
          AND method = 'parser'
        LIMIT 1
        """,
        (session_id, flag),
    ).fetchone()
    return row is not None


def record_capture_gap_event(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    existing_raw_id: str,
    incoming_raw_id: str,
    stored_message_count: int,
    incoming_message_count: int,
) -> None:
    row = conn.execute(
        """
        SELECT MAX(position) + 1
        FROM (
            SELECT position FROM session_events WHERE session_id = ?
            UNION ALL
            SELECT position FROM session_agent_policies WHERE session_id = ?
            UNION ALL
            SELECT position FROM session_provider_usage_events WHERE session_id = ?
        )
        """,
        (session_id, session_id, session_id),
    ).fetchone()
    position = int(row[0] or 0) if row is not None else 0
    summary = (
        "Skipped lower-precedence DOM browser-capture fallback "
        f"{incoming_raw_id!r}; existing raw {existing_raw_id!r} has "
        f"{stored_message_count} message(s), incoming fallback has {incoming_message_count}."
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO session_events (
            session_id, source_message_id, position, event_type, summary, occurred_at_ms
        ) VALUES (?, NULL, ?, 'capture_gap', ?, NULL)
        """,
        (session_id, position, summary),
    )


__all__ = ["record_capture_gap_event", "session_has_parser_ingest_flag", "stored_message_count"]
