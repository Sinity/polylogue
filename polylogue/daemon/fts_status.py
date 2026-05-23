"""FTS readiness projection for daemon status payloads."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel

from polylogue.storage.sqlite.connection_profile import open_readonly_connection


class FTSReadiness(BaseModel):
    messages_ready: bool = False
    action_events_ready: bool = False
    message_indexed_count: int = 0
    message_indexable_count: int = 0
    action_event_indexed_count: int = 0
    action_event_count: int = 0
    coverage_pct: float = 0.0


def fts_readiness_info(dbf: Path) -> dict[str, object]:
    """Check FTS table presence and source-count parity through bounded probes."""
    if not dbf.exists():
        return {"messages_ready": False, "action_events_ready": False, "coverage_pct": 0.0}
    try:
        conn = open_readonly_connection(dbf)
        try:
            tables = _table_names(conn)
            message_indexable_count = (
                _count(conn, "SELECT COUNT(*) FROM messages WHERE text IS NOT NULL") if "messages" in tables else 0
            )
            message_indexed_count = (
                _count(conn, "SELECT COUNT(*) FROM messages_fts_docsize") if "messages_fts_docsize" in tables else 0
            )
            action_event_count = _count(conn, "SELECT COUNT(*) FROM action_events") if "action_events" in tables else 0
            action_event_indexed_count = (
                _count(conn, "SELECT COUNT(*) FROM action_events_fts_docsize")
                if "action_events_fts_docsize" in tables
                else 0
            )
        finally:
            conn.close()
    except sqlite3.Error:
        return {"messages_ready": False, "action_events_ready": False, "coverage_pct": 0.0}

    message_ready = "messages_fts" in tables and message_indexed_count == message_indexable_count
    action_ready = "action_events_fts" in tables and action_event_indexed_count == action_event_count
    coverage_pct = 100.0 if message_indexable_count == 0 else 100 * message_indexed_count / message_indexable_count
    return {
        "messages_ready": message_ready,
        "action_events_ready": action_ready,
        "message_indexed_count": message_indexed_count,
        "message_indexable_count": message_indexable_count,
        "action_event_indexed_count": action_event_indexed_count,
        "action_event_count": action_event_count,
        "coverage_pct": round(max(0.0, coverage_pct), 1),
    }


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name IN (
            'messages',
            'messages_fts',
            'messages_fts_docsize',
            'action_events',
            'action_events_fts',
            'action_events_fts_docsize'
          )
        """
    ).fetchall()
    return {str(row[0]) for row in rows}


def _count(conn: sqlite3.Connection, sql: str) -> int:
    return int(conn.execute(sql).fetchone()[0] or 0)
