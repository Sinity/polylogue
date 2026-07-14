"""Session-scoped FTS repair probes."""

from __future__ import annotations

import sqlite3
from typing import Any, cast


def _row_int(row: sqlite3.Row | tuple[object, ...] | None, key: int | str) -> int:
    if row is None:
        return 0
    try:
        return int(cast(Any, row)[key])
    except (TypeError, ValueError):
        return 0


def session_fts_needs_repair_sync(conn: sqlite3.Connection, session_id: str) -> bool:
    """Return whether one session has missing message FTS rows."""
    if not session_id:
        return False
    if not _table_exists(conn, "messages_fts_docsize"):
        return True
    missing_blocks = _row_int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM blocks AS b
            LEFT JOIN messages_fts_docsize AS d ON d.id = b.rowid
            WHERE b.session_id = ?
              AND d.id IS NULL
              AND b.search_text != ''
            """,
            (session_id,),
        ).fetchone(),
        0,
    )
    return missing_blocks > 0


def repair_session_fts_if_needed_sync(conn: sqlite3.Connection, session_id: str) -> bool:
    """Repair missing FTS rows for one session and report whether work ran."""
    if not session_fts_needs_repair_sync(conn, session_id):
        return False
    from polylogue.storage.fts.fts_lifecycle import repair_message_fts_index_sync
from polylogue.storage.sqlite.introspection import (
    table_exists,
)

    repair_message_fts_index_sync(conn, [session_id], record_exact_snapshot=False)
    return True


__all__ = ["repair_session_fts_if_needed_sync", "session_fts_needs_repair_sync"]
