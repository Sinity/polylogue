from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from .action_event_rebuild_runtime import (
    action_event_repair_candidates_sync,
    rebuild_action_event_read_model_sync,
)
from .backends.connection import connection_context, open_read_connection
from .fts_lifecycle import (
    _chunked as _chunked,
)
from .fts_lifecycle import (
    ensure_fts_index_sync,
    fts_index_status_sync,
    rebuild_fts_index_sync,
    repair_fts_index_sync,
)
from .search_cache import invalidate_search_cache


def ensure_index(conn: sqlite3.Connection) -> None:
    """Ensure the FTS5 table exists on the supplied connection."""
    ensure_fts_index_sync(conn)


def rebuild_index(conn: sqlite3.Connection | None = None) -> None:
    """Rebuild the entire FTS5 search index from persisted message rows."""

    def _do(db_conn: sqlite3.Connection) -> None:
        action_targets = action_event_repair_candidates_sync(db_conn)
        if action_targets:
            rebuild_action_event_read_model_sync(db_conn, conversation_ids=action_targets)
        rebuild_fts_index_sync(db_conn)
        db_conn.commit()
        invalidate_search_cache()

    with connection_context(conn) as db_conn:
        _do(db_conn)


def update_index_for_conversations(conversation_ids: Sequence[str], conn: sqlite3.Connection | None = None) -> None:
    """Repair FTS rows for specific conversations from persisted message rows."""
    changed = bool(conversation_ids)

    def _do(db_conn: sqlite3.Connection) -> None:
        action_targets = _action_event_repair_targets_sync(db_conn, conversation_ids)
        if action_targets:
            rebuild_action_event_read_model_sync(db_conn, conversation_ids=action_targets)
        repair_fts_index_sync(db_conn, conversation_ids)
        db_conn.commit()
        if changed:
            invalidate_search_cache()

    with connection_context(conn) as db_conn:
        _do(db_conn)


def index_status(conn: sqlite3.Connection | None = None) -> dict[str, object]:
    if conn is not None:
        return fts_index_status_sync(conn)
    with open_read_connection(None) as fallback_conn:
        return fts_index_status_sync(fallback_conn)


def _action_event_repair_targets_sync(
    conn: sqlite3.Connection,
    conversation_ids: Sequence[str],
) -> list[str]:
    if not conversation_ids:
        return []
    candidate_ids = action_event_repair_candidates_sync(conn)
    if not candidate_ids:
        return []
    allowed = set(conversation_ids)
    return [conversation_id for conversation_id in candidate_ids if conversation_id in allowed]


__all__ = [
    "_chunked",
    "rebuild_index",
    "update_index_for_conversations",
    "index_status",
    "ensure_index",
]
