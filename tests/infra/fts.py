"""Test-side FTS rebuild helper over the production FTS lifecycle owner."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from polylogue.storage.fts.freshness import record_fts_invariant_snapshot_sync
from polylogue.storage.fts.fts_lifecycle import (
    fts_invariant_snapshot_sync,
    rebuild_fts_index_sync,
    repair_fts_index_sync,
)
from polylogue.storage.search.cache import invalidate_search_cache
from polylogue.storage.sqlite.connection import connection_context


def rebuild_fts(conn: sqlite3.Connection | None = None) -> None:
    """Rebuild the whole FTS5 index from persisted blocks on the configured archive."""
    with connection_context(conn) as db_conn:
        rebuild_fts_index_sync(db_conn)
        db_conn.commit()
    invalidate_search_cache()


def repair_fts_for_sessions(session_ids: Sequence[str], conn: sqlite3.Connection | None = None) -> None:
    """Repair FTS rows for specific sessions from persisted blocks."""
    with connection_context(conn) as db_conn:
        repair_fts_index_sync(db_conn, session_ids)
        record_fts_invariant_snapshot_sync(db_conn, fts_invariant_snapshot_sync(db_conn))
        db_conn.commit()
    if session_ids:
        invalidate_search_cache()


__all__ = ["rebuild_fts", "repair_fts_for_sessions"]
