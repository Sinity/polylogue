"""Test-side FTS rebuild helper over the production FTS lifecycle owner."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from contextlib import closing
from pathlib import Path

from polylogue.storage.fts.fts_lifecycle import (
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
        db_conn.commit()
    if session_ids:
        invalidate_search_cache()


def completed_fts_readiness(db_path: Path, projection: Callable[[], dict[str, object]]) -> dict[str, object]:
    """Read the real completed collector while a fixture owns a stable SQLite pin.

    Opening/closing the last WAL reader can change the file fingerprint. Keep
    the canonical reader alive before submission and through projection; do not
    turn a valid refreshing observation into an asserted readiness verdict.
    """
    from polylogue.daemon.fts_status import _fts_readiness_registry
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    with closing(open_readonly_connection(db_path)) as pinned:
        pinned.execute("BEGIN")
        pinned.execute("SELECT name FROM sqlite_schema LIMIT 1").fetchone()
        registry = _fts_readiness_registry(db_path)
        registry.request_refresh("fts_readiness")
        with registry._lock:
            attempt = registry._pending["fts_readiness"]
        assert attempt.thread is not None
        attempt.thread.join()
        assert attempt.done.is_set()
        return projection()


__all__ = ["completed_fts_readiness", "rebuild_fts", "repair_fts_for_sessions"]
