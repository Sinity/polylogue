"""FTS5 automerge tuning for the daemon write path (#1851).

FTS5's default ``automerge=8`` setting merges segments during every write
that accumulates ≥8 level-0 segments.  On a mature archive the existing
segments are hundreds of MB; each small ingest batch adds one tiny
level-0 segment and may trigger a merge of the large existing segments,
writing 8–12 MiB to the WAL regardless of the actual append size.

The fix is two-part:

1. At daemon startup, persist ``automerge=0`` for each FTS surface so
   that no merge is ever triggered by a write.  The setting lives in the
   FTS5 ``%_config`` table and survives connection closure.

2. A periodic background pass calls ``merge=N`` (bounded work-units) for
   each surface, amortising merge cost over time instead of paying it on
   every write.
"""

from __future__ import annotations

import sqlite3
from contextlib import suppress
from pathlib import Path

from polylogue.logging import get_logger

logger = get_logger(__name__)

# FTS5 surfaces managed by this module.  Must be kept in sync with the
# trigger names in fts_lifecycle.py.
_FTS_SURFACES = ("messages_fts", "session_work_events_fts", "threads_fts")

# Work-unit budget per periodic merge call.  500 units bounds each call
# to roughly 2–4 MiB of WAL writes so the periodic merge never becomes an
# unbounded write event of its own.
_PERIODIC_MERGE_WORK_UNITS = 500


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'shadow') AND name = ? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def configure_fts_automerge_sync(conn: sqlite3.Connection) -> list[str]:
    """Persist ``automerge=0`` for each present FTS surface.

    Returns the list of surfaces that were configured.  Surfaces that are
    not yet present in the schema (fresh archive before first ingest) are
    silently skipped.

    The FTS5 ``%_config`` table stores the setting durably; it persists
    across connection cycles so this call is needed only once per daemon
    startup — not per write.
    """
    configured: list[str] = []
    for surface in _FTS_SURFACES:
        if not _table_exists(conn, surface):
            continue
        with suppress(sqlite3.OperationalError):
            conn.execute(f"INSERT INTO {surface}({surface}, rank) VALUES('automerge', 0)")
            configured.append(surface)
    if configured:
        conn.commit()
        logger.info("daemon: FTS5 automerge=0 set for %s", ", ".join(configured))
    return configured


def run_periodic_fts_merge_sync(db: Path) -> None:
    """Run a bounded FTS5 merge pass on every present surface.

    Called from the daemon's periodic maintenance loop.  Each surface gets
    at most ``_PERIODIC_MERGE_WORK_UNITS`` work units, bounding the WAL
    cost to a predictable ceiling instead of merging all pending segments
    in one unbounded pass.

    Skips gracefully if:
    - ``db`` does not exist (fresh startup race).
    - A surface is absent (not yet created by a first ingest).
    - SQLite is busy (another writer holds the lock).
    """
    if not db.exists():
        return
    conn: sqlite3.Connection | None = None
    try:
        from polylogue.storage.sqlite.connection_profile import open_connection

        conn = open_connection(db, timeout=5.0)
        merged: list[str] = []
        for surface in _FTS_SURFACES:
            if not _table_exists(conn, surface):
                continue
            with suppress(sqlite3.OperationalError):
                conn.execute(f"INSERT INTO {surface}({surface}, rank) VALUES('merge', {_PERIODIC_MERGE_WORK_UNITS})")
                merged.append(surface)
        if merged:
            conn.commit()
            logger.debug("daemon: FTS5 periodic merge (%d units) for %s", _PERIODIC_MERGE_WORK_UNITS, ", ".join(merged))
    except sqlite3.OperationalError:
        logger.debug("daemon: FTS5 periodic merge skipped — database busy")
    except Exception:
        logger.warning("daemon: FTS5 periodic merge failed", exc_info=True)
    finally:
        if conn is not None:
            with suppress(Exception):
                conn.close()


__all__ = [
    "configure_fts_automerge_sync",
    "run_periodic_fts_merge_sync",
]
