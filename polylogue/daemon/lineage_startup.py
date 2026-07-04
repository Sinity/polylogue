"""Daemon startup lineage readiness checks."""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from pathlib import Path

from polylogue.logging import get_logger
from polylogue.paths import active_index_db_path
from polylogue.storage.sqlite.archive_tiers.write import repair_stale_prefix_branch_points
from polylogue.storage.sqlite.connection_profile import DB_TIMEOUT, open_daemon_connection

logger = get_logger(__name__)


def _open_lineage_startup_write_connection(db_path: Path) -> sqlite3.Connection:
    return open_daemon_connection(db_path, timeout=DB_TIMEOUT)


def ensure_lineage_startup_readiness_sync(*, limit: int | None = None) -> int:
    """Repair bounded lineage rows that can be corrected from existing evidence."""
    db = active_index_db_path()
    if not db.exists():
        return 0
    conn: sqlite3.Connection | None = None
    try:
        conn = _open_lineage_startup_write_connection(db)
        repaired = repair_stale_prefix_branch_points(conn, limit=limit)
        conn.commit()
        if repaired:
            logger.info("daemon: repaired %d stale prefix-sharing branch point(s) on startup", repaired)
        return repaired
    except Exception:
        logger.warning("daemon: lineage startup readiness failed", exc_info=True)
        return 0
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()


async def ensure_lineage_startup_readiness(*, limit: int | None = None) -> int:
    """Run lineage startup readiness without blocking the event loop."""
    return await asyncio.to_thread(ensure_lineage_startup_readiness_sync, limit=limit)


__all__ = [
    "ensure_lineage_startup_readiness",
    "ensure_lineage_startup_readiness_sync",
]
