"""Daemon startup lineage readiness checks."""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from pathlib import Path

from polylogue.logging import WARNING, emit
from polylogue.paths import archive_root
from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.sqlite.archive_tiers.write import repair_stale_prefix_branch_points
from polylogue.storage.sqlite.connection_profile import DB_TIMEOUT, open_daemon_connection


def _open_lineage_startup_write_connection(db_path: Path) -> sqlite3.Connection:
    return open_daemon_connection(db_path, timeout=DB_TIMEOUT)


def ensure_lineage_startup_readiness_sync(*, limit: int | None = None) -> int:
    """Repair bounded lineage rows that can be corrected from existing evidence."""
    db = resolve_active_index_path(archive_root())
    if not db.exists():
        return 0
    conn: sqlite3.Connection | None = None
    try:
        conn = _open_lineage_startup_write_connection(db)
        repaired = repair_stale_prefix_branch_points(conn, limit=limit)
        conn.commit()
        emit(
            "daemon.lineage.startup_repair",
            outcome="ok" if repaired else "empty",
            repaired=repaired,
            path=db,
        )
        return repaired
    except Exception as exc:
        emit(
            "daemon.lineage.startup_repair_failed",
            level=WARNING,
            outcome="degraded",
            reason="startup_readiness_failed",
            path=db,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
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
