"""Daemon startup lineage census.

Prefix-sharing branch points have exactly one producer: the scoped,
in-transaction refinement inside ``_resolve_session_graph``
(``storage/sqlite/archive_tiers/write.py``), which runs over the session ids
the write itself touched. A branch point that still dangles once that
transaction commits means the producer left it behind, and every session on
the far side of that edge composes to its own tail only.

This module measures that condition on daemon start and **reports** it. It
deliberately corrects nothing. An archive-wide corrective sweep on every
restart repairs whatever the producer got wrong before anybody can see it, so
the producer defect never surfaces -- the mask polylogue-6kur AC4 forbids.
The index tier is rebuildable, so its recovery route is reconvergence through
the daemon, not an out-of-band edit to rows the producer owns.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from polylogue.logging import WARNING, emit
from polylogue.paths import archive_root
from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.sqlite.archive_tiers.write import count_dangling_prefix_branch_points
from polylogue.storage.sqlite.connection_profile import open_readonly_connection


@dataclass(frozen=True, slots=True)
class LineageStartupCensus:
    """What the startup census saw, as a value the caller must decide on."""

    #: Composing prefix-sharing edges whose branch point names no message row.
    dangling_edges: int
    #: Distinct sessions that therefore compose to their own tail only.
    dangling_sessions: int
    #: False when the census could not be taken at all (unreadable index).
    measured: bool = True

    @property
    def converged(self) -> bool:
        """True only when the census ran and found nothing to report."""
        return self.measured and self.dangling_edges == 0


def _open_lineage_census_connection(db_path: Path) -> sqlite3.Connection:
    """A query-only reader: this component must not be able to write at all."""
    return open_readonly_connection(db_path)


def census_lineage_startup_sync() -> LineageStartupCensus:
    """Report dangling prefix-sharing branch points. Never corrects them."""
    db = resolve_active_index_path(archive_root())
    if not db.exists():
        return LineageStartupCensus(dangling_edges=0, dangling_sessions=0)
    conn: sqlite3.Connection | None = None
    try:
        conn = _open_lineage_census_connection(db)
        edges, sessions = count_dangling_prefix_branch_points(conn)
    except Exception as exc:
        emit(
            "daemon.lineage.startup_census_failed",
            level=WARNING,
            outcome="degraded",
            reason="startup_census_failed",
            path=db,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return LineageStartupCensus(dangling_edges=0, dangling_sessions=0, measured=False)
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()
    census = LineageStartupCensus(dangling_edges=edges, dangling_sessions=sessions)
    if census.converged:
        emit("daemon.lineage.startup_census", outcome="ok", rows=0, sessions=0, path=db)
    else:
        emit(
            "daemon.lineage.startup_census",
            level=WARNING,
            outcome="degraded",
            reason="dangling_prefix_branch_points",
            rows=census.dangling_edges,
            sessions=census.dangling_sessions,
            path=db,
        )
    return census


async def census_lineage_startup() -> LineageStartupCensus:
    """Run the lineage startup census without blocking the event loop."""
    return await asyncio.to_thread(census_lineage_startup_sync)


__all__ = [
    "LineageStartupCensus",
    "census_lineage_startup",
    "census_lineage_startup_sync",
]
