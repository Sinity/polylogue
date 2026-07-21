"""Bounded ops.db drift-magnitude history for FTS freshness (polylogue-1xc.12).

``fts_freshness_state`` (index.db, see ``storage/fts/freshness.py``) is O(1)
*current* state -- a snapshot fit for a hot-path readiness check or a
Prometheus gauge scrape, but not a trend. This module appends a bounded
time-series sample of the same counters to ``ops.db`` (a sibling database
file) whenever an exact FTS invariant snapshot is recorded, so an operator
can see drift MAGNITUDE trend across time, not just today's boolean
ready/stale verdict.

Best-effort telemetry, matching ``record_route_observation``'s contract: a
missing or unreachable ``ops.db`` (e.g. a synthetic single-file test
fixture, or an archive mid-bootstrap) is swallowed, never raised -- this is
never allowed to turn an index.db write/repair path into an ops.db write
dependency.
"""

from __future__ import annotations

import contextlib
import sqlite3
import time
from pathlib import Path

from polylogue.logging import get_logger

logger = get_logger(__name__)

_FRESHNESS_SURFACE_COLUMNS = (
    "surface",
    "state",
    "source_rows",
    "indexed_rows",
    "missing_rows",
    "excess_rows",
    "duplicate_rows",
    "identity_mismatch_rows",
)


def _index_db_path_sync(conn: sqlite3.Connection) -> Path | None:
    """Return the main database file path backing ``conn``, if any."""
    try:
        rows = conn.execute("PRAGMA database_list").fetchall()
    except sqlite3.Error:
        logger.debug("fts drift sampling: PRAGMA database_list failed", exc_info=True)
        return None
    for row in rows:
        if str(row[1]) == "main" and row[2]:
            return Path(str(row[2]))
    return None


def sample_fts_drift_to_ops_sync(conn: sqlite3.Connection) -> int:
    """Append one ops.db drift sample per recorded FTS surface.

    Reads the just-recorded ``fts_freshness_state`` rows on ``conn`` (an
    index.db connection) and appends a bounded sample for each to the
    sibling ``ops.db``. Returns the number of samples written; returns 0 on
    any failure (missing table, missing sibling file, unreachable
    connection) without raising.
    """
    try:
        from polylogue.storage.fts.freshness import ensure_fts_freshness_table_sync

        ensure_fts_freshness_table_sync(conn)
        rows = conn.execute(f"SELECT {', '.join(_FRESHNESS_SURFACE_COLUMNS)} FROM fts_freshness_state").fetchall()
    except sqlite3.Error:
        logger.debug("fts drift sampling: could not read fts_freshness_state", exc_info=True)
        return 0
    if not rows:
        return 0

    index_db_path = _index_db_path_sync(conn)
    if index_db_path is None:
        return 0
    ops_db_path = index_db_path.with_name("ops.db")
    if not ops_db_path.exists():
        return 0

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
    from polylogue.storage.sqlite.archive_tiers.ops_write import record_fts_drift_sample
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.connection_profile import open_connection

    ops_conn: sqlite3.Connection | None = None
    try:
        ops_conn = open_connection(ops_db_path, timeout=5.0)
        initialize_archive_tier(ops_conn, ArchiveTier.OPS)
        sampled_at_ms = int(time.time() * 1000)
        written = 0
        for row in rows:
            values = dict(zip(_FRESHNESS_SURFACE_COLUMNS, row, strict=True))
            record_fts_drift_sample(
                ops_conn,
                surface=str(values["surface"]),
                state=str(values["state"]),
                source_rows=int(values["source_rows"] or 0),
                indexed_rows=int(values["indexed_rows"] or 0),
                missing_rows=int(values["missing_rows"] or 0),
                excess_rows=int(values["excess_rows"] or 0),
                duplicate_rows=int(values["duplicate_rows"] or 0),
                identity_mismatch_rows=int(values["identity_mismatch_rows"] or 0),
                sampled_at_ms=sampled_at_ms,
            )
            written += 1
        return written
    except sqlite3.Error:
        logger.debug("fts drift sampling: ops.db write failed", exc_info=True)
        return 0
    finally:
        if ops_conn is not None:
            with contextlib.suppress(sqlite3.Error):
                ops_conn.close()


__all__ = ["sample_fts_drift_to_ops_sync"]
