"""Lightweight SQLite maintenance helpers shared by daemon and CLI writes."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OPTIMIZE_ANALYSIS_LIMIT = 1_000
ARCHIVE_TIER_OPTIMIZE_FILES = ("source.db", "index.db", "embeddings.db", "user.db", "ops.db")


@dataclass(frozen=True, slots=True)
class SqliteOptimizeObservation:
    """Outcome of one bounded planner-stat maintenance pass."""

    reason: str
    ran: bool
    analysis_limit: int
    elapsed_s: float = 0.0
    error: str | None = None


def maybe_optimize_sqlite(
    conn: sqlite3.Connection,
    *,
    reason: str,
    analysis_limit: int = DEFAULT_OPTIMIZE_ANALYSIS_LIMIT,
) -> SqliteOptimizeObservation:
    """Run bounded ``PRAGMA optimize`` on an existing write connection.

    SQLite's optimizer stats are lightweight derived state, but letting them
    depend only on a long-lived daemon means daemon-off archives can silently
    lose planner-quality visibility. This helper is intentionally bounded and
    single-shot so normal ingest/user writes can refresh touched-table stats
    without turning into a broad ANALYZE pass.
    """

    started = time.perf_counter()
    try:
        conn.execute(f"PRAGMA analysis_limit = {int(analysis_limit)}")
        conn.execute("PRAGMA optimize")
    except sqlite3.Error as exc:
        return SqliteOptimizeObservation(
            reason=reason,
            ran=False,
            analysis_limit=analysis_limit,
            elapsed_s=round(time.perf_counter() - started, 6),
            error=str(exc),
        )
    return SqliteOptimizeObservation(
        reason=reason,
        ran=True,
        analysis_limit=analysis_limit,
        elapsed_s=round(time.perf_counter() - started, 6),
    )


def maybe_optimize_archive_tiers(
    archive_root: Path,
    *,
    reason: str,
    analysis_limit: int = DEFAULT_OPTIMIZE_ANALYSIS_LIMIT,
    timeout_s: float = 30.0,
) -> tuple[SqliteOptimizeObservation, ...]:
    """Run bounded planner-stat maintenance for every existing archive tier."""
    from polylogue.storage.sqlite.connection_profile import open_daemon_connection

    observations: list[SqliteOptimizeObservation] = []
    for filename in ARCHIVE_TIER_OPTIMIZE_FILES:
        db = archive_root / filename
        if not db.exists():
            continue
        conn: sqlite3.Connection | None = None
        try:
            conn = open_daemon_connection(db, timeout=timeout_s)
            observations.append(maybe_optimize_sqlite(conn, reason=reason, analysis_limit=analysis_limit))
        except sqlite3.Error as exc:
            observations.append(
                SqliteOptimizeObservation(
                    reason=reason,
                    ran=False,
                    analysis_limit=analysis_limit,
                    error=str(exc),
                )
            )
        finally:
            if conn is not None:
                conn.close()
    return tuple(observations)


__all__ = [
    "ARCHIVE_TIER_OPTIMIZE_FILES",
    "DEFAULT_OPTIMIZE_ANALYSIS_LIMIT",
    "SqliteOptimizeObservation",
    "maybe_optimize_archive_tiers",
    "maybe_optimize_sqlite",
]
