"""Backfill ``sessions.created_at_ms``/``updated_at_ms`` from message evidence.

The live-write fallback (``write_parsed_session_to_archive`` in
``storage/sqlite/archive_tiers/write.py``, polylogue-m3p9) now derives a
session's ``created_at_ms``/``updated_at_ms`` from its messages' own
``occurred_at_ms`` whenever the provider payload carries no session-level
timestamp. That fix is forward-only: it only ever touches rows on their
*next* write (re-ingest, append, or full replace). This module is the
bounded backfill for the rest of the already-archived rows -- 65,946 of
83,198 sessions (79%) had NULL ``created_at_ms``/``updated_at_ms`` before
this fix landed -- so `since:` filters, recency ordering, and --by
year/month histograms stop silently excluding four-fifths of the archive
without waiting for every session to be re-ingested.

Index tier is rebuildable (index.db, schema regime "derived"): this is an
ordinary bounded UPDATE from an aggregate over ``messages``, not a durable
migration. It only ever *fills in* NULLs -- COALESCE keeps any existing
provider-supplied timestamp untouched -- so it is safe to run repeatedly and
safe to skip (the next full archive rebuild reproduces the same derivation
via the write path).

Split out from ``storage/repair.py`` for the same reason
``message_type_backfill.py`` was: keep that module under its file-size
budget (``docs/plans/file-size-budgets.yaml``) by moving one self-contained
backfill's SQL and result plumbing into its own module.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from polylogue.config import Config
from polylogue.logging import get_logger
from polylogue.maintenance.models import MaintenanceCategory
from polylogue.maintenance.targets import build_maintenance_target_catalog

logger = get_logger(__name__)
_TARGET_NAME = "session_timestamp_backfill"

# Per-session min/max of message occurred_at_ms, for sessions missing either
# session-level timestamp. Only messages that themselves carry a timestamp
# contribute to the aggregate; sessions with no timestamped messages either
# (a genuinely undatable session) are correctly excluded, not backdated.
_CANDIDATE_AGGREGATE_SQL = """
    SELECT s.session_id, MIN(m.occurred_at_ms) AS derived_created_at_ms, MAX(m.occurred_at_ms) AS derived_updated_at_ms
    FROM sessions s
    JOIN messages m ON m.session_id = s.session_id AND m.occurred_at_ms IS NOT NULL
    WHERE s.created_at_ms IS NULL OR s.updated_at_ms IS NULL
    GROUP BY s.session_id
"""


@dataclass
class BackfillResult:
    """Result of a single session-timestamp backfill pass.

    Mirrors ``storage.repair.RepairResult`` so the orchestrator in
    ``storage/repair.py`` can wrap this directly without re-coercion.
    """

    name: str
    category: MaintenanceCategory
    destructive: bool
    repaired_count: int
    success: bool
    detail: str = ""


def count_sessions_missing_timestamps_sync(conn: sqlite3.Connection) -> int:
    """Count sessions with a NULL ``created_at_ms``/``updated_at_ms`` that
    at least one of their own messages could resolve.

    This is the preview count for the backfill: it matches
    ``repaired_count`` after ``_backfill_pass`` runs, because both walk the
    exact same candidate aggregate.
    """
    return sum(1 for _row in conn.execute(_CANDIDATE_AGGREGATE_SQL))


def _backfill_pass(conn: sqlite3.Connection) -> int:
    """Fill NULL ``created_at_ms``/``updated_at_ms`` from message evidence.

    Uses SQLite's ``UPDATE ... FROM`` (3.33+) against the same aggregate
    ``count_sessions_missing_timestamps_sync`` previews, so preview and
    actual counts never drift apart. ``COALESCE`` on both sides means a
    session with one of the two timestamps already set (provider gave
    ``updated_at`` but not ``created_at``, say) only has the missing one
    filled -- the existing value is never overwritten.

    Returns the exact number of session rows touched, from
    ``cursor.rowcount`` (not ``conn.total_changes``, which is monotonic
    across the whole connection lifetime).
    """
    cursor = conn.execute(
        f"""
        UPDATE sessions
        SET created_at_ms = COALESCE(sessions.created_at_ms, agg.derived_created_at_ms),
            updated_at_ms = COALESCE(sessions.updated_at_ms, agg.derived_updated_at_ms)
        FROM ({_CANDIDATE_AGGREGATE_SQL}) AS agg
        WHERE sessions.session_id = agg.session_id
          AND (sessions.created_at_ms IS NULL OR sessions.updated_at_ms IS NULL)
        """
    )
    return cursor.rowcount


def preview_backfill(*, count: int) -> BackfillResult:
    """Preview handler for the #m3p9 session-timestamp backfill."""
    spec = build_maintenance_target_catalog().resolve_name(_TARGET_NAME)
    assert spec is not None, "session_timestamp_backfill must be registered in the catalog"
    return BackfillResult(
        name=spec.name,
        category=spec.category,
        destructive=spec.destructive,
        repaired_count=count,
        success=True,
        detail=(
            f"Would: derive created_at_ms/updated_at_ms for {count:,} session(s) from message evidence"
            if count
            else "No sessions need timestamp backfill"
        ),
    )


def run_backfill(_config: Config, *, dry_run: bool = False) -> BackfillResult:
    """Backfill NULL session timestamps from message evidence (#m3p9).

    Idempotent: sessions already fully timestamped are excluded by the
    WHERE clause on the next pass, and sessions with no timestamped
    messages either stay NULL (never backdated to the run's wall clock).
    """
    from contextlib import closing

    from polylogue.paths import active_index_db_path
    from polylogue.storage.sqlite.connection_profile import open_connection

    spec = build_maintenance_target_catalog().resolve_name(_TARGET_NAME)
    assert spec is not None, "session_timestamp_backfill must be registered in the catalog"

    try:
        with closing(open_connection(active_index_db_path())) as conn:
            conn.row_factory = sqlite3.Row
            if dry_run:
                return preview_backfill(count=count_sessions_missing_timestamps_sync(conn))

            updated = _backfill_pass(conn)
            conn.commit()
            logger.info("session_timestamp_backfill_complete", updated=updated)
            return BackfillResult(
                name=spec.name,
                category=spec.category,
                destructive=spec.destructive,
                repaired_count=updated,
                success=True,
                detail=f"Derived created_at_ms/updated_at_ms for {updated:,} session(s) from message evidence",
            )
    except Exception as exc:
        logger.exception("session_timestamp_backfill_failed", error=str(exc))
        return BackfillResult(
            name=spec.name,
            category=spec.category,
            destructive=spec.destructive,
            repaired_count=0,
            success=False,
            detail=f"Backfill failed: {exc}",
        )


__all__ = [
    "BackfillResult",
    "count_sessions_missing_timestamps_sync",
    "preview_backfill",
    "run_backfill",
]
