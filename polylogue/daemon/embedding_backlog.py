"""Bounded ambient embedding backlog drain for the daemon."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.core.enums import OperationStatus
from polylogue.logging import get_logger
from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock
from polylogue.storage.introspection import table_exists as _table_exists
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from polylogue.daemon.embedding_owner import EmbeddingConvergenceResult
    from polylogue.storage.embeddings.reconcile import EmbeddingOrphanReconcileReport

logger = get_logger(__name__)

EMBEDDING_BACKLOG_RETRY_INTERVAL_SECONDS = 60
EMBEDDING_ORPHAN_RECONCILE_INTERVAL_SECONDS = 900
EMBEDDING_ORPHAN_RECONCILE_MAX_COUNT = 500
EMBEDDING_ORPHAN_RECONCILE_QUIET_WINDOW_MS = 5 * 60 * 1000


def recover_embedding_catchup_receipts(archive_root: Path) -> int:
    """Mark unfinished catch-up receipts interrupted so the next tick retries."""
    ops_db = archive_root / "ops.db"
    if not ops_db.exists():
        return 0
    from polylogue.storage.sqlite.archive_tiers.bootstrap import open_initialized_tier_connection
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    with open_initialized_tier_connection(ops_db, ArchiveTier.OPS) as conn:
        updated = conn.execute(
            """
            UPDATE embedding_catchup_runs
            SET status = 'interrupted', finished_at_ms = ?,
                error_message = COALESCE(error_message, 'daemon restarted before catch-up receipt completed')
            WHERE status IN ('accepted', 'pending', 'running')
            """,
            (int(time.time() * 1000),),
        ).rowcount
        conn.commit()
    return int(updated)


async def periodic_embedding_backlog_check(
    *,
    catch_up_complete: asyncio.Event | None = None,
    converge: Callable[[Sequence[str] | None], Awaitable[EmbeddingConvergenceResult]] | None = None,
) -> None:
    """Periodically run the same authoritative embedding derivation as ingest."""
    from polylogue.daemon.cli import _await_catch_up_gate
    from polylogue.paths import archive_root

    db = archive_root() / "index.db"
    await _await_catch_up_gate(catch_up_complete, loop_name="embedding backlog catch-up")
    callback = converge
    if callback is None:
        # Direct daemon startup remains safe while callers migrate to retained
        # composition: it still drives the common adapter, never the former
        # backlog runner.
        from polylogue.daemon.embedding_owner import compose_embedding_convergence
        from polylogue.daemon.execution import daemon_compute_adapter
        from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge, daemon_write_coordinator

        callback = compose_embedding_convergence(
            db,
            compute_adapter=daemon_compute_adapter(),
            write_bridge=DaemonWriteThreadBridge(daemon_write_coordinator(), asyncio.get_running_loop()),
        ).callback
    while True:
        await asyncio.sleep(EMBEDDING_BACKLOG_RETRY_INTERVAL_SECONDS)
        try:
            result = await callback(None)
            if result.deferred_reason is not None:
                logger.info("embed: backlog deferred by policy: %s", result.deferred_reason)
            elif result.report is not None and result.report.done:
                logger.info("embed: converged %d message partition(s)", int(result.report.done))
        except sqlite3.OperationalError as exc:
            if is_transient_sqlite_lock(exc):
                logger.info("embed: archive busy; retrying backlog on next tick: %s", exc)
                continue
            logger.warning("embed: backlog check failed", exc_info=True)
        except Exception:
            logger.warning("embed: backlog check failed", exc_info=True)


async def periodic_embedding_orphan_reconcile_check(
    *,
    catch_up_complete: asyncio.Event | None = None,
) -> None:
    """Periodically reconcile one bounded batch of orphan embedding rows.

    An index rebuild (full re-ingest, an index reset followed by convergence, a provider
    full-replace parse) can leave ``embeddings.db`` rows pointing at
    message/session identities that no longer exist in the rebuilt
    ``index.db`` (polylogue-1dk1). This drains that debt in the background,
    the same way :func:`periodic_embedding_backlog_check` drains pending
    embed work; manual CLI (``polylogue maintenance embedding-orphan-reconcile``)
    remains a read-only diagnostic preview.
    """
    from polylogue.daemon.cli import _await_catch_up_gate
    from polylogue.paths import archive_root

    db = archive_root() / "index.db"
    await _await_catch_up_gate(catch_up_complete, loop_name="embedding orphan reconcile")
    while True:
        await asyncio.sleep(EMBEDDING_ORPHAN_RECONCILE_INTERVAL_SECONDS)
        try:
            from polylogue.daemon.write_coordinator import daemon_write_coordinator

            report = await daemon_write_coordinator().run_sync(
                "maintenance.embedding_orphan_reconcile",
                reconcile_embedding_orphans_once,
                db,
            )
            if report is not None and (report.removed_message_rows or report.removed_status_rows):
                logger.info(
                    "embed: reconciled %d orphan message row(s), %d orphan status row(s) more_pending=%s",
                    report.removed_message_rows,
                    report.removed_status_rows,
                    report.more_pending,
                )
        except sqlite3.OperationalError as exc:
            if is_transient_sqlite_lock(exc):
                logger.info("embed: archive busy; retrying orphan reconcile on next tick: %s", exc)
                continue
            logger.warning("embed: orphan reconcile check failed", exc_info=True)
        except Exception:
            logger.warning("embed: orphan reconcile check failed", exc_info=True)


def reconcile_embedding_orphans_once(db_path: Path) -> EmbeddingOrphanReconcileReport | None:
    """Run one bounded daemon orphan-reconciliation pass, or ``None`` if inapplicable."""
    index_db = _active_archive_index_path(db_path)
    if index_db is None:
        return None
    embeddings_db = db_path.parent / "embeddings.db"
    if not embeddings_db.exists():
        return None

    from polylogue.storage.embeddings.reconcile import reconcile_embedding_orphans

    return reconcile_embedding_orphans(
        index_db,
        embeddings_db,
        dry_run=False,
        max_count=EMBEDDING_ORPHAN_RECONCILE_MAX_COUNT,
        quiet_window_ms=EMBEDDING_ORPHAN_RECONCILE_QUIET_WINDOW_MS,
        mutation_authority="daemon-coordinator",
    )


def _active_archive_index_path(db_path: Path) -> Path | None:
    from polylogue.paths import archive_root
    from polylogue.storage.archive_identity import ArchiveLocation, resolve_active_index_path

    # Build candidates: active_db, then the archive-rooted-at-db_path resolution
    candidates = []
    active_db = resolve_active_index_path(archive_root())
    if active_db.name == "index.db" and active_db.exists():
        candidates.append(active_db)

    # db_path lives directly in its archive root; resolve that root's active
    # index.db (following any .index-active-pointer) as the fallback candidate.
    sibling_db = ArchiveLocation.resolve(db_path.parent).active_index_path
    if sibling_db.exists():
        candidates.append(sibling_db)

    # Use the first existing candidate and check for sessions table
    index_db = next((candidate for candidate in dict.fromkeys(candidates)), None)
    if index_db is None:
        return None
    try:
        # Presence inspection only: a schema-skewed index must still reach the
        # reconcile route, which owns the typed refusal.
        with closing(open_readonly_connection(index_db, timeout=5.0, validate_schema=False)) as conn:
            return index_db if _table_exists(conn, "sessions") else None
    except Exception:
        logger.warning("embed: failed to inspect archive index", exc_info=True)
        return None


def _archive_embedding_catchup_estimated_cost_this_month(ops_db: Path) -> float:
    if not ops_db.exists():
        return 0.0
    try:
        with closing(open_readonly_connection(ops_db, timeout=5.0)) as conn:
            if not _table_exists(conn, "embedding_catchup_runs"):
                return 0.0
            row = conn.execute(
                """
                SELECT COALESCE(SUM(estimated_cost_usd), 0.0)
                FROM embedding_catchup_runs
                WHERE strftime('%Y-%m', started_at_ms / 1000, 'unixepoch') = strftime('%Y-%m', 'now')
                """
            ).fetchone()
            return float(row[0] or 0.0) if row is not None else 0.0
    except Exception:
        logger.warning("embed: failed to inspect archive catch-up spend", exc_info=True)
        return 0.0


def _upsert_archive_embedding_catchup_run(
    ops_db: Path,
    *,
    status: OperationStatus,
    started_at_ms: int,
    run_id: str | None = None,
    finished_at_ms: int | None = None,
    scanned_sessions: int = 0,
    embedded_sessions: int = 0,
    skipped_sessions: int = 0,
    error_count: int = 0,
    embedded_messages: int = 0,
    estimated_cost_usd: float | None = None,
    error_message: str | None = None,
) -> str:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import open_initialized_tier_connection
    from polylogue.storage.sqlite.archive_tiers.ops_write import upsert_embedding_catchup_run
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    ops_db.parent.mkdir(parents=True, exist_ok=True)

    with open_initialized_tier_connection(ops_db, ArchiveTier.OPS) as conn:
        return upsert_embedding_catchup_run(
            conn,
            run_id=run_id,
            status=status,
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            scanned_sessions=scanned_sessions,
            embedded_sessions=embedded_sessions,
            skipped_sessions=skipped_sessions,
            error_count=error_count,
            embedded_messages=embedded_messages,
            estimated_cost_usd=estimated_cost_usd,
            error_message=error_message,
        )


def embedding_catchup_estimated_cost_this_month(conn: sqlite3.Connection) -> float:
    if not _table_exists(conn, "embedding_catchup_runs"):
        return 0.0
    row = conn.execute(
        """
        SELECT COALESCE(SUM(estimated_cost_usd), 0.0)
        FROM embedding_catchup_runs
        WHERE strftime('%Y-%m', started_at) = strftime('%Y-%m', 'now')
        """
    ).fetchone()
    return float(row[0] or 0.0) if row is not None else 0.0


__all__ = [
    "EMBEDDING_BACKLOG_RETRY_INTERVAL_SECONDS",
    "EMBEDDING_ORPHAN_RECONCILE_INTERVAL_SECONDS",
    "EMBEDDING_ORPHAN_RECONCILE_MAX_COUNT",
    "EMBEDDING_ORPHAN_RECONCILE_QUIET_WINDOW_MS",
    "embedding_catchup_estimated_cost_this_month",
    "periodic_embedding_backlog_check",
    "periodic_embedding_orphan_reconcile_check",
    "reconcile_embedding_orphans_once",
]
