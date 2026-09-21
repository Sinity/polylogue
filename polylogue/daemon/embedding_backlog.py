"""Bounded ambient embedding backlog drain for the daemon."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.core.enums import OperationStatus
from polylogue.core.sqlite_introspection import table_exists as _table_exists
from polylogue.daemon.periodic import PassOutcome, daemon_periodic_runner, watcher_registered_gate
from polylogue.logging import WARNING, emit, span
from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from polylogue.config import PolylogueConfig
    from polylogue.daemon.embedding_owner import EmbeddingConvergenceResult
    from polylogue.storage.embeddings.reconcile import EmbeddingOrphanReconcileReport

# Statuses that end a catch-up receipt. Everything else in the persisted
# vocabulary is unfinished and must be swept at startup: the receipt's own
# transitions only ever leave ``running``, so a row parked in any other
# non-terminal state was permanent debt no sweep would ever reach
# (polylogue-f7bf9). Deriving the sweep set as the complement keeps a newly
# added OperationStatus member recovered by default rather than stranded by
# an IN-list nobody updated.
TERMINAL_CATCHUP_RECEIPT_STATUSES: frozenset[str] = frozenset(
    {
        OperationStatus.COMPLETED.value,
        OperationStatus.FAILED.value,
        OperationStatus.INTERRUPTED.value,
        OperationStatus.REJECTED.value,
        "completed_with_failures",
    }
)
UNFINISHED_CATCHUP_RECEIPT_STATUSES: tuple[str, ...] = tuple(
    sorted({status.value for status in OperationStatus} - TERMINAL_CATCHUP_RECEIPT_STATUSES)
)

EMBEDDING_BACKLOG_RETRY_INTERVAL_SECONDS = 60
EMBEDDING_ORPHAN_RECONCILE_INTERVAL_SECONDS = 900
EMBEDDING_ORPHAN_RECONCILE_MAX_COUNT = 500
EMBEDDING_ORPHAN_RECONCILE_QUIET_WINDOW_MS = 5 * 60 * 1000


def embedding_convergence_unavailable_reason(config: PolylogueConfig) -> str | None:
    """Why embedding convergence can do no work this process, or ``None``.

    This is the selection-time reading of the two permanent deferrals
    :func:`polylogue.daemon.embedding_owner.compose_embedding_convergence`
    decides at composition time. Both are configuration read once per
    process, so a daemon that composes one of them will keep returning it
    unchanged until it restarts: scheduling
    :func:`periodic_embedding_backlog_check` against it buys an identical
    refusal on every ingest wake and never a pass that can progress.

    The composer owns the authority; this mirrors it so the composition root
    can withhold :attr:`~polylogue.daemon.services.ServiceCapability.EMBEDDINGS`
    before a task exists. The two are held together by
    ``test_the_capability_predicate_agrees_with_the_composed_callback``.
    """

    if not bool(config.embedding_enabled):
        return "disabled"
    if not config.voyage_api_key:
        return "provider_unavailable"
    return None


def recover_embedding_catchup_receipts(archive_root: Path) -> int:
    """Mark unfinished catch-up receipts interrupted so the next tick retries."""
    ops_db = archive_root / "ops.db"
    if not ops_db.exists():
        return 0
    from polylogue.storage.sqlite.archive_tiers.bootstrap import open_initialized_tier_connection
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    placeholders = ", ".join("?" for _ in UNFINISHED_CATCHUP_RECEIPT_STATUSES)
    with open_initialized_tier_connection(ops_db, ArchiveTier.OPS) as conn:
        updated = conn.execute(
            f"""
            UPDATE embedding_catchup_runs
            SET status = 'interrupted', finished_at_ms = ?,
                error_message = COALESCE(error_message, 'daemon restarted before catch-up receipt completed')
            WHERE status IN ({placeholders})
            """,
            (int(time.time() * 1000), *UNFINISHED_CATCHUP_RECEIPT_STATUSES),
        ).rowcount
        conn.commit()
    return int(updated)


async def periodic_embedding_backlog_check(
    *,
    watcher_registered: asyncio.Event | None = None,
    converge: Callable[[Sequence[str] | None], Awaitable[EmbeddingConvergenceResult]] | None = None,
    wakeup: asyncio.Event | None = None,
) -> None:
    """Periodically run the same authoritative embedding derivation as ingest."""
    from polylogue.paths import archive_root

    db = archive_root() / "index.db"
    resolved_callback = converge

    async def once() -> PassOutcome | None:
        """Run one backlog pass and report whether it drained the backlog.

        The return value is what gives this loop drain-cycle accounting in
        :class:`~polylogue.daemon.periodic.PeriodicRunner`. ``None`` means the
        pass reported nothing to account for: a policy deferral refused the
        work rather than finding none of it, and a failure did not establish
        either. Only a completed pass that converged zero messages is a drain.
        """
        nonlocal resolved_callback
        if resolved_callback is None:
            # Composed lazily on the first tick, i.e. right after the gate
            # this loop waits on releases -- matching the prior ordering
            # where composition ran once, immediately after the catch-up
            # wait, never before it and never once per tick.
            from polylogue.daemon.embedding_owner import compose_embedding_convergence
            from polylogue.daemon.execution import daemon_compute_adapter
            from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge, daemon_write_coordinator

            resolved_callback = compose_embedding_convergence(
                db,
                compute_adapter=daemon_compute_adapter(),
                write_bridge=DaemonWriteThreadBridge(daemon_write_coordinator(), asyncio.get_running_loop()),
            ).callback
        with span("daemon.embed.backlog_pass") as pass_span:
            try:
                result = await resolved_callback(None)
            except sqlite3.OperationalError as exc:
                if is_transient_sqlite_lock(exc):
                    pass_span.skipped(reason="archive_busy", error_detail=str(exc))
                else:
                    pass_span.degraded(
                        "backlog_check_failed",
                        error_type=type(exc).__name__,
                        error_detail=str(exc),
                    )
            except Exception as exc:
                pass_span.degraded(
                    "backlog_check_failed",
                    error_type=type(exc).__name__,
                    error_detail=str(exc),
                )
            else:
                if result.deferred_reason is not None:
                    # A policy deferral leaves the backlog unconverged. It is a
                    # refusal to do the work, never a pass that found nothing.
                    pass_span.refused("deferred_by_policy", error_detail=str(result.deferred_reason))
                elif result.report is not None and result.report.done:
                    pass_span.ok(messages=int(result.report.done))
                    return PassOutcome.PROGRESSED
                else:
                    pass_span.empty(messages=0)
                    return PassOutcome.DRAINED
            return None

    await daemon_periodic_runner().run(
        "embedding_backlog",
        once,
        interval_s=EMBEDDING_BACKLOG_RETRY_INTERVAL_SECONDS,
        gate=watcher_registered_gate(watcher_registered),
        wakeup=wakeup,
        run_first=False,
        on_error="record",
        error_event=None,
    )


async def periodic_embedding_orphan_reconcile_check(
    *,
    watcher_registered: asyncio.Event | None = None,
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
    from polylogue.paths import archive_root

    db = archive_root() / "index.db"

    async def once() -> None:
        with span("daemon.embed.orphan_reconcile") as pass_span:
            try:
                from polylogue.daemon.write_coordinator import daemon_write_coordinator

                report = await daemon_write_coordinator().run_sync(
                    "maintenance.embedding_orphan_reconcile",
                    reconcile_embedding_orphans_once,
                    db,
                )
            except sqlite3.OperationalError as exc:
                if is_transient_sqlite_lock(exc):
                    pass_span.skipped(reason="archive_busy", error_detail=str(exc))
                else:
                    pass_span.degraded(
                        "orphan_reconcile_failed",
                        error_type=type(exc).__name__,
                        error_detail=str(exc),
                    )
            except Exception as exc:
                pass_span.degraded(
                    "orphan_reconcile_failed",
                    error_type=type(exc).__name__,
                    error_detail=str(exc),
                )
            else:
                if report is None:
                    pass_span.skipped(reason="reconcile_not_applicable")
                    return
                removed = int(report.removed_message_rows) + int(report.removed_status_rows)
                if removed:
                    pass_span.ok(
                        removed=removed,
                        messages=int(report.removed_message_rows),
                        rows=int(report.removed_status_rows),
                        more_pending=bool(report.more_pending),
                    )
                else:
                    pass_span.empty(removed=0, more_pending=bool(report.more_pending))

    await daemon_periodic_runner().run(
        "embedding_orphan_reconcile",
        once,
        interval_s=EMBEDDING_ORPHAN_RECONCILE_INTERVAL_SECONDS,
        gate=watcher_registered_gate(watcher_registered),
        run_first=False,
        on_error="record",
        error_event=None,
    )


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
    except Exception as exc:
        emit(
            "daemon.embed.index_probe_failed",
            level=WARNING,
            outcome="degraded",
            reason="index_unreadable",
            path=index_db,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
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
    except Exception as exc:
        emit(
            "daemon.embed.spend_probe_failed",
            level=WARNING,
            outcome="degraded",
            reason="catchup_spend_unreadable",
            path=ops_db,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
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
    "TERMINAL_CATCHUP_RECEIPT_STATUSES",
    "UNFINISHED_CATCHUP_RECEIPT_STATUSES",
    "embedding_catchup_estimated_cost_this_month",
    "embedding_convergence_unavailable_reason",
    "periodic_embedding_backlog_check",
    "periodic_embedding_orphan_reconcile_check",
    "reconcile_embedding_orphans_once",
]
