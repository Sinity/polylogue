"""Bounded ambient blob-GC drain for the daemon.

The loop delegates each bounded pass to the daemon write coordinator. The
storage implementation owns liveness, publication reservations, and
crash-consistent unlink accounting.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.daemon.periodic import daemon_periodic_runner, watcher_registered_gate
from polylogue.logging import span
from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock

if TYPE_CHECKING:
    from polylogue.storage.blob_gc import BlobGCResult

BLOB_GC_INTERVAL_SECONDS = 900
BLOB_GC_MAX_BATCH = 200
BLOB_PUBLICATION_RECONCILIATION_INTERVAL_SECONDS = BLOB_GC_INTERVAL_SECONDS
BLOB_PUBLICATION_RECONCILIATION_MAX_BATCH = BLOB_GC_MAX_BATCH


async def periodic_blob_gc_check(*, watcher_registered: asyncio.Event | None = None) -> None:
    """Periodically reclaim one bounded batch of unreferenced, aged-out blobs."""
    from polylogue.paths import archive_root, source_db_path

    async def once() -> None:
        with span("daemon.blob_gc.pass") as pass_span:
            try:
                from polylogue.daemon.write_coordinator import daemon_write_coordinator

                result = await daemon_write_coordinator().run_sync(
                    "maintenance.blob_gc",
                    run_blob_gc_once,
                    source_db_path(),
                    archive_root() / "blob",
                )
            except sqlite3.OperationalError as exc:
                if is_transient_sqlite_lock(exc):
                    pass_span.skipped(reason="archive_busy", error_detail=str(exc))
                else:
                    pass_span.degraded(
                        "reclaim_failed",
                        error_type=type(exc).__name__,
                        error_detail=str(exc),
                    )
            except Exception as exc:
                pass_span.degraded(
                    "reclaim_failed",
                    error_type=type(exc).__name__,
                    error_detail=str(exc),
                )
            else:
                if result is None:
                    pass_span.skipped(reason="gc_not_applicable")
                elif result.blocked_reason is not None:
                    # A refused pass is never an empty one: the counts are the
                    # work already done before the refusal, not a clean sweep.
                    pass_span.degraded(
                        "gc_blocked",
                        error_detail=result.blocked_reason,
                        removed=result.deleted_count,
                        bytes=result.reclaimed_bytes,
                    )
                elif result.deleted_count:
                    pass_span.ok(removed=result.deleted_count, bytes=result.reclaimed_bytes)
                else:
                    pass_span.empty(removed=0, bytes=0)

    await daemon_periodic_runner().run(
        "blob_gc",
        once,
        interval_s=BLOB_GC_INTERVAL_SECONDS,
        gate=watcher_registered_gate(watcher_registered),
        run_first=False,
        on_error="record",
        error_event=None,
    )


async def periodic_blob_publication_reconciliation_check(*, watcher_registered: asyncio.Event | None = None) -> None:
    """Periodically clear only terminal publication reservations.

    The storage reconciler retains unreferenced reservations whose blob is
    still present. Those unresolved rows are intentionally left for explicit
    abandonment policy. Referenced and blob-missing rows are safe to clear,
    but only while the archive-wide publisher exclusion is held.
    """
    from polylogue.daemon.cli import _reconcile_blob_publications

    after_publication_id: str | None = None

    async def once() -> None:
        nonlocal after_publication_id
        with span("daemon.blob_publication.reconcile") as pass_span:
            try:
                outcome = await _reconcile_blob_publications(
                    actor="maintenance.blob_publication_reconciliation",
                    max_count=BLOB_PUBLICATION_RECONCILIATION_MAX_BATCH,
                    after_publication_id=after_publication_id,
                )
            except sqlite3.OperationalError as exc:
                if is_transient_sqlite_lock(exc):
                    pass_span.skipped(reason="archive_busy", error_detail=str(exc))
                else:
                    pass_span.degraded(
                        "reconcile_failed",
                        error_type=type(exc).__name__,
                        error_detail=str(exc),
                    )
            except Exception as exc:
                pass_span.degraded(
                    "reconcile_failed",
                    error_type=type(exc).__name__,
                    error_detail=str(exc),
                )
            else:
                if outcome is None or outcome.scanned < BLOB_PUBLICATION_RECONCILIATION_MAX_BATCH:
                    after_publication_id = None
                else:
                    after_publication_id = outcome.last_scanned_publication_id
                if outcome is None:
                    pass_span.skipped(reason="reconciliation_not_applicable")
                elif outcome.scanned:
                    pass_span.ok(scanned=outcome.scanned, more_pending=after_publication_id is not None)
                else:
                    pass_span.empty(scanned=0)

    await daemon_periodic_runner().run(
        "blob_publication_reconciliation",
        once,
        interval_s=BLOB_PUBLICATION_RECONCILIATION_INTERVAL_SECONDS,
        gate=watcher_registered_gate(watcher_registered),
        run_first=False,
        on_error="record",
        error_event=None,
    )


def run_blob_gc_once(source_db_path_arg: Path, blob_dir: Path) -> BlobGCResult | None:
    """Run one bounded daemon GC pass, preserving namespace-loss blockers."""
    from polylogue.storage.blob_gc import run_blob_gc_report

    if not source_db_path_arg.is_file():
        return None
    return run_blob_gc_report(source_db_path_arg, blob_dir, max_batch=BLOB_GC_MAX_BATCH, dry_run=False)


__all__ = [
    "BLOB_GC_INTERVAL_SECONDS",
    "BLOB_GC_MAX_BATCH",
    "BLOB_PUBLICATION_RECONCILIATION_INTERVAL_SECONDS",
    "BLOB_PUBLICATION_RECONCILIATION_MAX_BATCH",
    "periodic_blob_gc_check",
    "periodic_blob_publication_reconciliation_check",
    "run_blob_gc_once",
]
