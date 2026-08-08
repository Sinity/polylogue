"""Bounded ambient blob-GC drain for the daemon.

``polylogue ops maintenance blob-gc`` is a mechanical, non-judgment
operation (delete a content-addressed blob once its refcount is zero and
it clears the generation-age gate — see ``storage/blob_gc.py``'s module
docstring for the full safety invariant list) with no daemon-side
equivalent before this module: the CLI command was the *only* way to
reclaim disk space, in the same shape ``embedding-orphan-reconcile`` was
before its own daemon loop was added. This mirrors that precedent —
periodic ambient drain in bounded batches, same as
``periodic_embedding_orphan_reconcile_check``.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.logging import get_logger
from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock

if TYPE_CHECKING:
    from polylogue.storage.blob_gc import BlobGCResult

logger = get_logger(__name__)

BLOB_GC_INTERVAL_SECONDS = 900
BLOB_GC_MAX_BATCH = 200
BLOB_PUBLICATION_RECONCILIATION_INTERVAL_SECONDS = BLOB_GC_INTERVAL_SECONDS
BLOB_PUBLICATION_RECONCILIATION_MAX_BATCH = BLOB_GC_MAX_BATCH


async def periodic_blob_gc_check(*, catch_up_complete: asyncio.Event | None = None) -> None:
    """Periodically reclaim one bounded batch of unreferenced, aged-out blobs."""
    from polylogue.daemon.cli import _await_catch_up_gate
    from polylogue.paths import archive_root, source_db_path

    await _await_catch_up_gate(catch_up_complete, loop_name="blob gc check")
    while True:
        await asyncio.sleep(BLOB_GC_INTERVAL_SECONDS)
        try:
            from polylogue.daemon.write_coordinator import daemon_write_coordinator

            result = await daemon_write_coordinator().run_sync(
                "maintenance.blob_gc",
                run_blob_gc_once,
                source_db_path(),
                archive_root() / "blob",
            )
            if result is not None and result.deleted_count:
                logger.info(
                    "blob gc: reclaimed %d blob(s), %d byte(s)",
                    result.deleted_count,
                    result.reclaimed_bytes,
                )
        except sqlite3.OperationalError as exc:
            if is_transient_sqlite_lock(exc):
                logger.info("blob gc: archive busy; retrying on next tick: %s", exc)
                continue
            logger.warning("blob gc: periodic reclaim failed", exc_info=True)
        except Exception:
            logger.warning("blob gc: periodic reclaim failed", exc_info=True)


async def periodic_blob_publication_reconciliation_check(*, catch_up_complete: asyncio.Event | None = None) -> None:
    """Periodically clear only terminal publication reservations.

    The storage reconciler retains unreferenced reservations whose blob is
    still present. Those unresolved rows are intentionally left for explicit
    abandonment policy. Referenced and blob-missing rows are safe to clear,
    but only while the archive-wide publisher exclusion is held.
    """
    from polylogue.daemon.cli import _await_catch_up_gate, _reconcile_blob_publications

    await _await_catch_up_gate(catch_up_complete, loop_name="blob publication reconciliation")
    after_publication_id: str | None = None
    while True:
        await asyncio.sleep(BLOB_PUBLICATION_RECONCILIATION_INTERVAL_SECONDS)
        try:
            outcome = await _reconcile_blob_publications(
                actor="maintenance.blob_publication_reconciliation",
                max_count=BLOB_PUBLICATION_RECONCILIATION_MAX_BATCH,
                after_publication_id=after_publication_id,
            )
            if outcome is None or outcome.scanned < BLOB_PUBLICATION_RECONCILIATION_MAX_BATCH:
                after_publication_id = None
            else:
                after_publication_id = outcome.last_scanned_publication_id
        except sqlite3.OperationalError as exc:
            if is_transient_sqlite_lock(exc):
                logger.info("blob publication reconciliation: archive busy; retrying on next tick: %s", exc)
                continue
            logger.warning("blob publication reconciliation: periodic pass failed", exc_info=True)
        except Exception:
            logger.warning("blob publication reconciliation: periodic pass failed", exc_info=True)


def run_blob_gc_once(source_db_path_arg: Path, blob_dir: Path) -> BlobGCResult | None:
    """Run one bounded daemon blob-GC pass, or ``None`` if the blob store is absent."""
    from polylogue.storage.blob_gc import run_blob_gc_report

    if not blob_dir.is_dir():
        return None
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
