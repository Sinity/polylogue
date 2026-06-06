"""Consolidated archive write side effects.

This module is the ONLY place where post-write side effects run:
- FTS repair for changed session IDs
- Search cache invalidation
- Readiness recording

Every archive write path MUST route through this module or the
ArchiveWriteGateway that wraps it.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from polylogue.archive.write_gateway import WriteOperation, WriteResult

logger = logging.getLogger(__name__)


def commit_archive_write_effects(
    conn: sqlite3.Connection,
    op: WriteOperation,
    payload: dict[str, Any],
) -> WriteResult:
    """Run the canonical post-write side effects for an archive write.

    This function:
    1. Ensures FTS triggers exist without dropping live triggers
    2. Repairs message FTS for changed session IDs
    3. Repairs action-event FTS for changed session IDs
    4. Commits the transaction
    5. Invalidates the search cache

    Parameters
    ----------
    conn:
        Open SQLite connection. The caller owns the connection lifecycle.
    op:
        Write operation type (ingest, delete, tag_update, etc.).
    payload:
        Operation payload. Expected keys:
        - ``changed_session_ids``: sequence of session IDs whose
          FTS rows should be repaired.
        - ``_connection``: (optional) forwarded from the gateway when an
          external connection is already in use.

    Returns
    -------
    WriteResult with status, rows_affected, and operation_id.
    """
    from polylogue.storage.fts.fts_lifecycle import (
        ensure_fts_triggers_sync,
        repair_action_fts_index_sync,
        repair_message_fts_index_sync,
    )

    changed_ids: Sequence[str] = payload.get("changed_session_ids", [])
    sorted_ids = sorted(set(changed_ids)) if changed_ids else []
    blob_hashes: list[str] = payload.get("_blob_hashes", [])
    operation_id: str = payload.get("_operation_id", "")
    db_path: str | None = payload.get("_db_path")
    repair_message_fts = bool(payload.get("repair_message_fts", True))
    repair_action_fts = bool(payload.get("repair_action_fts", True))

    # Acquire blob GC leases before the main data commit so a concurrent GC
    # run sees them. Uses a separate connection (immediate commit) because
    # leases must be visible to other connections before *this* transaction
    # commits its blob references.
    has_lease = bool(blob_hashes and operation_id)
    if has_lease and db_path:
        from polylogue.storage.blob_gc import acquire_blob_leases

        acquire_blob_leases(db_path, blob_hashes, operation_id)

    # The lease was committed on its own connection, so rolling back ``conn``
    # on failure does NOT undo it. It must be released on every exit path or
    # the row leaks into ``pending_blob_refs`` forever and the blob it names
    # can never be GC'd (#1746). ``lease_released`` tracks the success path so
    # the ``finally`` only runs the recovery release after a failure.
    lease_released = False
    try:
        t_trigger = time.perf_counter()
        ensure_fts_triggers_sync(conn)
        trigger_elapsed_s = time.perf_counter() - t_trigger
        message_fts_elapsed_s = 0.0
        action_fts_elapsed_s = 0.0
        if sorted_ids and repair_message_fts:
            t_message = time.perf_counter()
            repair_message_fts_index_sync(conn, sorted_ids)
            message_fts_elapsed_s = time.perf_counter() - t_message
        if sorted_ids and repair_action_fts:
            t_action = time.perf_counter()
            repair_action_fts_index_sync(conn, sorted_ids)
            action_fts_elapsed_s = time.perf_counter() - t_action
        t_commit = time.perf_counter()
        conn.commit()
        commit_elapsed_s = time.perf_counter() - t_commit
        total_effect_elapsed_s = trigger_elapsed_s + message_fts_elapsed_s + action_fts_elapsed_s + commit_elapsed_s
        if total_effect_elapsed_s >= 1.0:
            logger.info(
                "slow_archive_write_effects operation=%s sessions=%d ensure_fts_triggers_s=%.3f "
                "message_fts_s=%.3f action_fts_s=%.3f commit_s=%.3f total_s=%.3f",
                op.value,
                len(sorted_ids),
                trigger_elapsed_s,
                message_fts_elapsed_s,
                action_fts_elapsed_s,
                commit_elapsed_s,
                total_effect_elapsed_s,
            )

        # Release blob GC leases after successful commit — the blob references
        # are now durable and the GC can safely clean unreferenced blobs.
        if has_lease:
            from polylogue.storage.blob_gc import release_operation_leases

            release_operation_leases(conn, operation_id)
            conn.commit()
            lease_released = True
    finally:
        if has_lease and not lease_released:
            _release_leases_on_failure(db_path, operation_id)

    if sorted_ids:
        _invalidate_search_cache()

    return WriteResult(
        operation_id=str(uuid4()),
        operation=op,
        rows_affected=len(sorted_ids),
        status="committed",
    )


def _release_leases_on_failure(db_path: str | None, operation_id: str) -> None:
    """Release leaked blob leases after a failed write-effects pass.

    The leases were committed on a separate connection, so the failing
    transaction's rollback cannot undo them. Open a fresh immediate-commit
    connection to drop them, mirroring ``acquire_blob_leases``. Any error here
    is logged and suppressed so it never masks the original failure that is
    propagating out of the ``finally``; the daemon-startup
    ``sweep_orphaned_blob_leases`` is the durable backstop if this best-effort
    release also fails.
    """
    if not db_path:
        return
    try:
        from polylogue.storage.blob_gc import release_operation_leases
        from polylogue.storage.sqlite.connection_profile import open_connection

        conn = open_connection(db_path)
        try:
            release_operation_leases(conn, operation_id)
            conn.commit()
        finally:
            conn.close()
        logger.warning(
            "Released leaked blob leases for operation %s after write-effects failure",
            operation_id,
        )
    except Exception:  # pragma: no cover - defensive: never mask the original error
        logger.warning(
            "Failed to release leaked blob leases for operation %s after write-effects failure",
            operation_id,
            exc_info=True,
        )


def _invalidate_search_cache() -> None:
    """Invalidate the search cache after archive mutations."""
    from polylogue.storage.search.cache import invalidate_search_cache

    invalidate_search_cache()


__all__ = ["commit_archive_write_effects"]
