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
        3. Commits the transaction
        4. Invalidates the search cache

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

    Note
    ----
    An earlier revision of this function also acquired/released a blob-GC
    lease here, keyed by ``_blob_hashes``/``_operation_id`` payload entries.
    No production caller ever populated those keys (polylogue-v7e0), so the
    branch never executed; it was removed rather than left as unreachable
    code. GC's sole defense against a blob write racing a concurrent
    ``blob-gc`` run is now the age floor documented on
    ``polylogue.storage.blob_gc.MIN_AGE_S`` — see ``docs/internals.md``
    "GC concurrency model" for the current, lease-free contract.
    """
    from polylogue.storage.fts.fts_lifecycle import (
        ensure_fts_triggers_sync,
        repair_message_fts_index_sync,
    )

    changed_ids: Sequence[str] = payload.get("changed_session_ids", [])
    sorted_ids = sorted(set(changed_ids)) if changed_ids else []
    repair_message_fts = bool(payload.get("repair_message_fts", True))

    t_trigger = time.perf_counter()
    ensure_fts_triggers_sync(conn)
    trigger_elapsed_s = time.perf_counter() - t_trigger
    message_fts_elapsed_s = 0.0
    if sorted_ids and repair_message_fts:
        t_message = time.perf_counter()
        repair_message_fts_index_sync(conn, sorted_ids, record_exact_snapshot=False)
        message_fts_elapsed_s = time.perf_counter() - t_message
    t_commit = time.perf_counter()
    conn.commit()
    commit_elapsed_s = time.perf_counter() - t_commit
    total_effect_elapsed_s = trigger_elapsed_s + message_fts_elapsed_s + commit_elapsed_s
    if total_effect_elapsed_s >= 1.0:
        logger.info(
            "slow_archive_write_effects operation=%s sessions=%d ensure_fts_triggers_s=%.3f "
            "message_fts_s=%.3f commit_s=%.3f total_s=%.3f",
            op.value,
            len(sorted_ids),
            trigger_elapsed_s,
            message_fts_elapsed_s,
            commit_elapsed_s,
            total_effect_elapsed_s,
        )

    if sorted_ids:
        _invalidate_search_cache()

    return WriteResult(
        operation_id=str(uuid4()),
        operation=op,
        rows_affected=len(sorted_ids),
        status="committed",
    )


def _invalidate_search_cache() -> None:
    """Invalidate the search cache after archive mutations."""
    from polylogue.storage.search.cache import invalidate_search_cache

    invalidate_search_cache()


__all__ = ["commit_archive_write_effects"]
