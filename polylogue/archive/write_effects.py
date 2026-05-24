"""Consolidated archive write side effects.

This module is the ONLY place where post-write side effects run:
- FTS repair for changed conversation IDs
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

_MESSAGE_FTS_TRIGGER_NAMES = (
    "messages_fts_ai",
    "messages_fts_ad",
    "messages_fts_au",
    "content_blocks_fts_ai",
    "content_blocks_fts_ad",
    "content_blocks_fts_au",
)
_ACTION_FTS_TRIGGER_NAMES = ("action_events_fts_ai", "action_events_fts_ad", "action_events_fts_au")


def commit_archive_write_effects(
    conn: sqlite3.Connection,
    op: WriteOperation,
    payload: dict[str, Any],
) -> WriteResult:
    """Run the canonical post-write side effects for an archive write.

    This function:
    1. Ensures FTS triggers exist without dropping live triggers
    2. Repairs message FTS for changed conversation IDs
    3. Repairs action-event FTS for changed conversation IDs
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
        - ``changed_conversation_ids``: sequence of conversation IDs whose
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

    changed_ids: Sequence[str] = payload.get("changed_conversation_ids", [])
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
    if blob_hashes and operation_id and db_path:
        from polylogue.storage.blob_gc import acquire_blob_leases

        acquire_blob_leases(db_path, blob_hashes, operation_id)

    message_triggers_live = _all_triggers_present(conn, _MESSAGE_FTS_TRIGGER_NAMES)
    action_triggers_live = _all_triggers_present(conn, _ACTION_FTS_TRIGGER_NAMES)

    t_trigger = time.perf_counter()
    ensure_fts_triggers_sync(conn)
    trigger_elapsed_s = time.perf_counter() - t_trigger
    message_fts_elapsed_s = 0.0
    action_fts_elapsed_s = 0.0
    if sorted_ids and repair_message_fts and not message_triggers_live:
        t_message = time.perf_counter()
        repair_message_fts_index_sync(conn, sorted_ids)
        message_fts_elapsed_s = time.perf_counter() - t_message
    if sorted_ids and repair_action_fts and not action_triggers_live:
        t_action = time.perf_counter()
        repair_action_fts_index_sync(conn, sorted_ids)
        action_fts_elapsed_s = time.perf_counter() - t_action
    t_commit = time.perf_counter()
    conn.commit()
    commit_elapsed_s = time.perf_counter() - t_commit
    total_effect_elapsed_s = trigger_elapsed_s + message_fts_elapsed_s + action_fts_elapsed_s + commit_elapsed_s
    if total_effect_elapsed_s >= 1.0:
        logger.info(
            "slow_archive_write_effects operation=%s conversations=%d ensure_fts_triggers_s=%.3f "
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
    if blob_hashes and operation_id:
        from polylogue.storage.blob_gc import release_operation_leases

        release_operation_leases(conn, operation_id)
        conn.commit()

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


def _all_triggers_present(conn: sqlite3.Connection, names: Sequence[str]) -> bool:
    if not names:
        return True
    placeholders = ", ".join("?" for _ in names)
    rows = conn.execute(
        f"SELECT name FROM sqlite_master WHERE type = 'trigger' AND name IN ({placeholders})",
        tuple(names),
    ).fetchall()
    return {str(row[0]) for row in rows} == set(names)


__all__ = ["commit_archive_write_effects"]
