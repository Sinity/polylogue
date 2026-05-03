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
    1. Restores FTS triggers (suspended during bulk writes)
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
        repair_fts_index_sync,
        restore_fts_triggers_sync,
    )

    changed_ids: Sequence[str] = payload.get("changed_conversation_ids", [])
    sorted_ids = sorted(set(changed_ids)) if changed_ids else []

    restore_fts_triggers_sync(conn)
    if sorted_ids:
        repair_fts_index_sync(conn, sorted_ids)
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


__all__ = ["commit_archive_write_effects"]
