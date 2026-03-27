"""Canonical FTS5 lifecycle helpers shared by sync and async surfaces."""

from __future__ import annotations

from polylogue.storage.fts_lifecycle_async import (
    ensure_fts_index_async,
    fts_index_status_async,
    rebuild_fts_index_async,
    repair_fts_index_async,
)
from polylogue.storage.fts_lifecycle_sql import chunked as _chunked
from polylogue.storage.fts_lifecycle_sync import (
    ensure_fts_index_sync,
    fts_index_status_sync,
    rebuild_fts_index_sync,
    repair_fts_index_sync,
    replace_fts_rows_for_messages_sync,
)

__all__ = [
    "_chunked",
    "ensure_fts_index_async",
    "ensure_fts_index_sync",
    "fts_index_status_async",
    "fts_index_status_sync",
    "rebuild_fts_index_async",
    "rebuild_fts_index_sync",
    "repair_fts_index_async",
    "repair_fts_index_sync",
    "replace_fts_rows_for_messages_sync",
]
