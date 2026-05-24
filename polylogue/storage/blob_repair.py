"""Blob cleanup helpers for archive repair surfaces."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from polylogue.config import Config


@dataclass(frozen=True)
class BlobRepairOutcome:
    repaired_count: int
    success: bool
    detail: str


def count_orphaned_blobs_sync(conn: sqlite3.Connection) -> int:
    from polylogue.storage.blob_store import get_blob_store

    db_raw_ids = {row[0] for row in conn.execute("SELECT raw_id FROM raw_conversations")}
    return get_blob_store().detect_orphans(db_raw_ids).orphan_count


def repair_orphaned_blobs_data(config: Config, dry_run: bool = False) -> BlobRepairOutcome:
    from polylogue.storage.blob_store import get_blob_store
    from polylogue.storage.sqlite.connection import open_read_connection

    blob_store = get_blob_store()
    with open_read_connection(config.db_path) as conn:
        db_raw_ids = {row[0] for row in conn.execute("SELECT raw_id FROM raw_conversations")}

    detect_result = blob_store.detect_orphans(db_raw_ids)
    if detect_result.orphan_count == 0:
        return BlobRepairOutcome(0, True, "No orphaned blobs found")

    orphan_hashes = {h for h in blob_store.iter_all() if h not in db_raw_ids}
    cleanup_result = blob_store.cleanup_orphans(orphan_hashes, dry_run=dry_run)
    if dry_run:
        return BlobRepairOutcome(
            cleanup_result.would_delete_count,
            True,
            (
                f"Would: delete {cleanup_result.would_delete_count} orphaned blobs "
                f"({cleanup_result.would_delete_bytes:,} bytes)"
            ),
        )
    return BlobRepairOutcome(
        cleanup_result.deleted_count,
        cleanup_result.errors == 0,
        (
            f"Deleted {cleanup_result.deleted_count} orphaned blobs ({cleanup_result.deleted_bytes:,} bytes)"
            + (f" with {cleanup_result.errors} errors" if cleanup_result.errors else "")
        ),
    )


__all__ = ["BlobRepairOutcome", "count_orphaned_blobs_sync", "repair_orphaned_blobs_data"]
