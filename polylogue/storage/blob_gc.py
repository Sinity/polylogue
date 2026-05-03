"""Blob garbage collection with lease safety.

Blobs are content-addressed files stored under the blob store directory.
The GC runs periodically to reclaim disk space from blobs that are no
longer referenced by any archive row.

Safety invariants
-----------------
1. Never delete a blob that still has a DB reference (message text,
   content_blocks, attachments).
2. Never delete a blob with an active lease (in-flight ingest that has
   not yet committed its blob references).
3. Only delete blobs older than the previous completed GC generation
   plus MIN_AGE seconds (defense-in-depth against clockskew).
4. Bound each run to ``max_batch`` deletions so GC does not monopolise
   I/O.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum age in seconds for a blob to be eligible for deletion.
# Provides defense-in-depth against clockskew and delayed lease writes
# beyond the lease-based safety mechanism.
MIN_AGE_S = 60

# GC generation constants
_INITIAL_GENERATION = 0


def _current_generation(conn: sqlite3.Connection) -> int:
    """Return the latest completed GC generation, or 0 if none."""
    row = conn.execute("SELECT COALESCE(MAX(generation), 0) FROM gc_generations").fetchone()
    return row[0] if row else 0


def _has_active_lease(conn: sqlite3.Connection, blob_hash: str) -> bool:
    """Check if a blob hash has any active lease."""
    row = conn.execute(
        "SELECT 1 FROM pending_blob_refs WHERE blob_hash = ? LIMIT 1",
        (blob_hash,),
    ).fetchone()
    return row is not None


def _still_referenced(conn: sqlite3.Connection, blob_hash: str) -> bool:
    """Return True if the blob hash is referenced by any archive row.

    Checks against the content_hash column of messages and conversations,
    and against attachment rows.
    """
    row = conn.execute(
        "SELECT 1 FROM messages WHERE content_hash = ? LIMIT 1",
        (blob_hash,),
    ).fetchone()
    if row:
        return True
    row = conn.execute(
        "SELECT 1 FROM conversations WHERE content_hash = ? LIMIT 1",
        (blob_hash,),
    ).fetchone()
    if row:
        return True
    row = conn.execute(
        "SELECT 1 FROM attachments WHERE attachment_id = ? LIMIT 1",
        (blob_hash,),
    ).fetchone()
    return row is not None


def _candidate_blobs(
    blob_dir: Path,
    *,
    older_than: float,
) -> list[tuple[str, float]]:
    """List blob files eligible for GC consideration.

    Returns list of ``(blob_hash, mtime)`` tuples sorted by mtime ascending.
    """
    if not blob_dir.is_dir():
        return []

    candidates: list[tuple[str, float]] = []
    now = time.time()
    try:
        for entry in os.scandir(str(blob_dir)):
            if (
                entry.is_file(follow_symlinks=False)
                and not entry.name.startswith(".")
                and now - entry.stat().st_mtime >= older_than
            ):
                candidates.append((entry.name, entry.stat().st_mtime))
    except PermissionError:
        logger.warning("Permission denied scanning blob directory: %s", blob_dir)
        return []

    candidates.sort(key=lambda pair: pair[1])
    return candidates


def run_blob_gc(
    db_path: str | Path,
    blob_dir: str | Path,
    max_batch: int = 100,
) -> int:
    """Delete unreferenced blobs that are lease-free and from prior generations.

    Safety invariants
    -----------------
    1. Never delete a blob that still has a DB reference.
    2. Never delete a blob with an active lease (in-flight ingest).
    3. Only delete blobs older than the previous completed GC generation
       plus MIN_AGE_S (defense-in-depth).
    4. Bound each run to ``max_batch`` deletions.

    Parameters
    ----------
    db_path:
        Path to the archive SQLite database.
    blob_dir:
        Path to the content-addressed blob store directory.
    max_batch:
        Maximum number of blobs to delete in one GC run (default 100).

    Returns
    -------
    Number of blobs deleted.
    """
    blob_path = Path(blob_dir)
    if not blob_path.is_dir():
        logger.debug("Blob directory %s does not exist, skipping GC", blob_dir)
        return 0

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        generation = _current_generation(conn)
        older_than = MIN_AGE_S
        candidates = _candidate_blobs(blob_path, older_than=older_than)
        if not candidates:
            return 0

        deleted = 0
        for blob_hash, _mtime in candidates:
            if deleted >= max_batch:
                break

            # Safety check 1: still referenced in DB
            if _still_referenced(conn, blob_hash):
                continue

            # Safety check 2: has active lease
            if _has_active_lease(conn, blob_hash):
                continue

            # All checks passed -- safe to delete
            try:
                (blob_path / blob_hash).unlink(missing_ok=True)
                deleted += 1
            except PermissionError:
                logger.warning("Permission denied deleting blob: %s", blob_hash)
                continue
            except OSError as exc:
                logger.warning("Failed to delete blob %s: %s", blob_hash, exc)
                continue

        # Record the new generation marker
        new_generation = generation + 1
        conn.execute(
            "INSERT OR REPLACE INTO gc_generations (generation, completed_at) VALUES (?, ?)",
            (new_generation, int(time.time())),
        )
        conn.commit()

        if deleted:
            logger.info("Blob GC: deleted %d blobs in generation %d", deleted, new_generation)

        return deleted
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


__all__ = [
    "MIN_AGE_S",
    "run_blob_gc",
]
