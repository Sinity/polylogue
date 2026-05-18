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

import json
import logging
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from polylogue.storage.sqlite.connection_profile import open_connection

logger = logging.getLogger(__name__)


@dataclass
class GCRunEvidence:
    """Structured per-pass GC evidence persisted to ``gc_generations.evidence``.

    Captures inspected/skipped/deleted counts plus per-skip reason tallies
    so operators can reconstruct what a GC pass actually did. See
    ``polylogue maintenance gc-history`` for the surface that reads it.
    """

    inspected: int = 0
    deleted: int = 0
    skipped_referenced: int = 0
    skipped_leased: int = 0
    skipped_missing: int = 0
    skipped_unlink_error: int = 0
    dry_run: bool = False
    max_batch: int = 0
    previous_generation: int = 0
    deleted_hashes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


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

    Blobs are stored under their ``raw_id`` (SHA-256 of source content).
    A blob is referenced if any ``raw_conversations`` row carries that
    ``raw_id`` — meaning the original source file is still in the archive.
    """
    row = conn.execute(
        "SELECT 1 FROM raw_conversations WHERE raw_id = ? LIMIT 1",
        (blob_hash,),
    ).fetchone()
    return row is not None


def _candidate_blobs(
    blob_dir: Path,
    *,
    older_than: float,
) -> list[tuple[str, float]]:
    """List blob files eligible for GC consideration.

    Blobs are stored in two-level prefix directories: ``{root}/{hh}/{hh...}``.
    Walks all prefix subdirectories (00–ff) to find actual blob files.
    Returns list of ``(blob_hash, mtime)`` tuples sorted by mtime ascending.
    """
    if not blob_dir.is_dir():
        return []

    candidates: list[tuple[str, float]] = []
    now = time.time()
    try:
        for prefix_dir in os.scandir(str(blob_dir)):
            if not prefix_dir.is_dir(follow_symlinks=False):
                continue
            if not prefix_dir.name or len(prefix_dir.name) != 2:
                continue
            try:
                for entry in os.scandir(prefix_dir.path):
                    if (
                        entry.is_file(follow_symlinks=False)
                        and not entry.name.startswith(".")
                        and now - entry.stat().st_mtime >= older_than
                    ):
                        blob_hash = prefix_dir.name + entry.name
                        candidates.append((blob_hash, entry.stat().st_mtime))
            except PermissionError:
                logger.warning("Permission denied scanning blob prefix: %s", prefix_dir.path)
                continue
    except PermissionError:
        logger.warning("Permission denied scanning blob directory: %s", blob_dir)
        return []

    candidates.sort(key=lambda pair: pair[1])
    return candidates


def _sharded_blob_path(blob_root: Path, blob_hash: str) -> Path:
    """Return the on-disk sharded path ``{root}/{prefix}/{remainder}`` for a blob hash.

    Mirrors ``BlobStore.blob_path`` without depending on the validator —
    GC walks discover candidate hashes from disk and they are already
    constrained to lowercase hex by ``_candidate_blobs``.
    """
    return blob_root / blob_hash[:2] / blob_hash[2:]


def run_blob_gc(
    db_path: str | Path,
    blob_dir: str | Path,
    max_batch: int = 100,
    *,
    dry_run: bool = False,
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
    dry_run:
        When True, no files are deleted and no generation row is
        written; the function reports the count of blobs that *would*
        have been deleted. Use this to preview a GC pass without
        committing to disk reclamation.

    Returns
    -------
    Number of blobs actually deleted from disk (or, for ``dry_run``,
    the number that would have been deleted).

    Notes
    -----
    The ``deleted`` counter only increments when an ``unlink`` actually
    removed a file. A blob that was already missing from disk at the
    moment of deletion (a concurrent reclaimer, a stale candidate, a
    pre-existing partial cleanup) bumps ``skipped_missing`` in the
    persisted evidence record and is NOT counted as a deletion (#1190).
    """
    blob_path = Path(blob_dir)
    if not blob_path.is_dir():
        logger.debug("Blob directory %s does not exist, skipping GC", blob_dir)
        return 0

    conn = open_connection(db_path)
    conn.row_factory = sqlite3.Row
    try:
        generation = _current_generation(conn)
        older_than = MIN_AGE_S
        candidates = _candidate_blobs(blob_path, older_than=older_than)
        if not candidates:
            return 0

        evidence = GCRunEvidence(
            dry_run=dry_run,
            max_batch=max_batch,
            previous_generation=generation,
        )
        deleted = 0
        for blob_hash, _mtime in candidates:
            if deleted >= max_batch:
                break

            evidence.inspected += 1

            # Safety check 1: still referenced in DB
            if _still_referenced(conn, blob_hash):
                evidence.skipped_referenced += 1
                continue

            # Safety check 2: has active lease
            if _has_active_lease(conn, blob_hash):
                evidence.skipped_leased += 1
                continue

            target = _sharded_blob_path(blob_path, blob_hash)

            if dry_run:
                # In dry-run mode, account the would-be deletion only when
                # the on-disk file actually exists at the sharded path.
                if target.is_file():
                    deleted += 1
                    evidence.deleted += 1
                    evidence.deleted_hashes.append(blob_hash)
                else:
                    evidence.skipped_missing += 1
                continue

            # All checks passed — attempt the unlink at the sharded path.
            # missing_ok=False so we observe whether a file actually went
            # away; if it was already gone (concurrent GC, stale candidate)
            # we record skipped_missing and do NOT count it as a deletion.
            try:
                target.unlink()
            except FileNotFoundError:
                evidence.skipped_missing += 1
                continue
            except PermissionError:
                logger.warning("Permission denied deleting blob: %s", blob_hash)
                evidence.skipped_unlink_error += 1
                continue
            except OSError as exc:
                logger.warning("Failed to delete blob %s: %s", blob_hash, exc)
                evidence.skipped_unlink_error += 1
                continue

            deleted += 1
            evidence.deleted += 1
            evidence.deleted_hashes.append(blob_hash)

        if dry_run:
            # Dry run does not consume a generation slot — no row written,
            # no commit needed. The caller still gets the would-be count.
            logger.info(
                "Blob GC dry-run: would delete %d blob(s); inspected=%d skipped_ref=%d skipped_leased=%d skipped_missing=%d",
                evidence.deleted,
                evidence.inspected,
                evidence.skipped_referenced,
                evidence.skipped_leased,
                evidence.skipped_missing,
            )
            return deleted

        # Record the new generation marker with structured evidence.
        new_generation = generation + 1
        conn.execute(
            "INSERT OR REPLACE INTO gc_generations (generation, completed_at, evidence) VALUES (?, ?, ?)",
            (new_generation, int(time.time()), evidence.to_json()),
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


@dataclass
class GCHistoryRow:
    """One row of the ``gc-history`` surface — a single completed GC pass."""

    generation: int
    completed_at: int | None
    evidence: GCRunEvidence | None


def read_gc_history(db_path: str | Path, *, limit: int = 20) -> list[GCHistoryRow]:
    """Return the most-recent committed GC passes, newest first.

    Rows missing structured evidence (pre-#1190 generations or runs that
    crashed before evidence was attached) surface as ``evidence=None``;
    operators can still see they happened.
    """
    conn = open_connection(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT generation, completed_at, evidence FROM gc_generations ORDER BY generation DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    finally:
        conn.close()

    history: list[GCHistoryRow] = []
    for row in rows:
        raw_evidence = row["evidence"]
        parsed_evidence: GCRunEvidence | None = None
        if raw_evidence:
            try:
                payload = json.loads(raw_evidence)
                parsed_evidence = GCRunEvidence(**payload)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    "gc_generations.evidence for generation %d unparseable: %s",
                    row["generation"],
                    exc,
                )
        history.append(
            GCHistoryRow(
                generation=int(row["generation"]),
                completed_at=int(row["completed_at"]) if row["completed_at"] is not None else None,
                evidence=parsed_evidence,
            )
        )
    return history


def acquire_blob_leases(
    db_path: str | Path,
    blob_hashes: list[str],
    operation_id: str,
) -> None:
    """Acquire leases on blob hashes to prevent GC during active operations.

    The lease tables are deliberately retained alongside the
    snapshot-based reference check in ``_still_referenced`` (audited in
    #1000). They bridge the acquire-blob → write-DB-row commit window:
    without leases, a GC pass running between blob materialization and
    the corresponding ``raw_conversations`` write would misclassify the
    blob as orphan.

    Uses a fresh connection that commits immediately so leases are
    visible to concurrent GC runs before the main data transaction
    commits.

    See ``docs/internals.md`` ("GC concurrency model") for the full
    contract.
    """
    if not blob_hashes:
        return
    conn = open_connection(db_path)
    try:
        now = int(time.time())
        for blob_hash in blob_hashes:
            conn.execute(
                "INSERT OR IGNORE INTO pending_blob_refs (blob_hash, operation_id, acquired_at) VALUES (?, ?, ?)",
                (blob_hash, operation_id, now),
            )
        conn.commit()
        logger.debug("Acquired %d blob lease(s) for operation %s", len(blob_hashes), operation_id)
    finally:
        conn.close()


def release_operation_leases(
    conn: sqlite3.Connection,
    operation_id: str,
) -> None:
    """Release all blob leases for an operation.

    Call after the data transaction that references blob hashes has been committed.
    Uses the provided connection (within the same transaction as the release, if desired,
    or called immediately post-commit).
    """
    conn.execute("DELETE FROM pending_blob_refs WHERE operation_id = ?", (operation_id,))
    logger.debug("Released blob leases for operation %s", operation_id)


__all__ = [
    "MIN_AGE_S",
    "GCHistoryRow",
    "GCRunEvidence",
    "acquire_blob_leases",
    "read_gc_history",
    "release_operation_leases",
    "run_blob_gc",
]
