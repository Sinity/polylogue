"""Blob garbage collection.

Blobs are content-addressed files stored under the blob store directory.
The GC runs periodically to reclaim disk space from blobs that are no
longer referenced by any archive row.

Safety invariants
-----------------
1. Never delete a blob that still has a DB reference (message text,
   content_blocks, attachments).
2. Never delete a blob with a durable publication receipt. Archive
   orchestration commits per-publication receipt IDs before exposing final
   paths, and exact reference transactions consume only their own receipts.
3. Serialize the final reference/reservation recheck and unlink under the
   source-tier write lock.
4. Only delete blobs older than the previous completed GC generation
   plus MIN_AGE seconds (defense-in-depth against clockskew and any
   uninstrumented writer path).
5. Bound each run to ``max_batch`` deletions so GC does not monopolise
   I/O.

A prior revision carried a late write-effects lease mechanism
(``pending_blob_refs``, ``acquire_blob_leases``/``release_operation_leases``)
that no production caller populated. Source schema v4 replaces it with an
archive-owned batched publisher over a substrate-neutral BlobStore, closing
the actual final-path-visibility -> reference-commit window.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from polylogue.storage.sqlite.connection_profile import open_connection

logger = logging.getLogger(__name__)


@dataclass
class GCRunEvidence:
    """In-memory per-pass GC tally used for operator log lines.

    Aggregates inspected/skipped/deleted counts during a ``run_blob_gc`` pass
    so the summary log line can describe what happened. The durable record is
    the typed ``gc_generations`` row (``reclaimed_count`` / ``reclaimed_bytes``);
    this tally is not persisted (#1743 — no JSON evidence escape hatch).
    """

    inspected: int = 0
    deleted: int = 0
    skipped_referenced: int = 0
    skipped_reserved: int = 0
    skipped_missing: int = 0
    skipped_unlink_error: int = 0
    dry_run: bool = False
    max_batch: int = 0


@dataclass
class BlobGCResult:
    """Machine-readable summary of one blob-GC pass."""

    db_path: str
    blob_dir: str
    dry_run: bool
    max_batch: int
    candidate_count: int = 0
    inspected_count: int = 0
    would_delete_count: int = 0
    deleted_count: int = 0
    reclaimed_bytes: int = 0
    skipped_referenced: int = 0
    skipped_reserved: int = 0
    skipped_missing: int = 0
    skipped_unlink_error: int = 0
    generation_id: str | None = None
    generation_written: bool = False
    older_than_s: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "db_path": self.db_path,
            "blob_dir": self.blob_dir,
            "dry_run": self.dry_run,
            "max_batch": self.max_batch,
            "candidate_count": self.candidate_count,
            "inspected_count": self.inspected_count,
            "would_delete_count": self.would_delete_count,
            "deleted_count": self.deleted_count,
            "reclaimed_bytes": self.reclaimed_bytes,
            "skipped_referenced": self.skipped_referenced,
            "skipped_reserved": self.skipped_reserved,
            "skipped_missing": self.skipped_missing,
            "skipped_unlink_error": self.skipped_unlink_error,
            "generation_id": self.generation_id,
            "generation_written": self.generation_written,
            "older_than_s": self.older_than_s,
        }


# Minimum age in seconds for a blob to be eligible for deletion.
#
# Publication reservations provide the exact acquire-to-reference defense.
# This floor remains defense-in-depth for legacy/uninstrumented paths and
# clock skew; it is not used to infer that a live publisher has expired.
MIN_AGE_S = 60


def _previous_generation_completed_at(conn: sqlite3.Connection) -> int | None:
    """Return the completion epoch (seconds) of the latest completed generation.

    Used by ``run_blob_gc`` to enforce safety invariant #2: a blob must be
    older than the previous generation's completion before it is eligible for
    deletion (defense-in-depth against clock skew).

    The durable column is ``completed_at_ms``; this returns whole seconds so the
    age gate can compare directly against ``time.time()``. In-progress
    generations (``completed_at_ms IS NULL``) are ignored.
    """
    row = conn.execute(
        "SELECT completed_at_ms FROM gc_generations "
        "WHERE completed_at_ms IS NOT NULL "
        "ORDER BY completed_at_ms DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    completed_at_ms = row[0]
    return int(completed_at_ms) // 1000 if completed_at_ms is not None else None


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _database_has_table(path: Path, table: str) -> bool:
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return False
    try:
        return _table_exists(conn, table)
    finally:
        conn.close()


def _blob_hash_bytes(blob_hash: str) -> bytes | None:
    if len(blob_hash) != 64:
        return None
    try:
        return bytes.fromhex(blob_hash)
    except ValueError:
        return None


def _archive_reference_surfaces(
    conn: sqlite3.Connection,
    blob_hash: str,
    *,
    surface_prefix: str,
) -> list[str]:
    blob_bytes = _blob_hash_bytes(blob_hash)
    if blob_bytes is None:
        return []

    surfaces: list[str] = []
    for table in ("raw_sessions", "blob_refs", "attachments"):
        if not _table_exists(conn, table):
            continue
        row = conn.execute(f"SELECT 1 FROM {table} WHERE blob_hash = ? LIMIT 1", (blob_bytes,)).fetchone()
        if row is not None:
            surfaces.append(f"{surface_prefix}.{table}")
    return surfaces


def _reference_surfaces(
    conn: sqlite3.Connection,
    blob_hash: str,
    *,
    source_db_path: Path | None = None,
    source_conn: sqlite3.Connection | None = None,
    index_conn: sqlite3.Connection | None = None,
) -> list[str]:
    surfaces = _archive_reference_surfaces(conn, blob_hash, surface_prefix="current")

    if source_conn is not None:
        source_prefix = source_db_path.name if source_db_path is not None else "source"
        surfaces.extend(_archive_reference_surfaces(source_conn, blob_hash, surface_prefix=source_prefix))
    elif source_db_path is not None and source_db_path.exists():
        try:
            source_conn = sqlite3.connect(f"file:{source_db_path}?mode=ro", uri=True)
            try:
                surfaces.extend(_archive_reference_surfaces(source_conn, blob_hash, surface_prefix=source_db_path.name))
            finally:
                source_conn.close()
        except sqlite3.Error as exc:
            logger.warning("Could not inspect archive source blob references in %s: %s", source_db_path, exc)

    if index_conn is not None:
        surfaces.extend(_archive_reference_surfaces(index_conn, blob_hash, surface_prefix="index.db"))

    if _table_exists(conn, "raw_sessions"):
        row = conn.execute(
            "SELECT 1 FROM raw_sessions WHERE raw_id = ? LIMIT 1",
            (blob_hash,),
        ).fetchone()
        if row is not None:
            surfaces.append("current.raw_sessions")

    return surfaces


def _still_referenced(
    conn: sqlite3.Connection,
    blob_hash: str,
    *,
    source_db_path: Path | None = None,
) -> bool:
    """Return True if the blob hash is referenced by any archive row."""
    return bool(_reference_surfaces(conn, blob_hash, source_db_path=source_db_path))


def _has_publication_reservation(conn: sqlite3.Connection, blob_hash: str) -> bool:
    blob_bytes = _blob_hash_bytes(blob_hash)
    if blob_bytes is None or not _table_exists(conn, "blob_publication_reservations"):
        return False
    return (
        conn.execute(
            "SELECT 1 FROM blob_publication_reservations WHERE blob_hash = ? LIMIT 1",
            (blob_bytes,),
        ).fetchone()
        is not None
    )


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
    """Delete unreferenced blobs old enough to clear the generation-age gate.

    Safety invariants
    -----------------
    1. Never delete a blob that still has a DB reference.
    2. Only delete blobs older than the previous completed GC generation
       plus MIN_AGE_S (defense-in-depth; the sole protection against an
       in-flight ingest, see the ``MIN_AGE_S`` docstring).
    3. Bound each run to ``max_batch`` deletions.

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
    result = run_blob_gc_report(db_path, blob_dir, max_batch=max_batch, dry_run=dry_run)
    return result.would_delete_count if dry_run else result.deleted_count


def run_blob_gc_report(
    db_path: str | Path,
    blob_dir: str | Path,
    max_batch: int = 100,
    *,
    dry_run: bool = False,
) -> BlobGCResult:
    """Run blob GC and return an operator-facing report.

    ``run_blob_gc`` remains the compatibility API returning only the affected
    count. This report form exposes the same pass as machine-readable counts
    for CLI dry-runs and maintenance logs.
    """
    blob_path = Path(blob_dir)
    db_path_obj = Path(db_path)
    report = BlobGCResult(
        db_path=str(db_path_obj),
        blob_dir=str(blob_path),
        dry_run=dry_run,
        max_batch=int(max_batch),
    )
    if not blob_path.is_dir():
        logger.debug("Blob directory %s does not exist, skipping GC", blob_dir)
        return report

    sibling_source_db = db_path_obj.with_name("source.db")
    control_db_path = (
        sibling_source_db
        if db_path_obj.name != "source.db" and _database_has_table(sibling_source_db, "gc_generations")
        else db_path_obj
    )
    # Filesystem enumeration is deliberately outside the destructive source
    # lock. The lock protects only the bounded final recheck+unlink window.
    with sqlite3.connect(f"file:{control_db_path}?mode=ro", uri=True) as planning_conn:
        prev_completed_at = _previous_generation_completed_at(planning_conn)
    older_than = float(MIN_AGE_S)
    if prev_completed_at is not None:
        older_than = max(older_than, time.time() - prev_completed_at)
    report.older_than_s = older_than
    candidates = _candidate_blobs(blob_path, older_than=older_than)
    report.candidate_count = len(candidates)
    if not candidates:
        return report

    sibling_index_db = db_path_obj if db_path_obj.name == "index.db" else db_path_obj.with_name("index.db")
    evidence = GCRunEvidence(dry_run=dry_run, max_batch=max_batch)
    shortlist: list[tuple[str, float]] = []

    # Filter the filesystem candidates without holding the source-tier write
    # lock. Referenced archives can contain millions of old blobs, so walking
    # past those rows under BEGIN IMMEDIATE would otherwise make max_batch a
    # deletion bound but not a lock-time bound. Every shortlisted candidate is
    # checked again under the destructive lock below.
    planning_conn = sqlite3.connect(f"file:{control_db_path}?mode=ro", uri=True)
    planning_conn.row_factory = sqlite3.Row
    planning_source_conn: sqlite3.Connection | None = None
    planning_index_conn: sqlite3.Connection | None = None
    try:
        if control_db_path != sibling_source_db and sibling_source_db.exists():
            planning_source_conn = sqlite3.connect(f"file:{sibling_source_db}?mode=ro", uri=True)
        if control_db_path != sibling_index_db and sibling_index_db.exists():
            planning_index_conn = sqlite3.connect(f"file:{sibling_index_db}?mode=ro", uri=True)
        for blob_hash, mtime in candidates:
            if len(shortlist) >= max_batch:
                break
            evidence.inspected += 1
            if _reference_surfaces(
                planning_conn,
                blob_hash,
                source_db_path=(sibling_source_db if planning_source_conn is not None else None),
                source_conn=planning_source_conn,
                index_conn=planning_index_conn,
            ):
                evidence.skipped_referenced += 1
                continue
            if _has_publication_reservation(planning_conn, blob_hash):
                evidence.skipped_reserved += 1
                continue
            shortlist.append((blob_hash, mtime))
    finally:
        if planning_source_conn is not None:
            planning_source_conn.close()
        if planning_index_conn is not None:
            planning_index_conn.close()
        planning_conn.close()

    connection_uri = f"file:{control_db_path}?mode=ro" if dry_run else str(control_db_path)
    conn = sqlite3.connect(connection_uri, uri=dry_run)
    conn.row_factory = sqlite3.Row
    source_conn: sqlite3.Connection | None = None
    index_conn: sqlite3.Connection | None = None
    affected = 0
    reclaimed_bytes = 0
    started_at_ms = int(time.time() * 1000)
    try:
        if not dry_run:
            conn.execute("BEGIN IMMEDIATE")
        if control_db_path != sibling_source_db and sibling_source_db.exists():
            source_conn = sqlite3.connect(f"file:{sibling_source_db}?mode=ro", uri=True)
        if control_db_path != sibling_index_db and sibling_index_db.exists():
            index_conn = sqlite3.connect(f"file:{sibling_index_db}?mode=ro", uri=True)

        for blob_hash, _mtime in shortlist:
            if _reference_surfaces(
                conn,
                blob_hash,
                source_db_path=(sibling_source_db if source_conn is not None else None),
                source_conn=source_conn,
                index_conn=index_conn,
            ):
                evidence.skipped_referenced += 1
                continue
            if _has_publication_reservation(conn, blob_hash):
                evidence.skipped_reserved += 1
                continue

            target = _sharded_blob_path(blob_path, blob_hash)
            if dry_run:
                if target.is_file():
                    affected += 1
                    evidence.deleted += 1
                else:
                    evidence.skipped_missing += 1
                continue

            try:
                freed_bytes = target.stat().st_size
            except OSError:
                freed_bytes = 0
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
            affected += 1
            evidence.deleted += 1
            reclaimed_bytes += freed_bytes

        if dry_run:
            report.would_delete_count = affected
        else:
            generation_id = f"gc-{uuid4().hex}"
            conn.execute(
                "INSERT INTO gc_generations "
                "(generation_id, started_at_ms, completed_at_ms, reclaimed_count, reclaimed_bytes) "
                "VALUES (?, ?, ?, ?, ?)",
                (generation_id, started_at_ms, int(time.time() * 1000), affected, reclaimed_bytes),
            )
            conn.commit()
            report.generation_id = generation_id
            report.generation_written = True
            report.deleted_count = affected
            report.reclaimed_bytes = reclaimed_bytes
        report.inspected_count = evidence.inspected
        report.skipped_referenced = evidence.skipped_referenced
        report.skipped_reserved = evidence.skipped_reserved
        report.skipped_missing = evidence.skipped_missing
        report.skipped_unlink_error = evidence.skipped_unlink_error
        return report
    except Exception:
        if not dry_run:
            conn.rollback()
        raise
    finally:
        if source_conn is not None:
            source_conn.close()
        if index_conn is not None:
            index_conn.close()
        conn.close()


@dataclass
class GCHistoryRow:
    """One row of the ``gc-history`` surface — a single completed GC pass."""

    generation_id: str
    started_at_ms: int
    completed_at_ms: int | None
    reclaimed_count: int
    reclaimed_bytes: int


def read_gc_history(db_path: str | Path, *, limit: int = 20) -> list[GCHistoryRow]:
    """Return the most-recent committed GC passes, newest first.

    Each row carries the typed reclaim counters (``reclaimed_count`` /
    ``reclaimed_bytes``) recorded by ``run_blob_gc``. Per-skip diagnostics are
    in-process log detail only and are not persisted (#1743).
    """
    conn = open_connection(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT generation_id, started_at_ms, completed_at_ms, reclaimed_count, reclaimed_bytes "
            "FROM gc_generations ORDER BY started_at_ms DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    finally:
        conn.close()

    return [
        GCHistoryRow(
            generation_id=str(row["generation_id"]),
            started_at_ms=int(row["started_at_ms"]),
            completed_at_ms=int(row["completed_at_ms"]) if row["completed_at_ms"] is not None else None,
            reclaimed_count=int(row["reclaimed_count"]),
            reclaimed_bytes=int(row["reclaimed_bytes"]),
        )
        for row in rows
    ]


__all__ = [
    "BlobGCResult",
    "MIN_AGE_S",
    "GCHistoryRow",
    "GCRunEvidence",
    "read_gc_history",
    "run_blob_gc",
    "run_blob_gc_report",
]
