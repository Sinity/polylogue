"""Blob garbage collection.

Blobs are content-addressed files stored under the blob store directory.
The GC runs periodically to reclaim disk space from blobs that are no
longer referenced by any archive row.

Safety invariants
-----------------
1. Never delete a blob that still has a DB reference (message text,
   content_blocks, attachments).
2. Only delete blobs older than the previous completed GC generation
   plus MIN_AGE seconds (defense-in-depth against clockskew and any
   acquire-blob -> commit-row window a slow ingest leaves open — see
   ``MIN_AGE_S`` below for the sizing rationale).
3. Bound each run to ``max_batch`` deletions so GC does not monopolise
   I/O.

A prior revision of this module also carried a lease mechanism
(``pending_blob_refs``, ``acquire_blob_leases``/``release_operation_leases``)
intended to bridge the acquire-blob -> write-DB-row commit window more
tightly than a timing heuristic. It was removed (polylogue-v7e0) after a
race-window audit found no production caller ever populated the lease
payload keys (``_blob_hashes``/``_operation_id`` on
``commit_archive_write_effects``) — the mechanism was fully wired end to
end but never reachable, so the invariant it described never actually
engaged. See ``docs/internals.md`` "GC concurrency model" for the current,
lease-free contract and why ``MIN_AGE_S`` alone is judged sufficient.
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
            "skipped_missing": self.skipped_missing,
            "skipped_unlink_error": self.skipped_unlink_error,
            "generation_id": self.generation_id,
            "generation_written": self.generation_written,
            "older_than_s": self.older_than_s,
        }


# Minimum age in seconds for a blob to be eligible for deletion.
#
# This is the SOLE defense-in-depth mechanism protecting an in-flight
# ingest's blob from a concurrent GC pass (polylogue-v7e0 removed the
# never-reachable lease mechanism that this comment used to describe as
# a second layer). The exposure window this must outlast is
# blob-write-to-disk -> referencing-row-commit for a single ingest
# operation. 60s comfortably covers ordinary ingest batches (parse +
# materialize + FTS repair for a normal session count in the low
# hundreds of ms to a few seconds per CLAUDE.md's convergence docs). The
# documented risk class is the memory-bounded streaming path for
# multi-GiB Claude Code JSONL (`sources/dispatch.py`), where a single
# session's parse+materialize could plausibly exceed 60s; an operator
# who runs `polylogue maintenance blob-gc --yes` concurrently with such
# an ingest, within that window, is the one scenario this floor does not
# fully cover. Mitigation until a real per-write lease lands: avoid
# running blob-gc manually while a large corpus import is active; the
# generation-age gate below (`gc_generations`) adds a second, independent
# margin on top of this floor.
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
    for table in ("raw_sessions", "blob_refs"):
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

    source_db_path: Path | None = db_path_obj.with_name("source.db")
    if source_db_path == db_path_obj:
        source_db_path = None
    conn = open_connection(db_path)
    conn.row_factory = sqlite3.Row
    source_conn: sqlite3.Connection | None = None
    try:
        if source_db_path is not None and source_db_path.exists():
            try:
                source_conn = sqlite3.connect(f"file:{source_db_path}?mode=ro", uri=True)
            except sqlite3.Error as exc:
                logger.warning("Could not inspect archive source blob references in %s: %s", source_db_path, exc)

        started_at_ms = int(time.time() * 1000)
        # Safety invariant #2: a candidate must be older than BOTH the static
        # MIN_AGE_S floor AND the previous completed GC generation. The
        # generation high-water mark prevents a blob created during the same
        # window as the previous GC pass from being reclaimed before its
        # eventual reference can land (defense-in-depth against clock skew).
        # Without a prior generation the floor applies.
        prev_completed_at = _previous_generation_completed_at(conn)
        older_than = float(MIN_AGE_S)
        if prev_completed_at is not None:
            since_prev_generation = time.time() - prev_completed_at
            older_than = max(older_than, since_prev_generation)
        report.older_than_s = older_than
        candidates = _candidate_blobs(blob_path, older_than=older_than)
        report.candidate_count = len(candidates)
        if not candidates:
            return report

        evidence = GCRunEvidence(dry_run=dry_run, max_batch=max_batch)
        deleted = 0
        reclaimed_bytes = 0
        for blob_hash, _mtime in candidates:
            if deleted >= max_batch:
                break

            evidence.inspected += 1

            # Safety check 1: still referenced in DB
            if _reference_surfaces(
                conn,
                blob_hash,
                source_db_path=source_db_path,
                source_conn=source_conn,
            ):
                evidence.skipped_referenced += 1
                continue

            target = _sharded_blob_path(blob_path, blob_hash)

            if dry_run:
                # In dry-run mode, account the would-be deletion only when
                # the on-disk file actually exists at the sharded path.
                if target.is_file():
                    deleted += 1
                    evidence.deleted += 1
                else:
                    evidence.skipped_missing += 1
                continue

            # All checks passed — attempt the unlink at the sharded path.
            # Stat the size first so a successful unlink can attribute the freed
            # bytes to reclaimed_bytes; missing_ok=False so we observe whether a
            # file actually went away. If it was already gone (concurrent GC,
            # stale candidate) we record skipped_missing and do NOT count it.
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

            deleted += 1
            evidence.deleted += 1
            reclaimed_bytes += freed_bytes

        if dry_run:
            # Dry run does not consume a generation slot — no row written,
            # no commit needed. The caller still gets the would-be count.
            logger.info(
                "Blob GC dry-run: would delete %d blob(s); inspected=%d skipped_ref=%d skipped_missing=%d",
                evidence.deleted,
                evidence.inspected,
                evidence.skipped_referenced,
                evidence.skipped_missing,
            )
            report.inspected_count = evidence.inspected
            report.would_delete_count = evidence.deleted
            report.skipped_referenced = evidence.skipped_referenced
            report.skipped_missing = evidence.skipped_missing
            report.skipped_unlink_error = evidence.skipped_unlink_error
            return report

        # Record the completed generation with typed reclaim counters. The
        # per-skip tally stays an in-memory log aggregate; the durable record is
        # reclaimed_count / reclaimed_bytes (#1743 — no JSON evidence column).
        generation_id = f"gc-{uuid4().hex}"
        conn.execute(
            "INSERT INTO gc_generations "
            "(generation_id, started_at_ms, completed_at_ms, reclaimed_count, reclaimed_bytes) "
            "VALUES (?, ?, ?, ?, ?)",
            (generation_id, started_at_ms, int(time.time() * 1000), deleted, reclaimed_bytes),
        )
        conn.commit()

        if deleted:
            logger.info(
                "Blob GC: deleted %d blob(s), reclaimed %d byte(s) in generation %s",
                deleted,
                reclaimed_bytes,
                generation_id,
            )

        report.generation_id = generation_id
        report.generation_written = True
        report.inspected_count = evidence.inspected
        report.reclaimed_bytes = reclaimed_bytes
        report.deleted_count = deleted
        report.skipped_referenced = evidence.skipped_referenced
        report.skipped_missing = evidence.skipped_missing
        report.skipped_unlink_error = evidence.skipped_unlink_error
        return report
    except Exception:
        conn.rollback()
        raise
    finally:
        if source_conn is not None:
            source_conn.close()
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
