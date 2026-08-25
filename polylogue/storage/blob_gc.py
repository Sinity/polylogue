"""Blob garbage collection.

Blobs are content-addressed files stored under the blob store directory.
The GC runs periodically to reclaim disk space from blobs that are no
longer referenced by any archive row.

Safety invariants
-----------------
1. Never delete a blob that still has a DB reference (message text,
   content_blocks, attachments). A reference surface that cannot be read is
   a blocker, never an answer: an unavailable tier aborts the pass rather
   than being treated as holding no references.
2. Never delete a blob with a durable publication receipt. Archive
   orchestration commits per-publication receipt IDs before exposing final
   paths, and exact reference transactions consume only their own receipts.
3. Serialize the final reference/reservation recheck and unlink under a
   write lock on *every* tier the recheck reads -- the control tier and each
   sibling reference tier -- so no concurrent writer can commit a reference
   between the check and the unlink. Tiers are locked in a fixed order
   (control, source, index).
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

Relationship to ``RawAuthorityVerdict`` (polylogue-w6hql/ds4b4 item 4): this
module deliberately never consults a raw's authority verdict (VERIFIED /
SOLE_COPY / SUPERSEDED / DIVERGED / UNCHECKED,
``polylogue.core.enums.RawAuthorityVerdict``) when deciding whether a blob is
GC-eligible. Invariant #1 above (row reference) is a strictly stronger, cheaper
check that subsumes verdict semantics: a raw carrying a "non-authoritative"
verdict (SUPERSEDED/DIVERGED/UNCHECKED) still has a live ``raw_sessions`` row
(and a ``blob_refs`` row for its payload) for as long as governance keeps that
history, so its blob stays protected regardless of what the verdict says about
its authority. See
``tests/unit/storage/test_blob_gc_raw_authority_verdict_invariant.py`` for the
regression proof across every verdict value.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from contextlib import closing, suppress
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from polylogue.storage.blob_liveness import (
    LivenessState,
    inspect_blob_liveness,
    inspect_blob_reservation,
    validated_blob_ref_liveness_joins,
)
from polylogue.storage.blob_liveness import (
    blob_refs_has_ref_type_column as _blob_refs_has_ref_type_column,
)
from polylogue.storage.hook_payload_ref_reconciliation import prepare_match_stage
from polylogue.storage.introspection import table_exists as _table_exists
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
    blocked_reason: str | None = None

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
            "blocked_reason": self.blocked_reason,
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


def _database_has_table(path: Path, table: str) -> bool:
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return False
    try:
        return _table_exists(conn, table)
    finally:
        conn.close()


def _open_recheck_connection(path: Path, *, dry_run: bool) -> sqlite3.Connection:
    """Open a sibling reference tier for the final recheck window.

    A destructive pass takes a write transaction so no other process can
    commit a new reference between the recheck and the unlink. A dry run
    reads only and takes no lock, so it never blocks an ingest.
    """
    if dry_run:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn = sqlite3.connect(str(path))
    conn.execute("BEGIN IMMEDIATE")
    return conn


def _reference_tier_blockers(tier_paths: dict[str, Path]) -> tuple[str, ...]:
    """Return a reason per reference tier that cannot be read.

    A tier counts as available when its path resolves to a file that SQLite
    can open read-only. A dangling symlink (the shape a reset or a swapped
    generation leaves behind) is unavailable, not empty.
    """
    blockers: list[str] = []
    for tier, path in tier_paths.items():
        if not path.exists():
            blockers.append(f"{tier} tier is unavailable at {path} (path does not resolve)")
            continue
        try:
            with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as probe:
                probe.execute("SELECT 1").fetchone()
        except sqlite3.Error as exc:
            blockers.append(f"{tier} tier at {path} could not be opened for reading: {exc}")
    return tuple(blockers)


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


def unlink_unreferenced_blob_hashes_under_exclusion(
    source_db_path: Path,
    index_db_path: Path,
    blob_root: Path,
    blob_hashes: set[str],
) -> tuple[int, int, tuple[str, ...]]:
    """Unlink specified hashes after GC's exact final liveness recheck.

    Raw-retention has independently authorized row deletion, but never a
    separate blob authority. It delegates its physical deletion to this
    bounded GC seam: publisher exclusion, source then index write locks, and
    the final canonical-owner/reservation recheck all happen immediately
    before unlink.
    """
    if not blob_hashes:
        return 0, 0, ()
    from polylogue.storage.blob_publication import exclude_archive_blob_publishers

    deleted = 0
    deleted_bytes = 0
    errors: list[str] = []
    try:
        same_database = source_db_path.resolve(strict=False) == index_db_path.resolve(strict=False)
    except OSError as exc:
        return 0, 0, (f"could not resolve blob liveness database paths: {exc}",)
    with exclude_archive_blob_publishers(source_db_path):
        source_conn = sqlite3.connect(source_db_path)
        index_conn: sqlite3.Connection | None = None
        try:
            source_conn.execute("BEGIN IMMEDIATE")
            if same_database:
                index_conn = source_conn
            else:
                index_conn = sqlite3.connect(index_db_path)
                index_conn.execute("BEGIN IMMEDIATE")
            preflight = inspect_blob_liveness(source_conn, "", index_conn=index_conn, require_index=True)
            if preflight.state is LivenessState.BLOCKED:
                return 0, 0, preflight.blockers
            # This transaction excludes source and index writers.  Reuse one
            # fully-attested legacy hook stage through the bounded candidate
            # pass; per-hash inspection only checks its cheap generation.
            try:
                legacy_hook_stage = prepare_match_stage(source_conn)
            except Exception as exc:
                return 0, 0, (f"legacy hook rekey matcher failed: {exc}",)
            for blob_hash in sorted(blob_hashes):
                decision = inspect_blob_liveness(
                    source_conn,
                    blob_hash,
                    index_conn=index_conn,
                    require_index=True,
                    legacy_hook_stage=legacy_hook_stage,
                )
                if decision.state is LivenessState.BLOCKED:
                    return deleted, deleted_bytes, decision.blockers
                if decision.state is LivenessState.LIVE:
                    continue
                reservation = inspect_blob_reservation(source_conn, blob_hash)
                if reservation.state is LivenessState.BLOCKED:
                    return deleted, deleted_bytes, reservation.blockers
                if reservation.state is LivenessState.LIVE:
                    continue
                target = _sharded_blob_path(blob_root, blob_hash)
                try:
                    size = target.stat().st_size
                    target.unlink()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    errors.append(f"{blob_hash[:16]}: {exc}")
                    continue
                deleted += 1
                deleted_bytes += size
            source_conn.commit()
        except Exception:
            source_conn.rollback()
            raise
        finally:
            if index_conn is not None and index_conn is not source_conn:
                if index_conn.in_transaction:
                    index_conn.rollback()
                index_conn.close()
            source_conn.close()
    return deleted, deleted_bytes, tuple(errors)


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

    from polylogue.storage.archive_identity import ArchiveLocation

    sibling_index_db = ArchiveLocation.resolve(db_path_obj.parent).active_index_path
    # Fail closed on an unreadable reference tier before a single candidate is
    # considered. Proceeding without one silently converts "cannot tell" into
    # "not referenced" for every blob only that tier knows about.
    #
    # The index tier carries ``attachments.blob_hash``: on the live archive
    # 1,240 distinct attachment payloads (~425 MB) are reachable through it and
    # nowhere else. It is *rebuildable* and reached through a symlink into a
    # generations directory, so a reset or an interrupted generation promotion
    # leaves the path absent. This is the same fail-closed rule the
    # per-ref-type join already applies to a missing referent table or column.
    required_tiers: dict[str, Path] = {"index": sibling_index_db}
    if db_path_obj.name == "index.db":
        # A split file set names its durable source tier explicitly; a
        # single-file fixture's control database *is* its source surface.
        required_tiers["source"] = sibling_source_db
    blockers = _reference_tier_blockers(required_tiers)
    if blockers:
        reason = "; ".join(blockers)
        logger.error("Blob GC refused to run: %s", reason)
        report.blocked_reason = reason
        return report
    # Filesystem enumeration is deliberately outside the destructive source
    # lock. The lock protects only the bounded final recheck+unlink window.
    with closing(sqlite3.connect(f"file:{control_db_path}?mode=ro", uri=True)) as planning_conn:
        prev_completed_at = _previous_generation_completed_at(planning_conn)
    older_than = float(MIN_AGE_S)
    if prev_completed_at is not None:
        older_than = max(older_than, time.time() - prev_completed_at)
    report.older_than_s = older_than
    candidates = _candidate_blobs(blob_path, older_than=older_than)
    report.candidate_count = len(candidates)
    if not candidates:
        return report

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
        # The source sibling is optional: for a single-file set the control
        # database is itself the source surface. The index sibling is not --
        # it was proven readable above, so it is opened unconditionally and a
        # failure here aborts the pass rather than silently omitting the tier.
        if control_db_path != sibling_source_db and sibling_source_db.exists():
            planning_source_conn = sqlite3.connect(f"file:{sibling_source_db}?mode=ro", uri=True)
        if control_db_path != sibling_index_db:
            planning_index_conn = sqlite3.connect(f"file:{sibling_index_db}?mode=ro", uri=True)
        # Fail the entire destructive pass closed before candidate selection
        # when a current owner surface cannot be evaluated.  Per-hash calls
        # below retain their long-standing aliases for testable snapshot and
        # final-recheck orchestration; the schema contract itself has one
        # owner in blob_liveness.
        planning_source = planning_source_conn or planning_conn
        planning_index = planning_index_conn or (planning_conn if control_db_path == sibling_index_db else None)
        preflight = inspect_blob_liveness(
            planning_source,
            "",
            index_conn=planning_index,
            require_index=True,
        )
        if preflight.state is LivenessState.BLOCKED:
            report.blocked_reason = "; ".join(preflight.blockers)
            logger.error("Blob GC refused to run: %s", report.blocked_reason)
            return report
        try:
            planning_legacy_hook_stage = prepare_match_stage(planning_source)
        except Exception as exc:
            report.blocked_reason = f"legacy hook rekey matcher failed: {exc}"
            logger.error("Blob GC refused to run: %s", report.blocked_reason)
            return report
        for blob_hash, mtime in candidates:
            if len(shortlist) >= max_batch:
                break
            evidence.inspected += 1
            decision = inspect_blob_liveness(
                planning_source,
                blob_hash,
                index_conn=planning_index,
                require_index=True,
                legacy_hook_stage=planning_legacy_hook_stage,
            )
            if decision.state is LivenessState.BLOCKED:
                report.blocked_reason = "; ".join(decision.blockers)
                logger.error("Blob GC refused to run: %s", report.blocked_reason)
                return report
            if decision.state is LivenessState.LIVE:
                evidence.skipped_referenced += 1
                continue
            reservation = inspect_blob_reservation(planning_source, blob_hash)
            if reservation.state is LivenessState.BLOCKED:
                report.blocked_reason = "; ".join(reservation.blockers)
                logger.error("Blob GC refused to run: %s", report.blocked_reason)
                return report
            if reservation.state is LivenessState.LIVE:
                evidence.skipped_reserved += 1
                continue
            shortlist.append((blob_hash, mtime))
    finally:
        if planning_source_conn is not None:
            planning_source_conn.close()
        if planning_index_conn is not None:
            planning_index_conn.close()
        planning_conn.close()

    if not shortlist:
        report.inspected_count = evidence.inspected
        report.skipped_referenced = evidence.skipped_referenced
        report.skipped_reserved = evidence.skipped_reserved
        if dry_run:
            report.would_delete_count = 0
            return report
        history_conn = sqlite3.connect(str(control_db_path))
        try:
            now_ms = int(time.time() * 1000)
            generation_id = f"gc-{uuid4().hex}"
            history_conn.execute(
                "INSERT INTO gc_generations "
                "(generation_id, started_at_ms, completed_at_ms, reclaimed_count, reclaimed_bytes) "
                "VALUES (?, ?, ?, 0, 0)",
                (generation_id, now_ms, now_ms),
            )
            history_conn.commit()
            report.generation_id = generation_id
            report.generation_written = True
        finally:
            history_conn.close()
        return report

    connection_uri = f"file:{control_db_path}?mode=ro" if dry_run else str(control_db_path)
    conn = sqlite3.connect(connection_uri, uri=dry_run)
    conn.row_factory = sqlite3.Row
    source_conn: sqlite3.Connection | None = None
    index_conn: sqlite3.Connection | None = None
    affected = 0
    reclaimed_bytes = 0
    started_at_ms = int(time.time() * 1000)
    try:
        # Invariant 3 spans every tier the recheck reads, not just the control
        # tier. A read-only sibling connection excludes no writer, so a
        # concurrent commit could create the very reference this pass is about
        # to decide does not exist. Each sibling therefore takes its own
        # BEGIN IMMEDIATE for the recheck+unlink window. Tiers are always
        # locked in the same order (control, then source, then index) so two
        # GC passes cannot deadlock against each other.
        if not dry_run:
            conn.execute("BEGIN IMMEDIATE")
        if control_db_path != sibling_source_db and sibling_source_db.exists():
            source_conn = _open_recheck_connection(sibling_source_db, dry_run=dry_run)
        if control_db_path != sibling_index_db:
            index_conn = _open_recheck_connection(sibling_index_db, dry_run=dry_run)
        recheck_source = source_conn or conn
        recheck_index = index_conn or (conn if control_db_path == sibling_index_db else None)
        recheck_preflight = inspect_blob_liveness(recheck_source, "", index_conn=recheck_index, require_index=True)
        if recheck_preflight.state is LivenessState.BLOCKED:
            report.blocked_reason = "; ".join(recheck_preflight.blockers)
            logger.error("Blob GC refused final recheck: %s", report.blocked_reason)
            return report
        try:
            recheck_legacy_hook_stage = prepare_match_stage(recheck_source)
        except Exception as exc:
            report.blocked_reason = f"legacy hook rekey matcher failed: {exc}"
            logger.error("Blob GC refused final recheck: %s", report.blocked_reason)
            return report

        for blob_hash, _mtime in shortlist:
            decision = inspect_blob_liveness(
                recheck_source,
                blob_hash,
                index_conn=recheck_index,
                require_index=True,
                legacy_hook_stage=recheck_legacy_hook_stage,
            )
            if decision.state is LivenessState.BLOCKED:
                report.blocked_reason = "; ".join(decision.blockers)
                logger.error("Blob GC refused final recheck: %s", report.blocked_reason)
                return report
            if decision.state is LivenessState.LIVE:
                evidence.skipped_referenced += 1
                continue
            reservation = inspect_blob_reservation(recheck_source, blob_hash)
            if reservation.state is LivenessState.BLOCKED:
                report.blocked_reason = "; ".join(reservation.blockers)
                logger.error("Blob GC refused final recheck: %s", report.blocked_reason)
                return report
            if reservation.state is LivenessState.LIVE:
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
        # The sibling tiers are read during the recheck and never written, so
        # their transactions are always rolled back; the lock existed only to
        # exclude a concurrent reference commit.
        for sibling in (index_conn, source_conn):
            if sibling is None:
                continue
            if not dry_run:
                with suppress(sqlite3.Error):
                    sibling.rollback()
            sibling.close()
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


@dataclass(frozen=True, slots=True)
class OrphanedBlobRefCensus:
    """Standing count of ``blob_refs`` rows whose referent no longer exists.

    A "orphaned" row here is exactly the shape blob GC's liveness join
    (``_blob_refs_still_live``) treats as dead: its ``ref_type`` names a
    referent table, but no row in that table has the ``ref_id`` this row
    claims. Their *count* is operator-relevant evidence of how much write-time
    drift (deleted rows, since-fixed bugs like the hook-payload one this census
    was built for) has accumulated. Unavailable schemas and unknown ref types
    are counted as dispositions rather than treated as dead. Intended to be
    read by a daemon health/expensive tier; wiring that in is left to the
    caller (polylogue-tfzw0 explicitly defers "wire into health tiers" as
    optional).
    """

    total: int
    by_ref_type: dict[str, int]
    scanned_count: int = 0
    ref_type_counts: dict[str, int] | None = None
    unknown_ref_types: dict[str, int] | None = None
    unavailable_ref_types: dict[str, int] | None = None
    schema_unavailable_count: int = 0
    deferred_by_ref_type: dict[str, int] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "scanned_count": self.scanned_count,
            "ref_type_counts": dict(self.ref_type_counts or {}),
            "total": self.total,
            "by_ref_type": dict(self.by_ref_type),
            "unknown_ref_types": dict(self.unknown_ref_types or {}),
            "unavailable_ref_types": dict(self.unavailable_ref_types or {}),
            "schema_unavailable_count": self.schema_unavailable_count,
            "deferred_by_ref_type": dict(self.deferred_by_ref_type or {}),
        }

    def to_privacy_safe_dict(self) -> dict[str, object]:
        """Serialize aggregate counts without exposing database-derived names."""
        payload = self.to_dict()
        known_ref_types = {ref_type for ref_type, _table, _column in validated_blob_ref_liveness_joins()}
        ref_type_counts = self.ref_type_counts or {}
        payload["ref_type_counts"] = {
            ref_type: count for ref_type, count in ref_type_counts.items() if ref_type in known_ref_types
        }
        payload.pop("unknown_ref_types", None)
        payload["unknown_ref_type_count"] = sum(
            count for ref_type, count in ref_type_counts.items() if ref_type not in known_ref_types
        )
        return payload


def census_orphaned_blob_refs(conn: sqlite3.Connection) -> OrphanedBlobRefCensus:
    """Project the canonical blob-ref classifier into the GC census surface."""
    if not _table_exists(conn, "blob_refs") or not _blob_refs_has_ref_type_column(conn):
        count = (
            int(conn.execute("SELECT COUNT(*) FROM blob_refs").fetchone()[0]) if _table_exists(conn, "blob_refs") else 0
        )
        return OrphanedBlobRefCensus(total=0, by_ref_type={}, schema_unavailable_count=count)
    from polylogue.storage.blob_ref_liveness import classify_blob_ref_liveness

    classification = classify_blob_ref_liveness(conn)
    ref_type_counts = classification.ref_type_counts
    unknown_ref_types = {ref_type: ref_type_counts[ref_type] for ref_type in classification.unknown_ref_types}
    unavailable_ref_types = {ref_type: ref_type_counts[ref_type] for ref_type in classification.unavailable_ref_types}
    deferred_by_ref_type = (
        {"raw_payload": classification.rekeyable_hook_payload_count}
        if classification.rekeyable_hook_payload_count
        else {}
    )
    return OrphanedBlobRefCensus(
        total=classification.orphaned_count,
        by_ref_type=classification.orphaned_by_ref_type,
        scanned_count=classification.scanned_count,
        ref_type_counts=ref_type_counts,
        unknown_ref_types=unknown_ref_types,
        unavailable_ref_types=unavailable_ref_types,
        deferred_by_ref_type=deferred_by_ref_type,
    )


__all__ = [
    "BlobGCResult",
    "MIN_AGE_S",
    "GCHistoryRow",
    "GCRunEvidence",
    "OrphanedBlobRefCensus",
    "census_orphaned_blob_refs",
    "inspect_blob_liveness",
    "read_gc_history",
    "run_blob_gc",
    "run_blob_gc_report",
]
