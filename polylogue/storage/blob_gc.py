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
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from polylogue.storage.blob_liveness import (
    BlobLiveness,
    LivenessState,
    inspect_blob_liveness,
    inspect_blob_reservation,
    validated_blob_ref_liveness_joins,
)
from polylogue.storage.blob_liveness import (
    blob_refs_has_ref_type_column as _blob_refs_has_ref_type_column,
)
from polylogue.storage.hook_payload_ref_reconciliation import HookPayloadRefMatchStage, prepare_match_stage
from polylogue.storage.introspection import table_exists as _table_exists
from polylogue.storage.sqlite.connection_profile import open_connection
from polylogue.version import VERSION_INFO

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


@dataclass(frozen=True, slots=True)
class _GCMemberIntent:
    """One exact, already-planned filesystem deletion candidate."""

    blob_hash: str
    mtime_ns: int
    size_bytes: int


def _gc_member_table_available(conn: sqlite3.Connection) -> bool:
    return _table_exists(conn, "gc_generation_members")


def _gc_identity_snapshot(
    source_conn: sqlite3.Connection,
    index_conn: sqlite3.Connection,
    *,
    archive_root: Path,
) -> tuple[int, int, str, str, str]:
    """Return the current canonical identities GC can actually observe.

    Source has no independent generation token.  Its schema version and the
    archive identity digest bind it to its durable file.  Index exposes both
    its schema version and the active archive generation identity.  The build
    version is the only code identity available at this substrate layer.
    """
    from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation

    source_version = int(source_conn.execute("PRAGMA user_version").fetchone()[0] or 0)
    index_version = int(index_conn.execute("PRAGMA user_version").fetchone()[0] or 0)
    identity = ArchiveIdentity.resolve_location(ArchiveLocation.resolve(archive_root))
    code_identity = f"{VERSION_INFO.version}:{VERSION_INFO.commit or 'unknown'}"
    return source_version, index_version, identity.active_generation, identity.authority_identity_digest, code_identity


def _commit_gc_generation_intent(
    control_db_path: Path,
    *,
    generation_id: str,
    started_at_ms: int,
    members: tuple[_GCMemberIntent, ...],
    identity: tuple[int, int, str, str, str],
) -> None:
    """Commit a generation and all exact member intents before any unlink."""
    source_schema_version, index_schema_version, index_generation, archive_digest, code_identity = identity
    with sqlite3.connect(control_db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if not _gc_member_table_available(conn):
            raise RuntimeError("blob GC durable member-intent schema is unavailable")
        conn.execute(
            "INSERT INTO gc_generations "
            "(generation_id, started_at_ms, completed_at_ms, reclaimed_count, reclaimed_bytes) "
            "VALUES (?, ?, NULL, 0, 0)",
            (generation_id, started_at_ms),
        )
        conn.executemany(
            "INSERT INTO gc_generation_members "
            "(generation_id, blob_hash, candidate_liveness, candidate_mtime_ns, candidate_size_bytes, "
            "source_schema_version, index_schema_version, index_generation, archive_identity_digest, "
            "code_identity, intent_committed_at_ms, outcome) "
            "VALUES (?, ?, 'unreferenced', ?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
            (
                (
                    generation_id,
                    bytes.fromhex(member.blob_hash),
                    member.mtime_ns,
                    member.size_bytes,
                    source_schema_version,
                    index_schema_version,
                    index_generation,
                    archive_digest,
                    code_identity,
                    started_at_ms,
                )
                for member in members
            ),
        )
        conn.commit()


def _commit_gc_member_outcome(
    source_conn: sqlite3.Connection,
    *,
    generation_id: str,
    blob_hash: str,
    outcome: str,
    detail: str | None = None,
) -> None:
    """Commit one post-recheck member result after its filesystem action."""
    cursor = source_conn.execute(
        "UPDATE gc_generation_members SET outcome = ?, outcome_at_ms = ?, outcome_detail = ? "
        "WHERE generation_id = ? AND blob_hash = ? AND outcome = 'pending'",
        (outcome, int(time.time() * 1000), detail, generation_id, bytes.fromhex(blob_hash)),
    )
    if cursor.rowcount != 1:
        raise RuntimeError("blob GC member outcome lost its exact pending intent")
    source_conn.commit()


def _finalize_gc_generation(control_db_path: Path, generation_id: str) -> bool:
    """Complete one generation only once every durable member is explained."""
    with sqlite3.connect(control_db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT completed_at_ms FROM gc_generations WHERE generation_id = ?", (generation_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError("blob GC generation disappeared before finalization")
        if row[0] is not None:
            conn.rollback()
            return True
        pending = conn.execute(
            "SELECT COUNT(*) FROM gc_generation_members WHERE generation_id = ? AND outcome = 'pending'",
            (generation_id,),
        ).fetchone()[0]
        if pending:
            conn.rollback()
            return False
        reclaimed_count, reclaimed_bytes = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(candidate_size_bytes), 0) "
            "FROM gc_generation_members WHERE generation_id = ? AND outcome = 'removed'",
            (generation_id,),
        ).fetchone()
        conn.execute(
            "UPDATE gc_generations SET completed_at_ms = ?, reclaimed_count = ?, reclaimed_bytes = ? "
            "WHERE generation_id = ? AND completed_at_ms IS NULL",
            (int(time.time() * 1000), reclaimed_count, reclaimed_bytes, generation_id),
        )
        conn.commit()
    return True


def _pending_gc_generation(control_db_path: Path) -> tuple[str | None, str | None]:
    """Return the one restartable member generation, or a fail-closed reason."""
    with closing(sqlite3.connect(f"file:{control_db_path}?mode=ro", uri=True)) as conn:
        if not _gc_member_table_available(conn):
            return None, "blob GC durable member-intent schema is unavailable"
        rows = conn.execute(
            "SELECT generation_id FROM gc_generations WHERE completed_at_ms IS NULL ORDER BY started_at_ms, generation_id"
        ).fetchall()
        if not rows:
            return None, None
        if len(rows) != 1:
            return None, "multiple incomplete blob GC generations require operator investigation"
        generation_id = str(rows[0][0])
        member_count = conn.execute(
            "SELECT COUNT(*) FROM gc_generation_members WHERE generation_id = ?", (generation_id,)
        ).fetchone()[0]
        if member_count == 0:
            return None, "incomplete legacy blob GC generation has no exact member intent"
        return generation_id, None


def _final_gc_member_liveness(
    source_conn: sqlite3.Connection,
    index_conn: sqlite3.Connection | None,
    blob_hash: str,
    *,
    legacy_hook_stage: HookPayloadRefMatchStage,
) -> tuple[BlobLiveness, BlobLiveness]:
    """Production seam for fault injection around GC's final locked recheck."""
    return (
        inspect_blob_liveness(
            source_conn,
            blob_hash,
            index_conn=index_conn,
            require_index=True,
            legacy_hook_stage=legacy_hook_stage,
        ),
        inspect_blob_reservation(source_conn, blob_hash),
    )


def _execute_gc_generation_members(
    *,
    control_db_path: Path,
    sibling_index_db: Path,
    blob_root: Path,
    generation_id: str,
    report: BlobGCResult,
    evidence: GCRunEvidence,
) -> tuple[int, int]:
    """Run pending members with locked rechecks and per-member outcomes.

    Each member gets a fresh source transaction.  That keeps the exact
    ordering durable intent -> final locked liveness -> unlink -> durable
    outcome, while a crash after an unlink leaves the committed intent pending
    for restart reconciliation rather than inventing a successful deletion.
    """
    with closing(sqlite3.connect(f"file:{control_db_path}?mode=ro", uri=True)) as history:
        members = [
            str(row[0]).lower()
            for row in history.execute(
                "SELECT hex(blob_hash) FROM gc_generation_members "
                "WHERE generation_id = ? AND outcome = 'pending' ORDER BY blob_hash",
                (generation_id,),
            )
        ]
    deleted_now = 0
    reclaimed_bytes_now = 0
    for blob_hash in members:
        source_conn = sqlite3.connect(control_db_path)
        index_conn: sqlite3.Connection | None = None
        try:
            source_conn.execute("BEGIN IMMEDIATE")
            if control_db_path != sibling_index_db:
                index_conn = sqlite3.connect(sibling_index_db)
                index_conn.execute("BEGIN IMMEDIATE")
            recheck_index = index_conn or (source_conn if control_db_path == sibling_index_db else None)
            preflight = inspect_blob_liveness(source_conn, "", index_conn=recheck_index, require_index=True)
            if preflight.state is LivenessState.BLOCKED:
                report.blocked_reason = "; ".join(preflight.blockers)
                return deleted_now, reclaimed_bytes_now
            try:
                legacy_hook_stage = prepare_match_stage(source_conn)
            except Exception as exc:
                report.blocked_reason = f"legacy hook rekey matcher failed: {exc}"
                return deleted_now, reclaimed_bytes_now
            decision, reservation = _final_gc_member_liveness(
                source_conn,
                recheck_index,
                blob_hash,
                legacy_hook_stage=legacy_hook_stage,
            )
            if decision.state is LivenessState.BLOCKED:
                report.blocked_reason = "; ".join(decision.blockers)
                return deleted_now, reclaimed_bytes_now
            if reservation.state is LivenessState.BLOCKED:
                report.blocked_reason = "; ".join(reservation.blockers)
                return deleted_now, reclaimed_bytes_now
            if decision.state is LivenessState.LIVE or reservation.state is LivenessState.LIVE:
                _commit_gc_member_outcome(
                    source_conn,
                    generation_id=generation_id,
                    blob_hash=blob_hash,
                    outcome="skipped_still_live",
                    detail="canonical liveness or publication reservation became live",
                )
                evidence.skipped_referenced += decision.state is LivenessState.LIVE
                evidence.skipped_reserved += reservation.state is LivenessState.LIVE
                continue
            target = _sharded_blob_path(blob_root, blob_hash)
            try:
                freed_bytes = target.stat().st_size
            except FileNotFoundError:
                # Only an already-pending exact member can attribute absence.
                _commit_gc_member_outcome(
                    source_conn,
                    generation_id=generation_id,
                    blob_hash=blob_hash,
                    outcome="reconciled_removed",
                    detail="blob absent during fresh final liveness recheck",
                )
                evidence.skipped_missing += 1
                continue
            except OSError:
                freed_bytes = 0
            try:
                target.unlink()
            except FileNotFoundError:
                _commit_gc_member_outcome(
                    source_conn,
                    generation_id=generation_id,
                    blob_hash=blob_hash,
                    outcome="reconciled_removed",
                    detail="blob disappeared after final liveness recheck",
                )
                evidence.skipped_missing += 1
                continue
            except OSError as exc:
                _commit_gc_member_outcome(
                    source_conn,
                    generation_id=generation_id,
                    blob_hash=blob_hash,
                    outcome="failed",
                    detail=str(exc),
                )
                evidence.skipped_unlink_error += 1
                continue
            # The next call commits an outcome in the same transaction that
            # held every liveness writer lock across the unlink.  If it fails,
            # rollback leaves this exact intent pending for a restart to
            # reconcile the now-absent file.
            _commit_gc_member_outcome(
                source_conn,
                generation_id=generation_id,
                blob_hash=blob_hash,
                outcome="removed",
            )
            deleted_now += 1
            reclaimed_bytes_now += freed_bytes
            evidence.deleted += 1
        except Exception:
            if source_conn.in_transaction:
                source_conn.rollback()
            raise
        finally:
            if index_conn is not None:
                if index_conn.in_transaction:
                    index_conn.rollback()
                index_conn.close()
            source_conn.close()
    _finalize_gc_generation(control_db_path, generation_id)
    return deleted_now, reclaimed_bytes_now


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

    with exclude_archive_blob_publishers(source_db_path):
        if not _database_has_table(source_db_path, "gc_generation_members"):
            return 0, 0, ("blob GC durable member-intent schema is unavailable",)
        members: list[_GCMemberIntent] = []
        try:
            with (
                closing(sqlite3.connect(f"file:{source_db_path}?mode=ro", uri=True)) as source_conn,
                closing(sqlite3.connect(f"file:{index_db_path}?mode=ro", uri=True)) as index_conn,
            ):
                preflight = inspect_blob_liveness(source_conn, "", index_conn=index_conn, require_index=True)
                if preflight.state is LivenessState.BLOCKED:
                    return 0, 0, preflight.blockers
                legacy_hook_stage = prepare_match_stage(source_conn)
                for blob_hash in sorted(blob_hashes):
                    try:
                        stat = _sharded_blob_path(blob_root, blob_hash).stat()
                    except OSError:
                        # A caller may hand us a stale candidate set.  No
                        # file means no possible unlink and therefore no
                        # member intent or success attribution.
                        continue
                    decision = inspect_blob_liveness(
                        source_conn,
                        blob_hash,
                        index_conn=index_conn,
                        require_index=True,
                        legacy_hook_stage=legacy_hook_stage,
                    )
                    reservation = inspect_blob_reservation(source_conn, blob_hash)
                    if decision.state is LivenessState.BLOCKED:
                        return 0, 0, decision.blockers
                    if reservation.state is LivenessState.BLOCKED:
                        return 0, 0, reservation.blockers
                    if decision.state is LivenessState.LIVE or reservation.state is LivenessState.LIVE:
                        continue
                    members.append(_GCMemberIntent(blob_hash, stat.st_mtime_ns, stat.st_size))
                identity = _gc_identity_snapshot(source_conn, index_conn, archive_root=source_db_path.parent)
        except Exception as exc:
            return 0, 0, (f"blob GC planning failed: {exc}",)
        if not members:
            return 0, 0, ()
        generation_id = f"gc-{uuid4().hex}"
        _commit_gc_generation_intent(
            source_db_path,
            generation_id=generation_id,
            started_at_ms=int(time.time() * 1000),
            members=tuple(members),
            identity=identity,
        )
        report = BlobGCResult(str(source_db_path), str(blob_root), False, len(members), generation_id=generation_id)
        evidence = GCRunEvidence(max_batch=len(members))
        deleted, deleted_bytes = _execute_gc_generation_members(
            control_db_path=source_db_path,
            sibling_index_db=index_db_path,
            blob_root=blob_root,
            generation_id=generation_id,
            report=report,
            evidence=evidence,
        )
        if report.blocked_reason is not None:
            return deleted, deleted_bytes, (report.blocked_reason,)
        with closing(sqlite3.connect(f"file:{source_db_path}?mode=ro", uri=True)) as source_conn:
            errors = tuple(
                f"{str(row[0])[:16]}: {row[1]}"
                for row in source_conn.execute(
                    "SELECT hex(blob_hash), outcome_detail FROM gc_generation_members "
                    "WHERE generation_id = ? AND outcome = 'failed' ORDER BY blob_hash",
                    (generation_id,),
                )
            )
        return deleted, deleted_bytes, errors


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
    if not _database_has_table(control_db_path, "gc_generation_members"):
        report.blocked_reason = "blob GC durable member-intent schema is unavailable"
        logger.error("Blob GC refused to run: %s", report.blocked_reason)
        return report

    if not dry_run:
        pending_generation, pending_blocker = _pending_gc_generation(control_db_path)
        if pending_blocker is not None:
            report.blocked_reason = pending_blocker
            logger.error("Blob GC refused to run: %s", report.blocked_reason)
            return report
        if pending_generation is not None:
            evidence = GCRunEvidence(dry_run=False, max_batch=max_batch)
            deleted, reclaimed_bytes = _execute_gc_generation_members(
                control_db_path=control_db_path,
                sibling_index_db=sibling_index_db,
                blob_root=blob_path,
                generation_id=pending_generation,
                report=report,
                evidence=evidence,
            )
            report.generation_id = pending_generation
            report.generation_written = True
            report.deleted_count = deleted
            report.reclaimed_bytes = reclaimed_bytes
            report.skipped_referenced = evidence.skipped_referenced
            report.skipped_reserved = evidence.skipped_reserved
            report.skipped_missing = evidence.skipped_missing
            report.skipped_unlink_error = evidence.skipped_unlink_error
            return report

    if not blob_path.is_dir():
        logger.debug("Blob directory %s does not exist, skipping GC", blob_dir)
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

    if not dry_run:
        members: list[_GCMemberIntent] = []
        for blob_hash, _mtime in shortlist:
            try:
                stat = _sharded_blob_path(blob_path, blob_hash).stat()
            except OSError:
                # No intent means no deletion attribution.  A vanished
                # planning candidate is neither an unlink nor a success.
                continue
            members.append(_GCMemberIntent(blob_hash, stat.st_mtime_ns, stat.st_size))
        with closing(sqlite3.connect(f"file:{control_db_path}?mode=ro", uri=True)) as identity_source:
            if control_db_path == sibling_index_db:
                identity_index = identity_source
                identity = _gc_identity_snapshot(identity_source, identity_index, archive_root=db_path_obj.parent)
            else:
                with closing(sqlite3.connect(f"file:{sibling_index_db}?mode=ro", uri=True)) as identity_index:
                    identity = _gc_identity_snapshot(identity_source, identity_index, archive_root=db_path_obj.parent)
        generation_id = f"gc-{uuid4().hex}"
        started_at_ms = int(time.time() * 1000)
        _commit_gc_generation_intent(
            control_db_path,
            generation_id=generation_id,
            started_at_ms=started_at_ms,
            members=tuple(members),
            identity=identity,
        )
        if not members:
            _finalize_gc_generation(control_db_path, generation_id)
            report.generation_id = generation_id
            report.generation_written = True
            report.inspected_count = evidence.inspected
            report.skipped_referenced = evidence.skipped_referenced
            report.skipped_reserved = evidence.skipped_reserved
            return report
        deleted, reclaimed_bytes = _execute_gc_generation_members(
            control_db_path=control_db_path,
            sibling_index_db=sibling_index_db,
            blob_root=blob_path,
            generation_id=generation_id,
            report=report,
            evidence=evidence,
        )
        report.generation_id = generation_id
        report.generation_written = True
        report.deleted_count = deleted
        report.reclaimed_bytes = reclaimed_bytes
        report.inspected_count = evidence.inspected
        report.skipped_referenced = evidence.skipped_referenced
        report.skipped_reserved = evidence.skipped_reserved
        report.skipped_missing = evidence.skipped_missing
        report.skipped_unlink_error = evidence.skipped_unlink_error
        return report

    if not shortlist:
        report.inspected_count = evidence.inspected
        report.skipped_referenced = evidence.skipped_referenced
        report.skipped_reserved = evidence.skipped_reserved
        return report

    # The mutation route returned above.  Dry runs retain the same canonical
    # final check but deliberately create no generation/member history.
    assert dry_run
    conn = sqlite3.connect(f"file:{control_db_path}?mode=ro", uri=True)
    index_conn: sqlite3.Connection | None = None
    affected = 0
    try:
        if control_db_path != sibling_index_db:
            index_conn = sqlite3.connect(f"file:{sibling_index_db}?mode=ro", uri=True)
        recheck_index = index_conn or (conn if control_db_path == sibling_index_db else None)
        recheck_preflight = inspect_blob_liveness(conn, "", index_conn=recheck_index, require_index=True)
        if recheck_preflight.state is LivenessState.BLOCKED:
            report.blocked_reason = "; ".join(recheck_preflight.blockers)
            logger.error("Blob GC refused final recheck: %s", report.blocked_reason)
            return report
        try:
            recheck_legacy_hook_stage = prepare_match_stage(conn)
        except Exception as exc:
            report.blocked_reason = f"legacy hook rekey matcher failed: {exc}"
            logger.error("Blob GC refused final recheck: %s", report.blocked_reason)
            return report

        for blob_hash, _mtime in shortlist:
            decision = inspect_blob_liveness(
                conn,
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
            reservation = inspect_blob_reservation(conn, blob_hash)
            if reservation.state is LivenessState.BLOCKED:
                report.blocked_reason = "; ".join(reservation.blockers)
                logger.error("Blob GC refused final recheck: %s", report.blocked_reason)
                return report
            if reservation.state is LivenessState.LIVE:
                evidence.skipped_reserved += 1
                continue

            target = _sharded_blob_path(blob_path, blob_hash)
            if target.is_file():
                affected += 1
            else:
                evidence.skipped_missing += 1

        report.would_delete_count = affected
        report.inspected_count = evidence.inspected
        report.skipped_referenced = evidence.skipped_referenced
        report.skipped_reserved = evidence.skipped_reserved
        report.skipped_missing = evidence.skipped_missing
        report.skipped_unlink_error = evidence.skipped_unlink_error
        return report
    finally:
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
