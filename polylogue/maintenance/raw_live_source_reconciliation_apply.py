"""Promote quarantined raw evidence proven correct by live-source verification.

polylogue-u19l: ~59GB of ``raw_sessions`` rows sit at
``revision_authority='quarantined'``. ``polylogue.storage.live_source_reconciliation``
classifies each quarantined row against its live source file's *current*
bytes, but is deliberately read-only -- it never mutates ``source.db``. This
module is the "act" half: an explicitly operator-invoked actuator that
promotes ONLY the two verdicts that are safe to act on without human
judgment (``exact_match`` and the historical Codex ``codex_header_strip_match``
variant). Rows classified ``diverged`` or ``source_missing`` are never
touched here -- ``source_missing`` includes genuinely irreplaceable
browser-capture spool files (cleaned up post-ingest by design) and
``diverged`` rows need case-by-case follow-up, not blanket action.

## Design decision: an evidence-provenance column, not a new authority value

``raw_sessions.revision_authority`` is a ``CHECK``-constrained closed
vocabulary (``asserted`` / ``byte_proven`` / ``quarantined``) on a ``STRICT``
table -- widening it needs a full table rebuild, not an additive ``ALTER``.
This module promotes a reconciled row to the existing ``byte_proven`` value
(an accurate description: we now have byte-level proof the archived content
is correct) but records the *different* evidence mechanism in a new,
additive ``revision_authority_evidence`` column (migration 017): ``NULL``
means "whatever produced this row's revision_authority before this column
existed" (the pre-existing revision-graph reconciler); ``'live_source_verification_v1'``
means "promoted because live-source verification re-confirmed the archived
bytes are still present at the recorded source path", independent of any
revision-graph chain reasoning. This is a deliberate choice to make the
provenance distinction durably queryable rather than silently overloading
``byte_proven``'s existing meaning -- see polylogue-u19l notes for the full
investigation this design responds to.

This module deliberately never touches ``predecessor_raw_id`` /
``baseline_raw_id`` / ``acquisition_generation`` -- those revision-chain
linkage fields are the pre-existing reconciler's concern
(``storage/sqlite/archive_tiers/revision_governance.py``), and live-source
verification makes no claim about a row's position in a revision chain, only
that its archived bytes are still verifiable against the live source. If a
promoted row's structural offsets happen to make it an eligible parent for
``_promote_contiguous_append_evidence``'s existing cascade, that cascade may
fire on the next convergence pass exactly as it already does for
reconciler-derived ``byte_proven`` rows -- that is pre-existing, already
reviewed behavior this module does not invoke directly.

## Safety

* Classification is re-run *live*, inside the same write transaction the
  promotion runs in -- this module never trusts a previously computed report
  (a stale report is exactly the failure mode ``--apply`` must not repeat).
* Dry-run (the default) never opens a write transaction and never mutates
  anything.
* ``dry_run=False`` requires a verified backup manifest for the ``source``
  tier, validated with the identical gate durable-tier schema migrations use
  (:func:`polylogue.storage.sqlite.migration_runner.validate_migration_backup_manifest`).
  That gate fails closed if the manifest is missing, was signed for a
  different tier/archive, or does not match ``source.db``'s current bytes
  (rejecting a stale backup, not just a missing one).
* Every promoted row gets an immutable receipt in
  ``raw_live_source_reconciliation_receipts`` in the same transaction as its
  ``revision_authority`` update.
* Never performs blob GC or ``VACUUM``. Promoting ``revision_authority`` only
  makes a row's blob byte-GC-eligible under the *existing* retention
  machinery (``polylogue/storage/raw_retention.py``); running that machinery
  is a separate, later, explicitly operator-invoked step.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from polylogue.config import Config
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason
from polylogue.paths import render_root
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.live_source_reconciliation import (
    LIVE_SOURCE_VERIFICATION_EVIDENCE,
    LiveSourceReconciliationPlan,
    plan_quarantined_live_source_reconciliation,
    receipt_verdict_for,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import validate_migration_backup_manifest

#: Bumped whenever the comparison/promotion semantics change -- recorded on
#: every receipt row so a future audit can tell which tool version produced
#: a given promotion.
TOOL_VERSION = "raw-live-source-reconciliation-apply-v1"


class RawLiveSourceReconciliationApplyError(RuntimeError):
    """Raised when applying a live-source reconciliation promotion is refused."""


@dataclass(frozen=True, slots=True)
class RawLiveSourceReconciliationApplyReport:
    """Outcome of one classify-then-promote pass (dry-run or applied)."""

    scanned_count: int
    promoted_count: int
    promoted_bytes: int
    codex_header_strip_match_count: int
    diverged_count: int
    diverged_bytes: int
    source_missing_count: int
    source_missing_bytes: int
    promoted_raw_ids: tuple[str, ...]
    applied: bool
    backup_manifest: Path | None = None

    @classmethod
    def from_plan(
        cls,
        plan: LiveSourceReconciliationPlan,
        *,
        applied: bool,
        promoted_raw_ids: tuple[str, ...] = (),
        backup_manifest: Path | None = None,
    ) -> RawLiveSourceReconciliationApplyReport:
        promoted_bytes = (
            sum(c.blob_size for c in plan.exact_match if c.raw_id in set(promoted_raw_ids)) if applied else 0
        )
        return cls(
            scanned_count=plan.scanned_count,
            promoted_count=len(promoted_raw_ids) if applied else len(plan.exact_match),
            promoted_bytes=promoted_bytes if applied else plan.exact_match_bytes,
            codex_header_strip_match_count=plan.codex_header_strip_match_count,
            diverged_count=len(plan.diverged),
            diverged_bytes=plan.diverged_bytes,
            source_missing_count=len(plan.source_missing),
            source_missing_bytes=plan.source_missing_bytes,
            promoted_raw_ids=promoted_raw_ids,
            applied=applied,
            backup_manifest=backup_manifest,
        )


def _offline_config(archive_root: Path) -> Config:
    return Config(archive_root=archive_root, render_root=render_root(), sources=[])


def _checkpoint_live_tier(conn: sqlite3.Connection) -> None:
    """Truncate the WAL so a subsequent fingerprint check sees a clean file.

    Mirrors ``migration_runner._checkpoint_live_tier`` -- the backup
    fingerprint check rejects a nonempty WAL, so a live tier that has
    buffered writes in its WAL must be checkpointed before validating a
    backup manifest against it.
    """
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    except sqlite3.Error as exc:
        raise RawLiveSourceReconciliationApplyError("could not checkpoint source.db before backup validation") from exc
    if row is None:
        raise RawLiveSourceReconciliationApplyError("could not checkpoint source.db before backup validation")


def apply_raw_live_source_reconciliation(
    archive_root: Path,
    *,
    backup_manifest: Path | None = None,
    limit: int | None = None,
    dry_run: bool = True,
) -> RawLiveSourceReconciliationApplyReport:
    """Classify quarantined ``raw_sessions`` rows live, then promote the safe ones.

    ``dry_run=True`` (the default) never opens a write transaction. It runs
    the same classifier a real apply would and reports what it would do.

    ``dry_run=False`` requires ``backup_manifest`` and re-runs classification
    live, inside the same ``BEGIN IMMEDIATE`` write transaction the
    promotion UPDATE/INSERT pair runs in, so nothing acted on can be stale
    relative to what gets promoted.
    """
    source_db = archive_root / "source.db"
    if not source_db.exists():
        raise FileNotFoundError(f"no source.db at {source_db}")
    blob_store = BlobStore(archive_root / "blob")

    if dry_run:
        conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
        try:
            plan = plan_quarantined_live_source_reconciliation(conn, blob_store=blob_store, limit=limit)
        finally:
            conn.close()
        return RawLiveSourceReconciliationApplyReport.from_plan(plan, applied=False)

    if backup_manifest is None:
        raise RawLiveSourceReconciliationApplyError(
            "applying raw-live-source-reconciliation requires a verified backup manifest (--backup-manifest)"
        )
    if reason := offline_maintenance_block_reason(_offline_config(archive_root), active=True, dry_run=False):
        raise RawLiveSourceReconciliationApplyError(reason)

    conn = sqlite3.connect(source_db)
    promoted: list[str] = []
    try:
        _checkpoint_live_tier(conn)
        # Precheck (lock-free, fail fast): reject a missing/stale/wrong-tier
        # manifest before paying for a write-lock acquisition.
        validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=conn)

        conn.execute("BEGIN IMMEDIATE")
        try:
            # Authoritative re-validation now that the write lock is held --
            # matches migrate_archive_tier's pattern: a concurrent write
            # between the precheck above and this lock acquisition would
            # make the backup stale, and the fingerprint check inside
            # validate_migration_backup_manifest rejects that.
            validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=conn)

            # Re-classify live, under the write lock -- never trust the
            # precheck's or any caller's previously computed plan.
            plan = plan_quarantined_live_source_reconciliation(conn, blob_store=blob_store, limit=limit)
            compared_at_ms = int(time.time() * 1000)

            for candidate in plan.exact_match:
                assert candidate.source_path is not None  # exact_match rows always have a source_path
                cursor = conn.execute(
                    """
                    UPDATE raw_sessions
                    SET revision_authority = 'byte_proven',
                        revision_authority_evidence = ?
                    WHERE raw_id = ? AND revision_authority = 'quarantined'
                    """,
                    (LIVE_SOURCE_VERIFICATION_EVIDENCE, candidate.raw_id),
                )
                if cursor.rowcount != 1:
                    # Defensive: the row is no longer quarantined under this
                    # same locked transaction's own read -- should not
                    # happen given the classification above ran on this
                    # connection under this same lock, but skipping instead
                    # of asserting keeps this pass conservative.
                    continue
                conn.execute(
                    """
                    INSERT INTO raw_live_source_reconciliation_receipts (
                        raw_id, verdict, previous_revision_authority, source_path,
                        blob_hash, blob_size, compared_at_ms, tool_version,
                        backup_manifest_path, detail
                    ) VALUES (?, ?, 'quarantined', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.raw_id,
                        receipt_verdict_for(candidate),
                        candidate.source_path,
                        candidate.blob_hash,
                        candidate.blob_size,
                        compared_at_ms,
                        TOOL_VERSION,
                        str(backup_manifest),
                        candidate.comparison.detail or "",
                    ),
                )
                promoted.append(candidate.raw_id)

            quick_check = conn.execute("PRAGMA quick_check").fetchone()
            if quick_check is None or str(quick_check[0]).lower() != "ok":
                raise RawLiveSourceReconciliationApplyError(
                    f"source.db quick_check failed after promotion: {quick_check!r}"
                )
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        else:
            conn.commit()
    finally:
        conn.close()

    return RawLiveSourceReconciliationApplyReport.from_plan(
        plan,
        applied=True,
        promoted_raw_ids=tuple(promoted),
        backup_manifest=backup_manifest,
    )


__all__ = [
    "TOOL_VERSION",
    "RawLiveSourceReconciliationApplyError",
    "RawLiveSourceReconciliationApplyReport",
    "apply_raw_live_source_reconciliation",
]
