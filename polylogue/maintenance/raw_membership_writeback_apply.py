"""Propagate already-decided membership verdicts onto ``raw_sessions.revision_authority``.

polylogue-lb39z (Phase 1, item 2): ``polylogue.storage.raw_membership_writeback``
classifies quarantined ``raw_sessions`` rows whose ``raw_session_memberships``
row already says ``applied`` / ``superseded_equivalent`` / ``superseded_prefix``
with membership ``revision_authority='byte_proven'`` -- a decided fact that was
never written back. This is pure fact transfer, not new judgment: the system
already decided, the verdict was simply never copied to the column the rest
of the archive reads authority from.

This module is the "act" half, following the identical safety pattern as the
already-merged ``raw_live_source_reconciliation_apply`` (polylogue-u19l):

* Dry-run by default. ``dry_run=False`` requires a verified backup manifest
  for the ``source`` tier, validated with the same gate durable-tier schema
  migrations use.
* Classification is re-run *live*, inside the same write transaction the
  promotion runs in -- never trusts a previously computed plan.
* Every promoted row gets an immutable receipt in
  ``raw_membership_writeback_receipts`` in the same transaction as its
  ``revision_authority`` update.
* Deliberately never touches ``predecessor_raw_id`` / ``baseline_raw_id`` /
  ``acquisition_generation`` (revision-graph linkage owned by
  ``classify_raw_revision_cohort``) or ``revision_authority_evidence`` (a
  closed vocabulary scoped to live-source-verification promotions) -- this is
  a distinct, independently-verifiable evidence mechanism, and its own
  receipt row is the durable record of it.
* Never performs blob GC or ``VACUUM``.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from polylogue.config import Config
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason
from polylogue.paths import render_root
from polylogue.storage.raw_membership_writeback import (
    RawMembershipWriteBackPlan,
    WriteBackCandidate,
    plan_raw_membership_writeback,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import validate_migration_backup_manifest

TOOL_VERSION = "raw-membership-writeback-apply-v1"


class RawMembershipWriteBackApplyError(RuntimeError):
    """Raised when applying a membership write-back promotion is refused."""


@dataclass(frozen=True, slots=True)
class RawMembershipWriteBackApplyReport:
    scanned_count: int
    promoted_count: int
    promoted_bytes: int
    promoted_raw_ids: tuple[str, ...]
    applied: bool
    backup_manifest: Path | None = None

    @classmethod
    def from_plan(
        cls,
        plan: RawMembershipWriteBackPlan,
        *,
        applied: bool,
        promoted_raw_ids: tuple[str, ...] = (),
        backup_manifest: Path | None = None,
    ) -> RawMembershipWriteBackApplyReport:
        by_id = {candidate.raw_id: candidate for candidate in plan.candidates}
        promoted_bytes = (
            sum(by_id[raw_id].blob_size for raw_id in promoted_raw_ids if raw_id in by_id)
            if applied
            else plan.candidate_bytes
        )
        return cls(
            scanned_count=plan.scanned_count,
            promoted_count=len(promoted_raw_ids) if applied else len(plan.candidates),
            promoted_bytes=promoted_bytes,
            promoted_raw_ids=promoted_raw_ids,
            applied=applied,
            backup_manifest=backup_manifest,
        )


def _offline_config(archive_root: Path) -> Config:
    return Config(archive_root=archive_root, render_root=render_root(), sources=[])


def _checkpoint_live_tier(conn: sqlite3.Connection) -> None:
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    except sqlite3.Error as exc:
        raise RawMembershipWriteBackApplyError("could not checkpoint source.db before backup validation") from exc
    if row is None:
        raise RawMembershipWriteBackApplyError("could not checkpoint source.db before backup validation")


def _promote(
    conn: sqlite3.Connection, candidate: WriteBackCandidate, *, promoted_at_ms: int, backup_manifest: Path
) -> bool:
    cursor = conn.execute(
        """
        UPDATE raw_sessions
        SET revision_authority = 'byte_proven'
        WHERE raw_id = ? AND revision_authority = 'quarantined'
        """,
        (candidate.raw_id,),
    )
    if cursor.rowcount != 1:
        # Defensive: no longer quarantined under this same locked
        # transaction's own read -- should not happen given the
        # classification above ran on this connection under this same lock,
        # but skipping instead of asserting keeps this pass conservative.
        return False
    conn.execute(
        """
        INSERT INTO raw_membership_writeback_receipts (
            raw_id, logical_source_key, provider_session_id, membership_decision,
            previous_revision_authority, promoted_at_ms, tool_version,
            backup_manifest_path, detail
        ) VALUES (?, ?, ?, ?, 'quarantined', ?, ?, ?, ?)
        """,
        (
            candidate.raw_id,
            candidate.logical_source_key,
            candidate.provider_session_id,
            candidate.decision,
            promoted_at_ms,
            TOOL_VERSION,
            str(backup_manifest),
            "",
        ),
    )
    return True


def apply_raw_membership_writeback(
    archive_root: Path,
    *,
    backup_manifest: Path | None = None,
    limit: int | None = None,
    dry_run: bool = True,
) -> RawMembershipWriteBackApplyReport:
    """Classify quarantined raws with a decided membership verdict, then promote them.

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

    if dry_run:
        conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
        try:
            plan = plan_raw_membership_writeback(conn, limit=limit)
        finally:
            conn.close()
        return RawMembershipWriteBackApplyReport.from_plan(plan, applied=False)

    if backup_manifest is None:
        raise RawMembershipWriteBackApplyError(
            "applying raw-membership-writeback requires a verified backup manifest (--backup-manifest)"
        )
    if reason := offline_maintenance_block_reason(_offline_config(archive_root), active=True, dry_run=False):
        raise RawMembershipWriteBackApplyError(reason)

    conn = sqlite3.connect(source_db)
    promoted: list[str] = []
    try:
        _checkpoint_live_tier(conn)
        validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=conn)

        conn.execute("BEGIN IMMEDIATE")
        try:
            validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=conn)

            plan = plan_raw_membership_writeback(conn, limit=limit)
            promoted_at_ms = int(time.time() * 1000)

            for candidate in plan.candidates:
                if _promote(conn, candidate, promoted_at_ms=promoted_at_ms, backup_manifest=backup_manifest):
                    promoted.append(candidate.raw_id)

            quick_check = conn.execute("PRAGMA quick_check").fetchone()
            if quick_check is None or str(quick_check[0]).lower() != "ok":
                raise RawMembershipWriteBackApplyError(f"source.db quick_check failed after promotion: {quick_check!r}")
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        else:
            conn.commit()
    finally:
        conn.close()

    return RawMembershipWriteBackApplyReport.from_plan(
        plan,
        applied=True,
        promoted_raw_ids=tuple(promoted),
        backup_manifest=backup_manifest,
    )


__all__ = [
    "TOOL_VERSION",
    "RawMembershipWriteBackApplyError",
    "RawMembershipWriteBackApplyReport",
    "apply_raw_membership_writeback",
]
