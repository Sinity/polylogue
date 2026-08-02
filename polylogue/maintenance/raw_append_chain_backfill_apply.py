"""Promote membershipless append raws proven correct by live-source verification.

polylogue-lb39z (Phase 1, item 3): 2,712 ``raw_sessions`` rows are
``revision_kind='append'``, ``revision_authority='quarantined'``, and have no
``raw_session_memberships`` row at all -- a genuine fixed point, because the
only mechanism that ever promotes an append raw
(``_promote_contiguous_append_evidence``,
``storage/sqlite/archive_tiers/revision_governance.py``) requires its
byte-contiguous predecessor to already be ``byte_proven``. When the
predecessor itself is stuck quarantined, no amount of re-running that cascade
resolves the child. :mod:`polylogue.storage.raw_append_chain_backfill`
classifies each such row against its live source file's *current* bytes at
its own recorded ``[append_start_offset:append_end_offset)`` window -- a proof
that does not depend on any ancestor's authority. This module is the "act"
half, following the identical safety pattern as the already-merged
``raw_live_source_reconciliation_apply`` (polylogue-u19l) and
``raw_membership_writeback_apply`` (this bead's item 2):

* Dry-run by default. ``dry_run=False`` requires a verified backup manifest
  for the ``source`` tier, validated with the same gate durable-tier schema
  migrations use.
* Classification is re-run *live*, inside the same write transaction the
  promotion runs in -- never trusts a previously computed plan.
* Every promoted row gets an immutable receipt in
  ``raw_append_chain_backfill_receipts`` in the same transaction as its
  ``revision_authority`` update.
* Deliberately never touches ``predecessor_raw_id`` / ``baseline_raw_id`` /
  ``acquisition_generation`` -- exactly like polylogue-u19l's actuator, that
  revision-graph linkage remains ``classify_raw_revision_cohort``'s /
  ``_promote_contiguous_append_evidence``'s concern; the existing cascade
  picks a newly proven row up for free on the next normal convergence pass,
  either resolving it as its true predecessor's child (once that ancestor is
  itself proven) or using it as a newly eligible parent for whatever append
  fragment sits downstream -- unblocking a stuck chain one proven link at a
  time.
* Records ``revision_authority_evidence='live_source_verification_v1'`` --
  the identical evidence value polylogue-u19l's actuator uses, because the
  proof mechanism (byte-window comparison against the live source file) is
  the same mechanism; only the target population (membershipless append
  rows, specifically) differs, and that population distinction is what this
  module's own dedicated receipt table records.
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
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.live_source_reconciliation import LIVE_SOURCE_VERIFICATION_EVIDENCE
from polylogue.storage.raw_append_chain_backfill import (
    AppendChainBackfillPlan,
    plan_append_chain_backfill,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import validate_migration_backup_manifest

TOOL_VERSION = "raw-append-chain-backfill-apply-v1"


class RawAppendChainBackfillApplyError(RuntimeError):
    """Raised when applying an append-chain backfill promotion is refused."""


@dataclass(frozen=True, slots=True)
class RawAppendChainBackfillApplyReport:
    scanned_count: int
    promoted_count: int
    promoted_bytes: int
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
        plan: AppendChainBackfillPlan,
        *,
        applied: bool,
        promoted_raw_ids: tuple[str, ...] = (),
        backup_manifest: Path | None = None,
    ) -> RawAppendChainBackfillApplyReport:
        promoted_bytes = (
            sum(c.blob_size for c in plan.exact_match if c.raw_id in set(promoted_raw_ids))
            if applied
            else plan.exact_match_bytes
        )
        return cls(
            scanned_count=plan.scanned_count,
            promoted_count=len(promoted_raw_ids) if applied else len(plan.exact_match),
            promoted_bytes=promoted_bytes,
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
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    except sqlite3.Error as exc:
        raise RawAppendChainBackfillApplyError("could not checkpoint source.db before backup validation") from exc
    if row is None:
        raise RawAppendChainBackfillApplyError("could not checkpoint source.db before backup validation")


def apply_raw_append_chain_backfill(
    archive_root: Path,
    *,
    backup_manifest: Path | None = None,
    limit: int | None = None,
    dry_run: bool = True,
) -> RawAppendChainBackfillApplyReport:
    """Classify membershipless append raws live, then promote the exact matches.

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
            plan = plan_append_chain_backfill(conn, blob_store=blob_store, limit=limit)
        finally:
            conn.close()
        return RawAppendChainBackfillApplyReport.from_plan(plan, applied=False)

    if backup_manifest is None:
        raise RawAppendChainBackfillApplyError(
            "applying raw-append-chain-backfill requires a verified backup manifest (--backup-manifest)"
        )
    if reason := offline_maintenance_block_reason(_offline_config(archive_root), active=True, dry_run=False):
        raise RawAppendChainBackfillApplyError(reason)

    conn = sqlite3.connect(source_db)
    promoted: list[str] = []
    try:
        _checkpoint_live_tier(conn)
        validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=conn)

        conn.execute("BEGIN IMMEDIATE")
        try:
            validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=conn)

            plan = plan_append_chain_backfill(conn, blob_store=blob_store, limit=limit)
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
                    # Defensive: no longer quarantined under this same locked
                    # transaction's own read -- should not happen given the
                    # classification above ran on this connection under this
                    # same lock, but skipping instead of asserting keeps this
                    # pass conservative.
                    continue
                conn.execute(
                    """
                    INSERT INTO raw_append_chain_backfill_receipts (
                        raw_id, logical_source_key, source_path, blob_hash, blob_size,
                        append_start_offset, append_end_offset, matched_after_codex_header_strip,
                        previous_revision_authority, compared_at_ms, tool_version,
                        backup_manifest_path, detail
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'quarantined', ?, ?, ?, ?)
                    """,
                    (
                        candidate.raw_id,
                        candidate.logical_source_key,
                        candidate.source_path,
                        candidate.blob_hash,
                        candidate.blob_size,
                        candidate.append_start_offset,
                        candidate.append_end_offset,
                        1 if candidate.comparison.matched_after_codex_header_strip else 0,
                        compared_at_ms,
                        TOOL_VERSION,
                        str(backup_manifest),
                        candidate.comparison.detail or "",
                    ),
                )
                promoted.append(candidate.raw_id)

            quick_check = conn.execute("PRAGMA quick_check").fetchone()
            if quick_check is None or str(quick_check[0]).lower() != "ok":
                raise RawAppendChainBackfillApplyError(f"source.db quick_check failed after promotion: {quick_check!r}")
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        else:
            conn.commit()
    finally:
        conn.close()

    return RawAppendChainBackfillApplyReport.from_plan(
        plan,
        applied=True,
        promoted_raw_ids=tuple(promoted),
        backup_manifest=backup_manifest,
    )


__all__ = [
    "TOOL_VERSION",
    "RawAppendChainBackfillApplyError",
    "RawAppendChainBackfillApplyReport",
    "apply_raw_append_chain_backfill",
]
