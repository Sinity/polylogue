"""Apply the provable subset of orphaned hook-payload blob-ref reconciliation.

See :mod:`polylogue.storage.hook_payload_ref_reconciliation` for the
classification this module acts on (polylogue-tfzw0). Follows the same
dry-run-by-default / verified-backup-manifest-gated shape as the other
source-tier repair actuators (``raw_append_chain_backfill_apply``,
``raw_membership_writeback_apply``, ``raw_live_source_reconciliation_apply``):

* Dry-run by default; ``dry_run=False`` requires a verified backup manifest
  for the ``source`` tier.
* Classification is re-run *live*, inside the same write transaction the
  re-key runs in -- never trusts a previously computed plan.
* For each confirmed match: set ``raw_hook_events.blob_hash``, delete the
  stale ``ref_type='raw_payload'`` row, and insert the corrected
  ``ref_type='hook_payload'`` row keyed by the hook event's own id -- the
  same shape ``write_source_hook_event`` writes for a hook event captured
  after v22.

Unlike the byte-proof actuators above, this module does not mint a receipt
row: there is no external evidence being witnessed, just an exact-match
recomputation of an existing deterministic id from data already in the
archive. The re-keyed ``blob_refs``/``raw_hook_events`` rows are themselves
the durable record of what changed.

This module never runs against the live archive as part of any automated
pipeline -- applying it is an explicit operator action
(``dry_run=False`` + a backup manifest), same as every other actuator in this
package.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from polylogue.config import Config
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason
from polylogue.paths import render_root
from polylogue.storage.hook_payload_ref_reconciliation import (
    HookPayloadRefReconciliationPlan,
    plan_hook_payload_ref_reconciliation,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import validate_migration_backup_manifest

TOOL_VERSION = "hook-payload-ref-reconciliation-apply-v1"


class HookPayloadRefReconciliationApplyError(RuntimeError):
    """Raised when applying hook-payload-ref reconciliation is refused."""


@dataclass(frozen=True, slots=True)
class HookPayloadRefReconciliationApplyReport:
    scanned_count: int
    matched_count: int
    matched_bytes: int
    unmatched_count: int
    reconciled_hook_event_ids: tuple[str, ...]
    applied: bool
    backup_manifest: Path | None = None

    @classmethod
    def from_plan(
        cls,
        plan: HookPayloadRefReconciliationPlan,
        *,
        applied: bool,
        reconciled_hook_event_ids: tuple[str, ...] = (),
        backup_manifest: Path | None = None,
    ) -> HookPayloadRefReconciliationApplyReport:
        return cls(
            scanned_count=plan.scanned_count,
            matched_count=len(plan.matched),
            matched_bytes=plan.matched_bytes,
            unmatched_count=plan.unmatched_count,
            reconciled_hook_event_ids=reconciled_hook_event_ids or tuple(c.hook_event_id for c in plan.matched),
            applied=applied,
            backup_manifest=backup_manifest,
        )


def _offline_config(archive_root: Path) -> Config:
    return Config(archive_root=archive_root, render_root=render_root(), sources=[])


def apply_hook_payload_ref_reconciliation(
    archive_root: Path,
    *,
    backup_manifest: Path | None = None,
    dry_run: bool = True,
) -> HookPayloadRefReconciliationApplyReport:
    """Re-key the provable subset of orphaned hook-payload blob refs.

    ``dry_run=True`` (the default) never opens a write transaction; it runs
    the same classifier a real apply would and reports what it would do.

    ``dry_run=False`` requires ``backup_manifest`` and re-runs classification
    live, inside the same ``BEGIN IMMEDIATE`` write transaction the re-key
    UPDATE/DELETE/INSERT triples run in.
    """
    source_db = archive_root / "source.db"
    if not source_db.exists():
        raise FileNotFoundError(f"no source.db at {source_db}")

    if dry_run:
        conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
        try:
            plan = plan_hook_payload_ref_reconciliation(conn)
        finally:
            conn.close()
        return HookPayloadRefReconciliationApplyReport.from_plan(plan, applied=False)

    if backup_manifest is None:
        raise HookPayloadRefReconciliationApplyError(
            "applying hook-payload-ref-reconciliation requires a verified backup manifest (--backup-manifest)"
        )
    if reason := offline_maintenance_block_reason(_offline_config(archive_root), active=True, dry_run=False):
        raise HookPayloadRefReconciliationApplyError(reason)

    conn = sqlite3.connect(source_db)
    reconciled: list[str] = []
    try:
        try:
            row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        except sqlite3.Error as exc:
            raise HookPayloadRefReconciliationApplyError(
                "could not checkpoint source.db before backup validation"
            ) from exc
        if row is None:
            raise HookPayloadRefReconciliationApplyError("could not checkpoint source.db before backup validation")
        validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=conn)

        conn.execute("BEGIN IMMEDIATE")
        try:
            validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=conn)

            plan = plan_hook_payload_ref_reconciliation(conn)
            for candidate in plan.matched:
                deleted = conn.execute(
                    "DELETE FROM blob_refs WHERE blob_hash = ? AND ref_type = 'raw_payload' AND ref_id = ?",
                    (candidate.blob_hash, candidate.orphaned_ref_id),
                )
                if deleted.rowcount != 1:
                    # Defensive: this exact row was read under this same
                    # locked transaction above -- should not happen, but
                    # skipping instead of asserting keeps this pass
                    # conservative like the sibling actuators.
                    continue
                conn.execute(
                    """
                    INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
                    VALUES (?, ?, 'hook_payload', ?, ?, ?)
                    """,
                    (
                        candidate.blob_hash,
                        candidate.hook_event_id,
                        candidate.source_path,
                        candidate.size_bytes,
                        candidate.acquired_at_ms,
                    ),
                )
                conn.execute(
                    "UPDATE raw_hook_events SET blob_hash = ? WHERE hook_event_id = ? AND blob_hash IS NULL",
                    (candidate.blob_hash, candidate.hook_event_id),
                )
                reconciled.append(candidate.hook_event_id)

            quick_check = conn.execute("PRAGMA quick_check").fetchone()
            if quick_check is None or str(quick_check[0]).lower() != "ok":
                raise HookPayloadRefReconciliationApplyError(
                    f"source.db quick_check failed after reconciliation: {quick_check!r}"
                )
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        else:
            conn.commit()
    finally:
        conn.close()

    return HookPayloadRefReconciliationApplyReport.from_plan(
        plan,
        applied=True,
        reconciled_hook_event_ids=tuple(reconciled),
        backup_manifest=backup_manifest,
    )


__all__ = [
    "TOOL_VERSION",
    "HookPayloadRefReconciliationApplyError",
    "HookPayloadRefReconciliationApplyReport",
    "apply_hook_payload_ref_reconciliation",
]
