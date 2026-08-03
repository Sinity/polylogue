"""Promote quarantined, logical-key-less raws proven byte-identical to an already-indexed raw.

polylogue-6753s: :mod:`polylogue.storage.raw_byte_duplicate_supersession`
classifies ``raw_sessions`` rows that are ``revision_authority='quarantined'``
with ``logical_source_key IS NULL`` -- invisible to every existing
reconciliation path, since all of them key off ``logical_source_key`` -- and
whose ``blob_hash`` matches some *other* raw that already has a materialized
session in ``index.db``. This module is the "act" half, following the
identical safety pattern as every other actuator in this family
(``raw_live_source_reconciliation_apply``, ``raw_membership_writeback_apply``,
``raw_append_chain_backfill_apply``):

* Dry-run by default. ``dry_run=False`` requires a verified backup manifest
  for the ``source`` tier, validated with the same gate durable-tier schema
  migrations use.
* Classification is re-run *live*, inside the same write transaction the
  promotion runs in -- never trusts a previously computed plan.
* Every promoted row gets an immutable receipt in
  ``raw_byte_duplicate_supersession_receipts`` in the same transaction as its
  ``revision_authority`` update.
* Deliberately never touches ``predecessor_raw_id`` / ``baseline_raw_id`` /
  ``acquisition_generation`` / ``revision_authority_evidence`` (a closed
  vocabulary scoped to live-source-verification promotions) -- this is a
  distinct, independently-verifiable evidence mechanism, and its own receipt
  row is the durable record of it, exactly like ``raw_membership_writeback``
  already established for its own distinct mechanism.
* Never touches, re-parses, or writes to ``index.db``. ``index.db`` is opened
  strictly read-only, purely to re-confirm which raw_ids are indexed at apply
  time; it is never mutated, and the duplicate's own indexed twin (its
  ``raw_sessions`` row, its ``sessions`` row) is never modified.
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
from polylogue.storage.raw_byte_duplicate_supersession import (
    ByteDuplicateSupersessionPlan,
    plan_byte_duplicate_supersession,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import validate_migration_backup_manifest

TOOL_VERSION = "raw-byte-duplicate-supersession-apply-v1"


class RawByteDuplicateSupersessionApplyError(RuntimeError):
    """Raised when applying a byte-duplicate supersession promotion is refused."""


@dataclass(frozen=True, slots=True)
class RawByteDuplicateSupersessionApplyReport:
    scanned_count: int
    promoted_count: int
    promoted_bytes: int
    novel_count: int
    promoted_raw_ids: tuple[str, ...]
    applied: bool
    backup_manifest: Path | None = None

    @classmethod
    def from_plan(
        cls,
        plan: ByteDuplicateSupersessionPlan,
        *,
        applied: bool,
        promoted_raw_ids: tuple[str, ...] = (),
        backup_manifest: Path | None = None,
    ) -> RawByteDuplicateSupersessionApplyReport:
        by_id = {candidate.raw_id: candidate for candidate in plan.duplicates}
        promoted_bytes = (
            sum(by_id[raw_id].blob_size for raw_id in promoted_raw_ids if raw_id in by_id)
            if applied
            else plan.duplicate_bytes
        )
        return cls(
            scanned_count=plan.scanned_count,
            promoted_count=len(promoted_raw_ids) if applied else len(plan.duplicates),
            promoted_bytes=promoted_bytes,
            novel_count=plan.novel_count,
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
        raise RawByteDuplicateSupersessionApplyError("could not checkpoint source.db before backup validation") from exc
    if row is None:
        raise RawByteDuplicateSupersessionApplyError("could not checkpoint source.db before backup validation")


def apply_raw_byte_duplicate_supersession(
    archive_root: Path,
    *,
    backup_manifest: Path | None = None,
    limit: int | None = None,
    dry_run: bool = True,
) -> RawByteDuplicateSupersessionApplyReport:
    """Classify quarantined, logical-key-less raws live, then promote proven byte-duplicates.

    ``dry_run=True`` (the default) never opens a write transaction on
    ``source.db`` and never touches ``index.db`` beyond a read-only open. It
    runs the same classifier a real apply would and reports what it would do.

    ``dry_run=False`` requires ``backup_manifest`` and re-runs classification
    live, inside the same ``BEGIN IMMEDIATE`` write transaction on
    ``source.db`` the promotion UPDATE/INSERT pair runs in, so nothing acted
    on can be stale relative to what gets promoted. ``index.db`` is opened
    read-only for the classification re-run and is never written to.
    """
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    if not source_db.exists():
        raise FileNotFoundError(f"no source.db at {source_db}")
    if not index_db.exists():
        raise FileNotFoundError(f"no index.db at {index_db}")

    if dry_run:
        source_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
        index_conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
        try:
            plan = plan_byte_duplicate_supersession(source_conn, index_conn, limit=limit)
        finally:
            source_conn.close()
            index_conn.close()
        return RawByteDuplicateSupersessionApplyReport.from_plan(plan, applied=False)

    if backup_manifest is None:
        raise RawByteDuplicateSupersessionApplyError(
            "applying raw-byte-duplicate-supersession requires a verified backup manifest (--backup-manifest)"
        )
    if reason := offline_maintenance_block_reason(_offline_config(archive_root), active=True, dry_run=False):
        raise RawByteDuplicateSupersessionApplyError(reason)

    index_conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
    conn = sqlite3.connect(source_db)
    promoted: list[str] = []
    try:
        _checkpoint_live_tier(conn)
        validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=conn)

        conn.execute("BEGIN IMMEDIATE")
        try:
            validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=conn)

            plan = plan_byte_duplicate_supersession(conn, index_conn, limit=limit)
            promoted_at_ms = int(time.time() * 1000)

            for candidate in plan.duplicates:
                cursor = conn.execute(
                    """
                    UPDATE raw_sessions
                    SET revision_authority = 'byte_proven'
                    WHERE raw_id = ? AND revision_authority = 'quarantined' AND logical_source_key IS NULL
                    """,
                    (candidate.raw_id,),
                )
                if cursor.rowcount != 1:
                    # Defensive: no longer quarantined/logical-key-less under
                    # this same locked transaction's own read -- should not
                    # happen given the classification above ran on this same
                    # connection under this same lock, but skipping instead
                    # of asserting keeps this pass conservative.
                    continue
                conn.execute(
                    """
                    INSERT INTO raw_byte_duplicate_supersession_receipts (
                        raw_id, blob_hash, blob_size, duplicate_of_raw_id, duplicate_of_session_id,
                        previous_revision_authority, promoted_at_ms, tool_version,
                        backup_manifest_path, detail
                    ) VALUES (?, ?, ?, ?, ?, 'quarantined', ?, ?, ?, ?)
                    """,
                    (
                        candidate.raw_id,
                        candidate.blob_hash,
                        candidate.blob_size,
                        candidate.duplicate_of_raw_id,
                        candidate.duplicate_of_session_id,
                        promoted_at_ms,
                        TOOL_VERSION,
                        str(backup_manifest),
                        "",
                    ),
                )
                promoted.append(candidate.raw_id)

            quick_check = conn.execute("PRAGMA quick_check").fetchone()
            if quick_check is None or str(quick_check[0]).lower() != "ok":
                raise RawByteDuplicateSupersessionApplyError(
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
        index_conn.close()

    return RawByteDuplicateSupersessionApplyReport.from_plan(
        plan,
        applied=True,
        promoted_raw_ids=tuple(promoted),
        backup_manifest=backup_manifest,
    )


__all__ = [
    "TOOL_VERSION",
    "RawByteDuplicateSupersessionApplyError",
    "RawByteDuplicateSupersessionApplyReport",
    "apply_raw_byte_duplicate_supersession",
]
