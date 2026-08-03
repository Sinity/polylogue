"""Promote one representative raw per fully-quarantined byte-identical group; mark the rest proven duplicates.

polylogue-zm4w8: :mod:`polylogue.storage.raw_quarantine_group_dedup` classifies
``(source_path, blob_hash)`` groups among quarantined ``raw_sessions`` rows
where every member is quarantined and none has an indexed twin anywhere --
genuine repeated acquisitions of the same real content, never materialized.
This module is the "act" half, following the same safety pattern as every
other actuator in this family (``raw_byte_duplicate_supersession_apply``,
``raw_live_source_reconciliation_apply``, ``raw_membership_writeback_apply``):

* Dry-run by default (report only, zero mutation).
* ``dry_run=False`` requires a verified backup manifest for the ``source``
  tier, validated with the same gate durable-tier schema migrations use.
* Every marked-duplicate row gets an immutable receipt in
  ``raw_quarantine_group_dedup_receipts`` recording exactly which
  representative raw and materialized session it was superseded by.
* Never deletes blobs or runs GC/VACUUM -- that is a separate, later,
  operator-invoked step.

Unlike every sibling actuator, this one is genuinely two-phase and cannot run
as a single locked ``source.db`` transaction end to end: promoting the
representative raw to a real indexed session goes through the production
async ingest pipeline (``ParsingService.parse_from_raw`` ->
``write_parsed_session_to_archive``, which owns its own transaction against
``index.db``, plus ``refresh_session_insights_bulk``). Only once that
materialization has genuinely landed (re-verified by reading ``index.db``
for a ``sessions`` row at that raw_id, not merely trusted from the ingest
call's return value) does phase two run: a single locked ``source.db``
transaction that re-verifies each duplicate is still quarantined and marks
it, exactly like the sibling actuators' single-phase writes. If
materialization fails or produces no indexed session for a group's
representative (parse error, refused write, non-session content), that
whole group is left untouched -- never guessed at.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from polylogue.config import Config
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason
from polylogue.paths import render_root
from polylogue.storage.raw_quarantine_group_dedup import (
    RawQuarantineGroupDedupPlan,
    plan_raw_quarantine_group_dedup,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import validate_migration_backup_manifest

TOOL_VERSION = "raw-quarantine-group-dedup-apply-v1"


class RawQuarantineGroupDedupApplyError(RuntimeError):
    """Raised when applying a quarantine-group-dedup promotion is refused."""


@dataclass(frozen=True, slots=True)
class RawQuarantineGroupDedupPromotion:
    """One group's outcome: the representative materialized, and the duplicates marked (or planned)."""

    source_path: str
    blob_hash: bytes
    blob_size: int
    representative_raw_id: str
    #: Empty string in a dry-run report (nothing was actually materialized).
    representative_session_id: str
    duplicate_raw_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RawQuarantineGroupDedupApplyReport:
    scanned_count: int
    group_count: int
    already_resolved_group_count: int
    promotions: tuple[RawQuarantineGroupDedupPromotion, ...]
    applied: bool
    backup_manifest: Path | None = None

    @property
    def promoted_count(self) -> int:
        return len(self.promotions)

    @property
    def marked_duplicate_count(self) -> int:
        return sum(len(promotion.duplicate_raw_ids) for promotion in self.promotions)

    @property
    def marked_duplicate_bytes(self) -> int:
        return sum(promotion.blob_size * len(promotion.duplicate_raw_ids) for promotion in self.promotions)


def _offline_config(archive_root: Path) -> Config:
    return Config(archive_root=archive_root, render_root=render_root(), sources=[])


def _checkpoint_live_tier(conn: sqlite3.Connection) -> None:
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    except sqlite3.Error as exc:
        raise RawQuarantineGroupDedupApplyError("could not checkpoint source.db before backup validation") from exc
    if row is None:
        raise RawQuarantineGroupDedupApplyError("could not checkpoint source.db before backup validation")


def _dry_run_report(plan: RawQuarantineGroupDedupPlan) -> RawQuarantineGroupDedupApplyReport:
    promotions = tuple(
        RawQuarantineGroupDedupPromotion(
            source_path=group.source_path,
            blob_hash=group.blob_hash,
            blob_size=group.blob_size,
            representative_raw_id=group.representative_raw_id,
            representative_session_id="",
            duplicate_raw_ids=group.duplicate_raw_ids,
        )
        for group in plan.groups
    )
    return RawQuarantineGroupDedupApplyReport(
        scanned_count=plan.scanned_count,
        group_count=len(plan.groups),
        already_resolved_group_count=plan.already_resolved_group_count,
        promotions=promotions,
        applied=False,
    )


async def apply_raw_quarantine_group_dedup(
    archive_root: Path,
    *,
    backup_manifest: Path | None = None,
    limit: int | None = None,
    dry_run: bool = True,
) -> RawQuarantineGroupDedupApplyReport:
    """Classify quarantined groups, materialize one representative per group, mark the rest.

    ``dry_run=True`` (the default) never opens a write transaction on either
    tier and never runs the ingest pipeline. It runs the same classifier a
    real apply would and reports what it would do.

    ``dry_run=False`` requires ``backup_manifest``. Phase one re-classifies
    live (read-only) and materializes each group's representative through
    the real production ingest pipeline; phase two re-verifies and marks the
    duplicates inside a single locked ``source.db`` transaction. See the
    module docstring for why this cannot be a single atomic transaction like
    the sibling actuators.
    """
    source_db = archive_root / "source.db"
    if not source_db.exists():
        raise FileNotFoundError(f"no source.db at {source_db}")

    from polylogue.storage.archive_identity import resolve_active_index_path

    index_db = resolve_active_index_path(archive_root)
    if not index_db.exists():
        raise FileNotFoundError(f"no index.db at {index_db}")

    if dry_run:
        source_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
        index_conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
        try:
            plan = plan_raw_quarantine_group_dedup(source_conn, index_conn, limit=limit)
        finally:
            source_conn.close()
            index_conn.close()
        return _dry_run_report(plan)

    if backup_manifest is None:
        raise RawQuarantineGroupDedupApplyError(
            "applying raw-quarantine-group-dedup requires a verified backup manifest (--backup-manifest)"
        )

    config = _offline_config(archive_root)
    if reason := offline_maintenance_block_reason(config, active=True, dry_run=False):
        raise RawQuarantineGroupDedupApplyError(reason)

    precheck_conn = sqlite3.connect(source_db)
    try:
        _checkpoint_live_tier(precheck_conn)
        validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=precheck_conn)
    finally:
        precheck_conn.close()

    source_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    index_conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
    try:
        plan = plan_raw_quarantine_group_dedup(source_conn, index_conn, limit=limit)
    finally:
        source_conn.close()
        index_conn.close()

    if not plan.groups:
        return RawQuarantineGroupDedupApplyReport(
            scanned_count=plan.scanned_count,
            group_count=0,
            already_resolved_group_count=plan.already_resolved_group_count,
            promotions=(),
            applied=True,
            backup_manifest=backup_manifest,
        )

    from polylogue.pipeline.services.ingest_batch import refresh_session_insights_bulk
    from polylogue.pipeline.services.parsing import ParsingService
    from polylogue.storage.repository import SessionRepository
    from polylogue.storage.sqlite import create_backend

    backend = create_backend(db_path=config.db_path)
    repository = SessionRepository(backend=backend)
    parser = ParsingService(repository=repository, archive_root=archive_root, config=config)

    promotions: list[RawQuarantineGroupDedupPromotion] = []
    try:
        for group in plan.groups:
            await parser.parse_from_raw(raw_ids=[group.representative_raw_id], force_write=True)
            async with backend.connection() as conn:
                cursor = await conn.execute(
                    "SELECT session_id FROM sessions WHERE raw_id = ?", (group.representative_raw_id,)
                )
                row = await cursor.fetchone()
            if row is None:
                # Materialization produced no indexed session for this raw
                # (parse error, refused write, non-session content, ...) --
                # leave the whole group untouched rather than guessing.
                continue
            representative_session_id = str(row[0])
            await refresh_session_insights_bulk(backend, [representative_session_id])
            promotions.append(
                RawQuarantineGroupDedupPromotion(
                    source_path=group.source_path,
                    blob_hash=group.blob_hash,
                    blob_size=group.blob_size,
                    representative_raw_id=group.representative_raw_id,
                    representative_session_id=representative_session_id,
                    duplicate_raw_ids=group.duplicate_raw_ids,
                )
            )
    finally:
        await backend.close()

    if not promotions:
        return RawQuarantineGroupDedupApplyReport(
            scanned_count=plan.scanned_count,
            group_count=len(plan.groups),
            already_resolved_group_count=plan.already_resolved_group_count,
            promotions=(),
            applied=True,
            backup_manifest=backup_manifest,
        )

    write_conn = sqlite3.connect(source_db)
    try:
        _checkpoint_live_tier(write_conn)
        validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=write_conn)

        write_conn.execute("BEGIN IMMEDIATE")
        try:
            validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=write_conn)
            marked_at_ms = int(time.time() * 1000)
            for promotion in promotions:
                for duplicate_raw_id in promotion.duplicate_raw_ids:
                    existing_cursor = write_conn.execute(
                        "SELECT blob_size FROM raw_sessions WHERE raw_id = ? AND revision_authority = 'quarantined'",
                        (duplicate_raw_id,),
                    )
                    existing = existing_cursor.fetchone()
                    if existing is None:
                        # Defensive: no longer quarantined under this same
                        # locked transaction's own read -- skip rather than
                        # assert, exactly like the sibling actuators.
                        continue
                    update_cursor = write_conn.execute(
                        """
                        UPDATE raw_sessions
                        SET revision_authority = 'byte_proven'
                        WHERE raw_id = ? AND revision_authority = 'quarantined'
                        """,
                        (duplicate_raw_id,),
                    )
                    if update_cursor.rowcount != 1:
                        continue
                    write_conn.execute(
                        """
                        INSERT INTO raw_quarantine_group_dedup_receipts (
                            raw_id, source_path, blob_hash, blob_size,
                            representative_raw_id, representative_session_id,
                            promoted_at_ms, tool_version, backup_manifest_path, detail
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            duplicate_raw_id,
                            promotion.source_path,
                            promotion.blob_hash,
                            int(existing[0]),
                            promotion.representative_raw_id,
                            promotion.representative_session_id,
                            marked_at_ms,
                            TOOL_VERSION,
                            str(backup_manifest),
                            "",
                        ),
                    )

            quick_check = write_conn.execute("PRAGMA quick_check").fetchone()
            if quick_check is None or str(quick_check[0]).lower() != "ok":
                raise RawQuarantineGroupDedupApplyError(
                    f"source.db quick_check failed after promotion: {quick_check!r}"
                )
        except Exception:
            if write_conn.in_transaction:
                write_conn.rollback()
            raise
        else:
            write_conn.commit()
    finally:
        write_conn.close()

    return RawQuarantineGroupDedupApplyReport(
        scanned_count=plan.scanned_count,
        group_count=len(plan.groups),
        already_resolved_group_count=plan.already_resolved_group_count,
        promotions=tuple(promotions),
        applied=True,
        backup_manifest=backup_manifest,
    )


__all__ = [
    "TOOL_VERSION",
    "RawQuarantineGroupDedupApplyError",
    "RawQuarantineGroupDedupApplyReport",
    "RawQuarantineGroupDedupPromotion",
    "apply_raw_quarantine_group_dedup",
]
