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

Unlike every sibling actuator, this one is genuinely two-phase per group and
cannot run as a single locked ``source.db`` transaction end to end: promoting
the representative raw to a real indexed session goes through the production
async ingest pipeline (``ParsingService.parse_from_raw`` ->
``write_parsed_session_to_archive``, which owns its own transaction against
``index.db``, plus ``refresh_session_insights_bulk``). Only once that
materialization has genuinely landed (re-verified by reading ``index.db``
for a ``sessions`` row at that raw_id, not merely trusted from the ingest
call's return value) does phase two run for THAT group: a single locked
``source.db`` transaction that re-verifies each duplicate is still
quarantined and marks it, exactly like the sibling actuators' single-phase
writes.

Materialize-then-mark runs **per group**, not batched across the whole plan
(explicit design decision, CodeRabbit review on PR #3697): if group N's
representative fails to materialize or raises, groups 1..N-1 already
recorded their receipts and are durably done, and group N+1 onward still get
attempted. Batching phase two until after every group's phase one completed
would mean one failing group anywhere in the plan leaves every
already-materialized group's duplicates permanently unreachable -- the
classifier excludes a group from its universe the moment any member's
``blob_hash`` has an indexed twin (see
``polylogue.storage.raw_quarantine_group_dedup``), so an already-materialized
representative with un-marked duplicates would never be reclassified as
"in scope" again by a later run.

If materialization produces anything other than **exactly one** indexed
session for a group's representative (parse error, refused write,
non-session content, or a multi-session capture file materializing more than
one ``sessions`` row from a single raw), that whole group is left untouched
-- never guessed at. The exactly-one invariant is a deliberate, conservative
choice: this actuator's whole premise is "one raw, one piece of content, one
representative session" and a raw that turns out to violate that (a grouped
multi-session payload) doesn't fit the receipt schema's singular
``representative_session_id`` column without inventing ambiguous
multi-session semantics -- skipping is safer than guessing which of several
materialized sessions is "the" representative.

If materialization *raises* for a group's representative (a genuine parse
error, a database error, ...), that exception is caught and logged, and that
one group is skipped exactly like the no-session case -- it does NOT abort
the run. Every earlier group's promotion is already durably committed (phase
two per group, above), and every later group still gets attempted. A single
bad row in an otherwise-clean ~1,800-group backlog must not block the rest.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.config import Config
from polylogue.logging import get_logger
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason
from polylogue.paths import render_root
from polylogue.storage.raw_quarantine_group_dedup import (
    RawQuarantineGroup,
    RawQuarantineGroupDedupPlan,
    plan_raw_quarantine_group_dedup,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import validate_migration_backup_manifest

if TYPE_CHECKING:
    from polylogue.pipeline.services.parsing import ParsingService
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

logger = get_logger(__name__)

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
    """Checkpoint the WAL and fail closed if it could not be fully truncated.

    ``PRAGMA wal_checkpoint(TRUNCATE)`` always returns one row
    ``(busy, log, checkpointed)``. ``busy=1`` means another connection held a
    lock that blocked the checkpoint -- the WAL was NOT truncated, and a
    subsequent backup-manifest fingerprint check would silently attest
    against a tier that still has uncheckpointed frames. Only ``busy=0`` is a
    genuine success.
    """
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    except sqlite3.Error as exc:
        raise RawQuarantineGroupDedupApplyError("could not checkpoint source.db before backup validation") from exc
    if row is None:
        raise RawQuarantineGroupDedupApplyError("could not checkpoint source.db before backup validation")
    if int(row[0]) != 0:
        raise RawQuarantineGroupDedupApplyError(
            "source.db WAL checkpoint was blocked by another connection; retry when the tier is idle"
        )


async def _materialize_group_representative(
    parser: ParsingService,
    backend: SQLiteBackend,
    group: RawQuarantineGroup,
) -> str | None:
    """Materialize a group's representative raw; return its session_id iff exactly one session resulted.

    Returns ``None`` (the whole group is left untouched by the caller) when
    materialization produces zero sessions or produces MORE than one session
    for this raw_id (a multi-session capture file) -- see the module
    docstring for why more-than-one is treated the same as zero rather than
    guessed at. Propagates any exception ``parse_from_raw`` raises; the
    caller is responsible for catching it so one bad group doesn't abort the
    rest of the run (see the module docstring).
    """
    from polylogue.pipeline.services.ingest_batch import refresh_session_insights_bulk

    await parser.parse_from_raw(raw_ids=[group.representative_raw_id], force_write=True)
    async with backend.connection() as conn:
        cursor = await conn.execute(
            "SELECT session_id FROM sessions WHERE raw_id = ? ORDER BY session_id",
            (group.representative_raw_id,),
        )
        session_rows = list(await cursor.fetchall())
    if len(session_rows) != 1:
        # Zero: parse error, refused write, non-session content. More than
        # one: a multi-session capture file -- this actuator's whole premise
        # is one raw/one representative session, so an ambiguous
        # materialization is left untouched rather than guessed at.
        return None
    representative_session_id = str(session_rows[0][0])
    await refresh_session_insights_bulk(backend, [representative_session_id])
    return representative_session_id


def _mark_group_duplicates(
    source_db: Path,
    backup_manifest: Path,
    group: RawQuarantineGroup,
    representative_session_id: str,
) -> RawQuarantineGroupDedupPromotion | None:
    """Mark one group's duplicates 'byte_proven' with a receipt, in its own locked transaction.

    Runs per group (not batched across the whole plan) so a later group's
    materialization failure can never leave an earlier, already-materialized
    group's duplicates permanently unmarked -- see the module docstring.
    Returns ``None`` only in the defensive case where every duplicate in the
    group is no longer quarantined by the time this transaction's lock is
    held (the representative's materialization still stands regardless).
    """
    write_conn = sqlite3.connect(source_db)
    try:
        _checkpoint_live_tier(write_conn)
        validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=write_conn)

        write_conn.execute("BEGIN IMMEDIATE")
        try:
            validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=write_conn)
            marked_at_ms = int(time.time() * 1000)
            marked_any = False
            for duplicate_raw_id in group.duplicate_raw_ids:
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
                        group.source_path,
                        group.blob_hash,
                        int(existing[0]),
                        group.representative_raw_id,
                        representative_session_id,
                        marked_at_ms,
                        TOOL_VERSION,
                        str(backup_manifest),
                        "",
                    ),
                )
                marked_any = True

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

    if not marked_any:
        return None
    return RawQuarantineGroupDedupPromotion(
        source_path=group.source_path,
        blob_hash=group.blob_hash,
        blob_size=group.blob_size,
        representative_raw_id=group.representative_raw_id,
        representative_session_id=representative_session_id,
        duplicate_raw_ids=group.duplicate_raw_ids,
    )


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

    ``dry_run=False`` requires ``backup_manifest``. Re-classifies live
    (read-only) once, then processes each group in turn: materialize the
    representative through the real production ingest pipeline, then --
    immediately, only for that group -- re-verify and mark its duplicates
    inside their own locked ``source.db`` transaction. See the module
    docstring for why this runs per group rather than as one batched pass or
    a single atomic transaction like the sibling actuators.
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

    from polylogue.pipeline.services.parsing import ParsingService
    from polylogue.storage.repository import SessionRepository
    from polylogue.storage.sqlite import create_backend

    backend = create_backend(db_path=config.db_path)
    repository = SessionRepository(backend=backend)
    parser = ParsingService(repository=repository, archive_root=archive_root, config=config)

    promotions: list[RawQuarantineGroupDedupPromotion] = []
    try:
        for group in plan.groups:
            try:
                representative_session_id = await _materialize_group_representative(parser, backend, group)
            except Exception:
                # One bad group (a genuine parse/database error) must not
                # abort the rest of an otherwise-clean run -- every earlier
                # group's promotion is already durably committed below, and
                # every later group still gets attempted. See module docstring.
                logger.exception(
                    "raw-quarantine-group-dedup: materializing representative %s failed; leaving group at %s untouched",
                    group.representative_raw_id,
                    group.source_path,
                )
                continue
            if representative_session_id is None:
                # Zero or more-than-one indexed session for this raw -- leave
                # the whole group untouched rather than guessing (see
                # _materialize_group_representative's docstring).
                continue

            promotion = _mark_group_duplicates(source_db, backup_manifest, group, representative_session_id)
            if promotion is not None:
                promotions.append(promotion)
    finally:
        await backend.close()

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
