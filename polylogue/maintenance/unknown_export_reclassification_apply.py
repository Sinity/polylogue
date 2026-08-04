"""Apply the narrow durable repair for pre-fix ChatGPT browser captures.

The storage classifier is intentionally report-first and can recognize more
than one browser-capture provider. This actuator is narrower: it re-runs that
classifier under the source-tier write lock and mutates only candidates whose
complete envelope proves ``session.provider == "chatgpt"``. It updates
``source.db``'s durable origin and capture mode, writes one immutable receipt,
and leaves ``index.db`` alone. The normal reparse/materialization route is
responsible for deriving the corrected generated session identity afterward.

Dry-run is the default. Apply requires a verified source-tier backup manifest.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from polylogue.config import Config
from polylogue.core.enums import Origin, Provider
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason
from polylogue.paths import render_root
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import validate_migration_backup_manifest
from polylogue.storage.unknown_export_reclassification import (
    CHATGPT_BROWSER_CAPTURE_SOURCE_PATH_LIKE,
    UnknownExportReclassificationCandidate,
    UnknownExportReclassificationPlan,
    plan_unknown_export_reclassification,
)

TOOL_VERSION = "unknown-export-reclassification-apply-v1"


class UnknownExportReclassificationApplyError(RuntimeError):
    """Raised when the durable unknown-export repair is refused."""


@dataclass(frozen=True, slots=True)
class UnknownExportReclassificationApplyReport:
    """Receipt-shaped outcome of one dry-run or apply pass."""

    scanned_count: int
    reclassifiable_count: int
    reclassifiable_bytes: int
    chatgpt_reclassifiable_count: int
    chatgpt_reclassifiable_bytes: int
    non_chatgpt_reclassifiable_count: int
    still_unknown_count: int
    blob_missing_count: int
    reclassified_count: int
    reclassified_bytes: int
    reclassified_raw_ids: tuple[str, ...]
    applied: bool
    source_path_like: str | None
    backup_manifest: Path | None = None
    index_reparse_required: bool = True
    index_rows_touched: int = 0

    @classmethod
    def from_plan(
        cls,
        plan: UnknownExportReclassificationPlan,
        *,
        applied: bool,
        source_path_like: str | None,
        reclassified_raw_ids: tuple[str, ...] = (),
        backup_manifest: Path | None = None,
    ) -> UnknownExportReclassificationApplyReport:
        chatgpt = plan.chatgpt_reclassifiable
        by_id = {candidate.raw_id: candidate for candidate in chatgpt}
        reclassified_bytes = (
            sum(by_id[raw_id].blob_size for raw_id in reclassified_raw_ids if raw_id in by_id)
            if applied
            else sum(candidate.blob_size for candidate in chatgpt)
        )
        return cls(
            scanned_count=plan.scanned_count,
            reclassifiable_count=len(plan.reclassifiable),
            reclassifiable_bytes=sum(candidate.blob_size for candidate in plan.reclassifiable),
            chatgpt_reclassifiable_count=len(chatgpt),
            chatgpt_reclassifiable_bytes=sum(candidate.blob_size for candidate in chatgpt),
            non_chatgpt_reclassifiable_count=len(plan.non_chatgpt_reclassifiable),
            still_unknown_count=len(plan.still_unknown),
            blob_missing_count=len(plan.blob_missing),
            reclassified_count=len(reclassified_raw_ids) if applied else len(chatgpt),
            reclassified_bytes=reclassified_bytes,
            reclassified_raw_ids=reclassified_raw_ids,
            applied=applied,
            source_path_like=source_path_like,
            backup_manifest=backup_manifest,
        )


def _offline_config(archive_root: Path) -> Config:
    return Config(archive_root=archive_root, render_root=render_root(), sources=[])


def _checkpoint_live_tier(conn: sqlite3.Connection) -> None:
    """Truncate the WAL before validating the source-tier backup fingerprint."""
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    except sqlite3.Error as exc:
        raise UnknownExportReclassificationApplyError(
            "could not checkpoint source.db before backup validation"
        ) from exc
    if row is None:
        raise UnknownExportReclassificationApplyError("could not checkpoint source.db before backup validation")


def _receipt_table_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'raw_unknown_export_reclassification_receipts'"
    ).fetchone()
    return row is not None


def _promote(
    conn: sqlite3.Connection,
    candidate: UnknownExportReclassificationCandidate,
    *,
    reclassified_at_ms: int,
    backup_manifest: Path,
) -> bool:
    """Apply one already-filtered ChatGPT candidate and write its receipt."""
    # Keep this guard local to the mutation. The plan is re-run under the same
    # lock, but the exact provider/origin checks make the write contract
    # auditable even if this helper is reused later.
    if candidate.recovered_provider is not Provider.CHATGPT:
        return False
    if candidate.recovered_origin is not Origin.CHATGPT_EXPORT:
        return False

    raw_id = candidate.raw_id
    row = conn.execute("SELECT blob_hash FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()
    if row is None:
        return False
    blob_hash = bytes(row[0])
    cursor = conn.execute(
        """
        UPDATE raw_sessions
        SET origin = 'chatgpt-export', capture_mode = 'chatgpt'
        WHERE raw_id = ? AND origin = 'unknown-export'
        """,
        (raw_id,),
    )
    if cursor.rowcount != 1:
        return False

    conn.execute(
        """
        INSERT INTO raw_unknown_export_reclassification_receipts (
            raw_id, previous_origin, new_origin, previous_capture_mode,
            new_capture_mode, embedded_provider, source_path, blob_hash,
            blob_size, reclassified_at_ms, tool_version, backup_manifest_path,
            index_reparse_required, detail
        ) VALUES (?, 'unknown-export', 'chatgpt-export', ?, 'chatgpt', 'chatgpt', ?, ?, ?, ?, ?, ?, 1, ?)
        """,
        (
            raw_id,
            candidate.previous_capture_mode.value if isinstance(candidate.previous_capture_mode, Provider) else None,
            candidate.source_path,
            blob_hash,
            candidate.blob_size,
            reclassified_at_ms,
            TOOL_VERSION,
            str(backup_manifest),
            "generated index identity deferred to normal reparse",
        ),
    )
    return True


def apply_unknown_export_reclassification(
    archive_root: Path,
    *,
    backup_manifest: Path | None = None,
    source_path_like: str | None = CHATGPT_BROWSER_CAPTURE_SOURCE_PATH_LIKE,
    limit: int | None = None,
    dry_run: bool = True,
) -> UnknownExportReclassificationApplyReport:
    """Classify and optionally reclassify only proven ChatGPT browser captures."""
    source_db = archive_root / "source.db"
    if not source_db.exists():
        raise FileNotFoundError(f"no source.db at {source_db}")
    blob_store = BlobStore(archive_root / "blob")

    if dry_run:
        conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
        try:
            plan = plan_unknown_export_reclassification(
                conn,
                blob_store=blob_store,
                source_path_like=source_path_like,
                limit=limit,
            )
        finally:
            conn.close()
        return UnknownExportReclassificationApplyReport.from_plan(
            plan, applied=False, source_path_like=source_path_like
        )

    if backup_manifest is None:
        raise UnknownExportReclassificationApplyError(
            "applying unknown-export reclassification requires a verified backup manifest (--backup-manifest)"
        )
    if reason := offline_maintenance_block_reason(_offline_config(archive_root), active=True, dry_run=False):
        raise UnknownExportReclassificationApplyError(reason)

    conn = sqlite3.connect(source_db)
    reclassified: list[str] = []
    try:
        _checkpoint_live_tier(conn)
        validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=conn)
        if not _receipt_table_exists(conn):
            raise UnknownExportReclassificationApplyError(
                "source.db schema is missing raw_unknown_export_reclassification_receipts; "
                "migrate the source tier before applying"
            )

        conn.execute("BEGIN IMMEDIATE")
        try:
            validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=conn)
            plan = plan_unknown_export_reclassification(
                conn,
                blob_store=blob_store,
                source_path_like=source_path_like,
                limit=limit,
            )
            reclassified_at_ms = int(time.time() * 1000)
            for candidate in plan.chatgpt_reclassifiable:
                if _promote(
                    conn,
                    candidate,
                    reclassified_at_ms=reclassified_at_ms,
                    backup_manifest=backup_manifest,
                ):
                    reclassified.append(candidate.raw_id)

            quick_check = conn.execute("PRAGMA quick_check").fetchone()
            if quick_check is None or str(quick_check[0]).lower() != "ok":
                raise UnknownExportReclassificationApplyError(
                    f"source.db quick_check failed after reclassification: {quick_check!r}"
                )
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        else:
            conn.commit()
    finally:
        conn.close()

    return UnknownExportReclassificationApplyReport.from_plan(
        plan,
        applied=True,
        source_path_like=source_path_like,
        reclassified_raw_ids=tuple(reclassified),
        backup_manifest=backup_manifest,
    )


__all__ = [
    "TOOL_VERSION",
    "UnknownExportReclassificationApplyError",
    "UnknownExportReclassificationApplyReport",
    "apply_unknown_export_reclassification",
]
