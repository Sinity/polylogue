"""Read-only audit and guarded repair for acquired blob-reference closure."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from polylogue.config import Config
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason
from polylogue.paths import render_root
from polylogue.storage.attachment_relink import (
    MAX_ATTACHMENT_SAMPLE_LIMIT,
    OrphanedAttachmentRelinkPlan,
    RawSessionParser,
    RelinkableAttachment,
    UnrecoverableAttachmentReason,
    plan_orphaned_attachment_relink,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import (
    validate_backup_manifest_covers_derived_tier,
    validate_migration_backup_manifest,
)

TOOL_VERSION = "blob-reference-closure-v1"


class BlobReferenceClosureError(RuntimeError):
    """Raised when a guarded closure repair cannot prove its write set."""


class BlobReferenceBlockerKind(StrEnum):
    """Typed reasons a closure row cannot be repaired by this route."""

    RAW_NONEXACT_REFERENCE = "raw_nonexact_reference"
    RAW_MISSING_AUTHORITATIVE_FIELD = "raw_missing_authoritative_field"
    ATTACHMENT_NO_AUTHORITATIVE_RAW = "attachment_no_authoritative_raw"
    ATTACHMENT_MESSAGE_MISSING = "attachment_message_missing"


@dataclass(frozen=True, slots=True)
class BlobReferenceClosureBlocker:
    kind: BlobReferenceBlockerKind
    object_id: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind.value, "object_id": self.object_id, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class RawBlobReferenceCandidate:
    raw_id: str
    blob_hash: bytes
    source_path: str
    blob_size: int
    acquired_at_ms: int

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_id": self.raw_id,
            "blob_hash": self.blob_hash.hex(),
            "source_path": self.source_path,
            "blob_size": self.blob_size,
            "acquired_at_ms": self.acquired_at_ms,
        }


@dataclass(frozen=True, slots=True)
class BlobReferenceClosurePlan:
    raw_candidates: tuple[RawBlobReferenceCandidate, ...]
    attachment_candidates: tuple[RelinkableAttachment, ...]
    blockers: tuple[BlobReferenceClosureBlocker, ...]
    raw_rows_scanned: int
    raw_rows_total: int
    attachment_orphan_count: int
    attachment_blockers_sampled: bool = False

    @property
    def candidate_count(self) -> int:
        return len(self.raw_candidates) + len(self.attachment_candidates)

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_candidate_count": len(self.raw_candidates),
            "attachment_candidate_count": len(self.attachment_candidates),
            "candidate_count": self.candidate_count,
            "raw_rows_scanned": self.raw_rows_scanned,
            "raw_rows_total": self.raw_rows_total,
            "attachment_orphan_count": self.attachment_orphan_count,
            "attachment_blockers_sampled": self.attachment_blockers_sampled,
            "blocker_count": len(self.blockers),
            "blockers": [blocker.to_dict() for blocker in self.blockers],
        }


@dataclass(frozen=True, slots=True)
class BlobReferenceClosureReport:
    archive_root: str
    dry_run: bool
    applied: bool
    plan: BlobReferenceClosurePlan
    raw_repaired_count: int = 0
    attachment_repaired_count: int = 0
    backup_manifest: Path | None = None
    receipt_path: Path | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_root": self.archive_root,
            "dry_run": self.dry_run,
            "applied": self.applied,
            "raw_repaired_count": self.raw_repaired_count,
            "attachment_repaired_count": self.attachment_repaired_count,
            "backup_manifest": str(self.backup_manifest) if self.backup_manifest is not None else None,
            "receipt_path": str(self.receipt_path) if self.receipt_path is not None else None,
            "plan": self.plan.to_dict(),
        }


def _offline_config(archive_root: Path) -> Config:
    return Config(archive_root=archive_root, render_root=render_root(), sources=[])


def _open_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def raw_reference_closure_predicate(raw_alias: str = "r", ref_alias: str = "b") -> str:
    """Return the canonical exact-one raw-payload reference predicate."""
    return f"""
        (
            (
            SELECT COUNT(*) FROM blob_refs {ref_alias}
            WHERE {ref_alias}.ref_type = 'raw_payload'
              AND {ref_alias}.ref_id = {raw_alias}.raw_id
              AND {ref_alias}.blob_hash = {raw_alias}.blob_hash
            ) != 1
            OR (
            SELECT COUNT(*) FROM blob_refs {ref_alias}
            WHERE {ref_alias}.ref_type = 'raw_payload'
              AND {ref_alias}.ref_id = {raw_alias}.raw_id
            ) != 1
        )
    """


def _raw_candidates_and_blockers(
    conn: sqlite3.Connection,
) -> tuple[list[RawBlobReferenceCandidate], list[BlobReferenceClosureBlocker], int]:
    rows = conn.execute(
        f"""
        SELECT r.raw_id, r.blob_hash, r.source_path, r.blob_size, r.acquired_at_ms,
               (SELECT COUNT(*) FROM blob_refs b
                WHERE b.ref_type = 'raw_payload' AND b.ref_id = r.raw_id) AS ref_count,
               (SELECT COUNT(*) FROM blob_refs b
                WHERE b.ref_type = 'raw_payload' AND b.ref_id = r.raw_id
                  AND b.blob_hash = r.blob_hash) AS exact_count
        FROM raw_sessions r
        WHERE {raw_reference_closure_predicate()}
        ORDER BY r.raw_id
        """
    ).fetchall()
    candidates: list[RawBlobReferenceCandidate] = []
    blockers: list[BlobReferenceClosureBlocker] = []
    for raw_id, blob_hash, source_path, blob_size, acquired_at_ms, ref_count, exact_count in rows:
        if source_path is None or blob_size is None or acquired_at_ms is None:
            blockers.append(
                BlobReferenceClosureBlocker(
                    BlobReferenceBlockerKind.RAW_MISSING_AUTHORITATIVE_FIELD,
                    str(raw_id),
                    "raw_sessions lacks source_path, blob_size, or acquired_at_ms",
                )
            )
        elif exact_count == 0 and ref_count == 0:
            candidates.append(
                RawBlobReferenceCandidate(
                    raw_id=str(raw_id),
                    blob_hash=bytes(blob_hash),
                    source_path=str(source_path),
                    blob_size=int(blob_size),
                    acquired_at_ms=int(acquired_at_ms),
                )
            )
        else:
            blockers.append(
                BlobReferenceClosureBlocker(
                    BlobReferenceBlockerKind.RAW_NONEXACT_REFERENCE,
                    str(raw_id),
                    f"expected exactly one exact raw_payload ref; ref_count={ref_count}, exact_count={exact_count}",
                )
            )
    total = int(conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0])
    return candidates, blockers, total


def _attachment_blockers(plan: OrphanedAttachmentRelinkPlan) -> list[BlobReferenceClosureBlocker]:
    blockers: list[BlobReferenceClosureBlocker] = []
    for item in plan.unrecoverable_samples[:MAX_ATTACHMENT_SAMPLE_LIMIT]:
        if item.reason_kind is UnrecoverableAttachmentReason.MESSAGE_MISSING:
            kind = BlobReferenceBlockerKind.ATTACHMENT_MESSAGE_MISSING
        else:
            kind = BlobReferenceBlockerKind.ATTACHMENT_NO_AUTHORITATIVE_RAW
        blockers.append(BlobReferenceClosureBlocker(kind, item.attachment_id, item.reason))
    return blockers


def _acquired_attachment_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        """
        SELECT a.attachment_id
        FROM attachments a
        WHERE a.acquisition_status = 'acquired'
          AND NOT EXISTS (SELECT 1 FROM attachment_refs r WHERE r.attachment_id = a.attachment_id)
        """
    ).fetchall()
    return {str(row[0]) for row in rows}


def _plan_connections(
    index_conn: sqlite3.Connection,
    source_conn: sqlite3.Connection,
    *,
    archive_root: Path,
    sample_limit: int,
    raw_session_parser: RawSessionParser | None = None,
) -> BlobReferenceClosurePlan:
    raw_candidates, raw_blockers, raw_total = _raw_candidates_and_blockers(source_conn)
    acquired_attachment_ids = _acquired_attachment_ids(index_conn)
    attachment_plan = plan_orphaned_attachment_relink(
        index_conn,
        source_conn,
        archive_root=archive_root,
        blob_root=archive_root / "blob",
        raw_row_limit=None,
        sample_limit=MAX_ATTACHMENT_SAMPLE_LIMIT,
        raw_session_parser=raw_session_parser,
    )
    return BlobReferenceClosurePlan(
        raw_candidates=tuple(raw_candidates),
        attachment_candidates=tuple(
            candidate for candidate in attachment_plan.eligible if candidate.attachment_id in acquired_attachment_ids
        ),
        blockers=tuple(
            raw_blockers
            + [
                blocker
                for blocker in _attachment_blockers(attachment_plan)
                if blocker.object_id in acquired_attachment_ids
            ]
        ),
        raw_rows_scanned=attachment_plan.raw_rows_scanned,
        raw_rows_total=raw_total,
        attachment_orphan_count=len(acquired_attachment_ids),
        attachment_blockers_sampled=attachment_plan.unrecoverable_samples_truncated,
    )


def plan_blob_reference_closure(
    archive_root: Path,
    *,
    sample_limit: int = 30,
    raw_session_parser: RawSessionParser | None = None,
) -> BlobReferenceClosurePlan:
    """Build a complete, read-only plan from durable source and index evidence."""
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    if not source_db.exists() or not index_db.exists():
        raise FileNotFoundError("blob-reference closure requires source.db and index.db")
    source_conn = _open_ro(source_db)
    index_conn = _open_ro(index_db)
    try:
        return _plan_connections(
            index_conn,
            source_conn,
            archive_root=archive_root,
            sample_limit=sample_limit,
            raw_session_parser=raw_session_parser,
        )
    finally:
        index_conn.close()
        source_conn.close()


def _plan_digest(plan: BlobReferenceClosurePlan) -> str:
    attachment_payload: list[dict[str, object]] = []
    for attachment in plan.attachment_candidates:
        attachment_payload.append(
            {
                "attachment_id": attachment.attachment_id,
                "session_id": attachment.session_id,
                "message_id": attachment.message_id,
                "position": attachment.position,
                "upload_origin": attachment.upload_origin,
                "source_url": attachment.source_url,
                "caption": attachment.caption,
                "raw_id": attachment.raw_id,
                "native_ids": attachment.native_ids,
            }
        )
    payload = {
        "raw": [candidate.to_dict() for candidate in plan.raw_candidates],
        "attachments": attachment_payload,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_receipt(path: Path, *, archive_root: Path, plan: BlobReferenceClosurePlan, backup_manifest: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise BlobReferenceClosureError(f"receipt already exists: {path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "kind": "blob_reference_closure",
                "tool_version": TOOL_VERSION,
                "phase": "prepared",
                "archive_root": str(archive_root),
                "backup_manifest": str(backup_manifest),
                "prepared_at_ms": int(time.time() * 1000),
                "plan_digest": _plan_digest(plan),
                "plan": plan.to_dict(),
            },
            handle,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_receipt_directory(path)


def _append_receipt(path: Path, phase: str, **extra: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        json.dump({"kind": "blob_reference_closure", "phase": phase, **extra}, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_receipt_directory(path)


def _fsync_receipt_directory(path: Path) -> None:
    """Durably publish a newly created or extended receipt directory entry."""
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _validate_backups(backup_manifest: Path, source_conn: sqlite3.Connection, index_conn: sqlite3.Connection) -> None:
    validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=source_conn)
    validate_backup_manifest_covers_derived_tier(backup_manifest, ArchiveTier.INDEX, connection=index_conn)


def reconcile_blob_reference_closure(
    archive_root: Path,
    *,
    backup_manifest: Path | None = None,
    receipt_path: Path | None = None,
    dry_run: bool = True,
    sample_limit: int = 30,
    raw_session_parser: RawSessionParser | None = None,
) -> BlobReferenceClosureReport:
    """Plan closure repair, or add only deterministic exact references.

    Apply is offline, backup-gated, and additive. It never deletes or replaces
    an existing reference. Attachment ownership is accepted only when a full
    raw reparse reproduces the attachment identity and its message exists in
    the current index.
    """
    if dry_run:
        return BlobReferenceClosureReport(
            archive_root=str(archive_root),
            dry_run=True,
            applied=False,
            plan=plan_blob_reference_closure(
                archive_root,
                sample_limit=sample_limit,
                raw_session_parser=raw_session_parser,
            ),
        )
    if backup_manifest is None:
        raise BlobReferenceClosureError("apply requires a verified backup manifest covering source.db and index.db")
    if receipt_path is None:
        raise BlobReferenceClosureError("apply requires an explicit receipt path")
    if reason := offline_maintenance_block_reason(_offline_config(archive_root), active=True, dry_run=False):
        raise BlobReferenceClosureError(reason)

    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    source_conn = sqlite3.connect(source_db)
    index_conn: sqlite3.Connection | None = sqlite3.connect(index_db)
    assert index_conn is not None
    source_conn.execute("PRAGMA foreign_keys = ON")
    index_conn.execute("PRAGMA foreign_keys = ON")
    plan: BlobReferenceClosurePlan | None = None
    source_repaired = 0
    attachment_repaired = 0
    prepared = False
    committed = False
    attached_index = False
    try:
        try:
            assert index_conn is not None
            _validate_backups(backup_manifest, source_conn, index_conn)
            plan = _plan_connections(
                index_conn,
                source_conn,
                archive_root=archive_root,
                sample_limit=sample_limit,
                raw_session_parser=raw_session_parser,
            )
            _write_receipt(receipt_path, archive_root=archive_root, plan=plan, backup_manifest=backup_manifest)
            prepared = True

            # A single connection and attached index database give SQLite one
            # transaction boundary for both tiers. Planning and backup checks
            # happen before ATTACH, so every conflict is known before either
            # tier is mutated.
            index_conn.close()
            index_conn = None
            source_conn.execute("ATTACH DATABASE ? AS index_tier", (str(index_db),))
            attached_index = True
            source_conn.execute("BEGIN IMMEDIATE")
            for candidate in plan.raw_candidates:
                source_conn.execute(
                    """
                    INSERT INTO blob_refs (
                        blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms
                    ) VALUES (?, ?, 'raw_payload', ?, ?, ?)
                    """,
                    (
                        candidate.blob_hash,
                        candidate.raw_id,
                        candidate.source_path,
                        candidate.blob_size,
                        candidate.acquired_at_ms,
                    ),
                )
                exact = source_conn.execute(
                    """
                    SELECT COUNT(*) FROM blob_refs b
                    JOIN raw_sessions r ON r.raw_id = b.ref_id AND r.blob_hash = b.blob_hash
                    WHERE b.ref_type = 'raw_payload' AND b.ref_id = ?
                    """,
                    (candidate.raw_id,),
                ).fetchone()[0]
                if exact != 1:
                    raise BlobReferenceClosureError(f"raw exact-match check failed after insert: {candidate.raw_id}")
                source_repaired += 1
            for attachment_candidate in plan.attachment_candidates:
                source_conn.execute(
                    """
                    INSERT INTO index_tier.attachment_refs (
                        attachment_id, session_id, message_id, position, upload_origin, source_url, caption
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        attachment_candidate.attachment_id,
                        attachment_candidate.session_id,
                        attachment_candidate.message_id,
                        attachment_candidate.position,
                        attachment_candidate.upload_origin,
                        attachment_candidate.source_url,
                        attachment_candidate.caption,
                    ),
                )
                source_conn.execute(
                    """
                    UPDATE index_tier.attachments
                    SET ref_count = (
                        SELECT COUNT(*) FROM index_tier.attachment_refs
                        WHERE index_tier.attachment_refs.attachment_id = index_tier.attachments.attachment_id
                    )
                    WHERE attachment_id = ?
                    """,
                    (attachment_candidate.attachment_id,),
                )
                for id_kind, native_id in attachment_candidate.native_ids:
                    source_conn.execute(
                        """
                        INSERT OR IGNORE INTO index_tier.attachment_native_ids (ref_id, id_kind, native_id)
                        VALUES (?, ?, ?)
                        """,
                        (
                            f"{attachment_candidate.message_id}:attachment:{attachment_candidate.position}",
                            id_kind,
                            native_id,
                        ),
                    )
                exact = source_conn.execute(
                    "SELECT COUNT(*) FROM index_tier.attachment_refs WHERE attachment_id = ?",
                    (attachment_candidate.attachment_id,),
                ).fetchone()[0]
                if exact < 1:
                    raise BlobReferenceClosureError(
                        f"attachment reference check failed after insert: {attachment_candidate.attachment_id}"
                    )
                attachment_repaired += 1
            source_conn.commit()
            committed = True
            _append_receipt(receipt_path, "source_committed", repaired_count=source_repaired)
            _append_receipt(receipt_path, "index_committed", repaired_count=attachment_repaired)
            _append_receipt(
                receipt_path,
                "committed",
                raw_repaired_count=source_repaired,
                attachment_repaired_count=attachment_repaired,
            )
        except Exception as exc:
            if source_conn.in_transaction:
                source_conn.rollback()
            if prepared:
                with suppress(OSError):
                    _append_receipt(
                        receipt_path,
                        "committed_receipt_incomplete" if committed else "aborted",
                        error=str(exc),
                    )
            raise
    finally:
        if attached_index:
            with suppress(sqlite3.Error):
                source_conn.execute("DETACH DATABASE index_tier")
        if index_conn is not None:
            index_conn.close()
        source_conn.close()

    assert plan is not None
    return BlobReferenceClosureReport(
        archive_root=str(archive_root),
        dry_run=False,
        applied=True,
        plan=plan,
        raw_repaired_count=source_repaired,
        attachment_repaired_count=attachment_repaired,
        backup_manifest=backup_manifest,
        receipt_path=receipt_path,
    )


def closure_counts(source_conn: sqlite3.Connection, index_conn: sqlite3.Connection) -> dict[str, int]:
    """Return exact structural closure counts without parsing or mutation."""
    raw_missing = int(
        source_conn.execute(
            f"""
            SELECT COUNT(*) FROM raw_sessions r
            WHERE {raw_reference_closure_predicate()}
            """
        ).fetchone()[0]
    )
    attachment_missing = int(
        index_conn.execute(
            """
            SELECT COUNT(*) FROM attachments a
            WHERE a.acquisition_status = 'acquired'
              AND NOT EXISTS (SELECT 1 FROM attachment_refs r WHERE r.attachment_id = a.attachment_id)
            """
        ).fetchone()[0]
    )
    return {"raw_missing_exact_count": raw_missing, "acquired_attachment_missing_ref_count": attachment_missing}


__all__ = [
    "BlobReferenceBlockerKind",
    "BlobReferenceClosureBlocker",
    "BlobReferenceClosureError",
    "BlobReferenceClosurePlan",
    "BlobReferenceClosureReport",
    "RawBlobReferenceCandidate",
    "closure_counts",
    "plan_blob_reference_closure",
    "raw_reference_closure_predicate",
    "reconcile_blob_reference_closure",
]
