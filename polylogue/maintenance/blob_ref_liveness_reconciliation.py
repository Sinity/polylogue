"""Dry-run-first reconciliation of historical source-tier blob references."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from polylogue.config import Config
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason
from polylogue.paths import render_root
from polylogue.storage.blob_ref_liveness import (
    BlobRefLivenessClassification,
    classify_blob_ref_liveness,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import validate_migration_backup_manifest

TOOL_VERSION = "blob-ref-liveness-reconciliation-v1"


class BlobRefLivenessReconciliationError(RuntimeError):
    """Raised when reconciliation cannot prove a safe source-tier apply."""


@dataclass(frozen=True, slots=True)
class BlobRefLivenessReconciliationReport:
    source_db: str
    dry_run: bool
    classification: BlobRefLivenessClassification
    applied: bool
    deleted_count: int
    receipt_path: Path | None = None
    backup_manifest: Path | None = None

    def to_dict(self, *, sample_limit: int = 30) -> dict[str, object]:
        return {
            "source_db": self.source_db,
            "dry_run": self.dry_run,
            "applied": self.applied,
            "deleted_count": self.deleted_count,
            "receipt_path": str(self.receipt_path) if self.receipt_path is not None else None,
            "backup_manifest": str(self.backup_manifest) if self.backup_manifest is not None else None,
            **self.classification.to_dict(sample_limit=sample_limit),
        }


def _offline_config(archive_root: Path) -> Config:
    return Config(archive_root=archive_root, render_root=render_root(), sources=[])


def _checkpoint_source_db(conn: sqlite3.Connection) -> None:
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    except sqlite3.Error as exc:
        raise BlobRefLivenessReconciliationError("could not checkpoint source.db before backup validation") from exc
    if row is None:
        raise BlobRefLivenessReconciliationError("could not checkpoint source.db before backup validation")


def _candidate_digest(classification: BlobRefLivenessClassification) -> str:
    encoded = json.dumps(
        [candidate.to_dict() for candidate in classification.candidates],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_prepared_receipt(
    receipt_path: Path,
    source_db: Path,
    classification: BlobRefLivenessClassification,
    backup_manifest: Path,
) -> None:
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "kind": "blob_ref_liveness_reconciliation",
        "phase": "prepared",
        "tool_version": TOOL_VERSION,
        "source_db": str(source_db),
        "backup_manifest": str(backup_manifest),
        "prepared_at_ms": int(time.time() * 1000),
        "candidate_count": len(classification.candidates),
        "candidate_digest": _candidate_digest(classification),
        "orphaned_by_ref_type": dict(classification.orphaned_by_ref_type),
        "referent_joins": [
            {"ref_type": ref_type, "referent_table": table, "referent_column": column}
            for ref_type, table, column in classification.ref_type_joins
        ],
    }
    try:
        with receipt_path.open("x", encoding="utf-8") as handle:
            rows = [header]
            rows.extend({"kind": "candidate", **candidate.to_dict()} for candidate in classification.candidates)
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise BlobRefLivenessReconciliationError(f"receipt already exists: {receipt_path}") from exc


def _append_receipt_footer(
    receipt_path: Path,
    *,
    phase: str,
    deleted_count: int | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "kind": "blob_ref_liveness_reconciliation",
        "phase": phase,
        "completed_at_ms": int(time.time() * 1000),
    }
    if deleted_count is not None:
        payload["deleted_count"] = deleted_count
    if error is not None:
        payload["error"] = error
    with receipt_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _delete_candidates(conn: sqlite3.Connection, classification: BlobRefLivenessClassification) -> int:
    conn.execute(
        """
        CREATE TEMP TABLE blob_ref_liveness_candidates (
            blob_hash BLOB NOT NULL,
            ref_type TEXT NOT NULL,
            ref_id TEXT NOT NULL,
            PRIMARY KEY (blob_hash, ref_type, ref_id)
        ) STRICT
        """
    )
    conn.executemany(
        "INSERT INTO blob_ref_liveness_candidates (blob_hash, ref_type, ref_id) VALUES (?, ?, ?)",
        (
            (bytes.fromhex(candidate.blob_hash), candidate.ref_type, candidate.ref_id)
            for candidate in classification.candidates
        ),
    )
    deleted = conn.execute(
        """
        DELETE FROM blob_refs
        WHERE EXISTS (
            SELECT 1
            FROM blob_ref_liveness_candidates AS c
            WHERE c.blob_hash = blob_refs.blob_hash
              AND c.ref_type = blob_refs.ref_type
              AND c.ref_id = blob_refs.ref_id
        )
        """
    )
    return max(0, int(deleted.rowcount))


def reconcile_blob_ref_liveness(
    archive_root: Path,
    *,
    backup_manifest: Path | None = None,
    receipt_path: Path | None = None,
    dry_run: bool = True,
) -> BlobRefLivenessReconciliationReport:
    """Classify orphaned source-tier refs, or apply the exact locked plan.

    Apply requires both a verified source-tier backup manifest and a receipt
    path. Classification is repeated after ``BEGIN IMMEDIATE`` and the
    receipt is fsynced before the set-based DELETE starts.
    """

    source_db = archive_root / "source.db"
    if not source_db.exists():
        raise FileNotFoundError(f"no source.db at {source_db}")

    if dry_run:
        with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True) as conn:
            dry_classification = classify_blob_ref_liveness(conn)
        return BlobRefLivenessReconciliationReport(
            source_db=str(source_db),
            dry_run=True,
            classification=dry_classification,
            applied=False,
            deleted_count=0,
        )

    if backup_manifest is None:
        raise BlobRefLivenessReconciliationError(
            "applying blob-ref liveness reconciliation requires a verified backup manifest (--backup-manifest)"
        )
    if receipt_path is None:
        raise BlobRefLivenessReconciliationError(
            "applying blob-ref liveness reconciliation requires a receipt path (--receipt-file)"
        )
    if reason := offline_maintenance_block_reason(_offline_config(archive_root), active=True, dry_run=False):
        raise BlobRefLivenessReconciliationError(reason)

    conn = sqlite3.connect(source_db)
    classification: BlobRefLivenessClassification | None = None
    prepared = False
    try:
        _checkpoint_source_db(conn)
        validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=conn)
            classification = classify_blob_ref_liveness(conn)
            if not classification.safe_to_apply:
                raise BlobRefLivenessReconciliationError(
                    "ref types cannot be proven with source-tier joins: "
                    f"unknown={classification.unknown_ref_types!r}, "
                    f"unavailable={classification.unavailable_ref_types!r}"
                )
            _write_prepared_receipt(receipt_path, source_db, classification, backup_manifest)
            prepared = True
            deleted_count = _delete_candidates(conn, classification)
            if deleted_count != len(classification.candidates):
                raise BlobRefLivenessReconciliationError(
                    f"candidate/delete count mismatch: planned={len(classification.candidates)} deleted={deleted_count}"
                )
            quick_check = conn.execute("PRAGMA quick_check").fetchone()
            if quick_check is None or str(quick_check[0]).lower() != "ok":
                raise BlobRefLivenessReconciliationError(f"source.db quick_check failed: {quick_check!r}")
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        else:
            conn.commit()
    except Exception as exc:
        if prepared:
            with suppress(OSError):
                _append_receipt_footer(receipt_path, phase="aborted", error=str(exc))
        raise
    finally:
        conn.close()

    assert classification is not None
    try:
        _append_receipt_footer(receipt_path, phase="committed", deleted_count=len(classification.candidates))
    except OSError as exc:
        raise BlobRefLivenessReconciliationError(
            f"source.db committed but could not finalize receipt {receipt_path}"
        ) from exc
    return BlobRefLivenessReconciliationReport(
        source_db=str(source_db),
        dry_run=False,
        classification=classification,
        applied=True,
        deleted_count=len(classification.candidates),
        receipt_path=receipt_path,
        backup_manifest=backup_manifest,
    )


__all__ = [
    "BlobRefLivenessReconciliationError",
    "BlobRefLivenessReconciliationReport",
    "TOOL_VERSION",
    "reconcile_blob_ref_liveness",
]
