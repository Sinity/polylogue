"""Apply a reviewed terminal disposition to retained historical raw failures.

The live writer records a typed ``raw_artifacts`` outcome as soon as it knows
a malformed or non-conversation payload is terminal. Historical rows created
before that route retain the bytes and parser diagnostic but lack the outcome.
This actuator closes only an explicit JSONL manifest, after a verified source
backup, and writes one immutable receipt per changed raw. It never clears the
original parser error, deletes bytes, runs GC, or retries a payload.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from polylogue.config import Config
from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason
from polylogue.paths import render_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import validate_migration_backup_manifest

TOOL_VERSION = "raw-failure-disposition-apply-v1"
_TERMINAL_KINDS = frozenset(
    {
        RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT,
        RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE,
    }
)


class RawFailureDispositionApplyError(RuntimeError):
    """Raised when a historical disposition would weaken raw evidence."""


@dataclass(frozen=True, slots=True)
class RawFailureDisposition:
    raw_id: str
    disposition_kind: RawFailureEvidenceKind
    detail: str


@dataclass(frozen=True, slots=True)
class RawFailureDispositionPlan:
    manifest_sha256: str
    candidates: tuple[RawFailureDisposition, ...]


@dataclass(frozen=True, slots=True)
class RawFailureDispositionApplyReport:
    manifest_sha256: str
    candidate_count: int
    disposed_raw_ids: tuple[str, ...]
    applied: bool
    backup_manifest: Path | None = None


def _offline_config(archive_root: Path) -> Config:
    return Config(archive_root=archive_root, render_root=render_root(), sources=[])


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw_failure_disposition_manifest(path: Path) -> RawFailureDispositionPlan:
    """Load an exact, immutable JSONL disposition manifest.

    Entries must name one raw id, a closed terminal kind, and a sanitized
    reason. Duplicate raw ids and unknown keys are rejected so the operator's
    reviewed manifest cannot silently collapse or broaden its scope.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RawFailureDispositionApplyError(f"could not read disposition manifest: {path}") from exc
    if not lines:
        raise RawFailureDispositionApplyError("disposition manifest is empty")
    candidates: list[RawFailureDisposition] = []
    seen: set[str] = set()
    for line_no, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RawFailureDispositionApplyError(f"manifest line {line_no} is not JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {"raw_id", "disposition_kind", "detail"}:
            raise RawFailureDispositionApplyError(
                f"manifest line {line_no} must contain exactly raw_id, disposition_kind, detail"
            )
        raw_id = payload["raw_id"]
        detail = payload["detail"]
        if not isinstance(raw_id, str) or not raw_id or not isinstance(detail, str) or not detail.strip():
            raise RawFailureDispositionApplyError(f"manifest line {line_no} has invalid raw_id or detail")
        try:
            kind = RawFailureEvidenceKind(str(payload["disposition_kind"]))
        except ValueError as exc:
            raise RawFailureDispositionApplyError(f"manifest line {line_no} has unknown disposition kind") from exc
        if kind not in _TERMINAL_KINDS:
            raise RawFailureDispositionApplyError(f"manifest line {line_no} is not a terminal disposition")
        if raw_id in seen:
            raise RawFailureDispositionApplyError(f"manifest repeats raw_id: {raw_id}")
        seen.add(raw_id)
        candidates.append(RawFailureDisposition(raw_id=raw_id, disposition_kind=kind, detail=detail.strip()))
    return RawFailureDispositionPlan(manifest_sha256=_manifest_sha256(path), candidates=tuple(candidates))


def _validate_candidate(conn: sqlite3.Connection, candidate: RawFailureDisposition) -> sqlite3.Row:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT r.raw_id, r.origin, r.source_path, r.source_index, r.blob_hash, r.blob_size,
               r.parse_error, r.validation_status, a.artifact_id
        FROM raw_sessions AS r
        JOIN raw_artifacts AS a
          ON a.raw_id = r.raw_id
         AND a.origin = r.origin
         AND a.source_path = r.source_path
         AND a.source_index = r.source_index
        WHERE r.raw_id = ?
          AND r.parse_error IS NOT NULL
          AND TRIM(r.parse_error) != ''
          AND NOT EXISTS (
              SELECT 1 FROM raw_failure_disposition_receipts AS receipt WHERE receipt.raw_id = r.raw_id
          )
        """,
        (candidate.raw_id,),
    ).fetchone()
    if row is None:
        raise RawFailureDispositionApplyError(
            f"raw {candidate.raw_id} is missing, has no matching artifact, is not failed, or was already disposed"
        )
    return cast(sqlite3.Row, row)


def _checkpoint_live_tier(conn: sqlite3.Connection) -> None:
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    except sqlite3.Error as exc:
        raise RawFailureDispositionApplyError("could not checkpoint source.db before backup validation") from exc
    if row is None:
        raise RawFailureDispositionApplyError("could not checkpoint source.db before backup validation")


def _apply_candidate(
    conn: sqlite3.Connection,
    candidate: RawFailureDisposition,
    *,
    manifest_sha256: str,
    backup_manifest: Path,
    disposed_at_ms: int,
) -> str:
    row = _validate_candidate(conn, candidate)
    updated = conn.execute(
        """
        UPDATE raw_artifacts
        SET artifact_kind = ?, support_status = ?, classification_reason = ?,
            parse_as_session = 0, schema_eligible = 0, last_observed_at_ms = ?
        WHERE artifact_id = ?
        """,
        (
            candidate.disposition_kind.value,
            candidate.disposition_kind.support_status.value,
            candidate.disposition_kind.value,
            disposed_at_ms,
            row["artifact_id"],
        ),
    )
    if updated.rowcount != 1:
        raise RawFailureDispositionApplyError(f"artifact changed during disposition: {candidate.raw_id}")
    conn.execute(
        """
        INSERT INTO raw_failure_disposition_receipts (
            raw_id, artifact_id, origin, source_path, source_index, blob_hash, blob_size,
            previous_parse_error, previous_validation_status, disposition_kind, manifest_sha256,
            disposed_at_ms, tool_version, backup_manifest_path, detail
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["raw_id"],
            row["artifact_id"],
            row["origin"],
            row["source_path"],
            row["source_index"],
            row["blob_hash"],
            row["blob_size"],
            row["parse_error"],
            row["validation_status"],
            candidate.disposition_kind.value,
            manifest_sha256,
            disposed_at_ms,
            TOOL_VERSION,
            str(backup_manifest),
            candidate.detail,
        ),
    )
    return str(row["raw_id"])


def apply_raw_failure_dispositions(
    archive_root: Path,
    *,
    manifest_path: Path,
    backup_manifest: Path | None = None,
    dry_run: bool = True,
) -> RawFailureDispositionApplyReport:
    """Validate and, with backup authorization, apply terminal dispositions."""
    source_db = archive_root / "source.db"
    if not source_db.exists():
        raise FileNotFoundError(f"no source.db at {source_db}")
    plan = load_raw_failure_disposition_manifest(manifest_path)
    if dry_run:
        conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
        try:
            for candidate in plan.candidates:
                _validate_candidate(conn, candidate)
        finally:
            conn.close()
        return RawFailureDispositionApplyReport(plan.manifest_sha256, len(plan.candidates), (), False)
    if backup_manifest is None:
        raise RawFailureDispositionApplyError(
            "applying raw-failure dispositions requires a verified backup manifest (--backup-manifest)"
        )
    if reason := offline_maintenance_block_reason(_offline_config(archive_root), active=True, dry_run=False):
        raise RawFailureDispositionApplyError(reason)
    conn = sqlite3.connect(source_db)
    disposed: list[str] = []
    try:
        _checkpoint_live_tier(conn)
        validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=conn)
            disposed_at_ms = int(time.time() * 1000)
            for candidate in plan.candidates:
                disposed.append(
                    _apply_candidate(
                        conn,
                        candidate,
                        manifest_sha256=plan.manifest_sha256,
                        backup_manifest=backup_manifest,
                        disposed_at_ms=disposed_at_ms,
                    )
                )
            quick_check = conn.execute("PRAGMA quick_check").fetchone()
            if quick_check is None or str(quick_check[0]).lower() != "ok":
                raise RawFailureDispositionApplyError(
                    f"source.db quick_check failed after disposition: {quick_check!r}"
                )
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        else:
            conn.commit()
    finally:
        conn.close()
    return RawFailureDispositionApplyReport(
        plan.manifest_sha256, len(plan.candidates), tuple(disposed), True, backup_manifest
    )


__all__ = [
    "RawFailureDisposition",
    "RawFailureDispositionApplyError",
    "RawFailureDispositionApplyReport",
    "RawFailureDispositionPlan",
    "apply_raw_failure_dispositions",
    "load_raw_failure_disposition_manifest",
]
