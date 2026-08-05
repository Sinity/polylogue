"""Dry-run-first reconciliation of historical source-tier blob references."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Iterable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from polylogue.config import Config
from polylogue.maintenance.offline_guard import running_daemon_pid
from polylogue.paths import render_root
from polylogue.storage.blob_gc import BLOB_REF_LIVENESS_JOIN
from polylogue.storage.blob_ref_liveness import (
    BlobRefLivenessCandidate,
    BlobRefLivenessClassification,
    classify_blob_ref_liveness,
    stage_blob_ref_liveness,
)
from polylogue.storage.hook_payload_ref_reconciliation import _deterministic_raw_session_id_udf
from polylogue.storage.introspection import table_exists as _table_exists
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import (
    validate_migration_backup_live_fingerprint,
    validate_migration_backup_manifest,
)

TOOL_VERSION = "blob-ref-liveness-reconciliation-v1"
BATCH_SIZE = 1_000
_RECOVERY_TERMINAL_PHASES = {
    "committed",
    "aborted",
    "recovered_committed",
    "recovered_rolled_back",
    "recovered_partial",
    "indeterminate",
}


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


def _offline_apply_block_reason(archive_root: Path) -> str | None:
    """Refuse this offline-only repair whenever its archive daemon is live."""

    daemon_pid = running_daemon_pid(_offline_config(archive_root))
    if daemon_pid is None:
        return None
    return (
        f"Refusing offline blob-ref liveness reconciliation while polylogued PID {daemon_pid} is running. "
        "Stop polylogued before applying this repair."
    )


def _checkpoint_source_db(conn: sqlite3.Connection) -> None:
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    except sqlite3.Error as exc:
        raise BlobRefLivenessReconciliationError("could not checkpoint source.db before backup validation") from exc
    if row is None:
        raise BlobRefLivenessReconciliationError("could not checkpoint source.db before backup validation")


def _candidate_digest(candidates: Iterable[BlobRefLivenessCandidate]) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for candidate in candidates:
        if not first:
            digest.update(b",")
        digest.update(json.dumps(candidate.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8"))
        first = False
    digest.update(b"]")
    return digest.hexdigest()


def _iter_staged_candidates(conn: sqlite3.Connection, table_name: str) -> Iterator[BlobRefLivenessCandidate]:
    for row in conn.execute(
        f"""
        SELECT blob_hash, ref_type, ref_id, source_path, size_bytes,
               acquired_at_ms, referent_table, referent_column
        FROM {table_name}
        ORDER BY ref_type, ref_id, blob_hash
        """
    ):
        yield BlobRefLivenessCandidate(
            blob_hash=bytes(row[0]).hex(),
            ref_type=str(row[1]),
            ref_id=str(row[2]),
            source_path=str(row[3]) if row[3] is not None else None,
            size_bytes=int(row[4]),
            acquired_at_ms=int(row[5]),
            referent_table=str(row[6]),
            referent_column=str(row[7]),
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_prepared_receipt(
    receipt_path: Path,
    source_db: Path,
    classification: BlobRefLivenessClassification,
    backup_manifest: Path,
    *,
    candidates: Iterable[BlobRefLivenessCandidate] | None = None,
    candidate_digest: str | None = None,
) -> None:
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "kind": "blob_ref_liveness_reconciliation",
        "phase": "prepared",
        "tool_version": TOOL_VERSION,
        "source_db": str(source_db),
        "backup_manifest": str(backup_manifest),
        "prepared_at_ms": int(time.time() * 1000),
        "candidate_count": classification.orphaned_count,
        "candidate_digest": candidate_digest
        or _candidate_digest(classification.candidates if candidates is None else candidates),
        "orphaned_by_ref_type": dict(classification.orphaned_by_ref_type),
        "referent_joins": [
            {"ref_type": ref_type, "referent_table": table, "referent_column": column}
            for ref_type, table, column in classification.ref_type_joins
        ],
    }
    try:
        with receipt_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(header, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            for candidate in classification.candidates if candidates is None else candidates:
                handle.write(
                    json.dumps({"kind": "candidate", **candidate.to_dict()}, sort_keys=True, separators=(",", ":"))
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(receipt_path.parent)
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
    _fsync_directory(receipt_path.parent)


def _receipt_candidate(row: dict[str, object]) -> tuple[bytes, str, str]:
    try:
        return bytes.fromhex(str(row["blob_hash"])), str(row["ref_type"]), str(row["ref_id"])
    except (KeyError, ValueError) as exc:
        raise BlobRefLivenessReconciliationError("prepared receipt contains an invalid candidate") from exc


def _stage_receipt_candidates(conn: sqlite3.Connection, receipt_path: Path) -> tuple[int, str, dict[str, object]]:
    table_name = "blob_ref_liveness_receipt_candidates"
    conn.execute(f"DROP TABLE IF EXISTS temp.{table_name}")
    conn.execute(
        f"""
        CREATE TEMP TABLE {table_name} (
            blob_hash BLOB NOT NULL,
            ref_type TEXT NOT NULL,
            ref_id TEXT NOT NULL,
            PRIMARY KEY (blob_hash, ref_type, ref_id)
        ) STRICT
        """
    )
    header: dict[str, object] | None = None
    terminal_phases: list[str] = []
    digest = hashlib.sha256()
    digest.update(b"[")
    first_candidate = True
    candidate_count = 0
    with receipt_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BlobRefLivenessReconciliationError(f"receipt is not valid JSON: {receipt_path}") from exc
            if not isinstance(row, dict):
                raise BlobRefLivenessReconciliationError(f"receipt row is not an object: {receipt_path}")
            if row.get("kind") == "blob_ref_liveness_reconciliation":
                if header is None:
                    header = row
                else:
                    phase = row.get("phase")
                    if phase is not None and str(phase) in _RECOVERY_TERMINAL_PHASES:
                        terminal_phases.append(str(phase))
            elif row.get("kind") == "candidate":
                candidate = _receipt_candidate(row)
                conn.execute(f"INSERT INTO {table_name} (blob_hash, ref_type, ref_id) VALUES (?, ?, ?)", candidate)
                if not first_candidate:
                    digest.update(b",")
                digest.update(
                    json.dumps(
                        {
                            "blob_hash": candidate[0].hex(),
                            "ref_id": candidate[2],
                            "ref_type": candidate[1],
                            "source_path": row.get("source_path"),
                            "size_bytes": row.get("size_bytes"),
                            "acquired_at_ms": row.get("acquired_at_ms"),
                            "referent_table": row.get("referent_table"),
                            "referent_column": row.get("referent_column"),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                first_candidate = False
                candidate_count += 1
    if header is None or header.get("phase") != "prepared":
        raise BlobRefLivenessReconciliationError(f"receipt does not contain a prepared plan: {receipt_path}")
    if terminal_phases:
        raise BlobRefLivenessReconciliationError(f"receipt is already terminal: {receipt_path}")
    receipt_candidate_count = header.get("candidate_count")
    if not isinstance(receipt_candidate_count, int) or receipt_candidate_count != candidate_count:
        raise BlobRefLivenessReconciliationError(f"prepared receipt candidate count mismatch: {receipt_path}")
    digest.update(b"]")
    if str(header.get("candidate_digest")) != digest.hexdigest():
        raise BlobRefLivenessReconciliationError(f"prepared receipt candidate digest mismatch: {receipt_path}")
    return candidate_count, table_name, header


def _recover_prepared_receipt(source_db: Path, receipt_path: Path) -> str:
    """Resolve crashes between bounded batch commits and receipt progress.

    Each batch is one SQLite transaction. Exact receipt keys therefore show a
    fully rolled-back plan, a fully committed plan, or a partial batch train.
    Recovery appends durable evidence but never performs another mutation.
    """

    with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True) as conn:
        candidate_count, table_name, _header = _stage_receipt_candidates(conn, receipt_path)
        present_count = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM {table_name} AS c
                JOIN blob_refs AS b
                  ON b.blob_hash = c.blob_hash
                 AND b.ref_type = c.ref_type
                 AND b.ref_id = c.ref_id
                """
            ).fetchone()[0]
        )
    if present_count == candidate_count:
        outcome = "recovered_rolled_back"
    elif present_count == 0:
        outcome = "recovered_committed"
    else:
        outcome = "recovered_partial"
    _append_receipt_footer(
        receipt_path,
        phase=outcome,
        deleted_count=candidate_count - present_count,
        error=None if outcome != "recovered_partial" else f"{present_count} candidate(s) remain present",
    )
    return outcome


def _stage_candidate_batch(conn: sqlite3.Connection, candidate_table: str, batch_table: str, *, batch_size: int) -> int:
    conn.execute(f"DROP TABLE IF EXISTS temp.{batch_table}")
    conn.execute(
        f"""
        CREATE TEMP TABLE {batch_table} AS
        SELECT blob_hash, ref_type, ref_id, source_path, size_bytes,
               acquired_at_ms, referent_table, referent_column
        FROM {candidate_table}
        ORDER BY ref_type, ref_id, blob_hash
        LIMIT ?
        """,
        (batch_size,),
    )
    conn.execute(f"CREATE INDEX {batch_table}_source_hash ON {batch_table}(ref_type, source_path, blob_hash)")
    return int(conn.execute(f"SELECT COUNT(*) FROM {batch_table}").fetchone()[0])


def _delete_candidate_batch(conn: sqlite3.Connection, candidate_table: str, batch_table: str) -> int:
    planned_count = int(conn.execute(f"SELECT COUNT(*) FROM {batch_table}").fetchone()[0])
    deleted = conn.execute(
        f"""
        DELETE FROM blob_refs
        WHERE rowid IN (
            SELECT b.rowid
            FROM blob_refs AS b
            JOIN {batch_table} AS c
              ON c.blob_hash = b.blob_hash
             AND c.ref_type = b.ref_type
             AND c.ref_id = b.ref_id
        )
        """
    )
    deleted_count = max(0, int(deleted.rowcount))
    if deleted_count != planned_count:
        raise BlobRefLivenessReconciliationError(
            f"candidate/delete count mismatch: planned={planned_count} deleted={deleted_count}"
        )
    conn.execute(
        f"""
        DELETE FROM {candidate_table}
        WHERE EXISTS (
            SELECT 1 FROM {batch_table} AS b
            WHERE b.blob_hash = {candidate_table}.blob_hash
              AND b.ref_type = {candidate_table}.ref_type
              AND b.ref_id = {candidate_table}.ref_id
        )
        """
    )
    return deleted_count


def _validate_locked_candidate_plan(conn: sqlite3.Connection, candidate_table: str, expected_count: int) -> None:
    present_count = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM {candidate_table} AS c
            JOIN blob_refs AS b
              ON b.blob_hash = c.blob_hash
             AND b.ref_type = c.ref_type
             AND b.ref_id = c.ref_id
            """
        ).fetchone()[0]
    )
    if present_count != expected_count:
        raise BlobRefLivenessReconciliationError(
            f"prepared candidate presence mismatch: planned={expected_count} present={present_count}"
        )
    referent_branches = [
        f"""
        SELECT c.blob_hash, c.ref_type, c.ref_id
        FROM {candidate_table} AS c
        JOIN {table} AS r ON r.{column} = c.ref_id
        WHERE c.ref_type = ?
        """
        for ref_type, (table, column) in {
            ref_type: (table, column) for ref_type, table, column in BLOB_REF_LIVENESS_JOIN
        }.items()
    ]
    if referent_branches:
        live_referent_count = int(
            conn.execute(
                f"SELECT COUNT(*) FROM ({' UNION ALL '.join(referent_branches)})",
                tuple(ref_type for ref_type, _table, _column in BLOB_REF_LIVENESS_JOIN),
            ).fetchone()[0]
        )
        if live_referent_count:
            raise BlobRefLivenessReconciliationError(
                f"prepared candidate referents became live under lock: {live_referent_count}"
            )
    conn.execute("DROP TABLE IF EXISTS temp.blob_ref_liveness_locked_hooks")
    if _table_exists(conn, "raw_hook_events"):
        conn.execute(
            f"""
            CREATE TEMP TABLE blob_ref_liveness_locked_hooks AS
            SELECT h.hook_event_id, h.origin, h.native_id, h.source_path, h.blob_hash
            FROM raw_hook_events AS h
            WHERE EXISTS (
                SELECT 1
                FROM {candidate_table} AS c
                WHERE c.ref_type = 'raw_payload'
                  AND c.source_path IS h.source_path
                  AND (h.blob_hash IS NULL OR h.blob_hash = c.blob_hash)
            )
              AND NOT EXISTS (
                SELECT 1 FROM blob_refs AS b
                WHERE b.blob_hash = h.blob_hash
                  AND b.ref_type = 'hook_payload'
                  AND b.ref_id = h.hook_event_id
            )
            """
        )
    else:
        conn.execute(
            """
            CREATE TEMP TABLE blob_ref_liveness_locked_hooks (
                hook_event_id TEXT NOT NULL,
                origin TEXT,
                native_id TEXT,
                source_path TEXT,
                blob_hash BLOB
            )
            """
        )
    conn.execute(
        "CREATE INDEX blob_ref_liveness_locked_hooks_source_hash "
        "ON blob_ref_liveness_locked_hooks(source_path, blob_hash)"
    )
    conn.create_function("polylogue_deterministic_raw_session_id", 5, _deterministic_raw_session_id_udf)
    rekeyable = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT c.blob_hash, c.ref_id
                FROM {candidate_table} AS c
                JOIN blob_ref_liveness_locked_hooks AS h
                  ON h.source_path IS c.source_path
                 AND h.blob_hash = c.blob_hash
                 AND polylogue_deterministic_raw_session_id(h.origin, c.source_path, 0, c.blob_hash, h.native_id) = c.ref_id
                WHERE c.ref_type = 'raw_payload'
                GROUP BY c.blob_hash, c.ref_id
                HAVING COUNT(*) = 1
                UNION ALL
                SELECT c.blob_hash, c.ref_id
                FROM {candidate_table} AS c
                JOIN blob_ref_liveness_locked_hooks AS h
                  ON h.source_path IS c.source_path
                 AND h.blob_hash IS NULL
                 AND polylogue_deterministic_raw_session_id(h.origin, c.source_path, 0, c.blob_hash, h.native_id) = c.ref_id
                WHERE c.ref_type = 'raw_payload'
                  AND (SELECT COUNT(*) FROM {candidate_table} AS c2
                       WHERE c2.ref_type = 'raw_payload' AND c2.source_path IS c.source_path) = 1
                  AND (SELECT COUNT(*) FROM blob_ref_liveness_locked_hooks AS h2
                       WHERE h2.source_path IS h.source_path AND h2.blob_hash IS NULL) = 1
            )
            """
        ).fetchone()[0]
    )
    if rekeyable:
        raise BlobRefLivenessReconciliationError(
            f"prepared candidates became rekeyable hook payloads under lock: {rekeyable}"
        )


def reconcile_blob_ref_liveness(
    archive_root: Path,
    *,
    backup_manifest: Path | None = None,
    receipt_path: Path | None = None,
    dry_run: bool = True,
) -> BlobRefLivenessReconciliationReport:
    """Classify orphaned source-tier refs, or apply the exact locked plan.

    Apply requires both a verified source-tier backup manifest and a receipt
    path. A compact SQLite plan and prepared receipt are built before
    ownership; exact candidate keys and rekeyable hook matches are rechecked
    after ``BEGIN IMMEDIATE`` before bounded deletes start.
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
    if receipt_path.exists():
        outcome = _recover_prepared_receipt(source_db, receipt_path)
        raise BlobRefLivenessReconciliationError(
            f"recovered existing prepared receipt as {outcome}: {receipt_path}; choose a fresh receipt path before retrying"
        )
    if reason := _offline_apply_block_reason(archive_root):
        raise BlobRefLivenessReconciliationError(reason)

    pre_conn = sqlite3.connect(source_db)
    staged_plan = None
    try:
        _checkpoint_source_db(pre_conn)
        validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=pre_conn)
        staged_plan = stage_blob_ref_liveness(pre_conn)
        classification = staged_plan.classification
        if not classification.safe_to_apply:
            raise BlobRefLivenessReconciliationError(
                "ref types cannot be proven with source-tier joins: "
                f"unknown={classification.unknown_ref_types!r}, "
                f"unavailable={classification.unavailable_ref_types!r}, "
                f"rekeyable_hook_payloads={classification.rekeyable_hook_payload_count!r}. "
                "Run hook-payload reference reconciliation before deleting orphan refs."
            )
        candidates = _iter_staged_candidates(pre_conn, staged_plan.candidate_table)
        candidate_digest = _candidate_digest(_iter_staged_candidates(pre_conn, staged_plan.candidate_table))
        _write_prepared_receipt(
            receipt_path,
            source_db,
            classification,
            backup_manifest,
            candidates=candidates,
            candidate_digest=candidate_digest,
        )
        expected_count = classification.orphaned_count
        # Temp tables survive a commit. Clearing the staging transaction here
        # lets this same connection carry the exact plan into ownership without
        # reparsing the durable receipt under the write lock.
        pre_conn.commit()
    except Exception:
        pre_conn.close()
        raise

    conn = pre_conn
    deleted_count = 0
    first_batch = True
    batch_table = "blob_ref_liveness_batch"
    try:
        while True:
            batch_count = _stage_candidate_batch(
                conn,
                staged_plan.candidate_table,
                batch_table,
                batch_size=BATCH_SIZE,
            )
            if batch_count == 0:
                break
            conn.execute("BEGIN IMMEDIATE")
            try:
                if first_batch:
                    # This is the ownership-boundary attestation. The helper
                    # re-authenticates the receipt and validates the cached
                    # full backup inventory, rehashing only if its stat
                    # signature changed since the pre-lock validation.
                    validate_migration_backup_live_fingerprint(backup_manifest, ArchiveTier.SOURCE, connection=conn)
                _validate_locked_candidate_plan(conn, batch_table, batch_count)
                batch_deleted = _delete_candidate_batch(conn, staged_plan.candidate_table, batch_table)
            except Exception:
                if conn.in_transaction:
                    conn.rollback()
                # Leave the prepared receipt without a terminal footer. A
                # retry can inspect exact key presence and record rollback,
                # commit, or partial-batch recovery without guessing.
                raise
            else:
                conn.commit()
            deleted_count += batch_deleted
            first_batch = False
            _append_receipt_footer(receipt_path, phase="batch_committed", deleted_count=deleted_count)
        if deleted_count != expected_count:
            raise BlobRefLivenessReconciliationError(
                f"candidate/delete count mismatch: planned={expected_count} deleted={deleted_count}"
            )
    finally:
        conn.close()

    try:
        _append_receipt_footer(receipt_path, phase="committed", deleted_count=deleted_count)
    except OSError as exc:
        raise BlobRefLivenessReconciliationError(
            f"source.db committed but could not finalize receipt {receipt_path}"
        ) from exc

    try:
        with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True) as verify_conn:
            quick_check = verify_conn.execute("PRAGMA quick_check").fetchone()
        if quick_check is None or str(quick_check[0]).lower() != "ok":
            with suppress(OSError):
                _append_receipt_footer(receipt_path, phase="post_check_failed", error=f"quick_check={quick_check!r}")
            raise BlobRefLivenessReconciliationError(f"source.db quick_check failed after commit: {quick_check!r}")
    finally:
        pass

    assert staged_plan is not None
    return BlobRefLivenessReconciliationReport(
        source_db=str(source_db),
        dry_run=False,
        classification=staged_plan.classification,
        applied=True,
        deleted_count=expected_count,
        receipt_path=receipt_path,
        backup_manifest=backup_manifest,
    )


__all__ = [
    "BlobRefLivenessReconciliationError",
    "BlobRefLivenessReconciliationReport",
    "TOOL_VERSION",
    "reconcile_blob_ref_liveness",
]
