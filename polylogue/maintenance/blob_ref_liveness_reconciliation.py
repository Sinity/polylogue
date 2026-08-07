"""Dry-run-first reconciliation of historical source-tier blob references."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Concatenate, ParamSpec, TypeVar, cast

if TYPE_CHECKING:
    from polylogue.storage.blob_gc import OrphanedBlobRefCensus

from polylogue.config import Config
from polylogue.daemon.write_coordinator import daemon_write_lease_active
from polylogue.maintenance.offline_guard import running_daemon_pid
from polylogue.paths import render_root
from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation
from polylogue.storage.blob_ref_liveness import (
    BlobRefLivenessCandidate,
    BlobRefLivenessCandidateDigest,
    BlobRefLivenessClassification,
    classify_blob_ref_liveness,
    digest_blob_ref_liveness_candidates,
    stage_blob_ref_liveness,
    validated_blob_ref_liveness_joins,
)
from polylogue.storage.hook_payload_ref_reconciliation import _deterministic_raw_session_id_udf
from polylogue.storage.introspection import table_exists as _table_exists
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.durable_change_train import (
    DurableChangeTrainError,
    DurableSourceContinuitySemanticError,
    DurableSourceTrainMissingError,
    assert_source_continuity_apply_allowed,
    clear_source_continuity_pending_intent,
    mark_source_continuity_pending_intent_terminal,
    refresh_released_source_train_continuity,
    write_source_continuity_pending_intent,
)
from polylogue.storage.sqlite.migration_runner import (
    capture_durable_database_evidence,
    validate_migration_backup_live_fingerprint,
    validate_migration_backup_manifest,
)

TOOL_VERSION = "blob-ref-liveness-reconciliation-v1"
BATCH_SIZE = 1_000
_LOCKED_HOOK_TABLE = "blob_ref_liveness_locked_hooks"
_LOCKED_MISSING_HOOK_TABLE = "blob_ref_liveness_locked_missing_hooks"
_LOCKED_IDENTITY_MATCH_TABLE = "blob_ref_liveness_locked_identity_matches"
_LOCKED_LEGACY_HOOK_COUNTS = "blob_ref_liveness_locked_legacy_hook_counts"
_RECOVERY_TERMINAL_PHASES = {
    "committed",
    "aborted",
    "recovered_committed",
    "recovered_rolled_back",
    "recovered_partial",
    "indeterminate",
    "postcondition_failed",
}


class BlobRefLivenessReconciliationError(RuntimeError):
    """Raised when reconciliation cannot prove a safe source-tier apply."""


_ArchiveOwnedParams = ParamSpec("_ArchiveOwnedParams")
_ArchiveOwnedResult = TypeVar("_ArchiveOwnedResult")


def _archive_owned(
    function: Callable[Concatenate[Path, _ArchiveOwnedParams], _ArchiveOwnedResult],
) -> Callable[Concatenate[Path, _ArchiveOwnedParams], _ArchiveOwnedResult]:
    """Hold the archive lease across the complete liveness operation."""

    @wraps(function)
    def wrapped(
        archive_root: Path, *args: _ArchiveOwnedParams.args, **kwargs: _ArchiveOwnedParams.kwargs
    ) -> _ArchiveOwnedResult:
        if bool(kwargs.get("dry_run", True)):
            return function(archive_root, *args, **kwargs)
        with OwnedArchiveLocation.acquire(
            ArchiveLocation.resolve(archive_root),
            owner_id=f"blob-ref-liveness:{os.getpid()}",
        ):
            return function(archive_root, *args, **kwargs)

    return cast(Callable[Concatenate[Path, _ArchiveOwnedParams], _ArchiveOwnedResult], wrapped)


@dataclass(frozen=True, slots=True)
class BlobRefLivenessReconciliationReport:
    source_db: str
    dry_run: bool
    classification: BlobRefLivenessClassification
    applied: bool
    deleted_count: int
    receipt_path: Path | None = None
    backup_manifest: Path | None = None
    post_classification: BlobRefLivenessClassification | None = None
    continuity_refresh_receipt: Path | None = None
    continuity_refresh_error: str | None = None
    continuity_refresh_pending: bool = False

    def to_dict(self, *, sample_limit: int = 30) -> dict[str, object]:
        return {
            "source_db": self.source_db,
            "dry_run": self.dry_run,
            "applied": self.applied,
            "deleted_count": self.deleted_count,
            "receipt_path": str(self.receipt_path) if self.receipt_path is not None else None,
            "backup_manifest": str(self.backup_manifest) if self.backup_manifest is not None else None,
            "continuity_refresh_receipt": (
                str(self.continuity_refresh_receipt) if self.continuity_refresh_receipt is not None else None
            ),
            "continuity_refresh_error": self.continuity_refresh_error,
            "continuity_refresh_pending": self.continuity_refresh_pending,
            "post_classification": self.post_classification.to_dict() if self.post_classification is not None else None,
            **self.classification.to_dict(sample_limit=sample_limit),
        }


def _offline_config(archive_root: Path) -> Config:
    return Config(archive_root=archive_root, render_root=render_root(), sources=[])


def _offline_apply_block_reason(archive_root: Path) -> str | None:
    """Refuse this offline-only repair whenever its archive daemon is live."""

    if daemon_write_lease_active():
        return "Refusing offline blob-ref liveness reconciliation while a daemon writer lease is active."
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
    if row is None or len(row) != 3:
        raise BlobRefLivenessReconciliationError("could not checkpoint source.db before backup validation")
    busy, log_frames, checkpointed_frames = (int(value) for value in row)
    if busy != 0 or log_frames != checkpointed_frames:
        raise BlobRefLivenessReconciliationError(
            "source.db WAL checkpoint was not clean; stop every writer and retry when no frames remain busy"
        )


def _source_data_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA data_version").fetchone()
    if row is None:
        raise BlobRefLivenessReconciliationError("could not read source.db data version")
    return int(row[0])


def _candidate_digest(candidates: Iterable[BlobRefLivenessCandidate]) -> str:
    return digest_blob_ref_liveness_candidates(candidates)


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
    backup_manifest_sha256: str | None = None,
) -> None:
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, object] = {
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
    if backup_manifest_sha256 is None and backup_manifest.is_file():
        backup_manifest_sha256 = hashlib.sha256(backup_manifest.read_bytes()).hexdigest()
    if backup_manifest_sha256 is not None:
        header["backup_manifest_sha256"] = backup_manifest_sha256
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
    post_orphaned_count: int | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "kind": "blob_ref_liveness_reconciliation",
        "phase": phase,
        "completed_at_ms": int(time.time() * 1000),
    }
    if deleted_count is not None:
        payload["deleted_count"] = deleted_count
    if post_orphaned_count is not None:
        payload["post_orphaned_count"] = post_orphaned_count
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


def _stage_receipt_candidates(
    conn: sqlite3.Connection,
    receipt_path: Path,
    *,
    allow_postcondition_failed: bool = False,
) -> tuple[int, str, dict[str, object]]:
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
    digest = BlobRefLivenessCandidateDigest()
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
                try:
                    digest.update(
                        BlobRefLivenessCandidate(
                            blob_hash=candidate[0].hex(),
                            ref_type=candidate[1],
                            ref_id=candidate[2],
                            source_path=str(row["source_path"]) if row.get("source_path") is not None else None,
                            size_bytes=int(row["size_bytes"]),
                            acquired_at_ms=int(row["acquired_at_ms"]),
                            referent_table=str(row["referent_table"]),
                            referent_column=str(row["referent_column"]),
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise BlobRefLivenessReconciliationError(
                        f"prepared receipt contains an invalid candidate: {receipt_path}"
                    ) from exc
                candidate_count += 1
    if header is None or header.get("phase") != "prepared":
        raise BlobRefLivenessReconciliationError(f"receipt does not contain a prepared plan: {receipt_path}")
    if terminal_phases and not (allow_postcondition_failed and terminal_phases == ["postcondition_failed"]):
        raise BlobRefLivenessReconciliationError(f"receipt is already terminal: {receipt_path}")
    receipt_candidate_count = header.get("candidate_count")
    if not isinstance(receipt_candidate_count, int) or receipt_candidate_count != candidate_count:
        raise BlobRefLivenessReconciliationError(f"prepared receipt candidate count mismatch: {receipt_path}")
    if str(header.get("candidate_digest")) != digest.hexdigest():
        raise BlobRefLivenessReconciliationError(f"prepared receipt candidate digest mismatch: {receipt_path}")
    return candidate_count, table_name, header


def _recover_prepared_receipt(
    source_db: Path,
    receipt_path: Path,
    *,
    allow_postcondition_failed: bool = False,
    postcondition_check: Callable[[], None] | None = None,
) -> str:
    """Resolve crashes between bounded batch commits and receipt progress.

    Each batch is one SQLite transaction. Exact receipt keys therefore show a
    fully rolled-back plan, a fully committed plan, or a partial batch train.
    Recovery appends durable evidence but never performs another mutation.
    """

    with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True) as conn:
        candidate_count, table_name, _header = _stage_receipt_candidates(
            conn,
            receipt_path,
            allow_postcondition_failed=allow_postcondition_failed,
        )
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
    if outcome == "recovered_committed" and postcondition_check is not None:
        postcondition_check()
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


def _stage_locked_legacy_hook_counts(conn: sqlite3.Connection, hook_table: str) -> str:
    """Aggregate legacy null-hash evidence once for the locked snapshot."""

    conn.execute(f"DROP TABLE IF EXISTS temp.{_LOCKED_LEGACY_HOOK_COUNTS}")
    conn.execute(
        f"""
        CREATE TEMP TABLE {_LOCKED_LEGACY_HOOK_COUNTS} AS
        SELECT source_path, COUNT(*) AS hook_count
        FROM {hook_table}
        WHERE blob_hash IS NULL
        GROUP BY source_path
        """
    )
    conn.execute(
        f"CREATE UNIQUE INDEX {_LOCKED_LEGACY_HOOK_COUNTS}_source_path ON {_LOCKED_LEGACY_HOOK_COUNTS}(source_path)"
    )
    return _LOCKED_LEGACY_HOOK_COUNTS


def _stage_locked_hook_snapshot(conn: sqlite3.Connection, candidate_table: str) -> str:
    """Capture all hook evidence relevant to the complete staged plan once."""

    for table in (
        _LOCKED_HOOK_TABLE,
        _LOCKED_MISSING_HOOK_TABLE,
        _LOCKED_IDENTITY_MATCH_TABLE,
        _LOCKED_LEGACY_HOOK_COUNTS,
    ):
        conn.execute(f"DROP TABLE IF EXISTS temp.{table}")
    conn.create_function("polylogue_deterministic_raw_session_id", 5, _deterministic_raw_session_id_udf)
    if _table_exists(conn, "raw_hook_events"):
        conn.execute(
            f"""
            CREATE TEMP TABLE {_LOCKED_HOOK_TABLE} AS
            SELECT h.hook_event_id, h.origin, h.native_id, h.source_path, h.blob_hash
            FROM (
                SELECT h.hook_event_id, h.origin, h.native_id, h.source_path, h.blob_hash
                FROM raw_hook_events AS h
                JOIN (
                    SELECT DISTINCT source_path, blob_hash
                    FROM {candidate_table}
                    WHERE ref_type = 'raw_payload' AND source_path IS NOT NULL
                ) AS c
                  ON c.source_path IS h.source_path
                 AND c.blob_hash = h.blob_hash
                UNION
                SELECT h.hook_event_id, h.origin, h.native_id, h.source_path, h.blob_hash
                FROM raw_hook_events AS h
                JOIN (
                    SELECT DISTINCT source_path
                    FROM {candidate_table}
                    WHERE ref_type = 'raw_payload' AND source_path IS NOT NULL
                ) AS c
                  ON c.source_path IS h.source_path
                WHERE h.blob_hash IS NULL
            ) AS h
            """
        )
    else:
        conn.execute(
            f"""
            CREATE TEMP TABLE {_LOCKED_HOOK_TABLE} (
                hook_event_id TEXT NOT NULL,
                origin TEXT,
                native_id TEXT,
                source_path TEXT,
                blob_hash BLOB
            )
            """
        )
    conn.execute(f"CREATE INDEX {_LOCKED_HOOK_TABLE}_source_hash ON {_LOCKED_HOOK_TABLE}(source_path, blob_hash)")
    conn.execute(
        f"""
        CREATE TEMP TABLE {_LOCKED_MISSING_HOOK_TABLE} AS
        SELECT h.hook_event_id, h.origin, h.native_id, h.source_path, h.blob_hash
        FROM {_LOCKED_HOOK_TABLE} AS h
        WHERE NOT EXISTS (
            SELECT 1 FROM blob_refs AS b
            WHERE b.ref_type = 'hook_payload'
              AND b.ref_id = h.hook_event_id
        )
        """
    )
    conn.execute(
        f"CREATE INDEX {_LOCKED_MISSING_HOOK_TABLE}_source_hash ON {_LOCKED_MISSING_HOOK_TABLE}(source_path, blob_hash)"
    )
    _stage_locked_legacy_hook_counts(conn, _LOCKED_HOOK_TABLE)
    conn.execute(
        f"""
        CREATE TEMP TABLE {_LOCKED_IDENTITY_MATCH_TABLE} AS
        SELECT c.blob_hash, c.ref_id, h.hook_event_id, h.blob_hash AS hook_blob_hash
        FROM {candidate_table} AS c
        JOIN {_LOCKED_HOOK_TABLE} AS h
          ON h.source_path IS c.source_path
         AND h.blob_hash = c.blob_hash
         AND polylogue_deterministic_raw_session_id(
               h.origin, c.source_path, 0, c.blob_hash, h.native_id
             ) = c.ref_id
        WHERE c.ref_type = 'raw_payload'
        UNION ALL
        SELECT c.blob_hash, c.ref_id, h.hook_event_id, h.blob_hash AS hook_blob_hash
        FROM {candidate_table} AS c
        JOIN (
            SELECT source_path
            FROM {candidate_table}
            WHERE ref_type = 'raw_payload'
            GROUP BY source_path
            HAVING COUNT(*) = 1
        ) AS unique_candidate_paths
          ON unique_candidate_paths.source_path IS c.source_path
        JOIN {_LOCKED_LEGACY_HOOK_COUNTS} AS unique_legacy_paths
          ON unique_legacy_paths.source_path IS c.source_path
         AND unique_legacy_paths.hook_count = 1
        JOIN {_LOCKED_HOOK_TABLE} AS h
          ON h.source_path IS c.source_path
         AND h.blob_hash IS NULL
         AND polylogue_deterministic_raw_session_id(
               h.origin, c.source_path, 0, c.blob_hash, h.native_id
             ) = c.ref_id
        WHERE c.ref_type = 'raw_payload'
        """
    )
    conn.execute(
        f"CREATE INDEX {_LOCKED_IDENTITY_MATCH_TABLE}_candidate ON {_LOCKED_IDENTITY_MATCH_TABLE}(blob_hash, ref_id)"
    )
    return _LOCKED_HOOK_TABLE


def _validate_locked_candidate_plan(
    conn: sqlite3.Connection,
    candidate_table: str,
    expected_count: int,
    *,
    locked_hook_table: str | None = None,
) -> None:
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
    content_mismatch_count = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM {candidate_table} AS c
            JOIN blob_refs AS b
              ON b.blob_hash = c.blob_hash
             AND b.ref_type = c.ref_type
             AND b.ref_id = c.ref_id
            WHERE NOT (
                b.source_path IS c.source_path
                AND b.size_bytes = c.size_bytes
                AND b.acquired_at_ms = c.acquired_at_ms
            )
            """
        ).fetchone()[0]
    )
    if content_mismatch_count:
        raise BlobRefLivenessReconciliationError(
            f"prepared candidate content changed under lock: {content_mismatch_count} row(s)"
        )
    referent_branches = [
        f"""
        SELECT c.blob_hash, c.ref_type, c.ref_id
        FROM {candidate_table} AS c
        JOIN {table} AS r ON r.{column} = c.ref_id
        WHERE c.ref_type = ?
        """
        for ref_type, (table, column) in {
            ref_type: (table, column) for ref_type, table, column in validated_blob_ref_liveness_joins()
        }.items()
    ]
    if referent_branches:
        live_referent_count = int(
            conn.execute(
                f"SELECT COUNT(*) FROM ({' UNION ALL '.join(referent_branches)})",
                tuple(ref_type for ref_type, _table, _column in validated_blob_ref_liveness_joins()),
            ).fetchone()[0]
        )
        if live_referent_count:
            raise BlobRefLivenessReconciliationError(
                f"prepared candidate referents became live under lock: {live_referent_count}"
            )
    if locked_hook_table is None:
        _stage_locked_hook_snapshot(conn, candidate_table)
    legacy_ambiguous_path_count = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT candidates.source_path
                FROM (
                    SELECT source_path, COUNT(*) AS candidate_count
                    FROM {candidate_table}
                    WHERE ref_type = 'raw_payload'
                    GROUP BY source_path
                ) AS candidates
                JOIN (
                    SELECT source_path, hook_count
                    FROM {_LOCKED_LEGACY_HOOK_COUNTS}
                ) AS hooks
                  ON hooks.source_path IS candidates.source_path
                WHERE candidates.candidate_count > 1 OR hooks.hook_count > 1
            )
            """
        ).fetchone()[0]
    )
    if legacy_ambiguous_path_count:
        raise BlobRefLivenessReconciliationError(
            "prepared candidates have ambiguous legacy hook evidence under lock: "
            f"{legacy_ambiguous_path_count} source path(s)"
        )
    duplicate_known_hash_count = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT i.blob_hash, i.ref_id
                FROM {_LOCKED_IDENTITY_MATCH_TABLE} AS i
                JOIN {candidate_table} AS c
                  ON c.blob_hash = i.blob_hash
                 AND c.ref_id = i.ref_id
                GROUP BY i.blob_hash, i.ref_id
                HAVING COUNT(DISTINCT i.hook_event_id) > 1
                   AND COUNT(DISTINCT CASE WHEN i.hook_blob_hash IS NOT NULL THEN i.hook_event_id END) > 1
            )
            """
        ).fetchone()[0]
    )
    if duplicate_known_hash_count:
        raise BlobRefLivenessReconciliationError(
            "prepared candidates have duplicate known-hash hook evidence under lock: "
            f"{duplicate_known_hash_count} candidate(s)"
        )
    duplicate_identity_count = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT i.blob_hash, i.ref_id
                FROM {_LOCKED_IDENTITY_MATCH_TABLE} AS i
                JOIN {candidate_table} AS c
                  ON c.blob_hash = i.blob_hash
                 AND c.ref_id = i.ref_id
                GROUP BY i.blob_hash, i.ref_id
                HAVING COUNT(DISTINCT i.hook_event_id) > 1
            )
            """
        ).fetchone()[0]
    )
    if duplicate_identity_count:
        raise BlobRefLivenessReconciliationError(
            "prepared candidates have duplicate hook identity evidence under lock: "
            f"{duplicate_identity_count} candidate(s)"
        )
    rekeyable = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT i.blob_hash, i.ref_id
                FROM {_LOCKED_IDENTITY_MATCH_TABLE} AS i
                JOIN {_LOCKED_MISSING_HOOK_TABLE} AS h
                  ON h.hook_event_id = i.hook_event_id
                JOIN {candidate_table} AS c
                  ON c.blob_hash = i.blob_hash
                 AND c.ref_id = i.ref_id
                WHERE c.ref_type = 'raw_payload'
                  AND i.hook_blob_hash IS NOT NULL
                GROUP BY i.blob_hash, i.ref_id
                HAVING COUNT(*) = 1
                UNION ALL
                SELECT i.blob_hash, i.ref_id
                FROM {_LOCKED_IDENTITY_MATCH_TABLE} AS i
                JOIN {_LOCKED_MISSING_HOOK_TABLE} AS h
                  ON h.hook_event_id = i.hook_event_id
                JOIN {candidate_table} AS c
                  ON c.blob_hash = i.blob_hash
                 AND c.ref_id = i.ref_id
                JOIN {_LOCKED_LEGACY_HOOK_COUNTS} AS hooks
                  ON hooks.source_path IS c.source_path
                WHERE c.ref_type = 'raw_payload'
                  AND i.hook_blob_hash IS NULL
                  AND (SELECT COUNT(*) FROM {candidate_table} AS c2
                       WHERE c2.ref_type = 'raw_payload' AND c2.source_path IS c.source_path) = 1
                  AND hooks.hook_count = 1
                GROUP BY i.blob_hash, i.ref_id
                HAVING COUNT(*) = 1
            )
            """
        ).fetchone()[0]
    )
    if rekeyable:
        raise BlobRefLivenessReconciliationError(
            f"prepared candidates became rekeyable hook payloads under lock: {rekeyable}"
        )


@_archive_owned
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

    archive_root = archive_root.resolve()
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
    backup_manifest = backup_manifest.resolve()
    receipt_path = receipt_path.resolve()
    if receipt_path.exists():
        outcome = _recover_prepared_receipt(source_db, receipt_path)
        raise BlobRefLivenessReconciliationError(
            f"recovered existing prepared receipt as {outcome}: {receipt_path}; choose a fresh receipt path before retrying"
        )
    if reason := _offline_apply_block_reason(archive_root):
        raise BlobRefLivenessReconciliationError(reason)
    try:
        assert_source_continuity_apply_allowed(archive_root)
    except DurableChangeTrainError as exc:
        raise BlobRefLivenessReconciliationError(str(exc)) from exc

    pre_conn = sqlite3.connect(source_db)
    staged_plan = None
    staged_data_version: int | None = None
    try:
        _checkpoint_source_db(pre_conn)
        validated_backup_digest = (
            hashlib.sha256(backup_manifest.read_bytes()).hexdigest() if backup_manifest.is_file() else None
        )
        validate_migration_backup_manifest(backup_manifest, ArchiveTier.SOURCE, connection=pre_conn)
        if (
            validated_backup_digest is not None
            and hashlib.sha256(backup_manifest.read_bytes()).hexdigest() != validated_backup_digest
        ):
            raise BlobRefLivenessReconciliationError("backup manifest changed during validation")
        pre_mutation_evidence = capture_durable_database_evidence(pre_conn, ArchiveTier.SOURCE)
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
            backup_manifest_sha256=validated_backup_digest,
        )
        pending_intent = write_source_continuity_pending_intent(
            archive_root,
            mutation_receipt=receipt_path,
            backup_manifest=backup_manifest,
            pre_mutation_evidence=pre_mutation_evidence,
            operation_id=candidate_digest,
            evidence_ref=f"proof:blob-ref-liveness:{candidate_digest}",
        )
        expected_count = classification.orphaned_count
        staged_data_version = _source_data_version(pre_conn)
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
    source_data_version: int | None = None
    locked_hook_table: str | None = None
    batch_table = "blob_ref_liveness_batch"
    try:
        while True:
            batch_count = _stage_candidate_batch(
                conn,
                staged_plan.candidate_table,
                batch_table,
                batch_size=BATCH_SIZE,
            )
            if batch_count == 0 and not first_batch:
                break
            conn.execute("BEGIN IMMEDIATE")
            try:
                if first_batch:
                    # This is the ownership-boundary attestation. The helper
                    # re-authenticates the receipt and validates the cached
                    # full backup inventory, rehashing only if its stat
                    # signature changed since the pre-lock validation.
                    validate_migration_backup_live_fingerprint(backup_manifest, ArchiveTier.SOURCE, connection=conn)
                    # Capture hook evidence against the complete staged
                    # population before any bounded delete can commit. The
                    # snapshot remains valid for later batches because apply
                    # is offline-only and the same connection owns the plan.
                    assert staged_data_version is not None
                    if _source_data_version(conn) != staged_data_version:
                        raise BlobRefLivenessReconciliationError(
                            "source.db changed after liveness plan staging; refusing stale plan"
                        )
                    locked_hook_table = _stage_locked_hook_snapshot(conn, staged_plan.candidate_table)
                    _validate_locked_candidate_plan(
                        conn,
                        staged_plan.candidate_table,
                        expected_count,
                        locked_hook_table=locked_hook_table,
                    )
                else:
                    assert locked_hook_table is not None
                    assert source_data_version is not None
                    if _source_data_version(conn) != source_data_version:
                        raise BlobRefLivenessReconciliationError(
                            "source.db changed after locked hook snapshot; refusing stale bounded plan"
                        )
                    _validate_locked_candidate_plan(
                        conn,
                        batch_table,
                        batch_count,
                        locked_hook_table=locked_hook_table,
                    )
                batch_deleted = (
                    _delete_candidate_batch(conn, staged_plan.candidate_table, batch_table) if batch_count else 0
                )
            except Exception:
                if conn.in_transaction:
                    conn.rollback()
                # Leave the prepared receipt without a terminal footer. A
                # retry can inspect exact key presence and record rollback,
                # commit, or partial-batch recovery without guessing.
                raise
            else:
                if first_batch:
                    # Read the external-writer marker while write ownership is
                    # still held. The local commit does not advance this
                    # connection's PRAGMA data_version, so later batches can
                    # detect writes in the gap after this commit.
                    source_data_version = _source_data_version(conn)
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

    post_classification: BlobRefLivenessClassification
    try:
        with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True) as verify_conn:
            quick_check = verify_conn.execute("PRAGMA quick_check").fetchone()
            if quick_check is None or str(quick_check[0]).lower() != "ok":
                raise BlobRefLivenessReconciliationError(f"source.db quick_check failed after commit: {quick_check!r}")
            post_classification = classify_blob_ref_liveness(verify_conn)
        if post_classification.orphaned_count != 0:
            raise BlobRefLivenessReconciliationError(
                f"liveness postcondition failed: {post_classification.orphaned_count} orphaned blob_refs row(s) remain"
            )
        if not post_classification.safe_to_apply:
            raise BlobRefLivenessReconciliationError(
                "liveness postcondition failed: source-tier ref types are no longer fully proven"
            )
    except Exception as exc:
        with suppress(OSError):
            _append_receipt_footer(receipt_path, phase="postcondition_failed", error=str(exc))
        raise

    try:
        _append_receipt_footer(
            receipt_path,
            phase="committed",
            deleted_count=deleted_count,
            post_orphaned_count=post_classification.orphaned_count,
        )
    except OSError as exc:
        raise BlobRefLivenessReconciliationError(
            f"source.db committed but could not finalize receipt {receipt_path}"
        ) from exc

    continuity_refresh_receipt: Path | None = None
    continuity_refresh_error: str | None = None
    continuity_refresh_pending = False
    try:
        continuity_refresh_receipt = refresh_released_source_train_continuity(
            archive_root,
            mutation_receipt=receipt_path,
            backup_manifest=backup_manifest,
            pre_mutation_evidence=pre_mutation_evidence,
            operation_id=candidate_digest,
            evidence_ref=f"proof:blob-ref-liveness:{candidate_digest}",
        )
        try:
            clear_source_continuity_pending_intent(pending_intent)
        except Exception as cleanup_exc:
            # The refresh is durable. Startup can consume the remaining
            # intent idempotently, so preserve the committed report and mark
            # only the cleanup residual as pending.
            continuity_refresh_error = f"pending intent cleanup failed: {cleanup_exc}"
            continuity_refresh_pending = True
    except DurableSourceTrainMissingError as exc:
        try:
            clear_source_continuity_pending_intent(pending_intent)
        except Exception as cleanup_exc:
            continuity_refresh_error = f"{exc}; pending intent cleanup failed: {cleanup_exc}"
            continuity_refresh_pending = True
        else:
            # A fresh archive has no released source train to refresh. The
            # committed source mutation is complete and there is no pending
            # continuity recovery obligation in this case.
            continuity_refresh_error = None
    except DurableSourceContinuitySemanticError as exc:
        # Semantic continuity rejection cannot become valid by retrying the
        # same committed source mutation. Preserve it durably for startup to
        # consume without hiding the fail-closed train mismatch.
        try:
            mark_source_continuity_pending_intent_terminal(pending_intent, error=exc)
        except Exception as terminalization_exc:
            continuity_refresh_error = f"{exc}; pending intent terminalization failed: {terminalization_exc}"
        else:
            continuity_refresh_error = str(exc)
        continuity_refresh_pending = True
    except Exception as exc:
        # The source deletion and its committed receipt are already durable.
        # Keep the report truthful while leaving the train fail-closed until a
        # separate continuity refresh succeeds. This boundary also normalizes
        # filesystem and SQLite failures after the irreversible commit.
        continuity_refresh_error = str(exc)
        continuity_refresh_pending = True

    assert staged_plan is not None
    return BlobRefLivenessReconciliationReport(
        source_db=str(source_db),
        dry_run=False,
        classification=staged_plan.classification,
        applied=True,
        deleted_count=expected_count,
        receipt_path=receipt_path,
        backup_manifest=backup_manifest,
        post_classification=post_classification,
        continuity_refresh_receipt=continuity_refresh_receipt,
        continuity_refresh_error=continuity_refresh_error,
        continuity_refresh_pending=continuity_refresh_pending,
    )


def census_blob_ref_liveness(archive_root: Path) -> OrphanedBlobRefCensus:
    """Return the privacy-safe source-tier blob-ref census without mutation."""
    from polylogue.storage.blob_gc import census_orphaned_blob_refs

    source_db = archive_root / "source.db"
    with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True) as conn:
        return census_orphaned_blob_refs(conn)


__all__ = [
    "BlobRefLivenessReconciliationError",
    "BlobRefLivenessReconciliationReport",
    "TOOL_VERSION",
    "reconcile_blob_ref_liveness",
]
