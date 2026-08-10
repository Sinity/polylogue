"""Census and bounded artifact-only reclassification for quarantined raws."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, cast

from polylogue.config import Config
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason
from polylogue.operations.durable_change_train import (
    ArchiveOwnershipError,
    acquire_durable_archive_ownership,
)
from polylogue.paths import render_root
from polylogue.storage.archive_identity import ArchiveLocation, assert_owns_archive_location, resolve_active_index_path
from polylogue.storage.artifacts.raw_authority_census import (
    MAX_APPLY_ROWS,
    RawAuthorityArtifactCensus,
    scan_quarantined_raw_authority,
    write_artifact_observations,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.index_generation import RebuildLease
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import MigrationError, validate_migration_backup_manifest

DEFAULT_APPLY_LIMIT = 500


class RawAuthorityArtifactCensusError(RuntimeError):
    """Raised when the artifact-only actuator is refused."""


@dataclass(frozen=True, slots=True)
class _CheckpointState:
    census_id: str
    candidate_count: int
    universe_sha256: str
    universe_complete: bool
    snapshot_max_raw_rowid: int
    materialized_after_rowid: int
    index_generation: str
    index_identity_sha256: str
    next_after_raw_id: str | None


@dataclass(frozen=True, slots=True)
class RawAuthorityArtifactCensusReport:
    census: RawAuthorityArtifactCensus
    mode: str
    observations_written: int
    receipt: dict[str, object]
    receipt_id: str | None = None
    receipt_path: Path | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "observations_written": self.observations_written,
            "receipt": self.receipt,
            "receipt_id": self.receipt_id,
            "receipt_path": str(self.receipt_path) if self.receipt_path is not None else None,
        }


def _offline_config(archive_root: Path) -> Config:
    return Config(archive_root=archive_root, render_root=render_root(), sources=[])


def _checkpoint_source_tier(conn: sqlite3.Connection) -> None:
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    except sqlite3.Error as exc:
        raise RawAuthorityArtifactCensusError("could not checkpoint source.db before backup validation") from exc
    if row is None or int(row[0]) != 0:
        raise RawAuthorityArtifactCensusError("source.db WAL checkpoint was blocked; retry when the tier is idle")


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve(strict=False)),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _json_value(value: object) -> object:
    return {"bytes": bytes(value).hex()} if isinstance(value, (bytes, bytearray, memoryview)) else value


def _validate_source_backup(path: Path, conn: sqlite3.Connection) -> dict[str, object]:
    try:
        verification_receipt = validate_migration_backup_manifest(path, ArchiveTier.SOURCE, connection=conn)
    except MigrationError as exc:
        raise RawAuthorityArtifactCensusError(str(exc)) from exc
    receipt_path = Path(verification_receipt)
    return {
        "manifest": {"path": str(path.resolve(strict=False)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "verification_receipt": {
            "path": str(receipt_path.resolve(strict=False)),
            "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        },
    }


def _write_immutable_receipt(path: Path, payload: dict[str, object]) -> None:
    """Create one receipt file and refuse to replace an existing receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise RawAuthorityArtifactCensusError(f"receipt already exists and is immutable: {path}") from exc
    try:
        path.chmod(0o444)
    except OSError as exc:
        raise RawAuthorityArtifactCensusError(f"could not make receipt immutable: {path}") from exc


def _validate_receipt_destination(path: Path) -> None:
    """Reject an unusable receipt path before an apply transaction starts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RawAuthorityArtifactCensusError(f"receipt already exists and is immutable: {path}")
    probe: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.probe-", dir=path.parent, delete=False) as handle:
            probe = Path(handle.name)
        probe.chmod(0o444)
    except OSError as exc:
        raise RawAuthorityArtifactCensusError(f"receipt path is not writable: {path}") from exc
    finally:
        if probe is not None:
            probe.unlink(missing_ok=True)


def _open_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30.0)


def _index_authority_identity(location: ArchiveLocation) -> tuple[str, str]:
    return location.active_generation, _canonical_digest({"active_generation": location.active_generation})


def _create_checkpoint(conn: sqlite3.Connection, *, location: ArchiveLocation, now_ms: int) -> _CheckpointState:
    census_id = f"raw-authority-artifact-census:{uuid.uuid4().hex}"
    snapshot_max_raw_rowid = int(conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM raw_sessions").fetchone()[0])
    index_generation, index_identity_sha256 = _index_authority_identity(location)
    conn.execute(
        """
        INSERT INTO raw_authority_artifact_census_checkpoints (
            census_id, universe_sha256, candidate_count, universe_complete,
            snapshot_max_raw_rowid, materialized_after_rowid,
            index_generation, index_identity_sha256, next_after_raw_id, created_at_ms
        ) VALUES (?, ?, 0, 0, ?, 0, ?, ?, NULL, ?)
        """,
        (census_id, _canonical_digest([]), snapshot_max_raw_rowid, index_generation, index_identity_sha256, now_ms),
    )
    return _checkpoint_state(conn, census_id)


def _extend_universe_digest(digest: str, raw_id: str) -> str:
    return hashlib.sha256(bytes.fromhex(digest) + b"\0" + raw_id.encode("utf-8")).hexdigest()


def _materialize_checkpoint_candidates(
    conn: sqlite3.Connection, checkpoint: _CheckpointState, *, limit: int
) -> _CheckpointState:
    if checkpoint.universe_complete:
        return checkpoint
    rows = conn.execute(
        """
        SELECT rowid, raw_id FROM raw_sessions
        WHERE revision_authority = 'quarantined'
          AND parse_error IS NULL
          AND rowid > ?
          AND rowid <= ?
        ORDER BY rowid
        LIMIT ?
        """,
        (checkpoint.materialized_after_rowid, checkpoint.snapshot_max_raw_rowid, limit + 1),
    ).fetchall()
    if not rows:
        conn.execute(
            "UPDATE raw_authority_artifact_census_checkpoints SET universe_complete = 1 WHERE census_id = ?",
            (checkpoint.census_id,),
        )
        return _checkpoint_state(conn, checkpoint.census_id)
    digest = checkpoint.universe_sha256
    members: list[tuple[str, int, str]] = []
    for offset, (_, raw_id) in enumerate(rows):
        raw_id_text = str(raw_id)
        digest = _extend_universe_digest(digest, raw_id_text)
        members.append((checkpoint.census_id, checkpoint.candidate_count + offset, raw_id_text))
    conn.executemany(
        """
        INSERT INTO raw_authority_artifact_census_checkpoint_members (census_id, ordinal, raw_id)
        VALUES (?, ?, ?)
        """,
        members,
    )
    conn.execute(
        """
        UPDATE raw_authority_artifact_census_checkpoints
        SET universe_sha256 = ?, candidate_count = ?, materialized_after_rowid = ?, universe_complete = ?
        WHERE census_id = ?
        """,
        (
            digest,
            checkpoint.candidate_count + len(members),
            int(rows[-1][0]),
            int(len(rows) <= limit),
            checkpoint.census_id,
        ),
    )
    return _checkpoint_state(conn, checkpoint.census_id)


def _checkpoint_state(conn: sqlite3.Connection, census_id: str) -> _CheckpointState:
    row = conn.execute(
        """
        SELECT candidate_count, universe_sha256, universe_complete, snapshot_max_raw_rowid,
               materialized_after_rowid, index_generation, index_identity_sha256,
               next_after_raw_id, completed_at_ms
        FROM raw_authority_artifact_census_checkpoints
        WHERE census_id = ?
        """,
        (census_id,),
    ).fetchone()
    if row is None:
        raise RawAuthorityArtifactCensusError(f"unknown durable census checkpoint: {census_id}")
    (
        candidate_count,
        universe_sha256,
        universe_complete,
        snapshot_max_raw_rowid,
        materialized_after_rowid,
        index_generation,
        index_identity_sha256,
        next_after_raw_id,
        completed_at_ms,
    ) = row
    if completed_at_ms is not None:
        raise RawAuthorityArtifactCensusError("durable census checkpoint is already complete")
    return _CheckpointState(
        census_id=census_id,
        candidate_count=int(candidate_count),
        universe_sha256=str(universe_sha256),
        universe_complete=bool(universe_complete),
        snapshot_max_raw_rowid=int(snapshot_max_raw_rowid),
        materialized_after_rowid=int(materialized_after_rowid),
        index_generation=str(index_generation),
        index_identity_sha256=str(index_identity_sha256),
        next_after_raw_id=str(next_after_raw_id) if next_after_raw_id is not None else None,
    )


def _checkpoint_page_raw_ids(
    conn: sqlite3.Connection, *, checkpoint: _CheckpointState, after_raw_id: str | None, limit: int
) -> tuple[tuple[str, ...], bool]:
    if after_raw_id is None:
        after_ordinal = -1
    else:
        cursor = conn.execute(
            "SELECT ordinal FROM raw_authority_artifact_census_checkpoint_members WHERE census_id = ? AND raw_id = ?",
            (checkpoint.census_id, after_raw_id),
        ).fetchone()
        if cursor is None:
            raise RawAuthorityArtifactCensusError("durable continuation cursor is absent from the candidate universe")
        after_ordinal = int(cursor[0])
    rows = conn.execute(
        """
        SELECT raw_id FROM raw_authority_artifact_census_checkpoint_members
        WHERE census_id = ?
          AND ordinal > ?
        ORDER BY ordinal
        LIMIT ?
        """,
        (checkpoint.census_id, after_ordinal, limit + 1),
    ).fetchall()
    selected = tuple(str(row[0]) for row in rows[:limit])
    return selected, len(rows) > limit or not checkpoint.universe_complete


def _page_inventory_digest(conn: sqlite3.Connection, raw_ids: tuple[str, ...]) -> str:
    if not raw_ids:
        return _canonical_digest({"raw_sessions": [], "raw_artifacts": []})
    records: dict[str, list[list[object]]] = {"raw_sessions": [], "raw_artifacts": []}
    for start in range(0, len(raw_ids), 500):
        selected = raw_ids[start : start + 500]
        marks = ",".join("?" for _ in selected)
        for table, columns in (
            ("raw_sessions", "raw_id, revision_authority, parse_error, blob_hash, blob_size"),
            ("raw_artifacts", "raw_id, artifact_kind, support_status, parse_as_session, schema_eligible"),
        ):
            rows = conn.execute(
                f"SELECT {columns} FROM {table} WHERE raw_id IN ({marks}) ORDER BY raw_id",
                selected,
            ).fetchall()
            records[table].extend([[_json_value(value) for value in row] for row in rows])
    return _canonical_digest(records)


def _snapshot_census(
    archive_root: Path,
    location: ArchiveLocation,
    *,
    limit: int | None,
    after_raw_id: str | None,
) -> RawAuthorityArtifactCensus:
    source_conn = _open_readonly(archive_root / "source.db")
    index_conn = _open_readonly(location.active_index_path)
    try:
        source_conn.execute("BEGIN")
        index_conn.execute("BEGIN")
        source_conn.execute("SELECT 1 FROM raw_sessions LIMIT 1").fetchone()
        index_conn.execute("SELECT 1 FROM sessions LIMIT 1").fetchone()
        return scan_quarantined_raw_authority(
            source_conn,
            index_conn,
            blob_store=BlobStore(archive_root / "blob"),
            limit=limit,
            after_raw_id=after_raw_id,
        )
    finally:
        source_conn.close()
        index_conn.close()


def _scan(
    archive_root: Path,
    *,
    limit: int | None,
    after_raw_id: str | None,
) -> RawAuthorityArtifactCensus:
    source_db = archive_root / "source.db"
    index_db = resolve_active_index_path(archive_root)
    if not source_db.is_file():
        raise FileNotFoundError(f"no source.db at {source_db}")
    if not index_db.is_file():
        raise FileNotFoundError(f"no index.db at {index_db}")
    del index_db
    try:
        ownership = acquire_durable_archive_ownership(
            archive_root,
            owner_id="maintenance:raw-authority-artifact-census:dry-run",
        )
    except ArchiveOwnershipError as exc:
        raise RawAuthorityArtifactCensusError(str(exc)) from exc
    with ownership, RebuildLease(archive_root):
        current_location = ArchiveLocation.resolve(archive_root)
        try:
            assert_owns_archive_location(ownership, current_location)
        except ArchiveOwnershipError as exc:
            raise RawAuthorityArtifactCensusError(str(exc)) from exc
        return _snapshot_census(archive_root, current_location, limit=limit, after_raw_id=after_raw_id)


def run_raw_authority_artifact_census(
    archive_root: Path,
    *,
    apply: bool = False,
    backup_manifest: Path | None = None,
    limit: int | None = None,
    after_raw_id: str | None = None,
    receipt_path: Path | None = None,
    census_id: str | None = None,
) -> RawAuthorityArtifactCensusReport:
    """Run one census, optionally persisting only artifact observations.

    Dry-run is a full read-only census and requires an immutable receipt. Apply
    is bounded to 500 rows by default and logically upserts only ``raw_artifacts``.
    It also performs the established ``source.db`` WAL checkpoint required by
    backup validation, which can change SQLite's physical main/WAL layout. It
    never changes ``raw_sessions`` rows, revision authority, index rows, or
    blob storage. A verified source-tier backup manifest is required for apply.
    """
    receipt: dict[str, object] = {}
    receipt_id: str | None = None
    if limit is not None and limit < 0:
        raise RawAuthorityArtifactCensusError("--limit must be non-negative")
    scan_limit: int | None = None
    if apply:
        scan_limit = DEFAULT_APPLY_LIMIT if limit is None else limit
        if scan_limit <= 0:
            raise RawAuthorityArtifactCensusError("--limit must be positive for --apply")
        if scan_limit > MAX_APPLY_ROWS:
            raise RawAuthorityArtifactCensusError(f"--limit cannot exceed {MAX_APPLY_ROWS} for --apply")
        if backup_manifest is None:
            raise RawAuthorityArtifactCensusError("--apply requires --backup-manifest")
        assert backup_manifest is not None
        assert scan_limit is not None
        if receipt_path is not None:
            raise RawAuthorityArtifactCensusError("--receipt is only supported for the read-only dry-run census")
        if census_id is None and after_raw_id is not None:
            raise RawAuthorityArtifactCensusError(
                "--after-raw-id requires --census-id from the preceding durable receipt"
            )
    elif receipt_path is None:
        raise RawAuthorityArtifactCensusError("read-only dry-run requires --receipt for an immutable census receipt")
    if receipt_path is not None:
        _validate_receipt_destination(receipt_path)
    if apply:
        assert backup_manifest is not None
        assert scan_limit is not None
        if reason := offline_maintenance_block_reason(_offline_config(archive_root), active=True, dry_run=False):
            raise RawAuthorityArtifactCensusError(reason)
        try:
            ownership = acquire_durable_archive_ownership(
                archive_root,
                owner_id="maintenance:raw-authority-artifact-census",
            )
        except ArchiveOwnershipError as exc:
            raise RawAuthorityArtifactCensusError(str(exc)) from exc
        with ownership:
            try:
                location = ArchiveLocation.resolve(archive_root)
                assert_owns_archive_location(ownership, location)
            except ArchiveOwnershipError as exc:
                raise RawAuthorityArtifactCensusError(str(exc)) from exc
            source_db = archive_root / "source.db"
            index_db = location.active_index_path
            if not source_db.is_file():
                raise FileNotFoundError(f"no source.db at {source_db}")
            if not index_db.is_file():
                raise FileNotFoundError(f"no index.db at {index_db}")
            source_conn = sqlite3.connect(source_db, timeout=30.0)
            index_conn = _open_readonly(index_db)
            try:
                _validate_source_backup(backup_manifest, source_conn)
                _checkpoint_source_tier(source_conn)
                source_conn.execute("BEGIN IMMEDIATE")
                try:
                    backup_evidence = _validate_source_backup(backup_manifest, source_conn)
                    observed_at_ms = int(time.time() * 1000)
                    if census_id is None:
                        checkpoint = _create_checkpoint(source_conn, location=location, now_ms=observed_at_ms)
                        expected_after_raw_id = None
                    else:
                        checkpoint = _checkpoint_state(source_conn, census_id)
                        expected_after_raw_id = checkpoint.next_after_raw_id
                        if after_raw_id != expected_after_raw_id:
                            raise RawAuthorityArtifactCensusError(
                                "continuation cursor must equal the durable checkpoint next_after_raw_id"
                            )
                        index_generation, index_identity_sha256 = _index_authority_identity(location)
                        if (
                            index_generation != checkpoint.index_generation
                            or index_identity_sha256 != checkpoint.index_identity_sha256
                        ):
                            raise RawAuthorityArtifactCensusError(
                                "active index authority changed since the preceding durable census page"
                            )
                    checkpoint = _materialize_checkpoint_candidates(source_conn, checkpoint, limit=scan_limit)
                    page_raw_ids, has_more = _checkpoint_page_raw_ids(
                        source_conn,
                        checkpoint=checkpoint,
                        after_raw_id=expected_after_raw_id,
                        limit=scan_limit,
                    )
                    index_conn.execute("BEGIN")
                    census = scan_quarantined_raw_authority(
                        source_conn,
                        index_conn,
                        blob_store=BlobStore(archive_root / "blob"),
                        raw_ids=page_raw_ids,
                        total_quarantined_count=checkpoint.candidate_count,
                        has_more=has_more,
                        after_raw_id=expected_after_raw_id,
                    )
                    census = RawAuthorityArtifactCensus(
                        total_quarantined_count=census.total_quarantined_count,
                        entries=census.entries,
                        after_raw_id=census.after_raw_id,
                        has_more=has_more,
                        page_next_after_raw_id=page_raw_ids[-1] if has_more and page_raw_ids else None,
                    )
                    before_inventory = _page_inventory_digest(source_conn, page_raw_ids)
                    observations_written = write_artifact_observations(source_conn, census.artifact_observations())
                    after_inventory = _page_inventory_digest(source_conn, page_raw_ids)
                    evidence = {
                        "backup": backup_evidence,
                        "archive": {
                            "configured_root": str(archive_root.resolve(strict=False)),
                            "active_index_path": str(location.active_index_path.resolve(strict=False)),
                            "active_generation": location.active_generation,
                            "source_tier": _file_identity(source_db),
                            "active_index": _file_identity(index_db),
                        },
                        "checkpoint": {
                            "census_id": checkpoint.census_id,
                            "universe_sha256": checkpoint.universe_sha256,
                            "candidate_count": checkpoint.candidate_count,
                            "universe_complete": checkpoint.universe_complete,
                            "snapshot_max_raw_rowid": checkpoint.snapshot_max_raw_rowid,
                            "index_generation": checkpoint.index_generation,
                            "index_identity_sha256": checkpoint.index_identity_sha256,
                            "page_raw_ids_sha256": _canonical_digest(list(page_raw_ids)),
                        },
                        "inventory": {"before_sha256": before_inventory, "after_sha256": after_inventory},
                        "command": {"operation": "raw-authority-artifact-census", "limit": scan_limit},
                    }
                    receipt = census.receipt_payload(
                        mode="apply",
                        observations_written=observations_written,
                        observed_at_ms=observed_at_ms,
                        evidence=evidence,
                    )
                    receipt_sha256 = str(receipt["receipt_sha256"])
                    receipt_id = f"raw-authority-artifact-census:{receipt_sha256}"
                    source_conn.execute(
                        """
                        INSERT INTO raw_authority_artifact_census_receipts (
                            receipt_id, receipt_sha256, receipt_json, backup_manifest_path, applied_at_ms, tool_version
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            receipt_id,
                            receipt_sha256,
                            json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                            str(backup_manifest),
                            observed_at_ms,
                            str(receipt["tool_version"]),
                        ),
                    )
                    source_conn.execute(
                        """
                        UPDATE raw_authority_artifact_census_checkpoints
                        SET next_after_raw_id = ?, last_receipt_id = ?, completed_at_ms = ?
                        WHERE census_id = ?
                        """,
                        (
                            census.next_after_raw_id,
                            receipt_id,
                            observed_at_ms if census.next_after_raw_id is None else None,
                            checkpoint.census_id,
                        ),
                    )
                except Exception:
                    if source_conn.in_transaction:
                        source_conn.rollback()
                    raise
                else:
                    source_conn.commit()
            finally:
                source_conn.close()
                index_conn.close()
        mode = "apply"
    else:
        census = _scan(archive_root, limit=limit, after_raw_id=after_raw_id)
        observations_written = 0
        mode = "dry_run"
    if not apply:
        receipt = census.receipt_payload(
            mode=mode,
            observations_written=observations_written,
            observed_at_ms=int(time.time() * 1000),
        )
    if not apply and receipt_path is not None:
        _write_immutable_receipt(receipt_path, receipt)
    return RawAuthorityArtifactCensusReport(
        census=census,
        mode=mode,
        observations_written=observations_written,
        receipt=receipt,
        receipt_id=receipt_id,
        receipt_path=receipt_path,
    )


def render_report(report: RawAuthorityArtifactCensusReport, *, stdout: TextIO | None = None) -> None:
    """Render a compact human report without exposing source paths/content."""
    stream = stdout
    print(f"mode: {report.mode}", file=stream)
    print(
        f"quarantined rows: {report.census.scanned_count}/{report.census.total_quarantined_count}"
        f"  truncated={report.census.truncated}",
        file=stream,
    )
    counts = cast(dict[str, int], report.receipt["counts"])
    bytes_by_bucket = cast(dict[str, int], report.receipt["bytes_by_bucket"])
    for bucket, count in counts.items():
        size = bytes_by_bucket[bucket]
        print(f"{bucket}: {count} ({size} bytes)", file=stream)
    print(f"artifact observations written: {report.observations_written}", file=stream)
    if report.receipt_path is not None:
        print(f"immutable receipt: {report.receipt_path}", file=stream)
    if report.receipt_id is not None:
        print(f"durable receipt: {report.receipt_id}", file=stream)


__all__ = [
    "DEFAULT_APPLY_LIMIT",
    "RawAuthorityArtifactCensusError",
    "RawAuthorityArtifactCensusReport",
    "render_report",
    "run_raw_authority_artifact_census",
]
