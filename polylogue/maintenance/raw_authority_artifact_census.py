"""Census and bounded artifact-only reclassification for quarantined raws."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
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
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import MigrationError, validate_migration_backup_manifest

DEFAULT_APPLY_LIMIT = 500


class RawAuthorityArtifactCensusError(RuntimeError):
    """Raised when the artifact-only actuator is refused."""


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


def _validate_source_backup(path: Path, conn: sqlite3.Connection) -> None:
    try:
        validate_migration_backup_manifest(path, ArchiveTier.SOURCE, connection=conn)
    except MigrationError as exc:
        raise RawAuthorityArtifactCensusError(str(exc)) from exc


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
    source_conn = _open_readonly(source_db)
    index_conn = _open_readonly(index_db)
    try:
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


def run_raw_authority_artifact_census(
    archive_root: Path,
    *,
    apply: bool = False,
    backup_manifest: Path | None = None,
    limit: int | None = None,
    after_raw_id: str | None = None,
    receipt_path: Path | None = None,
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
                    _validate_source_backup(backup_manifest, source_conn)
                    census = scan_quarantined_raw_authority(
                        source_conn,
                        index_conn,
                        blob_store=BlobStore(archive_root / "blob"),
                        limit=scan_limit,
                        after_raw_id=after_raw_id,
                    )
                    observations_written = write_artifact_observations(source_conn, census.artifact_observations())
                    observed_at_ms = int(time.time() * 1000)
                    receipt = census.receipt_payload(
                        mode="apply",
                        observations_written=observations_written,
                        observed_at_ms=observed_at_ms,
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
                    quick_check = source_conn.execute("PRAGMA quick_check").fetchone()
                    if quick_check is None or str(quick_check[0]).lower() != "ok":
                        raise RawAuthorityArtifactCensusError(f"source.db quick_check failed: {quick_check!r}")
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
