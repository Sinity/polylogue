"""Backup-gated reconciliation of one live cursor-authority violation.

The command is intentionally narrow.  It proves one source path from the
canonical raw-frontier projection, then runs that path through
``LiveBatchProcessor.ingest_files``.  It never edits a cursor or accepted head
itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import tempfile
import time
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path

from polylogue.api import Polylogue
from polylogue.config import Config
from polylogue.core.enums import IngestOutcome
from polylogue.operations.durable_change_train import acquire_durable_archive_ownership
from polylogue.pipeline.ingest_outcomes import IngestAttemptDisposition
from polylogue.sources.live.batch import (
    LiveBatchProcessor,
    cursor_authority_path_digest,
    scoped_cursor_authority_authorization,
)
from polylogue.sources.live.batch_support import sha256_range_from_path
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.metrics import LiveBatchMetrics
from polylogue.sources.live.watcher import WatchSource
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.backup_attestation import BackupAttestationError, verify_verification_receipt
from polylogue.storage.raw_retention import RawFrontierIntegrityProjection, raw_frontier_integrity_projection

PLAN_FORMAT = "polylogue.cursor-authority-reconciliation-plan.v1"
RECEIPT_FORMAT = "polylogue.cursor-authority-reconciliation-receipt.v1"
ARCHIVE_ROOT = Path("/realm/db/polylogue")
_REQUIRED_TIERS = ("source", "index", "ops", "audit")


class CursorAuthorityReconciliationError(RuntimeError):
    """A reconciliation precondition or postcondition was not proven."""


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CursorAuthorityReconciliationError(f"backup blob inventory is unreadable: {path}") from exc
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> tuple[int, str]:
    """Return size and digest from one descriptor observation."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CursorAuthorityReconciliationError(f"required archive file is unreadable: {path}") from exc
    return size, digest.hexdigest()


def _identity_digest(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _path_identity(path: Path) -> dict[str, str]:
    return {
        "path_digest": cursor_authority_path_digest(path),
        "basename": path.name,
    }


def _stat_observation(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _archive_root() -> Path:
    """Return the fixed archive root for this command.

    This deliberately does not call ``polylogue.paths.archive_root`` or read
    any ambient archive-root environment variable.
    """

    return ARCHIVE_ROOT


def _read_private_source_path(path_file: Path) -> Path:
    descriptor: int | None = None
    try:
        descriptor = os.open(path_file, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise CursorAuthorityReconciliationError(f"source path file is unreadable: {path_file}") from exc
    try:
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise CursorAuthorityReconciliationError("source path file must be a regular single-linked file")
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise CursorAuthorityReconciliationError("source path file must be owned by the operator and mode 0600")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    except OSError as exc:
        raise CursorAuthorityReconciliationError(f"source path file is unreadable: {path_file}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        lines = b"".join(chunks).decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise CursorAuthorityReconciliationError("source path file must be UTF-8 text") from exc
    if len(lines) != 1 or not lines[0].strip():
        raise CursorAuthorityReconciliationError("source path file must contain exactly one non-empty path")
    candidate = Path(lines[0])
    if not candidate.is_absolute():
        raise CursorAuthorityReconciliationError("selected source path must be absolute")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise CursorAuthorityReconciliationError("selected source path does not resolve") from exc
    if not resolved.is_file():
        raise CursorAuthorityReconciliationError("selected source path must be a regular file")
    return resolved


def _sqlite_snapshot(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise CursorAuthorityReconciliationError(f"required archive tier is missing: {path}")
    try:
        with tempfile.TemporaryDirectory(prefix="polylogue-sqlite-snapshot-") as temporary_dir:
            snapshot_path = Path(temporary_dir) / "snapshot.db"
            with (
                closing(sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)) as source_conn,
                closing(sqlite3.connect(snapshot_path)) as snapshot_conn,
            ):
                source_conn.execute("PRAGMA query_only = ON")
                source_conn.backup(snapshot_conn)
                snapshot_conn.commit()
            size_bytes, sha256 = _file_fingerprint(snapshot_path)
            with closing(sqlite3.connect(f"file:{snapshot_path.resolve()}?mode=ro", uri=True)) as conn:
                conn.execute("PRAGMA query_only = ON")
                user_version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
                schema_version = int(conn.execute("PRAGMA schema_version").fetchone()[0] or 0)
                schema_rows = conn.execute(
                    "SELECT type, name, tbl_name, sql FROM sqlite_schema "
                    "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name, tbl_name"
                ).fetchall()
                quick_check = tuple(str(row[0]) for row in conn.execute("PRAGMA quick_check"))
    except (OSError, sqlite3.Error) as exc:
        raise CursorAuthorityReconciliationError(f"could not read SQLite tier {path}: {exc}") from exc
    schema_digest = _canonical_digest(
        [[str(value) if value is not None else None for value in row] for row in schema_rows]
    )
    return {
        "size_bytes": size_bytes,
        "sha256": sha256,
        "user_version": user_version,
        "schema_version": schema_version,
        "schema_sha256": schema_digest,
        "quick_check": list(quick_check),
    }


def _tier_snapshots(root: Path) -> dict[str, dict[str, object]]:
    location = ArchiveLocation.resolve(root)
    snapshots: dict[str, dict[str, object]] = {}
    for tier in _REQUIRED_TIERS:
        tier_path = location.active_index_path if tier == "index" else root / f"{tier}.db"
        snapshots[tier] = _sqlite_snapshot(tier_path)
    return snapshots


def _active_index_binding(root: Path) -> dict[str, object]:
    location = ArchiveLocation.resolve(root)
    return {
        "path": _path_identity(location.active_index_path),
        "generation_digest": _identity_digest(location.active_generation),
    }


def _code_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[2],
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _deployed_package_sha() -> str:
    package_root = Path(__file__).parents[1]
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        if any(part == "__pycache__" for part in path.parts):
            continue
        digest.update(str(path.relative_to(package_root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _projection_for(root: Path) -> RawFrontierIntegrityProjection:
    from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot

    return raw_frontier_integrity_projection(
        root,
        raw_materialization_readiness_snapshot(root),
        sample_limit=100,
    )


def _private_projection(projection: RawFrontierIntegrityProjection) -> dict[str, object]:
    def redact(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: (
                    cursor_authority_path_digest(Path(item))
                    if key == "source_path" and isinstance(item, str)
                    else _identity_digest(item)
                    if key in {"source_path", "logical_source_key", "accepted_raw_id", "raw_id", "session_id"}
                    and isinstance(item, str)
                    else None
                    if key in {"source_path", "logical_source_key", "accepted_raw_id", "raw_id", "session_id"}
                    and item is not None
                    else redact(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    redacted = redact(projection.to_dict())
    if not isinstance(redacted, dict):
        raise CursorAuthorityReconciliationError("raw-frontier projection did not produce a mapping")
    return redacted


def _cursor_rows(root: Path) -> list[tuple[str, int]]:
    with closing(sqlite3.connect(f"file:{(root / 'ops.db').resolve()}?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            "SELECT source_path, byte_offset FROM ingest_cursor "
            "WHERE COALESCE(excluded, 0) = 0 AND byte_offset IS NOT NULL"
        ).fetchall()
    result: list[tuple[str, int]] = []
    for row in rows:
        if not isinstance(row[0], str) or not row[0]:
            raise CursorAuthorityReconciliationError("ingest cursor has an invalid source path")
        result.append((row[0], _required_nonnegative_int(row[1], "ingest cursor byte_offset")))
    return result


def _find_path_by_digest(root: Path, digest: str) -> Path:
    matches = [
        Path(path).resolve()
        for path, _offset in _cursor_rows(root)
        if cursor_authority_path_digest(Path(path)) == digest
    ]
    if len(matches) != 1:
        raise CursorAuthorityReconciliationError("plan path digest does not identify exactly one current cursor path")
    return matches[0]


def _head_details(root: Path, source_path: Path, projection: RawFrontierIntegrityProjection) -> dict[str, object]:
    if projection.cursor_ahead_count != 1 or len(projection.cursor_ahead_samples) != 1:
        raise CursorAuthorityReconciliationError("reconciliation requires exactly one true cursor-ahead row")
    sample = projection.cursor_ahead_samples[0]
    if Path(sample.source_path).resolve() != source_path.resolve():
        raise CursorAuthorityReconciliationError("selected source path is not the sole cursor-ahead path")
    index_path = ArchiveLocation.resolve(root).active_index_path
    with closing(sqlite3.connect(f"file:{index_path.resolve()}?mode=ro", uri=True)) as conn:
        head = conn.execute(
            "SELECT logical_source_key, accepted_raw_id, accepted_source_revision, "
            "accepted_content_hash, accepted_frontier_kind, accepted_frontier, "
            "acquisition_generation, append_end_offset "
            "FROM raw_revision_heads WHERE logical_source_key = ?",
            (sample.logical_source_key,),
        ).fetchone()
    if head is None:
        raise CursorAuthorityReconciliationError("accepted head is missing for the selected path")
    with closing(sqlite3.connect(f"file:{(root / 'source.db').resolve()}?mode=ro", uri=True)) as conn:
        raw = conn.execute(
            "SELECT raw_id, source_path, blob_hash, blob_size, revision_authority FROM raw_sessions WHERE raw_id = ?",
            (str(head[1]),),
        ).fetchone()
    if raw is None or Path(str(raw[1])).resolve() != source_path.resolve():
        raise CursorAuthorityReconciliationError("accepted head does not match the recorded source path")
    if str(head[4]) != "byte" or str(raw[4]) != "byte_proven":
        raise CursorAuthorityReconciliationError("accepted head is not byte-authoritative")
    logical_source_key = head[0]
    if not isinstance(logical_source_key, str) or not logical_source_key:
        raise CursorAuthorityReconciliationError("accepted head has an invalid logical source key")
    frontier = _required_nonnegative_int(head[5], "accepted head frontier")
    blob_hash = bytes(raw[2]).hex() if isinstance(raw[2], bytes) else str(raw[2]).lower()
    blob_size = _required_nonnegative_int(raw[3], "accepted raw blob size")
    try:
        bytes.fromhex(blob_hash)
    except ValueError as exc:
        raise CursorAuthorityReconciliationError("accepted raw has an invalid blob hash") from exc
    if blob_size != frontier or len(blob_hash) != 64:
        raise CursorAuthorityReconciliationError("accepted raw does not bind a complete byte frontier")
    cursor_matches = [offset for path, offset in _cursor_rows(root) if Path(path).resolve() == source_path.resolve()]
    if len(cursor_matches) != 1:
        raise CursorAuthorityReconciliationError("selected source path has no unique current cursor row")
    cursor_offset = cursor_matches[0]
    before_stat = _stat_observation(source_path)
    prefix_digest, bytes_read = sha256_range_from_path(source_path, start_offset=0, end_offset=frontier)
    after_stat = _stat_observation(source_path)
    if before_stat != after_stat:
        raise CursorAuthorityReconciliationError("source mutated during accepted-frontier hashing")
    if prefix_digest != blob_hash:
        raise CursorAuthorityReconciliationError("source prefix does not match the accepted raw blob hash")
    return {
        "logical_source_key": cursor_authority_path_digest(Path(logical_source_key)),
        "cursor_byte_offset": cursor_offset,
        "accepted_frontier": frontier,
        "accepted_raw_id_digest": _canonical_digest(str(head[1])),
        "accepted_blob_hash_digest": _canonical_digest(blob_hash),
        "source_prefix_digest": prefix_digest,
        "source_prefix_bytes": bytes_read,
        "source_stat": list(after_stat),
    }


def _require_healthy_projection_siblings(projection: RawFrontierIntegrityProjection) -> None:
    if not projection.available:
        raise CursorAuthorityReconciliationError("raw-frontier projection is unavailable")
    if projection.broken_head_status != "healthy" or projection.missing_source_raw_status != "healthy":
        raise CursorAuthorityReconciliationError("raw-frontier sibling projections are not healthy")


def _build_plan(root: Path, source_path: Path) -> dict[str, object]:
    tiers = _tier_snapshots(root)
    projection = _projection_for(root)
    path_digest = cursor_authority_path_digest(source_path)
    _require_healthy_projection_siblings(projection)
    if projection.cursor_ahead_count == 0:
        raise CursorAuthorityReconciliationError("cursor authority is incomparable or has no selected violation")
    if projection.cursor_ahead_count != 1:
        raise CursorAuthorityReconciliationError("refusing to guess among multiple cursor-ahead rows")
    if projection.broken_head_count or projection.missing_source_raw_count:
        raise CursorAuthorityReconciliationError(
            "global raw-frontier violation set is not exactly one cursor-ahead row"
        )
    details = _head_details(root, source_path, projection)
    plan: dict[str, object] = {
        "format": PLAN_FORMAT,
        "archive_identity": _path_identity(root),
        "active_index": _active_index_binding(root),
        "code_sha": _code_sha(),
        "deployed_package_sha": _deployed_package_sha(),
        "tier_fingerprints": tiers,
        "source_schema_versions": {tier: tiers[tier]["user_version"] for tier in _REQUIRED_TIERS},
        "selected_path_digest": path_digest,
        "observed_at_ms": int(time.time() * 1000),
        "status": "planned",
        "cursor_byte_offset": details["cursor_byte_offset"],
        "accepted_frontier": details["accepted_frontier"],
        "accepted_raw_id_digest": details["accepted_raw_id_digest"],
        "accepted_blob_hash_digest": details["accepted_blob_hash_digest"],
        "source_prefix_digest": details["source_prefix_digest"],
        "before_projection": _private_projection(projection),
    }
    plan["plan_digest"] = _canonical_digest(plan)
    return plan


def _load_plan(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CursorAuthorityReconciliationError(f"invalid reconciliation plan: {path}") from exc
    if not isinstance(payload, dict) or payload.get("format") != PLAN_FORMAT:
        raise CursorAuthorityReconciliationError("unsupported reconciliation plan format")
    digest = payload.get("plan_digest")
    unsigned = dict(payload)
    unsigned.pop("plan_digest", None)
    if not isinstance(digest, str) or _canonical_digest(unsigned) != digest:
        raise CursorAuthorityReconciliationError("reconciliation plan digest mismatch")
    return payload


def _backup_root(manifest_path: Path) -> Path:
    if manifest_path.is_dir():
        root = manifest_path
    elif manifest_path.is_file() and manifest_path.name == "manifest.json":
        root = manifest_path.parent
    else:
        raise CursorAuthorityReconciliationError(
            "backup manifest must be manifest.json or a verified full-evidence backup directory"
        )
    if not root.is_dir() or not (root / "manifest.json").is_file():
        raise CursorAuthorityReconciliationError("backup manifest must be a verified full-evidence backup directory")
    return root


def _validated_blob_inventory(
    root: Path,
    manifest: Mapping[str, object],
    receipt: Mapping[str, object],
) -> dict[str, object]:
    """Re-hash the current backup blob files and compare them with the receipt."""

    if manifest.get("blob_inventory_file") != "blob-inventory.json":
        raise CursorAuthorityReconciliationError("backup uses a noncanonical blob inventory path")
    inventory_path = root / "blob-inventory.json"
    try:
        inventory_metadata = inventory_path.lstat()
    except OSError as exc:
        raise CursorAuthorityReconciliationError("backup blob inventory is unreadable") from exc
    if stat.S_ISLNK(inventory_metadata.st_mode) or not stat.S_ISREG(inventory_metadata.st_mode):
        raise CursorAuthorityReconciliationError("backup blob inventory is not a regular file")
    if inventory_metadata.st_nlink != 1:
        raise CursorAuthorityReconciliationError("backup blob inventory must not be hard-linked")
    inventory_evidence = receipt.get("blob_inventory_file")
    if not isinstance(inventory_evidence, dict):
        raise CursorAuthorityReconciliationError("backup blob inventory lacks authenticated file evidence")
    if (
        inventory_evidence.get("path") != "blob-inventory.json"
        or inventory_evidence.get("present") is not True
        or inventory_evidence.get("size_bytes") != inventory_metadata.st_size
        or inventory_evidence.get("sha256") != _sha256_file(inventory_path)
    ):
        raise CursorAuthorityReconciliationError("backup blob inventory does not match its verification receipt")
    try:
        declared = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CursorAuthorityReconciliationError("backup blob inventory is unreadable") from exc
    expected = receipt.get("blobs")
    if not isinstance(declared, list) or not isinstance(expected, list):
        raise CursorAuthorityReconciliationError("backup blob inventory is not fully attested")

    declared_by_hash: dict[str, dict[str, object]] = {}
    for item in declared:
        if not isinstance(item, dict) or not isinstance(item.get("blob_hash"), str):
            raise CursorAuthorityReconciliationError("backup blob inventory contains an invalid row")
        blob_hash = str(item["blob_hash"]).lower()
        if len(blob_hash) != 64 or any(character not in "0123456789abcdef" for character in blob_hash):
            raise CursorAuthorityReconciliationError("backup blob inventory contains an invalid blob hash")
        if blob_hash in declared_by_hash:
            raise CursorAuthorityReconciliationError("backup blob inventory contains duplicate blob hashes")
        declared_by_hash[blob_hash] = item

    actual_rows: list[dict[str, object]] = []
    blob_root = root / "blob"
    if blob_root.is_symlink():
        raise CursorAuthorityReconciliationError("backup blob root is a symlink")
    for path in sorted(blob_root.rglob("*")):
        if path.is_symlink():
            raise CursorAuthorityReconciliationError("backup blob inventory contains a symlink")
        if not path.is_file():
            continue
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise CursorAuthorityReconciliationError(f"backup blob is unreadable: {path}") from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise CursorAuthorityReconciliationError(f"backup blob must be a single-linked regular file: {path}")
        relative = path.relative_to(root).as_posix()
        if len(path.parent.name) != 2 or len(path.name) != 62:
            raise CursorAuthorityReconciliationError(f"backup blob path is not content-addressed: {relative}")
        blob_hash = f"{path.parent.name}{path.name}".lower()
        if relative != f"blob/{blob_hash[:2]}/{blob_hash[2:]}":
            raise CursorAuthorityReconciliationError(f"backup blob path is not canonical: {relative}")
        if blob_hash not in declared_by_hash:
            raise CursorAuthorityReconciliationError("backup contains a blob absent from blob-inventory.json")
        size_bytes, sha256 = _file_fingerprint(path)
        if sha256 != blob_hash:
            raise CursorAuthorityReconciliationError(f"backup blob digest does not match its path: {relative}")
        declared_item = declared_by_hash[blob_hash]
        protection = declared_item.get("protection")
        if not isinstance(protection, list) or not all(isinstance(value, str) for value in protection):
            raise CursorAuthorityReconciliationError("backup blob inventory has invalid protection metadata")
        if declared_item.get("size_bytes") != size_bytes:
            raise CursorAuthorityReconciliationError("backup blob size disagrees with blob-inventory.json")
        actual_rows.append(
            {
                "blob_hash": blob_hash,
                "path": relative,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "protection": sorted(str(value) for value in protection),
            }
        )

    actual_rows.sort(key=lambda item: str(item["blob_hash"]))
    expected_rows: list[dict[str, object]] = []
    for item in expected:
        if not isinstance(item, dict):
            raise CursorAuthorityReconciliationError("backup verification receipt contains an invalid blob row")
        expected_rows.append(
            {
                "blob_hash": item.get("blob_hash"),
                "path": item.get("path"),
                "size_bytes": item.get("size_bytes"),
                "sha256": item.get("sha256"),
                "protection": sorted(str(value) for value in item.get("protection", []))
                if isinstance(item.get("protection"), list)
                else item.get("protection"),
            }
        )
    expected_rows.sort(key=lambda item: str(item["blob_hash"]))
    if expected_rows != actual_rows:
        raise CursorAuthorityReconciliationError(
            "current backup blob inventory does not match its verification receipt"
        )
    if len(declared_by_hash) != len(actual_rows):
        raise CursorAuthorityReconciliationError("blob-inventory.json contains a missing backup blob")
    manifest_count = manifest.get("blob_count")
    if isinstance(manifest_count, int) and manifest_count != len(actual_rows):
        raise CursorAuthorityReconciliationError("backup manifest blob count does not match current blob inventory")
    total_size_bytes = 0
    for item in actual_rows:
        row_size_bytes = item["size_bytes"]
        if not isinstance(row_size_bytes, int):
            raise CursorAuthorityReconciliationError("current backup blob inventory has an invalid size")
        total_size_bytes += row_size_bytes
    return {
        "count": len(actual_rows),
        "size_bytes": total_size_bytes,
        "inventory_digest": _canonical_digest(actual_rows),
    }


def _validate_backup(manifest_path: Path, plan: Mapping[str, object]) -> dict[str, object]:
    root = _backup_root(manifest_path)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        receipt = json.loads((root / "verification-receipt.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CursorAuthorityReconciliationError("backup lacks a readable verification receipt") from exc
    if not isinstance(manifest, dict) or manifest.get("profile") != "full_evidence":
        raise CursorAuthorityReconciliationError("apply requires a full_evidence backup")
    if not isinstance(receipt, dict) or receipt.get("verdict") != "success":
        raise CursorAuthorityReconciliationError("backup verification receipt is not successful")
    included = {str(item) for item in manifest.get("included_tiers", []) if isinstance(item, str)}
    if {f"{tier}.db" for tier in _REQUIRED_TIERS} - included:
        raise CursorAuthorityReconciliationError("full-evidence backup lacks source/index/ops/audit rollback evidence")
    verification = receipt.get("verification")
    required_verification = ("source_blobs_resolved", "index_attachment_blobs_resolved", "blob_inventory_exact")
    if not isinstance(verification, dict) or any(verification.get(key) is not True for key in required_verification):
        raise CursorAuthorityReconciliationError("backup lacks complete blob rollback evidence")
    if not (root / "blob").is_dir() or not (root / "blob-inventory.json").is_file():
        raise CursorAuthorityReconciliationError("backup lacks blob rollback evidence")
    archive_root = _archive_root()
    location = ArchiveLocation.resolve(archive_root)
    expected_active_index = plan.get("active_index")
    if expected_active_index is not None and expected_active_index != _active_index_binding(archive_root):
        raise CursorAuthorityReconciliationError("active index generation changed since planning")
    try:
        verify_verification_receipt(
            receipt,
            tier="source",
            live_tier_path=location.configured_tier("source").configured_path,
        )
        verify_verification_receipt(
            receipt,
            tier="user",
            live_tier_path=location.configured_tier("user").configured_path,
        )
    except BackupAttestationError as exc:
        raise CursorAuthorityReconciliationError("backup verification receipt attestation is invalid") from exc
    manifest_sha256 = _sha256_file(root / "manifest.json")
    if receipt.get("manifest_sha256") != manifest_sha256:
        raise CursorAuthorityReconciliationError("backup manifest does not match its verification receipt")
    blob_inventory = _validated_blob_inventory(root, manifest, receipt)
    declared = manifest.get("tier_source_fingerprints")
    expected = plan.get("tier_fingerprints")
    if not isinstance(declared, dict) or not isinstance(expected, dict):
        raise CursorAuthorityReconciliationError("plan or backup lacks tier fingerprints")
    for tier in _REQUIRED_TIERS:
        artifact = declared.get(f"{tier}.db")
        expected_tier = expected.get(tier)
        if not isinstance(artifact, dict) or not isinstance(expected_tier, dict):
            raise CursorAuthorityReconciliationError(f"backup lacks {tier} fingerprint")
        for key in ("size_bytes", "sha256", "user_version"):
            if artifact.get(key) != expected_tier.get(key):
                raise CursorAuthorityReconciliationError(f"backup {tier} fingerprint does not match the plan")
        backup_tier = root / f"{tier}.db"
        if not backup_tier.is_file():
            raise CursorAuthorityReconciliationError(f"backup {tier} tier is missing")
        actual = _sqlite_snapshot(backup_tier)
        if actual.get("sha256") != expected_tier.get("sha256") or actual.get("size_bytes") != expected_tier.get(
            "size_bytes"
        ):
            raise CursorAuthorityReconciliationError(f"backup {tier} image does not match the plan fingerprint")
        if tier == "index" and expected_active_index is not None:
            source_fingerprint = artifact.get("path")
            if (
                not isinstance(source_fingerprint, str)
                or Path(source_fingerprint).resolve() != location.active_index_path.resolve()
            ):
                raise CursorAuthorityReconciliationError(
                    "backup index fingerprint does not bind the active index generation"
                )
    return {
        "root": _path_identity(root),
        "manifest_sha256": manifest_sha256,
        "blob_inventory": blob_inventory,
    }


def _quick_checks(root: Path) -> dict[str, list[str]]:
    checks: dict[str, list[str]] = {}
    location = ArchiveLocation.resolve(root)
    for tier in ("source", "index", "ops", "audit"):
        tier_path = location.active_index_path if tier == "index" else root / f"{tier}.db"
        with closing(sqlite3.connect(f"file:{tier_path.resolve()}?mode=ro", uri=True)) as conn:
            checks[tier] = [str(row[0]) for row in conn.execute("PRAGMA quick_check")]
        if checks[tier] != ["ok"]:
            raise CursorAuthorityReconciliationError(f"{tier}.db quick_check failed: {checks[tier]}")
    return checks


def _write_atomic_json(path: Path, payload: Mapping[str, object], *, refuse_existing: bool) -> None:
    if refuse_existing and path.exists():
        raise CursorAuthorityReconciliationError(f"output path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if refuse_existing:
            try:
                os.link(temporary_path, path)
            except FileExistsError as exc:
                raise CursorAuthorityReconciliationError(f"output path already exists: {path}") from exc
            temporary_path.unlink()
        else:
            os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _require_daemon_stopped(root: Path) -> None:
    config = Config(
        archive_root=root,
        render_root=root / "render",
        sources=[],
        db_path=ArchiveLocation.resolve(root).active_index_path,
    )
    from polylogue.maintenance.offline_guard import running_daemon_pid

    if running_daemon_pid(config) is not None:
        raise CursorAuthorityReconciliationError("daemon must be stopped for cursor-authority reconciliation")


def build_reconciliation_plan(*, source_path_file: Path, output_plan: Path) -> dict[str, object]:
    root = _archive_root()
    _require_daemon_stopped(root)
    source_path = _read_private_source_path(source_path_file)
    plan = _build_plan(root, source_path)
    _write_atomic_json(output_plan, plan, refuse_existing=True)
    return plan


def _find_recovery_attempt(
    root: Path,
    source_path: Path,
    plan_observed_at_ms: int,
    *,
    plan_digest: str,
    path_digest: str,
) -> dict[str, object] | None:
    with closing(sqlite3.connect(f"file:{(root / 'ops.db').resolve()}?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            "SELECT attempt_id, status, source_path, source_paths_json, finished_at_ms, "
            "outcome_code, retryable, diagnostic, remediation FROM ingest_attempts "
            "ORDER BY COALESCE(finished_at_ms, heartbeat_at_ms, started_at_ms) DESC LIMIT 50"
        ).fetchall()
        event_rows = conn.execute(
            "SELECT attempt_id, payload_json FROM daemon_stage_events "
            "WHERE stage = 'planning' ORDER BY observed_at_ms DESC LIMIT 200"
        ).fetchall()
    planning_bindings: dict[str, bool] = {}
    for attempt_id, payload_json in event_rows:
        try:
            payload = json.loads(str(payload_json))
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        planning_bindings[str(attempt_id)] = (
            payload.get("cursor_authority_plan_digest") == plan_digest
            and payload.get("cursor_authority_path_digest") == path_digest
        )
    for (
        attempt_id,
        status,
        single_path,
        paths_json,
        finished_at_ms,
        outcome_code,
        retryable,
        diagnostic,
        remediation,
    ) in rows:
        if str(status) not in {"completed", "completed_with_failures"}:
            continue
        if not isinstance(finished_at_ms, int) or finished_at_ms <= plan_observed_at_ms:
            continue
        if not planning_bindings.get(str(attempt_id), False):
            continue
        values: list[str] = []
        if isinstance(paths_json, str):
            try:
                decoded = json.loads(paths_json)
            except ValueError:
                decoded = []
            if isinstance(decoded, list):
                values.extend(str(value) for value in decoded if isinstance(value, str))
        if not values and single_path:
            values.append(str(single_path))
        if any(Path(value).resolve() == source_path.resolve() for value in values):
            return {
                "attempt_id": str(attempt_id),
                "status": str(status),
                "finished_at_ms": finished_at_ms,
                "outcome_code": None if outcome_code is None else str(outcome_code),
                "retryable": None if retryable is None else bool(retryable),
                "diagnostic": None if diagnostic is None else str(diagnostic),
                "remediation": None if remediation is None else str(remediation),
            }
    return None


async def _normal_ingest(
    root: Path, source_path: Path, plan: Mapping[str, object]
) -> tuple[LiveBatchMetrics, dict[str, object]]:
    from polylogue.sources.live import watcher as live_watcher

    async with Polylogue(archive_root=root, db_path=ArchiveLocation.resolve(root).active_index_path) as polylogue:
        cursor = CursorStore(root / "ops.db", initialize=False, ops_db_path=root / "ops.db")
        processor = LiveBatchProcessor(
            polylogue,
            (WatchSource(name=source_path.parent.name, root=source_path.parent),),
            cursor=cursor,
            parser_fingerprint=lambda: live_watcher._PARSER_FINGERPRINT,
        )
        with scoped_cursor_authority_authorization(
            source_path_digest=str(plan["selected_path_digest"]),
            cursor_byte_offset=_plan_int(plan, "cursor_byte_offset"),
            accepted_frontier=_plan_int(plan, "accepted_frontier"),
            plan_digest=str(plan["plan_digest"]),
            force_full_ingest=True,
        ):
            metrics = await processor.ingest_files([source_path], emit_event=False)
        attempt = _find_recovery_attempt(
            root,
            source_path,
            _plan_int(plan, "observed_at_ms"),
            plan_digest=str(plan["plan_digest"]),
            path_digest=str(plan["selected_path_digest"]),
        )
        if attempt is None:
            raise CursorAuthorityReconciliationError("completed ingest attempt lacks reconciliation planning binding")
        return metrics, attempt


def _plan_int(plan: Mapping[str, object], key: str) -> int:
    value = plan.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CursorAuthorityReconciliationError(f"reconciliation plan field {key} is not an integer")
    return value


def _required_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CursorAuthorityReconciliationError(f"{field} is missing or invalid")
    return value


def _before_projection(plan: Mapping[str, object]) -> dict[str, object]:
    value = plan.get("before_projection")
    if not isinstance(value, dict):
        raise CursorAuthorityReconciliationError("plan lacks the before-projection authority census")
    return value


def _require_selected_path_in_before_projection(plan: Mapping[str, object], source_path: Path) -> None:
    before_projection = _before_projection(plan)
    samples = before_projection.get("cursor_ahead_samples")
    selected_path_digest = cursor_authority_path_digest(source_path)
    if not isinstance(samples, list) or not any(
        isinstance(sample, dict) and sample.get("source_path") == selected_path_digest for sample in samples
    ):
        raise CursorAuthorityReconciliationError(
            "plan does not bind the selected path to a previously observed cursor-ahead violation"
        )


def _same_plan_bindings(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    def comparable(plan: Mapping[str, object]) -> dict[str, object]:
        value = dict(plan)
        value.pop("observed_at_ms", None)
        value.pop("plan_digest", None)
        return value

    return comparable(left) == comparable(right)


def _changed_rows() -> dict[str, int | None]:
    return {"cursor": None, "accepted_head_direct_writes": 0}


def _typed_retryable_attempt(attempt: Mapping[str, object]) -> bool:
    outcome_code = attempt.get("outcome_code")
    retryable = attempt.get("retryable")
    if not isinstance(outcome_code, str) or retryable is not True:
        return False
    try:
        outcome = IngestOutcome(outcome_code)
    except ValueError:
        return False
    return IngestAttemptDisposition(outcome=outcome).retryable is True


def _redacted_metrics(metrics: LiveBatchMetrics | None) -> dict[str, object] | None:
    if metrics is None:
        return None
    payload = metrics.to_payload()
    payload["failed_paths"] = [_path_identity(Path(str(path))) for path in metrics.failed_paths]
    payload["new_sessions"] = [
        {"source_name_digest": _identity_digest(source_name), "session_id_digest": _identity_digest(session_id)}
        for source_name, session_id in metrics.new_sessions
    ]
    payload["updated_sessions"] = [
        {"source_name_digest": _identity_digest(source_name), "session_id_digest": _identity_digest(session_id)}
        for source_name, session_id in metrics.updated_sessions
    ]
    return payload


def _receipt_payload(
    *,
    plan: Mapping[str, object],
    backup: Mapping[str, object],
    root: Path,
    verdict: str,
    before_projection: Mapping[str, object],
    after_projection: RawFrontierIntegrityProjection | None,
    metrics: LiveBatchMetrics | None,
    attempt_id: str | None,
    attempt_observation: str,
    evidence: Mapping[str, object],
    tolerate_state_errors: bool = False,
) -> dict[str, object]:
    try:
        tier_fingerprints: object = _tier_snapshots(root)
        quick_check: object = _quick_checks(root)
    except Exception:
        if not tolerate_state_errors:
            raise
        tier_fingerprints = None
        quick_check = None
    return {
        "format": RECEIPT_FORMAT,
        "verdict": verdict,
        "archive_identity": {
            "root": _path_identity(root),
            "active_index": plan.get("active_index"),
        },
        "plan_digest": plan["plan_digest"],
        "backup": dict(backup),
        "before_projection": dict(before_projection),
        "after_projection": _private_projection(after_projection) if after_projection is not None else None,
        "metrics": _redacted_metrics(metrics),
        "changed_rows": _changed_rows(),
        "ingest_attempt_id": attempt_id,
        "ingest_attempt_observation": attempt_observation,
        "operation": attempt_observation,
        "code_sha": plan.get("code_sha"),
        "deployed_package_sha": plan.get("deployed_package_sha"),
        "tier_fingerprints": tier_fingerprints,
        "quick_check": quick_check,
        "evidence": dict(evidence),
    }


def apply_reconciliation(*, plan_path: Path, backup_manifest: Path, receipt: Path) -> dict[str, object]:
    plan = _load_plan(plan_path)
    if plan.get("status") != "planned":
        raise CursorAuthorityReconciliationError("only a planned one-path reconciliation can be applied")
    root = _archive_root()
    _require_daemon_stopped(root)
    if receipt.exists():
        raise CursorAuthorityReconciliationError(f"output path already exists: {receipt}")
    before_projection = _before_projection(plan)
    owner = acquire_durable_archive_ownership(root, owner_id=f"cursor-authority-reconcile:{os.getpid()}")
    with owner:
        backup_evidence = _validate_backup(backup_manifest, plan)
        current_path = _find_path_by_digest(root, str(plan["selected_path_digest"]))
        _require_selected_path_in_before_projection(plan, current_path)
        try:
            current_plan = _build_plan(root, current_path)
        except CursorAuthorityReconciliationError:
            current_plan = None
        if current_plan is None or not _same_plan_bindings(current_plan, plan):
            recovery_projection = _projection_for(root)
            _require_healthy_projection_siblings(recovery_projection)
            recovery_attempt = _find_recovery_attempt(
                root,
                current_path,
                _plan_int(plan, "observed_at_ms"),
                plan_digest=str(plan["plan_digest"]),
                path_digest=str(plan["selected_path_digest"]),
            )
            if recovery_attempt is None or recovery_projection.cursor_ahead_count != 0:
                raise CursorAuthorityReconciliationError("plan bindings changed before archive ownership")
            if recovery_projection.cursor_ahead_status != "healthy":
                raise CursorAuthorityReconciliationError("recovery did not prove a healthy cursor frontier")
            before_gap_count = before_projection.get("cursor_authority_gap_count")
            if (
                not isinstance(before_gap_count, int)
                or recovery_projection.cursor_authority_gap_count != before_gap_count
            ):
                raise CursorAuthorityReconciliationError(
                    "recovered ingest changed the pre-existing incomparable cursor population"
                )
            recovered_receipt_payload = _receipt_payload(
                plan=plan,
                backup=backup_evidence,
                root=root,
                verdict="reconciled",
                before_projection=before_projection,
                after_projection=recovery_projection,
                metrics=None,
                attempt_id=str(recovery_attempt["attempt_id"]),
                attempt_observation="observed",
                evidence={
                    "raw_frontier_worsening": False,
                    "invalid_ahead_reconciliation": False,
                    "changed_pre_existing_populations": False,
                    "attempt_outcome_code": recovery_attempt.get("outcome_code"),
                },
                tolerate_state_errors=False,
            )
            recovered_receipt_payload["receipt_digest"] = _canonical_digest(recovered_receipt_payload)
            _write_atomic_json(receipt, recovered_receipt_payload, refuse_existing=True)
            return recovered_receipt_payload
        current_plan = _build_plan(root, current_path)
        if not _same_plan_bindings(current_plan, plan):
            raise CursorAuthorityReconciliationError("plan bindings changed after archive ownership")
        metrics: LiveBatchMetrics | None = None
        attempt: dict[str, object] | None = None
        attempt_id = "unknown"
        after_projection: RawFrontierIntegrityProjection | None = None
        evidence: dict[str, object] = {
            "raw_frontier_worsening": False,
            "invalid_ahead_reconciliation": False,
            "changed_pre_existing_populations": False,
        }
        try:
            metrics, attempt = asyncio.run(_normal_ingest(root, current_path, plan))
            attempt_id = str(attempt["attempt_id"])
            after_projection = _projection_for(root)
            _require_healthy_projection_siblings(after_projection)
            if after_projection.broken_head_count or after_projection.missing_source_raw_count:
                evidence["raw_frontier_worsening"] = True
                raise CursorAuthorityReconciliationError("reconciliation introduced unrelated raw-frontier worsening")
            if after_projection.cursor_ahead_count:
                if (
                    metrics.succeeded_file_count != 0
                    or str(current_path) not in metrics.failed_paths
                    or metrics.time_budget_exceeded
                    or not _typed_retryable_attempt(attempt)
                ):
                    evidence["invalid_ahead_reconciliation"] = True
                    raise CursorAuthorityReconciliationError(
                        "cursor-ahead postcondition lacks a typed retryable deferral outcome"
                    )
                verdict = "typed_deferred"
            else:
                if after_projection.cursor_ahead_status != "healthy":
                    raise CursorAuthorityReconciliationError("reconciliation did not prove a healthy cursor frontier")
                verdict = "reconciled"
            before_gap_count = before_projection.get("cursor_authority_gap_count")
            if not isinstance(before_gap_count, int) or after_projection.cursor_authority_gap_count != before_gap_count:
                evidence["changed_pre_existing_populations"] = True
                raise CursorAuthorityReconciliationError(
                    "reconciliation changed the pre-existing incomparable cursor population"
                )
            receipt_payload = _receipt_payload(
                plan=plan,
                backup=backup_evidence,
                root=root,
                verdict=verdict,
                before_projection=before_projection,
                after_projection=after_projection,
                metrics=metrics,
                attempt_id=attempt_id,
                attempt_observation="performed",
                evidence={**evidence, "attempt_outcome_code": attempt.get("outcome_code") if attempt else None},
                tolerate_state_errors=False,
            )
        except Exception as exc:
            if after_projection is None:
                try:
                    after_projection = _projection_for(root)
                except Exception:
                    after_projection = None
            failure_payload = _receipt_payload(
                plan=plan,
                backup=backup_evidence,
                root=root,
                verdict="failed",
                before_projection=before_projection,
                after_projection=after_projection,
                metrics=metrics,
                attempt_id=attempt_id,
                attempt_observation="performed",
                evidence=evidence,
                tolerate_state_errors=True,
            )
            failure_message = str(exc).replace(str(root), "<archive-root>")
            if "current_path" in locals():
                failure_message = failure_message.replace(str(current_path), f"<source:{current_path.name}>")
            failure_payload["error"] = {"type": type(exc).__name__, "message": failure_message}
            failure_payload["receipt_digest"] = _canonical_digest(failure_payload)
            _write_atomic_json(receipt, failure_payload, refuse_existing=True)
            if isinstance(exc, CursorAuthorityReconciliationError):
                raise
            raise CursorAuthorityReconciliationError("cursor-authority reconciliation failed after ingest") from exc
        receipt_payload["receipt_digest"] = _canonical_digest(receipt_payload)
        _write_atomic_json(receipt, receipt_payload, refuse_existing=True)
        return receipt_payload


__all__ = [
    "ARCHIVE_ROOT",
    "PLAN_FORMAT",
    "RECEIPT_FORMAT",
    "CursorAuthorityReconciliationError",
    "apply_reconciliation",
    "build_reconciliation_plan",
    "cursor_authority_path_digest",
]
