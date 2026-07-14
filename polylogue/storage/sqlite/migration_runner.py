"""Versioned additive migrations for durable archive tiers."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import stat
from contextlib import closing
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from polylogue.storage.backup_attestation import (
    VERIFICATION_RECEIPT_FORMAT,
    BackupAttestationError,
    verify_verification_receipt,
)
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

DURABLE_MIGRATION_TIERS: frozenset[ArchiveTier] = frozenset({ArchiveTier.SOURCE, ArchiveTier.USER})
_MIGRATION_NAME_RE = re.compile(r"^(?P<version>\d{3,})_[a-z0-9_]+\.sql$")
_VERIFICATION_RECEIPT_FILE = "verification-receipt.json"
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
_ADDITIVE_NO_BACKUP_MARKER = "-- migration-safety: additive-no-backup"


class MigrationError(RuntimeError):
    """Raised when a durable tier cannot be migrated safely."""


@dataclass(frozen=True, slots=True)
class MigrationStep:
    tier: ArchiveTier
    version: int
    name: str
    sql: str
    requires_backup: bool


@dataclass(frozen=True, slots=True)
class MigrationResult:
    tier: ArchiveTier
    from_version: int
    to_version: int
    applied_versions: tuple[int, ...]
    backup_receipt: Path | None = None


def _migration_package(tier: ArchiveTier) -> str:
    return f"polylogue.storage.sqlite.migrations.{tier.value}"


def _load_migrations(tier: ArchiveTier) -> tuple[MigrationStep, ...]:
    if tier not in DURABLE_MIGRATION_TIERS:
        return ()
    try:
        files = resources.files(_migration_package(tier))
    except ModuleNotFoundError:
        return ()
    steps: list[MigrationStep] = []
    for item in sorted(files.iterdir(), key=lambda path: path.name):
        match = _MIGRATION_NAME_RE.match(item.name)
        if match is None:
            continue
        sql = item.read_text(encoding="utf-8")
        steps.append(
            MigrationStep(
                tier=tier,
                version=int(match.group("version")),
                name=item.name,
                sql=sql,
                requires_backup=_ADDITIVE_NO_BACKUP_MARKER not in sql,
            )
        )
    versions = [step.version for step in steps]
    if len(versions) != len(set(versions)):
        raise MigrationError(f"duplicate {tier.value} migration versions: {versions}")
    return tuple(steps)


def _backup_manifest_path(path: Path) -> Path:
    return path / "manifest.json" if path.is_dir() else path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_real_backup_directory(path: Path, *, label: str) -> Path:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise MigrationError(f"migration {label} is missing: {path}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise MigrationError(f"migration {label} is not a real directory: {path}")
    return path.resolve(strict=True)


def _require_regular_backup_artifact(path: Path, *, backup_root: Path, label: str) -> None:
    root_resolved = _require_real_backup_directory(backup_root, label="backup root")
    try:
        relative = path.relative_to(backup_root)
    except ValueError as exc:
        raise MigrationError(f"migration {label} is outside the backup root: {path}") from exc
    current = backup_root
    for part in relative.parts[:-1]:
        current /= part
        _require_real_backup_directory(current, label=f"{label} parent")
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise MigrationError(f"migration {label} is missing: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise MigrationError(f"migration {label} is not a real regular file: {path}")
    if metadata.st_nlink != 1:
        raise MigrationError(f"migration {label} has multiple hard links: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved):
        raise MigrationError(f"migration {label} resolves outside the backup root: {path}")


def _regular_backup_blob_files(backup_root: Path) -> list[Path]:
    blob_root = backup_root / "blob"
    if not blob_root.exists() and not blob_root.is_symlink():
        return []
    _require_real_backup_directory(blob_root, label="backup blob root")
    files: list[Path] = []
    for candidate in sorted(blob_root.rglob("*")):
        metadata = candidate.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            _require_real_backup_directory(candidate, label="backup blob directory")
            continue
        _require_regular_backup_artifact(candidate, backup_root=backup_root, label="backup blob")
        files.append(candidate)
    return files


def _reject_sqlite_sidecars(path: Path) -> None:
    for suffix in _SQLITE_SIDECAR_SUFFIXES:
        sidecar = Path(f"{path}{suffix}")
        if sidecar.exists() or sidecar.is_symlink():
            raise MigrationError(f"migration backup tier has an unbound SQLite sidecar: {sidecar}")


def _backup_artifact_inventory(backup_root: Path) -> list[dict[str, object]]:
    _require_real_backup_directory(backup_root, label="backup root")
    rows: list[dict[str, object]] = []
    for candidate in sorted(backup_root.rglob("*")):
        relative = candidate.relative_to(backup_root)
        if relative == Path(_VERIFICATION_RECEIPT_FILE):
            continue
        metadata = candidate.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            _require_real_backup_directory(candidate, label="backup artifact directory")
            rows.append({"path": str(relative), "type": "directory"})
            continue
        if candidate.name.endswith(_SQLITE_SIDECAR_SUFFIXES):
            raise MigrationError(f"migration backup contains an unbound SQLite sidecar: {candidate}")
        _require_regular_backup_artifact(candidate, backup_root=backup_root, label="backup artifact")
        rows.append(
            {
                "path": str(relative),
                "type": "file",
                "size_bytes": metadata.st_size,
                "sha256": _sha256_file(candidate),
            }
        )
    return rows


@dataclass(frozen=True, slots=True)
class _BackupInventoryCacheEntry:
    stat_signature: tuple[tuple[str, int, int], ...]
    artifact_inventory: tuple[dict[str, object], ...]


# Process-lifetime cache: a durable migration runs as a short-lived actuator
# invocation (devtools schema fast-forward, `polylogue ops` maintenance, or
# one daemon-startup migration), so this never needs cross-process
# persistence or eviction -- it exists to collapse the four SHA-256 scans of
# one immutable backup tree that `migrate_archive_tier` otherwise performs
# per activation (pre-BEGIN + in-transaction, times two durable tiers) into
# one.
_backup_artifact_inventory_cache: dict[Path, _BackupInventoryCacheEntry] = {}


def _backup_root_stat_signature(backup_root: Path) -> tuple[tuple[str, int, int], ...]:
    """Cheap, content-free fingerprint used only to decide whether the
    expensive SHA-256 scan below can be skipped.

    This is deliberately not evidence by itself: every entry the cache
    returns still carries the SHA-256 computed the last time the signature
    changed.  A stat-identical-but-content-tampered backup (same size and
    mtime, different bytes) is outside this actuator's threat model already
    -- ``backup_attestation.py`` documents it is "not a privilege boundary
    against arbitrary code running as the same Unix user" -- and any
    genuine artifact/manifest/receipt mutation changes size or mtime and is
    still caught below.
    """
    entries: list[tuple[str, int, int]] = []
    for candidate in sorted(backup_root.rglob("*")):
        relative = candidate.relative_to(backup_root)
        if relative == Path(_VERIFICATION_RECEIPT_FILE):
            continue
        metadata = candidate.lstat()
        entries.append((str(relative), metadata.st_size, metadata.st_mtime_ns))
    return tuple(entries)


def _cached_backup_artifact_inventory(backup_root: Path) -> list[dict[str, object]]:
    """Reuse a SHA-256'd backup artifact inventory while its bytes are unchanged."""
    resolved = backup_root.resolve(strict=True)
    signature = _backup_root_stat_signature(resolved)
    cached = _backup_artifact_inventory_cache.get(resolved)
    if cached is not None and cached.stat_signature == signature:
        return [dict(item) for item in cached.artifact_inventory]
    inventory = _backup_artifact_inventory(resolved)
    # Guard a scan-time mutation race: only cache a result whose signature is
    # still what it was before the (potentially slow) hashing pass began.
    if _backup_root_stat_signature(resolved) == signature:
        _backup_artifact_inventory_cache[resolved] = _BackupInventoryCacheEntry(
            stat_signature=signature,
            artifact_inventory=tuple(dict(item) for item in inventory),
        )
    else:
        _backup_artifact_inventory_cache.pop(resolved, None)
    return inventory


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sqlite_user_version(path: Path) -> int:
    with closing(sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)) as conn:
        return int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)


def _json_int(value: object, default: int = -1) -> int:
    return value if isinstance(value, int) else default


def _json_str_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _connection_main_path(conn: sqlite3.Connection) -> Path:
    for _seq, name, filename in conn.execute("PRAGMA database_list"):
        if name == "main" and filename:
            return Path(str(filename))
    raise MigrationError("migration backup receipt validation requires a file-backed SQLite connection")


def _checkpoint_live_tier(conn: sqlite3.Connection) -> None:
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    except sqlite3.Error as exc:
        raise MigrationError("migration backup receipt validation could not checkpoint the live tier") from exc
    if row is None:
        raise MigrationError("migration backup receipt validation could not checkpoint the live tier")
    busy, log_frames, checkpointed_frames = (int(value) for value in row)
    if busy or log_frames != checkpointed_frames:
        raise MigrationError("migration backup receipt validation could not quiesce the live tier")


def _receipt_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(_VERIFICATION_RECEIPT_FILE)


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MigrationError(f"migration backup {label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise MigrationError(f"migration backup {label} must be a JSON object: {path}")
    return payload


def _validate_tier_artifact(
    backup_root: Path,
    artifact: dict[str, object],
    *,
    file_evidence: dict[str, dict[str, object]],
    live_tier_path: Path | None = None,
) -> None:
    tier = artifact.get("tier")
    if not isinstance(tier, str) or tier not in {item.value for item in ArchiveTier}:
        raise MigrationError("migration backup receipt tier artifact has an invalid tier")
    filename = artifact.get("path")
    if not isinstance(filename, str) or filename != f"{tier}.db":
        raise MigrationError("migration backup receipt tier artifact path is not canonical")
    artifact_path = backup_root / filename
    if not artifact_path.exists() and not artifact_path.is_symlink():
        raise MigrationError(f"migration backup receipt references missing tier artifact: {artifact_path}")
    _require_regular_backup_artifact(artifact_path, backup_root=backup_root, label="backup tier")
    _reject_sqlite_sidecars(artifact_path)
    if live_tier_path is not None and artifact_path.samefile(live_tier_path):
        raise MigrationError(f"migration backup tier artifact aliases the live tier: {filename}")
    current_evidence = file_evidence.get(filename, {})
    if _json_int(artifact.get("size_bytes")) != _json_int(current_evidence.get("size_bytes")):
        raise MigrationError(f"migration backup tier artifact size mismatch: {filename}")
    if str(artifact.get("sha256")) != str(current_evidence.get("sha256")):
        raise MigrationError(f"migration backup tier artifact hash mismatch: {filename}")
    if _json_int(artifact.get("user_version")) != _sqlite_user_version(artifact_path):
        raise MigrationError(f"migration backup tier artifact user_version mismatch: {filename}")
    source_fingerprint = artifact.get("source_fingerprint")
    if not isinstance(source_fingerprint, dict) or any(
        artifact.get(field) != source_fingerprint.get(field) for field in ("size_bytes", "sha256", "user_version")
    ):
        raise MigrationError(f"migration backup tier artifact does not match its live source fingerprint: {filename}")


def _validated_receipt_artifacts(
    backup_root: Path,
    manifest: dict[str, object],
    receipt: dict[str, object],
    *,
    target_tier: str,
    live_tier_path: Path,
    file_evidence: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    included = _json_str_list(manifest.get("included_tiers"))
    if len(included) != len(set(included)) or any(
        name not in {f"{tier.value}.db" for tier in ArchiveTier} for name in included
    ):
        raise MigrationError("migration backup manifest has non-canonical included tiers")
    receipt_tiers = _json_str_list(receipt.get("included_tiers"))
    if receipt_tiers != included:
        raise MigrationError("migration backup receipt included tiers do not match the manifest")
    artifacts = receipt.get("tier_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(included):
        raise MigrationError("migration backup receipt does not bind every included tier artifact")
    by_tier: dict[str, dict[str, object]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise MigrationError("migration backup receipt tier artifact is invalid")
        tier = artifact.get("tier")
        if not isinstance(tier, str) or f"{tier}.db" not in included or tier in by_tier:
            raise MigrationError("migration backup receipt tier artifacts do not match included tiers")
        _validate_tier_artifact(
            backup_root,
            artifact,
            file_evidence=file_evidence,
            live_tier_path=live_tier_path if tier == target_tier else None,
        )
        by_tier[tier] = artifact
    if set(by_tier) != {name.removesuffix(".db") for name in included}:
        raise MigrationError("migration backup receipt tier artifacts do not match included tiers")
    return by_tier


def _validate_live_source_fingerprint(conn: sqlite3.Connection, artifact: dict[str, object]) -> None:
    fingerprint = artifact.get("source_fingerprint")
    if not isinstance(fingerprint, dict):
        raise MigrationError("migration backup receipt is missing the live source fingerprint")
    live_path = _connection_main_path(conn)
    wal_path = live_path.with_name(f"{live_path.name}-wal")
    if wal_path.exists() and wal_path.stat().st_size:
        raise MigrationError("migration backup receipt live tier changed before the migration lock")
    recorded_path_value = fingerprint.get("path")
    recorded_path = Path(str(recorded_path_value)) if recorded_path_value else None
    if recorded_path is not None and live_path.resolve(strict=False) != recorded_path.resolve(strict=False):
        raise MigrationError(
            f"migration backup receipt was recorded for {recorded_path}, not the live tier {live_path}"
        )
    if _json_int(fingerprint.get("size_bytes")) != live_path.stat().st_size:
        raise MigrationError("migration backup receipt live tier size mismatch")
    if str(fingerprint.get("sha256")) != _sha256_file(live_path):
        raise MigrationError("migration backup receipt live tier hash mismatch")
    if _json_int(fingerprint.get("user_version")) != _sqlite_user_version(live_path):
        raise MigrationError("migration backup receipt live tier user_version mismatch")


def _inventory_path(backup_root: Path, manifest: dict[str, object]) -> tuple[str, Path]:
    inventory_file = manifest.get("blob_inventory_file", "blob-inventory.json")
    if not isinstance(inventory_file, str):
        raise MigrationError("migration backup blob inventory file is invalid")
    relative = Path(inventory_file)
    if relative.is_absolute() or ".." in relative.parts or relative.name != inventory_file:
        raise MigrationError("migration backup blob inventory file is not canonical")
    return inventory_file, backup_root / relative


def _blob_inventory_file_evidence(
    backup_root: Path,
    manifest: dict[str, object],
    *,
    file_evidence: dict[str, dict[str, object]],
) -> dict[str, object]:
    inventory_file, inventory_path = _inventory_path(backup_root, manifest)
    if not inventory_path.exists() and not inventory_path.is_symlink():
        return {"path": inventory_file, "present": False, "size_bytes": 0, "sha256": None}
    _require_regular_backup_artifact(
        inventory_path,
        backup_root=backup_root,
        label="backup blob inventory",
    )
    evidence = file_evidence.get(inventory_file, {})
    return {
        "path": inventory_file,
        "present": True,
        "size_bytes": evidence.get("size_bytes"),
        "sha256": evidence.get("sha256"),
    }


def _current_blob_inventory(
    backup_root: Path,
    manifest: dict[str, object],
    *,
    file_evidence: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    _inventory_file, inventory_path = _inventory_path(backup_root, manifest)
    declared: dict[str, dict[str, object]] = {}
    if inventory_path.exists() or inventory_path.is_symlink():
        _require_regular_backup_artifact(
            inventory_path,
            backup_root=backup_root,
            label="backup blob inventory",
        )
        try:
            raw_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MigrationError(f"migration backup blob inventory is not valid JSON: {inventory_path}") from exc
        if not isinstance(raw_inventory, list):
            raise MigrationError(f"migration backup blob inventory is not a JSON list: {inventory_path}")
        for item in raw_inventory:
            if isinstance(item, dict) and "blob_hash" in item:
                declared[str(item["blob_hash"])] = item
    rows: list[dict[str, object]] = []
    for blob_path in _regular_backup_blob_files(backup_root):
        blob_hash = blob_path.parent.name + blob_path.name
        if (
            len(blob_hash) != 64
            or any(character not in "0123456789abcdef" for character in blob_hash)
            or blob_path.relative_to(backup_root) != Path("blob") / blob_hash[:2] / blob_hash[2:]
        ):
            raise MigrationError(f"migration backup blob inventory has a non-canonical blob path: {blob_path}")
        declared_item = declared.get(blob_hash, {})
        protection = declared_item.get("protection")
        relative_path = str(blob_path.relative_to(backup_root))
        evidence = file_evidence.get(relative_path, {})
        rows.append(
            {
                "blob_hash": blob_hash,
                "path": relative_path,
                "size_bytes": evidence.get("size_bytes"),
                "sha256": evidence.get("sha256"),
                "protection": _json_str_list(protection),
            }
        )
    rows.sort(key=lambda item: str(item["blob_hash"]))
    return rows


def _validate_blob_inventory(
    backup_root: Path,
    manifest: dict[str, object],
    receipt: dict[str, object],
    *,
    file_evidence: dict[str, dict[str, object]],
) -> None:
    current = _current_blob_inventory(backup_root, manifest, file_evidence=file_evidence)
    if receipt.get("blob_inventory_file") != _blob_inventory_file_evidence(
        backup_root,
        manifest,
        file_evidence=file_evidence,
    ):
        raise MigrationError("migration backup receipt blob inventory file mismatch")
    expected_root = str(receipt.get("blob_inventory_root_sha256") or "")
    if expected_root != _canonical_json_sha256(current):
        raise MigrationError("migration backup receipt blob inventory mismatch")
    expected_blobs = receipt.get("blobs")
    if expected_blobs != current:
        raise MigrationError("migration backup receipt blob metadata mismatch")
    for blob in current:
        if str(blob["blob_hash"]) != str(blob["sha256"]):
            raise MigrationError(f"migration backup blob hash mismatch: {blob['path']}")


def validate_migration_backup_manifest(
    path: Path, tier: ArchiveTier, *, connection: sqlite3.Connection | None = None
) -> Path:
    """Validate that ``path`` has a successful backup verification receipt."""
    manifest_path = _backup_manifest_path(path)
    if not manifest_path.exists() and not manifest_path.is_symlink():
        raise MigrationError(f"migration requires an existing backup manifest; missing {manifest_path}")
    backup_root = manifest_path.parent
    _require_real_backup_directory(backup_root, label="backup root")
    _require_regular_backup_artifact(manifest_path, backup_root=backup_root, label="backup manifest")
    payload = _load_json(manifest_path, label="manifest")
    if payload.get("format") != "polylogue-backup-v1":
        raise MigrationError(f"migration backup manifest has unsupported format: {manifest_path}")
    included = set(_json_str_list(payload.get("included_tiers")))
    if f"{tier.value}.db" not in included:
        raise MigrationError(f"migration backup manifest does not include {tier.value}.db: {manifest_path}")
    receipt_path = _receipt_path(manifest_path)
    if not receipt_path.exists() and not receipt_path.is_symlink():
        raise MigrationError(f"migration requires a successful backup verification receipt; missing {receipt_path}")
    _require_regular_backup_artifact(receipt_path, backup_root=backup_root, label="backup verification receipt")
    receipt = _load_json(receipt_path, label="verification receipt")
    if receipt.get("format") != VERIFICATION_RECEIPT_FORMAT:
        raise MigrationError(f"migration backup receipt has unsupported format: {receipt_path}")
    if connection is None:
        raise MigrationError("migration backup receipt authentication requires the live tier connection")
    live_tier_path = _connection_main_path(connection).resolve(strict=False)
    try:
        verify_verification_receipt(receipt, tier=tier.value, live_tier_path=live_tier_path)
    except BackupAttestationError as exc:
        raise MigrationError(f"migration backup receipt authentication failed: {exc}") from exc
    if receipt.get("verdict") != "success":
        raise MigrationError(f"migration backup receipt is not a successful verification: {receipt_path}")
    artifact_inventory = _cached_backup_artifact_inventory(backup_root)
    file_evidence = {str(item["path"]): item for item in artifact_inventory if item.get("type") == "file"}
    manifest_evidence = file_evidence.get("manifest.json", {})
    if _json_int(receipt.get("manifest_size_bytes")) != _json_int(manifest_evidence.get("size_bytes")):
        raise MigrationError("migration backup receipt does not match manifest size")
    if receipt.get("manifest_sha256") != manifest_evidence.get("sha256"):
        raise MigrationError("migration backup receipt does not match manifest bytes")
    artifacts = _validated_receipt_artifacts(
        backup_root,
        payload,
        receipt,
        target_tier=tier.value,
        live_tier_path=live_tier_path,
        file_evidence=file_evidence,
    )
    artifact = artifacts.get(tier.value)
    if artifact is None:
        raise MigrationError(f"migration backup receipt does not include {tier.value}.db: {receipt_path}")
    _validate_blob_inventory(backup_root, payload, receipt, file_evidence=file_evidence)
    if receipt.get("artifact_inventory") != artifact_inventory:
        raise MigrationError("migration backup receipt does not match the closed artifact inventory")
    _validate_live_source_fingerprint(connection, artifact)
    return receipt_path


def _execute_migration_sql(conn: sqlite3.Connection, sql: str) -> None:
    statement = ""
    for line in sql.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            if statement.strip():
                conn.execute(statement)
            statement = ""
    if statement.strip():
        raise MigrationError("migration SQL ended with an incomplete statement")


def migrate_archive_tier(
    conn: sqlite3.Connection,
    tier: ArchiveTier,
    *,
    backup_manifest: Path | None,
) -> MigrationResult:
    """Apply additive migrations for one durable tier."""
    if tier not in DURABLE_MIGRATION_TIERS:
        raise MigrationError(f"{tier.value} tier does not support in-place migrations")
    _checkpoint_live_tier(conn)
    current_version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
    target_version = ARCHIVE_VERSION_BY_TIER[tier]
    if current_version == target_version:
        return MigrationResult(
            tier=tier,
            from_version=current_version,
            to_version=target_version,
            applied_versions=(),
        )
    if current_version == 0:
        raise MigrationError(f"{tier.value} tier is empty; initialize it fresh instead of migrating")
    if current_version > target_version:
        raise MigrationError(
            f"{tier.value} tier version {current_version} is newer than this runtime expects ({target_version})"
        )

    steps = tuple(step for step in _load_migrations(tier) if current_version < step.version <= target_version)
    expected_versions = tuple(range(current_version + 1, target_version + 1))
    actual_versions = tuple(step.version for step in steps)
    if actual_versions != expected_versions:
        raise MigrationError(
            f"{tier.value} migration chain is incomplete: expected {expected_versions}, found {actual_versions}"
        )
    requires_backup = any(step.requires_backup for step in steps)
    if requires_backup and backup_manifest is None:
        raise MigrationError(f"{tier.value} migration requires a verified backup manifest")
    if requires_backup:
        assert backup_manifest is not None
        validate_migration_backup_manifest(backup_manifest, tier, connection=conn)

    try:
        conn.execute("BEGIN IMMEDIATE")
        backup_receipt = (
            validate_migration_backup_manifest(backup_manifest, tier, connection=conn)
            if requires_backup and backup_manifest is not None
            else None
        )
        start_version = current_version
        applied: list[int] = []
        for step in steps:
            before = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
            if before != step.version - 1:
                raise MigrationError(
                    f"{tier.value} migration {step.name} expected version {step.version - 1}, found {before}"
                )
            _execute_migration_sql(conn, step.sql)
            # Saved-query migration now runs after v8 has installed the
            # definition-version column consumed by the canonical identity API.
            if tier is ArchiveTier.USER and step.version == 8:
                from polylogue.storage.sqlite.query_objects import migrate_saved_query_assertions

                migrate_saved_query_assertions(conn)
            conn.execute(f"PRAGMA user_version = {step.version}")
            applied.append(step.version)
        quick_check = conn.execute("PRAGMA quick_check").fetchone()
        if quick_check is None or str(quick_check[0]).lower() != "ok":
            raise MigrationError(f"{tier.value} migration quick_check failed: {quick_check!r}")
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise
    else:
        conn.commit()
    return MigrationResult(
        tier=tier,
        from_version=start_version,
        to_version=target_version,
        applied_versions=tuple(applied),
        backup_receipt=backup_receipt,
    )


__all__ = [
    "DURABLE_MIGRATION_TIERS",
    "MigrationError",
    "MigrationResult",
    "MigrationStep",
    "migrate_archive_tier",
    "validate_migration_backup_manifest",
]
