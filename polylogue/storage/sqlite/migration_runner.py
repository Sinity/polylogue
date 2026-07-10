"""Versioned additive migrations for durable archive tiers."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

DURABLE_MIGRATION_TIERS: frozenset[ArchiveTier] = frozenset({ArchiveTier.SOURCE, ArchiveTier.USER})
_MIGRATION_NAME_RE = re.compile(r"^(?P<version>\d{3,})_[a-z0-9_]+\.sql$")
_VERIFICATION_RECEIPT_FILE = "verification-receipt.json"
_VERIFICATION_RECEIPT_FORMAT = "polylogue-backup-verification-receipt-v1"


class MigrationError(RuntimeError):
    """Raised when a durable tier cannot be migrated safely."""


@dataclass(frozen=True, slots=True)
class MigrationStep:
    tier: ArchiveTier
    version: int
    name: str
    sql: str


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
        steps.append(
            MigrationStep(
                tier=tier,
                version=int(match.group("version")),
                name=item.name,
                sql=item.read_text(encoding="utf-8"),
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


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sqlite_user_version(path: Path) -> int:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
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


def _validate_tier_artifact(backup_root: Path, artifact: dict[str, object]) -> None:
    tier = artifact.get("tier")
    if not isinstance(tier, str) or tier not in {item.value for item in ArchiveTier}:
        raise MigrationError("migration backup receipt tier artifact has an invalid tier")
    filename = artifact.get("path")
    if not isinstance(filename, str) or filename != f"{tier}.db":
        raise MigrationError("migration backup receipt tier artifact path is not canonical")
    artifact_path = backup_root / filename
    if not artifact_path.exists():
        raise MigrationError(f"migration backup receipt references missing tier artifact: {artifact_path}")
    if _json_int(artifact.get("size_bytes")) != artifact_path.stat().st_size:
        raise MigrationError(f"migration backup tier artifact size mismatch: {filename}")
    if str(artifact.get("sha256")) != _sha256_file(artifact_path):
        raise MigrationError(f"migration backup tier artifact hash mismatch: {filename}")
    if _json_int(artifact.get("user_version")) != _sqlite_user_version(artifact_path):
        raise MigrationError(f"migration backup tier artifact user_version mismatch: {filename}")


def _validated_receipt_artifacts(
    backup_root: Path, manifest: dict[str, object], receipt: dict[str, object]
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
        _validate_tier_artifact(backup_root, artifact)
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


def _blob_inventory_file_evidence(backup_root: Path, manifest: dict[str, object]) -> dict[str, object]:
    inventory_file, inventory_path = _inventory_path(backup_root, manifest)
    if not inventory_path.exists():
        return {"path": inventory_file, "present": False, "size_bytes": 0, "sha256": None}
    return {
        "path": inventory_file,
        "present": True,
        "size_bytes": inventory_path.stat().st_size,
        "sha256": _sha256_file(inventory_path),
    }


def _current_blob_inventory(backup_root: Path, manifest: dict[str, object]) -> list[dict[str, object]]:
    _inventory_file, inventory_path = _inventory_path(backup_root, manifest)
    declared: dict[str, dict[str, object]] = {}
    if inventory_path.exists():
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
    blob_root = backup_root / "blob"
    if blob_root.exists():
        for blob_path in sorted(path for path in blob_root.rglob("*") if path.is_file()):
            if blob_path.is_symlink():
                raise MigrationError(f"migration backup blob inventory contains a symlink: {blob_path}")
            blob_hash = blob_path.parent.name + blob_path.name
            if (
                len(blob_hash) != 64
                or any(character not in "0123456789abcdef" for character in blob_hash)
                or blob_path.relative_to(backup_root) != Path("blob") / blob_hash[:2] / blob_hash[2:]
            ):
                raise MigrationError(f"migration backup blob inventory has a non-canonical blob path: {blob_path}")
            declared_item = declared.get(blob_hash, {})
            protection = declared_item.get("protection")
            rows.append(
                {
                    "blob_hash": blob_hash,
                    "path": str(blob_path.relative_to(backup_root)),
                    "size_bytes": blob_path.stat().st_size,
                    "sha256": _sha256_file(blob_path),
                    "protection": _json_str_list(protection),
                }
            )
    rows.sort(key=lambda item: str(item["blob_hash"]))
    return rows


def _validate_blob_inventory(backup_root: Path, manifest: dict[str, object], receipt: dict[str, object]) -> None:
    current = _current_blob_inventory(backup_root, manifest)
    if receipt.get("blob_inventory_file") != _blob_inventory_file_evidence(backup_root, manifest):
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
    if not manifest_path.exists():
        raise MigrationError(f"migration requires an existing backup manifest; missing {manifest_path}")
    payload = _load_json(manifest_path, label="manifest")
    if payload.get("format") != "polylogue-backup-v1":
        raise MigrationError(f"migration backup manifest has unsupported format: {manifest_path}")
    included = set(_json_str_list(payload.get("included_tiers")))
    if f"{tier.value}.db" not in included:
        raise MigrationError(f"migration backup manifest does not include {tier.value}.db: {manifest_path}")
    receipt_path = _receipt_path(manifest_path)
    if not receipt_path.exists():
        raise MigrationError(f"migration requires a successful backup verification receipt; missing {receipt_path}")
    receipt = _load_json(receipt_path, label="verification receipt")
    if receipt.get("format") != _VERIFICATION_RECEIPT_FORMAT:
        raise MigrationError(f"migration backup receipt has unsupported format: {receipt_path}")
    if receipt.get("verdict") != "success":
        raise MigrationError(f"migration backup receipt is not a successful verification: {receipt_path}")
    if _json_int(receipt.get("manifest_size_bytes")) != manifest_path.stat().st_size:
        raise MigrationError("migration backup receipt does not match manifest size")
    if receipt.get("manifest_sha256") != _sha256_file(manifest_path):
        raise MigrationError("migration backup receipt does not match manifest bytes")
    backup_root = manifest_path.parent
    artifacts = _validated_receipt_artifacts(backup_root, payload, receipt)
    artifact = artifacts.get(tier.value)
    if artifact is None:
        raise MigrationError(f"migration backup receipt does not include {tier.value}.db: {receipt_path}")
    _validate_blob_inventory(backup_root, payload, receipt)
    if connection is not None:
        _validate_live_source_fingerprint(connection, artifact)
    return receipt_path


def _execute_migration_sql(conn: sqlite3.Connection, sql: str) -> None:
    statements = [statement.strip() for statement in sql.split(";") if statement.strip()]
    for statement in statements:
        conn.execute(statement)


def migrate_archive_tier(
    conn: sqlite3.Connection,
    tier: ArchiveTier,
    *,
    backup_manifest: Path,
) -> MigrationResult:
    """Apply additive migrations for one durable tier."""
    if tier not in DURABLE_MIGRATION_TIERS:
        raise MigrationError(f"{tier.value} tier does not support in-place migrations")
    _checkpoint_live_tier(conn)
    validate_migration_backup_manifest(backup_manifest, tier, connection=conn)
    try:
        conn.execute("BEGIN IMMEDIATE")
        backup_receipt = validate_migration_backup_manifest(backup_manifest, tier, connection=conn)
        current_version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
        target_version = ARCHIVE_VERSION_BY_TIER[tier]
        if current_version == target_version:
            conn.rollback()
            return MigrationResult(
                tier=tier,
                from_version=current_version,
                to_version=target_version,
                applied_versions=(),
                backup_receipt=backup_receipt,
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

        start_version = current_version
        applied: list[int] = []
        for step in steps:
            before = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
            if before != step.version - 1:
                raise MigrationError(
                    f"{tier.value} migration {step.name} expected version {step.version - 1}, found {before}"
                )
            _execute_migration_sql(conn, step.sql)
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
