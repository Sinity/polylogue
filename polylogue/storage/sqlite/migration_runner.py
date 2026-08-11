"""Versioned additive migrations for durable archive tiers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import sqlite3
import stat
import time
import types
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import StrEnum
from importlib import resources
from pathlib import Path
from typing import Final, cast, get_args, get_origin, get_type_hints

from polylogue.storage.backup_attestation import (
    VERIFICATION_RECEIPT_FORMAT,
    BackupAttestationError,
    verify_verification_receipt,
)
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_LOGGER = logging.getLogger(__name__)

DURABLE_MIGRATION_TIERS: frozenset[ArchiveTier] = frozenset({ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT})
_MIGRATION_NAME_RE = re.compile(r"^(?P<version>\d{3,})_[a-z0-9_]+\.sql$")
_VERIFICATION_RECEIPT_FILE = "verification-receipt.json"
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
_ADDITIVE_NO_BACKUP_MARKER = "-- migration-safety: additive-no-backup"
_SQL_TRANSACTION_CONTROL_RE = re.compile(r"^(?:BEGIN|COMMIT|END|ROLLBACK|SAVEPOINT|RELEASE)\b", re.IGNORECASE)
DURABLE_CHANGE_TRAIN_FORMAT: Final = "polylogue.durable-change-train.v1"
DURABLE_MIGRATION_COLLISION_REPORT_FORMAT: Final = "polylogue.durable-migration-collisions.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


@dataclass(frozen=True, slots=True)
class DurableMigrationClaim:
    """One numbered durable migration's globally contended ownership key."""

    tier: ArchiveTier
    target_version: int
    slot: int
    path: str
    owner_ref: str
    sql_sha256: str
    requires_backup: bool

    @property
    def contention_key(self) -> tuple[str, int, int]:
        return (self.tier.value, self.target_version, self.slot)


@dataclass(frozen=True, slots=True)
class DurableMigrationCollision:
    """All claims that attempted to own one durable migration slot."""

    tier: ArchiveTier
    target_version: int
    slot: int
    claims: tuple[DurableMigrationClaim, ...]

    @property
    def contention_key(self) -> tuple[str, int, int]:
        return (self.tier.value, self.target_version, self.slot)


def durable_migration_claim_for_sql(
    tier: ArchiveTier,
    path: str | Path,
    sql: str,
    *,
    owner_ref: str | None = None,
) -> DurableMigrationClaim:
    """Build the shared p155 contention claim for one numbered SQL resource."""
    if tier not in DURABLE_MIGRATION_TIERS:
        raise MigrationError(f"{tier.value} tier cannot own a durable migration slot")
    migration_path = Path(path)
    match = _MIGRATION_NAME_RE.match(migration_path.name)
    if match is None:
        raise MigrationError(f"invalid numbered durable migration path: {migration_path}")
    version = int(match.group("version"))
    return DurableMigrationClaim(
        tier=tier,
        target_version=version,
        slot=version,
        path=str(migration_path),
        owner_ref=owner_ref or str(migration_path),
        sql_sha256=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
        requires_backup=_requires_migration_backup(sql),
    )


def find_durable_migration_collisions(
    claims: Iterable[DurableMigrationClaim],
) -> tuple[DurableMigrationCollision, ...]:
    """Return duplicate ``(tier, target version, numbered slot)`` ownership."""
    grouped: dict[tuple[str, int, int], list[DurableMigrationClaim]] = {}
    for claim in claims:
        grouped.setdefault(claim.contention_key, []).append(claim)
    collisions: list[DurableMigrationCollision] = []
    for (tier_value, target_version, slot), contenders in sorted(grouped.items()):
        if len(contenders) < 2:
            continue
        collisions.append(
            DurableMigrationCollision(
                tier=ArchiveTier(tier_value),
                target_version=target_version,
                slot=slot,
                claims=tuple(sorted(contenders, key=lambda item: (item.path, item.owner_ref))),
            )
        )
    return tuple(collisions)


def durable_migration_collision_report(claims: Iterable[DurableMigrationClaim]) -> dict[str, object]:
    """Emit the machine-readable collision report used by policy and admission."""
    collisions = find_durable_migration_collisions(claims)
    return {
        "format": DURABLE_MIGRATION_COLLISION_REPORT_FORMAT,
        "ok": not collisions,
        "collisions": [
            {
                "tier": collision.tier.value,
                "target_version": collision.target_version,
                "slot": collision.slot,
                "contention_key": [collision.tier.value, collision.target_version, collision.slot],
                "claims": [
                    {
                        "path": claim.path,
                        "owner_ref": claim.owner_ref,
                        "sql_sha256": claim.sql_sha256,
                        "requires_backup": claim.requires_backup,
                    }
                    for claim in collision.claims
                ],
                "recovery": "rebase and renumber every late rider onto the next unowned target slot",
            }
            for collision in collisions
        ],
    }


def _migration_package(tier: ArchiveTier) -> str:
    return f"polylogue.storage.sqlite.migrations.{tier.value}"


def _requires_migration_backup(sql: str) -> bool:
    """A migration opts out of the backup requirement only via a header directive.

    Substring matching would waive the backup requirement if the marker text
    ever appeared in a comment, SQL string literal, or later in the file --
    require it to be the file's first non-blank line instead.
    """
    first_nonblank = next((line.strip() for line in sql.splitlines() if line.strip()), "")
    return first_nonblank != _ADDITIVE_NO_BACKUP_MARKER


def _load_migrations(tier: ArchiveTier) -> tuple[MigrationStep, ...]:
    if tier not in DURABLE_MIGRATION_TIERS:
        return ()
    try:
        files = resources.files(_migration_package(tier))
    except ModuleNotFoundError:
        return ()
    steps: list[MigrationStep] = []
    claims: list[DurableMigrationClaim] = []
    for item in sorted(files.iterdir(), key=lambda path: path.name):
        if _MIGRATION_NAME_RE.match(item.name) is None:
            continue
        sql = item.read_text(encoding="utf-8")
        claim = durable_migration_claim_for_sql(
            tier,
            item.name,
            sql,
            owner_ref=f"polylogue/storage/sqlite/migrations/{tier.value}/{item.name}",
        )
        claims.append(claim)
        steps.append(
            MigrationStep(
                tier=tier,
                version=claim.target_version,
                name=item.name,
                sql=sql,
                requires_backup=claim.requires_backup,
            )
        )
    collisions = find_durable_migration_collisions(claims)
    if collisions:
        details = "; ".join(
            f"{collision.tier.value}/v{collision.target_version}/slot-{collision.slot}: "
            + ", ".join(f"{claim.path} ({claim.owner_ref})" for claim in collision.claims)
            for collision in collisions
        )
        raise MigrationError("durable migration ownership collision; rebase and renumber late riders: " + details)
    # Above the adoption floors, package resources must carry the matching
    # deterministic <slot>.train.json authority. Keep discovery here at the
    # existing migration choke point so every caller, including the daemon's
    # startup probe and the maintenance CLI, observes the same admission gate.
    from polylogue.storage.sqlite.durable_change_train import validate_durable_migration_sidecars

    try:
        validate_durable_migration_sidecars(tier, tuple((step.name, step.sql) for step in steps))
    except Exception as exc:
        if isinstance(exc, MigrationError):
            raise
        raise MigrationError(str(exc)) from exc
    return tuple(steps)


def _prepare_fresh_connection_for_target(
    connection: sqlite3.Connection,
    tier: ArchiveTier,
    target_version: int,
) -> None:
    """Project current canonical DDL to an earlier additive migration slot.

    The shipped bootstrap is necessarily at the newest version.  A durable
    archive can be paused between numbered trains, so parity for one historical
    step must exclude objects explicitly owned by later train riders.  Existing
    object rewrites remain in the comparison and fail closed.
    """
    runtime_target = ARCHIVE_VERSION_BY_TIER[tier]
    if target_version >= runtime_target:
        return
    steps = _load_migrations(tier)
    from polylogue.storage.sqlite.durable_change_train import validate_durable_migration_sidecars

    sidecars = validate_durable_migration_sidecars(tier, tuple((step.name, step.sql) for step in steps))
    future_refs = {
        schema_object
        for sidecar in sidecars
        if sidecar.slot > target_version
        for rider in sidecar.train.riders
        for schema_object in rider.schema_objects
    }
    drop_order = {"index": 0, "trigger": 0, "view": 0, "table": 1}
    for schema_object in sorted(
        future_refs,
        key=lambda item: (drop_order.get(item.partition(":")[0], 2), item),
    ):
        object_type, separator, object_name = schema_object.partition(":")
        if not separator or object_type not in drop_order:
            continue
        quoted_name = '"' + object_name.replace('"', '""') + '"'
        connection.execute(f"DROP {object_type.upper()} IF EXISTS {quoted_name}")
    if future_refs:
        connection.execute(f"PRAGMA user_version = {target_version}")
        connection.commit()


def durable_migration_claims(tier: ArchiveTier) -> tuple[DurableMigrationClaim, ...]:
    """Return the exact claims consumed by the shipped migration runner."""
    return tuple(
        DurableMigrationClaim(
            tier=step.tier,
            target_version=step.version,
            slot=step.version,
            path=step.name,
            owner_ref=f"polylogue/storage/sqlite/migrations/{tier.value}/{step.name}",
            sql_sha256=hashlib.sha256(step.sql.encode("utf-8")).hexdigest(),
            requires_backup=step.requires_backup,
        )
        for step in _load_migrations(tier)
    )


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
    live_tier_path: Path | None,
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


def _validate_backup_manifest_covers_tier(
    path: Path, tier: ArchiveTier, *, connection: sqlite3.Connection, require_attestation: bool
) -> Path:
    """Validate that ``path`` has a successful backup verification receipt.

    ``require_attestation`` gates the cryptographic HMAC attestation check.
    Attestations are only ever minted for durable tiers (source, user) by
    ``daemon/backup.py``'s ``_write_successful_verification_receipt`` -- a
    derived tier (index, embeddings) can never carry one, by design, so
    requiring it for those tiers would make backup-manifest validation
    permanently unsatisfiable rather than merely strict. Callers protecting a
    mutation against a derived tier (e.g. the agent-meta-sidecar purge, which
    deletes rows from index.db) still get every other guarantee here --
    manifest/receipt shape, tier inclusion, and a byte-exact live fingerprint
    recomputed from the current on-disk file -- just not the attestation,
    which durable-tier migrations (``migrate_archive_tier``) still require.
    """
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
    live_tier_path = _connection_main_path(connection).resolve(strict=False)
    if require_attestation:
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


def validate_migration_backup_manifest(
    path: Path, tier: ArchiveTier, *, connection: sqlite3.Connection | None = None
) -> Path:
    """Validate a backup manifest for a durable-tier migration (requires attestation)."""
    if connection is None:
        raise MigrationError("migration backup receipt authentication requires the live tier connection")
    return _validate_backup_manifest_covers_tier(path, tier, connection=connection, require_attestation=True)


def validate_migration_backup_live_fingerprint(
    path: Path, tier: ArchiveTier, *, connection: sqlite3.Connection
) -> Path:
    """Recheck receipt authority and the live tier with a cached inventory.

    Callers run ``validate_migration_backup_manifest`` before taking the live
    tier lock. Once ownership is held, this helper repeats the local HMAC and
    manifest binding checks, then validates the complete cached backup
    inventory. A stat change invalidates the process cache and re-hashes the
    tree, so source.db/blob mutations after the precheck are rejected without
    repeating the expensive scan when the tree is unchanged.
    """

    if tier not in DURABLE_MIGRATION_TIERS:
        raise MigrationError(f"{tier.value} tier is not a durable migration tier")
    manifest_path = _backup_manifest_path(path)
    if not manifest_path.exists() and not manifest_path.is_symlink():
        raise MigrationError(f"migration requires an existing backup manifest; missing {manifest_path}")
    backup_root = manifest_path.parent
    _require_real_backup_directory(backup_root, label="backup root")
    _require_regular_backup_artifact(manifest_path, backup_root=backup_root, label="backup manifest")
    manifest = _load_json(manifest_path, label="manifest")
    if manifest.get("format") != "polylogue-backup-v1":
        raise MigrationError(f"migration backup manifest has unsupported format: {manifest_path}")
    if f"{tier.value}.db" not in _json_str_list(manifest.get("included_tiers")):
        raise MigrationError(f"migration backup manifest does not include {tier.value}.db: {manifest_path}")
    receipt_path = _receipt_path(manifest_path)
    if not receipt_path.exists() and not receipt_path.is_symlink():
        raise MigrationError(f"migration requires a successful backup verification receipt; missing {receipt_path}")
    _require_regular_backup_artifact(receipt_path, backup_root=backup_root, label="backup verification receipt")
    receipt = _load_json(receipt_path, label="verification receipt")
    if receipt.get("format") != VERIFICATION_RECEIPT_FORMAT:
        raise MigrationError(f"migration backup receipt has unsupported format: {receipt_path}")
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
        manifest,
        receipt,
        target_tier=tier.value,
        live_tier_path=live_tier_path,
        file_evidence=file_evidence,
    )
    artifact = artifacts.get(tier.value)
    if artifact is None:
        raise MigrationError(f"migration backup receipt does not include {tier.value}.db: {receipt_path}")
    _validate_blob_inventory(backup_root, manifest, receipt, file_evidence=file_evidence)
    if receipt.get("artifact_inventory") != artifact_inventory:
        raise MigrationError("migration backup receipt does not match the closed artifact inventory")
    _validate_live_source_fingerprint(connection, artifact)
    return receipt_path


def validate_full_evidence_backup_for_audit_adoption(path: Path, *, archive_root: Path) -> tuple[Path, Path]:
    """Authorize creation of a missing audit tier in an established archive.

    Unlike a normal tier migration there is no live ``audit.db`` connection to
    attest.  The route therefore requires the complete pre-audit file set,
    validates the existing source/user attestations, and compares every
    retained tier's recorded source fingerprint with the still-offline live
    archive.  This is intentionally stricter than ordinary migration backup
    validation: an adoption is only safe when the backup is full evidence for
    this exact established archive, not merely a restorable subset.
    """
    manifest_path = _backup_manifest_path(path)
    if not manifest_path.exists() and not manifest_path.is_symlink():
        raise MigrationError(f"audit adoption requires an existing backup manifest; missing {manifest_path}")
    backup_root = manifest_path.parent
    _require_real_backup_directory(backup_root, label="backup root")
    _require_regular_backup_artifact(manifest_path, backup_root=backup_root, label="backup manifest")
    manifest = _load_json(manifest_path, label="manifest")
    if manifest.get("format") != "polylogue-backup-v1" or manifest.get("profile") != "full_evidence":
        raise MigrationError("audit adoption requires a verified full_evidence backup")
    included = set(_json_str_list(manifest.get("included_tiers")))
    required_tiers = {"source", "index", "embeddings", "user"}
    permitted_tiers = required_tiers | {"ops"}
    included_tiers = {name.removesuffix(".db") for name in included}
    if (
        not required_tiers.issubset(included_tiers)
        or included_tiers - permitted_tiers
        or len(included_tiers) != len(included)
    ):
        raise MigrationError("audit adoption backup must contain every non-optional established tier and no audit tier")
    receipt_path = _receipt_path(manifest_path)
    if not receipt_path.exists() and not receipt_path.is_symlink():
        raise MigrationError(
            f"audit adoption requires a successful backup verification receipt; missing {receipt_path}"
        )
    _require_regular_backup_artifact(receipt_path, backup_root=backup_root, label="backup verification receipt")
    receipt = _load_json(receipt_path, label="verification receipt")
    if receipt.get("format") != VERIFICATION_RECEIPT_FORMAT or receipt.get("verdict") != "success":
        raise MigrationError("audit adoption requires a successful backup verification receipt")
    archive_root = archive_root.resolve()
    try:
        if backup_root.samefile(archive_root):
            raise MigrationError("audit adoption backup root aliases the live archive root")
    except OSError as exc:
        raise MigrationError("cannot compare audit adoption backup root with the live archive") from exc
    for authority_tier in ("source", "user"):
        try:
            verify_verification_receipt(
                receipt,
                tier=authority_tier,
                live_tier_path=archive_root / f"{authority_tier}.db",
            )
        except BackupAttestationError as exc:
            raise MigrationError(f"audit adoption backup authentication failed: {exc}") from exc
    artifact_inventory = _cached_backup_artifact_inventory(backup_root)
    file_evidence = {str(item["path"]): item for item in artifact_inventory if item.get("type") == "file"}
    manifest_evidence = file_evidence.get("manifest.json", {})
    if _json_int(receipt.get("manifest_size_bytes")) != _json_int(manifest_evidence.get("size_bytes")):
        raise MigrationError("audit adoption backup receipt does not match manifest size")
    if receipt.get("manifest_sha256") != manifest_evidence.get("sha256"):
        raise MigrationError("audit adoption backup receipt does not match manifest bytes")
    artifacts = _validated_receipt_artifacts(
        backup_root,
        manifest,
        receipt,
        target_tier="audit",
        live_tier_path=archive_root / "audit.db",
        file_evidence=file_evidence,
    )
    _validate_blob_inventory(backup_root, manifest, receipt, file_evidence=file_evidence)
    if receipt.get("artifact_inventory") != artifact_inventory:
        raise MigrationError("audit adoption backup receipt does not match the closed artifact inventory")
    for tier in sorted(included_tiers):
        live_path = archive_root / f"{tier}.db"
        artifact = artifacts[tier]
        artifact_path = backup_root / f"{tier}.db"
        if not live_path.is_file():
            raise MigrationError(f"audit adoption live tier is missing: {live_path}")
        try:
            if artifact_path.samefile(live_path):
                raise MigrationError(f"audit adoption backup tier artifact aliases the live tier: {tier}.db")
        except OSError as exc:
            raise MigrationError(f"cannot compare audit adoption backup tier with live tier: {tier}.db") from exc
        fingerprint = artifact.get("source_fingerprint")
        if not isinstance(fingerprint, dict):
            raise MigrationError(f"audit adoption backup lacks a live source fingerprint for {tier}.db")
        if Path(str(fingerprint.get("path") or "")).resolve(strict=False) != live_path.resolve(strict=False):
            raise MigrationError(f"audit adoption backup belongs to a different archive tier: {tier}.db")
        wal_path = live_path.with_name(f"{live_path.name}-wal")
        if wal_path.exists() and wal_path.stat().st_size:
            raise MigrationError(f"audit adoption backup has live WAL divergence for {tier}.db")
        if _json_int(fingerprint.get("size_bytes")) != live_path.stat().st_size:
            raise MigrationError(f"audit adoption backup is stale for {tier}.db")
        if str(fingerprint.get("sha256")) != _sha256_file(live_path):
            raise MigrationError(f"audit adoption backup is stale for {tier}.db")
        if _json_int(fingerprint.get("user_version")) != _sqlite_user_version(live_path):
            raise MigrationError(f"audit adoption backup is stale for {tier}.db")
    return manifest_path, receipt_path


def validate_full_evidence_backup_for_adopted_audit_restore(
    path: Path, *, archive_root: Path, allow_source_continuity_rebind: bool = False
) -> tuple[Path, Path]:
    """Authorize replacing adopted ``audit.db`` from one exact backup.

    The audit file may be absent or unreadable, so its stable path authority is
    verified without opening it. Every other captured tier must still match
    the scratch-verified full-evidence snapshot byte for byte, except that a
    retry after continuity promotion may differ only in source.db's control
    row.
    """
    manifest_path = _backup_manifest_path(path)
    if not manifest_path.exists() and not manifest_path.is_symlink():
        raise MigrationError(f"adopted-audit restore requires an existing backup manifest; missing {manifest_path}")
    backup_root = manifest_path.parent
    _require_real_backup_directory(backup_root, label="backup root")
    _require_regular_backup_artifact(manifest_path, backup_root=backup_root, label="backup manifest")
    manifest = _load_json(manifest_path, label="manifest")
    if manifest.get("format") != "polylogue-backup-v1" or manifest.get("profile") != "full_evidence":
        raise MigrationError("adopted-audit restore requires a verified full_evidence backup")
    included = set(_json_str_list(manifest.get("included_tiers")))
    required_tiers = {"source", "index", "embeddings", "user", "audit"}
    permitted_tiers = required_tiers | {"ops"}
    included_tiers = {name.removesuffix(".db") for name in included}
    if (
        not required_tiers.issubset(included_tiers)
        or included_tiers - permitted_tiers
        or len(included_tiers) != len(included)
    ):
        raise MigrationError("adopted-audit restore backup must contain every non-optional tier including audit")
    receipt_path = _receipt_path(manifest_path)
    if not receipt_path.exists() and not receipt_path.is_symlink():
        raise MigrationError(
            f"adopted-audit restore requires a successful backup verification receipt; missing {receipt_path}"
        )
    _require_regular_backup_artifact(receipt_path, backup_root=backup_root, label="backup verification receipt")
    receipt = _load_json(receipt_path, label="verification receipt")
    if receipt.get("format") != VERIFICATION_RECEIPT_FORMAT or receipt.get("verdict") != "success":
        raise MigrationError("adopted-audit restore requires a successful backup verification receipt")
    archive_root = archive_root.resolve()
    try:
        if backup_root.samefile(archive_root):
            raise MigrationError("adopted-audit restore backup root aliases the live archive root")
    except OSError as exc:
        raise MigrationError("cannot compare adopted-audit restore backup root with the live archive") from exc
    for authority_tier in ("source", "user", "audit"):
        try:
            verify_verification_receipt(
                receipt, tier=authority_tier, live_tier_path=archive_root / f"{authority_tier}.db"
            )
        except BackupAttestationError as exc:
            raise MigrationError(f"adopted-audit restore backup authentication failed: {exc}") from exc
    artifact_inventory = _cached_backup_artifact_inventory(backup_root)
    file_evidence = {str(item["path"]): item for item in artifact_inventory if item.get("type") == "file"}
    manifest_evidence = file_evidence.get("manifest.json", {})
    if _json_int(receipt.get("manifest_size_bytes")) != _json_int(manifest_evidence.get("size_bytes")):
        raise MigrationError("adopted-audit restore receipt does not match manifest size")
    if receipt.get("manifest_sha256") != manifest_evidence.get("sha256"):
        raise MigrationError("adopted-audit restore receipt does not match manifest bytes")
    artifacts = _validated_receipt_artifacts(
        backup_root, manifest, receipt, target_tier="audit", live_tier_path=None, file_evidence=file_evidence
    )
    _validate_blob_inventory(backup_root, manifest, receipt, file_evidence=file_evidence)
    if receipt.get("artifact_inventory") != artifact_inventory:
        raise MigrationError("adopted-audit restore receipt does not match the closed artifact inventory")
    for tier in sorted(included_tiers):
        live_path = archive_root / f"{tier}.db"
        artifact_path = backup_root / f"{tier}.db"
        if live_path.is_file():
            try:
                if artifact_path.samefile(live_path):
                    raise MigrationError(f"adopted-audit restore backup tier artifact aliases the live tier: {tier}.db")
            except OSError as exc:
                raise MigrationError(
                    f"cannot compare adopted-audit restore backup tier with live tier: {tier}.db"
                ) from exc
        if tier not in {"source", "user"}:
            continue
        fingerprint = artifacts[tier].get("source_fingerprint")
        if not isinstance(fingerprint, dict):
            raise MigrationError(f"adopted-audit restore backup lacks a live source fingerprint for {tier}.db")
        if Path(str(fingerprint.get("path") or "")).resolve(strict=False) != live_path.resolve(strict=False):
            raise MigrationError(f"adopted-audit restore backup belongs to a different archive tier: {tier}.db")
        if not live_path.is_file():
            raise MigrationError(f"adopted-audit restore live tier is missing: {live_path}")
        wal_path = live_path.with_name(f"{live_path.name}-wal")
        if wal_path.exists() and wal_path.stat().st_size:
            raise MigrationError(f"adopted-audit restore has live WAL divergence for {tier}.db")
        if tier == "source" and allow_source_continuity_rebind:
            if _json_int(fingerprint.get("user_version")) != _sqlite_user_version(live_path):
                raise MigrationError("adopted-audit restore backup is stale for source.db")
            _validate_source_continuity_rebind_delta(artifact_path, live_path)
            continue
        if _json_int(fingerprint.get("size_bytes")) != live_path.stat().st_size:
            raise MigrationError(f"adopted-audit restore backup is stale for {tier}.db")
        if str(fingerprint.get("sha256")) != _sha256_file(live_path):
            raise MigrationError(f"adopted-audit restore backup is stale for {tier}.db")
        if _json_int(fingerprint.get("user_version")) != _sqlite_user_version(live_path):
            raise MigrationError(f"adopted-audit restore backup is stale for {tier}.db")
    return manifest_path, receipt_path


def _validate_source_continuity_rebind_delta(backup_path: Path, live_path: Path) -> None:
    """Allow a retrying restore to differ only in the source continuity table."""

    try:
        with sqlite3.connect(f"{live_path.resolve(strict=True).as_uri()}?mode=ro", uri=True) as connection:
            connection.execute(
                "ATTACH DATABASE ? AS backup_source", (f"{backup_path.resolve(strict=True).as_uri()}?mode=ro",)
            )
            schema_sql = """
                SELECT type, name, tbl_name, sql
                FROM {schema}.sqlite_schema
                WHERE name NOT LIKE 'sqlite_%'
                  AND name != 'audit_continuity_control'
                ORDER BY type, name
            """
            live_schema = connection.execute(schema_sql.format(schema="main")).fetchall()
            backup_schema = connection.execute(schema_sql.format(schema="backup_source")).fetchall()
            if live_schema != backup_schema:
                raise MigrationError("adopted-audit restore backup is stale for source.db")
            table_names = [str(row[1]) for row in live_schema if row[0] == "table"]
            for table_name in table_names:
                quoted = _quote_sqlite_identifier(table_name)
                live_count = int(connection.execute(f"SELECT COUNT(*) FROM main.{quoted}").fetchone()[0])
                backup_count = int(connection.execute(f"SELECT COUNT(*) FROM backup_source.{quoted}").fetchone()[0])
                if live_count != backup_count:
                    raise MigrationError("adopted-audit restore backup is stale for source.db")
                for left, right in (("main", "backup_source"), ("backup_source", "main")):
                    differs = connection.execute(
                        f"SELECT 1 FROM (SELECT * FROM {left}.{quoted} EXCEPT SELECT * FROM {right}.{quoted}) LIMIT 1"
                    ).fetchone()
                    if differs is not None:
                        raise MigrationError("adopted-audit restore backup is stale for source.db")
    except sqlite3.DatabaseError as exc:
        raise MigrationError("cannot compare adopted-audit restore source continuity delta") from exc


def validate_backup_manifest_covers_derived_tier(
    path: Path, tier: ArchiveTier, *, connection: sqlite3.Connection
) -> Path:
    """Validate a backup manifest covers a derived tier (index, embeddings) at its live fingerprint.

    For use by actuators that mutate a derived tier and want backup coverage
    as a safety net before doing so, without requiring the cryptographic
    attestation that only durable tiers (source, user) ever carry -- see
    ``_validate_backup_manifest_covers_tier``'s docstring.
    """
    if tier in DURABLE_MIGRATION_TIERS:
        raise MigrationError(f"{tier.value} is a durable tier; use validate_migration_backup_manifest instead")
    return _validate_backup_manifest_covers_tier(path, tier, connection=connection, require_attestation=False)


def _execute_migration_sql(conn: sqlite3.Connection, sql: str) -> None:
    statement = ""
    for line in sql.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            if statement.strip():
                leading_sql = re.sub(
                    r"(?is)^(?:\s|--[^\n]*(?:\n|$)|/\*.*?\*/)*",
                    "",
                    statement,
                )
                if _SQL_TRANSACTION_CONTROL_RE.match(leading_sql) is not None:
                    raise MigrationError(
                        "durable migration SQL must not control the existing transaction: "
                        f"{leading_sql.split(None, 1)[0].upper()}"
                    )
                conn.execute(statement)
                if not conn.in_transaction:
                    raise MigrationError("durable migration SQL escaped the existing transaction")
            statement = ""
    if statement.strip():
        raise MigrationError("migration SQL ended with an incomplete statement")


def _pending_migration_steps(
    conn: sqlite3.Connection,
    tier: ArchiveTier,
    *,
    current_version: int,
    target_version: int,
) -> tuple[MigrationStep, ...]:
    steps = tuple(step for step in _load_migrations(tier) if current_version < step.version <= target_version)
    expected_versions = tuple(range(current_version + 1, target_version + 1))
    actual_versions = tuple(step.version for step in steps)
    if actual_versions != expected_versions:
        raise MigrationError(
            f"{tier.value} migration chain is incomplete: expected {expected_versions}, found {actual_versions}"
        )
    return steps


def migrate_archive_tier(
    conn: sqlite3.Connection,
    tier: ArchiveTier,
    *,
    backup_manifest: Path | None,
    target_version: int | None = None,
) -> MigrationResult:
    """Apply additive migrations for one durable tier."""
    if tier not in DURABLE_MIGRATION_TIERS:
        raise MigrationError(f"{tier.value} tier does not support in-place migrations")
    _checkpoint_live_tier(conn)
    runtime_target_version = ARCHIVE_VERSION_BY_TIER[tier]
    if target_version is None:
        target_version = runtime_target_version
    if target_version > runtime_target_version:
        raise MigrationError(
            f"{tier.value} migration target {target_version} is newer than this runtime expects "
            f"({runtime_target_version})"
        )

    # Lock-free precheck: fail fast (no wasted write-lock acquisition) when a
    # backup manifest is required but missing. This read is intentionally not
    # authoritative -- a concurrent migrate_archive_tier call could change the
    # version before this one acquires BEGIN IMMEDIATE below, so every value
    # computed here is re-derived from a fresh read once the lock is held.
    precheck_version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
    if precheck_version == target_version:
        return MigrationResult(
            tier=tier,
            from_version=precheck_version,
            to_version=target_version,
            applied_versions=(),
        )
    if precheck_version == 0:
        raise MigrationError(f"{tier.value} tier is empty; initialize it fresh instead of migrating")
    if precheck_version > target_version:
        raise MigrationError(
            f"{tier.value} tier version {precheck_version} is newer than this runtime expects ({target_version})"
        )
    precheck_steps = _pending_migration_steps(
        conn, tier, current_version=precheck_version, target_version=target_version
    )
    from polylogue.storage.sqlite.durable_change_train import validate_durable_migration_sidecars

    validate_durable_migration_sidecars(tier, tuple((step.name, step.sql) for step in _load_migrations(tier)))
    precheck_requires_backup = any(step.requires_backup for step in precheck_steps)
    if precheck_requires_backup and backup_manifest is None:
        raise MigrationError(f"{tier.value} migration requires a verified backup manifest")
    if precheck_requires_backup:
        # Baseline validation before acquiring the write lock. The paired
        # post-lock call below re-validates with the same connection;
        # _validate_live_source_fingerprint rejects a nonempty WAL, so a
        # write that lands on the live tier between this call and BEGIN
        # IMMEDIATE is caught as "changed before the migration lock" instead
        # of migrating over data the verified backup never covered.
        assert backup_manifest is not None
        validate_migration_backup_manifest(backup_manifest, tier, connection=conn)

    # Durable-tier migrations rebuild tables via create-copy-drop-rename;
    # with foreign_keys ON, DROP TABLE on a referenced parent performs an
    # implicit DELETE FROM and fires ON DELETE CASCADE into every referencing
    # table. Production migration connections are bare sqlite3.connect (FK
    # OFF by SQLite default), but that safety was implicit -- enforce it by
    # construction here, outside the transaction (the pragma is a no-op
    # inside one), and restore the caller's state afterwards.
    foreign_keys_were_on = bool(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    if foreign_keys_were_on:
        conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("BEGIN IMMEDIATE")
        # Authoritative re-read: a concurrent migration may have advanced (or
        # completed) the tier between the precheck above and this lock
        # acquisition. Recomputing here instead of trusting the precheck
        # avoids failing the per-step version check below with a confusing
        # "expected version N, found M" instead of the correct no-op result.
        current_version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
        if current_version == target_version:
            conn.rollback()
            return MigrationResult(
                tier=tier,
                from_version=current_version,
                to_version=target_version,
                applied_versions=(),
            )
        if current_version > target_version:
            raise MigrationError(
                f"{tier.value} tier version {current_version} is newer than this runtime expects ({target_version})"
            )
        steps = _pending_migration_steps(conn, tier, current_version=current_version, target_version=target_version)
        all_steps = _load_migrations(tier)
        all_sidecars = validate_durable_migration_sidecars(
            tier,
            tuple((step.name, step.sql) for step in all_steps),
        )
        pending_versions = {step.version for step in steps}
        sidecars = tuple(sidecar for sidecar in all_sidecars if sidecar.slot in pending_versions)
        requires_backup = any(step.requires_backup for step in steps)
        if requires_backup and backup_manifest is None:
            raise MigrationError(f"{tier.value} migration requires a verified backup manifest")
        backup_receipt = (
            validate_migration_backup_manifest(backup_manifest, tier, connection=conn)
            if requires_backup and backup_manifest is not None
            else None
        )
        start_version = current_version
        pre_transaction_evidence = capture_durable_database_evidence(conn, tier) if sidecars else None
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
            if not conn.in_transaction:
                raise MigrationError("durable migration SQL escaped the existing transaction")
            applied.append(step.version)
        quick_check = conn.execute("PRAGMA quick_check").fetchone()
        if quick_check is None or str(quick_check[0]).lower() != "ok":
            raise MigrationError(f"{tier.value} migration quick_check failed: {quick_check!r}")
        if sidecars:
            assert pre_transaction_evidence is not None
            if not conn.in_transaction:
                raise MigrationError("durable migration validation lost the existing transaction")
            post_transaction_evidence = capture_durable_database_evidence(conn, tier)
            foreign_key_errors = tuple(conn.execute("PRAGMA foreign_key_check").fetchall())
            if foreign_key_errors:
                raise MigrationError(f"{tier.value} migration foreign_key_check failed: {foreign_key_errors!r}")
            drop_constraints = tuple(
                constraint for sidecar in sidecars for constraint in sidecar.train.drop_constraints
            )
            row_change_allowances = tuple(
                allowance for sidecar in sidecars for allowance in sidecar.train.row_change_allowances
            )
            row_parity = prove_durable_row_parity(
                pre_transaction_evidence,
                post_transaction_evidence,
                drop_constraints=drop_constraints,
                row_change_allowances=row_change_allowances,
            )
            if not row_parity.ok:
                raise MigrationError(
                    "durable migration row parity failed: " + "; ".join(row_parity.unauthorized_changes)
                )
            with closing(sqlite3.connect(":memory:")) as fresh_connection:
                fresh_connection.execute("PRAGMA foreign_keys = ON")
                fresh_connection.executescript(ARCHIVE_DDL_BY_TIER[tier])
                fresh_connection.execute(f"PRAGMA user_version = {target_version}")
                _prepare_fresh_connection_for_target(fresh_connection, tier, target_version)
                fresh_connection.commit()
                fresh_parity = prove_durable_fresh_ddl_parity(
                    tier,
                    target_version,
                    migrated_connection=conn,
                    fresh_connection=fresh_connection,
                    evidence_ref=f"proof:fresh-ddl-before-commit:{tier.value}:v{target_version}",
                )
            if not fresh_parity.matches:
                raise MigrationError(
                    f"{tier.value} migration canonical DDL parity failed: "
                    f"missing={fresh_parity.missing_objects}, unexpected={fresh_parity.unexpected_objects}, "
                    f"changed={fresh_parity.changed_objects}"
                )
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        if foreign_keys_were_on:
            conn.execute("PRAGMA foreign_keys = ON")
        raise
    else:
        conn.commit()
        if foreign_keys_were_on:
            conn.execute("PRAGMA foreign_keys = ON")
    return MigrationResult(
        tier=tier,
        from_version=start_version,
        to_version=target_version,
        applied_versions=tuple(applied),
        backup_receipt=backup_receipt,
    )


class DurableChangeTrainState(StrEnum):
    """Persisted lifecycle states for one durable schema-change train."""

    DECLARED = "declared"
    ADMITTED = "admitted"
    RESERVED = "reserved"
    BACKUP_AUTHORIZED = "backup-authorized"
    APPLIED = "applied"
    PROVEN = "proven"
    RELEASED = "released"
    FAILED = "failed"


class DurableFailureClassification(StrEnum):
    """Recovery branch selected from the observed durable-tier version."""

    ROLLED_BACK_TO_CURRENT = "rolled-back-to-current"
    COMMITTED_TARGET_UNPROVEN = "committed-target-unproven"
    INDETERMINATE = "indeterminate"


class DurableChangeTrainError(MigrationError):
    """Raised when durable change-train authority or evidence is invalid."""


class DurableChangeTrainApplyError(DurableChangeTrainError):
    """Apply failure carrying the machine-readable failed train manifest."""

    def __init__(self, message: str, *, failed_train: DurableChangeTrain) -> None:
        super().__init__(message)
        self.failed_train = failed_train


class DurableChangeTrainRecoveryError(DurableChangeTrainError):
    """Unsafe recovery branch carrying the machine-readable failed train."""

    def __init__(self, message: str, *, failed_train: DurableChangeTrain) -> None:
        super().__init__(message)
        self.failed_train = failed_train


@dataclass(frozen=True, slots=True)
class DurableRuntimeConsumer:
    """One production consumer that proves a rider is runtime-complete."""

    consumer_id: str
    production_ref: str
    behavior_proof_ref: str
    roles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DurableChangeRider:
    """One schema-and-runtime unit admitted onto a train before reservation."""

    rider_id: str
    owner_ref: str
    schema_objects: tuple[str, ...]
    runtime_consumers: tuple[DurableRuntimeConsumer, ...]
    behavior_proof_refs: tuple[str, ...]
    after_rider_ids: tuple[str, ...] = ()
    trust_floor_exception_ref: str | None = None


@dataclass(frozen=True, slots=True)
class DurableOrderingConstraint:
    """Explicit cross-rider ordering evidence."""

    before_rider_id: str
    after_rider_id: str
    evidence_ref: str


@dataclass(frozen=True, slots=True)
class DurableDropConstraint:
    """Authority required before a durable object may disappear."""

    object_ref: str
    after_rider_ids: tuple[str, ...]
    copy_forward_proof_ref: str
    consent_ref: str


@dataclass(frozen=True, slots=True)
class DurableRowChangeAllowance:
    """An explicitly reviewed data movement exception to row-count parity."""

    table: str
    reason_ref: str
    expected_delta: int | None = None


@dataclass(frozen=True, slots=True)
class DurableSchemaObjectEvidence:
    """Canonical structural fingerprint for one non-internal SQLite object."""

    object_type: str
    name: str
    table_name: str
    definition_sha256: str

    @property
    def object_ref(self) -> str:
        return f"{self.object_type}:{self.name}"


@dataclass(frozen=True, slots=True)
class DurableSchemaInventory:
    """Canonical inventory used by fresh-DDL and migrated-schema parity."""

    objects: tuple[DurableSchemaObjectEvidence, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class DurableFreshDDLParityProof:
    """Comparison between an upgraded database and a fresh canonical create."""

    tier: ArchiveTier
    target_version: int
    migrated_version: int
    fresh_version: int
    migrated_inventory_sha256: str
    fresh_inventory_sha256: str
    missing_objects: tuple[str, ...]
    unexpected_objects: tuple[str, ...]
    changed_objects: tuple[str, ...]
    evidence_ref: str
    matches: bool


@dataclass(frozen=True, slots=True)
class DurableWriterReservation:
    """Manifest binding to the archive-root writer lease and stopped daemon."""

    reservation_id: str
    owner_ref: str
    archive_root: str
    tier_path: str
    daemon_stopped_evidence_ref: str
    single_writer_evidence_ref: str
    reserved_at_ms: int
    active: bool
    released_at_ms: int | None = None
    release_evidence_ref: str | None = None


@dataclass(frozen=True, slots=True)
class DurableBackupAuthorization:
    """Exact backup authority bound to the live tier before apply."""

    mode: str
    live_tier_path: str
    live_user_version: int
    manifest_path: str | None
    manifest_sha256: str | None
    receipt_path: str | None
    receipt_sha256: str | None
    evidence_ref: str
    authorized_at_ms: int


@dataclass(frozen=True, slots=True)
class DurableDatabaseEvidence:
    """Integrity, version, schema, and row census for one durable tier."""

    tier: ArchiveTier
    user_version: int
    quick_check: tuple[str, ...]
    schema_inventory_sha256: str
    row_counts: tuple[tuple[str, int], ...]
    archive_identity_digest: str
    content_sha256: str
    observed_at_ms: int


@dataclass(frozen=True, slots=True)
class DurableRowParityProof:
    """Pre-existing table row-count comparison across apply."""

    ok: bool
    changed_tables: tuple[tuple[str, int, int], ...]
    missing_tables: tuple[str, ...]
    unauthorized_changes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DurableApplyEvidence:
    """Evidence emitted around the existing numbered migration runner."""

    pre: DurableDatabaseEvidence
    post: DurableDatabaseEvidence
    migration_result: MigrationResult
    row_parity: DurableRowParityProof
    recovered_after_interrupt: bool
    applied_at_ms: int


@dataclass(frozen=True, slots=True)
class DurableRuntimeConsumerResult:
    """Behavioral result for one admitted production consumer."""

    consumer_id: str
    behavior_proof_ref: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class DurableRestartConvergenceProof:
    """Post-restart durable tier and consumer convergence evidence."""

    evidence_ref: str
    observed_user_version: int
    observed_schema_inventory_sha256: str
    consumer_ids: tuple[str, ...]
    converged: bool
    observed_at_ms: int


@dataclass(frozen=True, slots=True)
class DurableTrainProof:
    """Complete proof bundle required before release."""

    fresh_ddl_parity: DurableFreshDDLParityProof
    runtime_consumers: tuple[DurableRuntimeConsumerResult, ...]
    restart_convergence: DurableRestartConvergenceProof
    proof_refs: tuple[str, ...]
    proven_at_ms: int


@dataclass(frozen=True, slots=True)
class DurableTrainFailure:
    """Persisted failure branch and exact operator recovery obligations."""

    phase: str
    previous_state: DurableChangeTrainState
    error_type: str
    error_message: str
    observed_user_version: int | None
    classification: DurableFailureClassification
    required_actions: tuple[str, ...]
    failed_at_ms: int
    pre_apply_evidence: DurableDatabaseEvidence | None = None


@dataclass(frozen=True, slots=True)
class DurableChangeTrain:
    """Immutable machine-readable authority for one durable migration slot."""

    manifest_format: str
    train_id: str
    tier: ArchiveTier
    current_version: int
    target_version: int
    slot: int
    owner_ref: str
    migration: DurableMigrationClaim
    riders: tuple[DurableChangeRider, ...]
    ordering_constraints: tuple[DurableOrderingConstraint, ...]
    drop_constraints: tuple[DurableDropConstraint, ...]
    row_change_allowances: tuple[DurableRowChangeAllowance, ...]
    backup_plan_ref: str | None
    state: DurableChangeTrainState
    revision: int
    declared_at_ms: int
    admitted_at_ms: int | None
    admission_evidence_ref: str | None
    fresh_ddl_parity: DurableFreshDDLParityProof | None
    reservation: DurableWriterReservation | None
    backup_authorization: DurableBackupAuthorization | None
    pre_apply_evidence: DurableDatabaseEvidence | None
    apply_evidence: DurableApplyEvidence | None
    proof: DurableTrainProof | None
    failure: DurableTrainFailure | None
    released_at_ms: int | None
    release_evidence_ref: str | None
    proof_refs: tuple[str, ...]
    source_continuity_evidence: DurableDatabaseEvidence | None = None

    @property
    def contention_key(self) -> tuple[str, int, int]:
        return (self.tier.value, self.target_version, self.slot)


def _durable_now_ms() -> int:
    return int(time.time() * 1000)


def _require_nonempty(value: str, *, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise DurableChangeTrainError(f"durable change train requires {label}")
    return normalized


def _append_proof_refs(existing: tuple[str, ...], *refs: str | None) -> tuple[str, ...]:
    values = list(existing)
    for ref in refs:
        if ref is None:
            continue
        normalized = _require_nonempty(ref, label="a non-empty proof reference")
        if normalized not in values:
            values.append(normalized)
    return tuple(values)


def _normalize_schema_sql(sql: str | None) -> str:
    """Normalize SQLite's representational DDL differences without weakening parity.

    SQLite stores identifiers quoted after ``ALTER TABLE ... RENAME`` even
    when the fresh canonical DDL leaves them bare.  Normalize only identifier
    quoting and layout.  String literals, object names, and PRAGMA-derived
    columns remain part of the inventory, so a missing constraint or changed
    table shape still fails parity.
    """
    if sql is None:
        return ""
    unquoted: list[str] = []
    string_literals: list[str] = []
    index = 0
    while index < len(sql):
        character = sql[index]
        if character == "'":
            closing = "]" if character == "[" else character
            start = index
            index += 1
            while index < len(sql):
                if sql[index] != closing:
                    index += 1
                    continue
                if index + 1 < len(sql) and sql[index + 1] == closing:
                    index += 2
                    continue
                index += 1
                break
            string_literals.append(sql[start:index])
            unquoted.append(f"\x00{len(string_literals) - 1}\x00")
            continue
        if character in {'"', "`", "["}:
            closing = "]" if character == "[" else character
            index += 1
            identifier: list[str] = []
            while index < len(sql):
                if sql[index] != closing:
                    identifier.append(sql[index])
                    index += 1
                    continue
                if index + 1 < len(sql) and sql[index + 1] == closing and closing != "]":
                    identifier.append(closing)
                    index += 2
                    continue
                index += 1
                break
            unquoted.append("".join(identifier))
            continue
        if sql.startswith("--", index):
            newline = sql.find("\n", index + 2)
            index = len(sql) if newline < 0 else newline + 1
            unquoted.append(" ")
            continue
        if sql.startswith("/*", index):
            closing_comment = sql.find("*/", index + 2)
            index = len(sql) if closing_comment < 0 else closing_comment + 2
            unquoted.append(" ")
            continue
        unquoted.append(character)
        index += 1
    collapsed = re.sub(r"\s+", " ", "".join(unquoted)).strip()
    collapsed = re.sub(r"\s*,\s*", ",", collapsed)
    collapsed = re.sub(r"\s*\(\s*", "(", collapsed)
    collapsed = re.sub(r"\s*\)", ")", collapsed)
    collapsed = re.sub(
        r"CHECK\(\((?P<column>[A-Za-z_][A-Za-z0-9_]*) IN\s*\((?P<values>[^()]*)\) OR (?P=column) IS NULL\)\)",
        r"CHECK(\g<column> IN(\g<values>))",
        collapsed,
    )
    for index, literal in enumerate(string_literals):
        collapsed = collapsed.replace(f"\x00{index}\x00", literal)
    return collapsed


def _quote_sqlite_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _schema_pragma_rows(conn: sqlite3.Connection, pragma: str, object_name: str) -> list[list[object]]:
    quoted = _quote_sqlite_identifier(object_name)
    return [list(row) for row in conn.execute(f"PRAGMA {pragma}({quoted})")]


def capture_durable_schema_inventory(conn: sqlite3.Connection) -> DurableSchemaInventory:
    """Capture the canonical object universe from SQLite itself, not a hand list."""
    rows = conn.execute(
        """
        SELECT type, name, tbl_name, sql
        FROM sqlite_schema
        WHERE name NOT LIKE 'sqlite_%'
          AND type IN ('table', 'index', 'trigger', 'view')
        ORDER BY type, name
        """
    ).fetchall()
    objects: list[DurableSchemaObjectEvidence] = []
    for raw_type, raw_name, raw_table_name, raw_sql in rows:
        object_type = str(raw_type)
        name = str(raw_name)
        table_name = str(raw_table_name)
        payload: dict[str, object] = {
            "type": object_type,
            "name": name,
            "table_name": table_name,
            "sql": _normalize_schema_sql(str(raw_sql) if raw_sql is not None else None),
        }
        if object_type == "table":
            payload["table_xinfo"] = _schema_pragma_rows(conn, "table_xinfo", name)
            payload["foreign_key_list"] = _schema_pragma_rows(conn, "foreign_key_list", name)
        elif object_type == "index":
            payload["index_xinfo"] = _schema_pragma_rows(conn, "index_xinfo", name)
        objects.append(
            DurableSchemaObjectEvidence(
                object_type=object_type,
                name=name,
                table_name=table_name,
                definition_sha256=_canonical_json_sha256(payload),
            )
        )
    inventory_payload = [
        {
            "object_type": item.object_type,
            "name": item.name,
            "table_name": item.table_name,
            "definition_sha256": item.definition_sha256,
        }
        for item in objects
    ]
    return DurableSchemaInventory(objects=tuple(objects), sha256=_canonical_json_sha256(inventory_payload))


def prove_durable_fresh_ddl_parity(
    tier: ArchiveTier,
    target_version: int,
    *,
    migrated_connection: sqlite3.Connection,
    fresh_connection: sqlite3.Connection,
    evidence_ref: str,
) -> DurableFreshDDLParityProof:
    """Prove a migration result has the same canonical inventory as fresh DDL."""
    if tier not in DURABLE_MIGRATION_TIERS:
        raise DurableChangeTrainError(f"fresh-DDL parity is not a durable-tier proof for {tier.value}")
    evidence = _require_nonempty(evidence_ref, label="fresh-DDL parity evidence")
    migrated_version = int(migrated_connection.execute("PRAGMA user_version").fetchone()[0] or 0)
    fresh_version = int(fresh_connection.execute("PRAGMA user_version").fetchone()[0] or 0)
    migrated = capture_durable_schema_inventory(migrated_connection)
    fresh = capture_durable_schema_inventory(fresh_connection)
    migrated_by_ref = {item.object_ref: item for item in migrated.objects}
    fresh_by_ref = {item.object_ref: item for item in fresh.objects}
    missing = tuple(sorted(set(fresh_by_ref) - set(migrated_by_ref)))
    unexpected = tuple(sorted(set(migrated_by_ref) - set(fresh_by_ref)))
    changed = tuple(
        sorted(
            object_ref
            for object_ref in set(migrated_by_ref) & set(fresh_by_ref)
            if migrated_by_ref[object_ref].definition_sha256 != fresh_by_ref[object_ref].definition_sha256
        )
    )
    matches = (
        migrated_version == target_version
        and fresh_version == target_version
        and not missing
        and not unexpected
        and not changed
        and migrated.sha256 == fresh.sha256
    )
    return DurableFreshDDLParityProof(
        tier=tier,
        target_version=target_version,
        migrated_version=migrated_version,
        fresh_version=fresh_version,
        migrated_inventory_sha256=migrated.sha256,
        fresh_inventory_sha256=fresh.sha256,
        missing_objects=missing,
        unexpected_objects=unexpected,
        changed_objects=changed,
        evidence_ref=evidence,
        matches=matches,
    )


def _durable_table_counts(conn: sqlite3.Connection) -> tuple[tuple[str, int], ...]:
    rows = conn.execute(
        """
        SELECT name, sql
        FROM sqlite_schema
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    counts: list[tuple[str, int]] = []
    for raw_name, raw_sql in rows:
        name = str(raw_name)
        sql = str(raw_sql or "").lstrip().upper()
        if sql.startswith("CREATE VIRTUAL TABLE"):
            continue
        quoted = _quote_sqlite_identifier(name)
        counts.append((name, int(conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])))
    return tuple(counts)


def capture_durable_database_evidence(
    conn: sqlite3.Connection,
    tier: ArchiveTier,
) -> DurableDatabaseEvidence:
    """Capture the pre/post/restart evidence used by the train state machine."""
    quick_check = tuple(str(row[0]) for row in conn.execute("PRAGMA quick_check"))
    inventory = capture_durable_schema_inventory(conn)
    live_path = _connection_main_path(conn)
    from polylogue.storage.archive_identity import ArchiveIdentity

    # Durable migration evidence must survive replacement of rebuildable
    # generations and creation of another durable tier. Bind a train to the
    # file it actually migrates, not to the whole durable pair: a source train
    # must remain valid when a previously absent user.db is initialized later.
    tier_identity = ArchiveIdentity.resolve(live_path.parent).tier(tier.value).stable_id
    archive_identity_digest = hashlib.sha256(tier_identity.encode("utf-8")).hexdigest()
    content_hasher = hashlib.sha256()
    for statement in conn.iterdump():
        content_hasher.update(statement.encode("utf-8"))
        content_hasher.update(b"\n")
    return DurableDatabaseEvidence(
        tier=tier,
        user_version=int(conn.execute("PRAGMA user_version").fetchone()[0] or 0),
        quick_check=quick_check,
        schema_inventory_sha256=inventory.sha256,
        row_counts=_durable_table_counts(conn),
        archive_identity_digest=archive_identity_digest,
        content_sha256=content_hasher.hexdigest(),
        observed_at_ms=_durable_now_ms(),
    )


def _archive_identity_continuity_matches(
    actual_digest: str,
    expected_digest: str,
    archive_root: Path,
    tier: ArchiveTier,
) -> bool:
    if actual_digest == expected_digest:
        return True
    from polylogue.storage.archive_identity import ArchiveIdentity

    identity = ArchiveIdentity.resolve(archive_root.resolve())
    legacy_digest = identity.authority_identity_digest
    tier_identity = identity.tier(tier.value).stable_id
    durable_digest = hashlib.sha256(tier_identity.encode("utf-8")).hexdigest()
    # Manifests written before the durable-tier identity split contain the
    # old full-archive digest. Admit that legacy evidence only when the
    # current archive still has the same legacy identity and the newly
    # captured evidence proves the durable identity is unchanged.
    return expected_digest == legacy_digest and actual_digest == durable_digest


def _assert_durable_database_continuity(
    actual: DurableDatabaseEvidence,
    expected: DurableDatabaseEvidence,
    *,
    label: str,
    archive_root: Path | None = None,
    connection: sqlite3.Connection | None = None,
) -> None:
    """Require the live durable file to retain its authenticated evidence."""
    identity_continuous = actual.archive_identity_digest == expected.archive_identity_digest
    if not identity_continuous and (archive_root is not None or connection is not None):
        resolved_archive_root = archive_root or _connection_main_path(cast(sqlite3.Connection, connection)).parent
        identity_continuous = _archive_identity_continuity_matches(
            actual.archive_identity_digest,
            expected.archive_identity_digest,
            resolved_archive_root,
            actual.tier,
        )
    if (
        actual.quick_check != expected.quick_check
        or actual.quick_check != ("ok",)
        or actual.user_version != expected.user_version
        or not identity_continuous
        or actual.content_sha256 != expected.content_sha256
    ):
        raise DurableChangeTrainError(f"{label} durable tier identity/content continuity proof failed")


def _drop_table_names(constraints: Sequence[DurableDropConstraint]) -> set[str]:
    names: set[str] = set()
    for constraint in constraints:
        prefix, separator, name = constraint.object_ref.partition(":")
        if separator and prefix == "table":
            names.add(name)
        elif not separator:
            names.add(prefix)
    return names


def prove_durable_row_parity(
    pre: DurableDatabaseEvidence,
    post: DurableDatabaseEvidence,
    *,
    drop_constraints: Sequence[DurableDropConstraint],
    row_change_allowances: Sequence[DurableRowChangeAllowance],
) -> DurableRowParityProof:
    """Require unchanged counts for old tables unless an admitted rider authorizes movement."""
    before = dict(pre.row_counts)
    after = dict(post.row_counts)
    dropped = _drop_table_names(drop_constraints)
    allowances = {allowance.table: allowance for allowance in row_change_allowances}
    changed: list[tuple[str, int, int]] = []
    missing: list[str] = []
    unauthorized: list[str] = []
    for table, before_count in sorted(before.items()):
        if table not in after:
            missing.append(table)
            if table not in dropped:
                unauthorized.append(f"{table}: missing without an admitted drop constraint")
            continue
        after_count = after[table]
        if after_count == before_count:
            continue
        changed.append((table, before_count, after_count))
        allowance = allowances.get(table)
        if allowance is None:
            unauthorized.append(f"{table}: row count changed {before_count}->{after_count} without an allowance")
            continue
        delta = after_count - before_count
        if allowance.expected_delta is not None and delta != allowance.expected_delta:
            unauthorized.append(
                f"{table}: row-count delta {delta} did not match admitted delta {allowance.expected_delta}"
            )
    return DurableRowParityProof(
        ok=not unauthorized,
        changed_tables=tuple(changed),
        missing_tables=tuple(missing),
        unauthorized_changes=tuple(unauthorized),
    )


def declare_durable_change_train(
    *,
    train_id: str,
    tier: ArchiveTier,
    current_version: int,
    target_version: int,
    slot: int,
    owner_ref: str,
    migration: DurableMigrationClaim,
    riders: Sequence[DurableChangeRider] = (),
    ordering_constraints: Sequence[DurableOrderingConstraint] = (),
    drop_constraints: Sequence[DurableDropConstraint] = (),
    row_change_allowances: Sequence[DurableRowChangeAllowance] = (),
    backup_plan_ref: str | None = None,
    declared_at_ms: int | None = None,
) -> DurableChangeTrain:
    """Declare one next-version train; no authority is granted at this stage."""
    if tier not in DURABLE_MIGRATION_TIERS:
        raise DurableChangeTrainError(f"{tier.value} is not a durable migration tier")
    if current_version < 1 or target_version != current_version + 1:
        raise DurableChangeTrainError(
            f"durable train must own exactly one contiguous version: current={current_version}, target={target_version}"
        )
    if slot != target_version:
        raise DurableChangeTrainError(
            f"durable train slot must equal target version {target_version}, found slot {slot}"
        )
    if migration.tier is not tier or migration.target_version != target_version or migration.slot != slot:
        raise DurableChangeTrainError("durable train migration claim does not match its tier/target/slot")
    train = DurableChangeTrain(
        manifest_format=DURABLE_CHANGE_TRAIN_FORMAT,
        train_id=_require_nonempty(train_id, label="train_id"),
        tier=tier,
        current_version=current_version,
        target_version=target_version,
        slot=slot,
        owner_ref=_require_nonempty(owner_ref, label="owner_ref"),
        migration=migration,
        riders=tuple(riders),
        ordering_constraints=tuple(ordering_constraints),
        drop_constraints=tuple(drop_constraints),
        row_change_allowances=tuple(row_change_allowances),
        backup_plan_ref=backup_plan_ref.strip() if backup_plan_ref and backup_plan_ref.strip() else None,
        state=DurableChangeTrainState.DECLARED,
        revision=0,
        declared_at_ms=declared_at_ms if declared_at_ms is not None else _durable_now_ms(),
        admitted_at_ms=None,
        admission_evidence_ref=None,
        fresh_ddl_parity=None,
        reservation=None,
        backup_authorization=None,
        pre_apply_evidence=None,
        apply_evidence=None,
        proof=None,
        failure=None,
        released_at_ms=None,
        release_evidence_ref=None,
        proof_refs=(),
    )
    validate_durable_change_train_manifest(train)
    return train


def add_durable_change_train_rider(
    train: DurableChangeTrain,
    rider: DurableChangeRider,
) -> DurableChangeTrain:
    """Add a rider only before admission; late riders must take the next train."""
    if train.state is not DurableChangeTrainState.DECLARED:
        raise DurableChangeTrainError(
            f"late rider {rider.rider_id!r} cannot join {train.train_id} after {train.state.value}; "
            f"declare a new {train.tier.value} train for target v{train.target_version + 1}"
        )
    if any(existing.rider_id == rider.rider_id for existing in train.riders):
        raise DurableChangeTrainError(f"duplicate rider id on durable train: {rider.rider_id}")
    updated = replace(train, riders=(*train.riders, rider), revision=train.revision + 1)
    validate_durable_change_train_manifest(updated)
    return updated


def _validate_riders(train: DurableChangeTrain) -> None:
    if not train.riders:
        raise DurableChangeTrainError("durable train admission requires at least one schema-and-runtime rider")
    rider_ids = [rider.rider_id for rider in train.riders]
    if len(rider_ids) != len(set(rider_ids)):
        raise DurableChangeTrainError(f"durable train has duplicate rider ids: {rider_ids}")
    known_riders = set(rider_ids)
    consumer_ids: set[str] = set()
    for rider in train.riders:
        _require_nonempty(rider.rider_id, label="rider_id")
        _require_nonempty(rider.owner_ref, label=f"owner for rider {rider.rider_id}")
        if not rider.schema_objects:
            raise DurableChangeTrainError(f"rider {rider.rider_id} is schema-empty")
        if any(not item.strip() for item in rider.schema_objects):
            raise DurableChangeTrainError(f"rider {rider.rider_id} has an empty schema object reference")
        production_refs = {consumer.production_ref for consumer in rider.runtime_consumers}
        if len(production_refs) < 2 and not rider.trust_floor_exception_ref:
            raise DurableChangeTrainError(
                f"rider {rider.rider_id} has fewer than two materially distinct runtime consumers and no "
                "trust-floor exception"
            )
        if not rider.runtime_consumers:
            raise DurableChangeTrainError(f"rider {rider.rider_id} is schema-only; no runtime consumer was declared")
        declared_behavior_refs = set(rider.behavior_proof_refs)
        if not declared_behavior_refs:
            raise DurableChangeTrainError(f"rider {rider.rider_id} has no behavioral proof references")
        for consumer in rider.runtime_consumers:
            _require_nonempty(consumer.consumer_id, label=f"consumer id for rider {rider.rider_id}")
            production_ref = _require_nonempty(
                consumer.production_ref,
                label=f"production reference for consumer {consumer.consumer_id}",
            )
            if production_ref.startswith(("tests/", "test:", "fixture:")):
                raise DurableChangeTrainError(
                    f"consumer {consumer.consumer_id} is test-only, not a production runtime consumer: {production_ref}"
                )
            behavior_ref = _require_nonempty(
                consumer.behavior_proof_ref,
                label=f"behavior proof for consumer {consumer.consumer_id}",
            )
            if behavior_ref not in declared_behavior_refs:
                raise DurableChangeTrainError(
                    f"consumer {consumer.consumer_id} behavior proof is not owned by rider {rider.rider_id}"
                )
            if consumer.consumer_id in consumer_ids:
                raise DurableChangeTrainError(f"runtime consumer is owned by multiple riders: {consumer.consumer_id}")
            consumer_ids.add(consumer.consumer_id)
        for predecessor in rider.after_rider_ids:
            if predecessor not in known_riders:
                raise DurableChangeTrainError(f"rider {rider.rider_id} orders after unknown rider {predecessor}")
    for constraint in train.ordering_constraints:
        if constraint.before_rider_id not in known_riders or constraint.after_rider_id not in known_riders:
            raise DurableChangeTrainError(
                "ordering constraint names unknown rider(s): "
                f"{constraint.before_rider_id} -> {constraint.after_rider_id}"
            )
        _require_nonempty(constraint.evidence_ref, label="ordering evidence")
    for drop_constraint in train.drop_constraints:
        _require_nonempty(drop_constraint.object_ref, label="drop object reference")
        _require_nonempty(drop_constraint.copy_forward_proof_ref, label="copy-forward proof")
        _require_nonempty(drop_constraint.consent_ref, label="destructive durable-change consent")
        unknown = set(drop_constraint.after_rider_ids) - known_riders
        if unknown:
            raise DurableChangeTrainError(
                f"drop constraint {drop_constraint.object_ref} orders after unknown riders: {sorted(unknown)}"
            )
    allowance_tables = [allowance.table for allowance in train.row_change_allowances]
    if len(allowance_tables) != len(set(allowance_tables)):
        raise DurableChangeTrainError(f"duplicate row-change allowances: {allowance_tables}")
    for allowance in train.row_change_allowances:
        _require_nonempty(allowance.table, label="row-change allowance table")
        _require_nonempty(allowance.reason_ref, label=f"row-change authority for {allowance.table}")
    _validate_rider_ordering(train)


def _validate_rider_ordering(train: DurableChangeTrain) -> None:
    edges: dict[str, set[str]] = {rider.rider_id: set() for rider in train.riders}
    for rider in train.riders:
        for predecessor in rider.after_rider_ids:
            edges[predecessor].add(rider.rider_id)
    for constraint in train.ordering_constraints:
        edges[constraint.before_rider_id].add(constraint.after_rider_id)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(rider_id: str) -> None:
        if rider_id in visited:
            return
        if rider_id in visiting:
            raise DurableChangeTrainError(f"durable train rider ordering contains a cycle at {rider_id}")
        visiting.add(rider_id)
        for successor in sorted(edges[rider_id]):
            visit(successor)
        visiting.remove(rider_id)
        visited.add(rider_id)

    for rider_id in sorted(edges):
        visit(rider_id)


def _train_owns_contention_key(train: DurableChangeTrain) -> bool:
    return train.state is not DurableChangeTrainState.RELEASED


def _reject_duplicate_train_ownership(
    train: DurableChangeTrain,
    active_trains: Sequence[DurableChangeTrain],
) -> None:
    for active in active_trains:
        if active.train_id == train.train_id or not _train_owns_contention_key(active):
            continue
        if active.contention_key == train.contention_key:
            raise DurableChangeTrainError(
                "durable train contention key already owned by "
                f"{active.train_id} ({active.owner_ref}): {train.contention_key}"
            )


def admit_durable_change_train(
    train: DurableChangeTrain,
    *,
    observed_current_version: int,
    fresh_ddl_parity: DurableFreshDDLParityProof,
    admission_evidence_ref: str,
    active_trains: Sequence[DurableChangeTrain] = (),
    migration_claims: Sequence[DurableMigrationClaim] | None = None,
    canonical_target_version: int | None = None,
    admitted_at_ms: int | None = None,
) -> DurableChangeTrain:
    """Admit exact migration/runtime/fresh-DDL wiring and freeze the rider set."""
    if train.state is not DurableChangeTrainState.DECLARED:
        raise DurableChangeTrainError(f"only a declared train may be admitted, found {train.state.value}")
    validate_durable_change_train_manifest(train)
    canonical_target = (
        ARCHIVE_VERSION_BY_TIER[train.tier] if canonical_target_version is None else canonical_target_version
    )
    if train.target_version != canonical_target:
        raise DurableChangeTrainError(
            f"stale durable train target v{train.target_version}; "
            f"shipped {train.tier.value} target is v{canonical_target}"
        )
    if observed_current_version != train.current_version:
        raise DurableChangeTrainError(
            f"stale durable train current v{train.current_version}; "
            f"live {train.tier.value} is v{observed_current_version}"
        )
    _reject_duplicate_train_ownership(train, active_trains)
    claims = tuple(migration_claims) if migration_claims is not None else durable_migration_claims(train.tier)
    collisions = find_durable_migration_collisions(claims)
    if collisions:
        details = "; ".join(
            f"{item.contention_key}: " + ", ".join(f"{claim.path} ({claim.owner_ref})" for claim in item.claims)
            for item in collisions
        )
        raise DurableChangeTrainError(
            "durable migration collision blocks train admission; rebase/renumber late riders: " + details
        )
    matching = tuple(claim for claim in claims if claim.contention_key == train.contention_key)
    if len(matching) != 1:
        raise DurableChangeTrainError(
            "durable train requires exactly one shipped migration claim for "
            f"{train.contention_key}; found {len(matching)}"
        )
    if matching[0] != train.migration:
        raise DurableChangeTrainError(
            "durable train migration bytes/owner/path differ from the shipped numbered migration claim"
        )
    if train.migration.requires_backup and not train.backup_plan_ref:
        raise DurableChangeTrainError("durable train migration requires backup authority but declares no backup plan")
    _validate_riders(train)
    if (
        fresh_ddl_parity.tier is not train.tier
        or fresh_ddl_parity.target_version != train.target_version
        or not fresh_ddl_parity.matches
    ):
        raise DurableChangeTrainError(
            "durable train admission requires matching fresh-DDL parity for its exact tier and target; "
            f"missing={fresh_ddl_parity.missing_objects}, unexpected={fresh_ddl_parity.unexpected_objects}, "
            f"changed={fresh_ddl_parity.changed_objects}"
        )
    evidence = _require_nonempty(admission_evidence_ref, label="admission evidence")
    updated = replace(
        train,
        state=DurableChangeTrainState.ADMITTED,
        revision=train.revision + 1,
        admitted_at_ms=admitted_at_ms if admitted_at_ms is not None else _durable_now_ms(),
        admission_evidence_ref=evidence,
        fresh_ddl_parity=fresh_ddl_parity,
        proof_refs=_append_proof_refs(train.proof_refs, evidence, fresh_ddl_parity.evidence_ref),
    )
    validate_durable_change_train_manifest(updated)
    return updated


def reserve_durable_change_train(
    train: DurableChangeTrain,
    *,
    reservation_id: str,
    reservation_owner_ref: str,
    archive_root: Path,
    tier_path: Path,
    daemon_stopped_evidence_ref: str,
    single_writer_evidence_ref: str,
    active_trains: Sequence[DurableChangeTrain] = (),
    reserved_at_ms: int | None = None,
) -> DurableChangeTrain:
    """Bind an admitted train to the stopped-daemon, archive-root writer lease."""
    if train.state is not DurableChangeTrainState.ADMITTED:
        raise DurableChangeTrainError(f"only an admitted train may reserve a writer, found {train.state.value}")
    validate_durable_change_train_manifest(train)
    _reject_duplicate_train_ownership(train, active_trains)
    reservation = _require_nonempty(reservation_id, label="writer reservation id")
    owner = _require_nonempty(reservation_owner_ref, label="writer reservation owner")
    if owner != train.owner_ref:
        raise DurableChangeTrainError(
            f"writer reservation owner {owner!r} does not match train owner {train.owner_ref!r}"
        )
    root = archive_root.resolve(strict=False)
    expected_tier_path = root / f"{train.tier.value}.db"
    actual_tier_path = tier_path.resolve(strict=False)
    if actual_tier_path != expected_tier_path.resolve(strict=False):
        raise DurableChangeTrainError(
            f"writer reservation tier path must be {expected_tier_path}, found {actual_tier_path}"
        )
    for active in active_trains:
        existing = active.reservation
        if active.train_id == train.train_id or existing is None or not existing.active:
            continue
        if Path(existing.archive_root).resolve(strict=False) != root:
            continue
        if existing.reservation_id != reservation or existing.owner_ref != owner:
            raise DurableChangeTrainError(
                f"second writer rejected for archive root {root}: {existing.reservation_id} owned by "
                f"{existing.owner_ref} is already active"
            )
    stopped_ref = _require_nonempty(daemon_stopped_evidence_ref, label="stopped-daemon evidence")
    writer_ref = _require_nonempty(single_writer_evidence_ref, label="single-writer lease evidence")
    writer = DurableWriterReservation(
        reservation_id=reservation,
        owner_ref=owner,
        archive_root=str(root),
        tier_path=str(actual_tier_path),
        daemon_stopped_evidence_ref=stopped_ref,
        single_writer_evidence_ref=writer_ref,
        reserved_at_ms=reserved_at_ms if reserved_at_ms is not None else _durable_now_ms(),
        active=True,
    )
    updated = replace(
        train,
        state=DurableChangeTrainState.RESERVED,
        revision=train.revision + 1,
        reservation=writer,
        proof_refs=_append_proof_refs(train.proof_refs, stopped_ref, writer_ref),
    )
    validate_durable_change_train_manifest(updated)
    return updated


def authorize_durable_change_train_backup(
    conn: sqlite3.Connection,
    train: DurableChangeTrain,
    *,
    backup_manifest: Path | None,
    evidence_ref: str,
    authorized_at_ms: int | None = None,
) -> DurableChangeTrain:
    """Bind the exact live bytes and authenticated backup receipt before apply."""
    if train.state is not DurableChangeTrainState.RESERVED:
        raise DurableChangeTrainError(f"only a reserved train may authorize backup, found {train.state.value}")
    validate_durable_change_train_manifest(train)
    if train.reservation is None or not train.reservation.active:
        raise DurableChangeTrainError("backup authorization requires an active writer reservation")
    live_path = _connection_main_path(conn).resolve(strict=False)
    if live_path != Path(train.reservation.tier_path).resolve(strict=False):
        raise DurableChangeTrainError(
            f"backup authorization connection {live_path} does not match reserved tier {train.reservation.tier_path}"
        )
    current_version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
    if current_version != train.current_version:
        raise DurableChangeTrainError(
            f"backup authorization is stale: train expects v{train.current_version}, live tier is v{current_version}"
        )
    manifest_path: Path | None = None
    receipt_path: Path | None = None
    mode = "additive-no-backup"
    if train.migration.requires_backup:
        if backup_manifest is None:
            raise DurableChangeTrainError(
                f"{train.tier.value} train {train.train_id} requires an authenticated backup before apply"
            )
        manifest_path = _backup_manifest_path(backup_manifest).resolve(strict=False)
        receipt_path = validate_migration_backup_manifest(backup_manifest, train.tier, connection=conn)
        mode = "verified-backup"
    elif backup_manifest is not None:
        manifest_path = _backup_manifest_path(backup_manifest).resolve(strict=False)
        receipt_path = validate_migration_backup_manifest(backup_manifest, train.tier, connection=conn)
        mode = "verified-backup"
    evidence = _require_nonempty(evidence_ref, label="backup authorization evidence")
    authorization_time = authorized_at_ms if authorized_at_ms is not None else _durable_now_ms()
    pre_apply_evidence = capture_durable_database_evidence(conn, train.tier)
    if pre_apply_evidence.user_version != train.current_version or pre_apply_evidence.quick_check != ("ok",):
        raise DurableChangeTrainError(
            "backup authorization could not bind valid pre-apply evidence: "
            f"version={pre_apply_evidence.user_version}, quick_check={pre_apply_evidence.quick_check}"
        )
    authorization = DurableBackupAuthorization(
        mode=mode,
        live_tier_path=str(live_path),
        live_user_version=current_version,
        manifest_path=str(manifest_path) if manifest_path is not None else None,
        manifest_sha256=_sha256_file(manifest_path) if manifest_path is not None else None,
        receipt_path=str(receipt_path) if receipt_path is not None else None,
        receipt_sha256=_sha256_file(receipt_path) if receipt_path is not None else None,
        evidence_ref=evidence,
        authorized_at_ms=authorization_time,
    )
    updated = replace(
        train,
        state=DurableChangeTrainState.BACKUP_AUTHORIZED,
        revision=train.revision + 1,
        backup_authorization=authorization,
        pre_apply_evidence=pre_apply_evidence,
        proof_refs=_append_proof_refs(
            train.proof_refs,
            evidence,
            str(receipt_path) if receipt_path else None,
            f"pre-apply-schema:{pre_apply_evidence.schema_inventory_sha256}",
        ),
    )
    validate_durable_change_train_manifest(updated)
    return updated


def _revalidate_backup_authorization(conn: sqlite3.Connection, train: DurableChangeTrain) -> Path | None:
    authorization = train.backup_authorization
    if authorization is None:
        raise DurableChangeTrainError("apply requires backup authorization evidence")
    live_path = _connection_main_path(conn).resolve(strict=False)
    if str(live_path) != authorization.live_tier_path:
        raise DurableChangeTrainError(
            f"authorized live tier path changed: {authorization.live_tier_path} -> {live_path}"
        )
    version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
    if version != authorization.live_user_version or version != train.current_version:
        raise DurableChangeTrainError(
            f"authorized live tier version changed: authorized v{authorization.live_user_version}, observed v{version}"
        )
    if authorization.mode == "additive-no-backup":
        if train.migration.requires_backup:
            raise DurableChangeTrainError("backup-required migration cannot apply under additive-no-backup authority")
        return None
    if authorization.manifest_path is None or authorization.receipt_path is None:
        raise DurableChangeTrainError("verified-backup authority is missing manifest or receipt paths")
    manifest = Path(authorization.manifest_path)
    receipt = Path(authorization.receipt_path)
    if _sha256_file(manifest) != authorization.manifest_sha256:
        raise DurableChangeTrainError("authorized backup manifest bytes changed before apply")
    if _sha256_file(receipt) != authorization.receipt_sha256:
        raise DurableChangeTrainError("authorized backup receipt bytes changed before apply")
    validated_receipt = validate_migration_backup_manifest(manifest, train.tier, connection=conn)
    if validated_receipt.resolve(strict=False) != receipt.resolve(strict=False):
        raise DurableChangeTrainError("backup validator returned a different receipt than the train authorized")
    return manifest


def _safe_user_version(conn: sqlite3.Connection) -> int | None:
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
    except sqlite3.Error:
        _LOGGER.warning("unable to read durable tier PRAGMA user_version", exc_info=True)
        return None


def _failed_change_train(
    train: DurableChangeTrain,
    *,
    phase: str,
    error: Exception,
    observed_user_version: int | None,
    pre_apply_evidence: DurableDatabaseEvidence | None,
) -> DurableChangeTrain:
    actions: tuple[str, ...]
    if observed_user_version == train.current_version:
        classification = DurableFailureClassification.ROLLED_BACK_TO_CURRENT
        actions = (
            "release the writer reservation",
            "retain the admitted train and exact rider set",
            "reserve the writer again",
            "reauthorize a backup against the unchanged live bytes",
            "retry the existing numbered migration runner",
        )
    elif observed_user_version == train.target_version:
        classification = DurableFailureClassification.COMMITTED_TARGET_UNPROVEN
        actions = (
            "do not reapply the numbered migration",
            "release the writer reservation",
            "capture post-apply integrity and row parity",
            "run runtime-consumer proofs",
            "restart and prove convergence before release",
        )
    else:
        classification = DurableFailureClassification.INDETERMINATE
        actions = (
            "keep the daemon stopped",
            "do not retry or release the train",
            "restore the exact authenticated backup bound by backup_authorization",
            "re-prove the restored current version before declaring a replacement train",
        )
    failure = DurableTrainFailure(
        phase=phase,
        previous_state=train.state,
        error_type=type(error).__name__,
        error_message=str(error),
        observed_user_version=observed_user_version,
        classification=classification,
        required_actions=actions,
        failed_at_ms=_durable_now_ms(),
        pre_apply_evidence=pre_apply_evidence,
    )
    failed = replace(
        train,
        state=DurableChangeTrainState.FAILED,
        revision=train.revision + 1,
        failure=failure,
    )
    validate_durable_change_train_manifest(failed)
    return failed


def apply_durable_change_train(
    conn: sqlite3.Connection,
    train: DurableChangeTrain,
    *,
    active_trains: Sequence[DurableChangeTrain] = (),
    applied_at_ms: int | None = None,
) -> DurableChangeTrain:
    """Apply by delegating to ``migrate_archive_tier``; this is not a second engine."""
    if train.state is not DurableChangeTrainState.BACKUP_AUTHORIZED:
        raise DurableChangeTrainError(f"only a backup-authorized train may apply, found {train.state.value}")
    validate_durable_change_train_manifest(train)
    if train.reservation is None or not train.reservation.active:
        raise DurableChangeTrainError("apply requires an active writer reservation")
    _reject_duplicate_train_ownership(train, active_trains)
    pre = train.pre_apply_evidence
    try:
        if pre is None:
            raise DurableChangeTrainError("apply requires persisted pre-apply evidence from authorization")
        backup_manifest = _revalidate_backup_authorization(conn, train)
        observed_pre = capture_durable_database_evidence(conn, train.tier)
        if (
            observed_pre.user_version != pre.user_version
            or observed_pre.quick_check != pre.quick_check
            or observed_pre.schema_inventory_sha256 != pre.schema_inventory_sha256
            or observed_pre.row_counts != pre.row_counts
        ):
            raise DurableChangeTrainError("live durable tier changed after pre-apply evidence was authorized")
        result = migrate_archive_tier(
            conn,
            train.tier,
            backup_manifest=backup_manifest,
            target_version=train.target_version,
        )
        if (
            result.from_version != train.current_version
            or result.to_version != train.target_version
            or result.applied_versions != (train.target_version,)
        ):
            raise DurableChangeTrainError(
                "existing migration runner did not apply this train's one exact slot: "
                f"from={result.from_version}, to={result.to_version}, applied={result.applied_versions}"
            )
        post = capture_durable_database_evidence(conn, train.tier)
        if post.user_version != train.target_version or post.quick_check != ("ok",):
            raise DurableChangeTrainError(
                f"post-apply evidence is invalid: version={post.user_version}, quick_check={post.quick_check}"
            )
        row_parity = prove_durable_row_parity(
            pre,
            post,
            drop_constraints=train.drop_constraints,
            row_change_allowances=train.row_change_allowances,
        )
        if not row_parity.ok:
            raise DurableChangeTrainError("post-apply row parity failed: " + "; ".join(row_parity.unauthorized_changes))
    except Exception as exc:
        failed = _failed_change_train(
            train,
            phase="apply",
            error=exc,
            observed_user_version=_safe_user_version(conn),
            pre_apply_evidence=pre,
        )
        raise DurableChangeTrainApplyError(
            f"durable change train {train.train_id} apply failed: {exc}",
            failed_train=failed,
        ) from exc
    apply_evidence = DurableApplyEvidence(
        pre=pre,
        post=post,
        migration_result=result,
        row_parity=row_parity,
        recovered_after_interrupt=False,
        applied_at_ms=applied_at_ms if applied_at_ms is not None else _durable_now_ms(),
    )
    updated = replace(
        train,
        state=DurableChangeTrainState.APPLIED,
        revision=train.revision + 1,
        apply_evidence=apply_evidence,
        proof_refs=_append_proof_refs(
            train.proof_refs,
            f"schema-inventory:{post.schema_inventory_sha256}",
            f"row-parity:{train.train_id}:v{train.target_version}",
        ),
    )
    validate_durable_change_train_manifest(updated)
    return updated


def record_durable_writer_release(
    train: DurableChangeTrain,
    *,
    evidence_ref: str,
    released_at_ms: int | None = None,
) -> DurableChangeTrain:
    """Record that the outer archive-root lease has exited before restart proof."""
    validate_durable_change_train_manifest(train)
    if train.reservation is None:
        raise DurableChangeTrainError("writer release requires a recorded reservation")
    if not train.reservation.active:
        raise DurableChangeTrainError("writer reservation is already released")
    if train.state not in {
        DurableChangeTrainState.APPLIED,
        DurableChangeTrainState.FAILED,
    }:
        raise DurableChangeTrainError(f"writer cannot be released from train state {train.state.value}")
    if (
        train.state is DurableChangeTrainState.FAILED
        and train.failure is not None
        and train.failure.classification is DurableFailureClassification.INDETERMINATE
    ):
        raise DurableChangeTrainRecoveryError(
            "indeterminate durable train must retain stopped-daemon/single-writer authority until exact restore",
            failed_train=train,
        )
    proof_ref = _require_nonempty(evidence_ref, label="writer release evidence")
    released = replace(
        train.reservation,
        active=False,
        released_at_ms=released_at_ms if released_at_ms is not None else _durable_now_ms(),
        release_evidence_ref=proof_ref,
    )
    updated = replace(
        train,
        reservation=released,
        revision=train.revision + 1,
        proof_refs=_append_proof_refs(train.proof_refs, proof_ref),
    )
    validate_durable_change_train_manifest(updated)
    return updated


def recover_durable_change_train(
    conn: sqlite3.Connection,
    train: DurableChangeTrain,
    *,
    recovery_evidence_ref: str,
    writer_release_evidence_ref: str,
) -> DurableChangeTrain:
    """Resume only the two versions whose recovery is unambiguous."""
    if train.state is not DurableChangeTrainState.FAILED or train.failure is None:
        raise DurableChangeTrainError("only a failed train with recovery evidence may recover")
    validate_durable_change_train_manifest(train)
    observed = _safe_user_version(conn)
    if observed != train.failure.observed_user_version:
        raise DurableChangeTrainError(
            "failed train recovery version changed: "
            f"failure observed {train.failure.observed_user_version}, now {observed}"
        )
    recovery_ref = _require_nonempty(recovery_evidence_ref, label="failure recovery evidence")
    release_ref = _require_nonempty(writer_release_evidence_ref, label="writer release evidence")
    reservation = train.reservation
    if reservation is None:
        raise DurableChangeTrainError("failed train recovery requires its original writer reservation")
    if train.failure.classification is DurableFailureClassification.ROLLED_BACK_TO_CURRENT:
        if observed != train.current_version:
            raise DurableChangeTrainError("rolled-back recovery requires the exact declared current version")
        pre = train.failure.pre_apply_evidence
        if pre is None:
            raise DurableChangeTrainError("rolled-back recovery lacks pre-apply evidence for continuity")
        _assert_durable_database_continuity(
            capture_durable_database_evidence(conn, train.tier),
            pre,
            label="rolled-back recovery",
            connection=conn,
        )
        updated = replace(
            train,
            state=DurableChangeTrainState.ADMITTED,
            revision=train.revision + 1,
            reservation=None,
            backup_authorization=None,
            pre_apply_evidence=None,
            apply_evidence=None,
            failure=None,
            proof_refs=_append_proof_refs(train.proof_refs, recovery_ref, release_ref),
        )
        validate_durable_change_train_manifest(updated)
        return updated
    if train.failure.classification is DurableFailureClassification.COMMITTED_TARGET_UNPROVEN:
        if observed != train.target_version:
            raise DurableChangeTrainError("committed-target recovery requires the exact declared target version")
        pre = train.failure.pre_apply_evidence
        if pre is None:
            raise DurableChangeTrainError("committed-target recovery lacks pre-apply evidence for row parity")
        post = capture_durable_database_evidence(conn, train.tier)
        if post.quick_check != ("ok",):
            raise DurableChangeTrainError(f"committed-target recovery quick_check failed: {post.quick_check}")
        row_parity = prove_durable_row_parity(
            pre,
            post,
            drop_constraints=train.drop_constraints,
            row_change_allowances=train.row_change_allowances,
        )
        if not row_parity.ok:
            raise DurableChangeTrainError(
                "committed-target recovery row parity failed: " + "; ".join(row_parity.unauthorized_changes)
            )
        authorization = train.backup_authorization
        backup_receipt = (
            Path(authorization.receipt_path)
            if authorization is not None and authorization.receipt_path is not None
            else None
        )
        apply_evidence = DurableApplyEvidence(
            pre=pre,
            post=post,
            migration_result=MigrationResult(
                tier=train.tier,
                from_version=train.current_version,
                to_version=train.target_version,
                applied_versions=(train.target_version,),
                backup_receipt=backup_receipt,
            ),
            row_parity=row_parity,
            recovered_after_interrupt=True,
            applied_at_ms=_durable_now_ms(),
        )
        released_reservation = replace(
            reservation,
            active=False,
            released_at_ms=_durable_now_ms(),
            release_evidence_ref=release_ref,
        )
        updated = replace(
            train,
            state=DurableChangeTrainState.APPLIED,
            revision=train.revision + 1,
            reservation=released_reservation,
            apply_evidence=apply_evidence,
            failure=None,
            proof_refs=_append_proof_refs(train.proof_refs, recovery_ref, release_ref),
        )
        validate_durable_change_train_manifest(updated)
        return updated
    raise DurableChangeTrainRecoveryError(
        "indeterminate durable train cannot recover automatically; "
        "keep the daemon stopped and restore the exact authenticated backup first",
        failed_train=train,
    )


def capture_durable_restart_convergence(
    conn: sqlite3.Connection,
    train: DurableChangeTrain,
    *,
    runtime_consumers: Sequence[DurableRuntimeConsumerResult],
    evidence_ref: str,
    observed_at_ms: int | None = None,
) -> DurableRestartConvergenceProof:
    """Capture the reopened database state after the operator/runtime restart."""
    validate_durable_change_train_manifest(train)
    if train.apply_evidence is None:
        raise DurableChangeTrainError("restart convergence requires apply evidence")
    evidence = capture_durable_database_evidence(conn, train.tier)
    consumer_ids = tuple(sorted(result.consumer_id for result in runtime_consumers if result.passed))
    expected_ids = tuple(sorted(consumer.consumer_id for rider in train.riders for consumer in rider.runtime_consumers))
    converged = (
        evidence.user_version == train.target_version
        and evidence.quick_check == ("ok",)
        and evidence.schema_inventory_sha256 == train.apply_evidence.post.schema_inventory_sha256
        and consumer_ids == expected_ids
    )
    return DurableRestartConvergenceProof(
        evidence_ref=_require_nonempty(evidence_ref, label="restart convergence evidence"),
        observed_user_version=evidence.user_version,
        observed_schema_inventory_sha256=evidence.schema_inventory_sha256,
        consumer_ids=consumer_ids,
        converged=converged,
        observed_at_ms=observed_at_ms if observed_at_ms is not None else _durable_now_ms(),
    )


def prove_durable_change_train(
    train: DurableChangeTrain,
    *,
    fresh_ddl_parity: DurableFreshDDLParityProof,
    runtime_consumers: Sequence[DurableRuntimeConsumerResult],
    restart_convergence: DurableRestartConvergenceProof,
    proof_refs: Sequence[str] = (),
    proven_at_ms: int | None = None,
) -> DurableChangeTrain:
    """Prove actual migrated bytes, production behavior, and restart convergence."""
    if train.state is not DurableChangeTrainState.APPLIED or train.apply_evidence is None:
        raise DurableChangeTrainError(f"only an applied train may be proven, found {train.state.value}")
    validate_durable_change_train_manifest(train)
    if train.reservation is None or train.reservation.active:
        raise DurableChangeTrainError("writer reservation must be released before restart convergence proof")
    if train.fresh_ddl_parity is None:
        raise DurableChangeTrainError("admission fresh-DDL parity evidence is missing")
    if (
        not fresh_ddl_parity.matches
        or fresh_ddl_parity.tier is not train.tier
        or fresh_ddl_parity.target_version != train.target_version
        or fresh_ddl_parity.migrated_inventory_sha256 != train.apply_evidence.post.schema_inventory_sha256
        or fresh_ddl_parity.fresh_inventory_sha256 != train.fresh_ddl_parity.fresh_inventory_sha256
    ):
        raise DurableChangeTrainError("actual post-apply bytes do not have admitted fresh-DDL parity")
    expected_consumers = {
        consumer.consumer_id: consumer for rider in train.riders for consumer in rider.runtime_consumers
    }
    result_by_id: dict[str, DurableRuntimeConsumerResult] = {}
    for result in runtime_consumers:
        if result.consumer_id in result_by_id:
            raise DurableChangeTrainError(f"duplicate runtime proof result: {result.consumer_id}")
        result_by_id[result.consumer_id] = result
    if set(result_by_id) != set(expected_consumers):
        raise DurableChangeTrainError(
            "runtime proof does not cover the admitted consumers exactly: "
            f"expected={sorted(expected_consumers)}, observed={sorted(result_by_id)}"
        )
    for consumer_id, declared in expected_consumers.items():
        result = result_by_id[consumer_id]
        if not result.passed:
            raise DurableChangeTrainError(f"runtime consumer proof failed: {consumer_id}: {result.detail}")
        if result.behavior_proof_ref != declared.behavior_proof_ref:
            raise DurableChangeTrainError(
                f"runtime consumer {consumer_id} proved a different behavior ref than admission"
            )
    expected_ids = tuple(sorted(expected_consumers))
    if (
        not restart_convergence.converged
        or restart_convergence.observed_user_version != train.target_version
        or restart_convergence.observed_schema_inventory_sha256 != train.apply_evidence.post.schema_inventory_sha256
        or restart_convergence.consumer_ids != expected_ids
    ):
        raise DurableChangeTrainError("restart did not converge the exact target schema and runtime consumer set")
    if train.apply_evidence.pre.quick_check != ("ok",) or train.apply_evidence.post.quick_check != ("ok",):
        raise DurableChangeTrainError("pre/post integrity evidence is not successful")
    if not train.apply_evidence.row_parity.ok:
        raise DurableChangeTrainError("row parity is not successful")
    references = train.proof_refs
    references = _append_proof_refs(
        references,
        fresh_ddl_parity.evidence_ref,
        restart_convergence.evidence_ref,
        *[result.behavior_proof_ref for result in runtime_consumers],
        *proof_refs,
    )
    proof = DurableTrainProof(
        fresh_ddl_parity=fresh_ddl_parity,
        runtime_consumers=tuple(sorted(runtime_consumers, key=lambda item: item.consumer_id)),
        restart_convergence=restart_convergence,
        proof_refs=references,
        proven_at_ms=proven_at_ms if proven_at_ms is not None else _durable_now_ms(),
    )
    updated = replace(
        train,
        state=DurableChangeTrainState.PROVEN,
        revision=train.revision + 1,
        proof=proof,
        proof_refs=references,
    )
    validate_durable_change_train_manifest(updated)
    return updated


def release_durable_change_train(
    train: DurableChangeTrain,
    *,
    evidence_ref: str,
    released_at_ms: int | None = None,
) -> DurableChangeTrain:
    """Release contention ownership only after proof and restart convergence."""
    if train.state is not DurableChangeTrainState.PROVEN or train.proof is None:
        raise DurableChangeTrainError(f"only a proven train may release, found {train.state.value}")
    validate_durable_change_train_manifest(train)
    if train.reservation is None or train.reservation.active:
        raise DurableChangeTrainError("train release requires a released writer reservation")
    if not train.proof.restart_convergence.converged:
        raise DurableChangeTrainError("train release requires restart convergence")
    proof_ref = _require_nonempty(evidence_ref, label="train release evidence")
    updated = replace(
        train,
        state=DurableChangeTrainState.RELEASED,
        revision=train.revision + 1,
        released_at_ms=released_at_ms if released_at_ms is not None else _durable_now_ms(),
        release_evidence_ref=proof_ref,
        proof_refs=_append_proof_refs(train.proof_refs, proof_ref),
    )
    validate_durable_change_train_manifest(updated)
    return updated


def _validate_sha256(value: str, *, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise DurableChangeTrainError(f"{label} is not a lowercase SHA-256 digest")


def _validate_proof_refs(proof_refs: tuple[str, ...]) -> None:
    if len(proof_refs) != len(set(proof_refs)):
        raise DurableChangeTrainError("durable train proof references are not unique")
    for proof_ref in proof_refs:
        _require_nonempty(proof_ref, label="proof reference")


def _validate_admission_evidence(train: DurableChangeTrain) -> None:
    parity = train.fresh_ddl_parity
    if train.admitted_at_ms is None or train.admitted_at_ms < train.declared_at_ms:
        raise DurableChangeTrainError("durable train admission timestamp is invalid")
    admission_ref = _require_nonempty(
        train.admission_evidence_ref or "",
        label="admission evidence",
    )
    if parity is None:
        raise DurableChangeTrainError("durable train admission lacks fresh-DDL parity")
    if (
        parity.tier is not train.tier
        or parity.target_version != train.target_version
        or parity.migrated_version != train.target_version
        or parity.fresh_version != train.target_version
    ):
        raise DurableChangeTrainError("fresh-DDL parity does not bind the train tier and target")
    _validate_sha256(parity.migrated_inventory_sha256, label="migrated fresh-DDL inventory")
    _validate_sha256(parity.fresh_inventory_sha256, label="canonical fresh-DDL inventory")
    parity_ref = _require_nonempty(parity.evidence_ref, label="fresh-DDL parity evidence")
    if (
        not parity.matches
        or parity.missing_objects
        or parity.unexpected_objects
        or parity.changed_objects
        or parity.migrated_inventory_sha256 != parity.fresh_inventory_sha256
    ):
        raise DurableChangeTrainError("admitted fresh-DDL parity is not an exact match")
    required_refs = {admission_ref, parity_ref}
    if not required_refs.issubset(train.proof_refs):
        raise DurableChangeTrainError("admission proof references are not retained by the train")
    if train.migration.requires_backup and not train.backup_plan_ref:
        raise DurableChangeTrainError("backup-required train lacks its admitted backup plan")


def _validate_writer_reservation(train: DurableChangeTrain) -> None:
    reservation = train.reservation
    if reservation is None:
        raise DurableChangeTrainError(f"{train.state.value} manifest lacks writer reservation")
    if reservation.owner_ref != train.owner_ref:
        raise DurableChangeTrainError("writer reservation owner does not match train owner")
    _require_nonempty(reservation.reservation_id, label="writer reservation id")
    _require_nonempty(reservation.owner_ref, label="writer reservation owner")
    archive_root = Path(_require_nonempty(reservation.archive_root, label="archive root"))
    tier_path = Path(_require_nonempty(reservation.tier_path, label="reserved tier path"))
    if not archive_root.is_absolute() or not tier_path.is_absolute():
        raise DurableChangeTrainError("writer reservation paths must be absolute")
    expected_tier_path = archive_root / f"{train.tier.value}.db"
    if tier_path.resolve(strict=False) != expected_tier_path.resolve(strict=False):
        raise DurableChangeTrainError("writer reservation tier path does not match its archive root and tier")
    stopped_ref = _require_nonempty(
        reservation.daemon_stopped_evidence_ref,
        label="stopped-daemon evidence",
    )
    writer_ref = _require_nonempty(
        reservation.single_writer_evidence_ref,
        label="single-writer evidence",
    )
    if not {stopped_ref, writer_ref}.issubset(train.proof_refs):
        raise DurableChangeTrainError("writer authority proof references are not retained by the train")
    if reservation.reserved_at_ms < (train.admitted_at_ms or train.declared_at_ms):
        raise DurableChangeTrainError("writer reservation timestamp predates admission")
    if reservation.active:
        if reservation.released_at_ms is not None or reservation.release_evidence_ref is not None:
            raise DurableChangeTrainError("active writer reservation contains release evidence")
        return
    release_ref = _require_nonempty(
        reservation.release_evidence_ref or "",
        label="writer release evidence",
    )
    if reservation.released_at_ms is None or reservation.released_at_ms < reservation.reserved_at_ms:
        raise DurableChangeTrainError("released writer reservation has an invalid timestamp")
    lifecycle_floor = reservation.reserved_at_ms
    if train.apply_evidence is not None:
        lifecycle_floor = max(lifecycle_floor, train.apply_evidence.applied_at_ms)
    if train.failure is not None:
        lifecycle_floor = max(lifecycle_floor, train.failure.failed_at_ms)
    if reservation.released_at_ms < lifecycle_floor:
        raise DurableChangeTrainError("writer release timestamp predates apply/failure evidence")
    if release_ref not in train.proof_refs:
        raise DurableChangeTrainError("writer release evidence is not retained by the train")


def _validate_database_evidence(
    evidence: DurableDatabaseEvidence,
    train: DurableChangeTrain,
    *,
    expected_version: int,
    label: str,
) -> None:
    if evidence.tier is not train.tier or evidence.user_version != expected_version:
        raise DurableChangeTrainError(f"{label} does not bind the train tier/version")
    if evidence.quick_check != ("ok",):
        raise DurableChangeTrainError(f"{label} does not contain a successful quick_check")
    _validate_sha256(evidence.schema_inventory_sha256, label=f"{label} schema inventory")
    _validate_sha256(evidence.archive_identity_digest, label=f"{label} archive identity")
    _validate_sha256(evidence.content_sha256, label=f"{label} content")
    if evidence.observed_at_ms < train.declared_at_ms:
        raise DurableChangeTrainError(f"{label} timestamp predates train declaration")
    tables = [table for table, _count in evidence.row_counts]
    if tables != sorted(tables) or len(tables) != len(set(tables)):
        raise DurableChangeTrainError(f"{label} row census is not unique and canonical")
    for table, count in evidence.row_counts:
        _require_nonempty(table, label=f"{label} row-census table")
        if count < 0:
            raise DurableChangeTrainError(f"{label} row census contains a negative count")


def _validate_backup_authorization(train: DurableChangeTrain) -> None:
    authorization = train.backup_authorization
    reservation = train.reservation
    pre = train.pre_apply_evidence
    if authorization is None or reservation is None or pre is None:
        raise DurableChangeTrainError(f"{train.state.value} manifest lacks backup authorization or pre-apply evidence")
    evidence_ref = _require_nonempty(
        authorization.evidence_ref,
        label="backup authorization evidence",
    )
    if evidence_ref not in train.proof_refs:
        raise DurableChangeTrainError("backup authorization evidence is not retained by the train")
    if authorization.authorized_at_ms < reservation.reserved_at_ms:
        raise DurableChangeTrainError("backup authorization timestamp predates writer reservation")
    if authorization.live_user_version != train.current_version:
        raise DurableChangeTrainError("backup authorization is bound to the wrong live version")
    live_path = Path(_require_nonempty(authorization.live_tier_path, label="authorized live tier path"))
    if live_path.resolve(strict=False) != Path(reservation.tier_path).resolve(strict=False):
        raise DurableChangeTrainError("backup authorization live path differs from the writer reservation")
    if authorization.mode == "additive-no-backup":
        if train.migration.requires_backup:
            raise DurableChangeTrainError("backup-required migration has additive-no-backup authority")
        if any(
            value is not None
            for value in (
                authorization.manifest_path,
                authorization.manifest_sha256,
                authorization.receipt_path,
                authorization.receipt_sha256,
            )
        ):
            raise DurableChangeTrainError("additive-no-backup authority contains backup artifacts")
    elif authorization.mode == "verified-backup":
        manifest_path = _require_nonempty(
            authorization.manifest_path or "",
            label="authorized backup manifest path",
        )
        receipt_path = _require_nonempty(
            authorization.receipt_path or "",
            label="authorized backup receipt path",
        )
        if not Path(manifest_path).is_absolute() or not Path(receipt_path).is_absolute():
            raise DurableChangeTrainError("authorized backup artifact paths must be absolute")
        _validate_sha256(
            authorization.manifest_sha256 or "",
            label="authorized backup manifest",
        )
        _validate_sha256(
            authorization.receipt_sha256 or "",
            label="authorized backup receipt",
        )
    else:
        raise DurableChangeTrainError(f"unsupported durable backup authorization mode: {authorization.mode}")
    _validate_database_evidence(
        pre,
        train,
        expected_version=train.current_version,
        label="pre-apply evidence",
    )
    if pre.observed_at_ms < authorization.authorized_at_ms:
        raise DurableChangeTrainError("pre-apply evidence predates backup authorization")


def _validate_apply_evidence(train: DurableChangeTrain) -> None:
    evidence = train.apply_evidence
    pre = train.pre_apply_evidence
    authorization = train.backup_authorization
    if evidence is None or pre is None or authorization is None:
        raise DurableChangeTrainError(f"{train.state.value} manifest lacks apply evidence")
    if evidence.pre != pre:
        raise DurableChangeTrainError("apply evidence does not retain the authorized pre-apply census")
    _validate_database_evidence(
        evidence.post,
        train,
        expected_version=train.target_version,
        label="post-apply evidence",
    )
    if evidence.post.observed_at_ms < pre.observed_at_ms:
        raise DurableChangeTrainError("post-apply evidence predates pre-apply evidence")
    if evidence.applied_at_ms < evidence.post.observed_at_ms:
        raise DurableChangeTrainError("apply timestamp predates post-apply evidence")
    result = evidence.migration_result
    if (
        result.tier is not train.tier
        or result.from_version != train.current_version
        or result.to_version != train.target_version
        or result.applied_versions != (train.target_version,)
    ):
        raise DurableChangeTrainError("apply evidence does not describe the train's one exact migration slot")
    if authorization.mode == "verified-backup":
        if result.backup_receipt is None or authorization.receipt_path is None:
            raise DurableChangeTrainError("verified-backup apply evidence lacks its receipt")
        if result.backup_receipt.resolve(strict=False) != Path(authorization.receipt_path).resolve(strict=False):
            raise DurableChangeTrainError("apply evidence names a different backup receipt than authorization")
    elif result.backup_receipt is not None:
        raise DurableChangeTrainError("additive-no-backup apply evidence unexpectedly names a receipt")
    if not evidence.row_parity.ok or evidence.row_parity.unauthorized_changes:
        raise DurableChangeTrainError("apply evidence contains unsuccessful row parity")


def _validate_train_proof(train: DurableChangeTrain) -> None:
    proof = train.proof
    apply_evidence = train.apply_evidence
    admitted_parity = train.fresh_ddl_parity
    if proof is None or apply_evidence is None or admitted_parity is None:
        raise DurableChangeTrainError(f"{train.state.value} manifest lacks proof evidence")
    parity = proof.fresh_ddl_parity
    if (
        parity.tier is not train.tier
        or parity.target_version != train.target_version
        or parity.migrated_version != train.target_version
        or parity.fresh_version != train.target_version
        or not parity.matches
        or parity.missing_objects
        or parity.unexpected_objects
        or parity.changed_objects
        or parity.migrated_inventory_sha256 != apply_evidence.post.schema_inventory_sha256
        or parity.fresh_inventory_sha256 != admitted_parity.fresh_inventory_sha256
    ):
        raise DurableChangeTrainError("proof fresh-DDL parity does not bind admitted and applied schema bytes")
    _validate_sha256(parity.migrated_inventory_sha256, label="proof migrated inventory")
    _validate_sha256(parity.fresh_inventory_sha256, label="proof fresh inventory")
    expected_consumers = {
        consumer.consumer_id: consumer for rider in train.riders for consumer in rider.runtime_consumers
    }
    consumer_ids = tuple(result.consumer_id for result in proof.runtime_consumers)
    if consumer_ids != tuple(sorted(expected_consumers)):
        raise DurableChangeTrainError("proof runtime consumers do not exactly match the admitted consumer set")
    for result in proof.runtime_consumers:
        declared = expected_consumers[result.consumer_id]
        if not result.passed or result.behavior_proof_ref != declared.behavior_proof_ref:
            raise DurableChangeTrainError(f"runtime consumer proof is invalid: {result.consumer_id}")
    restart = proof.restart_convergence
    if (
        not restart.converged
        or restart.observed_user_version != train.target_version
        or restart.observed_schema_inventory_sha256 != apply_evidence.post.schema_inventory_sha256
        or restart.consumer_ids != consumer_ids
    ):
        raise DurableChangeTrainError("restart convergence does not bind the target schema and consumers")
    _validate_sha256(
        restart.observed_schema_inventory_sha256,
        label="restart schema inventory",
    )
    if restart.observed_at_ms < apply_evidence.applied_at_ms:
        raise DurableChangeTrainError("restart convergence timestamp predates apply")
    reservation = train.reservation
    if reservation is None or reservation.released_at_ms is None or restart.observed_at_ms < reservation.released_at_ms:
        raise DurableChangeTrainError("restart convergence timestamp predates writer release")
    if proof.proven_at_ms < restart.observed_at_ms:
        raise DurableChangeTrainError("proof timestamp predates restart convergence")
    _validate_proof_refs(proof.proof_refs)
    required_refs = {
        parity.evidence_ref,
        restart.evidence_ref,
        *(result.behavior_proof_ref for result in proof.runtime_consumers),
    }
    if not required_refs.issubset(proof.proof_refs) or not set(proof.proof_refs).issubset(train.proof_refs):
        raise DurableChangeTrainError("proof bundle references are incomplete or not retained by the train")


def _validate_failure_evidence(train: DurableChangeTrain) -> None:
    failure = train.failure
    if failure is None:
        raise DurableChangeTrainError("failed manifest lacks exact recovery evidence")
    if failure.previous_state is not DurableChangeTrainState.BACKUP_AUTHORIZED:
        raise DurableChangeTrainError("failed manifest did not originate from backup-authorized apply")
    _require_nonempty(failure.phase, label="failure phase")
    _require_nonempty(failure.error_type, label="failure error type")
    _require_nonempty(failure.error_message, label="failure error message")
    if not failure.required_actions:
        raise DurableChangeTrainError("failed manifest has no operator recovery actions")
    for action in failure.required_actions:
        _require_nonempty(action, label="failure recovery action")
    if failure.failed_at_ms < (train.pre_apply_evidence.observed_at_ms if train.pre_apply_evidence else 0):
        raise DurableChangeTrainError("failure timestamp predates pre-apply evidence")
    if failure.pre_apply_evidence != train.pre_apply_evidence:
        raise DurableChangeTrainError("failure does not retain the authorized pre-apply evidence")
    if failure.observed_user_version == train.current_version:
        expected = DurableFailureClassification.ROLLED_BACK_TO_CURRENT
    elif failure.observed_user_version == train.target_version:
        expected = DurableFailureClassification.COMMITTED_TARGET_UNPROVEN
    else:
        expected = DurableFailureClassification.INDETERMINATE
    if failure.classification is not expected:
        raise DurableChangeTrainError("failure classification does not match the observed durable version")
    if (
        failure.classification is DurableFailureClassification.INDETERMINATE
        and train.reservation is not None
        and not train.reservation.active
    ):
        raise DurableChangeTrainError("indeterminate failure released stopped-daemon/single-writer authority")


def validate_durable_change_train_manifest(train: DurableChangeTrain) -> None:
    """Validate cross-field lifecycle invariants for loaded and transitioned manifests."""
    if train.manifest_format != DURABLE_CHANGE_TRAIN_FORMAT:
        raise DurableChangeTrainError(f"unsupported durable change train format: {train.manifest_format}")
    if train.source_continuity_evidence is not None and (
        train.tier is not ArchiveTier.SOURCE or train.state is not DurableChangeTrainState.RELEASED
    ):
        raise DurableChangeTrainError("source continuity evidence is only valid on a released source train")
    if train.tier not in DURABLE_MIGRATION_TIERS:
        raise DurableChangeTrainError(f"manifest tier is not durable: {train.tier.value}")
    if train.current_version < 1 or train.target_version != train.current_version + 1:
        raise DurableChangeTrainError("manifest does not own one contiguous target version")
    if train.slot != train.target_version:
        raise DurableChangeTrainError("manifest numbered slot does not equal its target version")
    if train.migration.contention_key != train.contention_key or train.migration.tier is not train.tier:
        raise DurableChangeTrainError("manifest migration claim does not match train contention key")
    _require_nonempty(train.migration.path, label="numbered migration path")
    _require_nonempty(train.migration.owner_ref, label="numbered migration owner")
    _validate_sha256(train.migration.sql_sha256, label="numbered migration SQL")
    if train.revision < 0 or train.declared_at_ms < 0:
        raise DurableChangeTrainError("manifest revision/timestamp is invalid")
    _require_nonempty(train.train_id, label="train_id")
    _require_nonempty(train.owner_ref, label="owner_ref")
    _validate_proof_refs(train.proof_refs)
    if train.state is DurableChangeTrainState.DECLARED:
        if train.proof_refs or any(
            value is not None
            for value in (
                train.admitted_at_ms,
                train.admission_evidence_ref,
                train.fresh_ddl_parity,
                train.reservation,
                train.backup_authorization,
                train.pre_apply_evidence,
                train.apply_evidence,
                train.proof,
                train.failure,
                train.released_at_ms,
                train.release_evidence_ref,
            )
        ):
            raise DurableChangeTrainError("declared manifest contains evidence from a later lifecycle state")
        return
    _validate_riders(train)
    _validate_admission_evidence(train)
    if train.state is DurableChangeTrainState.ADMITTED:
        if any(
            value is not None
            for value in (
                train.reservation,
                train.backup_authorization,
                train.pre_apply_evidence,
                train.apply_evidence,
                train.proof,
                train.failure,
                train.released_at_ms,
                train.release_evidence_ref,
            )
        ):
            raise DurableChangeTrainError("admitted manifest contains later lifecycle evidence")
        return
    _validate_writer_reservation(train)
    if train.state is DurableChangeTrainState.RESERVED:
        if (
            train.reservation is None
            or not train.reservation.active
            or any(
                value is not None
                for value in (
                    train.backup_authorization,
                    train.pre_apply_evidence,
                    train.apply_evidence,
                    train.proof,
                    train.failure,
                    train.released_at_ms,
                    train.release_evidence_ref,
                )
            )
        ):
            raise DurableChangeTrainError("reserved manifest has invalid reservation/later evidence")
        return
    _validate_backup_authorization(train)
    if train.state is DurableChangeTrainState.FAILED:
        if any(
            value is not None
            for value in (
                train.apply_evidence,
                train.proof,
                train.released_at_ms,
                train.release_evidence_ref,
            )
        ):
            raise DurableChangeTrainError("failed manifest contains apply/proof/release evidence")
        _validate_failure_evidence(train)
        return
    if train.failure is not None:
        raise DurableChangeTrainError(f"{train.state.value} manifest unexpectedly retains failure evidence")
    if train.state is DurableChangeTrainState.BACKUP_AUTHORIZED:
        if (
            train.reservation is None
            or not train.reservation.active
            or any(
                value is not None
                for value in (
                    train.apply_evidence,
                    train.proof,
                    train.released_at_ms,
                    train.release_evidence_ref,
                )
            )
        ):
            raise DurableChangeTrainError("backup-authorized manifest contains invalid later evidence")
        return
    _validate_apply_evidence(train)
    apply_evidence = train.apply_evidence
    if train.state is DurableChangeTrainState.APPLIED:
        if train.proof is not None or train.released_at_ms is not None or train.release_evidence_ref is not None:
            raise DurableChangeTrainError("applied manifest contains proof/release evidence")
        return
    if train.reservation is None or train.reservation.active:
        raise DurableChangeTrainError(f"{train.state.value} manifest still owns the writer")
    _validate_train_proof(train)
    if train.state is DurableChangeTrainState.PROVEN:
        if train.released_at_ms is not None or train.release_evidence_ref is not None:
            raise DurableChangeTrainError("proven manifest contains release evidence")
        if train.proof is None or train.proof.proof_refs != train.proof_refs:
            raise DurableChangeTrainError("proven manifest proof references differ from its proof bundle")
        return
    if train.state is DurableChangeTrainState.RELEASED:
        release_ref = _require_nonempty(
            train.release_evidence_ref or "",
            label="train release evidence",
        )
        if train.released_at_ms is None or train.proof is None:
            raise DurableChangeTrainError("released manifest lacks release evidence")
        if train.released_at_ms < train.proof.proven_at_ms:
            raise DurableChangeTrainError("train release timestamp predates proof")
        if release_ref not in train.proof_refs:
            raise DurableChangeTrainError("train release evidence is not retained by the manifest")
        if apply_evidence is None:
            raise DurableChangeTrainError("released manifest lacks apply evidence")
        if train.source_continuity_evidence is not None:
            _validate_database_evidence(
                train.source_continuity_evidence,
                train,
                expected_version=train.target_version,
                label="source continuity evidence",
            )
            apply_post = apply_evidence.post
            refreshed = train.source_continuity_evidence
            if (
                refreshed.schema_inventory_sha256 != apply_post.schema_inventory_sha256
                or refreshed.archive_identity_digest != apply_post.archive_identity_digest
            ):
                raise DurableChangeTrainError("source continuity evidence changed schema or archive identity")
            if refreshed.observed_at_ms < train.released_at_ms:
                raise DurableChangeTrainError("source continuity evidence predates train release")
            if not any(ref.startswith("proof:source-continuity-refresh:") for ref in train.proof_refs):
                raise DurableChangeTrainError("source continuity evidence is not retained by the train")
        return
    raise DurableChangeTrainError(f"unknown durable change train state: {train.state}")


def reconcile_interrupted_durable_change_train(
    conn: sqlite3.Connection,
    train: DurableChangeTrain,
    *,
    interruption_evidence_ref: str,
    writer_release_evidence_ref: str,
) -> DurableChangeTrain:
    """Reconcile a persisted pre-apply manifest after process interruption.

    A crash can leave the manifest in ``backup-authorized`` while SQLite is
    either still at ``current`` (transaction rolled back) or already at
    ``target`` (commit completed before the manifest write).  No other version
    is safe to infer.
    """
    if train.state is not DurableChangeTrainState.BACKUP_AUTHORIZED:
        raise DurableChangeTrainError(
            f"interruption reconciliation requires backup-authorized state, found {train.state.value}"
        )
    validate_durable_change_train_manifest(train)
    evidence_ref = _require_nonempty(interruption_evidence_ref, label="interruption evidence")
    observed = _safe_user_version(conn)
    interrupted = RuntimeError(f"operator observed interrupted apply ({evidence_ref})")
    failed = _failed_change_train(
        train,
        phase="apply-interrupted",
        error=interrupted,
        observed_user_version=observed,
        pre_apply_evidence=train.pre_apply_evidence,
    )
    if failed.failure is None:
        raise DurableChangeTrainError("interruption reconciliation failed to classify the train")
    if failed.failure.classification is DurableFailureClassification.INDETERMINATE:
        raise DurableChangeTrainRecoveryError(
            "interrupted durable train reached an indeterminate version; "
            "keep the daemon stopped and restore the exact authenticated backup",
            failed_train=failed,
        )
    recovered = recover_durable_change_train(
        conn,
        failed,
        recovery_evidence_ref=evidence_ref,
        writer_release_evidence_ref=writer_release_evidence_ref,
    )
    # The failed classification is an in-memory recovery decision. The
    # durable manifest records one atomic startup-recovery transition and
    # retains the recovery evidence in its proof references.
    recovered = replace(recovered, revision=train.revision + 1)
    validate_durable_change_train_manifest(recovered)
    return recovered


def _manifest_json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _manifest_json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, tuple | list):
        return [_manifest_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _manifest_json_value(item) for key, item in value.items()}
    if value is None or isinstance(value, str | int | bool | float):
        return value
    raise DurableChangeTrainError(f"durable manifest contains unsupported value type: {type(value).__name__}")


def durable_change_train_to_payload(train: DurableChangeTrain) -> dict[str, object]:
    """Encode one immutable train with a canonical payload checksum."""
    validate_durable_change_train_manifest(train)
    encoded = _manifest_json_value(train)
    if not isinstance(encoded, dict):
        raise DurableChangeTrainError("durable change train did not encode as an object")
    payload = cast(dict[str, object], encoded)
    payload["manifest_sha256"] = _canonical_json_sha256(payload)
    return payload


def _decode_manifest_value(annotation: object, value: object, *, label: str) -> object:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is types.UnionType:
        non_none = tuple(item for item in args if item is not type(None))
        if value is None and len(non_none) != len(args):
            return None
        errors: list[str] = []
        for candidate in non_none:
            try:
                return _decode_manifest_value(candidate, value, label=label)
            except DurableChangeTrainError as exc:
                errors.append(str(exc))
        raise DurableChangeTrainError(f"{label} does not match its union type: {'; '.join(errors)}")
    if origin is tuple:
        if not isinstance(value, list):
            raise DurableChangeTrainError(f"{label} must be a JSON array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                _decode_manifest_value(args[0], item, label=f"{label}[{index}]") for index, item in enumerate(value)
            )
        if len(value) != len(args):
            raise DurableChangeTrainError(f"{label} requires {len(args)} entries, found {len(value)}")
        return tuple(
            _decode_manifest_value(item_type, item, label=f"{label}[{index}]")
            for index, (item_type, item) in enumerate(zip(args, value, strict=True))
        )
    if annotation is str:
        if not isinstance(value, str):
            raise DurableChangeTrainError(f"{label} must be a string")
        return value
    if annotation is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise DurableChangeTrainError(f"{label} must be an integer")
        return value
    if annotation is bool:
        if not isinstance(value, bool):
            raise DurableChangeTrainError(f"{label} must be a boolean")
        return value
    if annotation is float:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise DurableChangeTrainError(f"{label} must be a number")
        return float(value)
    if annotation is Path:
        if not isinstance(value, str):
            raise DurableChangeTrainError(f"{label} must be a path string")
        return Path(value)
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        if not isinstance(value, str):
            raise DurableChangeTrainError(f"{label} must be an enum string")
        try:
            return annotation(value)
        except ValueError as exc:
            raise DurableChangeTrainError(f"{label} has unsupported value {value!r}") from exc
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, dict):
            raise DurableChangeTrainError(f"{label} must be a JSON object")
        dataclass_fields = fields(annotation)
        expected = {item.name for item in dataclass_fields}
        observed = set(value)
        missing = expected - observed
        unexpected = observed - expected
        if missing or unexpected:
            raise DurableChangeTrainError(
                f"{label} fields differ: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        hints = get_type_hints(annotation)
        decoded = {
            item.name: _decode_manifest_value(
                hints[item.name],
                value[item.name],
                label=f"{label}.{item.name}",
            )
            for item in dataclass_fields
        }
        return annotation(**decoded)
    raise DurableChangeTrainError(f"{label} has unsupported manifest annotation {annotation!r}")


def durable_change_train_from_payload(payload: Mapping[str, object]) -> DurableChangeTrain:
    """Strictly decode and validate a checksummed durable train manifest."""
    mutable = dict(payload)
    checksum = mutable.pop("manifest_sha256", None)
    if not isinstance(checksum, str) or checksum != _canonical_json_sha256(mutable):
        raise DurableChangeTrainError("durable change train manifest checksum mismatch")
    # v1 manifests written before source continuity refreshes omitted this
    # optional field. Preserve their checksum and decode them as no refresh.
    mutable.setdefault("source_continuity_evidence", None)
    decoded = _decode_manifest_value(DurableChangeTrain, mutable, label="train")
    if not isinstance(decoded, DurableChangeTrain):
        raise DurableChangeTrainError("durable change train payload decoded to the wrong type")
    validate_durable_change_train_manifest(decoded)
    return decoded


def _require_safe_manifest_parent(path: Path) -> Path:
    parent = path.parent
    try:
        metadata = parent.lstat()
    except FileNotFoundError as exc:
        raise DurableChangeTrainError(f"durable change train parent directory is missing: {parent}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise DurableChangeTrainError(f"durable change train parent is a symbolic link: {parent}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise DurableChangeTrainError(f"durable change train parent is not a real directory: {parent}")
    absolute = Path(os.path.abspath(parent))
    resolved = parent.resolve(strict=True)
    if resolved != absolute:
        raise DurableChangeTrainError(f"durable change train parent traverses a symbolic link: {parent}")
    return resolved


def load_durable_change_train_manifest(path: Path) -> DurableChangeTrain:
    """Load one real, single-linked, checksummed authority file."""
    _require_safe_manifest_parent(path)
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise DurableChangeTrainError(f"durable change train manifest is missing: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise DurableChangeTrainError(f"durable change train manifest is not a real single-linked file: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DurableChangeTrainError(f"durable change train manifest is invalid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise DurableChangeTrainError(f"durable change train manifest must be a JSON object: {path}")
    return durable_change_train_from_payload(cast(dict[str, object], raw))


def _fsync_manifest_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _acquire_manifest_write_lock(path: Path) -> int:
    lock_path = path.with_name(f".{path.name}.lock")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise DurableChangeTrainError(f"cannot open durable train manifest lock safely: {lock_path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise DurableChangeTrainError(f"durable train manifest lock is not a real single-linked file: {lock_path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def write_durable_change_train_manifest(
    path: Path,
    train: DurableChangeTrain,
    *,
    expected_revision: int,
) -> None:
    """Atomically persist authority with serialized optimistic revision control."""
    validate_durable_change_train_manifest(train)
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = _require_safe_manifest_parent(path)
    lock_descriptor = _acquire_manifest_write_lock(path)
    temporary: Path | None = None
    try:
        if path.exists() or path.is_symlink():
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise DurableChangeTrainError(f"refusing to replace unsafe train manifest path: {path}")
            current = load_durable_change_train_manifest(path)
            if current.revision != expected_revision:
                raise DurableChangeTrainError(
                    f"durable train manifest revision changed: expected {expected_revision}, found {current.revision}"
                )
            if train.revision != expected_revision + 1:
                raise DurableChangeTrainError(
                    "durable train manifest update must advance exactly one revision: "
                    f"current={expected_revision}, proposed={train.revision}"
                )
        elif expected_revision != -1:
            raise DurableChangeTrainError(
                f"durable train manifest does not exist for expected revision {expected_revision}: {path}"
            )
        payload = durable_change_train_to_payload(train)
        encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise DurableChangeTrainError("durable train manifest write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        temporary = None
        _fsync_manifest_directory(parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock_descriptor)


__all__ = [
    "DURABLE_CHANGE_TRAIN_FORMAT",
    "DURABLE_MIGRATION_COLLISION_REPORT_FORMAT",
    "DURABLE_MIGRATION_TIERS",
    "DurableApplyEvidence",
    "DurableBackupAuthorization",
    "DurableChangeRider",
    "DurableChangeTrain",
    "DurableChangeTrainApplyError",
    "DurableChangeTrainError",
    "DurableChangeTrainRecoveryError",
    "DurableChangeTrainState",
    "DurableDatabaseEvidence",
    "DurableDropConstraint",
    "DurableFailureClassification",
    "DurableFreshDDLParityProof",
    "DurableMigrationClaim",
    "DurableMigrationCollision",
    "DurableOrderingConstraint",
    "DurableRestartConvergenceProof",
    "DurableRowChangeAllowance",
    "DurableRowParityProof",
    "DurableRuntimeConsumer",
    "DurableRuntimeConsumerResult",
    "DurableSchemaInventory",
    "DurableSchemaObjectEvidence",
    "DurableTrainFailure",
    "DurableTrainProof",
    "DurableWriterReservation",
    "MigrationError",
    "MigrationResult",
    "MigrationStep",
    "add_durable_change_train_rider",
    "admit_durable_change_train",
    "apply_durable_change_train",
    "authorize_durable_change_train_backup",
    "capture_durable_database_evidence",
    "capture_durable_restart_convergence",
    "capture_durable_schema_inventory",
    "declare_durable_change_train",
    "durable_change_train_from_payload",
    "durable_change_train_to_payload",
    "durable_migration_claim_for_sql",
    "durable_migration_claims",
    "durable_migration_collision_report",
    "find_durable_migration_collisions",
    "load_durable_change_train_manifest",
    "migrate_archive_tier",
    "prove_durable_change_train",
    "prove_durable_fresh_ddl_parity",
    "prove_durable_row_parity",
    "reconcile_interrupted_durable_change_train",
    "record_durable_writer_release",
    "recover_durable_change_train",
    "release_durable_change_train",
    "reserve_durable_change_train",
    "validate_durable_change_train_manifest",
    "validate_backup_manifest_covers_derived_tier",
    "validate_migration_backup_live_fingerprint",
    "validate_full_evidence_backup_for_audit_adoption",
    "validate_full_evidence_backup_for_adopted_audit_restore",
    "validate_migration_backup_manifest",
    "write_durable_change_train_manifest",
]
