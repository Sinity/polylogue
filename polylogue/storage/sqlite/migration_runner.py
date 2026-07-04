"""Versioned additive migrations for durable archive tiers."""

from __future__ import annotations

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


def validate_migration_backup_manifest(path: Path, tier: ArchiveTier) -> Path:
    """Validate that ``path`` is a backup manifest containing ``tier``."""
    manifest_path = _backup_manifest_path(path)
    if not manifest_path.exists():
        raise MigrationError(f"migration requires an existing backup manifest; missing {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MigrationError(f"migration backup manifest is not valid JSON: {manifest_path}") from exc
    if payload.get("format") != "polylogue-backup-v1":
        raise MigrationError(f"migration backup manifest has unsupported format: {manifest_path}")
    included = {str(item) for item in payload.get("included_tiers") or []}
    if f"{tier.value}.db" not in included:
        raise MigrationError(f"migration backup manifest does not include {tier.value}.db: {manifest_path}")
    return manifest_path


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
    validate_migration_backup_manifest(backup_manifest, tier)
    current_version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
    target_version = ARCHIVE_VERSION_BY_TIER[tier]
    if current_version == target_version:
        return MigrationResult(tier=tier, from_version=current_version, to_version=target_version, applied_versions=())
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
    try:
        conn.execute("BEGIN IMMEDIATE")
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
        conn.rollback()
        raise
    else:
        conn.commit()
    return MigrationResult(
        tier=tier,
        from_version=start_version,
        to_version=target_version,
        applied_versions=tuple(applied),
    )


__all__ = [
    "DURABLE_MIGRATION_TIERS",
    "MigrationError",
    "MigrationResult",
    "MigrationStep",
    "migrate_archive_tier",
    "validate_migration_backup_manifest",
]
