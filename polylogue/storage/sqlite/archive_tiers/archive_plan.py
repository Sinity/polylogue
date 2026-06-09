"""Read-only planning for archive initialization."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


class ArchiveInitAction(StrEnum):
    """Operator action needed for one archive tier file."""

    CREATE = "create"
    REPLACE_WITH_BACKUP = "replace_with_backup"
    RECREATE_DISPOSABLE = "recreate_disposable"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ArchiveTierPlan:
    """Planned handling for one durability-tier database file."""

    tier: ArchiveTier
    path: Path
    durability: str
    exists: bool
    user_version: int | None
    expected_user_version: int
    backup_required: bool
    action: ArchiveInitAction
    backup_path: Path | None
    blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArchiveInitPlan:
    """Complete read-only initialization plan for an archive root."""

    archive_root: Path
    tiers: tuple[ArchiveTierPlan, ...]
    blockers: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.blockers


def build_archive_init_plan(
    *,
    archive_root: Path,
    replace_existing: bool = False,
) -> ArchiveInitPlan:
    """Inspect the current archive root and return the initialization plan.

    The planner is deliberately read-only. It does not create backups,
    initialize archive files, or delete disposable state; it only determines
    whether those operations are safe to start and records the exact file
    targets the mutating initialization command must use.
    """

    resolved_archive_root = archive_root.expanduser()

    blockers: list[str] = []

    tier_plans: list[ArchiveTierPlan] = []
    for tier in ArchiveTier:
        tier_plan = _plan_tier(
            archive_root=resolved_archive_root,
            tier=tier,
            replace_existing=replace_existing,
        )
        tier_plans.append(tier_plan)
        blockers.extend(tier_plan.blockers)

    return ArchiveInitPlan(
        archive_root=resolved_archive_root,
        tiers=tuple(tier_plans),
        blockers=tuple(blockers),
    )


def _plan_tier(
    *,
    archive_root: Path,
    tier: ArchiveTier,
    replace_existing: bool,
) -> ArchiveTierPlan:
    spec = ARCHIVE_TIER_SPECS[tier]
    path = archive_root / spec.filename
    exists = path.exists()
    user_version = _read_user_version(path) if exists and path.is_file() else None
    backup_path = path.with_name(f"{path.name}.pre-archive-init.bak") if exists and spec.backup_required else None
    blockers: list[str] = []

    if exists and not replace_existing:
        blockers.append(f"{tier.value} target already exists; rerun with replace_existing after backing it up: {path}")
        action = ArchiveInitAction.BLOCKED
    elif exists and spec.backup_required:
        action = ArchiveInitAction.REPLACE_WITH_BACKUP
    elif exists:
        action = ArchiveInitAction.RECREATE_DISPOSABLE
    else:
        action = ArchiveInitAction.CREATE

    return ArchiveTierPlan(
        tier=tier,
        path=path,
        durability=spec.durability,
        exists=exists,
        user_version=user_version,
        expected_user_version=spec.version,
        backup_required=spec.backup_required,
        action=action,
        backup_path=backup_path,
        blockers=tuple(blockers),
    )


def _read_user_version(path: Path) -> int | None:
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])
    except sqlite3.Error:
        return None
    finally:
        conn.close()


__all__ = [
    "ArchiveInitAction",
    "ArchiveInitPlan",
    "ArchiveTierPlan",
    "build_archive_init_plan",
]
