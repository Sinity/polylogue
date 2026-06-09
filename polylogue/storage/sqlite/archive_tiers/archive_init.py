"""Mutating archive initialization operations."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.archive_plan import (
    ArchiveInitAction,
    ArchiveInitPlan,
    ArchiveTierPlan,
    build_archive_init_plan,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database


class ArchiveInitBlockedError(RuntimeError):
    """Raised when archive initialization is requested for a blocked plan."""


@dataclass(frozen=True, slots=True)
class ArchiveTierInitResult:
    """Mutation result for one archive tier file."""

    tier: str
    path: Path
    action: ArchiveInitAction
    backup_path: Path | None
    initialized: bool


@dataclass(frozen=True, slots=True)
class ArchiveInitResult:
    """Mutation result for archive initialization."""

    tier_results: tuple[ArchiveTierInitResult, ...]


def initialize_archive_tier_files(
    *,
    archive_root: Path,
    replace_existing: bool = False,
) -> ArchiveInitResult:
    """Create the archive database file set after backup planning."""
    plan = build_archive_init_plan(
        archive_root=archive_root,
        replace_existing=replace_existing,
    )
    return initialize_archive_tier_files_from_plan(plan)


def initialize_archive_tier_files_from_plan(plan: ArchiveInitPlan) -> ArchiveInitResult:
    """Execute a previously inspected archive initialization plan."""
    if not plan.ready:
        raise ArchiveInitBlockedError("; ".join(plan.blockers))

    tier_results = tuple(_initialize_tier(tier_plan) for tier_plan in plan.tiers)
    return ArchiveInitResult(
        tier_results=tier_results,
    )


def _initialize_tier(tier_plan: ArchiveTierPlan) -> ArchiveTierInitResult:
    backup_path: Path | None = None
    if tier_plan.path.exists():
        if tier_plan.action is ArchiveInitAction.REPLACE_WITH_BACKUP:
            backup_path = tier_plan.backup_path
            if backup_path is None:
                raise ArchiveInitBlockedError(f"missing backup path for {tier_plan.tier.value}")
            _backup_sqlite_database(tier_plan.path, backup_path)
            tier_plan.path.unlink()
        elif tier_plan.action is ArchiveInitAction.RECREATE_DISPOSABLE:
            tier_plan.path.unlink()
        elif tier_plan.action is ArchiveInitAction.BLOCKED:
            raise ArchiveInitBlockedError(f"blocked archive initialization action for {tier_plan.tier.value}")

    initialize_archive_database(tier_plan.path, tier_plan.tier)
    return ArchiveTierInitResult(
        tier=tier_plan.tier.value,
        path=tier_plan.path,
        action=tier_plan.action,
        backup_path=backup_path,
        initialized=True,
    )


def _backup_sqlite_database(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    for suffix in ("-wal", "-shm"):
        sidecar = source.with_name(source.name + suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, destination.with_name(destination.name + suffix))


__all__ = [
    "ArchiveInitBlockedError",
    "ArchiveInitResult",
    "ArchiveTierInitResult",
    "initialize_archive_tier_files",
    "initialize_archive_tier_files_from_plan",
]
