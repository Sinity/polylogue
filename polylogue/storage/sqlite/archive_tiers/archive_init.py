"""Mutating archive initialization operations."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from polylogue.storage.archive_identity import (
    ArchiveLocation,
    OwnedArchiveLocation,
    assert_owns_archive_location,
)
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

    # Archive-init replaces durable tiers and recreates disposable state. Hold
    # the same stable archive ownership token as other offline maintenance so
    # a live daemon (or a competing rebuild) is refused before any backup,
    # unlink, or initialization can touch a target.
    plan.archive_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    owner = OwnedArchiveLocation.acquire(
        ArchiveLocation.resolve(plan.archive_root),
        owner_id=f"archive-init:{os.getpid()}",
    )
    with owner:
        generation_rotated = [False]
        tier_results = tuple(
            _initialize_tier(tier_plan, owner=owner, generation_rotated=generation_rotated) for tier_plan in plan.tiers
        )
    return ArchiveInitResult(
        tier_results=tier_results,
    )


def _initialize_tier(
    tier_plan: ArchiveTierPlan,
    *,
    owner: OwnedArchiveLocation | None = None,
    generation_rotated: list[bool] | None = None,
) -> ArchiveTierInitResult:
    def assert_current_owner() -> None:
        if owner is not None:
            # Replacing the active index intentionally changes its inode. Once
            # that target has been unlinked, validate the pinned root without
            # treating the expected generation rotation as a competing writer.
            if (generation_rotated and generation_rotated[0]) or (
                tier_plan.tier.value == "index" and not tier_plan.path.exists()
            ):
                assert_owns_archive_location(owner, owner.location)
            else:
                assert_owns_archive_location(owner, ArchiveLocation.resolve(tier_plan.path.parent))

    backup_path: Path | None = None
    if tier_plan.path.exists():
        if tier_plan.action is ArchiveInitAction.REPLACE_WITH_BACKUP:
            backup_path = tier_plan.backup_path
            if backup_path is None:
                raise ArchiveInitBlockedError(f"missing backup path for {tier_plan.tier.value}")
            assert_current_owner()
            _backup_sqlite_database(tier_plan.path, backup_path)
            assert_current_owner()
            tier_plan.path.unlink()
            if generation_rotated is not None and tier_plan.tier.value == "index":
                generation_rotated[0] = True
        elif tier_plan.action is ArchiveInitAction.RECREATE_DISPOSABLE:
            assert_current_owner()
            tier_plan.path.unlink()
            if generation_rotated is not None and tier_plan.tier.value == "index":
                generation_rotated[0] = True
        elif tier_plan.action is ArchiveInitAction.BLOCKED:
            raise ArchiveInitBlockedError(f"blocked archive initialization action for {tier_plan.tier.value}")

    assert_current_owner()
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
