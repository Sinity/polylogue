"""Apply declared durable-tier migrations when the daemon opens an archive.

Durable tiers (``source``, ``user``, ``audit``) evolve only by numbered
migrations shipped under ``storage/sqlite/migrations/<tier>/``. Opening an
archive whose durable tier stands below this runtime's declared version is
ordinary lifecycle, not maintenance: the daemon applies the pending durable
change trains before it serves anything, while it holds exclusive archive
ownership. A step that changes existing data (``requires_backup``) runs only
behind a backup the daemon takes here and scratch-verifies; the train then
binds that backup's authenticated receipt to the exact pre-apply bytes.

An additive step needs no backup. A tier already at the runtime's version, or
absent from a fresh root that bootstrap is about to create, is left alone.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

from polylogue.operations.durable_change_train import (
    PendingDurableMigration,
    execute_durable_change_train,
    pending_durable_migrations,
)
from polylogue.storage.archive_identity import OwnedArchiveLocation


def _backup_profile(tier: str) -> str:
    # ``user_overlays`` carries user.db and audit.db without blobs; source.db
    # travels with the blobs it references.
    return "rebuildable_cache_exclude" if tier == "source" else "user_overlays"


def pre_migration_backup(
    archive_root: Path, migration: PendingDurableMigration, *, archive_owner: OwnedArchiveLocation
) -> Path:
    """Take and scratch-verify the backup a data-changing migration requires."""

    from polylogue.daemon.backup import backup_archive

    output_dir = (
        archive_root
        / ".maintenance-state"
        / "pre-migration-backups"
        / f"{migration.tier.value}-v{migration.current_version}-to-v{migration.target_version}-{int(time.time() * 1000)}"
    )
    result = backup_archive(
        output_dir=output_dir,
        verify=True,
        profile=_backup_profile(migration.tier.value),  # type: ignore[arg-type]
        archive_root_path=archive_root,
        archive_owner=archive_owner,
    )
    if not result.ok or result.output_path is None:
        raise RuntimeError(
            f"pre-migration backup of {migration.tier.value}.db failed; refusing to migrate: {result.error}"
        )
    return Path(result.output_path) / "manifest.json"


def apply_declared_durable_migrations(
    archive_root: Path,
    *,
    archive_owner: OwnedArchiveLocation,
    write_lease: Callable[[str], AbstractContextManager[object]],
) -> tuple[PendingDurableMigration, ...]:
    """Apply every pending durable migration; return what was applied.

    The caller holds exclusive archive ownership for the whole call and keeps
    it afterwards, so the train's ownership release is a no-op here. Each
    backup and train runs inside ``write_lease`` so it is one writer with the
    daemon, not beside it.
    """

    applied: list[PendingDurableMigration] = []
    for migration in pending_durable_migrations(archive_root):
        with write_lease("daemon.durable_migration.apply"):
            manifest = (
                pre_migration_backup(archive_root, migration, archive_owner=archive_owner)
                if migration.requires_backup
                else None
            )
            execute_durable_change_train(
                archive_root,
                migration.tier,
                backup_manifest=manifest,
                daemon_stopped_evidence_ref="proof:daemon-open-before-serving",
                single_writer_evidence_ref="proof:archive-ownership-lock",
                release_archive_ownership=lambda: None,
            )
        applied.append(migration)
    return tuple(applied)


__all__ = ["apply_declared_durable_migrations", "pre_migration_backup"]
