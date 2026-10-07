"""Product-layer adapters for durable schema-change train operations."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, closing
from dataclasses import dataclass
from pathlib import Path

from polylogue.storage.archive_identity import ArchiveLocation, ArchiveOwnershipError, OwnedArchiveLocation
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from polylogue.storage.sqlite.durable_change_train import (
    DurableChangeTrainExecution,
)
from polylogue.storage.sqlite.durable_change_train import (
    execute_durable_change_train as _execute_durable_change_train,
)
from polylogue.storage.sqlite.durable_change_train import (
    reconcile_durable_change_train_startup as _reconcile_durable_change_train_startup,
)
from polylogue.storage.sqlite.migration_runner import (
    DurableChangeTrainError,
    DurableMigrationReplayProof,
    DurableRuntimeConsumerResult,
)


def acquire_durable_archive_ownership(root: Path, *, owner_id: str) -> OwnedArchiveLocation:
    """Acquire the stable archive lease shared by daemon and maintenance."""
    location = ArchiveLocation.resolve(root)
    return OwnedArchiveLocation.acquire(location, owner_id=owner_id)


@dataclass(frozen=True, slots=True)
class PendingDurableMigration:
    """One durable tier this runtime declares a newer schema version for."""

    tier: ArchiveTier
    current_version: int
    target_version: int
    requires_backup: bool


def assert_holds_archive_ownership(owner: OwnedArchiveLocation, archive_root: Path) -> None:
    """Refuse unless ``owner`` is the live exclusive ownership of ``archive_root``."""

    from polylogue.storage.archive_identity import assert_owns_archive_location

    assert_owns_archive_location(owner, ArchiveLocation.resolve(archive_root))


def initialize_fresh_archive_on_startup(
    archive_root: Path,
    *,
    archive_owner: OwnedArchiveLocation,
    write_lease: Callable[[str], AbstractContextManager[object]],
) -> None:
    """Resume declared fresh construction before ordinary migration admission."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import (
        DURABLE_MIGRATION_TIERS,
        archive_tier_spec,
        initialize_active_archive_root,
    )

    assert_holds_archive_ownership(archive_owner, archive_root)
    pending = (archive_root / ".maintenance-state/durable-change-trains/.bootstrap.pending").is_file()
    absent = all(not (archive_root / archive_tier_spec(tier).filename).exists() for tier in DURABLE_MIGRATION_TIERS)
    if pending or absent:
        # Existing bootstrap owns all-six baseline intent and numbered trains.
        # Established roots retain their separate durable migration admission.
        with write_lease("daemon.archive_bootstrap.startup"):
            initialize_active_archive_root(archive_root)


def pending_durable_migrations(archive_root: Path) -> tuple[PendingDurableMigration, ...]:
    """Name the next numbered step for every durable tier below this runtime's version."""

    from polylogue.storage.sqlite import migration_runner

    pending: list[PendingDurableMigration] = []
    for tier in sorted(migration_runner.DURABLE_MIGRATION_TIERS, key=lambda item: item.value):
        path = archive_root / f"{tier.value}.db"
        if not path.is_file():
            continue
        with closing(open_readonly_connection(path, validate_schema=False)) as conn:
            current = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
        target = ARCHIVE_VERSION_BY_TIER[tier]
        if current >= target:
            continue
        steps = tuple(
            claim
            for claim in migration_runner.durable_migration_claims(tier)
            if current < claim.target_version <= target
        )
        if {step.target_version for step in steps} != set(range(current + 1, target + 1)):
            # No declared route from this version: schema preflight reports
            # the skew; there is nothing to apply.
            continue
        # One numbered step at a time: each train advances exactly one slot,
        # and each data-changing step needs a backup of the bytes it changes.
        step = next(step for step in steps if step.target_version == current + 1)
        pending.append(
            PendingDurableMigration(
                tier=tier,
                current_version=current,
                target_version=current + 1,
                requires_backup=step.requires_backup,
            )
        )
    return tuple(pending)


def rehearse_pending_durable_migration(
    archive_root: Path,
    migration: PendingDurableMigration,
) -> DurableMigrationReplayProof:
    """Rehearse this pending tier's complete route to the shipped schema."""
    from polylogue.storage.sqlite import migration_runner

    tier_path = archive_root / f"{migration.tier.value}.db"
    with closing(open_readonly_connection(tier_path, validate_schema=False)) as source:
        return migration_runner.rehearse_durable_migration_chain(
            source,
            migration.tier,
            target_version=ARCHIVE_VERSION_BY_TIER[migration.tier],
            evidence_ref=(
                f"proof:daemon-chain-rehearsal:{migration.tier.value}:v{migration.current_version}-to-current"
            ),
        )


def execute_durable_change_train(
    archive_root: Path,
    tier: ArchiveTier,
    *,
    backup_manifest: Path | None,
    daemon_stopped_evidence_ref: str,
    single_writer_evidence_ref: str,
    runtime_consumer_results: tuple[DurableRuntimeConsumerResult, ...] | None = None,
    schema_replay_proof: DurableMigrationReplayProof | None = None,
    release_archive_ownership: Callable[[], None],
) -> DurableChangeTrainExecution:
    """Run one durable migration through the storage authority contract."""
    return _execute_durable_change_train(
        archive_root,
        tier,
        backup_manifest=backup_manifest,
        daemon_stopped_evidence_ref=daemon_stopped_evidence_ref,
        single_writer_evidence_ref=single_writer_evidence_ref,
        runtime_consumer_results=runtime_consumer_results,
        schema_replay_proof=schema_replay_proof,
        release_archive_ownership=release_archive_ownership,
    )


def reconcile_durable_change_trains_on_startup(root: Path) -> tuple[Path, ...]:
    """Run bootstrap train recovery before daemon surfaces open the archive."""
    return _reconcile_durable_change_train_startup(root)


def pre_migration_backup(
    archive_root: Path, migration: PendingDurableMigration, *, archive_owner: OwnedArchiveLocation
) -> Path:
    """Take and scratch-verify the backup a data-changing migration requires."""

    from polylogue.storage.backup_package import create_pre_migration_backup

    return create_pre_migration_backup(
        archive_root,
        tier=migration.tier.value,
        current_version=migration.current_version,
        target_version=migration.target_version,
        archive_owner=archive_owner,
    )


__all__ = [
    "pre_migration_backup",
    "acquire_durable_archive_ownership",
    "ArchiveOwnershipError",
    "DurableChangeTrainError",
    "DurableMigrationReplayProof",
    "OwnedArchiveLocation",
    "PendingDurableMigration",
    "assert_holds_archive_ownership",
    "execute_durable_change_train",
    "initialize_fresh_archive_on_startup",
    "pending_durable_migrations",
    "rehearse_pending_durable_migration",
    "reconcile_durable_change_trains_on_startup",
]
