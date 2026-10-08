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

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

from polylogue.operations.durable_change_train import (
    DurableMigrationReplayProof,
    OwnedArchiveLocation,
    PendingDurableMigration,
    execute_durable_change_train,
    pending_durable_migrations,
    pre_migration_backup,
    rehearse_pending_durable_migration,
)


def apply_declared_durable_migrations(
    archive_root: Path,
    *,
    archive_owner: OwnedArchiveLocation,
    write_lease: Callable[[str], AbstractContextManager[object]],
) -> tuple[PendingDurableMigration, ...]:
    """Apply the pending numbered step of each tier behind its declared version.

    Each step is one train; a data-changing step gets its own backup of the
    bytes it is about to change. Before any backup in a pass, every pending
    tier chain is replayed on a schema-only replica through current canonical
    DDL. Each live step is checked against its captured intermediate inventory.

    The caller holds exclusive archive ownership for the whole call and keeps
    it afterwards, so the train's ownership release is a no-op here. Each
    backup and train runs inside ``write_lease`` so it is one writer with the
    daemon, not beside it.
    """

    applied: list[PendingDurableMigration] = []
    while pending := pending_durable_migrations(archive_root):
        replay_proofs: dict[str, DurableMigrationReplayProof] = {}
        for migration in pending:
            replay_proofs[migration.tier.value] = rehearse_pending_durable_migration(archive_root, migration)
        for migration in pending:
            if migration in applied:
                raise RuntimeError(
                    f"{migration.tier.value}.db did not advance past v{migration.current_version}; refusing to loop"
                )
            _apply_step(
                archive_root,
                migration,
                archive_owner=archive_owner,
                write_lease=write_lease,
                schema_replay_proof=replay_proofs[migration.tier.value],
            )
            applied.append(migration)
    return tuple(applied)


def _apply_step(
    archive_root: Path,
    migration: PendingDurableMigration,
    *,
    archive_owner: OwnedArchiveLocation,
    write_lease: Callable[[str], AbstractContextManager[object]],
    schema_replay_proof: DurableMigrationReplayProof,
) -> None:
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
            schema_replay_proof=schema_replay_proof,
            release_archive_ownership=lambda: None,
        )


__all__ = ["apply_declared_durable_migrations"]
