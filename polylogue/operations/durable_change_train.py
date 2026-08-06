"""Product-layer adapters for durable schema-change train operations."""

from __future__ import annotations

import os
import stat
import uuid
from collections.abc import Callable
from pathlib import Path

from polylogue.storage.archive_identity import ArchiveLocation, ArchiveOwnershipError, OwnedArchiveLocation
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.durable_change_train import (
    DurableChangeTrainExecution,
)
from polylogue.storage.sqlite.durable_change_train import (
    execute_durable_change_train as _execute_durable_change_train,
)
from polylogue.storage.sqlite.durable_change_train import (
    reconcile_durable_change_train_startup as _reconcile_durable_change_train_startup,
)
from polylogue.storage.sqlite.migration_runner import DurableRuntimeConsumerResult, MigrationError


def acquire_durable_archive_ownership(root: Path, *, owner_id: str) -> OwnedArchiveLocation:
    """Acquire the stable archive lease shared by daemon and maintenance."""
    location = ArchiveLocation.resolve(root)
    return OwnedArchiveLocation.acquire(location, owner_id=owner_id)


def initialize_missing_durable_tier(path: Path, tier: ArchiveTier) -> int:
    """Initialize one absent durable tier while the caller owns the archive.

    This is deliberately separate from migration. A missing tier has no
    historical schema version to advance, while an existing path must never be
    replaced or interpreted as empty by this recovery route.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database

    try:
        parent_metadata = path.parent.lstat()
    except FileNotFoundError as exc:
        raise MigrationError(f"durable tier parent directory is missing: {path.parent}") from exc
    if (
        stat.S_ISLNK(parent_metadata.st_mode)
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(parent_metadata.st_mode) & 0o022
    ):
        raise MigrationError(f"durable tier parent is not a private owned directory: {path.parent}")

    try:
        path.lstat()
    except FileNotFoundError:
        pass
    else:
        raise MigrationError(f"{tier.value} tier already exists; refusing missing-tier initialization: {path}")

    # Build the database under an unguessable private sibling name, then
    # publish it with link(2). Unlike a check followed by SQLite's ordinary
    # create-open, link cannot replace a file or symlink that appears at the
    # target between the absence probe above and publication.
    staged = path.with_name(f".{path.name}.initialize-{uuid.uuid4().hex}.tmp")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    publication_descriptor: int | None = None
    staged_identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(staged, flags, 0o600)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise MigrationError(f"staged durable tier is not a real single-linked file: {staged}")
        staged_identity = (metadata.st_dev, metadata.st_ino)
        initialize_archive_database(staged, tier, allow_create=False)
        os.fsync(descriptor)
        initialized_metadata = staged.lstat()
        if (
            not stat.S_ISREG(initialized_metadata.st_mode)
            or initialized_metadata.st_nlink != 1
            or (initialized_metadata.st_dev, initialized_metadata.st_ino) != staged_identity
        ):
            raise MigrationError(f"staged durable tier identity changed during initialization: {staged}")
        anonymous_flag = getattr(os, "O_TMPFILE", 0)
        if not anonymous_flag:
            raise MigrationError("missing-tier initialization requires anonymous-file publication support")
        try:
            publication_descriptor = os.open(
                path.parent,
                os.O_RDWR | anonymous_flag | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
        except OSError as exc:
            raise MigrationError(f"cannot create anonymous durable-tier publication inode: {path.parent}") from exc
        offset = 0
        while chunk := os.pread(descriptor, 1024 * 1024, offset):
            written_offset = 0
            while written_offset < len(chunk):
                written = os.write(publication_descriptor, chunk[written_offset:])
                if written <= 0:
                    raise MigrationError("durable-tier publication copy made no progress")
                written_offset += written
            offset += len(chunk)
        os.fsync(publication_descriptor)
        publication_metadata = os.fstat(publication_descriptor)
        if (
            not stat.S_ISREG(publication_metadata.st_mode)
            or publication_metadata.st_size != initialized_metadata.st_size
        ):
            raise MigrationError(f"anonymous durable-tier publication copy is incomplete: {path}")
        publication_identity = (publication_metadata.st_dev, publication_metadata.st_ino)
        try:
            # O_TMPFILE plus link(2) publishes one descriptor-backed inode
            # without resolving the replaceable named staging path again.
            os.link(f"/proc/self/fd/{publication_descriptor}", path, follow_symlinks=True)
        except FileExistsError as exc:
            raise MigrationError(
                f"{tier.value} tier appeared during initialization; refusing to replace it: {path}"
            ) from exc
        published_metadata = path.lstat()
        if (
            not stat.S_ISREG(published_metadata.st_mode)
            or (published_metadata.st_dev, published_metadata.st_ino) != publication_identity
        ):
            raise MigrationError(f"published durable tier identity does not match the staged database: {path}")
        directory_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if publication_descriptor is not None:
            os.close(publication_descriptor)
        if descriptor is not None:
            os.close(descriptor)
        if staged_identity is not None:
            try:
                current = staged.lstat()
            except FileNotFoundError:
                pass
            else:
                if (current.st_dev, current.st_ino) == staged_identity:
                    staged.unlink()

    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER

    return ARCHIVE_VERSION_BY_TIER[tier]


def execute_durable_change_train(
    archive_root: Path,
    tier: ArchiveTier,
    *,
    backup_manifest: Path | None,
    daemon_stopped_evidence_ref: str,
    single_writer_evidence_ref: str,
    runtime_consumer_results: tuple[DurableRuntimeConsumerResult, ...] | None = None,
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
        release_archive_ownership=release_archive_ownership,
    )


def reconcile_durable_change_trains_on_startup(root: Path) -> tuple[Path, ...]:
    """Run bootstrap train recovery before daemon surfaces open the archive."""
    return _reconcile_durable_change_train_startup(root)


__all__ = [
    "acquire_durable_archive_ownership",
    "ArchiveOwnershipError",
    "execute_durable_change_train",
    "initialize_missing_durable_tier",
    "reconcile_durable_change_trains_on_startup",
]
