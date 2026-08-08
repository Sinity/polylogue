"""Product-layer adapters for durable schema-change train operations."""

from __future__ import annotations

import os
import sqlite3
import stat
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
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier

    def adoption_lstat(candidate: Path, description: str) -> os.stat_result | None:
        try:
            return candidate.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise MigrationError(f"cannot inspect {description}: {candidate}") from exc

    parent_metadata = adoption_lstat(path.parent, "durable tier parent directory")
    if parent_metadata is None:
        raise MigrationError(f"durable tier parent directory is missing: {path.parent}")
    if (
        stat.S_ISLNK(parent_metadata.st_mode)
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(parent_metadata.st_mode) & 0o022
    ):
        raise MigrationError(f"durable tier parent is not a private owned directory: {path.parent}")

    if adoption_lstat(path, "durable tier target") is not None:
        raise MigrationError(f"{tier.value} tier already exists; refusing missing-tier initialization: {path}")

    try:
        location = ArchiveLocation.resolve(path.parent)
    except Exception as exc:
        raise MigrationError(f"cannot inspect archive adoption markers: {path.parent}") from exc
    durable_siblings = tuple(path.parent / f"{sibling.value}.db" for sibling in ArchiveTier if sibling is not tier)
    existing_siblings: list[Path] = []
    for sibling in durable_siblings:
        if adoption_lstat(sibling, "archive tier") is not None:
            existing_siblings.append(sibling)
    adoption_markers: list[Path] = []
    active_pointer_marker = path.parent / ".index-active-pointer"
    if adoption_lstat(active_pointer_marker, "active index pointer") is not None:
        # Presence is enough. ArchiveLocation intentionally ignores a dangling
        # symlink via Path.exists(), but missing-tier initialization must treat
        # malformed or dangling adoption evidence as established and fail
        # closed rather than publishing an empty durable database.
        adoption_markers.append(active_pointer_marker)
    blob_path = path.parent / "blob"
    blob_metadata = adoption_lstat(blob_path, "retained blob path")
    if blob_metadata is not None:
        if stat.S_ISLNK(blob_metadata.st_mode) or not stat.S_ISDIR(blob_metadata.st_mode):
            adoption_markers.append(blob_path)
        else:
            try:
                blob_has_entries = next(blob_path.iterdir(), None) is not None
            except OSError as exc:
                raise MigrationError(f"cannot inspect retained blob path: {blob_path}") from exc
            if blob_has_entries:
                adoption_markers.append(blob_path)
    maintenance_state_root = path.parent / ".maintenance-state"
    maintenance_state_metadata = adoption_lstat(maintenance_state_root, "maintenance state parent")
    if maintenance_state_metadata is not None and (
        stat.S_ISLNK(maintenance_state_metadata.st_mode) or not stat.S_ISDIR(maintenance_state_metadata.st_mode)
    ):
        adoption_markers.append(maintenance_state_root)
    train_marker_root = path.parent / ".maintenance-state" / "durable-change-trains"
    train_marker_metadata = adoption_lstat(train_marker_root, "durable change-train adoption marker")
    if train_marker_metadata is not None:
        if stat.S_ISLNK(train_marker_metadata.st_mode) or not stat.S_ISDIR(train_marker_metadata.st_mode):
            adoption_markers.append(train_marker_root)
        else:
            try:
                marker_entries = tuple(train_marker_root.iterdir())
            except OSError as exc:
                raise MigrationError(
                    f"cannot inspect durable change-train adoption marker: {train_marker_root}"
                ) from exc
            if marker_entries:
                for marker_name in (".bootstrap", ".bootstrap.pending"):
                    marker_path = train_marker_root / marker_name
                    if adoption_lstat(marker_path, "durable bootstrap marker") is not None:
                        adoption_markers.append(marker_path)
                if train_marker_root not in adoption_markers:
                    adoption_markers.append(train_marker_root)
    if location.active_pointer is not None and active_pointer_marker not in adoption_markers:
        adoption_markers.append(active_pointer_marker)
    if existing_siblings or adoption_markers:
        details = ", ".join(str(item) for item in (*existing_siblings, *adoption_markers))
        raise MigrationError(
            f"cannot initialize missing {tier.value} tier in an established archive; adoption marker(s): {details}"
        )

    # Build the canonical database in memory before choosing the publication
    # substrate. Both publication paths link one exact serialized image and
    # never replace a target that appears concurrently.
    memory_database = sqlite3.connect(":memory:")
    try:
        initialize_archive_tier(memory_database, tier)
        initialized_image = memory_database.serialize()
    finally:
        memory_database.close()
    if not initialized_image:
        raise MigrationError(f"canonical {tier.value} tier initialization produced an empty database image")

    anonymous_flag = getattr(os, "O_TMPFILE", 0)
    if not anonymous_flag:
        raise MigrationError(
            f"cannot initialize missing {tier.value} tier: durable publication requires anonymous O_TMPFILE support"
        )
    publication_descriptor: int | None = None
    try:
        try:
            publication_descriptor = os.open(
                path.parent,
                os.O_RDWR | anonymous_flag | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
        except OSError as exc:
            raise MigrationError(
                f"cannot initialize missing {tier.value} tier: filesystem does not support anonymous durable publication"
            ) from exc

        assert publication_descriptor is not None
        descriptor = publication_descriptor
        source_offset = 0
        while source_offset < len(initialized_image):
            written_offset = 0
            chunk = initialized_image[source_offset : source_offset + 1024 * 1024]
            while written_offset < len(chunk):
                written = os.write(descriptor, chunk[written_offset:])
                if written <= 0:
                    raise MigrationError("durable-tier publication copy made no progress")
                written_offset += written
            source_offset += len(chunk)
        os.fsync(descriptor)
        publication_metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(publication_metadata.st_mode)
            or publication_metadata.st_nlink != 0
            or publication_metadata.st_size != len(initialized_image)
        ):
            raise MigrationError(f"durable-tier publication image is incomplete: {path}")
        publication_identity = (publication_metadata.st_dev, publication_metadata.st_ino)
        try:
            # Link the still-open anonymous inode directly. A named fallback
            # would let another same-UID process replace or modify the staged
            # bytes before publication, so unsupported filesystems fail closed.
            os.link(f"/proc/self/fd/{descriptor}", path, follow_symlinks=True)
        except FileExistsError as exc:
            raise MigrationError(
                f"{tier.value} tier appeared during initialization; refusing to replace it: {path}"
            ) from exc
        published_metadata = path.lstat()
        if (
            not stat.S_ISREG(published_metadata.st_mode)
            or (published_metadata.st_dev, published_metadata.st_ino) != publication_identity
        ):
            if (published_metadata.st_dev, published_metadata.st_ino) == publication_identity:
                path.unlink(missing_ok=True)
            raise MigrationError(f"published durable tier identity does not match the staged database: {path}")
        directory_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if publication_descriptor is not None:
            os.close(publication_descriptor)

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
