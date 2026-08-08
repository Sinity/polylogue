"""Product-layer adapters for durable schema-change train operations."""

from __future__ import annotations

import errno
import os
import sqlite3
import stat
import tempfile
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

    try:
        location = ArchiveLocation.resolve(path.parent)
    except Exception as exc:
        raise MigrationError(f"cannot inspect archive adoption markers: {path.parent}") from exc
    durable_siblings = tuple(
        path.parent / f"{sibling.value}.db"
        for sibling in (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT)
        if sibling is not tier
    )
    existing_siblings: list[Path] = []
    for sibling in durable_siblings:
        try:
            sibling.lstat()
        except FileNotFoundError:
            continue
        existing_siblings.append(sibling)
    adoption_markers: list[Path] = []
    active_pointer_marker = path.parent / ".index-active-pointer"
    try:
        active_pointer_marker.lstat()
    except FileNotFoundError:
        pass
    else:
        # Presence is enough. ArchiveLocation intentionally ignores a dangling
        # symlink via Path.exists(), but missing-tier initialization must treat
        # malformed or dangling adoption evidence as established and fail
        # closed rather than publishing an empty durable database.
        adoption_markers.append(active_pointer_marker)
    train_marker_root = path.parent / ".maintenance-state" / "durable-change-trains"
    try:
        train_marker_metadata = train_marker_root.lstat()
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISLNK(train_marker_metadata.st_mode) or not stat.S_ISDIR(train_marker_metadata.st_mode):
            adoption_markers.append(train_marker_root)
        else:
            try:
                if any(train_marker_root.iterdir()):
                    adoption_markers.append(train_marker_root)
            except OSError as exc:
                raise MigrationError(
                    f"cannot inspect durable change-train adoption marker: {train_marker_root}"
                ) from exc
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
    anonymous_unsupported = not anonymous_flag
    publication_descriptor: int | None = None
    named_publication_path: Path | None = None
    try:
        if not anonymous_unsupported:
            try:
                publication_descriptor = os.open(
                    path.parent,
                    os.O_RDWR | anonymous_flag | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                )
            except OSError as exc:
                if exc.errno in {errno.EISDIR, errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}:
                    anonymous_unsupported = True
                else:
                    raise MigrationError(
                        f"cannot create anonymous durable-tier publication inode: {path.parent}"
                    ) from exc
        if anonymous_unsupported:
            try:
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{path.name}.initialize-",
                    suffix=".tmp",
                    dir=path.parent,
                )
            except OSError as exc:
                raise MigrationError(f"cannot create durable-tier publication staging file: {path.parent}") from exc
            publication_descriptor = descriptor
            named_publication_path = Path(temporary_name)
            os.fchmod(descriptor, 0o600)

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
            or (not anonymous_unsupported and publication_metadata.st_nlink != 0)
            or publication_metadata.st_size != len(initialized_image)
        ):
            raise MigrationError(f"durable-tier publication image is incomplete: {path}")
        publication_identity = (publication_metadata.st_dev, publication_metadata.st_ino)
        if named_publication_path is not None:
            os.close(descriptor)
            publication_descriptor = None
        try:
            if named_publication_path is None:
                # O_TMPFILE plus link(2) publishes one descriptor-backed inode
                # without resolving a replaceable named staging path again.
                os.link(f"/proc/self/fd/{descriptor}", path, follow_symlinks=True)
            else:
                # mkstemp created this same-directory name with O_EXCL and
                # mode 0600. Closing it before link makes the fallback portable
                # to filesystems without O_TMPFILE while retaining no-replace
                # publication semantics.
                os.link(named_publication_path, path, follow_symlinks=True)
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
        if named_publication_path is not None:
            named_publication_path.unlink(missing_ok=True)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if publication_descriptor is not None:
            os.close(publication_descriptor)
        if named_publication_path is not None:
            named_publication_path.unlink(missing_ok=True)

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
