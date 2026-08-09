"""Product-layer adapters for durable schema-change train operations."""

from __future__ import annotations

import os
import sqlite3
import stat
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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


@dataclass(frozen=True, slots=True)
class DurableCleanupOutcome:
    """Operator-facing result of recovering a partially visible publication."""

    state: Literal["not_attempted", "target_absent", "cleaned", "uncertain"]
    code: str | None = None
    target: str | None = None
    detail: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "state": self.state,
            "code": self.code,
            "target": self.target,
            "detail": self.detail,
        }


class DurablePublicationError(MigrationError):
    """Publication failure carrying durable cleanup uncertainty for operators."""

    def __init__(self, message: str, *, cleanup: DurableCleanupOutcome | None = None) -> None:
        super().__init__(message)
        self.cleanup = cleanup


def acquire_durable_archive_ownership(root: Path, *, owner_id: str) -> OwnedArchiveLocation:
    """Acquire the stable archive lease shared by daemon and maintenance."""
    location = ArchiveLocation.resolve(root)
    return OwnedArchiveLocation.acquire(location, owner_id=owner_id)


def initialize_missing_durable_tier(path: Path, tier: ArchiveTier, *, directory_fd: int | None = None) -> int:
    """Initialize one absent durable tier while the caller owns the archive.

    This is deliberately separate from migration. A missing tier has no
    historical schema version to advance, while an existing path must never be
    replaced or interpreted as empty by this recovery route.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier

    archive_root = path.parent

    try:
        parent_metadata = archive_root.lstat()
    except FileNotFoundError as exc:
        raise MigrationError(f"durable tier parent directory is missing: {archive_root}") from exc
    except OSError as exc:
        raise MigrationError(f"cannot inspect durable tier parent directory: {archive_root}") from exc
    if parent_metadata is None:
        raise MigrationError(f"durable tier parent directory is missing: {archive_root}")
    if (
        stat.S_ISLNK(parent_metadata.st_mode)
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(parent_metadata.st_mode) & 0o022
    ):
        raise MigrationError(f"durable tier parent is not a private owned directory: {archive_root}")

    directory_descriptor: int | None = None
    try:
        if directory_fd is not None:
            directory_descriptor = os.dup(directory_fd)
        else:
            directory_descriptor = os.open(
                archive_root,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
    except OSError as exc:
        raise MigrationError(
            f"cannot publish {tier.value} tier at {path}: cannot anchor durable tier parent directory"
        ) from exc

    try:
        assert directory_descriptor is not None
        anchored_metadata = os.fstat(directory_descriptor)
        if (anchored_metadata.st_dev, anchored_metadata.st_ino) != (parent_metadata.st_dev, parent_metadata.st_ino):
            raise MigrationError(f"durable tier parent directory changed during validation: {archive_root}")

        def adoption_lstat(relative: str, description: str) -> os.stat_result | None:
            try:
                return os.stat(relative, dir_fd=directory_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                return None
            except OSError as exc:
                raise MigrationError(f"cannot inspect {description}: {archive_root / relative}") from exc

        def directory_entries(relative: str, metadata: os.stat_result, description: str) -> list[str]:
            try:
                child_descriptor = os.open(
                    relative,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_descriptor,
                )
            except OSError as exc:
                raise MigrationError(f"cannot inspect {description}: {archive_root / relative}") from exc
            try:
                child_metadata = os.fstat(child_descriptor)
                if (child_metadata.st_dev, child_metadata.st_ino) != (metadata.st_dev, metadata.st_ino):
                    raise MigrationError(
                        f"archive directory entry changed during validation: {archive_root / relative}"
                    )
                return os.listdir(child_descriptor)
            except OSError as exc:
                raise MigrationError(f"cannot inspect {description}: {archive_root / relative}") from exc
            finally:
                os.close(child_descriptor)

        target_name = path.name
        if adoption_lstat(target_name, "durable tier target") is not None:
            raise MigrationError(f"{tier.value} tier already exists; refusing missing-tier initialization: {path}")

        durable_siblings = tuple(f"{sibling.value}.db" for sibling in ArchiveTier if sibling is not tier)
        existing_siblings = [
            archive_root / sibling
            for sibling in durable_siblings
            if adoption_lstat(sibling, "archive tier") is not None
        ]
        adoption_markers: list[Path] = []
        active_pointer_marker = ".index-active-pointer"
        if adoption_lstat(active_pointer_marker, "active index pointer") is not None:
            # Presence is enough. ArchiveLocation intentionally ignores a dangling
            # symlink via Path.exists(), but missing-tier initialization must treat
            # malformed or dangling adoption evidence as established and fail
            # closed rather than publishing an empty durable database.
            adoption_markers.append(archive_root / active_pointer_marker)
        blob_relative = "blob"
        blob_metadata = adoption_lstat(blob_relative, "retained blob path")
        if blob_metadata is not None:
            if stat.S_ISLNK(blob_metadata.st_mode) or not stat.S_ISDIR(blob_metadata.st_mode):
                adoption_markers.append(archive_root / blob_relative)
            else:
                blob_has_entries = bool(directory_entries(blob_relative, blob_metadata, "retained blob path"))
                if blob_has_entries:
                    adoption_markers.append(archive_root / blob_relative)
        maintenance_state_relative = ".maintenance-state"
        maintenance_state_metadata = adoption_lstat(maintenance_state_relative, "maintenance state parent")
        if maintenance_state_metadata is not None and (
            stat.S_ISLNK(maintenance_state_metadata.st_mode) or not stat.S_ISDIR(maintenance_state_metadata.st_mode)
        ):
            adoption_markers.append(archive_root / maintenance_state_relative)
        train_marker_relative = ".maintenance-state/durable-change-trains"
        train_marker_metadata = adoption_lstat(train_marker_relative, "durable change-train adoption marker")
        if train_marker_metadata is not None:
            if stat.S_ISLNK(train_marker_metadata.st_mode) or not stat.S_ISDIR(train_marker_metadata.st_mode):
                adoption_markers.append(archive_root / train_marker_relative)
            else:
                marker_entries = tuple(
                    directory_entries(
                        train_marker_relative, train_marker_metadata, "durable change-train adoption marker"
                    )
                )
                if marker_entries:
                    for marker_name in (".bootstrap", ".bootstrap.pending"):
                        marker_relative = f"{train_marker_relative}/{marker_name}"
                        if adoption_lstat(marker_relative, "durable bootstrap marker") is not None:
                            adoption_markers.append(archive_root / marker_relative)
                    train_marker_path = archive_root / train_marker_relative
                    if train_marker_path not in adoption_markers:
                        adoption_markers.append(train_marker_path)
        retained_evidence_roots = (
            (".index-generations", "retained index-generation evidence"),
            (".index-rebuild-transactions", "retained index-rebuild transaction evidence"),
            (".maintenance-state/source-continuity-pending", "source-continuity recovery evidence"),
            (".maintenance-state/source-continuity-refreshes", "source-continuity refresh evidence"),
        )
        for evidence_relative, description in retained_evidence_roots:
            evidence_metadata = adoption_lstat(evidence_relative, description)
            if evidence_metadata is None:
                continue
            if stat.S_ISLNK(evidence_metadata.st_mode) or not stat.S_ISDIR(evidence_metadata.st_mode):
                adoption_markers.append(archive_root / evidence_relative)
                continue
            has_retained_evidence = bool(directory_entries(evidence_relative, evidence_metadata, description))
            if has_retained_evidence:
                adoption_markers.append(archive_root / evidence_relative)
        if existing_siblings or adoption_markers:
            details = ", ".join(str(item) for item in (*existing_siblings, *adoption_markers))
            raise MigrationError(
                f"cannot initialize missing {tier.value} tier in an established archive; adoption marker(s): {details}"
            )
    except BaseException:
        os.close(directory_descriptor)
        raise

    # Build the canonical database in memory before choosing the publication
    # substrate. Both publication paths link one exact serialized image and
    # never replace a target that appears concurrently.
    try:
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
                f"cannot initialize missing {tier.value} tier: filesystem does not support O_TMPFILE: {path}"
            )
    except BaseException:
        os.close(directory_descriptor)
        raise
    publication_descriptor: int | None = None
    publication_identity: tuple[int, int] | None = None
    published_target = False

    def cleanup_published_target(primary: BaseException) -> DurableCleanupOutcome:
        """Remove only our inode after a post-link publication failure."""
        if not published_target or publication_identity is None:
            return DurableCleanupOutcome("not_attempted")
        try:
            published_metadata = adoption_lstat(target_name, "published durable tier")
            if published_metadata is None:
                return DurableCleanupOutcome("target_absent", target=str(path))
        except FileNotFoundError:
            return DurableCleanupOutcome("target_absent", target=str(path))
        except MigrationError as exc:
            detail = f"could not inspect published durable tier during recovery: {path}: {exc}"
            primary.add_note(detail)
            return DurableCleanupOutcome("uncertain", "leaf_inspection_failed", str(path), detail)
        except OSError as exc:
            detail = f"could not inspect published durable tier during recovery: {path}: {exc}"
            primary.add_note(detail)
            return DurableCleanupOutcome("uncertain", "leaf_inspection_failed", str(path), detail)
        if (published_metadata.st_dev, published_metadata.st_ino) != publication_identity:
            detail = f"published durable tier changed before recovery; preserving foreign target: {path}"
            primary.add_note(detail)
            return DurableCleanupOutcome("uncertain", "leaf_replaced", str(path), detail)
        cleanup_name = f".{target_name}.cleanup-{uuid.uuid4().hex}.tmp"
        try:
            leaf_descriptor = os.open(
                target_name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory_descriptor,
            )
            try:
                leaf_metadata = os.fstat(leaf_descriptor)
                if (leaf_metadata.st_dev, leaf_metadata.st_ino) != publication_identity:
                    detail = f"published durable tier changed before recovery; preserving foreign target: {path}"
                    primary.add_note(detail)
                    return DurableCleanupOutcome("uncertain", "leaf_replaced", str(path), detail)
                current_metadata = adoption_lstat(target_name, "published durable tier")
                if current_metadata is None:
                    return DurableCleanupOutcome("target_absent", target=str(path))
                if (current_metadata.st_dev, current_metadata.st_ino) != publication_identity:
                    detail = f"published durable tier changed before recovery; preserving foreign target: {path}"
                    primary.add_note(detail)
                    return DurableCleanupOutcome("uncertain", "leaf_replaced", str(path), detail)
                os.rename(
                    target_name,
                    cleanup_name,
                    src_dir_fd=directory_descriptor,
                    dst_dir_fd=directory_descriptor,
                )
                moved_metadata = os.stat(cleanup_name, dir_fd=directory_descriptor, follow_symlinks=False)
                if (moved_metadata.st_dev, moved_metadata.st_ino) != publication_identity:
                    detail = f"published durable tier changed before recovery; preserving foreign target: {path}"
                    try:
                        os.rename(
                            cleanup_name,
                            target_name,
                            src_dir_fd=directory_descriptor,
                            dst_dir_fd=directory_descriptor,
                        )
                    except OSError as restore_exc:
                        primary.add_note(
                            f"could not restore foreign durable tier after checked cleanup: {path}: {restore_exc}"
                        )
                    return DurableCleanupOutcome("uncertain", "leaf_replaced", str(path), detail)
                os.unlink(cleanup_name, dir_fd=directory_descriptor)
            finally:
                os.close(leaf_descriptor)
            try:
                os.fsync(directory_descriptor)
            except OSError as exc:
                detail = f"could not fsync durable tier cleanup directory: {archive_root}: {exc}"
                primary.add_note(detail)
                return DurableCleanupOutcome("uncertain", "cleanup_directory_fsync_failed", str(path), detail)
            return DurableCleanupOutcome("cleaned", target=str(path))
        except FileNotFoundError:
            return DurableCleanupOutcome("target_absent", target=str(path))
        except MigrationError as exc:
            detail = f"could not inspect published durable tier during recovery: {path}: {exc}"
            primary.add_note(detail)
            return DurableCleanupOutcome("uncertain", "leaf_inspection_failed", str(path), detail)
        except OSError as exc:
            detail = f"could not remove partially published durable tier during recovery: {path}: {exc}"
            primary.add_note(detail)
            return DurableCleanupOutcome("uncertain", "leaf_unlink_failed", str(path), detail)

    try:
        try:
            publication_descriptor = os.open(
                ".",
                os.O_RDWR | anonymous_flag | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=directory_descriptor,
            )
        except OSError as exc:
            raise MigrationError(
                f"cannot initialize missing {tier.value} tier: anonymous durable publication failed: {path}"
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
            os.link(
                f"/proc/self/fd/{descriptor}",
                target_name,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=True,
            )
        except FileExistsError as exc:
            raise MigrationError(
                f"{tier.value} tier appeared during initialization; refusing to replace it: {path}"
            ) from exc
        except OSError as exc:
            raise MigrationError(
                f"cannot initialize missing {tier.value} tier: anonymous durable publication failed: {path}"
            ) from exc
        published_target = True
        published_metadata = adoption_lstat(target_name, "published durable tier")
        if published_metadata is None:
            raise MigrationError(f"published durable tier disappeared before identity validation: {path}")
        if (
            not stat.S_ISREG(published_metadata.st_mode)
            or (published_metadata.st_dev, published_metadata.st_ino) != publication_identity
        ):
            raise MigrationError(f"published durable tier identity does not match the staged database: {path}")
        os.fsync(directory_descriptor)
    except MigrationError as exc:
        cleanup = cleanup_published_target(exc)
        if published_target:
            raise DurablePublicationError(
                f"cannot publish {tier.value} tier at {path} via durable publication", cleanup=cleanup
            ) from exc
        raise
    except OSError as exc:
        cleanup = cleanup_published_target(exc)
        raise DurablePublicationError(
            f"cannot publish {tier.value} tier at {path} via durable publication", cleanup=cleanup
        ) from exc
    finally:
        if publication_descriptor is not None:
            try:
                os.close(publication_descriptor)
            except OSError as exc:
                cleanup = cleanup_published_target(exc)
                if cleanup.state == "uncertain":
                    exc.add_note(cleanup.detail or cleanup.code or "durable cleanup is uncertain")
                raise MigrationError(
                    f"cannot close {tier.value} tier publication at {path} after durable publication"
                ) from exc
        try:
            os.close(directory_descriptor)
        except OSError as exc:
            raise MigrationError(f"cannot close durable tier parent directory: {archive_root}") from exc

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
