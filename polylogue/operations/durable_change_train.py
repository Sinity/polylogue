"""Product-layer adapters for durable schema-change train operations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import stat
import sys
import time
from collections.abc import Callable
from contextlib import closing, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from polylogue.storage.archive_identity import ArchiveLocation, ArchiveOwnershipError, OwnedArchiveLocation
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.audit_continuity import AuditContinuityCoordinator
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
    DurableRuntimeConsumerResult,
    MigrationError,
    _canonical_json_sha256,
    capture_durable_schema_inventory,
    validate_full_evidence_backup_for_adopted_audit_restore,
    validate_full_evidence_backup_for_audit_adoption,
)

_AUDIT_ADOPTION_RECEIPT_FORMAT = "polylogue.audit-tier-adoption.v1"
_AUDIT_ADOPTION_RECEIPT_NAME = "audit-adoption.json"
_AUDIT_ADOPTION_CONTINUITY_FORMAT = "polylogue.audit-tier-continuity.v1"
_AUDIT_ADOPTION_CONTINUITY_NAME = "audit-continuity.json"
_AUDIT_ADOPTION_RESTORE_FORMAT = "polylogue.audit-tier-restore.v1"
_AUDIT_ADOPTION_RESTORE_NAME = re.compile(
    r"^audit-restore\.(?P<generation>[1-9][0-9]*)\.(?P<operation>[0-9a-f]{32})\.(?P<state>prepared|committed)\.json$"
)


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


def _close_publication_descriptor(descriptor: int) -> None:
    """Close a publication descriptor through one fault-injectable boundary."""
    os.close(descriptor)


def acquire_durable_archive_ownership(root: Path, *, owner_id: str) -> OwnedArchiveLocation:
    """Acquire the stable archive lease shared by daemon and maintenance."""
    location = ArchiveLocation.resolve(root)
    return OwnedArchiveLocation.acquire(location, owner_id=owner_id)


def initialize_missing_durable_tier(
    path: Path,
    tier: ArchiveTier,
    *,
    directory_fd: int | None = None,
    permit_established_archive: bool = False,
    prepare_initialized_image: Callable[[sqlite3.Connection], None] | None = None,
    pre_publish_check: Callable[[bytes], None] | None = None,
) -> int:
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

        def assert_no_adoption_evidence(*, check_target: bool = True) -> None:
            if check_target and adoption_lstat(target_name, "durable tier target") is not None:
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
            if maintenance_state_metadata is not None:
                if stat.S_ISLNK(maintenance_state_metadata.st_mode) or not stat.S_ISDIR(
                    maintenance_state_metadata.st_mode
                ):
                    adoption_markers.append(archive_root / maintenance_state_relative)
                else:
                    known_maintenance_children = {
                        "durable-change-trains",
                        "source-continuity-pending",
                        "source-continuity-refreshes",
                    }
                    for name in directory_entries(
                        maintenance_state_relative, maintenance_state_metadata, "maintenance state parent"
                    ):
                        if name not in known_maintenance_children:
                            adoption_markers.append(archive_root / maintenance_state_relative / name)
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
            if not permit_established_archive and (existing_siblings or adoption_markers):
                details = ", ".join(str(item) for item in (*existing_siblings, *adoption_markers))
                raise MigrationError(
                    f"cannot initialize missing {tier.value} tier in an established archive; "
                    f"adoption marker(s): {details}"
                )

        assert_no_adoption_evidence()
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
            if prepare_initialized_image is not None:
                prepare_initialized_image(memory_database)
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
        """Assess a published target without mutating an uncertain pathname.

        POSIX pathname operations do not provide a portable conditional unlink
        or rename keyed by ``(st_dev, st_ino)``. A checked ``rename`` can still
        move a foreign inode after the final identity check, and restoring it
        can overwrite a newer target. Preserve the target and surface the
        uncertainty unless it is already absent or has visibly changed.
        """
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
        detail = (
            f"published durable tier remains after publication failure; cleanup deferred because no conditional "
            f"inode removal is available: {path}"
        )
        primary.add_note(detail)
        return DurableCleanupOutcome("uncertain", "cleanup_not_atomic", str(path), detail)

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
        # Image construction can take long enough for retained archive evidence
        # to appear. Re-census immediately before the first visible link so an
        # empty durable tier is never adopted over a newly established archive.
        # ``link`` is the atomic no-replacement check for the target itself;
        # re-census only evidence whose appearance would otherwise make this
        # empty tier an unsafe adoption.
        if pre_publish_check is not None:
            pre_publish_check(initialized_image)
        else:
            assert_no_adoption_evidence(check_target=False)
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
        primary_exception = sys.exception()
        close_failures: list[tuple[BaseException, OSError]] = []
        if publication_descriptor is not None:
            try:
                _close_publication_descriptor(publication_descriptor)
            except OSError as exc:
                cleanup = cleanup_published_target(exc)
                if cleanup.state == "uncertain":
                    exc.add_note(cleanup.detail or cleanup.code or "durable cleanup is uncertain")
                failure: BaseException
                if published_target:
                    failure = DurablePublicationError(
                        f"cannot close {tier.value} tier publication at {path} after durable publication",
                        cleanup=cleanup,
                    )
                else:
                    failure = MigrationError(
                        f"cannot close {tier.value} tier publication at {path} after durable publication"
                    )
                close_failures.append((failure, exc))
        try:
            _close_publication_descriptor(directory_descriptor)
        except OSError as exc:
            failure = MigrationError(f"cannot close durable tier parent directory: {archive_root}")
            close_failures.append((failure, exc))
        if primary_exception is not None:
            for failure, _cause in close_failures:
                primary_exception.add_note(str(failure))
        elif close_failures:
            failure, cause = close_failures[0]
            raise failure from cause

    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER

    return ARCHIVE_VERSION_BY_TIER[tier]


def audit_adoption_receipt_path(archive_root: Path) -> Path:
    """Return the durable-change-train ledger location for audit adoption."""
    return archive_root / ".maintenance-state" / "durable-change-trains" / _AUDIT_ADOPTION_RECEIPT_NAME


def _audit_adoption_continuity_path(archive_root: Path) -> Path:
    """Return the immutable identity binding for the published audit file."""
    return archive_root / ".maintenance-state" / "durable-change-trains" / _AUDIT_ADOPTION_CONTINUITY_NAME


def _audit_adoption_authority_digest(archive_root: Path) -> str:
    """Bind adoption to the two irreplaceable archive authority tiers only."""
    from polylogue.storage.archive_identity import ArchiveIdentity

    durable_id = ArchiveIdentity.resolve(archive_root).durable_id
    return hashlib.sha256(f"source-user-authority:{durable_id}".encode()).hexdigest()


def _audit_schema_inventory_sha256() -> str:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier

    with closing(sqlite3.connect(":memory:")) as connection:
        initialize_archive_tier(connection, ArchiveTier.AUDIT)
        return capture_durable_schema_inventory(connection).sha256


def _open_audit_adoption_receipt_directory(
    path: Path,
    *,
    archive_root: Path,
    create: bool,
    archive_directory_fd: int | None = None,
) -> int:
    """Open the receipt parent without following any archive path component."""
    archive_root = archive_root.resolve()
    expected_paths = {audit_adoption_receipt_path(archive_root), _audit_adoption_continuity_path(archive_root)}
    try:
        relative = path.relative_to(archive_root)
    except ValueError:
        relative = Path()
    is_restore_record = (
        len(relative.parts) == 3
        and relative.parts[:2] == (".maintenance-state", "durable-change-trains")
        and _AUDIT_ADOPTION_RESTORE_NAME.fullmatch(relative.name) is not None
    )
    if path not in expected_paths and not is_restore_record:
        raise MigrationError(f"audit adoption receipt path is outside its fixed archive location: {path}")
    try:
        current_fd = (
            os.dup(archive_directory_fd)
            if archive_directory_fd is not None
            else os.open(
                archive_root,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        )
    except OSError as exc:
        raise MigrationError(f"cannot anchor audit adoption receipt to archive root: {archive_root}") from exc
    try:
        for component in path.parent.relative_to(archive_root).parts:
            try:
                next_fd = os.open(
                    component,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=current_fd,
                )
            except FileNotFoundError:
                if not create:
                    raise
                with suppress(FileExistsError):
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                next_fd = os.open(
                    component,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=current_fd,
                )
            metadata = os.fstat(next_fd)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) & 0o022
            ):
                os.close(next_fd)
                raise MigrationError(
                    f"audit adoption receipt parent is not a private owned directory: {archive_root / component}"
                )
            os.fsync(current_fd)
            os.fsync(next_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException as exc:
        os.close(current_fd)
        if isinstance(exc, (FileNotFoundError, MigrationError)):
            raise
        if isinstance(exc, OSError):
            raise MigrationError(
                f"audit adoption receipt path must not traverse outside archive-owned directories: {path}"
            ) from exc
        raise


def _write_immutable_audit_adoption_receipt(
    path: Path,
    payload: dict[str, object],
    *,
    archive_root: Path,
    archive_directory_fd: int | None = None,
    checksum_key: str = "receipt_sha256",
) -> None:
    """Publish one pre-publication receipt without replacement and fsync it."""
    unsigned = dict(payload)
    unsigned.pop(checksum_key, None)
    payload = {**unsigned, checksum_key: _canonical_json_sha256(unsigned)}
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor: int | None = None
    receipt_directory_fd: int | None = None
    temporary_name = f".{path.name}.{secrets.token_hex(16)}.tmp"
    published = False
    try:
        receipt_directory_fd = _open_audit_adoption_receipt_directory(
            path,
            archive_root=archive_root,
            create=True,
            archive_directory_fd=archive_directory_fd,
        )
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=receipt_directory_fd,
        )
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise MigrationError("immutable audit adoption receipt write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=receipt_directory_fd,
            dst_dir_fd=receipt_directory_fd,
            follow_symlinks=False,
        )
        published = True
        os.fsync(receipt_directory_fd)
        os.unlink(temporary_name, dir_fd=receipt_directory_fd)
        os.fsync(receipt_directory_fd)
    except FileExistsError as exc:
        raise MigrationError(f"audit adoption receipt already exists and is immutable: {path}") from exc
    except OSError as exc:
        raise MigrationError(f"cannot publish immutable audit adoption receipt: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if receipt_directory_fd is not None and not published:
            try:
                os.unlink(temporary_name, dir_fd=receipt_directory_fd)
                os.fsync(receipt_directory_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        if receipt_directory_fd is not None:
            os.close(receipt_directory_fd)


def _audit_adoption_image_binding(payload: dict[str, object]) -> tuple[str, int, int]:
    """Return the receipt-bound initial image digest, size, and durable marker."""
    expected_image_sha256 = payload.get("audit_image_sha256")
    expected_image_size = payload.get("audit_image_size")
    application_id = payload.get("audit_application_id")
    if (
        not isinstance(expected_image_sha256, str)
        or not isinstance(expected_image_size, int)
        or not isinstance(application_id, int)
    ):
        raise MigrationError("audit adoption receipt lacks a canonical audit image binding")
    return expected_image_sha256, expected_image_size, application_id


def _load_audit_adoption_receipt(archive_root: Path) -> tuple[Path, dict[str, object]] | None:
    receipt_path = audit_adoption_receipt_path(archive_root)
    try:
        receipt_directory_fd = _open_audit_adoption_receipt_directory(
            receipt_path,
            archive_root=archive_root,
            create=False,
        )
    except FileNotFoundError:
        return None
    receipt_fd: int | None = None
    try:
        receipt_fd = os.open(
            receipt_path.name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=receipt_directory_fd,
        )
        metadata = os.fstat(receipt_fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise MigrationError(f"invalid audit adoption receipt ownership or mode: {receipt_path}")
        with os.fdopen(receipt_fd, "r", encoding="utf-8") as stream:
            receipt_fd = None
            payload = json.load(stream)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationError(f"invalid audit adoption receipt: {receipt_path}") from exc
    finally:
        if receipt_fd is not None:
            os.close(receipt_fd)
        os.close(receipt_directory_fd)
    if not isinstance(payload, dict) or payload.get("format") != _AUDIT_ADOPTION_RECEIPT_FORMAT:
        raise MigrationError(f"audit adoption receipt format mismatch: {receipt_path}")
    digest = payload.get("receipt_sha256")
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    if not isinstance(digest, str) or digest != _canonical_json_sha256(unsigned):
        raise MigrationError(f"audit adoption receipt checksum mismatch: {receipt_path}")
    if payload.get("source_user_authority_digest") != _audit_adoption_authority_digest(archive_root):
        raise MigrationError("audit adoption receipt source/user authority mismatch")
    _audit_adoption_image_binding(payload)
    return receipt_path, payload


def _audit_file_identity(path: Path) -> tuple[int, int]:
    """Read the audit leaf identity without following a replacement symlink."""
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise MigrationError(f"cannot inspect adopted audit tier: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise MigrationError(f"adopted audit tier is not a regular file: {path}")
    return metadata.st_dev, metadata.st_ino


def _audit_live_metadata(audit_path: Path) -> tuple[int, int, tuple[str, ...]]:
    """Read the durable markers that remain valid after an in-place migration."""
    uri = f"{audit_path.resolve(strict=False).as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0] or 0)
        application_id = int(connection.execute("PRAGMA application_id").fetchone()[0] or 0)
        quick_check = tuple(str(row[0]) for row in connection.execute("PRAGMA quick_check"))
    return version, application_id, quick_check


def _audit_file_sha256(audit_path: Path) -> str:
    """Hash the exact regular-file image a continuity rebind is about to bless."""

    _audit_file_identity(audit_path)
    digest = hashlib.sha256()
    try:
        with audit_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MigrationError(f"cannot hash adopted audit tier: {audit_path}") from exc
    return digest.hexdigest()


def _validate_initial_audit_image(
    audit_path: Path,
    *,
    expected_image_sha256: str,
    expected_image_size: int,
    expected_application_id: int,
    expected_initial_version: int,
) -> tuple[int, int]:
    """Authenticate the receipt-bound initial image before binding its identity."""
    file_identity = _audit_file_identity(audit_path)
    try:
        audit_image = audit_path.read_bytes()
    except OSError as exc:
        raise MigrationError(f"cannot read adopted audit tier: {audit_path}") from exc
    if len(audit_image) != expected_image_size or hashlib.sha256(audit_image).hexdigest() != expected_image_sha256:
        raise MigrationError("audit adoption receipt does not match the published canonical audit image")
    version, application_id, quick_check = _audit_live_metadata(audit_path)
    if version != expected_initial_version or application_id != expected_application_id or quick_check != ("ok",):
        raise MigrationError("audit adoption receipt does not match the published canonical audit image")
    return file_identity


def _load_audit_adoption_continuity(archive_root: Path) -> dict[str, object] | None:
    """Load the immutable audit-file identity record, if publication reached it."""
    continuity_path = _audit_adoption_continuity_path(archive_root)
    try:
        continuity_directory_fd = _open_audit_adoption_receipt_directory(
            continuity_path,
            archive_root=archive_root,
            create=False,
        )
    except FileNotFoundError:
        return None
    continuity_fd: int | None = None
    try:
        continuity_fd = os.open(
            continuity_path.name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=continuity_directory_fd,
        )
        metadata = os.fstat(continuity_fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise MigrationError(f"invalid audit adoption continuity ownership or mode: {continuity_path}")
        with os.fdopen(continuity_fd, "r", encoding="utf-8") as stream:
            continuity_fd = None
            payload = json.load(stream)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationError(f"invalid audit adoption continuity record: {continuity_path}") from exc
    finally:
        if continuity_fd is not None:
            os.close(continuity_fd)
        os.close(continuity_directory_fd)
    if not isinstance(payload, dict) or payload.get("format") != _AUDIT_ADOPTION_CONTINUITY_FORMAT:
        raise MigrationError(f"audit adoption continuity format mismatch: {continuity_path}")
    digest = payload.get("continuity_sha256")
    unsigned = dict(payload)
    unsigned.pop("continuity_sha256", None)
    if not isinstance(digest, str) or digest != _canonical_json_sha256(unsigned):
        raise MigrationError(f"audit adoption continuity checksum mismatch: {continuity_path}")
    return payload


def _audit_restore_records(archive_root: Path) -> list[tuple[Path, dict[str, object]]]:
    """Read restore state through the fixed, no-follow archive ledger path."""
    marker_path = _audit_adoption_continuity_path(archive_root)
    try:
        directory_fd = _open_audit_adoption_receipt_directory(marker_path, archive_root=archive_root, create=False)
    except FileNotFoundError:
        return []
    records: list[tuple[Path, dict[str, object]]] = []
    try:
        for name in os.listdir(directory_fd):
            match = _AUDIT_ADOPTION_RESTORE_NAME.fullmatch(name)
            if match is None:
                continue
            path = marker_path.with_name(name)
            fd: int | None = None
            try:
                fd = os.open(
                    name, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd
                )
                metadata = os.fstat(fd)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(metadata.st_mode) & 0o022
                ):
                    raise MigrationError(f"invalid audit restore record ownership or mode: {path}")
                with os.fdopen(fd, "r", encoding="utf-8") as stream:
                    fd = None
                    payload = json.load(stream)
            except (OSError, json.JSONDecodeError) as exc:
                raise MigrationError(f"invalid audit restore record: {path}") from exc
            finally:
                if fd is not None:
                    os.close(fd)
            if not isinstance(payload, dict) or payload.get("format") != _AUDIT_ADOPTION_RESTORE_FORMAT:
                raise MigrationError(f"audit restore record format mismatch: {path}")
            checksum_key = "restore_sha256" if match["state"] == "prepared" else "continuity_sha256"
            checksum = payload.get(checksum_key)
            unsigned = dict(payload)
            unsigned.pop(checksum_key, None)
            if not isinstance(checksum, str) or checksum != _canonical_json_sha256(unsigned):
                raise MigrationError(f"audit restore record checksum mismatch: {path}")
            if payload.get("state") != match["state"] or payload.get("operation_id") != match["operation"]:
                raise MigrationError(f"audit restore record filename does not match its payload: {path}")
            if payload.get("generation") != int(match["generation"]):
                raise MigrationError(f"audit restore record generation mismatch: {path}")
            records.append((path, payload))
    finally:
        os.close(directory_fd)
    return records


def _latest_audit_adoption_continuity(
    archive_root: Path, *, allow_incomplete_restore: bool = False
) -> dict[str, object] | None:
    """Follow the immutable restore chain and expose its current generation."""
    continuity = _load_audit_adoption_continuity(archive_root)
    if continuity is None:
        return None
    current_digest = continuity.get("continuity_sha256")
    if not isinstance(current_digest, str):
        raise MigrationError("audit adoption continuity lacks its immutable checksum")
    records_by_generation: dict[int, dict[str, dict[str, object]]] = {}
    for _path, payload in _audit_restore_records(archive_root):
        generation = payload["generation"]
        state = payload["state"]
        assert isinstance(generation, int)
        assert isinstance(state, str)
        states = records_by_generation.setdefault(generation, {})
        if state in states:
            raise MigrationError("audit restore records contain duplicate generation state")
        states[state] = payload
    for expected_generation in range(1, len(records_by_generation) + 1):
        if expected_generation not in records_by_generation:
            raise MigrationError("audit restore continuity generations are not contiguous")
        states = records_by_generation[expected_generation]
        prepared = states.get("prepared")
        committed = states.get("committed")
        if prepared is None or committed is None:
            if allow_incomplete_restore and prepared is not None and committed is None:
                return continuity
            raise MigrationError(
                "adopted audit restore is prepared but incomplete; rerun maintenance migrate-tier audit "
                "--restore-adopted-audit with the same verified full_evidence backup"
            )
        if (
            prepared.get("previous_continuity_sha256") != current_digest
            or committed.get("previous_continuity_sha256") != current_digest
            or committed.get("prepared_restore_sha256") != prepared.get("restore_sha256")
            or committed.get("receipt_sha256") != continuity.get("receipt_sha256")
            or committed.get("source_user_authority_digest") != continuity.get("source_user_authority_digest")
        ):
            raise MigrationError("audit restore continuity chain does not match the adopted archive")
        next_digest = committed.get("continuity_sha256")
        if not isinstance(next_digest, str):
            raise MigrationError("committed audit restore record lacks its continuity checksum")
        continuity = committed
        current_digest = next_digest
    return continuity


def _write_audit_adoption_continuity(
    archive_root: Path,
    *,
    receipt_payload: dict[str, object],
    expected_initial_file_identity: tuple[int, int],
    expected_audit_image_sha256: str,
) -> None:
    """Publish the post-link audit identity that later detects stale replacement."""
    audit_path = archive_root / "audit.db"
    device, inode = _audit_file_identity(audit_path)
    if (device, inode) != expected_initial_file_identity:
        raise MigrationError("audit tier changed before recording adoption continuity")
    if _audit_file_sha256(audit_path) != expected_audit_image_sha256:
        raise MigrationError("audit image changed before recording adoption continuity")
    continuity_path = _audit_adoption_continuity_path(archive_root)
    payload: dict[str, object] = {
        "format": _AUDIT_ADOPTION_CONTINUITY_FORMAT,
        "receipt_sha256": receipt_payload["receipt_sha256"],
        "source_user_authority_digest": receipt_payload["source_user_authority_digest"],
        "audit_device": device,
        "audit_inode": inode,
        "audit_image_sha256": expected_audit_image_sha256,
    }
    unsigned = dict(payload)
    payload["continuity_sha256"] = _canonical_json_sha256(unsigned)
    _write_immutable_audit_adoption_receipt(
        continuity_path,
        payload,
        archive_root=archive_root,
        checksum_key="continuity_sha256",
    )
    # The immutable receipt remains operator evidence. The machine authority
    # is the cross-tier head, seeded with the authenticated initial image so a
    # byte-for-byte stale copy on the same inode cannot be blessed later.
    receipt_sha256 = receipt_payload.get("receipt_sha256")
    if not isinstance(receipt_sha256, str):
        raise MigrationError("audit adoption receipt lacks its checksum")
    AuditContinuityCoordinator(archive_root).seed_or_rebind(
        mutation_id=f"audit-adoption:{receipt_sha256}",
        now_ms=int(time.time() * 1000),
        evidence={
            "kind": "adoption",
            "receipt_sha256": receipt_sha256,
            "audit_image_sha256": expected_audit_image_sha256,
        },
    )
    if _audit_file_identity(audit_path) != (device, inode):
        raise MigrationError("audit tier changed while recording adoption continuity")


def _validate_audit_adoption_continuity(
    archive_root: Path,
    *,
    receipt_payload: dict[str, object],
    expected_initial_file_identity: tuple[int, int] | None,
) -> None:
    """Require the published audit path to retain its adopted live identity."""
    continuity = _latest_audit_adoption_continuity(archive_root)
    if continuity is None:
        if expected_initial_file_identity is None:
            raise MigrationError("audit adoption continuity is missing without an authenticated initial image")
        _write_audit_adoption_continuity(
            archive_root,
            receipt_payload=receipt_payload,
            expected_initial_file_identity=expected_initial_file_identity,
            expected_audit_image_sha256=cast(str, receipt_payload["audit_image_sha256"]),
        )
        continuity = _latest_audit_adoption_continuity(archive_root)
    assert continuity is not None
    expected = (continuity.get("audit_device"), continuity.get("audit_inode"))
    if (
        continuity.get("receipt_sha256") != receipt_payload.get("receipt_sha256")
        or continuity.get("source_user_authority_digest") != receipt_payload.get("source_user_authority_digest")
        or not all(isinstance(value, int) for value in expected)
        or _audit_file_identity(archive_root / "audit.db") != expected
    ):
        raise MigrationError("audit adoption continuity does not match the live audit tier")


def _validate_audit_adoption_recovery_evidence(payload: dict[str, object], *, archive_root: Path) -> None:
    manifest_value = payload.get("backup_manifest")
    receipt_value = payload.get("backup_verification_receipt")
    if not isinstance(manifest_value, str) or not isinstance(receipt_value, str):
        raise MigrationError("audit adoption receipt lacks backup recovery evidence")
    manifest_path = Path(manifest_value)
    verification_receipt = Path(receipt_value)
    try:
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        receipt_sha256 = hashlib.sha256(verification_receipt.read_bytes()).hexdigest()
    except OSError as exc:
        raise MigrationError("audit adoption recovery evidence is unavailable") from exc
    if (
        payload.get("backup_manifest_sha256") != manifest_sha256
        or payload.get("backup_verification_receipt_sha256") != receipt_sha256
    ):
        raise MigrationError("audit adoption recovery evidence no longer matches its immutable receipt")
    validated_manifest, validated_receipt = validate_full_evidence_backup_for_audit_adoption(
        manifest_path,
        archive_root=archive_root,
    )
    if validated_manifest != manifest_path.resolve() or validated_receipt != verification_receipt.resolve():
        raise MigrationError("audit adoption recovery evidence path changed")


def _recover_pending_audit_adoption(
    archive_root: Path,
    receipt_path: Path,
    payload: dict[str, object],
) -> None:
    """Complete a missing audit publication from its immutable, verified intent."""
    audit_path = archive_root / "audit.db"
    _validate_audit_adoption_recovery_evidence(payload, archive_root=archive_root)
    expected_sha256, expected_size, application_id = _audit_adoption_image_binding(payload)

    def prepare_initialized_image(connection: sqlite3.Connection) -> None:
        connection.execute(f"PRAGMA application_id = {application_id}")

    def revalidate_before_publish(initialized_image: bytes) -> None:
        if hashlib.sha256(initialized_image).hexdigest() != expected_sha256 or len(initialized_image) != expected_size:
            raise MigrationError("audit adoption receipt does not match its recoverable canonical audit image")
        _validate_audit_adoption_recovery_evidence(payload, archive_root=archive_root)
        if receipt_path != audit_adoption_receipt_path(archive_root):
            raise MigrationError("audit adoption receipt path changed during recovery")

    initialize_missing_durable_tier(
        audit_path,
        ArchiveTier.AUDIT,
        permit_established_archive=True,
        prepare_initialized_image=prepare_initialized_image,
        pre_publish_check=revalidate_before_publish,
    )


def recover_pending_audit_adoption(archive_root: Path) -> bool:
    """Publish a receipt-backed missing audit file before startup classification."""
    archive_root = archive_root.resolve()
    receipt = _load_audit_adoption_receipt(archive_root)
    audit_path = archive_root / "audit.db"
    if receipt is None or audit_path.is_file():
        return False
    if _latest_audit_adoption_continuity(archive_root) is not None:
        raise MigrationError(
            "adopted audit tier is missing after continuity was recorded; run maintenance migrate-tier audit "
            "--restore-adopted-audit --backup-manifest <verified-full-evidence>/manifest.json"
        )
    receipt_path, payload = receipt
    _recover_pending_audit_adoption(archive_root, receipt_path, payload)
    return True


def validate_audit_adoption_receipt(archive_root: Path, *, require_initial_image: bool = False) -> Path | None:
    """Validate a present adoption receipt before startup consumes its audit tier."""
    archive_root = archive_root.resolve()
    receipt = _load_audit_adoption_receipt(archive_root)
    if receipt is None:
        return None
    receipt_path, payload = receipt
    expected_image_sha256, expected_image_size, expected_application_id = _audit_adoption_image_binding(payload)
    expected_initial_version = payload.get("audit_user_version")
    if not isinstance(expected_initial_version, int):
        raise MigrationError("audit adoption receipt lacks its initial audit schema version")
    audit_path = archive_root / "audit.db"
    continuity = _latest_audit_adoption_continuity(archive_root)
    if not audit_path.is_file():
        if continuity is not None:
            raise MigrationError(
                "adopted audit tier is missing after continuity was recorded; run maintenance migrate-tier audit "
                "--restore-adopted-audit --backup-manifest <verified-full-evidence>/manifest.json"
            )
        _recover_pending_audit_adoption(archive_root, receipt_path, payload)
        require_initial_image = True
    initial_file_identity: tuple[int, int] | None = None
    if continuity is None or require_initial_image:
        initial_file_identity = _validate_initial_audit_image(
            audit_path,
            expected_image_sha256=expected_image_sha256,
            expected_image_size=expected_image_size,
            expected_application_id=expected_application_id,
            expected_initial_version=expected_initial_version,
        )
    else:
        version, application_id, quick_check = _audit_live_metadata(audit_path)
        if version < expected_initial_version or application_id != expected_application_id or quick_check != ("ok",):
            raise MigrationError("audit adoption receipt does not match the live audit tier")
    _validate_audit_adoption_continuity(
        archive_root,
        receipt_payload=payload,
        expected_initial_file_identity=initial_file_identity,
    )
    return receipt_path


def adopt_missing_audit_tier(
    path: Path,
    *,
    backup_manifest: Path,
    directory_fd: int,
    stopped_daemon_check: Callable[[], str],
) -> tuple[int, Path]:
    """Adopt canonical ``audit.db`` into an established, offline archive.

    The receipt is published first, so a crash cannot leave an unproven audit
    tier.  It names the authenticated full-evidence backup and expected
    canonical image; startup validates that immutable intent against the
    linked database before accepting it.
    """
    if path.name != "audit.db":
        raise MigrationError(f"established-archive adoption is only supported for audit.db: {path}")
    archive_root = path.parent.resolve()
    if path.exists() or path.is_symlink():
        raise MigrationError(f"audit tier already exists; refusing established-archive adoption: {path}")
    receipt_path = audit_adoption_receipt_path(archive_root)
    if _load_audit_adoption_receipt(archive_root) is not None:
        validate_audit_adoption_receipt(archive_root)
        return 1, receipt_path
    stopped_evidence = stopped_daemon_check()
    manifest_path, verification_receipt = validate_full_evidence_backup_for_audit_adoption(
        backup_manifest,
        archive_root=archive_root,
    )
    initial_authority_digest = _audit_adoption_authority_digest(archive_root)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    verification_receipt_sha256 = hashlib.sha256(verification_receipt.read_bytes()).hexdigest()
    application_id = (
        int.from_bytes(hashlib.sha256(f"{initial_authority_digest}:{manifest_sha256}".encode()).digest()[:4], "big")
        & 0x7FFFFFFF
    )
    if application_id == 0:
        application_id = 1
    payload: dict[str, object] = {}

    def prepare_initialized_image(connection: sqlite3.Connection) -> None:
        connection.execute(f"PRAGMA application_id = {application_id}")

    def revalidate_before_publish(initialized_image: bytes) -> None:
        if path.exists() or path.is_symlink():
            raise MigrationError(f"audit tier appeared during established-archive adoption: {path}")
        if stopped_daemon_check() != stopped_evidence:
            raise MigrationError("daemon stopped proof changed during audit adoption")
        validate_full_evidence_backup_for_audit_adoption(backup_manifest, archive_root=archive_root)
        if _audit_adoption_authority_digest(archive_root) != initial_authority_digest:
            raise MigrationError("source/user authority changed during audit adoption")
        payload.update(
            {
                "format": _AUDIT_ADOPTION_RECEIPT_FORMAT,
                "source_user_authority_digest": initial_authority_digest,
                "backup_manifest": str(manifest_path.resolve()),
                "backup_manifest_sha256": manifest_sha256,
                "backup_verification_receipt": str(verification_receipt.resolve()),
                "backup_verification_receipt_sha256": verification_receipt_sha256,
                "stopped_daemon_evidence_ref": stopped_evidence,
                "single_writer_evidence_ref": "proof:archive-ownership-lock",
                "audit_schema_inventory_sha256": _audit_schema_inventory_sha256(),
                "audit_user_version": ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT],
                "audit_application_id": application_id,
                "audit_image_sha256": hashlib.sha256(initialized_image).hexdigest(),
                "audit_image_size": len(initialized_image),
            }
        )
        _write_immutable_audit_adoption_receipt(
            receipt_path,
            payload,
            archive_root=archive_root,
            archive_directory_fd=directory_fd,
        )

    version = initialize_missing_durable_tier(
        path,
        ArchiveTier.AUDIT,
        directory_fd=directory_fd,
        permit_established_archive=True,
        prepare_initialized_image=prepare_initialized_image,
        pre_publish_check=revalidate_before_publish,
    )
    validate_audit_adoption_receipt(archive_root, require_initial_image=True)
    return version, receipt_path


def _audit_restore_artifact_binding(receipt_path: Path) -> tuple[str, int, int]:
    """Read the audit artifact facts after receipt authentication succeeded."""
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationError("cannot read adopted-audit restore verification receipt") from exc
    artifacts = receipt.get("tier_artifacts") if isinstance(receipt, dict) else None
    audit = (
        next((item for item in artifacts if isinstance(item, dict) and item.get("tier") == "audit"), None)
        if isinstance(artifacts, list)
        else None
    )
    if not isinstance(audit, dict):
        raise MigrationError("adopted-audit restore receipt lacks audit artifact evidence")
    sha256, size, version = audit.get("sha256"), audit.get("size_bytes"), audit.get("user_version")
    if not isinstance(sha256, str) or not isinstance(size, int) or not isinstance(version, int):
        raise MigrationError("adopted-audit restore receipt has invalid audit artifact evidence")
    return sha256, size, version


def _copy_restore_artifact(source: Path, *, directory_fd: int, temporary_name: str, sha256: str, size: int) -> None:
    """Copy an exact no-follow, unlinked backup artifact into the owned root."""
    source_fd: int | None = None
    target_fd: int | None = None
    try:
        source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
        source_metadata = os.fstat(source_fd)
        if not stat.S_ISREG(source_metadata.st_mode) or source_metadata.st_nlink != 1:
            raise MigrationError("adopted-audit restore artifact is not an unlinked regular file")
        target_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        digest = hashlib.sha256()
        copied = 0
        while chunk := os.read(source_fd, 1024 * 1024):
            digest.update(chunk)
            copied += len(chunk)
            offset = 0
            while offset < len(chunk):
                written = os.write(target_fd, chunk[offset:])
                if written <= 0:
                    raise MigrationError("adopted-audit restore artifact copy made no progress")
                offset += written
        if copied != size or digest.hexdigest() != sha256:
            raise MigrationError("adopted-audit restore artifact changed while it was copied")
        os.fsync(target_fd)
    finally:
        if target_fd is not None:
            os.close(target_fd)
        if source_fd is not None:
            os.close(source_fd)


def _remove_stale_restore_staging(*, directory_fd: int, temporary_name: str) -> None:
    """Remove one prior crash's private restore image before retrying its intent."""
    try:
        metadata = os.stat(temporary_name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise MigrationError(f"invalid stale adopted-audit restore staging file: {temporary_name}")
    os.unlink(temporary_name, dir_fd=directory_fd)
    os.fsync(directory_fd)


def _audit_file_matches_artifact(path: Path, *, sha256: str, size: int) -> bool:
    """Check whether an interrupted restore already published the intended image."""
    try:
        _audit_file_identity(path)
        if path.stat().st_size == size:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest() == sha256
    except OSError:
        return False
    return False


def restore_adopted_audit_tier(
    path: Path,
    *,
    backup_manifest: Path,
    directory_fd: int,
    stopped_daemon_check: Callable[[], str],
) -> Path:
    """Restore adopted ``audit.db`` and append its new continuity generation."""
    if path.name != "audit.db":
        raise MigrationError(f"adopted-audit restore is only supported for audit.db: {path}")
    archive_root = path.parent.resolve()
    receipt = _load_audit_adoption_receipt(archive_root)
    if receipt is None:
        raise MigrationError("adopted-audit restore requires an existing audit adoption receipt")
    _receipt_path, adoption = receipt
    continuity = _latest_audit_adoption_continuity(archive_root, allow_incomplete_restore=True)
    if continuity is None or continuity.get("receipt_sha256") != adoption.get("receipt_sha256"):
        raise MigrationError("adopted-audit restore requires completed continuity for this adoption receipt")
    stopped_evidence = stopped_daemon_check()
    manifest_path, verification_receipt = validate_full_evidence_backup_for_adopted_audit_restore(
        backup_manifest, archive_root=archive_root
    )
    artifact_sha256, artifact_size, artifact_version = _audit_restore_artifact_binding(verification_receipt)
    expected_application_id = adoption.get("audit_application_id")
    expected_initial_version = adoption.get("audit_user_version")
    if not isinstance(expected_application_id, int) or not isinstance(expected_initial_version, int):
        raise MigrationError("audit adoption receipt lacks its durable SQLite markers")
    backup_version, backup_application_id, backup_quick_check = _audit_live_metadata(manifest_path.parent / "audit.db")
    if (
        backup_version != artifact_version
        or backup_version < expected_initial_version
        or backup_application_id != expected_application_id
        or backup_quick_check != ("ok",)
    ):
        raise MigrationError("adopted-audit restore artifact does not belong to this audit adoption")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    verification_receipt_sha256 = hashlib.sha256(verification_receipt.read_bytes()).hexdigest()

    def revalidate_exact_backup() -> None:
        current_manifest, current_receipt = validate_full_evidence_backup_for_adopted_audit_restore(
            backup_manifest, archive_root=archive_root
        )
        if (
            current_manifest.resolve() != manifest_path.resolve()
            or current_receipt.resolve() != verification_receipt.resolve()
            or hashlib.sha256(current_manifest.read_bytes()).hexdigest() != manifest_sha256
            or hashlib.sha256(current_receipt.read_bytes()).hexdigest() != verification_receipt_sha256
        ):
            raise MigrationError("adopted-audit restore backup changed during the operation")

    previous_continuity_sha256 = continuity.get("continuity_sha256")
    if not isinstance(previous_continuity_sha256, str):
        raise MigrationError("adopted-audit restore continuity lacks its checksum")
    restore_records = _audit_restore_records(archive_root)
    committed_generations: list[int] = []
    pending_records: list[tuple[Path, dict[str, object]]] = []
    committed_operations: set[tuple[int, str]] = set()
    for record_path, payload in restore_records:
        generation_value = payload.get("generation")
        if payload.get("state") == "committed" and isinstance(generation_value, int):
            committed_generations.append(generation_value)
            operation_value = payload.get("operation_id")
            if isinstance(operation_value, str):
                committed_operations.add((generation_value, operation_value))
        elif payload.get("state") == "prepared":
            pending_records.append((record_path, payload))
    unresolved_records: list[tuple[Path, dict[str, object]]] = []
    for record_path, payload in pending_records:
        generation_value = payload.get("generation")
        operation_value = payload.get("operation_id")
        if not isinstance(generation_value, int) or not isinstance(operation_value, str):
            raise MigrationError("incomplete adopted-audit restore has invalid identity fields")
        if (generation_value, operation_value) not in committed_operations:
            unresolved_records.append((record_path, payload))
    pending_records = unresolved_records
    if len(pending_records) > 1:
        raise MigrationError("adopted-audit restore has multiple incomplete continuity records")
    if pending_records:
        _pending_path, pending_payload = pending_records[0]
        generation_value = pending_payload.get("generation")
        operation_value = pending_payload.get("operation_id")
        assert isinstance(generation_value, int)
        assert isinstance(operation_value, str)
        generation = generation_value
        operation_id = operation_value
    else:
        generation = 1 + max(committed_generations, default=0)
        operation_id = secrets.token_hex(16)
    base_payload: dict[str, object] = {
        "format": _AUDIT_ADOPTION_RESTORE_FORMAT,
        "generation": generation,
        "operation_id": operation_id,
        "previous_continuity_sha256": previous_continuity_sha256,
        "receipt_sha256": adoption["receipt_sha256"],
        "source_user_authority_digest": adoption["source_user_authority_digest"],
        "backup_manifest_sha256": manifest_sha256,
        "backup_verification_receipt_sha256": verification_receipt_sha256,
        "audit_artifact_sha256": artifact_sha256,
        "audit_artifact_size": artifact_size,
        "audit_artifact_user_version": artifact_version,
        "stopped_daemon_evidence_ref": stopped_evidence,
        "single_writer_evidence_ref": "proof:archive-ownership-lock",
    }
    if pending_records:
        prepared_path, prepared = pending_records[0]
        expected_prepared = {**base_payload, "state": "prepared"}
        if any(prepared.get(key) != value for key, value in expected_prepared.items()):
            raise MigrationError("incomplete adopted-audit restore does not match the supplied verified backup")
    else:
        prepared_path = _audit_adoption_continuity_path(archive_root).with_name(
            f"audit-restore.{generation}.{operation_id}.prepared.json"
        )
        prepared = {**base_payload, "state": "prepared"}
        _write_immutable_audit_adoption_receipt(
            prepared_path,
            prepared,
            archive_root=archive_root,
            archive_directory_fd=directory_fd,
            checksum_key="restore_sha256",
        )
    temporary_name = f".audit.db.restore-{operation_id}.tmp"
    _remove_stale_restore_staging(directory_fd=directory_fd, temporary_name=temporary_name)
    published = False
    try:
        if _audit_file_matches_artifact(archive_root / "audit.db", sha256=artifact_sha256, size=artifact_size):
            published = True
        else:
            _copy_restore_artifact(
                manifest_path.parent / "audit.db",
                directory_fd=directory_fd,
                temporary_name=temporary_name,
                sha256=artifact_sha256,
                size=artifact_size,
            )
        if stopped_daemon_check() != stopped_evidence:
            raise MigrationError("daemon stopped proof changed during adopted-audit restore")
        revalidate_exact_backup()
        if _audit_adoption_authority_digest(archive_root) != adoption.get("source_user_authority_digest"):
            raise MigrationError("source/user authority changed during adopted-audit restore")
        if not published:
            os.replace(temporary_name, path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            os.fsync(directory_fd)
            published = True
        if _audit_file_sha256(path) != artifact_sha256:
            raise MigrationError("adopted-audit restore published image changed before continuity rebind")
        identity = _audit_file_identity(path)
        version, application_id, quick_check = _audit_live_metadata(path)
        if version != artifact_version or application_id != expected_application_id or quick_check != ("ok",):
            raise MigrationError("adopted-audit restore published artifact is not the verified SQLite image")
        if stopped_daemon_check() != stopped_evidence:
            raise MigrationError("daemon stopped proof changed after adopted-audit restore publication")
        revalidate_exact_backup()
        committed_path = prepared_path.with_name(prepared_path.name.replace(".prepared.json", ".committed.json"))
        prepared_restore_sha256 = prepared.get("restore_sha256")
        if not isinstance(prepared_restore_sha256, str):
            prepared_restore_sha256 = _canonical_json_sha256(prepared)
        committed = {
            **base_payload,
            "state": "committed",
            "prepared_restore_sha256": prepared_restore_sha256,
            "audit_device": identity[0],
            "audit_inode": identity[1],
            "audit_image_sha256": artifact_sha256,
        }
        committed["continuity_sha256"] = _canonical_json_sha256(committed)
        AuditContinuityCoordinator(archive_root).seed_or_rebind(
            mutation_id=f"audit-restore:{operation_id}",
            now_ms=int(time.time() * 1000),
            evidence={
                "kind": "verified_restore",
                "restore_continuity_sha256": committed["continuity_sha256"],
                "audit_image_sha256": artifact_sha256,
            },
        )
        _write_immutable_audit_adoption_receipt(
            committed_path,
            committed,
            archive_root=archive_root,
            archive_directory_fd=directory_fd,
            checksum_key="continuity_sha256",
        )
        return committed_path
    finally:
        if not published:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=directory_fd)
                os.fsync(directory_fd)


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
    "adopt_missing_audit_tier",
    "audit_adoption_receipt_path",
    "ArchiveOwnershipError",
    "execute_durable_change_train",
    "initialize_missing_durable_tier",
    "reconcile_durable_change_trains_on_startup",
    "recover_pending_audit_adoption",
    "restore_adopted_audit_tier",
    "validate_audit_adoption_receipt",
]
