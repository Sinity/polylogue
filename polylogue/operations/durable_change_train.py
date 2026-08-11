"""Product-layer adapters for durable schema-change train operations."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import stat
import sys
from collections.abc import Callable
from contextlib import closing, suppress
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
from polylogue.storage.sqlite.migration_runner import (
    DurableRuntimeConsumerResult,
    MigrationError,
    _canonical_json_sha256,
    capture_durable_schema_inventory,
    validate_full_evidence_backup_for_audit_adoption,
)

_AUDIT_ADOPTION_RECEIPT_FORMAT = "polylogue.audit-tier-adoption.v1"
_AUDIT_ADOPTION_RECEIPT_NAME = "audit-adoption.json"


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
    expected_path = audit_adoption_receipt_path(archive_root)
    if path != expected_path:
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
) -> None:
    """Publish one pre-publication receipt without replacement and fsync it."""
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    payload = {**unsigned, "receipt_sha256": _canonical_json_sha256(unsigned)}
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
    from polylogue.storage.archive_identity import ArchiveIdentity

    if payload.get("archive_identity_digest") != ArchiveIdentity.resolve(archive_root).authority_identity_digest:
        raise MigrationError("audit adoption receipt archive identity mismatch")
    if payload.get("audit_schema_inventory_sha256") != _audit_schema_inventory_sha256():
        raise MigrationError("audit adoption receipt canonical audit DDL mismatch")
    _audit_adoption_image_binding(payload)
    return receipt_path, payload


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


def validate_audit_adoption_receipt(archive_root: Path, *, require_initial_image: bool = False) -> Path | None:
    """Validate a present adoption receipt before startup consumes its audit tier."""
    archive_root = archive_root.resolve()
    receipt = _load_audit_adoption_receipt(archive_root)
    if receipt is None:
        return None
    receipt_path, payload = receipt
    expected_image_sha256, expected_image_size, expected_application_id = _audit_adoption_image_binding(payload)
    audit_path = archive_root / "audit.db"
    if not audit_path.is_file():
        _recover_pending_audit_adoption(archive_root, receipt_path, payload)
        require_initial_image = True
    if require_initial_image:
        try:
            audit_image = audit_path.read_bytes()
        except OSError as exc:
            raise MigrationError(f"cannot read adopted audit tier: {audit_path}") from exc
        if len(audit_image) != expected_image_size or hashlib.sha256(audit_image).hexdigest() != expected_image_sha256:
            raise MigrationError("audit adoption receipt does not match the published canonical audit image")
    with closing(sqlite3.connect(f"file:{audit_path}?mode=ro", uri=True)) as connection:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0] or 0)
        application_id = int(connection.execute("PRAGMA application_id").fetchone()[0] or 0)
        quick_check = tuple(str(row[0]) for row in connection.execute("PRAGMA quick_check"))
        schema_digest = capture_durable_schema_inventory(connection).sha256
    if (
        version != 1
        or application_id != expected_application_id
        or quick_check != ("ok",)
        or schema_digest != _audit_schema_inventory_sha256()
    ):
        raise MigrationError("audit adoption receipt does not match a canonical audit v1 tier")
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
    from polylogue.storage.archive_identity import ArchiveIdentity

    initial_archive_identity_digest = ArchiveIdentity.resolve(archive_root).authority_identity_digest
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    verification_receipt_sha256 = hashlib.sha256(verification_receipt.read_bytes()).hexdigest()
    application_id = (
        int.from_bytes(
            hashlib.sha256(f"{initial_archive_identity_digest}:{manifest_sha256}".encode()).digest()[:4], "big"
        )
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
        archive_identity_digest = ArchiveIdentity.resolve(archive_root).authority_identity_digest
        if archive_identity_digest != initial_archive_identity_digest:
            raise MigrationError("archive identity changed during audit adoption")
        payload.update(
            {
                "format": _AUDIT_ADOPTION_RECEIPT_FORMAT,
                "archive_identity_digest": initial_archive_identity_digest,
                "backup_manifest": str(manifest_path.resolve()),
                "backup_manifest_sha256": manifest_sha256,
                "backup_verification_receipt": str(verification_receipt.resolve()),
                "backup_verification_receipt_sha256": verification_receipt_sha256,
                "stopped_daemon_evidence_ref": stopped_evidence,
                "single_writer_evidence_ref": "proof:archive-ownership-lock",
                "audit_schema_inventory_sha256": _audit_schema_inventory_sha256(),
                "audit_user_version": 1,
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
    "validate_audit_adoption_receipt",
]
