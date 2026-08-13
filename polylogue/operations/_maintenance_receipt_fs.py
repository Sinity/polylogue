"""Descriptor-pinned publication for retained offline-maintenance receipts."""

from __future__ import annotations

import os
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path


class MaintenanceReceiptPathError(RuntimeError):
    """A maintenance-state path could not be pinned without following links."""


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _simple_name(value: str, *, label: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise MaintenanceReceiptPathError(f"{label} is not a canonical single path component: {value!r}")
    return value


def _open_directory(path: Path, *, label: str) -> int:
    try:
        descriptor = os.open(path, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise MaintenanceReceiptPathError(f"cannot pin {label} without following links: {path}") from exc
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise MaintenanceReceiptPathError(f"{label} is not a real directory: {path}")
    return descriptor


def _open_directory_at(parent_fd: int, name: str, *, label: str) -> int:
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise MaintenanceReceiptPathError(f"cannot pin {label} without following links: {name}") from exc
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise MaintenanceReceiptPathError(f"{label} is not a real directory: {name}")
    return descriptor


def _remove_created_empty_child(parent_fd: int, name: str, *, expected: os.stat_result) -> None:
    """Remove only the empty child this operation created, through its pinned parent."""
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise MaintenanceReceiptPathError(f"cannot inspect fresh maintenance receipt directory: {name}") from exc
    if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
        raise MaintenanceReceiptPathError(f"fresh maintenance receipt directory changed before cleanup: {name}")
    try:
        os.rmdir(name, dir_fd=parent_fd)
    except OSError as exc:
        raise MaintenanceReceiptPathError(f"cannot remove fresh maintenance receipt directory: {name}") from exc


@contextmanager
def maintenance_receipt_directory(archive_root: Path, directory_name: str) -> Iterator[int]:
    """Yield a pinned child of an existing, non-symlink ``.maintenance-state``."""
    child_name = _simple_name(directory_name, label="maintenance receipt directory name")
    root_fd = _open_directory(archive_root, label="archive root")
    state_fd = -1
    child_fd = -1
    try:
        state_fd = _open_directory_at(root_fd, ".maintenance-state", label="maintenance state")
        try:
            child_fd = os.open(child_name, _DIRECTORY_FLAGS, dir_fd=state_fd)
        except FileNotFoundError:
            created_child = False
            try:
                os.mkdir(child_name, mode=0o700, dir_fd=state_fd)
                created_child = True
            except FileExistsError:
                pass
            child_fd = _open_directory_at(state_fd, child_name, label="maintenance receipt directory")
            if created_child:
                child_metadata = os.fstat(child_fd)
                try:
                    os.fsync(state_fd)
                except OSError as exc:
                    _remove_created_empty_child(state_fd, child_name, expected=child_metadata)
                    raise MaintenanceReceiptPathError(
                        f"cannot persist maintenance receipt directory: {child_name}"
                    ) from exc
        except OSError as exc:
            raise MaintenanceReceiptPathError(
                f"cannot pin maintenance receipt directory without following links: {child_name}"
            ) from exc
        if not stat.S_ISDIR(os.fstat(child_fd).st_mode):
            raise MaintenanceReceiptPathError(f"maintenance receipt directory is not real: {child_name}")
        yield child_fd
    finally:
        if child_fd >= 0:
            os.close(child_fd)
        if state_fd >= 0:
            os.close(state_fd)
        os.close(root_fd)


def read_optional_receipt(directory_fd: int, filename: str) -> bytes | None:
    """Read one regular, single-linked receipt relative to a pinned directory."""
    name = _simple_name(filename, label="maintenance receipt filename")
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise MaintenanceReceiptPathError(f"cannot open maintenance receipt without following links: {name}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise MaintenanceReceiptPathError(f"maintenance receipt is not a regular single-linked file: {name}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def atomic_replace_receipt(directory_fd: int, filename: str, payload: bytes) -> None:
    """Fsync and atomically replace one file within a pinned receipt directory."""
    name = _simple_name(filename, label="maintenance receipt filename")
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    except OSError as exc:
        raise MaintenanceReceiptPathError(f"cannot atomically publish maintenance receipt: {name}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=directory_fd)


__all__ = [
    "MaintenanceReceiptPathError",
    "atomic_replace_receipt",
    "maintenance_receipt_directory",
    "read_optional_receipt",
]
