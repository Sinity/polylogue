"""Small, typed primitives for durable filesystem publication."""

from __future__ import annotations

import errno
import fcntl
import os
import shutil
import stat
import tempfile
from contextlib import suppress
from pathlib import Path

#: ``FICLONE`` ioctl: share the source's extents instead of copying bytes.
_FICLONE = 0x40049409
_REFLINK_UNSUPPORTED = frozenset({errno.EOPNOTSUPP, errno.ENOTTY, errno.EINVAL, errno.EXDEV})


class DurableFilesystemError(OSError):
    """A durable filesystem operation could not complete its barriers."""


def _fsync_directory(path: Path) -> None:
    """Persist directory entries in ``path``."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise DurableFilesystemError(f"cannot fsync directory: {path}") from exc
    finally:
        os.close(descriptor)


def sync_directory(path: Path) -> None:
    """Persist directory entries in ``path``."""
    _fsync_directory(path)


def write_once(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    """Create ``path`` exactly once, persisting its bytes and directory entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except OSError as exc:
        raise DurableFilesystemError(f"cannot durably create: {path}") from exc


def atomic_replace(path: Path, payload: bytes, *, mode: int | None = None) -> None:
    """Durably write bytes to a temporary file and replace ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = -1
    temporary_path: Path | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary_path = Path(temporary)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            if mode is not None:
                os.fchmod(descriptor, mode)
            stream.write(payload)
            stream.flush()
            os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except OSError as exc:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        raise DurableFilesystemError(f"cannot durably replace: {path}") from exc
    finally:
        if temporary_path is not None:
            with suppress(FileNotFoundError):
                temporary_path.unlink()


def append_line(path: Path, line: str | bytes) -> None:
    """Append one line and persist both the file bytes and directory entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = line.encode("utf-8") if isinstance(line, str) else line
    if not payload.endswith(b"\n"):
        payload += b"\n"
    try:
        with path.open("ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except OSError as exc:
        raise DurableFilesystemError(f"cannot durably append: {path}") from exc


__all__ = ["DurableFilesystemError", "append_line", "atomic_replace", "sync_directory", "write_once"]


def reflink_into(source_fd: int, destination_fd: int) -> bool:
    """Clone ``source_fd``'s contents into ``destination_fd``; ``False`` when unsupported."""
    try:
        fcntl.ioctl(destination_fd, _FICLONE, source_fd)
    except OSError as exc:
        if exc.errno in _REFLINK_UNSUPPORTED:
            return False
        raise
    return True


def clone_or_copy_replace(source: Path, destination: Path) -> None:
    """Place a copy of regular file ``source`` at ``destination``.

    The bytes are cloned by reflink where the filesystem supports it and
    copied otherwise, into a temporary sibling that replaces ``destination``
    only once complete: a failure leaves any earlier ``destination`` intact.
    Anything but a regular file (a FIFO, socket or device) is refused before
    it is opened, since opening a FIFO for reading blocks.
    """
    info = source.stat()
    if not stat.S_ISREG(info.st_mode):
        raise OSError(errno.EINVAL, f"not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A short fixed prefix: embedding the destination name could push a valid
    # 255-byte member name past the filesystem's component limit.
    handle, temporary = tempfile.mkstemp(prefix=".stage-", dir=destination.parent)
    temporary_path = Path(temporary)
    try:
        with source.open("rb") as stream:
            if not reflink_into(stream.fileno(), handle):
                with os.fdopen(os.dup(handle), "wb") as target:
                    shutil.copyfileobj(stream, target)
        os.fsync(handle)
        os.close(handle)
        handle = -1
        # Mode and timestamps only: copystat would also copy BSD/macOS file
        # flags, and an immutable temporary could be neither renamed into
        # place nor cleaned up.
        shutil.copymode(source, temporary_path)
        os.utime(temporary_path, ns=(info.st_atime_ns, info.st_mtime_ns))
        os.replace(temporary_path, destination)
    except BaseException:
        if handle >= 0:
            os.close(handle)
        temporary_path.unlink(missing_ok=True)
        raise
