"""Small, typed primitives for durable filesystem publication."""

from __future__ import annotations

import errno
import fcntl
import os
import shutil
import stat
import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO

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


def sync_directory_ancestors(path: Path) -> None:
    """Persist a directory and its reachability, including newly made ancestors."""
    path = path.absolute()
    while True:
        _fsync_directory(path)
        if path.parent == path:
            return
        path = path.parent


def sync_tree(root: Path) -> None:
    """Persist an exclusively owned regular file tree before publishing authority.

    Entries stream through open directory iterators; files precede their
    containing directories and ancestors. Symlinks and special files refuse
    publication rather than syncing a different object's bytes.
    """
    root = root.absolute()
    pending = []
    try:
        if not stat.S_ISDIR(root.lstat().st_mode):
            raise DurableFilesystemError(f"publication root is not a real directory: {root}")
        pending.append((root, os.scandir(root)))
        while pending:
            directory, entries = pending[-1]
            entry = next(entries, None)
            if entry is None:
                entries.close()
                pending.pop()
                _fsync_directory(directory)
                continue
            path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                pending.append((path, os.scandir(path)))
                continue
            descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise DurableFilesystemError(f"publication artifact is not a regular file: {path}")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        sync_directory_ancestors(root.parent)
    except OSError as exc:
        raise DurableFilesystemError(f"cannot durably publish tree: {root}") from exc
    finally:
        for _, entries in pending:
            entries.close()


def write_once(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    """Create ``path`` exactly once, persisting its bytes and directory entry."""
    created: list[Path] = []
    missing: list[Path] = []
    parent = path.parent
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        created = list(reversed(missing))
        for directory in created:
            _fsync_directory(directory.parent)
        with path.open("xb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except OSError as exc:
        if "stream" in locals() and path.exists():
            with suppress(OSError):
                path.unlink()
                _fsync_directory(path.parent)
        raise DurableFilesystemError(f"cannot durably create: {path}") from exc


def atomic_replace(path: Path, payload: bytes, *, mode: int | None = None) -> None:
    """Durably write bytes to a temporary file and replace ``path``."""
    _atomic_publish(path, payload, mode=mode, replace_existing=True)


def atomic_create(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    """Publish complete durable bytes without replacing any existing path."""
    _atomic_publish(path, payload, mode=mode, replace_existing=False)


def _atomic_publish(path: Path, payload: bytes, *, mode: int | None, replace_existing: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = -1
    temporary_path: Path | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=".publish-", suffix=".tmp", dir=path.parent)
        temporary_path = Path(temporary)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            if mode is not None:
                os.fchmod(descriptor, mode)
            stream.write(payload)
            stream.flush()
            os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if replace_existing:
            os.replace(temporary_path, path)
        else:
            os.link(temporary_path, path)
        _fsync_directory(path.parent)
    except OSError as exc:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        raise DurableFilesystemError(f"cannot durably publish: {path}") from exc
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


__all__ = [
    "DurableFilesystemError",
    "append_line",
    "atomic_create",
    "atomic_replace",
    "sync_directory",
    "sync_directory_ancestors",
    "sync_tree",
    "write_once",
]


def reflink_into(source_fd: int, destination_fd: int) -> bool:
    """Clone ``source_fd``'s contents into ``destination_fd``; ``False`` when unsupported."""
    try:
        fcntl.ioctl(destination_fd, _FICLONE, source_fd)
    except OSError as exc:
        if exc.errno in _REFLINK_UNSUPPORTED:
            return False
        raise
    return True


def clone_or_copy_replace(
    source: BinaryIO, destination: Path, *, before_publish: Callable[[int], None] | None = None
) -> None:
    """Place a copy of the already accepted regular descriptor at ``destination``.

    The bytes are cloned by reflink where the filesystem supports it and
    copied otherwise, into a temporary sibling that replaces ``destination``
    only once complete: a failure leaves any earlier ``destination`` intact.
    ``before_publish`` receives the copied temporary's still-open descriptor
    after its file barrier and directly before rename. The acquisition owner
    can bind publication metadata to that exact candidate. The caller owns
    the source descriptor's lifetime and its namespace evidence.
    """
    info = os.fstat(source.fileno())
    if not stat.S_ISREG(info.st_mode):
        raise OSError(errno.EINVAL, f"not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A short fixed prefix: embedding the destination name could push a valid
    # 255-byte member name past the filesystem's component limit.
    handle, temporary = tempfile.mkstemp(prefix=".stage-", dir=destination.parent)
    temporary_path = Path(temporary)
    try:
        if not reflink_into(source.fileno(), handle):
            with os.fdopen(os.dup(handle), "wb") as target:
                shutil.copyfileobj(source, target)
        # Mode and timestamps only, and before the barrier so they are as
        # durable as the bytes: copystat would also copy BSD/macOS file
        # flags, and an immutable temporary could be neither renamed into
        # place nor cleaned up.
        os.fchmod(handle, stat.S_IMODE(info.st_mode))
        os.utime(handle, ns=(info.st_atime_ns, info.st_mtime_ns))
        os.fsync(handle)
        if before_publish is not None:
            before_publish(handle)
        os.close(handle)
        handle = -1
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if handle >= 0:
            os.close(handle)
        temporary_path.unlink(missing_ok=True)
        raise
