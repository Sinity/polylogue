"""Locked exact-byte staging shared by acquired request bodies."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

BODY_READ_CHUNK_BYTES = 1024 * 1024
STAGING_DIRNAME = ".staging"


class BodyIncompleteError(ValueError):
    """The request ended before its declared ``Content-Length``."""


class BodyStorageExhaustedError(RuntimeError):
    """The spool filesystem cannot hold an incoming body.

    Raised before any body byte is written: the declared length is reserved on
    disk first, so the only refusal is the physical one, and it is retryable.
    """

    def __init__(self, requested_bytes: int, available_bytes: int | None) -> None:
        super().__init__(f"spool storage cannot hold {requested_bytes} bytes (available: {available_bytes})")
        self.requested_bytes = requested_bytes
        self.available_bytes = available_bytes


class StagedBody:
    """A received body on disk, with the digest of its exact bytes.

    The staging file stays ``flock``-ed until :meth:`discard`, so a receiver
    starting beside this one (:func:`reap_stale_staging`) can tell a live
    upload from one a crashed process abandoned.
    """

    __slots__ = ("_lock_fd", "path", "sha256", "size_bytes", "adopted")

    def __init__(self, path: Path, size_bytes: int, sha256: str, lock_fd: int | None = None) -> None:
        self.path = path
        self.size_bytes = size_bytes
        self.sha256 = sha256
        self._lock_fd = lock_fd
        self.adopted = False

    def discard(self) -> None:
        self.path.unlink(missing_ok=True)
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None


def _available_bytes(directory: Path) -> int:
    stats = os.statvfs(directory)
    return stats.f_bavail * stats.f_frsize


def _reserve(fd: int, directory: Path, length: int) -> None:
    """Reserve ``length`` bytes for the staging file before writing any.

    ``posix_fallocate`` allocates the blocks, so concurrent uploads -- in this
    process or another sharing the spool -- cannot both be admitted into the
    same free space. Where the filesystem cannot allocate, free space is
    compared instead. A length no file offset can represent, or one past the
    filesystem's largest file, is the same physical refusal.
    """
    if length <= 0:
        return
    fallocate = getattr(os, "posix_fallocate", None)
    if fallocate is not None:
        try:
            fallocate(fd, 0, length)
            return
        except OverflowError as exc:
            raise BodyStorageExhaustedError(length, _available_bytes(directory)) from exc
        except OSError as exc:
            if is_storage_exhausted(exc):
                raise BodyStorageExhaustedError(length, _available_bytes(directory)) from exc
            if exc.errno not in {errno.EOPNOTSUPP, errno.EINVAL, errno.ENOSYS}:
                raise
    available = _available_bytes(directory)
    if length > available:
        raise BodyStorageExhaustedError(length, available)


def _locked_staging_file(staging: Path) -> tuple[int, Path]:
    """Create and lock a staging file that no concurrent reaper has removed."""
    while True:
        fd, name = tempfile.mkstemp(dir=staging, prefix=_STAGING_PREFIX, suffix=_STAGING_SUFFIX)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            if os.stat(name).st_ino == os.fstat(fd).st_ino:
                return fd, Path(name)
        except FileNotFoundError:
            pass
        os.close(fd)


_STAGING_PREFIX = ".capture-"
_STAGING_SUFFIX = ".tmp"


def stage_body(read: Callable[[int], bytes], length: int, *, spool_root: Path) -> StagedBody:
    """Copy exactly ``length`` body bytes into a staging file in the spool.

    The declared length is reserved on disk before the first read, so a body
    the filesystem cannot hold is refused with
    :class:`BodyStorageExhaustedError` without consuming space. Reads at
    most :data:`BODY_READ_CHUNK_BYTES` per call and hashes while writing,
    so no body is held in memory. The staged file is fsynced and stays locked
    until the caller publishes it by ``os.replace`` or discards it.
    """

    def chunks() -> Iterator[bytes]:
        remaining = length
        while remaining > 0:
            chunk = read(min(BODY_READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise BodyIncompleteError(f"request body ended {remaining} bytes before its declared length")
            remaining -= len(chunk)
            yield chunk

    return stage_body_chunks(chunks(), spool_root=spool_root, durable=True, reserved_length=length)


def stage_body_chunks(
    chunks: Iterator[bytes], *, spool_root: Path, durable: bool = True, reserved_length: int | None = None
) -> StagedBody:
    """Seal generated artifact bytes through the same locked staging owner.

    A received File reserves its known length before reading. A generated
    prefix/final artifact has no declared length: actual filesystem exhaustion
    remains a typed refusal, and no arbitrary body limit is substituted.
    Publication bytes are fsynced before their artifact owner adopts them.
    Transient JSON cells and responses are flushed for their immediate reader;
    their durable authority is the committed registry, not this scratch file.
    The caller's chunk/token allocation remains its own memory contract.
    """
    staging = spool_root / STAGING_DIRNAME
    staging.mkdir(parents=True, exist_ok=True)
    fd, path = _locked_staging_file(staging)
    digest = hashlib.sha256()
    size = 0
    try:
        if reserved_length is not None:
            _reserve(fd, staging, reserved_length)
        with os.fdopen(os.dup(fd), "wb") as handle:
            for chunk in chunks:
                size += len(chunk)
                handle.write(chunk)
                digest.update(chunk)
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
    except BaseException as exc:
        path.unlink(missing_ok=True)
        os.close(fd)
        if isinstance(exc, OSError) and is_storage_exhausted(exc):
            raise BodyStorageExhaustedError(size, None) from exc
        raise
    return StagedBody(path=path, size_bytes=size, sha256=digest.hexdigest(), lock_fd=fd)


def reap_stale_staging(spool_root: Path) -> int:
    """Remove staging files no live upload holds; return how many.

    A receiver that died mid-upload leaves its staging file behind, and it is
    invisible to the spool quota. Every live upload holds its file's lock, so
    a file whose lock can be taken belongs to no one.
    """
    staging = spool_root / STAGING_DIRNAME
    if not staging.is_dir():
        return 0
    reaped = 0
    for path in staging.glob(f"{_STAGING_PREFIX}*{_STAGING_SUFFIX}"):
        try:
            fd = os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            path.unlink(missing_ok=True)
            reaped += 1
        finally:
            os.close(fd)
    return reaped


def is_storage_exhausted(exc: OSError) -> bool:
    """Whether a staging failure is the spool's physical limit, not a fault.

    Space or quota exhaustion, or a body past the filesystem's largest file.
    """
    return exc.errno in {errno.ENOSPC, errno.EFBIG, getattr(errno, "EDQUOT", errno.ENOSPC)}
