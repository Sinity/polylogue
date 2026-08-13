"""Descriptor-anchored access to the archive-owned ``audit.db`` leaf."""

from __future__ import annotations

import fcntl
import os
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


class AuditLeafError(RuntimeError):
    """The audit pathname cannot prove it is one archive-owned database file."""


@dataclass(frozen=True, slots=True)
class _AuditLeafIdentity:
    device: int
    inode: int


class VerifiedAuditLeaf:
    """Keep one archive directory descriptor and verify its ``audit.db`` leaf.

    A writer holds the verified main leaf while SQLite opens a child path that
    is proven to resolve back to that descriptor's directory. The main leaf
    and any SQLite sidecar are checked before and after opening, so a
    replacement or redirected sidecar is rejected before a caller receives a
    connection.
    """

    def __init__(self, archive_root: Path, *, filename: str = "audit.db", lock_writer: bool = False) -> None:
        self._archive_root = archive_root
        self._filename = filename
        self._lock_writer = lock_writer
        self._directory_fd: int | None = None
        self._leaf_fd: int | None = None
        self._directory_identity: _AuditLeafIdentity | None = None
        self._identity: _AuditLeafIdentity | None = None
        self._anchored_path: Path | None = None
        self._writer_lock_held = False

    def __enter__(self) -> VerifiedAuditLeaf:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        try:
            self._directory_fd = os.open(self._archive_root, directory_flags | nofollow)
            directory_metadata = os.fstat(self._directory_fd)
            self._validate_directory(directory_metadata)
            self._directory_identity = _AuditLeafIdentity(directory_metadata.st_dev, directory_metadata.st_ino)
            expected = self._validate(self._lstat_leaf_metadata())
            self._leaf_fd = self._open_leaf()
            metadata = os.fstat(self._leaf_fd)
            self._identity = self._validate(metadata)
            if self._identity != expected:
                raise AuditLeafError(f"audit tier leaf changed while opening: {self._archive_root / self._filename}")
            if self._lock_writer:
                self._acquire_writer_lock()
            self._anchored_path = self._resolve_portable_child_path()
            self._assert_sidecar_namespace()
        except BaseException as exc:
            self._close_after_failed_enter()
            if isinstance(exc, AuditLeafError):
                raise
            raise AuditLeafError(f"cannot safely open audit tier leaf: {self._archive_root / self._filename}") from exc
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    @property
    def sqlite_uri(self) -> str:
        return f"{self.anchored_path.as_uri()}?mode=rw"

    @property
    def anchored_path(self) -> Path:
        """Return the descriptor-anchored path SQLite and byte readers may open."""

        if self._anchored_path is None:
            raise RuntimeError("audit leaf descriptor is closed")
        return self._anchored_path

    def assert_unchanged(self) -> None:
        """Require the current directory entry to retain the inspected inode."""

        if self._identity is None or self._directory_identity is None:
            raise RuntimeError("audit leaf descriptor is closed")
        try:
            current = self._validate(self._open_leaf_metadata())
            anchored = self._stat_path(self.anchored_path)
            anchored_directory = self._stat_path(self.anchored_path.parent)
            self._assert_sidecar_namespace()
        except OSError as exc:
            raise AuditLeafError(f"cannot revalidate audit tier leaf: {self._archive_root / self._filename}") from exc
        if (
            current != self._identity
            or _AuditLeafIdentity(anchored.st_dev, anchored.st_ino) != self._identity
            or _AuditLeafIdentity(anchored_directory.st_dev, anchored_directory.st_ino) != self._directory_identity
        ):
            raise AuditLeafError(f"audit tier leaf changed during SQLite open: {self._archive_root / self._filename}")

    def close(self) -> None:
        directory_fd, leaf_fd = self._directory_fd, self._leaf_fd
        writer_lock_held = self._writer_lock_held
        self._directory_fd = None
        self._leaf_fd = None
        self._directory_identity = None
        self._identity = None
        self._anchored_path = None
        self._writer_lock_held = False
        errors: list[OSError] = []
        if leaf_fd is not None:
            if writer_lock_held:
                try:
                    fcntl.flock(leaf_fd, fcntl.LOCK_UN)
                except OSError as exc:
                    errors.append(exc)
            try:
                os.close(leaf_fd)
            except OSError as exc:
                errors.append(exc)
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def _close_after_failed_enter(self) -> None:
        try:
            self.close()
        except OSError:
            return

    def _acquire_writer_lock(self) -> None:
        if self._leaf_fd is None:
            raise RuntimeError("audit leaf descriptor is closed")
        try:
            fcntl.flock(self._leaf_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise AuditLeafError(
                f"audit tier already has an active writer: {self._archive_root / self._filename}"
            ) from exc
        self._writer_lock_held = True

    def _open_leaf(self) -> int:
        if self._directory_fd is None:
            raise RuntimeError("audit leaf descriptor is closed")
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        return os.open(self._filename, flags, dir_fd=self._directory_fd)

    def _open_leaf_metadata(self) -> os.stat_result:
        descriptor = self._open_leaf()
        try:
            return os.fstat(descriptor)
        finally:
            os.close(descriptor)

    def _lstat_leaf_metadata(self) -> os.stat_result:
        if self._directory_fd is None:
            raise RuntimeError("audit leaf descriptor is closed")
        return os.stat(self._filename, dir_fd=self._directory_fd, follow_symlinks=False)

    def _resolve_portable_child_path(self) -> Path:
        if self._directory_fd is None or self._identity is None:
            raise RuntimeError("audit leaf descriptor is closed")
        directory = self._native_directory_path()
        if directory is not None:
            candidate = directory / self._filename
            if self._matches_identity(candidate, self._identity):
                return candidate
        descriptor_child = self._descriptor_child_path()
        if descriptor_child is not None:
            return descriptor_child
        raise AuditLeafError(f"cannot access audit tier through a verified descriptor: {self._archive_root}")

    def _descriptor_child_path(self) -> Path | None:
        """Return a descriptor-directory child only where the host proves it works."""

        if self._directory_fd is None or self._identity is None:
            raise RuntimeError("audit leaf descriptor is closed")
        for directory in (Path("/proc/self/fd"), Path("/dev/fd")):
            candidate = directory / str(self._directory_fd) / self._filename
            if self._matches_identity(candidate, self._identity):
                return candidate
        return None

    def _native_directory_path(self) -> Path | None:
        if self._directory_fd is None:
            raise RuntimeError("audit leaf descriptor is closed")
        request = getattr(fcntl, "F_GETPATH", None)
        if not isinstance(request, int):
            return None
        try:
            raw = fcntl.fcntl(self._directory_fd, request, b"\0" * 1024)
        except OSError:
            return None
        if not isinstance(raw, bytes):
            return None
        encoded = raw.split(b"\0", 1)[0]
        if not encoded:
            return None
        try:
            candidate = Path(os.fsdecode(encoded))
            directory = os.fstat(self._directory_fd)
            metadata = self._stat_path(candidate)
        except OSError:
            return None
        if (metadata.st_dev, metadata.st_ino) != (directory.st_dev, directory.st_ino):
            return None
        return candidate

    def _assert_sidecar_namespace(self) -> None:
        if self._directory_fd is None:
            raise RuntimeError("audit leaf descriptor is closed")
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            filename = f"{self._filename}{suffix}"
            try:
                expected = self._validate(
                    os.stat(filename, dir_fd=self._directory_fd, follow_symlinks=False),
                    description="audit tier sidecar",
                    filename=filename,
                )
            except FileNotFoundError:
                continue
            descriptor = os.open(
                filename,
                os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=self._directory_fd,
            )
            try:
                actual = self._validate(os.fstat(descriptor), description="audit tier sidecar", filename=filename)
            finally:
                os.close(descriptor)
            if actual != expected:
                raise AuditLeafError(f"audit tier sidecar changed while opening: {self._archive_root / filename}")

    @staticmethod
    def _stat_path(path: Path) -> os.stat_result:
        return os.stat(path)

    def _matches_identity(self, path: Path, identity: _AuditLeafIdentity) -> bool:
        try:
            metadata = self._stat_path(path)
        except OSError:
            return False
        return (metadata.st_dev, metadata.st_ino) == (identity.device, identity.inode)

    def _validate(
        self,
        metadata: os.stat_result,
        *,
        description: str = "audit tier",
        filename: str | None = None,
    ) -> _AuditLeafIdentity:
        path = self._archive_root / (filename or self._filename)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise AuditLeafError(f"{description} must be an archive-owned regular file with one link: {path}")
        if metadata.st_uid != os.geteuid():
            raise AuditLeafError(f"{description} must be owned by the current effective user: {path}")
        if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise AuditLeafError(f"{description} must not be writable by group or other: {path}")
        return _AuditLeafIdentity(metadata.st_dev, metadata.st_ino)

    def _validate_directory(self, metadata: os.stat_result) -> None:
        if metadata.st_uid != os.geteuid():
            raise AuditLeafError(
                f"audit tier directory must be owned by the current effective user: {self._archive_root}"
            )
        if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise AuditLeafError(f"audit tier directory must not be writable by group or other: {self._archive_root}")


@contextmanager
def open_verified_audit_connection(path: Path) -> Iterator[sqlite3.Connection]:
    """Open one writable audit connection pinned to an owned leaf descriptor."""

    with VerifiedAuditLeaf(path.parent, filename=path.name, lock_writer=True) as leaf:
        connection = sqlite3.connect(leaf.sqlite_uri, uri=True)
        try:
            leaf.assert_unchanged()
            yield connection
        finally:
            connection.close()
            leaf.assert_unchanged()


def assert_verified_audit_leaf(path: Path) -> None:
    """Check an existing audit leaf without exposing its descriptor to callers."""

    with VerifiedAuditLeaf(path.parent, filename=path.name):
        return


__all__ = ["AuditLeafError", "VerifiedAuditLeaf", "assert_verified_audit_leaf", "open_verified_audit_connection"]
