"""Descriptor-anchored access to the archive-owned ``audit.db`` leaf."""

from __future__ import annotations

import os
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class AuditLeafError(RuntimeError):
    """The audit pathname cannot prove it is one archive-owned database file."""


@dataclass(frozen=True, slots=True)
class _AuditLeafIdentity:
    device: int
    inode: int


class VerifiedAuditLeaf:
    """Keep one archive directory descriptor and verify its ``audit.db`` leaf.

    SQLite accepts a URI below ``/proc/self/fd/<directory-fd>``.  That binds
    all database and sidecar opens to the directory we inspected, rather than
    re-resolving the caller's mutable pathname.  The leaf is checked before
    and immediately after SQLite opens it, so a replace between those steps
    is rejected before any caller receives a connection.
    """

    def __init__(self, archive_root: Path, *, filename: str = "audit.db") -> None:
        self._archive_root = archive_root
        self._filename = filename
        self._directory_fd: int | None = None
        self._identity: _AuditLeafIdentity | None = None

    def __enter__(self) -> VerifiedAuditLeaf:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        try:
            self._directory_fd = os.open(self._archive_root, directory_flags | nofollow)
            self._validate(self._lstat_leaf_metadata())
            metadata = self._open_leaf_metadata()
        except OSError as exc:
            self.close()
            raise AuditLeafError(f"cannot safely open audit tier leaf: {self._archive_root / self._filename}") from exc
        self._identity = self._validate(metadata)
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    @property
    def sqlite_uri(self) -> str:
        return f"{self.anchored_path.as_uri()}?mode=rw"

    @property
    def anchored_path(self) -> Path:
        """Return the descriptor-anchored path SQLite and byte readers may open."""

        if self._directory_fd is None:
            raise RuntimeError("audit leaf descriptor is closed")
        return Path(f"/proc/self/fd/{self._directory_fd}/{self._filename}")

    def assert_unchanged(self) -> None:
        """Require the current directory entry to retain the inspected inode."""

        if self._identity is None:
            raise RuntimeError("audit leaf descriptor is closed")
        try:
            current = self._validate(self._open_leaf_metadata())
        except OSError as exc:
            raise AuditLeafError(f"cannot revalidate audit tier leaf: {self._archive_root / self._filename}") from exc
        if current != self._identity:
            raise AuditLeafError(f"audit tier leaf changed during SQLite open: {self._archive_root / self._filename}")

    def close(self) -> None:
        if self._directory_fd is not None:
            os.close(self._directory_fd)
            self._directory_fd = None
        self._identity = None

    def _open_leaf_metadata(self) -> os.stat_result:
        if self._directory_fd is None:
            raise RuntimeError("audit leaf descriptor is closed")
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self._filename, flags, dir_fd=self._directory_fd)
        try:
            return os.fstat(descriptor)
        finally:
            os.close(descriptor)

    def _lstat_leaf_metadata(self) -> os.stat_result:
        if self._directory_fd is None:
            raise RuntimeError("audit leaf descriptor is closed")
        return os.stat(self._filename, dir_fd=self._directory_fd, follow_symlinks=False)

    def _validate(self, metadata: os.stat_result) -> _AuditLeafIdentity:
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise AuditLeafError(
                f"audit tier must be an archive-owned regular file with one link: {self._archive_root / self._filename}"
            )
        return _AuditLeafIdentity(metadata.st_dev, metadata.st_ino)


@contextmanager
def open_verified_audit_connection(path: Path) -> Iterator[sqlite3.Connection]:
    """Open one writable audit connection pinned to an owned leaf descriptor."""

    with VerifiedAuditLeaf(path.parent, filename=path.name) as leaf:
        connection = sqlite3.connect(leaf.sqlite_uri, uri=True)
        try:
            leaf.assert_unchanged()
            yield connection
        finally:
            connection.close()


def assert_verified_audit_leaf(path: Path) -> None:
    """Check an existing audit leaf without exposing its descriptor to callers."""

    with VerifiedAuditLeaf(path.parent, filename=path.name):
        return


__all__ = ["AuditLeafError", "VerifiedAuditLeaf", "assert_verified_audit_leaf", "open_verified_audit_connection"]
