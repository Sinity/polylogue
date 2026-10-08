"""Seals for closed private files passed from preparation to publication."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.compute_cancel import check_compute_cancelled


class VerificationCancelledError(Exception):
    """A digest pass stopped at a chunk boundary because its caller was cancelled."""


def file_digest(path: Path, *, stop: Callable[[], bool] | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            check_compute_cancelled()
            if stop is not None and stop():
                raise VerificationCancelledError(str(path))
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedFileSeal:
    """Closed scratch-file bytes and the exact inode handed to publication."""

    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def capture(cls, path: Path) -> PreparedFileSeal:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("prepared file must be a regular private file")
            if stat.S_IMODE(os.fstat(fd).st_mode) != 0o400:
                os.fchmod(fd, 0o400)
            before = os.fstat(fd)
            digest = hashlib.sha256()
            while chunk := os.read(fd, 1024 * 1024):
                check_compute_cancelled()
                digest.update(chunk)
            after = os.fstat(fd)
            if file_identity(before) != file_identity(after) or file_identity(path.lstat()) != file_identity(after):
                raise ValueError(f"prepared file changed while sealing: {path}")
            return cls(digest.hexdigest(), *file_identity(after))
        finally:
            os.close(fd)

    def verify(self, path: Path, *, full: bool, stop: Callable[[], bool] | None = None) -> None:
        before = path.lstat()
        if file_identity(before) != self.identity:
            raise ValueError(f"prepared file identity changed: {path}")
        if full:
            if file_digest(path, stop=stop) != self.sha256:
                raise ValueError(f"prepared file content changed: {path}")
            after = path.lstat()
            if file_identity(after) != self.identity:
                raise ValueError(f"prepared file changed during verification: {path}")

    @property
    def identity(self) -> tuple[int, int, int, int, int]:
        return self.device, self.inode, self.size, self.mtime_ns, self.ctime_ns


def file_identity(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
