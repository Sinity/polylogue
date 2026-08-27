"""Shared clone primitive for offline devtools actuators."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path

_SQLITE_SIDECARS = ("-wal", "-shm", "-journal")


class CloneSupportError(RuntimeError):
    """Raised when a source file cannot be safely cloned."""


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_no_sidecars(path: Path) -> None:
    present = [str(Path(f"{path}{suffix}")) for suffix in _SQLITE_SIDECARS if Path(f"{path}{suffix}").exists()]
    if present:
        raise CloneSupportError(f"SQLite sidecars make clone proof ambiguous: {', '.join(present)}")


def _digest(path: Path) -> tuple[int, str]:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise CloneSupportError(f"clone source must be a regular single-linked file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return metadata.st_size, digest.hexdigest()


def reflink_clone(source: Path, destination: Path) -> None:
    """Create a new clone without accepting SQLite sidecars or overwrites."""
    source = Path(source)
    source_size, source_digest = _digest(source)
    if destination.is_symlink() or destination.exists():
        raise CloneSupportError(f"clone destination already exists: {destination}")
    _require_no_sidecars(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # cp's reflink is a copy optimization, never a correctness dependency.
    try:
        subprocess.run(
            ["cp", "--reflink=auto", "--preserve=mode,timestamps", str(source), str(destination)], check=True
        )
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise CloneSupportError(f"could not clone {source}: {exc}") from exc
    try:
        clone_size, clone_digest = _digest(destination)
        if (clone_size, clone_digest) != (source_size, source_digest):
            raise CloneSupportError(f"clone content changed during publication: {destination}")
        source_identity = os.stat(source, follow_symlinks=False)
        clone_identity = os.stat(destination, follow_symlinks=False)
        if (source_identity.st_dev, source_identity.st_ino) == (clone_identity.st_dev, clone_identity.st_ino):
            raise CloneSupportError(f"clone shares the source inode: {destination}")
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    _fsync_directory(destination.parent)


__all__ = ["CloneSupportError", "reflink_clone"]
