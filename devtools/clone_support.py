"""Shared clone primitive for offline devtools actuators."""

from __future__ import annotations

import os
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


def reflink_clone(source: Path, destination: Path) -> None:
    """Create a new clone without accepting SQLite sidecars or overwrites."""
    source = source.resolve(strict=True)
    if destination.exists():
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
    _fsync_directory(destination.parent)


__all__ = ["CloneSupportError", "reflink_clone"]
