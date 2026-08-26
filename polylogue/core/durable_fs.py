"""Small, typed primitives for durable filesystem publication."""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path


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
