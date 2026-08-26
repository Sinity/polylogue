"""Small, typed primitives for durable filesystem publication."""

from __future__ import annotations

import os
import uuid
from contextlib import suppress
from pathlib import Path


class DurableFilesystemError(OSError):
    """A durable filesystem operation could not complete its barriers."""


def sync_directory(path: Path) -> None:
    """Persist directory entries in ``path``."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise DurableFilesystemError(f"cannot fsync directory: {path}") from exc
    finally:
        os.close(descriptor)


def write_once(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    """Create ``path`` exactly once, persisting its bytes and directory entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            mode,
        )
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        sync_directory(path.parent)
    except OSError as exc:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        raise DurableFilesystemError(f"cannot durably create: {path}") from exc


def atomic_replace(path: Path, payload: bytes, *, mode: int | None = None) -> None:
    """Durably write bytes to a temporary file and replace ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
        sync_directory(path.parent)
    except OSError as exc:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        raise DurableFilesystemError(f"cannot durably replace: {path}") from exc
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


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
        sync_directory(path.parent)
    except OSError as exc:
        raise DurableFilesystemError(f"cannot durably append: {path}") from exc


__all__ = ["DurableFilesystemError", "append_line", "atomic_replace", "sync_directory", "write_once"]
