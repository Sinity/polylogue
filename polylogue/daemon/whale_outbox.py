"""Durable publication outbox for whale lifecycle receipts.

The ops database is telemetry and can be temporarily unavailable (or absent
while the daemon is starting). A receipt is therefore first committed to an
archive-local filesystem outbox. Delivery is idempotent and removal happens
only after the event ledger accepts the record.

The outbox is a descriptor-pinned filesystem boundary. Every archive and
outbox path component is opened with ``O_NOFOLLOW`` and retained descriptors
are used for all later operations. Receipt identity is cross-validated against
both its filename and its body before a record can be delivered or removed.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from polylogue.logging import get_logger
from polylogue.paths import archive_root

_RECOVERY_MARKER = ".json.recovery."
_RECOVERY_HEX_LENGTH = 32
_RENAME_NOREPLACE = 1
_RENAMEAT2_SYSCALLS = {"x86_64": 316, "aarch64": 276}
_LIBC = ctypes.CDLL(None, use_errno=True)

logger = get_logger(__name__)


def _rename_noreplace(src: str, dst: str, *, directory_fd: int) -> None:
    """Atomically move two names without replacing an existing destination."""
    syscall_number = _RENAMEAT2_SYSCALLS.get(os.uname().machine)
    if syscall_number is None:
        raise NotImplementedError("renameat2 is unavailable on this platform")
    result = _LIBC.syscall(
        ctypes.c_long(syscall_number),
        ctypes.c_int(directory_fd),
        ctypes.c_char_p(os.fsencode(src)),
        ctypes.c_int(directory_fd),
        ctypes.c_char_p(os.fsencode(dst)),
        ctypes.c_uint(_RENAME_NOREPLACE),
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), dst)
    raise OSError(error_number, os.strerror(error_number), src)


def _preserve_recovery_quarantine(directory_fd: int, quarantine: str, key: str) -> None:
    """Move quarantine to a unique recovery name without replacement races."""
    for _attempt in range(8):
        recovery_name = _recovery_name(key)
        try:
            _rename_noreplace(quarantine, recovery_name, directory_fd=directory_fd)
        except FileExistsError:
            continue
        except NotImplementedError:
            # The original ``.ack`` name is a durable fallback namespace;
            # list_pending validates and drains it on platforms without
            # renameat2 rather than stranding the receipt.
            return
        return
    # Forced allocation exhaustion also leaves the validated .ack fallback in
    # place. It remains visible to list_pending and startup recovery.


_OUTBOX_DIRNAME = "whale-receipt-outbox"
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC


def _simple_name(value: str, *, label: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{label} must be one path component")
    return value


def _validate_idempotency_key(value: str) -> str:
    name = _simple_name(value, label="idempotency key")
    if len(name) > 200 or any(character in "/\\" or ord(character) < 0x20 for character in name):
        raise ValueError("idempotency key contains unsafe characters")
    return name


def _validate_identity(value: object, *, label: str) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 400:
        return None
    if any(character in "/\\" or ord(character) < 0x20 for character in value):
        return None
    return value


def _validate_directory(descriptor: int, *, label: str) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise OSError(f"{label} is not a directory")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise PermissionError(f"{label} permissions must exclude group and other")


def _open_directory_chain(path: Path) -> int:
    """Open every component of an existing path without following links."""
    parts = path.parts
    if not parts:
        raise ValueError("directory path must not be empty")
    if path.is_absolute():
        descriptor = os.open(os.sep, _DIRECTORY_FLAGS)
        components = parts[1:]
    else:
        descriptor = os.open(".", _DIRECTORY_FLAGS)
        components = parts
    try:
        for component in components:
            _simple_name(component, label="directory path component")
            child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            try:
                if not stat.S_ISDIR(os.fstat(child).st_mode):
                    raise OSError(f"directory component {component} is not a directory")
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_directory(path: Path | str, *, label: str, parent_fd: int | None = None) -> int:
    if parent_fd is None:
        try:
            descriptor = _open_directory_chain(Path(path))
        except OSError as exc:
            raise OSError(f"cannot pin {label} without following links: {path}") from exc
        return descriptor
    name = _simple_name(os.fspath(path), label=f"{label} name")
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise OSError(f"cannot pin {label} without following links: {name}") from exc
    try:
        _validate_directory(descriptor, label=label)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


@contextmanager
def _outbox_directory(root: Path, *, create: bool) -> Iterator[tuple[int, Path] | None]:
    """Yield a pinned, private outbox child of a pinned archive root."""
    root_fd = _open_directory(root, label="archive root")
    outbox_fd = -1
    outbox_path = root / _OUTBOX_DIRNAME
    try:
        try:
            outbox_fd = _open_directory(_OUTBOX_DIRNAME, label="whale receipt outbox", parent_fd=root_fd)
        except OSError as exc:
            if not create and isinstance(exc.__cause__, FileNotFoundError):
                yield None
                return
            if not create or not isinstance(exc.__cause__, FileNotFoundError):
                raise
            with suppress(FileExistsError):
                os.mkdir(_OUTBOX_DIRNAME, 0o700, dir_fd=root_fd)
            outbox_fd = _open_directory(_OUTBOX_DIRNAME, label="whale receipt outbox", parent_fd=root_fd)
            os.fsync(root_fd)
        yield outbox_fd, outbox_path
    finally:
        if outbox_fd >= 0:
            os.close(outbox_fd)
        os.close(root_fd)


def _record_name_matches_key(name: str, key: str) -> bool:
    """Accept canonical names and explicit, recoverable quarantine names."""
    if name == f"{key}.json":
        return True
    prefix = f"{key}{_RECOVERY_MARKER}"
    if name.startswith(prefix) and name.endswith(".json"):
        token = name[len(prefix) : -len(".json")]
        return len(token) == _RECOVERY_HEX_LENGTH and all(character in "0123456789abcdef" for character in token)
    fallback_prefix = f".{key}.json."
    if not name.startswith(fallback_prefix) or not name.endswith(".ack"):
        return False
    fallback_token = name[len(fallback_prefix) : -len(".ack")]
    return len(fallback_token) == _RECOVERY_HEX_LENGTH and all(
        character in "0123456789abcdef" for character in fallback_token
    )


def _recovery_name(key: str) -> str:
    return f"{key}.json.recovery.{uuid.uuid4().hex}.json"


def _validated_record(name: str, value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not (name.endswith(".json") or name.endswith(".ack")):
        return None
    key = value.get("idempotency_key")
    try:
        safe_key = _validate_idempotency_key(key) if isinstance(key, str) else None
    except ValueError:
        return None
    if safe_key is None or not _record_name_matches_key(name, safe_key):
        return None
    if _validate_identity(value.get("kind"), label="kind") is None:
        return None
    if _validate_identity(value.get("operation_id"), label="operation id") is None:
        return None
    if not isinstance(value.get("payload"), dict):
        return None
    return value


def _enqueue_impl(
    *,
    kind: str,
    idempotency_key: str,
    operation_id: str,
    payload: dict[str, object],
    root: Path | None = None,
) -> tuple[Path, tuple[int, int]]:
    """Atomically persist one receipt and return its pinned identity."""
    key = _validate_idempotency_key(idempotency_key)
    if _validate_identity(kind, label="kind") is None:
        raise ValueError("whale receipt outbox requires a valid kind")
    if _validate_identity(operation_id, label="operation id") is None:
        raise ValueError("whale receipt outbox requires a valid operation id")
    archive = archive_root() if root is None else root
    with _outbox_directory(archive, create=True) as state:
        assert state is not None
        directory_fd, directory = state
        record = {
            "kind": kind,
            "idempotency_key": key,
            "operation_id": operation_id,
            "payload": payload,
        }
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        temporary = f".{key}.{uuid.uuid4().hex}.tmp"
        descriptor = -1
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=directory_fd,
            )
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            target_name = f"{key}.json"
            try:
                os.link(
                    temporary,
                    target_name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                try:
                    existing_metadata = os.stat(target_name, dir_fd=directory_fd, follow_symlinks=False)
                except OSError as exc:
                    raise ValueError("whale receipt idempotency target changed during admission") from exc
                existing = _read_record(directory_fd, target_name, existing_metadata)
                if existing is None or any(
                    existing.get(field) != record.get(field)
                    for field in ("kind", "idempotency_key", "operation_id", "payload")
                ):
                    raise ValueError("whale receipt idempotency key conflicts with an existing receipt") from None
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=directory_fd)
            os.fsync(directory_fd)
            final_metadata = os.stat(target_name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(final_metadata.st_mode) or final_metadata.st_nlink < 1:
                raise OSError("whale receipt target is not a regular file")
            identity = (final_metadata.st_dev, final_metadata.st_ino)
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=directory_fd)
            raise
    return directory / f"{key}.json", identity


def enqueue(
    *,
    kind: str,
    idempotency_key: str,
    operation_id: str,
    payload: dict[str, object],
    root: Path | None = None,
) -> Path:
    """Atomically persist one receipt before attempting SQLite publication."""
    path, _identity = _enqueue_impl(
        kind=kind,
        idempotency_key=idempotency_key,
        operation_id=operation_id,
        payload=payload,
        root=root,
    )
    return path


def enqueue_with_identity(
    *,
    kind: str,
    idempotency_key: str,
    operation_id: str,
    payload: dict[str, object],
    root: Path | None = None,
) -> tuple[Path, tuple[int, int]]:
    """Persist one receipt and retain its inode for immediate acknowledgement."""
    return _enqueue_impl(
        kind=kind,
        idempotency_key=idempotency_key,
        operation_id=operation_id,
        payload=payload,
        root=root,
    )


def _read_record(directory_fd: int, name: str, expected: os.stat_result) -> dict[str, Any] | None:
    try:
        descriptor = os.open(name, _FILE_FLAGS, dir_fd=directory_fd)
    except OSError:
        return None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or (metadata.st_dev, metadata.st_ino) != (expected.st_dev, expected.st_ino)
        ):
            return None
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as stream:
            value = json.load(stream)
        value = _validated_record(name, value)
        if value is None:
            return None
        value["_name"] = name
        value["_identity"] = (metadata.st_dev, metadata.st_ino)
        return value
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    finally:
        os.close(descriptor)


def _list_pending_pinned(*, root: Path | None = None) -> list[dict[str, Any]]:
    """Read pending records through one pinned directory."""
    archive = archive_root() if root is None else root
    with _outbox_directory(archive, create=False) as state:
        if state is None:
            return []
        directory_fd, directory = state
        entries: list[tuple[str, os.stat_result]] = []
        try:
            with os.scandir(directory_fd) as scan:
                for entry in scan:
                    if not (entry.name.endswith(".json") or entry.name.endswith(".ack")):
                        continue
                    metadata = entry.stat(follow_symlinks=False)
                    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                        logger.error("whale receipt outbox contains unsafe record: %s", entry.name)
                        continue
                    if stat.S_IMODE(metadata.st_mode) & 0o077:
                        logger.error("whale receipt outbox contains world-readable record: %s", entry.name)
                        continue
                    entries.append((entry.name, metadata))
        except OSError:
            logger.error("cannot enumerate whale receipt outbox", exc_info=True)
            return []
        pending: list[dict[str, Any]] = []
        for name, metadata in sorted(entries):
            value = _read_record(directory_fd, name, metadata)
            if value is not None:
                value["_path"] = directory / name
                pending.append(value)
        return pending


def list_pending(*, root: Path | None = None) -> list[dict[str, Any]]:
    """Read pending records, failing closed if any path component is unsafe."""
    try:
        return _list_pending_pinned(root=root)
    except OSError:
        logger.error("cannot safely open whale receipt outbox", exc_info=True)
        return []


def acknowledge(record: dict[str, Any]) -> None:
    """Remove exactly the previously-read receipt, never a replacement."""
    path = record.get("_path")
    if not isinstance(path, Path) or path.name != record.get("_name"):
        return
    if path.parent.name != _OUTBOX_DIRNAME:
        return
    name = path.name
    try:
        if _validated_record(name, record) is None:
            return
    except (TypeError, ValueError):
        return
    archive = path.parent.parent
    try:
        with _outbox_directory(archive, create=False) as state:
            if state is None:
                return
            directory_fd, _directory = state
            descriptor = -1
            try:
                descriptor = os.open(name, _FILE_FLAGS, dir_fd=directory_fd)
                metadata = os.fstat(descriptor)
                expected_identity = record.get("_identity")
                if not (
                    isinstance(expected_identity, tuple)
                    and len(expected_identity) == 2
                    and all(isinstance(part, int) for part in expected_identity)
                    and stat.S_ISREG(metadata.st_mode)
                    and metadata.st_nlink == 1
                    and not stat.S_IMODE(metadata.st_mode) & 0o077
                    and (metadata.st_dev, metadata.st_ino) == expected_identity
                ):
                    return
                with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as stream:
                    actual = json.load(stream)
                if _validated_record(name, actual) is None:
                    return
                for field in ("kind", "idempotency_key", "operation_id", "payload"):
                    if actual.get(field) != record.get(field):
                        return
                # A final stat followed by unlink(name) is not an identity
                # check: another process can replace name in that interval.
                # Move the pathname to a private quarantine name atomically,
                # then inspect the moved inode.  A replacement that won the
                # race is never unlinked; it is restored (without replacing a
                # newer pathname) and acknowledgement fails closed.
                quarantine = f".{name}.{uuid.uuid4().hex}.ack"
                try:
                    os.rename(name, quarantine, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
                except FileNotFoundError:
                    return
                moved = os.stat(quarantine, dir_fd=directory_fd, follow_symlinks=False)
                moved_identity = (moved.st_dev, moved.st_ino)
                if moved_identity != (metadata.st_dev, metadata.st_ino):
                    key = _validate_idempotency_key(str(actual["idempotency_key"]))
                    try:
                        _rename_noreplace(quarantine, name, directory_fd=directory_fd)
                    except FileExistsError:
                        _preserve_recovery_quarantine(directory_fd, quarantine, key)
                    except NotImplementedError:
                        _preserve_recovery_quarantine(directory_fd, quarantine, key)
                    return
                os.unlink(quarantine, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except FileNotFoundError:
                return
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
    except (OSError, ValueError, TypeError):
        logger.warning("cannot acknowledge whale receipt safely: %s", path, exc_info=True)


__all__ = ["acknowledge", "enqueue", "enqueue_with_identity", "list_pending"]
