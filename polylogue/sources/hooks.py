"""Durable local spool for Claude Code and Codex hook events.

Hook commands must return promptly and cannot rely on the archive daemon being
up.  They therefore atomically place one immutable envelope in ``pending``.
The daemon drains those envelopes into ``source.db`` and only moves a file to
``acknowledged`` after its ``raw_hook_events`` row has committed.  A crash in
between is safe: replay uses the stable event id as the source-tier key.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from polylogue.core.enums import Origin, Provider
from polylogue.logging import get_logger
from polylogue.paths import hooks_sidecar_dir
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveHookEvent, write_source_raw_session

logger = get_logger(__name__)

_SUPPORTED_PROVIDERS = frozenset({"claude-code", "codex"})
_PENDING_DIRNAME = "pending"
_ACKNOWLEDGED_DIRNAME = "acknowledged"
_SAFE_EVENT_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class HookSpoolRecordError(ValueError):
    """A pending spool file is not a valid Claude Code/Codex hook envelope."""


@dataclass(frozen=True, slots=True)
class HookSpoolDrainResult:
    """Outcome of one durable hook-spool drain attempt."""

    acknowledged: int
    failed: int


def pending_hook_spool_dir(root: Path | None = None) -> Path:
    """Return the directory that hook commands append to atomically."""

    return (root or hook_spool_root()) / _PENDING_DIRNAME


def acknowledged_hook_spool_dir(root: Path | None = None) -> Path:
    """Return the receipt directory for source-tier-acknowledged events."""

    return (root or hook_spool_root()) / _ACKNOWLEDGED_DIRNAME


def hook_spool_root() -> Path:
    """Resolve the hook spool root shared by producer and daemon.

    ``POLYLOGUE_HOOK_SIDECAR_DIR`` predates the durable spool and remains the
    documented operator override.  Applying it here, rather than only in the
    command-hook entrypoint, keeps the daemon on the same receipt path.
    """

    override = os.environ.get("POLYLOGUE_HOOK_SIDECAR_DIR", "").strip()
    return Path(override).expanduser() if override else hooks_sidecar_dir()


def enqueue_hook_event(
    *,
    event_type: str,
    session_id: str,
    provider: str,
    timestamp: str,
    payload: dict[str, object],
    root: Path | None = None,
    event_id: str | None = None,
) -> Path:
    """Atomically enqueue one hook event before the daemon receives it."""

    record: dict[str, object] = {
        "event_id": event_id or uuid4().hex,
        "event_type": event_type,
        "session_id": session_id,
        "timestamp": timestamp,
        "provider": provider,
        "payload": payload,
    }
    normalized = _validated_record(record)
    if not _SAFE_EVENT_ID.fullmatch(str(normalized["event_id"])):
        raise HookSpoolRecordError("hook spool event_id must contain only letters, digits, '_' or '-'")
    pending = pending_hook_spool_dir(root)
    pending.mkdir(parents=True, exist_ok=True)
    target = pending / f"{normalized['event_id']}.json"
    if target.exists():
        return target
    _atomic_json_write(target, normalized)
    return target


def drain_hook_event_spool(
    archive_root: Path,
    *,
    root: Path | None = None,
) -> HookSpoolDrainResult:
    """Persist every pending event and acknowledge only committed records.

    Pending files deliberately remain in place when archive writes or envelope
    validation fail.  The next daemon pass can retry a transient write failure;
    a malformed producer record remains inspectable rather than disappearing.
    """

    pending = pending_hook_spool_dir(root)
    if not pending.exists():
        return HookSpoolDrainResult(acknowledged=0, failed=0)
    acknowledged = 0
    failed = 0
    for path in sorted(pending.glob("*.json")):
        try:
            record = _read_record(path)
            _persist_record(archive_root, path, record)
            _acknowledge(path, root=root)
        except (HookSpoolRecordError, OSError, sqlite3.Error, ValueError):
            failed += 1
            logger.warning("hook spool event remains pending: %s", path, exc_info=True)
        else:
            acknowledged += 1
    return HookSpoolDrainResult(acknowledged=acknowledged, failed=failed)


def _read_record(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HookSpoolRecordError(f"invalid hook spool JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise HookSpoolRecordError(f"hook spool envelope must be an object: {path.name}")
    return _validated_record(value)


def _validated_record(value: dict[str, object]) -> dict[str, object]:
    required_text = ("event_id", "event_type", "session_id", "timestamp", "provider")
    for key in required_text:
        item = value.get(key)
        if not isinstance(item, str) or not item.strip():
            raise HookSpoolRecordError(f"hook spool envelope has no {key}")
    provider = str(value["provider"])
    if provider not in _SUPPORTED_PROVIDERS:
        raise HookSpoolRecordError(f"unsupported hook provider: {provider}")
    payload = value.get("payload")
    if not isinstance(payload, dict):
        raise HookSpoolRecordError("hook spool envelope payload must be an object")
    observed_at_ms = _timestamp_ms(str(value["timestamp"]))
    return {
        "event_id": str(value["event_id"]),
        "event_type": str(value["event_type"]),
        "session_id": str(value["session_id"]),
        "timestamp": str(value["timestamp"]),
        "provider": provider,
        "payload": dict(payload),
        "observed_at_ms": observed_at_ms,
    }


def _timestamp_ms(value: str) -> int:
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC).timestamp() * 1000)
    except ValueError as exc:
        raise HookSpoolRecordError(f"invalid hook timestamp: {value!r}") from exc


def _persist_record(archive_root: Path, path: Path, record: dict[str, object]) -> None:
    initialize_active_archive_root(archive_root)
    provider = Provider.from_string(str(record["provider"]))
    origin = Origin.from_string("claude-code-session" if provider is Provider.CLAUDE_CODE else "codex-session")
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    observed_at_ms_value = record["observed_at_ms"]
    if not isinstance(observed_at_ms_value, int):
        raise HookSpoolRecordError("hook spool envelope has an invalid observed timestamp")
    observed_at_ms = observed_at_ms_value
    source_path = str(path)
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=provider,
            payload=payload,
            source_path=source_path,
            acquired_at_ms=observed_at_ms,
        )
        write_source_raw_session(
            archive._ensure_source_conn(),
            origin=origin,
            source_path=source_path,
            source_index=0,
            payload=payload,
            acquired_at_ms=observed_at_ms,
            raw_id=raw_id,
            hook_event=ArchiveHookEvent(
                hook_event_id=f"hook:{record['event_id']}",
                origin=origin,
                source_path=source_path,
                event_type=str(record["event_type"]),
                payload=record,
                observed_at_ms=observed_at_ms,
                native_id=f"{record['session_id']}:{record['event_type']}:{record['event_id']}",
                session_native_id=str(record["session_id"]),
            ),
        )


def _acknowledge(path: Path, *, root: Path | None) -> None:
    acknowledged = acknowledged_hook_spool_dir(root)
    acknowledged.mkdir(parents=True, exist_ok=True)
    os.replace(path, acknowledged / path.name)
    _fsync_directory(acknowledged)


def _atomic_json_write(path: Path, payload: dict[str, object]) -> None:
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            output.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    """Persist an atomic rename's directory entry before returning success."""

    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "HookSpoolDrainResult",
    "HookSpoolRecordError",
    "acknowledged_hook_spool_dir",
    "drain_hook_event_spool",
    "enqueue_hook_event",
    "hook_spool_root",
    "pending_hook_spool_dir",
]
