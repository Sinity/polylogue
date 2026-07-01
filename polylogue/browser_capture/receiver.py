"""Local-only browser-capture receiver helpers."""

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

import orjson

from polylogue.browser_capture.models import (
    BROWSER_CAPTURE_EXTENSION_ORIGIN_WILDCARD,
    BrowserCaptureArchiveLifecycle,
    BrowserCaptureArchiveStatePayload,
    BrowserCaptureEnvelope,
    BrowserCaptureReceiverStatusPayload,
    BrowserPostCommand,
    BrowserPostCommandAckRequest,
    BrowserPostCommandRequest,
)
from polylogue.core.hashing import hash_text_short
from polylogue.paths import archive_root as default_archive_root
from polylogue.paths import browser_capture_spool_root

_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")

#: Environment flag that must equal ``"1"`` before the receiver will enqueue or
#: dispatch any outbound post command. Default OFF: the posting capability is a
#: high-authority action (it writes into a live ChatGPT/Claude thread through the
#: extension), so it cannot fire by accident — an operator must opt in explicitly.
BROWSER_POST_ENABLED_ENV = "POLYLOGUE_BROWSER_POST_ENABLED"

#: Subdirectory of the capture spool that holds queued post commands.
POST_COMMAND_QUEUE_DIRNAME = "post-commands"


@dataclass(frozen=True, slots=True)
class BrowserCaptureReceiverConfig:
    """Configuration for the localhost browser-capture receiver."""

    spool_path: Path
    archive_root: Path | None = None
    allowed_origins: frozenset[str] = frozenset({BROWSER_CAPTURE_EXTENSION_ORIGIN_WILDCARD})
    allow_remote: bool = False
    auth_token: str | None = None

    @classmethod
    def default(cls) -> BrowserCaptureReceiverConfig:
        return cls(spool_path=browser_capture_spool_root())

    def validate(self) -> None:
        """Validate configuration invariants."""
        if self.allow_remote and not self.auth_token:
            raise ValueError("--browser-capture-auth-token is required when --insecure-allow-remote is set")
        unauthenticated_web_origins = sorted(
            origin for origin in self.allowed_origins if not _is_extension_origin_pattern(origin)
        )
        if unauthenticated_web_origins and not self.auth_token:
            raise ValueError(
                "browser-capture web origins require --browser-capture-auth-token; "
                f"unauthenticated origins: {', '.join(unauthenticated_web_origins)}"
            )


@dataclass(frozen=True, slots=True)
class BrowserCaptureWriteResult:
    """Result of accepting a browser-capture envelope."""

    provider: str
    provider_session_id: str
    path: Path
    artifact_ref: str
    bytes_written: int
    replaced: bool


@dataclass(frozen=True, slots=True)
class _RawArchiveLookup:
    raw_row_exists: bool = False
    raw_id: str | None = None
    latest_failure: str | None = None
    failure_source: str | None = None


@dataclass(frozen=True, slots=True)
class _IndexArchiveLookup:
    indexed_session_exists: bool = False
    indexed_session_id: str | None = None
    indexed_message_count: int | None = None


def _is_extension_origin_pattern(origin: str) -> bool:
    return origin == BROWSER_CAPTURE_EXTENSION_ORIGIN_WILDCARD or origin.startswith("chrome-extension://")


def _safe_token(value: str) -> str:
    token = _SAFE_TOKEN.sub("-", value.strip()).strip(".-")
    return token[:96] if token else "session"


def capture_artifact_path(envelope: BrowserCaptureEnvelope, spool_path: Path | None = None) -> Path:
    """Return the deterministic source artifact path for an envelope."""
    root = spool_path if spool_path is not None else BrowserCaptureReceiverConfig.default().spool_path
    provider = _safe_token(envelope.provider.value)
    session = _safe_token(envelope.provider_session_id)
    suffix = hash_text_short(f"{envelope.provider.value}:{envelope.provider_session_id}", 12)
    return root / provider / f"{session}-{suffix}.json"


def capture_artifact_ref(envelope: BrowserCaptureEnvelope, spool_path: Path | None = None) -> str:
    """Return the bounded receiver-facing artifact reference for an envelope."""
    root = spool_path if spool_path is not None else BrowserCaptureReceiverConfig.default().spool_path
    return capture_artifact_path(envelope, root).relative_to(root).as_posix()


def capture_response_id(provider: str, provider_session_id: str, capture_id: str | None = None) -> str:
    """Return a response capture id without double-prefixing the provider."""
    prefix = f"{provider}:"
    value = capture_id or provider_session_id
    while value.startswith(f"{prefix}{prefix}"):
        value = value[len(prefix) :]
    return value if value.startswith(prefix) else f"{prefix}{value}"


def _open_readonly_sqlite(path: Path) -> sqlite3.Connection | None:
    if not path.exists():
        return None
    uri = f"file:{path.as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        return None
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name=? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}


def _origin_candidates_for_provider(provider: str) -> tuple[str, ...]:
    canonical = {
        "chatgpt": ("chatgpt", "chatgpt-export"),
        "openai": ("openai", "chatgpt-export"),
        "claude": ("claude", "claude-ai-export"),
        "claude-ai": ("claude-ai", "claude-ai-export"),
        "claude-code": ("claude-code", "claude-code-session"),
        "codex": ("codex", "codex-session"),
        "aistudio": ("aistudio", "aistudio-drive"),
        "gemini": ("gemini", "aistudio-drive"),
    }
    values = (provider, *canonical.get(provider, ()))
    return tuple(dict.fromkeys(value for value in values if value))


def _lookup_raw_archive_state(
    archive_root: Path,
    *,
    provider: str,
    provider_session_id: str,
    artifact_ref: str,
) -> _RawArchiveLookup:
    conn = _open_readonly_sqlite(archive_root / "source.db")
    if conn is None:
        return _RawArchiveLookup()
    try:
        if not _table_exists(conn, "raw_sessions"):
            return _RawArchiveLookup()
        columns = _columns(conn, "raw_sessions")
        select = ["raw_id"] if "raw_id" in columns else []
        for optional in ("parse_error", "validation_error", "validation_status"):
            if optional in columns:
                select.append(optional)
        if not select:
            return _RawArchiveLookup(raw_row_exists=True)
        where: list[str] = []
        params: list[object] = []
        if "native_id" in columns:
            where.append("native_id = ?")
            params.append(provider_session_id)
        if "origin" in columns and "native_id" in columns:
            origins = _origin_candidates_for_provider(provider)
            placeholders = ",".join("?" for _ in origins)
            where[-1] = f"(native_id = ? AND origin IN ({placeholders}))"
            params.extend(origins)
        if "source_path" in columns:
            where.append("source_path LIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like_suffix(artifact_ref)}")
        if not where:
            return _RawArchiveLookup()
        row = conn.execute(
            f"SELECT {', '.join(select)} FROM raw_sessions WHERE {' OR '.join(where)} ORDER BY rowid DESC LIMIT 1",
            tuple(params),
        ).fetchone()
        if row is None:
            return _RawArchiveLookup()
        row_keys = set(row.keys())
        latest_failure: str | None = None
        failure_source: str | None = None
        parse_error = row["parse_error"] if "parse_error" in row_keys else None
        validation_error = row["validation_error"] if "validation_error" in row_keys else None
        validation_status = (
            str(row["validation_status"]) if "validation_status" in row_keys and row["validation_status"] else None
        )
        if isinstance(parse_error, str) and parse_error:
            latest_failure = parse_error
            failure_source = "raw_parse"
        elif isinstance(validation_error, str) and validation_error:
            latest_failure = validation_error
            failure_source = "raw_validation"
        elif validation_status is not None and validation_status not in {"passed", "valid", "ok"}:
            latest_failure = validation_status
            failure_source = "raw_validation"
        return _RawArchiveLookup(
            raw_row_exists=True,
            raw_id=str(row["raw_id"]) if "raw_id" in row_keys and row["raw_id"] is not None else None,
            latest_failure=latest_failure,
            failure_source=failure_source,
        )
    except sqlite3.Error:
        return _RawArchiveLookup()
    finally:
        conn.close()


def _lookup_index_archive_state(
    archive_root: Path,
    *,
    raw_id: str | None,
    provider: str,
    provider_session_id: str,
) -> _IndexArchiveLookup:
    conn = _open_readonly_sqlite(archive_root / "index.db")
    if conn is None:
        return _IndexArchiveLookup()
    try:
        if not _table_exists(conn, "sessions"):
            return _IndexArchiveLookup()
        columns = _columns(conn, "sessions")
        select = ["session_id"] if "session_id" in columns else []
        if "message_count" in columns:
            select.append("message_count")
        if not select:
            return _IndexArchiveLookup(indexed_session_exists=True)
        where: list[str] = []
        params: list[object] = []
        if raw_id and "raw_id" in columns:
            where.append("raw_id = ?")
            params.append(raw_id)
        if "native_id" in columns:
            if "origin" in columns:
                origins = _origin_candidates_for_provider(provider)
                placeholders = ",".join("?" for _ in origins)
                where.append(f"(native_id = ? AND origin IN ({placeholders}))")
                params.extend((provider_session_id, *origins))
            else:
                where.append("native_id = ?")
                params.append(provider_session_id)
        if not where:
            return _IndexArchiveLookup()
        row = conn.execute(
            f"SELECT {', '.join(select)} FROM sessions WHERE {' OR '.join(where)} ORDER BY rowid DESC LIMIT 1",
            tuple(params),
        ).fetchone()
        if row is None:
            return _IndexArchiveLookup()
        row_keys = set(row.keys())
        session_id = str(row["session_id"]) if "session_id" in row_keys and row["session_id"] is not None else None
        message_count: int | None
        if "message_count" in row_keys and row["message_count"] is not None:
            message_count = int(row["message_count"])
        elif session_id is not None and _table_exists(conn, "messages"):
            count_row = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id=?", (session_id,)).fetchone()
            message_count = int(count_row[0] or 0) if count_row is not None else 0
        else:
            message_count = None
        return _IndexArchiveLookup(
            indexed_session_exists=True,
            indexed_session_id=session_id,
            indexed_message_count=message_count,
        )
    except (sqlite3.Error, ValueError):
        return _IndexArchiveLookup()
    finally:
        conn.close()


def _escape_like_suffix(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def write_capture_envelope(
    envelope: BrowserCaptureEnvelope,
    *,
    spool_path: Path | None = None,
) -> BrowserCaptureWriteResult:
    """Atomically write a browser-capture source artifact into the capture spool."""
    target = capture_artifact_path(envelope, spool_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = envelope.model_dump(mode="json", exclude_none=True)
    raw = orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    replaced = target.exists()
    with tempfile.NamedTemporaryFile("wb", dir=target.parent, prefix=f".{target.name}.", delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(raw)
        handle.write(b"\n")
    temp_path.replace(target)
    return BrowserCaptureWriteResult(
        provider=envelope.provider.value,
        provider_session_id=envelope.provider_session_id,
        path=target,
        artifact_ref=capture_artifact_ref(envelope, spool_path),
        bytes_written=target.stat().st_size,
        replaced=replaced,
    )


def receiver_status_payload(config: BrowserCaptureReceiverConfig) -> dict[str, object]:
    """Return JSON status for extension health checks."""
    return BrowserCaptureReceiverStatusPayload(
        spool_path=str(config.spool_path),
        spool_ready=True,
        allowed_origins=sorted(config.allowed_origins),
        allow_remote=config.allow_remote,
        auth_required=config.auth_token is not None,
        active=True,
        checked_at=datetime.now(UTC).isoformat(),
    ).model_dump(mode="json")


def existing_capture_state(
    provider: str,
    provider_session_id: str,
    *,
    spool_path: Path | None = None,
    archive_root: Path | None = None,
) -> dict[str, object]:
    """Return the local capture state visible to the browser extension."""
    envelope = BrowserCaptureEnvelope.model_validate(
        {
            "provenance": {
                "source_url": "about:blank",
                "captured_at": "1970-01-01T00:00:00+00:00",
                "adapter_name": "state-lookup",
            },
            "session": {
                "provider": provider,
                "provider_session_id": provider_session_id,
                "turns": [{"provider_turn_id": "lookup", "role": "system", "text": "lookup"}],
            },
        }
    )
    path = capture_artifact_path(envelope, spool_path)
    artifact_ref = capture_artifact_ref(envelope, spool_path)
    capture_id: str | None = None
    updated_at: str | None = None
    artifact_readable: bool | None = None
    spooled = path.exists()
    latest_failure: str | None = None
    failure_source: str | None = None
    if spooled:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_capture_id = payload.get("capture_id")
            raw_updated_at = payload.get("session", {}).get("updated_at")
            capture_id = raw_capture_id if isinstance(raw_capture_id, str) else None
            updated_at = raw_updated_at if isinstance(raw_updated_at, str) else None
        except (OSError, json.JSONDecodeError, AttributeError):
            artifact_readable = False
            latest_failure = "spool_unreadable"
            failure_source = "spool"
    root = archive_root if archive_root is not None else default_archive_root()
    raw = _lookup_raw_archive_state(
        root,
        provider=envelope.provider.value,
        provider_session_id=envelope.provider_session_id,
        artifact_ref=artifact_ref,
    )
    index = _lookup_index_archive_state(
        root,
        raw_id=raw.raw_id,
        provider=envelope.provider.value,
        provider_session_id=envelope.provider_session_id,
    )
    latest_failure = latest_failure or raw.latest_failure
    failure_source = failure_source or raw.failure_source
    lifecycle: BrowserCaptureArchiveLifecycle
    if latest_failure is not None:
        lifecycle = "failed"
    elif raw.raw_row_exists and index.indexed_session_exists and (index.indexed_message_count or 0) > 0:
        lifecycle = "archived"
    elif raw.raw_row_exists:
        lifecycle = "ingest_pending"
    elif spooled:
        lifecycle = "spooled_only"
    else:
        lifecycle = "missing"
    captured = lifecycle == "archived"
    return BrowserCaptureArchiveStatePayload(
        provider=envelope.provider.value,
        provider_session_id=envelope.provider_session_id,
        state=lifecycle,
        lifecycle=lifecycle,
        captured=captured,
        spooled=spooled,
        artifact_ref=artifact_ref,
        capture_id=capture_response_id(envelope.provider.value, envelope.provider_session_id, capture_id),
        updated_at=updated_at,
        artifact_readable=artifact_readable,
        raw_row_exists=raw.raw_row_exists,
        raw_id=raw.raw_id,
        indexed_session_exists=index.indexed_session_exists,
        indexed_session_id=index.indexed_session_id,
        indexed_message_count=index.indexed_message_count,
        latest_failure=latest_failure,
        failure_source=failure_source,
    ).model_dump(mode="json", exclude_none=True)


class BrowserPostDisabledError(RuntimeError):
    """Raised when a post command is requested while posting is disabled."""


class BrowserPostCommandConflictError(RuntimeError):
    """Raised when a queued post command id would overwrite an existing command."""


class BrowserPostCommandStateError(RuntimeError):
    """Raised when an ack targets a command state that cannot be acked."""


def browser_post_enabled() -> bool:
    """Return whether outbound posting is enabled via the env safety flag."""
    return os.environ.get(BROWSER_POST_ENABLED_ENV, "") == "1"


def post_command_queue_root(spool_path: Path | None = None) -> Path:
    """Return the directory that holds queued outbound post commands."""
    root = spool_path if spool_path is not None else BrowserCaptureReceiverConfig.default().spool_path
    return root / POST_COMMAND_QUEUE_DIRNAME


def _post_command_path(root: Path, command_id: str) -> Path:
    return root / f"{_safe_token(command_id)}.json"


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(raw)
        handle.write(b"\n")
    temp_path.replace(path)


def _write_post_command(root: Path, command: BrowserPostCommand) -> Path:
    target = _post_command_path(root, command.command_id)
    _atomic_write_json(target, command.model_dump(mode="json", exclude_none=True))
    return target


def _read_post_command(path: Path) -> BrowserPostCommand | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return BrowserPostCommand.model_validate(payload)
    except Exception:
        return None


def enqueue_post_command(
    request: BrowserPostCommandRequest,
    *,
    spool_path: Path | None = None,
) -> BrowserPostCommand:
    """Persist an outbound post command in the queue.

    Raises :class:`BrowserPostDisabledError` unless the
    ``POLYLOGUE_BROWSER_POST_ENABLED=1`` safety flag is set, so the capability
    cannot fire by accident.
    """
    if not browser_post_enabled():
        raise BrowserPostDisabledError(f"outbound posting is disabled; set {BROWSER_POST_ENABLED_ENV}=1 to enable")
    root = post_command_queue_root(spool_path)
    command_id = _safe_token(request.command_id) if request.command_id else uuid4().hex
    if _post_command_path(root, command_id).exists():
        raise BrowserPostCommandConflictError(f"post command already exists: {command_id}")
    now = datetime.now(UTC).isoformat()
    command = BrowserPostCommand(
        command_id=command_id,
        provider=request.provider,
        target=request.target,
        text=request.text,
        submit=request.submit,
        status="pending",
        created_at=now,
        updated_at=now,
    )
    _write_post_command(root, command)
    return command


def poll_post_commands(
    *,
    provider: str | None = None,
    spool_path: Path | None = None,
    claim: bool = True,
) -> list[BrowserPostCommand]:
    """Return pending post commands, optionally claiming them (pending -> dispatched).

    Returns an empty list when posting is disabled, so a manually-placed queue
    file is never dispatched while the safety flag is off. Claiming a command
    prevents a subsequent poll from re-dispatching it; an unacked dispatched
    command is intentionally not auto-retried.
    """
    if not browser_post_enabled():
        return []
    root = post_command_queue_root(spool_path)
    if not root.exists():
        return []
    commands: list[BrowserPostCommand] = []
    for path in sorted(root.glob("*.json")):
        command = _read_post_command(path)
        if command is None or command.status != "pending":
            continue
        if provider is not None and command.provider != provider:
            continue
        if claim:
            now = datetime.now(UTC).isoformat()
            command.status = "dispatched"
            command.dispatched_at = now
            command.updated_at = now
            _write_post_command(root, command)
        commands.append(command)
    return commands


def ack_post_command(
    command_id: str,
    ack: BrowserPostCommandAckRequest,
    *,
    spool_path: Path | None = None,
) -> BrowserPostCommand | None:
    """Record the extension's result for a dispatched post command."""
    root = post_command_queue_root(spool_path)
    path = _post_command_path(root, command_id)
    command = _read_post_command(path)
    if command is None:
        return None
    if command.status in {"submitted", "failed"}:
        return command
    if command.status != "dispatched":
        raise BrowserPostCommandStateError(f"cannot ack post command in {command.status!r} state")
    now = datetime.now(UTC).isoformat()
    command.status = ack.status
    command.detail = ack.detail
    command.observed_url = ack.observed_url
    command.acked_at = now
    command.updated_at = now
    _write_post_command(root, command)
    return command


__all__ = [
    "BROWSER_POST_ENABLED_ENV",
    "POST_COMMAND_QUEUE_DIRNAME",
    "BrowserCaptureReceiverConfig",
    "BrowserCaptureWriteResult",
    "BrowserPostCommandConflictError",
    "BrowserPostCommandStateError",
    "BrowserPostDisabledError",
    "ack_post_command",
    "browser_post_enabled",
    "capture_artifact_ref",
    "capture_response_id",
    "_is_extension_origin_pattern",
    "capture_artifact_path",
    "enqueue_post_command",
    "existing_capture_state",
    "poll_post_commands",
    "post_command_queue_root",
    "receiver_status_payload",
    "write_capture_envelope",
]
