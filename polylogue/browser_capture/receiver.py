"""Local-only browser-capture receiver helpers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import sqlite3
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import orjson

from polylogue.browser_capture.models import (
    BROWSER_CAPTURE_API_SCHEMA,
    BROWSER_CAPTURE_EXTENSION_ORIGIN_WILDCARD,
    BrowserBackfillCheckpointRecord,
    BrowserBackfillCheckpointRequest,
    BrowserCaptureArchiveLifecycle,
    BrowserCaptureArchiveStatePayload,
    BrowserCaptureEnvelope,
    BrowserCaptureReceiverStatusPayload,
    BrowserPostCommand,
    BrowserPostCommandAckRequest,
    BrowserPostCommandRequest,
)
from polylogue.core.hashing import hash_text_short
from polylogue.core.timestamps import parse_timestamp
from polylogue.logging import get_logger
from polylogue.paths import archive_root as default_archive_root
from polylogue.paths import browser_capture_receiver_token_path, browser_capture_spool_root

logger = get_logger(__name__)

_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")

#: Environment flag that must equal ``"1"`` before the receiver will enqueue or
#: dispatch any outbound post command. Default OFF: the posting capability is a
#: high-authority action (it writes into a live ChatGPT/Claude thread through the
#: extension), so it cannot fire by accident — an operator must opt in explicitly.
BROWSER_POST_ENABLED_ENV = "POLYLOGUE_BROWSER_POST_ENABLED"

#: Subdirectory of the capture spool that holds queued post commands.
POST_COMMAND_QUEUE_DIRNAME = "post-commands"

#: Subdirectory of the capture spool that holds mirrored backfill-ledger
#: checkpoints (polylogue-06zm). One file per extension_instance_id.
BACKFILL_CHECKPOINT_DIRNAME = "backfill-checkpoints"

# Backfill scheduling identifies the observer that acquired a snapshot rather
# than a semantic property of the provider session.  Keep it out of the spool
# deduplication fingerprint, while retaining any future semantic backfill
# metadata a provider might add.
_BACKFILL_OBSERVER_ATTRIBUTION_KEYS = frozenset({"job_id", "queue_id", "instance_id"})


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


#: Entropy (bytes, pre-base64) for an auto-minted receiver pairing token.
RECEIVER_TOKEN_ENTROPY_BYTES = 32

#: Env flag that must equal ``"1"`` before the receiver will start with no
#: bearer token at all. Default OFF: without a token, any local process (not
#: just a browser page) can read the spool/archive-lifecycle state and post
#: forged captures — auto-minting a token so unauthenticated requests are
#: refused by default closes that hole (polylogue-gnie), and this flag is the
#: explicit, logged escape hatch for the rare intentionally-open setup.
BROWSER_CAPTURE_ALLOW_NO_AUTH_ENV = "POLYLOGUE_BROWSER_CAPTURE_ALLOW_NO_AUTH"


def load_or_mint_receiver_token(path: Path | None = None, *, rotate: bool = False) -> str:
    """Return the receiver's persisted bearer token, minting one on first use.

    This is a local pairing secret (paste into the extension popup's
    "Receiver token" field), not an OAuth credential, so it gets a plain
    0600 file rather than :mod:`polylogue.sources.token_store`'s
    keyring-backed store. Written atomically (mkstemp + fchmod(0o600) before
    any bytes land, then ``os.replace``) so the token is never briefly
    world-readable under a permissive umask.
    """
    target = path if path is not None else browser_capture_receiver_token_path()
    if not rotate and target.exists():
        existing = target.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    token = secrets.token_urlsafe(RECEIVER_TOKEN_ENTROPY_BYTES)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token)
        os.replace(tmp_path, target)
    except BaseException:
        with suppress(FileNotFoundError):
            tmp_path.unlink()
        raise
    return token


def resolve_receiver_auth_token(
    explicit_token: str | None,
    *,
    allow_no_auth: bool = False,
    token_path: Path | None = None,
) -> str | None:
    """Return the bearer token the receiver should require before serving.

    An explicitly configured token always wins. Otherwise the receiver
    auto-mints/loads a persisted 0600 token so unauthenticated capture POSTs
    (and the GET status/archive-state/post-command routes, which share the
    same auth gate) are refused by default. ``allow_no_auth`` is the
    explicit, loudly-logged opt-out for the rare setup that wants the
    pre-gnie fully-open posture.
    """
    if explicit_token:
        return explicit_token
    if allow_no_auth:
        logger.warning(
            "browser_capture.auth_disabled",
            reason="allow_no_auth explicitly set",
            risk="any local process can read spool/archive state and post forged captures",
        )
        return None
    return load_or_mint_receiver_token(token_path)


@dataclass(frozen=True, slots=True)
class BrowserCaptureWriteResult:
    """Result of accepting a browser-capture envelope."""

    provider: str
    provider_session_id: str
    path: Path
    artifact_ref: str
    bytes_written: int
    replaced: bool
    deduplicated: bool
    dedup_content_hash: str
    capture_instance_id: str | None


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
    indexed_updated_at_ms: int | None = None


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


def _semantic_provider_meta(provider_meta: dict[str, object]) -> dict[str, object]:
    """Return provider metadata without backfill-observer attribution."""
    semantic_meta = dict(provider_meta)
    backfill = semantic_meta.get("backfill")
    if not isinstance(backfill, dict):
        return semantic_meta

    semantic_backfill = {
        key: value for key, value in backfill.items() if key not in _BACKFILL_OBSERVER_ATTRIBUTION_KEYS
    }
    if semantic_backfill:
        semantic_meta["backfill"] = semantic_backfill
    else:
        semantic_meta.pop("backfill", None)
    return semantic_meta


def capture_dedup_content_hash(envelope: BrowserCaptureEnvelope) -> str:
    """Hash capture content independently from observation-specific provenance.

    A browser extension instance and its capture timestamp identify *who saw*
    a snapshot, not a different provider session revision. Keeping them out of
    this hash lets two concurrently running instances converge on one spool
    artifact while the receiver can still echo each poster's attribution.
    """
    session = envelope.session.model_dump(mode="json", exclude_none=True)
    session["provider_meta"] = _semantic_provider_meta(envelope.session.provider_meta)
    payload = {
        "polylogue_capture_kind": envelope.polylogue_capture_kind,
        "schema_version": envelope.schema_version,
        "session": session,
        "provider_meta": _semantic_provider_meta(envelope.provider_meta),
        "raw_provider_payload": envelope.raw_provider_payload,
    }
    return hashlib.sha256(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)).hexdigest()


def _timestamp_ms(value: object) -> int | None:
    parsed = parse_timestamp(value if isinstance(value, (str, int, float)) else None)
    return int(parsed.timestamp() * 1000) if parsed is not None else None


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
        if "updated_at_ms" in columns:
            select.append("updated_at_ms")
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
            indexed_updated_at_ms=(
                int(row["updated_at_ms"]) if "updated_at_ms" in row_keys and row["updated_at_ms"] is not None else None
            ),
        )
    except (sqlite3.Error, ValueError):
        return _IndexArchiveLookup()
    finally:
        conn.close()


def _escape_like_suffix(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# Spool governor (kwsb.1): a hostile or runaway poster sending distinct
# (provider, provider_session_id) pairs creates a new file per capture —
# unlike a repeat capture of the SAME session, which replaces its existing
# file in place and never grows the spool. These bounds cap that growth.
SPOOL_MAX_FILES = 20_000
SPOOL_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB

#: Outbound post commands are small control-plane objects (a target, some
#: text, a submit flag), not archive content, so their queue gets a much
#: tighter bound than the capture spool. This still exists to cap a runaway
#: caller looping enqueue_post_command, not to bound legitimate use --
#: dispatched/acked commands are expected to be pruned by the extension's
#: poll/ack cycle long before this is reached.
POST_COMMAND_QUEUE_MAX_FILES = 5_000
POST_COMMAND_QUEUE_MAX_BYTES = 50 * 1024 * 1024  # 50 MiB


class SpoolQuotaExceededError(RuntimeError):
    """Raised when writing a new (non-replacing) capture would exceed the spool quota."""


# BrowserCaptureHTTPServer is a ThreadingHTTPServer — concurrent POSTs run on
# separate threads. Without this lock, multiple new-capture writes could all
# pass _check_spool_quota() before any of them lands (TOCTOU), overshooting
# the quota under load. Held across check-and-write, not just the check.
_SPOOL_WRITE_LOCK = threading.Lock()


@contextmanager
def _spool_file_lock(spool_root: Path) -> Iterator[None]:
    """Serialize writers from distinct receiver processes sharing a spool."""
    spool_root.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(spool_root / ".polylogue-browser-capture.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


@dataclass(frozen=True, slots=True)
class SpoolUsage:
    file_count: int
    total_bytes: int


def _spool_usage(spool_root: Path) -> SpoolUsage:
    file_count = 0
    total_bytes = 0
    if spool_root.exists():
        for path in spool_root.rglob("*.json"):
            try:
                total_bytes += path.stat().st_size
            except OSError:
                continue
            file_count += 1
    return SpoolUsage(file_count=file_count, total_bytes=total_bytes)


def _check_spool_quota(
    spool_root: Path,
    *,
    max_files: int,
    max_bytes: int,
    label: str = "capture spool",
) -> None:
    """Callers must pass max_files/max_bytes explicitly (not as defaults
    bound to the module constants) so tests can monkeypatch SPOOL_MAX_FILES/
    SPOOL_MAX_BYTES/POST_COMMAND_QUEUE_MAX_* and have it take effect --
    a default parameter value binds at function-definition time, before
    any monkeypatch runs."""
    usage = _spool_usage(spool_root)
    if usage.file_count >= max_files or usage.total_bytes >= max_bytes:
        raise SpoolQuotaExceededError(
            f"{label} quota exceeded: {usage.file_count} files, {usage.total_bytes} bytes "
            f"(limits: {max_files} files, {max_bytes} bytes)"
        )


def write_capture_envelope(
    envelope: BrowserCaptureEnvelope,
    *,
    spool_path: Path | None = None,
) -> BrowserCaptureWriteResult:
    """Atomically write a browser-capture source artifact into the capture spool.

    Raises :class:`SpoolQuotaExceededError` before writing a NEW artifact
    (one that does not replace an existing same-session file) once the
    spool quota is reached — replacing an existing capture never grows the
    spool and is always allowed. The quota check and the write are
    serialized against every other call (see ``_SPOOL_WRITE_LOCK``) so
    concurrent requests cannot all pass the check before any one write
    lands.
    """
    root = spool_path if spool_path is not None else BrowserCaptureReceiverConfig.default().spool_path
    target = capture_artifact_path(envelope, root)
    dedup_content_hash = capture_dedup_content_hash(envelope)
    with _SPOOL_WRITE_LOCK, _spool_file_lock(root):
        replaced = target.exists()
        if replaced:
            try:
                existing = BrowserCaptureEnvelope.model_validate(orjson.loads(target.read_bytes()))
            except (OSError, orjson.JSONDecodeError, ValueError):
                existing = None
            if existing is not None and capture_dedup_content_hash(existing) == dedup_content_hash:
                return BrowserCaptureWriteResult(
                    provider=envelope.provider.value,
                    provider_session_id=envelope.provider_session_id,
                    path=target,
                    artifact_ref=capture_artifact_ref(envelope, root),
                    bytes_written=target.stat().st_size,
                    replaced=True,
                    deduplicated=True,
                    dedup_content_hash=dedup_content_hash,
                    capture_instance_id=envelope.provenance.extension_instance_id,
                )
        else:
            _check_spool_quota(root, max_files=SPOOL_MAX_FILES, max_bytes=SPOOL_MAX_BYTES)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = envelope.model_dump(mode="json", exclude_none=True)
        raw = orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb", dir=target.parent, prefix=f".{target.name}.", delete=False
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(raw)
                handle.write(b"\n")
            temp_path.replace(target)
        except BaseException:
            if temp_path is not None:
                with suppress(FileNotFoundError):
                    temp_path.unlink()
            raise
    return BrowserCaptureWriteResult(
        provider=envelope.provider.value,
        provider_session_id=envelope.provider_session_id,
        path=target,
        artifact_ref=capture_artifact_ref(envelope, root),
        bytes_written=target.stat().st_size,
        replaced=replaced,
        deduplicated=False,
        dedup_content_hash=dedup_content_hash,
        capture_instance_id=envelope.provenance.extension_instance_id,
    )


def receiver_identity(config: BrowserCaptureReceiverConfig) -> str:
    """Return a stable, non-secret identity for one paired receiver.

    The persisted bearer token is the receiver's pairing root and remains stable
    across daemon restarts.  Hashing it with a domain separator produces a
    comparison-safe identifier without sending the token itself.  The explicit
    no-auth escape hatch has no token, so it falls back to the resolved spool
    path; that setup is already opt-in and local-only.
    """
    if config.auth_token:
        material = f"token:{config.auth_token}"
    else:
        material = f"no-auth-spool:{config.spool_path.expanduser().resolve()}"
    digest = hashlib.sha256(f"polylogue-browser-capture-receiver\0{material}".encode()).hexdigest()
    return f"rx-{digest[:20]}"


def receiver_status_payload(config: BrowserCaptureReceiverConfig) -> dict[str, object]:
    """Return JSON status for extension health checks."""
    return BrowserCaptureReceiverStatusPayload(
        api_schema=BROWSER_CAPTURE_API_SCHEMA,
        receiver_id=receiver_identity(config),
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
    spooled_updated_at_ms: int | None = None
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
            spooled_updated_at_ms = _timestamp_ms(updated_at)
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
    archive_current_for_spool = not (
        spooled_updated_at_ms is not None
        and index.indexed_updated_at_ms is not None
        and spooled_updated_at_ms > index.indexed_updated_at_ms
    )
    if latest_failure is not None:
        lifecycle = "failed"
    elif (
        raw.raw_row_exists
        and index.indexed_session_exists
        and (index.indexed_message_count or 0) > 0
        and archive_current_for_spool
    ):
        lifecycle = "archived"
    elif (
        spooled
        and raw.raw_row_exists
        and index.indexed_session_exists
        and (index.indexed_message_count or 0) > 0
        and not archive_current_for_spool
    ):
        lifecycle = "stale"
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
    cannot fire by accident. Raises :class:`SpoolQuotaExceededError` before
    writing a NEW command once the queue quota is reached, guarded by the
    same write lock the capture-spool path uses so concurrent enqueues
    cannot all pass the check before any one write lands (polylogue-gnie).
    """
    if not browser_post_enabled():
        raise BrowserPostDisabledError(f"outbound posting is disabled; set {BROWSER_POST_ENABLED_ENV}=1 to enable")
    root = post_command_queue_root(spool_path)
    command_id = _safe_token(request.command_id) if request.command_id else uuid4().hex
    with _SPOOL_WRITE_LOCK:
        if _post_command_path(root, command_id).exists():
            raise BrowserPostCommandConflictError(f"post command already exists: {command_id}")
        _check_spool_quota(
            root,
            max_files=POST_COMMAND_QUEUE_MAX_FILES,
            max_bytes=POST_COMMAND_QUEUE_MAX_BYTES,
            label="post-command queue",
        )
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


# ---- Backfill-ledger checkpoint mirror (polylogue-06zm) --------------------
#
# The extension's IndexedDB backfill ledger is the fast primary source; a
# local chrome.storage.local copy already survives an ordinary profile
# restart (polylogue-jlme.4). This mirror is the second fallback: a
# receiver-owned, credential-free copy that outlives BOTH of those when a
# profile is destructively re-seeded or the extension is fully reinstalled,
# so recovery has a durable place to look beyond the browser profile itself.
# One JSON file per extension_instance_id, overwritten on every checkpoint
# (last write wins) -- the service worker is single-threaded and only ever
# checkpoints its own ledger, so there is exactly one legitimate writer per
# instance id at a time. The file-count bound still guards against a buggy
# or hostile caller minting unbounded distinct instance ids.
BACKFILL_CHECKPOINT_MAX_FILES = 2_000
BACKFILL_CHECKPOINT_MAX_BYTES = 200 * 1024 * 1024  # 200 MiB


def backfill_checkpoint_root(spool_path: Path | None = None) -> Path:
    """Return the directory that holds mirrored backfill-ledger checkpoints."""
    root = spool_path if spool_path is not None else BrowserCaptureReceiverConfig.default().spool_path
    return root / BACKFILL_CHECKPOINT_DIRNAME


def _backfill_checkpoint_path(root: Path, instance_id: str) -> Path:
    return root / f"{_safe_token(instance_id)}.json"


def write_backfill_checkpoint(
    request: BrowserBackfillCheckpointRequest,
    *,
    spool_path: Path | None = None,
) -> BrowserBackfillCheckpointRecord:
    """Persist a credential-free backfill-ledger checkpoint mirror.

    Overwrites any prior checkpoint for the same ``extension_instance_id``
    (last write wins; see module comment above). Guarded by the same write
    lock and quota-check pattern the capture spool and post-command queue
    use, so a concurrent write cannot bypass the quota check (TOCTOU).
    """
    root = backfill_checkpoint_root(spool_path)
    with _SPOOL_WRITE_LOCK:
        target = _backfill_checkpoint_path(root, request.extension_instance_id)
        if not target.exists():
            _check_spool_quota(
                root,
                max_files=BACKFILL_CHECKPOINT_MAX_FILES,
                max_bytes=BACKFILL_CHECKPOINT_MAX_BYTES,
                label="backfill-checkpoint mirror",
            )
        record = BrowserBackfillCheckpointRecord(
            extension_instance_id=request.extension_instance_id,
            checkpoint=request.checkpoint,
            stored_at=datetime.now(UTC).isoformat(),
        )
        _atomic_write_json(target, record.model_dump(mode="json"))
    return record


def read_backfill_checkpoint(
    instance_id: str,
    *,
    spool_path: Path | None = None,
) -> BrowserBackfillCheckpointRecord | None:
    """Return the mirrored checkpoint for an extension instance, if any."""
    root = backfill_checkpoint_root(spool_path)
    path = _backfill_checkpoint_path(root, instance_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return BrowserBackfillCheckpointRecord.model_validate(payload)
    except Exception:
        return None


__all__ = [
    "BACKFILL_CHECKPOINT_DIRNAME",
    "BACKFILL_CHECKPOINT_MAX_BYTES",
    "BACKFILL_CHECKPOINT_MAX_FILES",
    "BROWSER_CAPTURE_ALLOW_NO_AUTH_ENV",
    "BROWSER_POST_ENABLED_ENV",
    "POST_COMMAND_QUEUE_DIRNAME",
    "RECEIVER_TOKEN_ENTROPY_BYTES",
    "BrowserCaptureReceiverConfig",
    "BrowserCaptureWriteResult",
    "BrowserPostCommandConflictError",
    "BrowserPostCommandStateError",
    "BrowserPostDisabledError",
    "ack_post_command",
    "backfill_checkpoint_root",
    "browser_post_enabled",
    "capture_artifact_ref",
    "capture_response_id",
    "_is_extension_origin_pattern",
    "capture_artifact_path",
    "enqueue_post_command",
    "existing_capture_state",
    "load_or_mint_receiver_token",
    "poll_post_commands",
    "post_command_queue_root",
    "read_backfill_checkpoint",
    "receiver_identity",
    "receiver_status_payload",
    "resolve_receiver_auth_token",
    "write_backfill_checkpoint",
    "write_capture_envelope",
]
