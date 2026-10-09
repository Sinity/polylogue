"""Local-only browser-capture receiver helpers."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import hmac
import io
import os
import re
import secrets
import sqlite3
import stat
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol

from polylogue.browser_capture.capture_stream import (
    AttachmentFact,
    CaptureEnvelopeError,
    CaptureSummary,
    StagedCapture,
    read_capture_state_fields,
    stage_capture_body,
    summarize_capture_file,
    summarize_capture_stream,
)
from polylogue.browser_capture.models import (
    BROWSER_CAPTURE_API_SCHEMA,
    BROWSER_CAPTURE_EXTENSION_ORIGIN_WILDCARD,
    BrowserCaptureAcceptedIdentity,
    BrowserCaptureArchiveLifecycle,
    BrowserCaptureArchiveStatePayload,
    BrowserCaptureEnvelope,
    BrowserCaptureReceiverStatusPayload,
)
from polylogue.core.durable_fs import atomic_replace, sync_directory
from polylogue.core.enums import Provider
from polylogue.core.hashing import hash_file, hash_text_short
from polylogue.core.json import dumps_bytes
from polylogue.core.raw_state import raw_state_authority
from polylogue.core.sqlite_introspection import table_exists as _table_exists
from polylogue.core.timestamps import to_epoch_ms
from polylogue.logging import get_logger
from polylogue.paths import archive_root as default_archive_root
from polylogue.paths import (
    browser_capture_receiver_identity_path,
    browser_capture_receiver_token_path,
    browser_capture_spool_root,
)
from polylogue.storage.archive_identity import ArchiveLocationError, resolve_active_index_path
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

logger = get_logger(__name__)

_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")

#: Subdirectory of the capture spool that holds mirrored backfill-ledger
#: checkpoints (polylogue-06zm). One file per extension_instance_id.
BACKFILL_CHECKPOINT_DIRNAME = "backfill-checkpoints"


@dataclass(frozen=True, slots=True)
class BrowserCaptureReceiverConfig:
    """Configuration for the localhost browser-capture receiver."""

    spool_path: Path
    archive_root: Path | None = None
    allowed_origins: frozenset[str] = frozenset({BROWSER_CAPTURE_EXTENSION_ORIGIN_WILDCARD})
    allow_remote: bool = False
    auth_token: str | None = field(default=None, repr=False)
    # The receiver pairing token and the daemon machine credential are distinct.
    api_auth_token: str | None = field(default=None, repr=False)
    api_allow_no_auth: bool = False

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


def _is_trusted_token_file(target: Path) -> bool:
    """Refuse to trust a token file this process does not exclusively own.

    Mirrors :func:`polylogue.daemon.api_auth._is_trusted_token_file` --
    reading an existing token file's bytes without first checking who put
    them there repeats the exact filesystem-boundary assumption polylogue-n6pz
    found reachable elsewhere. A symlink is never followed; a regular file
    must be owned by our own uid with no group/other permission bits.
    """
    try:
        info = target.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return False
    if info.st_uid != os.getuid():
        return False
    return stat.S_IMODE(info.st_mode) & 0o077 == 0


def load_or_mint_receiver_token(path: Path | None = None, *, rotate: bool = False) -> str:
    """Return the receiver's persisted bearer token, minting one on first use.

    This is a local pairing secret (paste into the extension popup's
    "Receiver token" field), not an OAuth credential, so it gets a plain
    0600 file rather than :mod:`polylogue.sources.token_store`'s
    keyring-backed store. Written atomically (mkstemp + fchmod(0o600) before
    any bytes land, then ``os.replace``) so the token is never briefly
    world-readable under a permissive umask. An existing file is trusted only
    if :func:`_is_trusted_token_file` passes; otherwise it is treated as
    absent and a fresh token is minted in its place.
    """
    target = path if path is not None else browser_capture_receiver_token_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(target.name + ".lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.fchmod(lock_fd, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return _load_or_mint_receiver_token_locked(target, rotate=rotate)
    finally:
        os.close(lock_fd)


def _load_or_mint_receiver_token_locked(target: Path, *, rotate: bool) -> str:
    if not rotate and target.exists():
        if _is_trusted_token_file(target):
            existing = target.read_text(encoding="utf-8").strip()
            if existing:
                return existing
        else:
            logger.warning(
                "browser_capture.receiver_token_file_untrusted",
                path=str(target),
                action="reminting",
                reason="existing token file is not an owner-only regular file we exclusively own",
            )
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


def persist_receiver_token(token: str, path: Path | None = None) -> str:
    """Publish an explicitly configured receiver token for native pairing.

    The token is normalized once, exactly as ``load_or_mint_receiver_token``
    reads the file back, so the receiver and the paired extension hold the
    same bearer credential.
    """
    token = token.strip()
    if not token:
        raise ValueError("receiver token must not be empty")
    target = path if path is not None else browser_capture_receiver_token_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(target.with_name(target.name + ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.fchmod(lock_fd, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        atomic_replace(target, token.encode("utf-8"), mode=0o600)
    finally:
        os.close(lock_fd)
    return token


#: Entropy (raw hex chars) for the auto-minted, non-secret receiver identity.
RECEIVER_IDENTITY_HEX_CHARS = 20


def load_or_mint_receiver_identity(path: Path | None = None) -> str:
    """Return the receiver's persisted stable pairing identity, minting one on first use.

    Unlike :func:`load_or_mint_receiver_token`, this value is non-secret by
    construction (safe to advertise in ``/v1/status`` and popup UI) and is
    stored in its own file so it survives token rotation untouched. Written
    atomically (mkstemp + ``os.replace``), matching the token-minting
    pattern, though no ``fchmod`` is needed since there is nothing secret to
    protect.
    """
    target = path if path is not None else browser_capture_receiver_identity_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Concurrent first-run native hosts (parallel health checks) each saw the
    # file absent and minted their own identity; the loser returned a value
    # the file no longer held. A lock beside the file, as the token minting
    # takes, makes the check and the publish one step, so every caller
    # returns what the file holds.
    lock_fd = os.open(target.with_name(target.name + ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return _load_or_mint_receiver_identity_locked(target)
    finally:
        os.close(lock_fd)


def _load_or_mint_receiver_identity_locked(target: Path) -> str:
    if target.exists():
        existing = target.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    identity = f"rx-{secrets.token_hex(RECEIVER_IDENTITY_HEX_CHARS // 2)}"
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(identity)
        os.replace(tmp_path, target)
    except BaseException:
        with suppress(FileNotFoundError):
            tmp_path.unlink()
        raise
    return identity


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
        return persist_receiver_token(explicit_token, token_path)
    if allow_no_auth:
        logger.warning(
            "browser_capture.auth_disabled",
            reason="allow_no_auth explicitly set",
            risk="any local process can read spool/archive state and post forged captures",
        )
        return None
    return load_or_mint_receiver_token(token_path)


class CaptureConvergence(StrEnum):
    """What the spool does with an incoming capture of a resident identity.

    One provider session keeps one artifact, so a second delivery is an
    ordinary revision of the same identity rather than a conflict. Only
    ``NAME_COLLISION`` is a genuine refusal: two different sessions claiming
    one artifact name.
    """

    PUBLISH = "publish"
    DUPLICATE = "duplicate"
    SUPERSEDED = "superseded"
    NAME_COLLISION = "name_collision"


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
    content_hash: str
    capture_id: str | None
    capture_instance_id: str | None
    accepted_identities: tuple[BrowserCaptureAcceptedIdentity, ...] = ()
    convergence: CaptureConvergence = CaptureConvergence.PUBLISH

    @property
    def outcome(self) -> Literal["accepted", "noop", "superseded"]:
        if self.convergence is CaptureConvergence.DUPLICATE:
            return "noop"
        if self.convergence is CaptureConvergence.SUPERSEDED:
            return "superseded"
        return "accepted"


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


class _CaptureIdentity(Protocol):
    @property
    def provider(self) -> Provider: ...

    @property
    def provider_session_id(self) -> str: ...


def capture_artifact_path(envelope: _CaptureIdentity, spool_path: Path | None = None) -> Path:
    """Return the deterministic source artifact path for a capture identity."""
    root = spool_path if spool_path is not None else BrowserCaptureReceiverConfig.default().spool_path
    provider = _safe_token(envelope.provider.value)
    session = _safe_token(envelope.provider_session_id)
    suffix = hash_text_short(f"{envelope.provider.value}:{envelope.provider_session_id}", 12)
    return root / provider / f"{session}-{suffix}.json"


def capture_artifact_ref(envelope: _CaptureIdentity, spool_path: Path | None = None) -> str:
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


def _capture_has_any_carrier(summary: CaptureSummary) -> bool:
    return any(fact.carrier_evidence for fact in summary.attachments)


def _capture_has_invalid_or_conflicting_carrier(summary: CaptureSummary) -> bool:
    return any(fact.has_invalid_or_conflicting_carrier for fact in summary.attachments)


def _capture_carrier_conflicts(
    incoming: CaptureSummary,
    existing: CaptureSummary,
) -> bool:
    """Reject carrier bytes that contradict an existing attachment identity.

    Attachments pair by their scope, provider attachment ID, and nonempty
    provider message owner, in observed order within that group. Ownerless
    occurrences retain their declared order within the scope and ID group.
    """
    incoming_groups = _scoped_attachment_facts(incoming)
    for key, previous_group in _scoped_attachment_facts(existing).items():
        current_group = incoming_groups.get(key, [])
        if len(current_group) < len(previous_group):
            return True
        for current, previous in zip(current_group, previous_group, strict=False):
            if current.identity != previous.identity:
                return True
            if (
                current.size_bytes is not None
                and previous.size_bytes is not None
                and current.size_bytes != previous.size_bytes
            ):
                return True
            current_evidence = current.carrier_evidence
            previous_evidence = previous.carrier_evidence
            if not current_evidence:
                continue
            if current.has_invalid_or_conflicting_carrier or previous.has_invalid_or_conflicting_carrier:
                return True
            if previous_evidence and current.effective_carrier[0] != previous.effective_carrier[0]:
                return True
    return False


def _scoped_attachment_facts(summary: CaptureSummary) -> dict[tuple[str, str, str | None], list[AttachmentFact]]:
    groups: dict[tuple[str, str, str | None], list[AttachmentFact]] = {}
    for fact in summary.attachments:
        owner = fact.message_provider_id or None
        groups.setdefault((fact.scope, fact.attachment_id, owner), []).append(fact)
    return groups


def _attachment_content_enrichment(incoming: CaptureSummary, existing: CaptureSummary) -> bool:
    """Accept only a valid carrier added to an otherwise identical capture."""
    incoming_attachments = incoming.attachments
    existing_attachments = existing.attachments
    if not incoming_attachments or len(incoming_attachments) != len(existing_attachments):
        return False
    if incoming.head.provenance.model_dump(mode="json", exclude_none=True) != existing.head.provenance.model_dump(
        mode="json", exclude_none=True
    ):
        return False
    if incoming.provenance_meta_digest != existing.provenance_meta_digest:
        return False
    # The envelope and session metadata are part of the carrierless fingerprint.
    if incoming.carrierless_fingerprint != existing.carrierless_fingerprint:
        return False

    added_carrier = False
    for incoming_attachment, existing_attachment in zip(incoming_attachments, existing_attachments, strict=True):
        if incoming_attachment.identity != existing_attachment.identity:
            return False
        if (
            incoming_attachment.size_bytes is not None
            and existing_attachment.size_bytes is not None
            and incoming_attachment.size_bytes != existing_attachment.size_bytes
        ):
            return False
        incoming_evidence = incoming_attachment.carrier_evidence
        existing_evidence = existing_attachment.carrier_evidence
        if not incoming_evidence:
            if existing_evidence:
                return False
            continue
        if incoming_attachment.has_invalid_or_conflicting_carrier:
            return False
        incoming_carrier, _ = incoming_attachment.effective_carrier
        existing_carrier, existing_valid = existing_attachment.effective_carrier
        if existing_carrier is None:
            added_carrier = True
            continue
        if existing_attachment.has_invalid_or_conflicting_carrier or not existing_valid:
            return False
        if incoming_carrier != existing_carrier:
            return False
    return added_carrier


def _session_update_evidence_ms(summary: CaptureSummary) -> int | None:
    """Return a session update timestamp only when it is independent evidence.

    Adapters without a provider-side update time fill ``session.updated_at``
    from ``provenance.captured_at``.  That fallback describes when the page was
    observed, not when the session changed, so it must not participate in the
    session-timestamp ordering below.
    """
    updated_at = to_epoch_ms(summary.head.session.updated_at, numeric_unit="seconds")
    captured_at = to_epoch_ms(summary.head.provenance.captured_at, numeric_unit="seconds")
    return None if updated_at is not None and updated_at == captured_at else updated_at


def _open_readonly_sqlite(path: Path) -> sqlite3.Connection | None:
    if not path.exists():
        return None
    try:
        # Column-tolerant diagnostic lookups: the caller inspects whatever
        # schema is present, so tier schema validation stays off here.
        conn = open_readonly_connection(path, validate_schema=False)
    except sqlite3.Error:
        return None
    conn.row_factory = sqlite3.Row
    return conn


def _columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    # ``sessions.session_id`` is a generated canonical identity. SQLite's
    # table_info projection omits generated columns, while table_xinfo includes
    # both ordinary and generated columns with the same ``name`` field.
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_xinfo({table_name})").fetchall()}


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
        for optional in ("parse_error", "validation_error", "validation_status", "parsed_at_ms", "validated_at_ms"):
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
        validation_authority = raw_state_authority(
            row["parsed_at_ms"] if "parsed_at_ms" in row_keys else None,
            row["validated_at_ms"] if "validated_at_ms" in row_keys else None,
        )
        if isinstance(parse_error, str) and parse_error:
            latest_failure = parse_error
            failure_source = "raw_parse"
        elif validation_authority == "validation" and isinstance(validation_error, str) and validation_error:
            latest_failure = validation_error
            failure_source = "raw_validation"
        elif (
            validation_authority == "validation"
            and validation_status is not None
            and validation_status not in {"passed", "valid", "ok"}
        ):
            latest_failure = validation_status
            failure_source = "raw_validation"
        elif validation_authority == "ambiguous" and validation_status not in {None, "passed", "valid", "ok"}:
            latest_failure = "raw validation and parse timestamps are indeterminate"
            failure_source = "raw_state_order"
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
    try:
        index_path = resolve_active_index_path(archive_root)
    except ArchiveLocationError:
        # Archive state is a best-effort capture acknowledgement. A malformed
        # active-generation pointer must not turn a receiver GET into a 500 or
        # make us consult the conventional shadow index instead.
        return _IndexArchiveLookup()
    conn = _open_readonly_sqlite(index_path)
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


class BrowserCaptureSpoolConflictError(RuntimeError):
    """Raised when an existing spool name cannot safely admit a capture.

    A spool artifact is an immutable envelope once published.  A malformed
    artifact or an artifact belonging to a different provider/session is not
    evidence that the incoming capture is a duplicate; replacing it would
    destroy the only durable evidence of the conflict.
    """


# Serialize admission and publication across request threads; the file lock
# below supplies the same original custody across receiver processes.
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


def _capture_is_newer_or_richer(incoming: CaptureSummary, existing: CaptureSummary) -> bool:
    """Prevent a stale, smaller snapshot from replacing a richer spool item.

    Two things are deliberately NOT freshness evidence here. A provider-native
    payload arriving over a DOM fallback is a fidelity improvement even when
    it lists fewer turns -- the archive boundary
    (``archive_tiers.ingest_precedence.browser_capture_precedence``) already
    admits exactly that, so discarding it at the spool made the two rules
    disagree and retained the lower-fidelity content permanently. And
    ``provenance.captured_at`` is an observation by one extension instance,
    not a session revision (see ``CaptureSummary.dedup_content_hash``): with
    two instances whose clocks are skewed it must not veto a strictly newer
    provider revision or a richer turn set.
    """
    incoming_updated = _session_update_evidence_ms(incoming)
    existing_updated = _session_update_evidence_ms(existing)
    incoming_captured = to_epoch_ms(incoming.head.provenance.captured_at, numeric_unit="seconds")
    existing_captured = to_epoch_ms(existing.head.provenance.captured_at, numeric_unit="seconds")
    incoming_turns = incoming.turn_count
    existing_turns = existing.turn_count
    native_over_fallback = incoming.has_native_provider_payload and not existing.has_native_provider_payload
    if incoming_turns < existing_turns and not native_over_fallback:
        return False
    if incoming_turns > existing_turns or native_over_fallback:
        return True
    if existing_updated is not None and incoming_updated is not None:
        if incoming_updated > existing_updated:
            return True
        if incoming_updated < existing_updated:
            return False
    incoming_observation = incoming.head.provenance
    existing_observation = existing.head.provenance
    if (
        incoming_observation.extension_instance_id is not None
        and incoming_observation.extension_instance_id == existing_observation.extension_instance_id
        and incoming_observation.acquisition_sequence is not None
        and existing_observation.acquisition_sequence is not None
    ):
        # These counters share one durable owner. Other instances' clocks and
        # counters cannot establish ordering within this acquisition history.
        return incoming_observation.acquisition_sequence > existing_observation.acquisition_sequence
    if existing_captured is not None and incoming_captured is not None and incoming_captured < existing_captured:
        return False
    # An absent update timestamp is unknown, not a change from an existing
    # provider timestamp.  Only compare that field when the incoming capture
    # carries independent session-side evidence.
    return (
        (incoming_updated is not None and incoming_updated != existing_updated)
        or incoming_captured != existing_captured
        or incoming_turns != existing_turns
    )


def _same_instance_acquisition_advances(incoming: CaptureSummary, existing: CaptureSummary) -> bool:
    incoming_observation = incoming.head.provenance
    existing_observation = existing.head.provenance
    return (
        incoming.provider is existing.provider
        and incoming.provider_session_id == existing.provider_session_id
        and incoming_observation.extension_instance_id is not None
        and incoming_observation.extension_instance_id == existing_observation.extension_instance_id
        and incoming_observation.acquisition_sequence is not None
        and existing_observation.acquisition_sequence is not None
        and incoming_observation.acquisition_sequence > existing_observation.acquisition_sequence
    )


def capture_convergence(incoming: CaptureSummary, existing: CaptureSummary) -> CaptureConvergence:
    """Decide what a resident artifact of the same name does to an incoming capture.

    The whole spool-admission rule for a destination that is already occupied,
    so a caller that must predict admission without writing — a restoration
    rehearsal — asks this rather than restating it.
    """
    if incoming.provider is not existing.provider or incoming.provider_session_id != existing.provider_session_id:
        return CaptureConvergence.NAME_COLLISION
    if _capture_has_invalid_or_conflicting_carrier(incoming):
        return CaptureConvergence.SUPERSEDED
    if existing.dedup_content_hash == incoming.dedup_content_hash:
        return CaptureConvergence.DUPLICATE
    if _attachment_content_enrichment(incoming, existing):
        return CaptureConvergence.PUBLISH
    # A carrier that contradicts a resident attachment is not freshness
    # evidence.  Otherwise ordinary newer/richer snapshots (for example a new
    # turn carrying an attachment) retain their existing admission semantics.
    if _capture_has_any_carrier(incoming) and _capture_carrier_conflicts(incoming, existing):
        return CaptureConvergence.SUPERSEDED
    if not _capture_is_newer_or_richer(incoming, existing):
        return CaptureConvergence.SUPERSEDED
    return CaptureConvergence.PUBLISH


def summarize_capture_envelope(envelope: BrowserCaptureEnvelope) -> CaptureSummary:
    """Summarize an in-memory envelope through the streamed admission reader."""
    return summarize_capture_stream(io.BytesIO(_envelope_bytes(envelope)))


def _envelope_bytes(envelope: BrowserCaptureEnvelope) -> bytes:
    payload = envelope.model_dump(mode="json", exclude_none=True)
    # Raw provider mapping order remains authoritative for canonical replay.
    # Streamed admission fingerprints normalize object order independently.
    return dumps_bytes(payload, indent=2) + b"\n"


def write_capture_envelope(
    envelope: BrowserCaptureEnvelope,
    *,
    spool_path: Path | None = None,
) -> BrowserCaptureWriteResult:
    """Serialize an envelope and admit it through the streamed spool route."""
    return write_capture_envelope_bytes(_envelope_bytes(envelope), spool_path=spool_path)


def write_capture_envelope_bytes(
    raw: bytes,
    *,
    spool_path: Path | None = None,
) -> BrowserCaptureWriteResult:
    """Admit envelope bytes already in memory through the streamed spool route.

    The bytes are staged exactly as the HTTP receiver stages a request body,
    then admitted by :func:`admit_staged_capture`; the published artifact is
    the byte sequence that was acquired.
    """
    root = spool_path if spool_path is not None else BrowserCaptureReceiverConfig.default().spool_path
    staged = stage_capture_body(io.BytesIO(raw).read, len(raw), spool_root=root)
    try:
        try:
            summary = summarize_capture_file(staged.path)
        except CaptureEnvelopeError as exc:
            raise BrowserCaptureSpoolConflictError("capture envelope is malformed") from exc
        return admit_staged_capture(staged, summary, spool_path=root)
    finally:
        staged.discard()


def _accepted_identities(
    summary: CaptureSummary,
    root: Path,
) -> tuple[BrowserCaptureAcceptedIdentity, ...]:
    """Project the message identities one retained capture artifact carries."""
    session_ref = f"{_capture_origin(summary.provider.value)}:{summary.provider_session_id}"
    artifact_ref = capture_artifact_ref(summary, root)
    return tuple(
        BrowserCaptureAcceptedIdentity(
            session_ref=session_ref,
            message_ref=f"{session_ref}:n:{turn_id}",
            evidence_ref=f"{artifact_ref}#message:{turn_id}",
            fidelity=fidelity,
            adapter_version=summary.head.provenance.adapter_version,
        )
        for turn_id, fidelity in summary.turn_identities
    )


def admit_staged_capture(
    staged: StagedCapture,
    summary: CaptureSummary,
    *,
    spool_path: Path | None = None,
) -> BrowserCaptureWriteResult:
    """Publish a staged capture into the spool, or keep the resident artifact.

    ``summary`` is the streamed summary of ``staged``. A resident artifact of
    the same name is summarized by the same streamed reader under the spool
    lock, so neither side is held whole. Admission and publication are
    serialized across writers. Staging reserves the actual physical storage;
    valid captures have no count quota. The caller discards ``staged``
    afterwards; a published file has already been moved away.
    """
    root = spool_path if spool_path is not None else BrowserCaptureReceiverConfig.default().spool_path
    target = capture_artifact_path(summary, root)
    convergence = CaptureConvergence.PUBLISH
    with _SPOOL_WRITE_LOCK, _spool_file_lock(root):
        replaced = target.exists()
        if replaced:
            try:
                existing = summarize_capture_file(target)
            except (OSError, CaptureEnvelopeError) as exc:
                raise BrowserCaptureSpoolConflictError(
                    f"existing capture artifact is unreadable or malformed: {target.name}"
                ) from exc
            convergence = capture_convergence(summary, existing)
            if convergence is CaptureConvergence.NAME_COLLISION:
                raise BrowserCaptureSpoolConflictError(f"capture artifact name collision for {target.name}")
            refresh_duplicate = (
                convergence is CaptureConvergence.DUPLICATE
                and _same_instance_acquisition_advances(summary, existing)
                and _capture_is_newer_or_richer(summary, existing)
            )
            if convergence is not CaptureConvergence.PUBLISH and not refresh_duplicate:
                # Both receipts identify the bytes and revision that stay.
                # A previous rename may have succeeded before its directory
                # barrier failed. Settle that path before acknowledging reuse.
                sync_directory(root)
                sync_directory(target.parent)
                return BrowserCaptureWriteResult(
                    provider=summary.provider.value,
                    provider_session_id=summary.provider_session_id,
                    path=target,
                    artifact_ref=capture_artifact_ref(summary, root),
                    bytes_written=target.stat().st_size,
                    replaced=True,
                    deduplicated=True,
                    dedup_content_hash=existing.dedup_content_hash,
                    content_hash=hash_file(target),
                    capture_id=existing.capture_id,
                    capture_instance_id=summary.head.provenance.extension_instance_id,
                    # The retained artifact is `existing`, so the identities
                    # this delivery acknowledges are its identities. Echoing
                    # the rejected incoming envelope told the extension a
                    # branch was captured whose messages were never written
                    # and are not in the spool.
                    accepted_identities=_accepted_identities(existing, root),
                    convergence=convergence,
                )
        target.parent.mkdir(parents=True, exist_ok=True)
        # Establish the provider entry in the durable spool root first.
        sync_directory(root)
        os.replace(staged.path, target)
        sync_directory(target.parent)
    return BrowserCaptureWriteResult(
        provider=summary.provider.value,
        provider_session_id=summary.provider_session_id,
        path=target,
        artifact_ref=capture_artifact_ref(summary, root),
        bytes_written=target.stat().st_size,
        replaced=replaced,
        deduplicated=convergence is CaptureConvergence.DUPLICATE,
        dedup_content_hash=summary.dedup_content_hash,
        content_hash=staged.sha256,
        capture_id=summary.capture_id,
        capture_instance_id=summary.head.provenance.extension_instance_id,
        accepted_identities=_accepted_identities(summary, root),
        convergence=convergence,
    )


def _capture_origin(provider: str) -> str:
    """Reduce provider-wire names to the public archive origin namespace."""
    return {"chatgpt": "chatgpt-export", "claude": "claude-ai-export", "claude-ai": "claude-ai-export"}.get(
        provider, "unknown-export"
    )


def receiver_identity(config: BrowserCaptureReceiverConfig) -> str:
    """Return a stable, non-secret identity for one paired receiver.

    Persisted independently of the bearer token and the spool path
    (polylogue-jlme.5): identity is minted once per archive root
    (:func:`load_or_mint_receiver_identity`) and then read back on every
    call, so it survives daemon restarts, spool relocation, *and* ordinary
    token rotation. The pre-jlme.5 design hashed the auth token to derive
    this id, which meant every token rotation silently changed a paired
    extension's trusted identity and forced an unnecessary re-pair — exactly
    the failure mode ``gnie``'s automatic credential refresh depends on not
    happening.
    """
    # `config` is kept as the call-site signature (every caller already has
    # one in hand) but no longer feeds the identity itself.
    _ = config
    return load_or_mint_receiver_identity()


#: Domain separator for a receiver attestation MAC, so a proof can never be
#: confused with any other HMAC keyed by the same bearer.
RECEIVER_ATTESTATION_DOMAIN = "polylogue-browser-capture-receiver-attestation/v1"


def receiver_attestation_proof(secret: str, receiver_id: str, challenge: str) -> str:
    """Return the proof that the holder of ``secret`` answered ``challenge``.

    HMAC-SHA256 keyed by the receiver bearer over the domain, the receiver
    identity, and the caller's fresh challenge. A process that does not hold
    the bearer cannot produce it, and the proof reveals nothing about the
    bearer, so a client can authenticate a loopback receiver before it
    releases or presents the durable credential.
    """
    message = f"{RECEIVER_ATTESTATION_DOMAIN}\n{receiver_id}\n{challenge}".encode()
    digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


RECEIVER_STATUS_REQUEST_DOMAIN = "polylogue-browser-capture-status-request/v1"
RECEIVER_STATUS_RESPONSE_DOMAIN = "polylogue-browser-capture-status-response/v1"


def receiver_status_proof(secret: str, receiver_id: str, challenge: str, *, payload_sha256: str | None = None) -> str:
    """Authenticate a status request or its exact staged JSON response bytes."""
    domain = RECEIVER_STATUS_REQUEST_DOMAIN if payload_sha256 is None else RECEIVER_STATUS_RESPONSE_DOMAIN
    message = f"{domain}\n{receiver_id}\n{challenge}"
    if payload_sha256 is not None:
        message += f"\n{payload_sha256}"
    digest = hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def attest_receiver(config: BrowserCaptureReceiverConfig, challenge: str) -> str | None:
    """Answer an attestation challenge, or ``None`` when auth is disabled."""
    if config.auth_token is None:
        return None
    return receiver_attestation_proof(config.auth_token, receiver_identity(config), challenge)


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
            raw_capture_id, raw_updated_at = read_capture_state_fields(path)
            capture_id = raw_capture_id if isinstance(raw_capture_id, str) else None
            updated_at = raw_updated_at if isinstance(raw_updated_at, str) else None
            spooled_updated_at_ms = to_epoch_ms(updated_at, numeric_unit="seconds")
        except (OSError, ValueError):
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


def backfill_checkpoint_root(spool_path: Path | None = None) -> Path:
    """Locate original checkpoint inputs retained for digest-bound inspection.

    New checkpoints use CaptureJobRegistry custody. Existing mirror bytes may
    contain unique acquired payloads and delivery metadata, so their directory
    remains an ordinary input to the registry's orphan census and raw reader.
    """
    root = spool_path if spool_path is not None else BrowserCaptureReceiverConfig.default().spool_path
    return root / BACKFILL_CHECKPOINT_DIRNAME


__all__ = [
    "BACKFILL_CHECKPOINT_DIRNAME",
    "BROWSER_CAPTURE_ALLOW_NO_AUTH_ENV",
    "RECEIVER_ATTESTATION_DOMAIN",
    "RECEIVER_IDENTITY_HEX_CHARS",
    "RECEIVER_TOKEN_ENTROPY_BYTES",
    "BrowserCaptureReceiverConfig",
    "BrowserCaptureWriteResult",
    "BrowserCaptureSpoolConflictError",
    "CaptureConvergence",
    "admit_staged_capture",
    "attest_receiver",
    "backfill_checkpoint_root",
    "capture_artifact_ref",
    "capture_convergence",
    "capture_response_id",
    "_is_extension_origin_pattern",
    "capture_artifact_path",
    "existing_capture_state",
    "load_or_mint_receiver_identity",
    "load_or_mint_receiver_token",
    "receiver_attestation_proof",
    "receiver_identity",
    "receiver_status_payload",
    "resolve_receiver_auth_token",
    "summarize_capture_envelope",
    "write_capture_envelope",
    "write_capture_envelope_bytes",
]
