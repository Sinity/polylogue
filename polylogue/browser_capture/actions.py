"""Receiver-authoritative transport for provider-neutral browser actions."""

from __future__ import annotations

import base64
import binascii
import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any, TypeVar, cast
from urllib.parse import urlparse
from uuid import uuid4

import orjson

from polylogue.browser_capture.models import (
    BrowserActionAttachment,
    BrowserActionEvent,
    BrowserActionIntent,
    BrowserActionReceipt,
    BrowserActionReconcileRequest,
    BrowserActionRequest,
    BrowserActionUpdateRequest,
)
from polylogue.paths import browser_capture_spool_root

ACTION_DIRNAME = "browser-actions"
ACTION_MAX_FILES = 5_000
ACTION_ATTACHMENT_MAX_BYTES = 16 * 1024 * 1024
ACTION_TOTAL_ATTACHMENT_MAX_BYTES = 16 * 1024 * 1024
ACTION_LEASE_SECONDS = 180
ACTION_EVENT_LIMIT = 100
_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")
_ACTION_LOCK = RLock()
_T = TypeVar("_T")


class BrowserActionConflictError(RuntimeError):
    """Raised when stable action identity or state conflicts."""


class BrowserActionLeaseError(RuntimeError):
    """Raised when an action is mutated by a non-owner."""


class BrowserActionQuotaError(RuntimeError):
    """Raised when action or attachment storage exceeds its bound."""


class BrowserActionStateError(RuntimeError):
    """Raised when durable action state is unreadable."""


def _safe_token(value: str) -> str:
    token = _SAFE_TOKEN.sub("-", value.strip()).strip("-.")
    if not token:
        raise ValueError("browser action token must not be empty")
    return token[:160]


def action_root(spool_path: Path | None = None) -> Path:
    return (spool_path if spool_path is not None else browser_capture_spool_root()) / ACTION_DIRNAME


def _action_dir(root: Path, action_id: str) -> Path:
    return root / _safe_token(action_id)


def _action_path(root: Path, action_id: str) -> Path:
    return _action_dir(root, action_id) / "action.json"


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_action(root: Path, action: BrowserActionIntent) -> None:
    payload = orjson.dumps(
        action.model_dump(mode="json", exclude_none=True),
        option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
    )
    _atomic_write(_action_path(root, action.action_id), payload + b"\n")


def _read_action(path: Path) -> BrowserActionIntent | None:
    try:
        return BrowserActionIntent.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise BrowserActionStateError(f"corrupt browser action state: {path.name}") from exc


def _serialized(function: Callable[..., _T]) -> Callable[..., _T]:
    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> _T:
        with _ACTION_LOCK:
            root = action_root(cast(Path | None, kwargs.get("spool_path")))
            root.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(root / ".queue.lock", os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                return function(*args, **kwargs)
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    return cast(Callable[..., _T], guarded)


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.isoformat()


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _event(
    action: BrowserActionIntent,
    kind: str,
    phase: str,
    *,
    detail: str | None = None,
    owner: str | None = None,
    retry_after_seconds: int | None = None,
) -> None:
    action.events = [
        *action.events,
        BrowserActionEvent(
            event_id=uuid4().hex,
            at=_iso(_now()),
            kind=kind,
            phase=phase,
            detail=detail,
            owner_instance_id=owner,
            retry_after_seconds=retry_after_seconds,
        ),
    ][-ACTION_EVENT_LIMIT:]


def browser_action_capabilities() -> dict[str, object]:
    """Return exact supported selections; clients must not infer fallbacks."""
    return {
        "chatgpt": {
            "operations": ["conversation.create", "conversation.reply"],
            "submit_policies": ["stage_only", "submit_once"],
            "attachments": True,
            "max_attachment_bytes": ACTION_ATTACHMENT_MAX_BYTES,
            "max_total_attachment_bytes": ACTION_TOTAL_ATTACHMENT_MAX_BYTES,
            "presentations": [
                {
                    "surface": "chat",
                    "model_slug": "gpt-5-6-pro",
                    "model_label": "GPT-5.6 Sol",
                    "effort_label": "Pro",
                    "project_targeting": "provider_route",
                }
            ],
        },
        "claude": {
            "operations": [],
            "submit_policies": [],
            "attachments": False,
            "presentations": [],
            "reason": "provider_action_adapter_not_implemented",
        },
    }


def _validate_capability(request: BrowserActionRequest) -> None:
    provider = browser_action_capabilities().get(request.provider)
    if not isinstance(provider, dict) or request.operation not in provider.get("operations", []):
        raise BrowserActionConflictError("requested provider operation is unsupported")
    presentations = provider.get("presentations", [])
    requested = request.presentation.model_dump(mode="json")
    if not any(
        isinstance(item, dict) and all(item.get(key) == value for key, value in requested.items())
        for item in presentations
    ):
        raise BrowserActionConflictError("requested provider presentation is unsupported")
    if request.submit_policy not in provider.get("submit_policies", []):
        raise BrowserActionConflictError("requested submit policy is unsupported")
    matching_presentation = next(
        (
            item
            for item in presentations
            if isinstance(item, dict) and all(item.get(key) == value for key, value in requested.items())
        ),
        None,
    )
    if request.target.project_ref and (
        not isinstance(matching_presentation, dict) or not matching_presentation.get("project_targeting")
    ):
        raise BrowserActionConflictError("requested provider presentation does not support project targeting")
    if request.provider == "chatgpt" and request.target.conversation_url:
        parsed = urlparse(request.target.conversation_url)
        if parsed.scheme != "https" or parsed.hostname != "chatgpt.com":
            raise BrowserActionConflictError("ChatGPT action target URL must use https://chatgpt.com")
        if request.operation == "conversation.reply" and request.target.conversation_id not in parsed.path.split("/"):
            raise BrowserActionConflictError("reply target URL does not contain the requested conversation id")


def _request_sha256(request: BrowserActionRequest) -> str:
    payload = request.model_dump(mode="json", exclude={"action_id", "idempotency_key"})
    return hashlib.sha256(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)).hexdigest()


def list_actions(*, spool_path: Path | None = None) -> list[BrowserActionIntent]:
    root = action_root(spool_path)
    if not root.exists():
        return []
    actions = [_read_action(path) for path in root.glob("*/action.json")]
    return sorted(
        (action for action in actions if action is not None),
        key=lambda action: (action.created_at, action.action_id),
        reverse=True,
    )


def get_action(action_id: str, *, spool_path: Path | None = None) -> BrowserActionIntent | None:
    return _read_action(_action_path(action_root(spool_path), action_id))


@_serialized
def enqueue_action(
    request: BrowserActionRequest,
    *,
    receiver_id: str,
    spool_path: Path | None = None,
) -> BrowserActionIntent:
    _validate_capability(request)
    root = action_root(spool_path)
    request_sha256 = _request_sha256(request)
    if request.action_id and _safe_token(request.action_id) != request.action_id:
        raise ValueError("browser action id must contain only letters, digits, dot, underscore, or hyphen")
    action_id = request.action_id or uuid4().hex
    if action_id == "capabilities":
        raise ValueError("browser action id is reserved by the receiver route contract")
    idempotency_key = request.idempotency_key or action_id
    for existing in list_actions(spool_path=spool_path):
        if existing.action_id == action_id or existing.idempotency_key == idempotency_key:
            if existing.request_sha256 == request_sha256:
                return existing
            raise BrowserActionConflictError("browser action identity already exists with different input")
    if sum(1 for _ in root.glob("*/action.json")) >= ACTION_MAX_FILES:
        raise BrowserActionQuotaError("browser action file quota exceeded")

    attachments: list[BrowserActionAttachment] = []
    decoded: list[tuple[BrowserActionAttachment, bytes]] = []
    total = 0
    attachment_names: set[str] = set()
    for index, item in enumerate(request.attachments, start=1):
        if item.name in attachment_names:
            raise ValueError(f"duplicate browser action attachment name: {item.name!r}")
        attachment_names.add(item.name)
        try:
            content = base64.b64decode(item.content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"invalid base64 for browser action attachment {item.name!r}") from exc
        if len(content) > ACTION_ATTACHMENT_MAX_BYTES:
            raise BrowserActionQuotaError("browser action attachment byte quota exceeded")
        total += len(content)
        if total > ACTION_TOTAL_ATTACHMENT_MAX_BYTES:
            raise BrowserActionQuotaError("browser action total attachment byte quota exceeded")
        digest = hashlib.sha256(content).hexdigest()
        attachment = BrowserActionAttachment(
            attachment_id=f"a{index}-{digest[:16]}",
            name=item.name,
            mime_type=item.mime_type,
            size_bytes=len(content),
            sha256=digest,
        )
        attachments.append(attachment)
        decoded.append((attachment, content))

    now = _now()
    action = BrowserActionIntent(
        action_id=action_id,
        idempotency_key=idempotency_key,
        request_sha256=request_sha256,
        receiver_id=receiver_id,
        provider=request.provider,
        operation=request.operation,
        target=request.target,
        text=request.text,
        attachments=attachments,
        presentation=request.presentation,
        submit_policy=request.submit_policy,
        created_at=_iso(now),
        updated_at=_iso(now),
    )
    _event(action, "created", "queued")
    try:
        for attachment, content in decoded:
            _atomic_write(_action_dir(root, action_id) / "attachments" / attachment.attachment_id, content)
        _write_action(root, action)
    except Exception:
        for path in sorted(_action_dir(root, action_id).glob("**/*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        with suppress(FileNotFoundError):
            _action_dir(root, action_id).rmdir()
        raise
    return action


@_serialized
def claim_action(
    owner_instance_id: str,
    *,
    spool_path: Path | None = None,
    lease_seconds: int = ACTION_LEASE_SECONDS,
) -> BrowserActionIntent | None:
    root = action_root(spool_path)
    now = _now()
    actions = list_actions(spool_path=spool_path)
    for action in actions:
        expiry = _parse(action.lease_expires_at)
        if action.status == "submit_intent" and (expiry is None or expiry <= now):
            action.status = "outcome_unknown"
            action.phase = "outcome_unknown"
            action.last_error = "submit executor lease expired after durable submit intent"
            action.lease_owner = None
            action.lease_expires_at = None
            action.updated_at = _iso(now)
            _event(action, "outcome_unknown", action.phase, detail=action.last_error)
            _write_action(root, action)
        elif action.status in {"leased", "preparing"} and (expiry is None or expiry <= now):
            action.status = "queued"
            action.phase = "lease_expired_before_submit"
            action.lease_owner = None
            action.lease_expires_at = None
            action.updated_at = _iso(now)
            _event(action, "requeued", action.phase)
            _write_action(root, action)
    if any(action.status in {"leased", "preparing", "submit_intent"} for action in actions):
        return None
    queued = sorted(
        (action for action in list_actions(spool_path=spool_path) if action.status == "queued"),
        key=lambda action: (action.created_at, action.action_id),
    )
    if not queued:
        return None
    action = queued[0]
    action.status = "leased"
    action.phase = "leased"
    action.lease_owner = owner_instance_id
    action.lease_expires_at = _iso(now + timedelta(seconds=max(30, lease_seconds)))
    action.updated_at = _iso(now)
    _event(action, "leased", action.phase, owner=owner_instance_id)
    _write_action(root, action)
    return action


def _validate_receipt(
    action: BrowserActionIntent,
    receipt: BrowserActionReceipt | None,
    *,
    submitted: bool,
    extension_owner: str | None = None,
) -> BrowserActionReceipt:
    if receipt is None:
        raise BrowserActionConflictError("terminal browser action outcome requires an exact receipt")
    if receipt.action_id != action.action_id or receipt.receiver_id != action.receiver_id:
        raise BrowserActionConflictError("browser action receipt identity mismatch")
    if extension_owner is not None and receipt.extension_instance_id != extension_owner:
        raise BrowserActionConflictError("browser action receipt extension identity mismatch")
    if (
        (receipt.observed_surface or "").lower() != action.presentation.surface
        or receipt.observed_model != action.presentation.model_label
        or receipt.observed_effort != action.presentation.effort_label
    ):
        raise BrowserActionConflictError("browser action receipt presentation mismatch")
    expected_project = re.search(r"g-p-[A-Za-z0-9]+", action.target.project_ref or "")
    if expected_project and receipt.observed_project_ref != expected_project.group(0):
        raise BrowserActionConflictError("browser action receipt project mismatch")
    if submitted and not all(
        (receipt.provider_conversation_id, receipt.provider_conversation_url, receipt.provider_turn_id)
    ):
        raise BrowserActionConflictError("submitted browser action receipt is incomplete")
    if submitted and not receipt.provider_evidence:
        raise BrowserActionConflictError("submitted browser action receipt lacks provider evidence")
    return receipt


@_serialized
def update_action(
    action_id: str,
    update: BrowserActionUpdateRequest,
    *,
    spool_path: Path | None = None,
) -> BrowserActionIntent | None:
    root = action_root(spool_path)
    action = _read_action(_action_path(root, action_id))
    if action is None:
        return None
    if action.status in {"outcome_unknown", "drafted", "submitted", "blocked", "failed", "cancelled"}:
        same_success = action.status in {"drafted", "submitted"} and (
            update.outcome == action.status and update.receipt == action.receipt
        )
        same_failure = action.status in {"outcome_unknown", "blocked", "failed"} and (
            update.outcome == action.failure_kind and update.detail == action.last_error
        )
        if same_success or same_failure:
            return action
        raise BrowserActionConflictError("browser action already has a different terminal outcome")
    if action.lease_owner != update.owner_instance_id:
        raise BrowserActionLeaseError("browser action update requires the current lease owner")
    now = _now()
    action.phase = update.phase
    action.updated_at = _iso(now)
    action.retry_after_seconds = update.retry_after_seconds
    action.last_error = update.detail if update.outcome not in {"progress", "drafted", "submitted"} else None
    if update.outcome in {"drafted", "submitted"}:
        _validate_receipt(
            action,
            update.receipt,
            submitted=update.outcome == "submitted",
            extension_owner=update.owner_instance_id,
        )
    if update.outcome == "progress":
        action.status = "submit_intent" if update.phase == "submit_intent" else "preparing"
        if update.phase == "submit_intent":
            action.submit_intent_at = _iso(now)
        action.lease_expires_at = _iso(now + timedelta(seconds=ACTION_LEASE_SECONDS))
    elif update.outcome in {"drafted", "submitted"}:
        action.status = "drafted" if update.outcome == "drafted" else "submitted"
        action.receipt = update.receipt
        action.lease_owner = None
        action.lease_expires_at = None
    elif update.outcome == "outcome_unknown":
        action.status = "outcome_unknown"
        action.failure_kind = "outcome_unknown"
        action.lease_owner = None
        action.lease_expires_at = None
    elif update.outcome in {
        "provider_warning",
        "rate_limited",
        "safety_locked",
        "auth_challenge",
        "capability_mismatch",
        "provider_drift",
    }:
        action.status = "blocked"
        action.failure_kind = update.outcome
        action.lease_owner = None
        action.lease_expires_at = None
    else:
        action.status = "failed"
        action.failure_kind = update.outcome
        action.lease_owner = None
        action.lease_expires_at = None
    _event(
        action,
        update.outcome,
        update.phase,
        detail=update.detail,
        owner=update.owner_instance_id,
        retry_after_seconds=update.retry_after_seconds,
    )
    _write_action(root, action)
    return action


@_serialized
def reconcile_action(
    action_id: str,
    request: BrowserActionReconcileRequest,
    *,
    spool_path: Path | None = None,
) -> BrowserActionIntent | None:
    root = action_root(spool_path)
    action = _read_action(_action_path(root, action_id))
    if action is None:
        return None
    if action.status != "outcome_unknown":
        raise BrowserActionConflictError("only an outcome_unknown browser action can be reconciled")
    _validate_receipt(action, request.receipt, submitted=request.resolution == "submitted")
    action.status = request.resolution
    action.phase = f"reconciled_{request.resolution}"
    action.receipt = request.receipt
    action.failure_kind = None
    action.last_error = None
    action.updated_at = _iso(_now())
    _event(action, "reconciled", action.phase, detail=request.detail)
    _write_action(root, action)
    return action


def read_action_attachment(
    action_id: str,
    attachment_id: str,
    *,
    spool_path: Path | None = None,
) -> tuple[BrowserActionAttachment, bytes] | None:
    root = action_root(spool_path)
    action = _read_action(_action_path(root, action_id))
    if action is None:
        return None
    attachment = next((item for item in action.attachments if item.attachment_id == attachment_id), None)
    if attachment is None:
        return None
    content = (_action_dir(root, action_id) / "attachments" / attachment.attachment_id).read_bytes()
    if len(content) != attachment.size_bytes or hashlib.sha256(content).hexdigest() != attachment.sha256:
        raise BrowserActionConflictError("browser action attachment integrity mismatch")
    return attachment, content


__all__ = [
    "ACTION_ATTACHMENT_MAX_BYTES",
    "ACTION_TOTAL_ATTACHMENT_MAX_BYTES",
    "BrowserActionConflictError",
    "BrowserActionLeaseError",
    "BrowserActionQuotaError",
    "BrowserActionStateError",
    "action_root",
    "browser_action_capabilities",
    "claim_action",
    "enqueue_action",
    "get_action",
    "list_actions",
    "read_action_attachment",
    "reconcile_action",
    "update_action",
]
