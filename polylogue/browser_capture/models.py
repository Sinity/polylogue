"""Typed browser-capture envelope shared by receiver and parser."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.core.json import is_json_document, json_document

BROWSER_CAPTURE_KIND: Literal["browser_llm_session"] = "browser_llm_session"
BROWSER_CAPTURE_SCHEMA_VERSION: Literal[1] = 1
BROWSER_CAPTURE_TRANSPORT_SOURCE: Literal["browser-extension"] = "browser-extension"
BROWSER_CAPTURE_RECEIVER: Literal["polylogue-browser-capture"] = "polylogue-browser-capture"
BROWSER_CAPTURE_API_SCHEMA: Literal["polylogue-browser-capture/v1"] = "polylogue-browser-capture/v1"
BROWSER_CAPTURE_EXTENSION_ORIGIN_WILDCARD: Literal["chrome-extension://*"] = "chrome-extension://*"
BrowserCaptureArchiveLifecycle = Literal["missing", "spooled_only", "ingest_pending", "archived", "stale", "failed"]
BrowserCaptureSessionKind = Literal["standard", "temporary"]


class BrowserCaptureAttachment(BaseModel):
    """Attachment reference observed in a browser-hosted LLM session."""

    provider_attachment_id: str
    message_provider_id: str | None = None
    name: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    url: str | None = None
    extracted_content: str | None = None
    inline_base64: str | None = None
    content_base64: str | None = None
    data: str | None = None
    provider_meta: dict[str, object] = Field(default_factory=dict)

    @field_validator("provider_meta", mode="before")
    @classmethod
    def coerce_provider_meta(cls, value: object) -> dict[str, object]:
        return dict(json_document(value))


class BrowserCaptureTurn(BaseModel):
    """Provider-neutral turn observed from the page."""

    provider_turn_id: str
    role: Role
    text: str | None = None
    timestamp: str | None = None
    ordinal: int = 0
    parent_turn_id: str | None = None
    attachments: list[BrowserCaptureAttachment] = Field(default_factory=list)
    provider_meta: dict[str, object] = Field(default_factory=dict)

    @field_validator("provider_meta", mode="before")
    @classmethod
    def coerce_provider_meta(cls, value: object) -> dict[str, object]:
        return dict(json_document(value))

    @field_validator("role", mode="before")
    @classmethod
    def coerce_role(cls, value: object) -> Role:
        if isinstance(value, Role):
            return value
        return Role.normalize(str(value) if value is not None else "unknown")

    @model_validator(mode="after")
    def require_content(self) -> BrowserCaptureTurn:
        if (self.text is None or not self.text.strip()) and not self.attachments:
            raise ValueError("browser capture turn must include text or attachments")
        return self


class BrowserCaptureInterruption(BaseModel):
    """A source-declared interval during which the capture adapter was not observing the page.

    Extensions can legitimately detect and report their own non-observation
    windows (the host was suspended, the tab was backgrounded, auth expired
    mid-session, etc). This is distinct from ordinary conversational silence,
    which leaves no signal in the archive at all — a declared interruption is
    positive evidence of a coverage gap, not an absence of evidence.
    """

    started_at: str
    ended_at: str
    reason: str


class BrowserCaptureProvenance(BaseModel):
    """How and where the capture was observed."""

    source_url: str
    page_title: str | None = None
    captured_at: str
    extension_id: str | None = None
    extension_instance_id: str | None = Field(default=None, min_length=1, max_length=128)
    browser_profile: str | None = None
    adapter_name: str
    adapter_version: str | None = None
    capture_mode: Literal["snapshot", "tail"] = "snapshot"
    capture_interruption: BrowserCaptureInterruption | None = None
    provider_meta: dict[str, object] = Field(default_factory=dict)

    @field_validator("provider_meta", mode="before")
    @classmethod
    def coerce_provider_meta(cls, value: object) -> dict[str, object]:
        return dict(json_document(value))


class BrowserCaptureSession(BaseModel):
    """A browser-visible provider session."""

    provider: Provider
    provider_session_id: str
    session_kind: BrowserCaptureSessionKind = "standard"
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    model: str | None = None
    turns: list[BrowserCaptureTurn] = Field(default_factory=list)
    attachments: list[BrowserCaptureAttachment] = Field(default_factory=list)
    provider_meta: dict[str, object] = Field(default_factory=dict)

    @field_validator("provider_meta", mode="before")
    @classmethod
    def coerce_provider_meta(cls, value: object) -> dict[str, object]:
        return dict(json_document(value))

    @field_validator("provider", mode="before")
    @classmethod
    def coerce_provider(cls, value: object) -> Provider:
        if isinstance(value, Provider):
            return value
        return Provider.from_string(str(value) if value is not None else None)

    @field_validator("session_kind", mode="before")
    @classmethod
    def coerce_session_kind(cls, value: object) -> BrowserCaptureSessionKind:
        if value in ("temporary", True):
            return "temporary"
        return "standard"

    @model_validator(mode="after")
    def require_turns(self) -> BrowserCaptureSession:
        if not self.turns:
            raise ValueError("browser capture session must include at least one turn")
        return self


class BrowserCaptureEnvelope(BaseModel):
    """Source artifact posted by the browser extension to Polylogue."""

    polylogue_capture_kind: Literal["browser_llm_session"] = BROWSER_CAPTURE_KIND
    schema_version: Literal[1] = BROWSER_CAPTURE_SCHEMA_VERSION
    capture_id: str | None = None
    source: Literal["browser-extension"] = BROWSER_CAPTURE_TRANSPORT_SOURCE
    provenance: BrowserCaptureProvenance
    session: BrowserCaptureSession
    provider_meta: dict[str, object] = Field(default_factory=dict)
    raw_provider_payload: dict[str, object] | None = None

    @field_validator("provider_meta", mode="before")
    @classmethod
    def coerce_provider_meta(cls, value: object) -> dict[str, object]:
        return dict(json_document(value))

    @field_validator("raw_provider_payload", mode="before")
    @classmethod
    def coerce_raw_provider_payload(cls, value: object) -> dict[str, object] | None:
        if value is None:
            return None
        payload: dict[str, object] = dict(json_document(value))
        return payload

    @model_validator(mode="after")
    def fill_capture_id(self) -> BrowserCaptureEnvelope:
        if self.capture_id is None:
            self.capture_id = f"{self.session.provider.value}:{self.session.provider_session_id}"
        return self

    @property
    def provider(self) -> Provider:
        return self.session.provider

    @property
    def provider_session_id(self) -> str:
        return self.session.provider_session_id


class BrowserCaptureReceiverStatusPayload(BaseModel):
    """Receiver readiness payload returned by ``GET /v1/status``."""

    ok: Literal[True] = True
    receiver: Literal["polylogue-browser-capture"] = BROWSER_CAPTURE_RECEIVER
    schema_version: Literal[1] = BROWSER_CAPTURE_SCHEMA_VERSION
    api_schema: Literal["polylogue-browser-capture/v1"] = BROWSER_CAPTURE_API_SCHEMA
    receiver_id: str = Field(min_length=8, max_length=80)
    spool_path: str
    spool_ready: bool
    allowed_origins: list[str]
    allow_remote: bool
    auth_required: bool
    active: bool
    checked_at: str


class BrowserCaptureArchiveStatePayload(BaseModel):
    """Capture-state payload returned by ``GET /v1/archive-state``."""

    provider: str
    provider_session_id: str
    state: BrowserCaptureArchiveLifecycle
    lifecycle: BrowserCaptureArchiveLifecycle
    captured: bool
    spooled: bool
    artifact_ref: str
    capture_id: str | None = None
    updated_at: str | None = None
    artifact_readable: bool | None = None
    raw_row_exists: bool = False
    raw_id: str | None = None
    indexed_session_exists: bool = False
    indexed_session_id: str | None = None
    indexed_message_count: int | None = None
    latest_failure: str | None = None
    failure_source: str | None = None


class BrowserCaptureAcceptedPayload(BaseModel):
    """Accepted-capture payload returned by ``POST /v1/browser-captures``."""

    ok: Literal[True] = True
    receiver: Literal["polylogue-browser-capture"] = BROWSER_CAPTURE_RECEIVER
    schema_version: Literal[1] = BROWSER_CAPTURE_SCHEMA_VERSION
    source: Literal["browser-extension"] = BROWSER_CAPTURE_TRANSPORT_SOURCE
    capture_id: str
    provider: str
    provider_session_id: str
    artifact_ref: str
    content_hash: str
    dedup_content_hash: str
    bytes_written: int
    replaced: bool
    deduplicated: bool
    capture_instance_id: str | None = None


class BrowserCaptureCapabilitiesPayload(BaseModel):
    """Receiver-declared browser-capture contract required before backfill."""

    ok: Literal[True] = True
    receiver: Literal["polylogue-browser-capture"] = BROWSER_CAPTURE_RECEIVER
    schema_version: Literal[1] = BROWSER_CAPTURE_SCHEMA_VERSION
    durable_ack_fields: tuple[Literal["receiver_request_id"], Literal["content_hash"]] = (
        "receiver_request_id",
        "content_hash",
    )


class BrowserBackfillCheckpointRequest(BaseModel):
    """Extension-submitted backfill-ledger checkpoint mirror (polylogue-06zm).

    The receiver becomes a durable second copy of the extension's IndexedDB
    backfill ledger (job/queue/revision state, already sanitized of provider
    credentials by the extension before it ever leaves the browser — see
    browser-extension/src/backfill/storage.js exportRecoveryCheckpoint), so a
    browser profile loss that also destroys the extension's local
    chrome.storage.local mirror still leaves a recoverable checkpoint.
    IndexedDB remains the fast primary source; this mirror is a fallback.

    The receiver treats ``checkpoint`` as an opaque JSON object: its internal
    job/queue/revision shape is the extension's concern, matching the browser
    capture envelope's own "receiver does not reinterpret payload structure"
    trust boundary (see BrowserCaptureHandler's class docstring).
    """

    extension_instance_id: str = Field(min_length=1, max_length=128)
    checkpoint: dict[str, object]

    @field_validator("checkpoint", mode="before")
    @classmethod
    def require_checkpoint_document(cls, value: object) -> dict[str, object]:
        # This field is the disaster-recovery mirror of the extension's
        # backfill ledger (polylogue-06zm): a malformed request must be
        # REJECTED, never silently coerced to {} and persisted as if it were
        # a legitimate checkpoint -- that would report HTTP 202 success while
        # overwriting a previously-good stored checkpoint with an empty one.
        if not is_json_document(value):
            raise ValueError("checkpoint must be a JSON object")
        return dict(value)


class BrowserBackfillCheckpointRecord(BaseModel):
    """Persisted checkpoint envelope, one per extension instance (last write wins)."""

    extension_instance_id: str
    checkpoint: dict[str, object]
    stored_at: str

    @field_validator("checkpoint", mode="before")
    @classmethod
    def require_checkpoint_document(cls, value: object) -> dict[str, object]:
        # Same durability rationale as BrowserBackfillCheckpointRequest above:
        # a checkpoint file on disk that fails to parse as a JSON object is
        # corrupt, not an empty-but-valid checkpoint. Raising here makes
        # read_backfill_checkpoint's `except Exception: return None` treat a
        # corrupt stored file as "no checkpoint found" rather than silently
        # presenting fabricated empty data as a real one.
        if not is_json_document(value):
            raise ValueError("checkpoint must be a JSON object")
        return dict(value)


class BrowserBackfillCheckpointAcceptedPayload(BaseModel):
    """Response to ``POST /v1/backfill-checkpoint``."""

    ok: Literal[True] = True
    receiver: Literal["polylogue-browser-capture"] = BROWSER_CAPTURE_RECEIVER
    schema_version: Literal[1] = BROWSER_CAPTURE_SCHEMA_VERSION
    extension_instance_id: str
    stored_at: str
    bytes_written: int


class BrowserBackfillCheckpointPayload(BaseModel):
    """Response to ``GET /v1/backfill-checkpoint``."""

    ok: Literal[True] = True
    receiver: Literal["polylogue-browser-capture"] = BROWSER_CAPTURE_RECEIVER
    schema_version: Literal[1] = BROWSER_CAPTURE_SCHEMA_VERSION
    extension_instance_id: str
    checkpoint: dict[str, object]
    stored_at: str


class BrowserCaptureErrorPayload(BaseModel):
    """Safe receiver error payload with no paths or stack traces."""

    ok: Literal[False] = False
    receiver: Literal["polylogue-browser-capture"] = BROWSER_CAPTURE_RECEIVER
    schema_version: Literal[1] = BROWSER_CAPTURE_SCHEMA_VERSION
    error: str


BrowserActionProvider = Literal["chatgpt", "claude"]
BrowserActionOperation = Literal["conversation.create", "conversation.reply"]
BrowserActionSubmitPolicy = Literal["stage_only", "submit_once"]
BrowserActionStatus = Literal[
    "queued",
    "leased",
    "preparing",
    "submit_intent",
    "outcome_unknown",
    "drafted",
    "submitted",
    "blocked",
    "failed",
    "cancelled",
]
BrowserActionOutcome = Literal[
    "progress",
    "drafted",
    "submitted",
    "outcome_unknown",
    "provider_warning",
    "rate_limited",
    "safety_locked",
    "auth_challenge",
    "network_error",
    "capability_mismatch",
    "provider_drift",
    "failed",
]


class BrowserActionTarget(BaseModel):
    """Provider-qualified destination for one browser action."""

    conversation_id: str = Field(default="new", min_length=1, max_length=255)
    conversation_url: str | None = Field(default=None, max_length=2_048)
    project_ref: str | None = Field(default=None, max_length=255)


class BrowserActionPresentation(BaseModel):
    """Exact provider UI selection requested at the submit boundary."""

    surface: Literal["chat"] = "chat"
    model_slug: str = Field(min_length=1, max_length=160)
    model_label: str = Field(min_length=1, max_length=200)
    effort_label: str = Field(min_length=1, max_length=120)


class BrowserActionAttachmentInput(BaseModel):
    """One caller-supplied attachment copied into receiver-owned storage."""

    name: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(default="application/octet-stream", min_length=1, max_length=255)
    content_base64: str

    @field_validator("name")
    @classmethod
    def require_safe_name(cls, value: str) -> str:
        if value != Path(value).name or value in {".", ".."} or any(ord(char) < 32 for char in value):
            raise ValueError("browser action attachment name must be a safe basename")
        return value


class BrowserActionAttachment(BaseModel):
    """Immutable metadata for receiver-copied action input bytes."""

    attachment_id: str
    name: str
    mime_type: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BrowserActionRequest(BaseModel):
    """Provider-neutral request to draft or submit one conversational turn."""

    action_id: str | None = Field(default=None, min_length=1, max_length=160)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)
    provider: BrowserActionProvider
    operation: BrowserActionOperation
    target: BrowserActionTarget = Field(default_factory=BrowserActionTarget)
    text: str = Field(min_length=1, max_length=200_000)
    attachments: list[BrowserActionAttachmentInput] = Field(default_factory=list, max_length=100)
    presentation: BrowserActionPresentation
    submit_policy: BrowserActionSubmitPolicy = "stage_only"

    @field_validator("text")
    @classmethod
    def require_nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("browser action text must not be empty")
        return value

    @model_validator(mode="after")
    def target_matches_operation(self) -> BrowserActionRequest:
        is_new = self.target.conversation_id == "new"
        if self.operation == "conversation.create" and not is_new:
            raise ValueError("conversation.create requires target conversation_id=new")
        if self.operation == "conversation.reply" and is_new:
            raise ValueError("conversation.reply requires an existing conversation id")
        return self


class BrowserActionReceipt(BaseModel):
    """Exact provider observation returned after draft or submit."""

    action_id: str
    receiver_id: str
    extension_instance_id: str
    provider_conversation_id: str | None = None
    provider_conversation_url: str | None = None
    provider_turn_id: str | None = None
    observed_surface: str | None = None
    observed_model: str | None = None
    observed_effort: str | None = None
    observed_project_ref: str | None = None
    provider_evidence: dict[str, object] = Field(default_factory=dict)
    observed_at: str


class BrowserActionEvent(BaseModel):
    event_id: str
    at: str
    kind: str
    phase: str
    detail: str | None = None
    owner_instance_id: str | None = None
    retry_after_seconds: int | None = Field(default=None, ge=1, le=86_400)


class BrowserActionIntent(BaseModel):
    """Durable receiver-authoritative browser action transport state."""

    polylogue_browser_action_kind: Literal["browser_action_intent"] = "browser_action_intent"
    schema_version: Literal[1] = 1
    contract: Literal["polylogue.browser-actions/v1"] = "polylogue.browser-actions/v1"
    capability_version: Literal[1] = 1
    action_id: str
    idempotency_key: str
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receiver_id: str
    provider: BrowserActionProvider
    operation: BrowserActionOperation
    target: BrowserActionTarget
    text: str
    attachments: list[BrowserActionAttachment] = Field(default_factory=list)
    presentation: BrowserActionPresentation
    submit_policy: BrowserActionSubmitPolicy
    status: BrowserActionStatus = "queued"
    phase: str = "queued"
    created_at: str
    updated_at: str
    lease_owner: str | None = None
    lease_expires_at: str | None = None
    submit_intent_at: str | None = None
    last_error: str | None = None
    failure_kind: str | None = None
    retry_after_seconds: int | None = Field(default=None, ge=1, le=86_400)
    receipt: BrowserActionReceipt | None = None
    events: list[BrowserActionEvent] = Field(default_factory=list)


class BrowserActionUpdateRequest(BaseModel):
    owner_instance_id: str = Field(min_length=1, max_length=200)
    outcome: BrowserActionOutcome = "progress"
    phase: str = Field(min_length=1, max_length=120)
    detail: str | None = Field(default=None, max_length=4_000)
    retry_after_seconds: int | None = Field(default=None, ge=1, le=86_400)
    receipt: BrowserActionReceipt | None = None


class BrowserActionReconcileRequest(BaseModel):
    """Explicitly bind a provider observation to an uncertain submit."""

    resolution: Literal["submitted", "drafted"]
    detail: str = Field(min_length=1, max_length=4_000)
    receipt: BrowserActionReceipt


class BrowserActionListPayload(BaseModel):
    ok: Literal[True] = True
    receiver: Literal["polylogue-browser-capture"] = BROWSER_CAPTURE_RECEIVER
    schema_version: Literal[1] = BROWSER_CAPTURE_SCHEMA_VERSION
    actions: list[BrowserActionIntent] = Field(default_factory=list)


class BrowserActionPayload(BaseModel):
    ok: Literal[True] = True
    receiver: Literal["polylogue-browser-capture"] = BROWSER_CAPTURE_RECEIVER
    schema_version: Literal[1] = BROWSER_CAPTURE_SCHEMA_VERSION
    action: BrowserActionIntent


class BrowserActionCapabilitiesPayload(BaseModel):
    ok: Literal[True] = True
    receiver: Literal["polylogue-browser-capture"] = BROWSER_CAPTURE_RECEIVER
    schema_version: Literal[1] = BROWSER_CAPTURE_SCHEMA_VERSION
    contract: Literal["polylogue.browser-actions/v1"] = "polylogue.browser-actions/v1"
    providers: dict[str, object]


def looks_like_browser_capture(payload: object) -> bool:
    """Return whether a payload is a browser-capture envelope."""
    if not isinstance(payload, dict):
        return False
    return (
        payload.get("polylogue_capture_kind") == BROWSER_CAPTURE_KIND
        and payload.get("schema_version") == BROWSER_CAPTURE_SCHEMA_VERSION
        and is_json_document(payload.get("session"))
        and is_json_document(payload.get("provenance"))
    )


__all__ = [
    "BROWSER_CAPTURE_API_SCHEMA",
    "BROWSER_CAPTURE_EXTENSION_ORIGIN_WILDCARD",
    "BROWSER_CAPTURE_KIND",
    "BROWSER_CAPTURE_RECEIVER",
    "BROWSER_CAPTURE_SCHEMA_VERSION",
    "BROWSER_CAPTURE_TRANSPORT_SOURCE",
    "BrowserBackfillCheckpointAcceptedPayload",
    "BrowserBackfillCheckpointPayload",
    "BrowserBackfillCheckpointRecord",
    "BrowserBackfillCheckpointRequest",
    "BrowserActionAttachment",
    "BrowserActionAttachmentInput",
    "BrowserActionCapabilitiesPayload",
    "BrowserActionEvent",
    "BrowserActionIntent",
    "BrowserActionListPayload",
    "BrowserActionOperation",
    "BrowserActionOutcome",
    "BrowserActionPayload",
    "BrowserActionPresentation",
    "BrowserActionProvider",
    "BrowserActionReceipt",
    "BrowserActionReconcileRequest",
    "BrowserActionRequest",
    "BrowserActionStatus",
    "BrowserActionSubmitPolicy",
    "BrowserActionTarget",
    "BrowserActionUpdateRequest",
    "BrowserCaptureAcceptedPayload",
    "BrowserCaptureCapabilitiesPayload",
    "BrowserCaptureArchiveLifecycle",
    "BrowserCaptureArchiveStatePayload",
    "BrowserCaptureAttachment",
    "BrowserCaptureEnvelope",
    "BrowserCaptureErrorPayload",
    "BrowserCaptureInterruption",
    "BrowserCaptureProvenance",
    "BrowserCaptureReceiverStatusPayload",
    "BrowserCaptureSession",
    "BrowserCaptureSessionKind",
    "BrowserCaptureTurn",
    "looks_like_browser_capture",
]
