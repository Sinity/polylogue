"""Typed browser-capture envelope shared by receiver and parser."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeGuard

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationInfo, field_validator, model_validator

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider, ToolOutcome
from polylogue.core.json import is_json_document, json_document
from polylogue.sources.parsers.base_models import ParsedFileEdit, ParsedWebConstruct

BROWSER_CAPTURE_KIND: Literal["browser_llm_session"] = "browser_llm_session"
BROWSER_CAPTURE_SCHEMA_VERSION: Literal[1] = 1
BROWSER_CAPTURE_TRANSPORT_SOURCE: Literal["browser-extension"] = "browser-extension"
BROWSER_CAPTURE_RECEIVER: Literal["polylogue-browser-capture"] = "polylogue-browser-capture"
BROWSER_CAPTURE_API_SCHEMA: Literal["polylogue-browser-capture/v1"] = "polylogue-browser-capture/v1"
BROWSER_CAPTURE_EXTENSION_ORIGIN_WILDCARD: Literal["chrome-extension://*"] = "chrome-extension://*"
BrowserCaptureArchiveLifecycle = Literal["missing", "spooled_only", "ingest_pending", "archived", "stale", "failed"]
BrowserCaptureSessionKind = Literal["standard", "temporary"]
BrowserCaptureTitleSource = Literal["provider", "page", "session-id"]
BrowserCaptureIdentityFidelity = Literal["native", "dom_degraded", "unknown"]
BrowserCaptureIdentityReason = Literal[
    "missing_conversation_id",
    "missing_message_id",
    "adapter_drift",
    "ambiguous",
    "receiver_disagreement",
    "unsupported_provider",
]


class BrowserCaptureIdentityObservation(BaseModel):
    """Provider-native identity evidence; DOM hints are explicitly non-authoritative."""

    origin: str
    provider_conversation_id: str | None = None
    provider_message_id: str | None = None
    parent_provider_message_id: str | None = None
    branch_context: str | None = None
    variant_index: int | None = Field(default=None, ge=0)
    content_fingerprint: str | None = None
    dom_ordinal: int | None = Field(default=None, ge=0)
    adapter_name: str
    adapter_version: str | None = None
    observed_at: str | None = None
    fidelity: BrowserCaptureIdentityFidelity = "unknown"
    degraded_reason: BrowserCaptureIdentityReason | None = None


class BrowserCaptureAcceptedIdentity(BaseModel):
    """The exact canonical identity accepted by the receiver."""

    session_ref: str
    message_ref: str | None = None
    evidence_ref: str | None = None
    fidelity: BrowserCaptureIdentityFidelity
    adapter_version: str | None = None


#: Attachment fields that carry the attachment's bytes as base64.
ATTACHMENT_CARRIER_FIELDS: tuple[str, ...] = ("content_base64", "inline_base64", "data")


class SpilledCarrier(str):
    """An attachment byte carrier already decoded into the blob store.

    The streamed ingest decode (``capture_stream.load_capture_for_ingest``)
    puts one in place of a base64 string so the carrier is never held beside
    the rest of the capture. JSON decoding cannot produce one, so a posted
    envelope cannot claim a blob. It reads as the empty string to shape and
    schema inspection; :func:`validate_capture_envelope` lifts it onto the
    validated attachment.
    """

    blob_hash: str
    size_bytes: int

    def __new__(cls, blob_hash: str, size_bytes: int) -> SpilledCarrier:
        carrier = super().__new__(cls, "")
        carrier.blob_hash = blob_hash
        carrier.size_bytes = size_bytes
        return carrier

    def __getnewargs__(self) -> tuple[str, int]:  # type: ignore[override]
        return (self.blob_hash, self.size_bytes)


class BrowserCaptureAttachment(BaseModel):
    """Attachment reference observed in a browser-hosted LLM session."""

    provider_attachment_id: str
    message_provider_id: str | None = None
    attachment_kind: str | None = None
    name: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    url: str | None = None
    extracted_content: str | None = None
    inline_base64: str | None = None
    content_base64: str | None = None
    data: str | None = None
    provider_meta: dict[str, object] = Field(default_factory=dict)
    _spilled_carriers: dict[str, SpilledCarrier] = PrivateAttr(default_factory=dict)

    @field_validator("provider_meta", mode="before")
    @classmethod
    def coerce_provider_meta(cls, value: object) -> dict[str, object]:
        return dict(json_document(value))

    def spilled_carrier(self, field_name: str) -> SpilledCarrier | None:
        """The blob a streamed decode moved ``field_name``'s bytes into, if any."""
        return self._spilled_carriers.get(field_name)


class BrowserCaptureBlock(BaseModel):
    """A single structured content block observed within a browser-captured turn.

    Deliberately modeled on ``ParsedContentBlock``
    (``polylogue/sources/parsers/base_models.py``) so a capture adapter that
    has observed real structure (a ChatGPT ``mapping`` node's ``recipient`` /
    ``content_type``, a Claude API tool_use/tool_result segment, ...) can
    carry that structure across the wire verbatim instead of flattening it
    into ``BrowserCaptureTurn.text`` prose. The parser
    (``polylogue/sources/parsers/browser_capture.py``) converts these 1:1 into
    ``ParsedContentBlock`` rows.

    Native preparation uses the canonical provider parser, including its
    constructs, file-edit evidence, signatures and normalized tool outcomes.
    Page adapters may leave those fields absent; typed validation preserves
    them whenever the canonical preparation supplies them.
    """

    type: BlockType
    text: str | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    tool_input: dict[str, object] | None = None
    media_type: str | None = None
    metadata: dict[str, object] | None = None
    # Structured tool-result outcome, mirrored from ParsedContentBlock: read
    # from the provider's own outcome fields when the capture adapter observed
    # them, never regex-guessed from rendered text.
    is_error: bool | None = None
    exit_code: int | None = None
    signature: str | None = None
    tool_outcome: ToolOutcome | None = None
    outcome_unknown_reason: str | None = None
    file_edit: ParsedFileEdit | None = None
    web_constructs: list[ParsedWebConstruct] = Field(default_factory=list)

    @field_validator("type", mode="before")
    @classmethod
    def coerce_type(cls, value: object) -> BlockType:
        return BlockType.from_string(str(value))

    @field_validator("tool_input", "metadata", mode="before")
    @classmethod
    def coerce_object_field(cls, value: object) -> dict[str, object] | None:
        if value is None:
            return None
        return dict(json_document(value))


class _NativeMessageWitness(Protocol):
    provider_message_id: str
    role: Role
    text: str | None
    timestamp: str | None
    parent_message_provider_id: str | None


@dataclass(frozen=True)
class _CanonicalNativeTurnWitness:
    """Process-local canonical parser evidence, never a wire mode flag.

    Callers retain the exact raw revision and complete canonical parsing before
    supplying this witness. The sequence can be the existing SQLite sink.
    """

    messages: Sequence[_NativeMessageWitness]

    def matches(self, turn: BrowserCaptureTurn) -> bool:
        if "ordinal" not in turn.model_fields_set or not 0 <= turn.ordinal < len(self.messages):
            return False
        native = self.messages[turn.ordinal]
        return (
            native.provider_message_id == turn.provider_turn_id
            and native.role == turn.role
            and native.text == turn.text
            and native.timestamp == turn.timestamp
            and native.parent_message_provider_id == turn.parent_turn_id
        )


class BrowserCaptureTurn(BaseModel):
    """Provider-neutral turn observed from the page."""

    provider_turn_id: str
    role: Role
    text: str | None = None
    timestamp: str | None = None
    ordinal: int = 0
    parent_turn_id: str | None = None
    attachments: list[BrowserCaptureAttachment] = Field(default_factory=list)
    # Typed structured content observed for this turn (tool_use/tool_result/
    # thinking/code/...). ``text`` remains a rendering of the turn -- kept for
    # search/display and for capture adapters that have not been upgraded to
    # emit structure yet -- but it is no longer the only content channel a
    # tool-shaped turn can populate (polylogue-ah21).
    blocks: list[BrowserCaptureBlock] = Field(default_factory=list)
    provider_meta: dict[str, object] = Field(default_factory=dict)
    identity_observation: BrowserCaptureIdentityObservation | None = None

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
    def require_content(self, info: ValidationInfo) -> BrowserCaptureTurn:
        if (self.text is None or not self.text.strip()) and not self.attachments and not self.blocks:
            witness = info.context.get("canonical_native_turns") if isinstance(info.context, dict) else None
            if not isinstance(witness, _CanonicalNativeTurnWitness) or not witness.matches(self):
                raise ValueError("browser capture turn must include text, blocks, or attachments")
        if (
            not self.provider_turn_id
            and any(not attachment.message_provider_id for attachment in self.attachments)
            and "ordinal" not in self.model_fields_set
        ):
            raise ValueError("an id-less attachment turn requires an explicit native ordinal")
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
    # The browser owner stores this counter as an exactly represented JS
    # integer. It orders acquisitions only within the declared instance.
    acquisition_sequence: int | None = Field(default=None, strict=True, ge=1, le=2**53 - 1)
    browser_profile: str | None = None
    adapter_name: str
    adapter_version: str | None = None
    capture_mode: Literal["snapshot", "tail"] = "snapshot"
    capture_interruption: BrowserCaptureInterruption | None = None
    provider_meta: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_acquisition_owner(self) -> BrowserCaptureProvenance:
        if self.acquisition_sequence is not None and self.extension_instance_id is None:
            raise ValueError("acquisition_sequence requires extension_instance_id")
        return self

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
    title_source: BrowserCaptureTitleSource | None = None
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
    raw_provider_payload: dict[str, object] | list[dict[str, object]] | None = None

    @field_validator("provider_meta", mode="before")
    @classmethod
    def coerce_provider_meta(cls, value: object) -> dict[str, object]:
        return dict(json_document(value))

    @field_validator("raw_provider_payload", mode="before")
    @classmethod
    def coerce_raw_provider_payload(cls, value: object) -> dict[str, object] | list[dict[str, object]] | None:
        if value is None:
            return None
        if isinstance(value, list):
            if not value or any(not is_json_document(record) for record in value):
                raise ValueError("native record payload must contain JSON objects")
            return [dict(json_document(record)) for record in value]
        payload: dict[str, object] = dict(json_document(value))
        if payload.get("polylogue_bridge_projection") == "chatgpt-native-compact-v1":
            raise ValueError("capture_retired_projection: synthetic compact payload is not provider-native evidence")
        return payload

    @model_validator(mode="after")
    def fill_capture_id(self) -> BrowserCaptureEnvelope:
        if isinstance(self.raw_provider_payload, list) and self.session.provider is not Provider.CODEX:
            raise ValueError("native record arrays are supported only for Codex")
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
    outcome: Literal["accepted", "noop", "superseded"]
    submitted_content_hash: str
    content_hash: str
    dedup_content_hash: str
    bytes_written: int
    replaced: bool
    deduplicated: bool
    capture_instance_id: str | None = None
    accepted_identities: list[BrowserCaptureAcceptedIdentity] = Field(default_factory=list)


class BrowserCaptureCapabilitiesPayload(BaseModel):
    """Receiver-declared browser-capture capabilities required by extensions."""

    ok: Literal[True] = True
    receiver: Literal["polylogue-browser-capture"] = BROWSER_CAPTURE_RECEIVER
    schema_version: Literal[1] = BROWSER_CAPTURE_SCHEMA_VERSION
    durable_ack_fields: tuple[
        Literal["receiver_request_id"], Literal["content_hash"], Literal["submitted_content_hash"], Literal["outcome"]
    ] = ("receiver_request_id", "content_hash", "submitted_content_hash", "outcome")
    assertion_candidates: Literal[True] = True


class BrowserCaptureErrorPayload(BaseModel):
    """Safe receiver error payload with no paths or stack traces."""

    ok: Literal[False] = False
    receiver: Literal["polylogue-browser-capture"] = BROWSER_CAPTURE_RECEIVER
    schema_version: Literal[1] = BROWSER_CAPTURE_SCHEMA_VERSION
    error: str


class BrowserCapturePairingRedeemRequest(BaseModel):
    """One-time pairing-code exchange request (polylogue-gnie)."""

    code: str = Field(min_length=1, max_length=32)


class BrowserCapturePairingRedeemPayload(BaseModel):
    """Successful pairing-code exchange: the receiver's current bearer token.

    Deliberately does not echo the redeemed code back, and callers must not
    log this payload -- it carries the same secret as `token show`.
    """

    ok: Literal[True] = True
    receiver: Literal["polylogue-browser-capture"] = BROWSER_CAPTURE_RECEIVER
    schema_version: Literal[1] = BROWSER_CAPTURE_SCHEMA_VERSION
    auth_token: str
    receiver_id: str


#: A receiver attestation challenge: 32 random bytes, base64url without
#: padding, so every accepted challenge carries the same fresh entropy.
RECEIVER_ATTESTATION_CHALLENGE_PATTERN = r"^[A-Za-z0-9_-]{43}$"


class BrowserCaptureReceiverAttestationRequest(BaseModel):
    """A client's fresh challenge to a receiver it has not yet trusted."""

    challenge: str = Field(pattern=RECEIVER_ATTESTATION_CHALLENGE_PATTERN)


class BrowserCaptureReceiverAttestationPayload(BaseModel):
    """Proof that this receiver holds the bearer, without revealing it."""

    ok: Literal[True] = True
    receiver: Literal["polylogue-browser-capture"] = BROWSER_CAPTURE_RECEIVER
    schema_version: Literal[1] = BROWSER_CAPTURE_SCHEMA_VERSION
    api_schema: str
    receiver_id: str
    proof: str


#: Capture-health telemetry kinds the extension may report (polylogue-3v1).
#: `capture_gap` is the completeness mismatch (page shows more messages than
#: were captured); `capture_error` is a failed capture attempt; `spool_backlog`
#: is the extension reporting its own offline-spool depth crossing a
#: self-chosen watermark (the receiver does not compute this -- it has no
#: visibility into the extension's local queue).
BrowserCaptureHealthEventKind = Literal["capture_gap", "capture_error", "spool_backlog", "provider_auth_broken"]


class BrowserCaptureHealthEventRequest(BaseModel):
    """One capture-health telemetry event reported by the extension."""

    event: BrowserCaptureHealthEventKind
    provider: str | None = None
    provider_session_id: str | None = None
    extension_instance_id: str | None = None
    visible_count: int | None = Field(default=None, ge=0)
    captured_count: int | None = Field(default=None, ge=0)
    reason: str | None = None
    detail: dict[str, object] = Field(default_factory=dict)


class BrowserCaptureHealthEventAcceptedPayload(BaseModel):
    """Accepted capture-health report, echoing its committed history/resume id."""

    ok: Literal[True] = True
    receiver: Literal["polylogue-browser-capture"] = BROWSER_CAPTURE_RECEIVER
    schema_version: Literal[1] = BROWSER_CAPTURE_SCHEMA_VERSION
    event_id: int


BrowserActionProvider = Literal["chatgpt", "claude"]
BrowserActionOperation = Literal["conversation.create", "conversation.reply"]
BrowserActionSubmitPolicy = Literal["stage_only", "submit_once"]
BrowserActionStatus = Literal[
    "awaiting_approval",
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
#: Why a browser action is held at ``awaiting_approval`` instead of being
#: claimable immediately after enqueue. Only ``destructive_submit`` is wired
#: to a real trigger today (polylogue-yyvg.7): a ``submit_once`` action whose
#: operation is ``conversation.reply`` posts one real, provider-visible turn
#: into an EXISTING conversation the operator may not currently be watching,
#: with no automatic undo. That is the "explicit submit/destructive approval"
#: category the redesign's AC3 names and the popup previously had no data to
#: back. A second AC3-named category, "destructive conflict", is deliberately
#: NOT modeled here: the receiver's ``claim_action`` already serializes to one
#: in-flight action at a time (see its "any(...) return None" guard below), so
#: there is no genuine two-actions-racing-for-one-resource scenario in the
#: current architecture to attach it to. Adding a value for it now would be an
#: unbacked, unfireable enum member; a future bead should add it once a real
#: collision scenario exists (e.g. multi-instance orchestration).
BrowserActionApprovalReason = Literal["destructive_submit"]
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

    model_config = ConfigDict(protected_namespaces=())

    surface: Literal["chat"] = "chat"
    model_slug: str = Field(min_length=1, max_length=160)
    model_label: str = Field(min_length=1, max_length=200)
    effort_label: str = Field(min_length=1, max_length=120)


class BrowserActionAttachmentInput(BaseModel):
    """One caller-supplied attachment copied into receiver-owned storage."""

    name: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(default="application/octet-stream", min_length=1, max_length=255)
    model_config = ConfigDict(extra="forbid")

    attachment_ref: str = Field(pattern=r"^[0-9a-f]{64}$")

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
    #: True once this action was ever held for explicit operator approval
    #: (stays True after approve/decline -- it is a historical fact, not a
    #: live gate; the live gate is ``status == "awaiting_approval"``).
    requires_operator_approval: bool = False
    approval_reason: BrowserActionApprovalReason | None = None
    approval_requested_at: str | None = None
    approved_at: str | None = None
    approved_by: str | None = None
    declined_at: str | None = None


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


class BrowserActionApprovalDecisionRequest(BaseModel):
    """The operator's explicit approve/decline decision on a held action.

    Unlike ``BrowserActionUpdateRequest``, this does not require a lease --
    an ``awaiting_approval`` action has never been leased, and approving one
    is exactly what makes it claimable for the first time.
    """

    extension_instance_id: str = Field(min_length=1, max_length=128)
    decision: Literal["approve", "decline"]
    detail: str | None = Field(default=None, max_length=4_000)


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


def has_chatgpt_native_payload(payload: object) -> TypeGuard[Mapping[str, object]]:
    """Whether a raw provider payload is a trusted ChatGPT conversation mapping."""
    return (
        isinstance(payload, dict)
        and payload.get("polylogue_bridge_projection") != "chatgpt-native-compact-v1"
        and isinstance(payload.get("mapping"), dict)
    )


def has_claude_ai_native_payload(payload: object) -> TypeGuard[Mapping[str, object]]:
    """Whether a raw provider payload is a trusted Claude.ai conversation body."""
    return isinstance(payload, dict) and isinstance(payload.get("chat_messages"), list)


def has_grok_native_payload(payload: object) -> TypeGuard[dict[str, object]]:
    """Identify the endpoint bundle carrier; the native parser validates identity."""
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("conversation"), dict)
        and isinstance(payload.get("responses"), (dict, list))
    )


def envelope_has_native_provider_payload(envelope: BrowserCaptureEnvelope) -> bool:
    """Whether this capture carries a provider-native transcript, not a DOM read.

    One definition, three readers: the parser routes a native payload to the
    provider parser and tags the session
    ``NATIVE_BROWSER_CAPTURE_INGEST_FLAG``; the conservation census lowers the
    same documents production materializes; and the receiver's spool
    admission must not discard that fidelity in favour of a DOM fallback that
    happens to list more turns. The archive boundary
    (``archive_tiers.ingest_precedence.browser_capture_precedence``) already
    admits a lower-count native capture over a DOM fallback, so a receiver
    rule that drops it first makes the two disagree and retains the
    lower-fidelity content permanently.
    """
    payload = envelope.raw_provider_payload
    if envelope.session.provider is Provider.CODEX:
        return isinstance(payload, list) and bool(payload)
    if envelope.session.provider is Provider.CHATGPT:
        return has_chatgpt_native_payload(payload)
    if envelope.session.provider is Provider.CLAUDE_AI:
        return has_claude_ai_native_payload(payload)
    if envelope.session.provider is Provider.GROK:
        return has_grok_native_payload(payload)
    return False


def _lift_spilled_carriers(raw_items: object, attachments: list[BrowserCaptureAttachment]) -> None:
    if not isinstance(raw_items, list):
        return
    for raw, attachment in zip(raw_items, attachments, strict=False):
        if not isinstance(raw, Mapping):
            continue
        spilled = {
            field_name: value
            for field_name in ATTACHMENT_CARRIER_FIELDS
            if isinstance(value := raw.get(field_name), SpilledCarrier)
        }
        if spilled:
            attachment._spilled_carriers = spilled


def validate_capture_envelope(
    payload: object, *, native_witness: _CanonicalNativeTurnWitness | None = None
) -> BrowserCaptureEnvelope:
    """Validate an envelope, keeping carriers a streamed decode spilled.

    Validation turns a :class:`SpilledCarrier` into a plain empty string;
    the blob it names is re-attached to the attachment at the same position.
    """
    envelope = BrowserCaptureEnvelope.model_validate(payload, context={"canonical_native_turns": native_witness})
    session = payload.get("session") if isinstance(payload, Mapping) else None
    if not isinstance(session, Mapping):
        return envelope
    _lift_spilled_carriers(session.get("attachments"), envelope.session.attachments)
    raw_turns = session.get("turns")
    if isinstance(raw_turns, list):
        for raw_turn, turn in zip(raw_turns, envelope.session.turns, strict=False):
            if isinstance(raw_turn, Mapping):
                _lift_spilled_carriers(raw_turn.get("attachments"), turn.attachments)
    return envelope


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
    "BrowserCaptureBlock",
    "BrowserActionApprovalDecisionRequest",
    "BrowserActionApprovalReason",
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
    "ATTACHMENT_CARRIER_FIELDS",
    "BrowserCaptureAttachment",
    "SpilledCarrier",
    "validate_capture_envelope",
    "BrowserCaptureEnvelope",
    "BrowserCaptureErrorPayload",
    "BrowserCaptureInterruption",
    "BrowserCaptureProvenance",
    "BrowserCaptureReceiverAttestationPayload",
    "BrowserCaptureReceiverAttestationRequest",
    "BrowserCaptureReceiverStatusPayload",
    "BrowserCaptureSession",
    "BrowserCaptureSessionKind",
    "BrowserCaptureTurn",
    "RECEIVER_ATTESTATION_CHALLENGE_PATTERN",
    "envelope_has_native_provider_payload",
    "has_chatgpt_native_payload",
    "has_claude_ai_native_payload",
    "looks_like_browser_capture",
]
