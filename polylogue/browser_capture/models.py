"""Typed browser-capture envelope shared by receiver and parser."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.core.json import is_json_document, json_document

BROWSER_CAPTURE_KIND: Literal["browser_llm_session"] = "browser_llm_session"
BROWSER_CAPTURE_SCHEMA_VERSION: Literal[1] = 1
BROWSER_CAPTURE_TRANSPORT_SOURCE: Literal["browser-extension"] = "browser-extension"
BROWSER_CAPTURE_RECEIVER: Literal["polylogue-browser-capture"] = "polylogue-browser-capture"
BROWSER_CAPTURE_EXTENSION_ORIGIN_WILDCARD: Literal["chrome-extension://*"] = "chrome-extension://*"
BrowserCaptureArchiveLifecycle = Literal["missing", "spooled_only", "ingest_pending", "archived", "failed"]
BrowserCaptureSessionKind = Literal["standard", "temporary"]


class BrowserCaptureAttachment(BaseModel):
    """Attachment reference observed in a browser-hosted LLM session."""

    provider_attachment_id: str
    message_provider_id: str | None = None
    name: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    url: str | None = None
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


class BrowserCaptureProvenance(BaseModel):
    """How and where the capture was observed."""

    source_url: str
    page_title: str | None = None
    captured_at: str
    extension_id: str | None = None
    browser_profile: str | None = None
    adapter_name: str
    adapter_version: str | None = None
    capture_mode: Literal["snapshot", "tail"] = "snapshot"
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
    bytes_written: int
    replaced: bool


class BrowserCaptureErrorPayload(BaseModel):
    """Safe receiver error payload with no paths or stack traces."""

    ok: Literal[False] = False
    receiver: Literal["polylogue-browser-capture"] = BROWSER_CAPTURE_RECEIVER
    schema_version: Literal[1] = BROWSER_CAPTURE_SCHEMA_VERSION
    error: str


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
    "BROWSER_CAPTURE_EXTENSION_ORIGIN_WILDCARD",
    "BROWSER_CAPTURE_KIND",
    "BROWSER_CAPTURE_RECEIVER",
    "BROWSER_CAPTURE_SCHEMA_VERSION",
    "BROWSER_CAPTURE_TRANSPORT_SOURCE",
    "BrowserCaptureAcceptedPayload",
    "BrowserCaptureArchiveLifecycle",
    "BrowserCaptureArchiveStatePayload",
    "BrowserCaptureAttachment",
    "BrowserCaptureEnvelope",
    "BrowserCaptureErrorPayload",
    "BrowserCaptureProvenance",
    "BrowserCaptureReceiverStatusPayload",
    "BrowserCaptureSession",
    "BrowserCaptureSessionKind",
    "BrowserCaptureTurn",
    "looks_like_browser_capture",
]
