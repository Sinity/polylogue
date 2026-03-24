"""Archive-core storage record models and shared serialization helpers."""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field, field_validator

from polylogue.lib.branch_type import BranchType
from polylogue.lib.hashing import hash_text
from polylogue.lib.json import dumps as json_dumps
from polylogue.lib.roles import Role
from polylogue.lib.security import sanitize_path as _sanitize_path_helper
from polylogue.types import (
    ArtifactSupportStatus,
    AttachmentId,
    ContentBlockType,
    ContentHash,
    ConversationId,
    MessageId,
    Provider,
    SemanticBlockType,
    ValidationMode,
    ValidationStatus,
)

MAX_ATTACHMENT_SIZE = 1024 * 1024 * 1024 * 1024
MAX_RAW_CONTENT_SIZE = 900 * 1024 * 1024
ACTION_EVENT_MATERIALIZER_VERSION = 1
SESSION_PRODUCT_MATERIALIZER_VERSION = 3
SESSION_INFERENCE_VERSION = 1
SESSION_INFERENCE_FAMILY = "heuristic_session_semantics"
SESSION_ENRICHMENT_VERSION = 1
SESSION_ENRICHMENT_FAMILY = "scored_session_enrichment"
MAINTENANCE_RUN_SCHEMA_VERSION = 1


class ConversationRecord(BaseModel):
    conversation_id: ConversationId
    provider_name: str
    provider_conversation_id: str
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    sort_key: float | None = None
    content_hash: ContentHash
    provider_meta: dict[str, object] | None = None
    metadata: dict[str, object] | None = None
    version: int = 1
    parent_conversation_id: ConversationId | None = None
    branch_type: BranchType | None = None
    raw_id: str | None = None

    @property
    def provider(self) -> Provider:
        return Provider.from_string(self.provider_name)

    @field_validator("conversation_id", "provider_conversation_id", "content_hash")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class ContentBlockRecord(BaseModel):
    block_id: str
    message_id: MessageId
    conversation_id: ConversationId
    block_index: int
    type: ContentBlockType
    text: str | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    tool_input: str | None = None
    media_type: str | None = None
    metadata: str | None = None
    semantic_type: SemanticBlockType | None = None

    @field_validator("type", mode="before")
    @classmethod
    def coerce_block_type(cls, v: object) -> ContentBlockType:
        return ContentBlockType.from_string(str(v))

    @field_validator("semantic_type", mode="before")
    @classmethod
    def coerce_semantic_type(cls, v: object) -> SemanticBlockType | None:
        if v is None:
            return None
        return SemanticBlockType.from_string(str(v))

    @classmethod
    def make_id(cls, message_id: str, block_index: int) -> str:
        return f"blk-{hash_text(f'{message_id}:{block_index}')[:16]}"


class MessageRecord(BaseModel):
    message_id: MessageId
    conversation_id: ConversationId
    provider_message_id: str | None = None
    role: Role | None = None
    text: str | None = None
    sort_key: float | None = None
    content_hash: ContentHash
    version: int = 1
    parent_message_id: MessageId | None = None
    branch_index: int = 0
    content_blocks: list[ContentBlockRecord] = Field(default_factory=list)
    provider_name: str = ""
    word_count: int = 0
    has_tool_use: int = 0
    has_thinking: int = 0

    @field_validator("role", mode="before")
    @classmethod
    def coerce_role(cls, v: object) -> Role | None:
        if v is None:
            return None
        if isinstance(v, Role):
            return v
        if isinstance(v, str) and not v.strip():
            return Role.UNKNOWN
        return Role.normalize(str(v))

    @property
    def role_typed(self) -> Role:
        return self.role or Role.UNKNOWN

    @field_validator("message_id", "conversation_id", "content_hash")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class AttachmentRecord(BaseModel):
    attachment_id: AttachmentId
    conversation_id: ConversationId
    message_id: MessageId | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    path: str | None = None
    provider_meta: dict[str, object] | None = None

    @field_validator("attachment_id", "conversation_id")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v

    @field_validator("path")
    @classmethod
    def sanitize_path(cls, v: str | None) -> str | None:
        return _sanitize_path_helper(v)

    @field_validator("size_bytes")
    @classmethod
    def validate_size_bytes(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v < 0:
            raise ValueError("size_bytes cannot be negative")
        if v > MAX_ATTACHMENT_SIZE:
            raise ValueError(f"size_bytes exceeds maximum ({MAX_ATTACHMENT_SIZE} bytes / 1TB)")
        return v


class RunRecord(BaseModel):
    run_id: str
    timestamp: str
    plan_snapshot: dict[str, Any] | None = None
    counts: dict[str, Any] | None = None
    drift: dict[str, Any] | None = None
    indexed: bool | None = None
    duration_ms: int | None = None


class PublicationRecord(BaseModel):
    publication_id: str
    publication_kind: str
    generated_at: str
    output_dir: str
    duration_ms: int | None = None
    manifest: dict[str, Any]

    @field_validator("publication_id", "publication_kind", "generated_at", "output_dir")
    @classmethod
    def publication_non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class RawConversationRecord(BaseModel):
    raw_id: str
    provider_name: str
    payload_provider: Provider | None = None
    source_name: str | None = None
    source_path: str
    source_index: int | None = None
    raw_content: bytes
    acquired_at: str
    file_mtime: str | None = None
    parsed_at: str | None = None
    parse_error: str | None = None
    validated_at: str | None = None
    validation_status: ValidationStatus | None = None
    validation_error: str | None = None
    validation_drift_count: int | None = None
    validation_provider: Provider | None = None
    validation_mode: ValidationMode | None = None

    @field_validator("raw_id", "provider_name", "source_path")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v

    @field_validator("raw_content")
    @classmethod
    def non_empty_bytes(cls, v: bytes) -> bytes:
        if not v:
            raise ValueError("raw_content cannot be empty")
        return v

    @field_validator("validation_status", mode="before")
    @classmethod
    def coerce_validation_status(cls, v: object) -> ValidationStatus | None:
        if v is None:
            return None
        return ValidationStatus.from_string(str(v))

    @field_validator("payload_provider", mode="before")
    @classmethod
    def coerce_payload_provider(cls, v: object) -> Provider | None:
        if v is None:
            return None
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v))

    @field_validator("validation_provider", mode="before")
    @classmethod
    def coerce_validation_provider(cls, v: object) -> Provider | None:
        if v is None:
            return None
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v))

    @field_validator("validation_mode", mode="before")
    @classmethod
    def coerce_validation_mode(cls, v: object) -> ValidationMode | None:
        if v is None:
            return None
        return ValidationMode.from_string(str(v))


class ArtifactObservationRecord(BaseModel):
    observation_id: str
    raw_id: str
    provider_name: str
    payload_provider: Provider | None = None
    source_name: str | None = None
    source_path: str
    source_index: int | None = None
    file_mtime: str | None = None
    wire_format: str | None = None
    artifact_kind: str
    classification_reason: str
    parse_as_conversation: bool
    schema_eligible: bool
    support_status: ArtifactSupportStatus
    malformed_jsonl_lines: int = 0
    decode_error: str | None = None
    bundle_scope: str | None = None
    cohort_id: str | None = None
    resolved_package_version: str | None = None
    resolved_element_kind: str | None = None
    resolution_reason: str | None = None
    link_group_key: str | None = None
    sidecar_agent_type: str | None = None
    first_observed_at: str
    last_observed_at: str

    @field_validator(
        "observation_id",
        "raw_id",
        "provider_name",
        "source_path",
        "artifact_kind",
        "classification_reason",
        "first_observed_at",
        "last_observed_at",
    )
    @classmethod
    def observation_non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v

    @field_validator("payload_provider", mode="before")
    @classmethod
    def coerce_observation_payload_provider(cls, v: object) -> Provider | None:
        if v is None:
            return None
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v))

    @field_validator("support_status", mode="before")
    @classmethod
    def coerce_support_status(cls, v: object) -> ArtifactSupportStatus:
        return ArtifactSupportStatus.from_string(str(v))


class ActionEventRecord(BaseModel):
    event_id: str
    conversation_id: ConversationId
    message_id: MessageId
    materializer_version: int = ACTION_EVENT_MATERIALIZER_VERSION
    source_block_id: str | None = None
    timestamp: str | None = None
    sort_key: float | None = None
    sequence_index: int
    provider_name: str | None = None
    action_kind: str
    tool_name: str | None = None
    normalized_tool_name: str
    tool_id: str | None = None
    affected_paths: tuple[str, ...] = ()
    cwd_path: str | None = None
    branch_names: tuple[str, ...] = ()
    command: str | None = None
    query_text: str | None = None
    url: str | None = None
    output_text: str | None = None
    search_text: str

    @field_validator(
        "event_id",
        "conversation_id",
        "message_id",
        "action_kind",
        "normalized_tool_name",
        "search_text",
    )
    @classmethod
    def action_non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


def _json_or_none(value: dict[str, object] | None) -> str | None:
    if value is None:
        return None
    return json_dumps(value)


def _json_array_or_none(value: tuple[str, ...] | list[str] | None) -> str | None:
    if not value:
        return None
    return json_dumps(list(value))


def _make_ref_id(attachment_id: AttachmentId, conversation_id: ConversationId, message_id: MessageId | None) -> str:
    seed = f"{attachment_id}:{conversation_id}:{message_id or ''}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"ref-{digest}"


__all__ = [
    "ACTION_EVENT_MATERIALIZER_VERSION",
    "ActionEventRecord",
    "AttachmentRecord",
    "ArtifactObservationRecord",
    "ContentBlockRecord",
    "ConversationRecord",
    "MAINTENANCE_RUN_SCHEMA_VERSION",
    "MAX_ATTACHMENT_SIZE",
    "MAX_RAW_CONTENT_SIZE",
    "MessageRecord",
    "PublicationRecord",
    "RawConversationRecord",
    "RunRecord",
    "SESSION_INFERENCE_FAMILY",
    "SESSION_INFERENCE_VERSION",
    "SESSION_PRODUCT_MATERIALIZER_VERSION",
    "_json_array_or_none",
    "_json_or_none",
    "_make_ref_id",
]
