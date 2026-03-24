"""Storage record models and validation."""

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

# Maximum reasonable file size (1TB)
MAX_ATTACHMENT_SIZE = 1024 * 1024 * 1024 * 1024

# SQLite SQLITE_MAX_LENGTH is 1 GB; keep raw blobs under 900 MB to leave headroom
MAX_RAW_CONTENT_SIZE = 900 * 1024 * 1024
ACTION_EVENT_MATERIALIZER_VERSION = 1
SESSION_PRODUCT_MATERIALIZER_VERSION = 1
MAINTENANCE_RUN_SCHEMA_VERSION = 1


class ConversationRecord(BaseModel):
    conversation_id: ConversationId
    provider_name: str
    provider_conversation_id: str
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    sort_key: float | None = None  # Pre-computed numeric updated_at for ORDER BY / WHERE
    content_hash: ContentHash
    provider_meta: dict[str, object] | None = None
    metadata: dict[str, object] | None = None
    version: int = 1
    # Branching support: links conversations in session trees
    parent_conversation_id: ConversationId | None = None
    branch_type: BranchType | None = None
    # Link to raw source data (FK to raw_conversations.raw_id)
    raw_id: str | None = None

    @property
    def provider(self) -> Provider:
        """Typed provider enum derived from provider_name string."""
        return Provider.from_string(self.provider_name)

    @field_validator("conversation_id", "provider_conversation_id", "content_hash")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class ContentBlockRecord(BaseModel):
    """A single structured content block belonging to a message.

    Content blocks are the canonical representation of message content.
    Block types: text, thinking, tool_use, tool_result, image, code, document.
    """

    block_id: str
    message_id: MessageId
    conversation_id: ConversationId
    block_index: int
    type: ContentBlockType
    text: str | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    tool_input: str | None = None  # JSON-serialized dict
    media_type: str | None = None
    metadata: str | None = None  # JSON-serialized dict
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
    text: str | None = None  # Concatenated text from text-type content blocks
    sort_key: float | None = None  # Pre-computed numeric timestamp for ORDER BY
    content_hash: ContentHash
    version: int = 1
    # Branching support: links messages in conversation trees
    parent_message_id: MessageId | None = None
    branch_index: int = 0  # 0 = mainline, >0 = branch sibling position
    # Content blocks loaded alongside the message (not stored in messages table)
    content_blocks: list[ContentBlockRecord] = Field(default_factory=list)
    # Precomputed analytics fields (stored in messages table, computed at insert time)
    provider_name: str = ''
    word_count: int = 0
    has_tool_use: int = 0   # 1 if any block.type in ('tool_use', 'tool_result')
    has_thinking: int = 0   # 1 if any block.type == 'thinking'

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
        """Typed role enum for callers that expect a non-null role."""
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
        """Sanitize path to prevent traversal attacks and other security issues."""
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
    """Durable persisted publication manifest."""

    publication_id: str
    publication_kind: str
    generated_at: str
    output_dir: str
    duration_ms: int | None = None
    manifest: dict[str, Any]

    @field_validator(
        "publication_id",
        "publication_kind",
        "generated_at",
        "output_dir",
    )
    @classmethod
    def publication_non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class RawConversationRecord(BaseModel):
    """Record storing original raw JSON/JSONL bytes before parsing.

    This enables honest, database-driven testing by preserving the exact
    input data that was parsed into conversations and messages.

    Note: The link to parsed conversations goes the OTHER way:
    conversations.raw_id → raw_conversations.raw_id
    This matches the data flow: acquire raw first, then parse.
    """
    raw_id: str  # SHA256 of raw_content
    provider_name: str  # Provenance/provider hint from acquisition time
    payload_provider: Provider | None = None  # Durable provider classification from decoded payload
    source_name: str | None = None  # Config source name (e.g., "inbox"), distinct from provider
    source_path: str
    source_index: int | None = None  # Position in bundle (e.g., conversations[3])
    raw_content: bytes  # Full JSON/JSONL bytes
    acquired_at: str  # ISO timestamp of acquisition
    file_mtime: str | None = None  # File modification time if available
    parsed_at: str | None = None  # ISO timestamp of last successful parse
    parse_error: str | None = None  # Error from last failed parse attempt
    validated_at: str | None = None  # ISO timestamp of last validation attempt
    validation_status: ValidationStatus | None = None
    validation_error: str | None = None  # Error from last failed validation attempt
    validation_drift_count: int | None = None  # Drift warnings seen during last validation
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
    """Durable observation of one source artifact, separate from raw blob dedupe."""

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
    """Durable canonical semantic action-event row."""

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


class SessionProfileRecord(BaseModel):
    """Durable canonical session-profile row."""

    conversation_id: ConversationId
    materializer_version: int = SESSION_PRODUCT_MATERIALIZER_VERSION
    materialized_at: str
    source_updated_at: str | None = None
    source_sort_key: float | None = None
    provider_name: str
    title: str | None = None
    first_message_at: str | None = None
    last_message_at: str | None = None
    primary_work_kind: str | None = None
    repo_paths: tuple[str, ...] = ()
    canonical_projects: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    auto_tags: tuple[str, ...] = ()
    message_count: int = 0
    work_event_count: int = 0
    word_count: int = 0
    tool_use_count: int = 0
    thinking_count: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    wall_duration_ms: int = 0
    payload: dict[str, Any]
    search_text: str

    @field_validator("conversation_id", "provider_name", "materialized_at", "search_text")
    @classmethod
    def profile_non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class SessionWorkEventRecord(BaseModel):
    """Durable canonical work-event row."""

    event_id: str
    conversation_id: ConversationId
    materializer_version: int = SESSION_PRODUCT_MATERIALIZER_VERSION
    materialized_at: str
    source_updated_at: str | None = None
    source_sort_key: float | None = None
    provider_name: str
    event_index: int
    kind: str
    confidence: float
    start_index: int
    end_index: int
    summary: str
    file_paths: tuple[str, ...] = ()
    tools_used: tuple[str, ...] = ()
    payload: dict[str, Any]
    search_text: str

    @field_validator(
        "event_id",
        "conversation_id",
        "materialized_at",
        "provider_name",
        "kind",
        "summary",
        "search_text",
    )
    @classmethod
    def work_event_non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class WorkThreadRecord(BaseModel):
    """Durable canonical work-thread row."""

    thread_id: str
    root_id: ConversationId
    materializer_version: int = SESSION_PRODUCT_MATERIALIZER_VERSION
    materialized_at: str
    start_time: str | None = None
    end_time: str | None = None
    dominant_project: str | None = None
    session_ids: tuple[str, ...] = ()
    session_count: int = 0
    depth: int = 0
    branch_count: int = 0
    total_messages: int = 0
    total_cost_usd: float = 0.0
    wall_duration_ms: int = 0
    work_event_breakdown: dict[str, int] | None = None
    payload: dict[str, Any]
    search_text: str

    @field_validator("thread_id", "root_id", "materialized_at", "search_text")
    @classmethod
    def work_thread_non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class SessionTagRollupRecord(BaseModel):
    """Durable provider/day tag rollup row used to build public tag products."""

    tag: str
    bucket_day: str
    provider_name: str
    materializer_version: int = SESSION_PRODUCT_MATERIALIZER_VERSION
    materialized_at: str
    source_updated_at: str | None = None
    source_sort_key: float | None = None
    conversation_count: int = 0
    explicit_count: int = 0
    auto_count: int = 0
    project_breakdown: dict[str, int]
    search_text: str

    @field_validator(
        "tag",
        "bucket_day",
        "provider_name",
        "materialized_at",
        "search_text",
    )
    @classmethod
    def tag_rollup_non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class DaySessionSummaryRecord(BaseModel):
    """Durable provider/day summary row used to build day/week products."""

    day: str
    provider_name: str
    materializer_version: int = SESSION_PRODUCT_MATERIALIZER_VERSION
    materialized_at: str
    source_updated_at: str | None = None
    source_sort_key: float | None = None
    conversation_count: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    total_wall_duration_ms: int = 0
    total_messages: int = 0
    total_words: int = 0
    work_event_breakdown: dict[str, int]
    projects_active: tuple[str, ...] = ()
    payload: dict[str, Any]
    search_text: str

    @field_validator("day", "provider_name", "materialized_at", "search_text")
    @classmethod
    def day_summary_non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class MaintenanceRunRecord(BaseModel):
    """Durable lineage record for maintenance previews and applies."""

    maintenance_run_id: str
    schema_version: int = MAINTENANCE_RUN_SCHEMA_VERSION
    executed_at: str
    mode: str
    preview: bool = False
    repair_selected: bool = False
    cleanup_selected: bool = False
    vacuum_requested: bool = False
    target_names: tuple[str, ...] = ()
    success: bool = True
    manifest: dict[str, Any]

    @field_validator("maintenance_run_id", "executed_at", "mode")
    @classmethod
    def maintenance_non_empty_string(cls, v: str) -> str:
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
    "MaintenanceRunRecord",
    "MAX_ATTACHMENT_SIZE",
    "MessageRecord",
    "PublicationRecord",
    "RawConversationRecord",
    "RunRecord",
    "SESSION_PRODUCT_MATERIALIZER_VERSION",
    "SessionTagRollupRecord",
    "SessionProfileRecord",
    "SessionWorkEventRecord",
    "DaySessionSummaryRecord",
    "WorkThreadRecord",
    "_json_array_or_none",
]
