"""Storage record models and validation."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, field_validator

from polylogue.errors import DatabaseError
from polylogue.lib.branch_type import BranchType
from polylogue.lib.hashing import hash_text
from polylogue.lib.json import dumps as json_dumps
from polylogue.lib.security import sanitize_path as _sanitize_path_helper
from polylogue.types import (
    AttachmentId,
    ArtifactSupportStatus,
    ContentBlockType,
    ContentHash,
    ConversationId,
    MessageId,
    PlanStage,
    Provider,
    SemanticBlockType,
    ValidationMode,
    ValidationStatus,
)

# Maximum reasonable file size (1TB)
MAX_ATTACHMENT_SIZE = 1024 * 1024 * 1024 * 1024

# SQLite SQLITE_MAX_LENGTH is 1 GB; keep raw blobs under 900 MB to leave headroom
MAX_RAW_CONTENT_SIZE = 900 * 1024 * 1024


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
    type: str
    text: str | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    tool_input: str | None = None  # JSON-serialized dict
    media_type: str | None = None
    metadata: str | None = None  # JSON-serialized dict
    semantic_type: str | None = None  # 'file_read'|'file_write'|'file_edit'|'shell'|'git'|'search'|'web'|'agent'|'subagent'|'thinking'|NULL

    @classmethod
    def make_id(cls, message_id: str, block_index: int) -> str:
        return f"blk-{hash_text(f'{message_id}:{block_index}')[:16]}"


class MessageRecord(BaseModel):
    message_id: MessageId
    conversation_id: ConversationId
    provider_message_id: str | None = None
    role: str | None = None
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

    @property
    def role_typed(self):
        """Typed Role enum derived from role string."""
        from polylogue.lib.roles import Role
        raw = (self.role or "").strip() or "unknown"
        return Role.normalize(raw)

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
    payload_provider: str | None = None  # Durable provider classification from decoded payload
    source_name: str | None = None  # Config source name (e.g., "inbox"), distinct from provider
    source_path: str
    source_index: int | None = None  # Position in bundle (e.g., conversations[3])
    raw_content: bytes  # Full JSON/JSONL bytes
    acquired_at: str  # ISO timestamp of acquisition
    file_mtime: str | None = None  # File modification time if available
    parsed_at: str | None = None  # ISO timestamp of last successful parse
    parse_error: str | None = None  # Error from last failed parse attempt
    validated_at: str | None = None  # ISO timestamp of last validation attempt
    validation_status: str | None = None  # "passed" | "failed" | "skipped"
    validation_error: str | None = None  # Error from last failed validation attempt
    validation_drift_count: int | None = None  # Drift warnings seen during last validation
    validation_provider: str | None = None  # Canonical provider used for validation schema
    validation_mode: str | None = None  # Validation mode used ("off" | "advisory" | "strict")

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


class ArtifactCohortSummary(BaseModel):
    """Aggregate summary for one observed artifact cohort."""

    provider_name: str
    payload_provider: Provider | None = None
    artifact_kind: str
    support_status: ArtifactSupportStatus
    cohort_id: str | None = None
    observation_count: int = 0
    unique_raw_ids: int = 0
    first_observed_at: str | None = None
    last_observed_at: str | None = None
    bundle_scope_count: int = 0
    sample_source_paths: list[str] = Field(default_factory=list)
    resolved_package_version: str | None = None
    resolved_element_kind: str | None = None
    resolution_reason: str | None = None
    link_group_count: int = 0
    linked_sidecar_count: int = 0

    @field_validator("payload_provider", mode="before")
    @classmethod
    def coerce_cohort_payload_provider(cls, v: object) -> Provider | None:
        if v is None:
            return None
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v))

    @field_validator("support_status", mode="before")
    @classmethod
    def coerce_cohort_support_status(cls, v: object) -> ArtifactSupportStatus:
        return ArtifactSupportStatus.from_string(str(v))


class PlanResult(BaseModel):
    timestamp: int
    stage: str = "all"
    counts: dict[str, int]
    details: dict[str, int] = Field(default_factory=dict)
    sources: list[str]
    cursors: dict[str, dict[str, Any]]


class RawConversationState(BaseModel):
    raw_id: str
    source_name: str | None = None
    source_path: str | None = None
    parsed_at: str | None = None
    parse_error: str | None = None
    payload_provider: str | None = None
    validation_status: str | None = None
    validation_provider: str | None = None


class RunResult(BaseModel):
    run_id: str
    counts: dict[str, int]
    drift: dict[str, dict[str, int]]
    indexed: bool
    index_error: str | None
    duration_ms: int
    render_failures: list[dict[str, str]] = []


class ExistingConversation(BaseModel):
    conversation_id: str
    content_hash: str


@dataclass(frozen=True)
class ConversationRenderProjection:
    """Repository-owned render projection preserving raw attachment layout."""

    conversation: ConversationRecord
    messages: list[MessageRecord]
    attachments: list[AttachmentRecord]


def _json_or_none(value: dict[str, object] | None) -> str | None:
    if value is None:
        return None
    return json_dumps(value)


def _make_ref_id(attachment_id: AttachmentId, conversation_id: ConversationId, message_id: MessageId | None) -> str:
    seed = f"{attachment_id}:{conversation_id}:{message_id or ''}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"ref-{digest}"




def _parse_json(raw: str | None, *, field: str = "", record_id: str = "") -> Any:
    """Parse a JSON string with diagnostic context on failure."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DatabaseError(
            f"Corrupt JSON in {field} for {record_id}: {exc} (value starts: {raw[:80]!r})"
        ) from exc


def _row_get(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    """Get a column value, returning default if the column doesn't exist.

    Handles schema version differences where optional columns may be absent.
    """
    try:
        return row[key]
    except (KeyError, IndexError):
        return default


def _row_to_conversation(row: sqlite3.Row) -> ConversationRecord:
    """Map a SQLite row to a ConversationRecord."""
    return ConversationRecord(
        conversation_id=row["conversation_id"],
        provider_name=row["provider_name"],
        provider_conversation_id=row["provider_conversation_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        sort_key=_row_get(row, "sort_key"),
        content_hash=row["content_hash"],
        provider_meta=_parse_json(row["provider_meta"], field="provider_meta", record_id=row["conversation_id"]),
        metadata=_parse_json(row["metadata"], field="metadata", record_id=row["conversation_id"]),
        version=row["version"],
        parent_conversation_id=_row_get(row, "parent_conversation_id"),
        branch_type=_row_get(row, "branch_type"),
        raw_id=_row_get(row, "raw_id"),
    )


def _row_to_message(row: sqlite3.Row) -> MessageRecord:
    """Map a SQLite row to a MessageRecord."""
    return MessageRecord(
        message_id=row["message_id"],
        conversation_id=row["conversation_id"],
        provider_message_id=_row_get(row, "provider_message_id"),
        role=_row_get(row, "role"),
        text=_row_get(row, "text"),
        sort_key=_row_get(row, "sort_key"),
        content_hash=row["content_hash"],
        version=row["version"],
        parent_message_id=_row_get(row, "parent_message_id"),
        branch_index=_row_get(row, "branch_index", 0) or 0,
        provider_name=_row_get(row, "provider_name", '') or '',
        word_count=_row_get(row, "word_count", 0) or 0,
        has_tool_use=_row_get(row, "has_tool_use", 0) or 0,
        has_thinking=_row_get(row, "has_thinking", 0) or 0,
    )


def _row_to_content_block(row: sqlite3.Row) -> ContentBlockRecord:
    """Map a SQLite row to a ContentBlockRecord."""
    return ContentBlockRecord(
        block_id=row["block_id"],
        message_id=MessageId(row["message_id"]),
        conversation_id=ConversationId(row["conversation_id"]),
        block_index=row["block_index"],
        type=row["type"],
        text=_row_get(row, "text"),
        tool_name=_row_get(row, "tool_name"),
        tool_id=_row_get(row, "tool_id"),
        tool_input=_row_get(row, "tool_input"),
        media_type=_row_get(row, "media_type"),
        metadata=_row_get(row, "metadata"),
        semantic_type=_row_get(row, "semantic_type"),
    )


def _row_to_raw_conversation(row: sqlite3.Row) -> RawConversationRecord:
    """Map a SQLite row to a RawConversationRecord."""
    return RawConversationRecord(
        raw_id=row["raw_id"],
        provider_name=row["provider_name"],
        payload_provider=_row_get(row, "payload_provider"),
        source_name=row["source_name"],
        source_path=row["source_path"],
        source_index=row["source_index"],
        raw_content=row["raw_content"],
        acquired_at=row["acquired_at"],
        file_mtime=row["file_mtime"],
        parsed_at=_row_get(row, "parsed_at"),
        parse_error=_row_get(row, "parse_error"),
        validated_at=_row_get(row, "validated_at"),
        validation_status=_row_get(row, "validation_status"),
        validation_error=_row_get(row, "validation_error"),
        validation_drift_count=_row_get(row, "validation_drift_count"),
        validation_provider=_row_get(row, "validation_provider"),
        validation_mode=_row_get(row, "validation_mode"),
    )


__all__ = [
    "AttachmentRecord",
    "ArtifactCohortSummary",
    "ArtifactObservationRecord",
    "ContentBlockRecord",
    "ConversationRecord",
    "MAX_ATTACHMENT_SIZE",
    "MessageRecord",
    "RawConversationRecord",
    "RunRecord",
    "_row_to_content_block",
]
