"""Storage record models and validation."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from pydantic import BaseModel, field_validator

from polylogue.lib.json import dumps as json_dumps
from polylogue.lib.security import sanitize_path as _sanitize_path_helper
from polylogue.types import AttachmentId, ContentHash, ConversationId, MessageId

# Valid provider name pattern: starts with letter, contains only letters, numbers, hyphens, underscores
_PROVIDER_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")

# Maximum reasonable file size (1TB)
MAX_ATTACHMENT_SIZE = 1024 * 1024 * 1024 * 1024


class ConversationRecord(BaseModel):
    conversation_id: ConversationId
    provider_name: str
    provider_conversation_id: str
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    content_hash: ContentHash
    provider_meta: dict[str, object] | None = None
    metadata: dict[str, object] | None = None
    version: int = 1
    # Branching support: links conversations in session trees
    parent_conversation_id: ConversationId | None = None
    branch_type: str | None = None  # "continuation", "sidechain", "fork"
    # Link to raw source data (FK to raw_conversations.raw_id)
    raw_id: str | None = None

    @field_validator("provider_name")
    @classmethod
    def validate_provider_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("provider_name cannot be empty")
        v = v.strip()
        if not _PROVIDER_NAME_PATTERN.match(v):
            raise ValueError(
                f"provider_name '{v}' is invalid. Must start with a letter and "
                "contain only letters, numbers, hyphens, and underscores."
            )
        return v

    @field_validator("conversation_id", "provider_conversation_id", "content_hash")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class MessageRecord(BaseModel):
    message_id: MessageId
    conversation_id: ConversationId
    provider_message_id: str | None = None
    role: str | None = None
    text: str | None = None
    timestamp: str | None = None
    content_hash: ContentHash
    provider_meta: dict[str, object] | None = None
    version: int = 1
    # Branching support: links messages in conversation trees
    parent_message_id: MessageId | None = None
    branch_index: int = 0  # 0 = mainline, >0 = branch sibling position

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
    provider_name: str
    source_name: str | None = None  # Config source name (e.g., "inbox"), distinct from provider
    source_path: str
    source_index: int | None = None  # Position in bundle (e.g., conversations[3])
    raw_content: bytes  # Full JSON/JSONL bytes
    acquired_at: str  # ISO timestamp of acquisition
    file_mtime: str | None = None  # File modification time if available

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


class PlanResult(BaseModel):
    timestamp: int
    counts: dict[str, int]
    sources: list[str]
    cursors: dict[str, dict[str, Any]]


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


def _json_or_none(value: dict[str, object] | None) -> str | None:
    if value is None:
        return None
    return json_dumps(value)


def _make_ref_id(attachment_id: AttachmentId, conversation_id: ConversationId, message_id: MessageId | None) -> str:
    seed = f"{attachment_id}:{conversation_id}:{message_id or ''}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"ref-{digest}"




__all__ = [
    "ConversationRecord",
    "MessageRecord",
    "AttachmentRecord",
    "RunRecord",
    "RawConversationRecord",
    "MAX_ATTACHMENT_SIZE",
    "_json_or_none",
    "_make_ref_id",
]
