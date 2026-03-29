"""Archive storage record models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from polylogue.lib.branch_type import BranchType
from polylogue.lib.hashing import hash_text
from polylogue.lib.roles import Role
from polylogue.lib.security import sanitize_path as _sanitize_path_helper
from polylogue.types import (
    AttachmentId,
    ContentBlockType,
    ContentHash,
    ConversationId,
    MessageId,
    Provider,
    SemanticBlockType,
)


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
