"""Archive storage record models."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from polylogue.archive.conversation.branch_type import BranchType
from polylogue.archive.message.types import MessageType
from polylogue.core.hashing import hash_text
from polylogue.core.json import json_document
from polylogue.core.security import sanitize_path as _sanitize_path_helper
from polylogue.lib.roles import Role
from polylogue.storage.run_state import RunCounts, RunCountsPayload
from polylogue.types import (
    AttachmentId,
    ContentBlockType,
    ContentHash,
    ConversationId,
    MessageId,
    Provider,
    SemanticBlockType,
)

JSONObject = dict[str, object]


def _coerce_json_object(value: object) -> JSONObject | None:
    if value is None:
        return None
    document = json_document(value)
    if not document:
        return None
    result: JSONObject = {}
    for key, item in document.items():
        result[key] = item
    return result


class ConversationRecord(BaseModel):
    conversation_id: ConversationId
    provider_name: str
    provider_conversation_id: str
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    sort_key: float | None = None
    content_hash: ContentHash
    provider_meta: JSONObject | None = None
    metadata: JSONObject | None = None
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

    @field_validator("provider_meta", "metadata", mode="before")
    @classmethod
    def coerce_json_document(cls, value: object) -> JSONObject | None:
        return _coerce_json_object(value)


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
    has_paste: int = 0
    message_type: MessageType = MessageType.MESSAGE

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

    @field_validator("message_type", mode="before")
    @classmethod
    def coerce_message_type(cls, v: object) -> MessageType:
        return MessageType.normalize(v)

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
    provider_meta: JSONObject | None = None

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

    @field_validator("provider_meta", mode="before")
    @classmethod
    def coerce_provider_meta(cls, value: object) -> JSONObject | None:
        return _coerce_json_object(value)


def _coerce_run_counts_payload(value: object) -> RunCountsPayload | None:
    if value is None:
        return None
    return RunCounts.model_validate(value).to_payload()


class RunRecord(BaseModel):
    run_id: str
    timestamp: str
    plan_snapshot: JSONObject | None = None
    counts: RunCountsPayload | None = None
    drift: JSONObject | None = None
    indexed: bool | None = None
    duration_ms: int | None = None

    @field_validator("plan_snapshot", mode="before")
    @classmethod
    def coerce_plan_snapshot(cls, value: object) -> JSONObject | None:
        return _coerce_json_object(value)

    @field_validator("counts", mode="before")
    @classmethod
    def coerce_counts(cls, value: object) -> RunCountsPayload | None:
        return _coerce_run_counts_payload(value)

    @field_validator("drift", mode="before")
    @classmethod
    def coerce_drift(cls, value: object) -> JSONObject | None:
        return _coerce_json_object(value)


class PublicationRecord(BaseModel):
    publication_id: str
    publication_kind: str
    generated_at: str
    output_dir: str
    duration_ms: int | None = None
    manifest: JSONObject

    @field_validator("publication_id", "publication_kind", "generated_at", "output_dir")
    @classmethod
    def publication_non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v

    @field_validator("manifest", mode="before")
    @classmethod
    def coerce_manifest(cls, value: object) -> JSONObject:
        return _coerce_json_object(value) or {}
