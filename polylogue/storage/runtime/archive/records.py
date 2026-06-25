"""Archive storage record models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, MaterialOrigin, Origin, SemanticBlockType, SessionKind
from polylogue.core.hashing import hash_text
from polylogue.core.json import json_document
from polylogue.core.security import sanitize_path as _sanitize_path_helper
from polylogue.core.timestamps import canonical_timestamp_text
from polylogue.types import AttachmentId, ContentHash, MessageId, SessionEventId, SessionId

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


class SessionRecord(BaseModel):
    session_id: SessionId
    native_id: str
    origin: Origin
    title: str | None = None
    session_kind: SessionKind = SessionKind.STANDARD
    created_at: str | None = None
    updated_at: str | None = None
    sort_key: float | None = None
    content_hash: ContentHash
    metadata: JSONObject | None = None
    version: int = 1
    parent_session_id: SessionId | None = None
    branch_type: BranchType | None = None
    raw_id: str | None = None
    working_directories_json: str | None = None
    git_branch: str | None = None
    git_repository_url: str | None = None

    @field_validator("origin", mode="before")
    @classmethod
    def coerce_origin(cls, value: object) -> Origin:
        if isinstance(value, Origin):
            return value
        return Origin.from_string(str(value))

    @field_validator("session_kind", mode="before")
    @classmethod
    def coerce_session_kind(cls, value: object) -> SessionKind:
        return SessionKind.normalize(value)

    @field_validator("session_id", "native_id", "content_hash")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v

    @field_validator("metadata", mode="before")
    @classmethod
    def coerce_json_document(cls, value: object) -> JSONObject | None:
        return _coerce_json_object(value)

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def coerce_archive_timestamp(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"Unsupported archive timestamp: {value!r}")
        if not isinstance(value, (str, int, float, datetime)):
            raise ValueError(f"Unsupported archive timestamp: {value!r}")
        canonical = canonical_timestamp_text(value)
        if canonical is None:
            raise ValueError(f"Unsupported archive timestamp: {value!r}")
        return canonical


class BlockRecord(BaseModel):
    block_id: str
    message_id: MessageId
    session_id: SessionId
    block_index: int
    type: BlockType
    text: str | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    tool_input: str | None = None
    metadata: str | None = None
    semantic_type: SemanticBlockType | None = None

    @field_validator("type", mode="before")
    @classmethod
    def coerce_block_type(cls, v: object) -> BlockType:
        return BlockType.from_string(str(v))

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
    model_config = ConfigDict(protected_namespaces=())

    message_id: MessageId
    session_id: SessionId
    provider_message_id: str | None = None
    role: Role | None = None
    text: str | None = None
    sort_key: float | None = None
    content_hash: ContentHash
    version: int = 1
    parent_message_id: MessageId | None = None
    branch_index: int = 0
    blocks: list[BlockRecord] = Field(default_factory=list)
    source_name: str = ""
    word_count: int = 0
    has_tool_use: int = 0
    has_thinking: int = 0
    has_paste: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    duration_ms: int | None = None
    model_name: str | None = None
    message_type: MessageType = MessageType.MESSAGE
    material_origin: MaterialOrigin = MaterialOrigin.UNKNOWN
    paste_boundary_state: str | None = None

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

    @field_validator("material_origin", mode="before")
    @classmethod
    def coerce_material_origin(cls, v: object) -> MaterialOrigin:
        return MaterialOrigin.normalize(v)

    @property
    def role_typed(self) -> Role:
        return self.role or Role.UNKNOWN

    @field_validator("message_id", "session_id", "content_hash")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class AttachmentRecord(BaseModel):
    attachment_id: AttachmentId
    session_id: SessionId
    message_id: MessageId | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    path: str | None = None
    display_name: str | None = None
    source_url: str | None = None
    caption: str | None = None
    attachment_native_id: str | None = None
    file_native_id: str | None = None
    drive_native_id: str | None = None
    upload_origin: str | None = None

    @field_validator("attachment_id", "session_id")
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


class SessionEventRecord(BaseModel):
    event_id: SessionEventId
    session_id: SessionId
    origin: str
    event_index: int
    event_type: str
    timestamp: str | None = None
    sort_key: float | None = None
    payload: JSONObject = Field(default_factory=dict)
    source_message_id: MessageId | None = None
    raw_id: str | None = None
    materializer_version: int = 1

    @field_validator("event_id", "session_id", "origin", "event_type")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v

    @field_validator("payload", mode="before")
    @classmethod
    def coerce_payload(cls, value: object) -> JSONObject:
        return _coerce_json_object(value) or {}
