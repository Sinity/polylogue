"""Typed parser contracts shared across provider parsers."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, Field, field_validator

from polylogue.archive.message.types import MessageType
from polylogue.lib.conversation.branch_type import BranchType
from polylogue.lib.roles import Role
from polylogue.lib.security import sanitize_path as _sanitize_path_helper
from polylogue.types import ContentBlockType, Provider


class ParsedContentBlock(BaseModel):
    """A single structured content block within a parsed message.

    Block types:
    - text: regular text content
    - thinking: extended reasoning traces
    - tool_use: tool invocation (tool_name, tool_id, tool_input required)
    - tool_result: tool response (tool_id, text required)
    - image: image reference (media_type, metadata for asset pointer)
    - code: code block, language-detected (text required)
    - document: document reference
    """

    type: ContentBlockType
    text: str | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    tool_input: Mapping[str, object] | None = None
    media_type: str | None = None
    metadata: dict[str, object] | None = None

    @field_validator("type", mode="before")
    @classmethod
    def coerce_type(cls, v: object) -> ContentBlockType:
        return ContentBlockType.from_string(str(v))


class ParsedMessage(BaseModel):
    provider_message_id: str
    role: Role
    text: str | None = None
    timestamp: str | None = None
    content_blocks: list[ParsedContentBlock] = Field(default_factory=list)
    message_type: MessageType = MessageType.MESSAGE
    # Optional transient parser metadata for direct parser consumers.
    # Canonical persistence uses content_blocks for messages and provider_meta for conversations/attachments.
    provider_meta: dict[str, object] | None = None
    parent_message_provider_id: str | None = None
    branch_index: int = 0

    @field_validator("role", mode="before")
    @classmethod
    def coerce_role(cls, v: object) -> Role:
        if isinstance(v, Role):
            return v
        return Role.normalize(str(v) if v is not None else "unknown")

    @field_validator("message_type", mode="before")
    @classmethod
    def coerce_message_type(cls, v: object) -> MessageType:
        return MessageType.normalize(v)


class ParsedAttachment(BaseModel):
    provider_attachment_id: str
    message_provider_id: str | None = None
    name: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    path: str | None = None
    provider_meta: dict[str, object] | None = None

    @field_validator("path")
    @classmethod
    def sanitize_path(cls, v: str | None) -> str | None:
        """Sanitize path to prevent traversal attacks and other security issues."""
        return _sanitize_path_helper(v)

    @field_validator("name")
    @classmethod
    def sanitize_name(cls, v: str | None) -> str | None:
        """Sanitize filename to prevent control chars and invalid names."""
        if v is None:
            return v

        v = v.replace("\x00", "")
        v = "".join(c for c in v if ord(c) >= 32 and ord(c) != 127)

        if v and v.strip(".") == "":
            v = "file"

        return v if v else None


class ParsedProviderEvent(BaseModel):
    """Non-message semantic artifact from a provider (compaction, turn context, etc.)."""

    event_type: str  # "compaction", "turn_context", etc.
    timestamp: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)


class ParsedConversation(BaseModel):
    provider_name: Provider
    provider_conversation_id: str
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    messages: list[ParsedMessage]
    attachments: list[ParsedAttachment] = Field(default_factory=list)
    provider_meta: dict[str, object] | None = None
    provider_events: list[ParsedProviderEvent] = Field(default_factory=list)
    parent_conversation_provider_id: str | None = None
    branch_type: BranchType | None = None

    @field_validator("provider_name", mode="before")
    @classmethod
    def coerce_provider(cls, v: object) -> Provider:
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v) if v is not None else "unknown")


class RawConversationData(BaseModel):
    """Container for raw conversation bytes with metadata.

    When ``blob_hash`` is set, the content has been written to the blob
    store and ``raw_bytes`` may be empty (only a detection prefix was
    needed). Consumers should load from the blob store using ``blob_hash``.
    """

    raw_bytes: bytes = b""
    source_path: str
    source_index: int | None = None
    file_mtime: str | None = None
    provider_hint: Provider | None = None
    blob_hash: str | None = None
    blob_size: int | None = None

    @field_validator("provider_hint", mode="before")
    @classmethod
    def coerce_provider_hint(cls, v: object) -> Provider | None:
        if v is None:
            return None
        return Provider.from_string(str(v))
