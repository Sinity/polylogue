"""Base parser models and message extraction utilities."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from polylogue.lib.branch_type import BranchType
from polylogue.lib.hashing import hash_text
from polylogue.lib.roles import Role, normalize_role
from polylogue.lib.security import sanitize_path as _sanitize_path_helper
from polylogue.types import Provider

__all__ = [
    "ParsedContentBlock",
    "ParsedMessage",
    "ParsedAttachment",
    "ParsedConversation",
    "RawConversationData",
    "normalize_role",
    "content_blocks_from_segments",
    "extract_messages_from_list",
    "attachment_from_meta",
]


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

    type: str
    text: str | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    tool_input: dict[str, object] | None = None
    media_type: str | None = None
    metadata: dict[str, object] | None = None


class ParsedMessage(BaseModel):
    provider_message_id: str
    role: Role
    text: str | None = None  # Concatenated text from text-type blocks (for FTS5 and rendering)
    timestamp: str | None = None
    content_blocks: list[ParsedContentBlock] = Field(default_factory=list)
    # raw provider API data — stored in message_meta table, not messages
    provider_meta: dict[str, object] | None = None
    parent_message_provider_id: str | None = None
    branch_index: int = 0

    @field_validator("role", mode="before")
    @classmethod
    def coerce_role(cls, v: object) -> Role:
        if isinstance(v, Role):
            return v
        return Role.normalize(str(v) if v is not None else "unknown")


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

        # Remove null bytes
        v = v.replace("\x00", "")

        # Remove control characters (ASCII < 32 and 127)
        v = "".join(c for c in v if ord(c) >= 32 and ord(c) != 127)

        # Reject dots-only names
        if v and v.strip(".") == "":
            # Return a default name instead of empty
            v = "file"

        return v if v else None


class ParsedConversation(BaseModel):
    provider_name: Provider
    provider_conversation_id: str
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    messages: list[ParsedMessage]
    attachments: list[ParsedAttachment] = Field(default_factory=list)
    provider_meta: dict[str, object] | None = None
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

    Used to pass raw data through the parsing pipeline alongside
    the parsed conversation, enabling honest database-driven testing.
    """
    raw_bytes: bytes
    source_path: str
    source_index: int | None = None
    file_mtime: str | None = None
    provider_hint: str | None = None  # Provider detected from path/content


def content_blocks_from_segments(content: object) -> list[ParsedContentBlock]:
    """Convert raw API content (str, list, dict) to ParsedContentBlock list.

    Handles the common Claude/Codex content format:
    - str: single text block
    - list: typed segment dicts (text, thinking, tool_use, tool_result, image)
    - other: returns empty list
    """
    if isinstance(content, str):
        return [ParsedContentBlock(type="text", text=content)] if content else []
    if not isinstance(content, list):
        return []
    blocks: list[ParsedContentBlock] = []
    for seg in content:
        if isinstance(seg, str):
            if seg:
                blocks.append(ParsedContentBlock(type="text", text=seg))
            continue
        if not isinstance(seg, dict):
            continue
        seg_type = seg.get("type", "text")
        if seg_type == "thinking":
            text = seg.get("thinking") or seg.get("text") or ""
            if text:
                blocks.append(ParsedContentBlock(type="thinking", text=text))
        elif seg_type == "tool_use":
            blocks.append(ParsedContentBlock(
                type="tool_use",
                tool_name=seg.get("name"),
                tool_id=seg.get("id"),
                tool_input=seg.get("input") if isinstance(seg.get("input"), dict) else None,
            ))
        elif seg_type == "tool_result":
            result_content = seg.get("content")
            result_text = None
            if isinstance(result_content, str):
                result_text = result_content
            elif isinstance(result_content, list):
                text_parts = [
                    b.get("text", "") for b in result_content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                result_text = "\n".join(p for p in text_parts if p) or None
            blocks.append(ParsedContentBlock(
                type="tool_result",
                tool_id=seg.get("tool_use_id"),
                text=result_text,
            ))
        elif seg_type in ("image", "document"):
            blocks.append(ParsedContentBlock(
                type=seg_type,
                media_type=seg.get("media_type"),
                metadata={k: v for k, v in seg.items() if k not in ("type", "media_type")},
            ))
        else:
            # Generic text block
            text = seg.get("text") or seg.get("content") or ""
            if text:
                blocks.append(ParsedContentBlock(type="text", text=str(text)))
    return blocks


def _make_attachment_id(seed: str) -> str:
    return f"att-{hash_text(seed)[:12]}"


def attachment_from_meta(meta: object, message_id: str | None, index: int) -> ParsedAttachment | None:
    if not isinstance(meta, dict):
        return None
    attachment_id = meta.get("id") or meta.get("file_id") or meta.get("fileId") or meta.get("uuid")
    name = meta.get("name") or meta.get("filename") or meta.get("file_name")
    if not attachment_id:
        if not name:
            return None
        seed = f"{message_id or 'msg'}:{name}:{index}"
        attachment_id = _make_attachment_id(seed)
    size_raw = meta.get("size") or meta.get("size_bytes") or meta.get("sizeBytes")
    size_bytes = None
    if isinstance(size_raw, (int, str)):
        try:
            size_bytes = int(size_raw)
        except ValueError:
            size_bytes = None
    mime_type = meta.get("mimeType") or meta.get("mime_type") or meta.get("content_type")
    return ParsedAttachment(
        provider_attachment_id=str(attachment_id),
        message_provider_id=message_id,
        name=name,
        mime_type=mime_type if isinstance(mime_type, str) else None,
        size_bytes=size_bytes,
        path=None,
        provider_meta=meta,
    )


def extract_messages_from_list(items: list[object]) -> list[ParsedMessage]:
    messages: list[ParsedMessage] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        # Support various schema conventions
        message_val = item.get("message")
        payload = message_val if isinstance(message_val, dict) else item

        role = Role.normalize(
            str(
                payload.get("role")
                or item.get("role")
                or payload.get("sender")
                or item.get("sender")
                or payload.get("author")
                or item.get("author")
                or "unknown"
            )
        )

        timestamp = (
            item.get("timestamp")
            or payload.get("timestamp")
            or payload.get("created_at")
            or item.get("created_at")
            or payload.get("create_time")
            or item.get("create_time")
        )

        text = None
        text_val = payload.get("text")
        if text_val is not None and isinstance(text_val, str):
            text = text_val
        else:
            content = payload.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, dict):
                parts = content.get("parts")
                if isinstance(parts, list):
                    text = "\n".join(str(part) for part in parts)
                else:
                    text_dict_val = content.get("text")
                    if text_dict_val is not None and isinstance(text_dict_val, str):
                        text = text_dict_val
            elif isinstance(content, list):
                # Simple concatenation for list of strings or dicts
                parts = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict):
                        part_text = part.get("text")
                        parts.append(part_text if isinstance(part_text, str) else "")
                text = "\n".join(parts)

        if text:
            msg_id = str(payload.get("id") or payload.get("uuid") or item.get("uuid") or item.get("id") or f"msg-{idx}")
            messages.append(
                ParsedMessage(
                    provider_message_id=msg_id,
                    role=role,
                    text=text,
                    timestamp=str(timestamp) if timestamp is not None else None,
                    provider_meta={"raw": item},
                )
            )
    return messages
