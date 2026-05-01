"""Canonicalization helpers for raw-to-record transformation."""

from __future__ import annotations

from polylogue.archive.viewport.models import ContentBlock
from polylogue.schemas.unified.unified import harmonize_parsed_message
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedConversation, ParsedMessage
from polylogue.types import ContentBlockType


def parsed_block_from_harmonized(block: ContentBlock) -> ParsedContentBlock | None:
    metadata: dict[str, object] | None = dict(block.raw) if isinstance(block.raw, dict) and block.raw else None

    if block.type.name == "TOOL_USE" and block.tool_call is not None:
        return ParsedContentBlock(
            type=ContentBlockType.TOOL_USE,
            text=block.text,
            tool_name=block.tool_call.name,
            tool_id=block.tool_call.id,
            tool_input=block.tool_call.input or None,
            metadata=metadata,
        )
    if block.type.name == "TOOL_RESULT":
        tool_id = None
        if isinstance(block.raw, dict):
            raw_tool_id = block.raw.get("tool_use_id") or block.raw.get("tool_id")
            if isinstance(raw_tool_id, str) and raw_tool_id:
                tool_id = raw_tool_id
        return ParsedContentBlock(
            type=ContentBlockType.TOOL_RESULT, text=block.text, tool_id=tool_id, metadata=metadata
        )
    if block.type.name == "CODE":
        if block.language:
            metadata = dict(metadata or {})
            metadata.setdefault("language", block.language)
        return ParsedContentBlock(type=ContentBlockType.CODE, text=block.text, metadata=metadata)
    if block.type.name == "THINKING":
        return ParsedContentBlock(type=ContentBlockType.THINKING, text=block.text, metadata=metadata)
    if block.type.name == "IMAGE":
        return ParsedContentBlock(
            type=ContentBlockType.IMAGE,
            text=block.text,
            media_type=block.mime_type,
            metadata=metadata,
        )
    if block.type.name in {"FILE", "AUDIO", "VIDEO"}:
        return ParsedContentBlock(
            type=ContentBlockType.DOCUMENT,
            text=block.text,
            media_type=block.mime_type,
            metadata=metadata,
        )
    if block.type.name in {"TEXT", "SYSTEM", "ERROR", "UNKNOWN"}:
        return ParsedContentBlock(type=ContentBlockType.TEXT, text=block.text, metadata=metadata)
    return None


def canonicalize_message_content(provider_name: str, message: ParsedMessage) -> ParsedMessage:
    harmonized = harmonize_parsed_message(
        provider_name,
        message.provider_meta,
        message_id=message.provider_message_id,
        role=str(message.role),
        text=message.text,
        timestamp=message.timestamp,
    )
    if harmonized is None:
        return message

    updates: dict[str, object] = {}
    if not message.text and harmonized.text:
        updates["text"] = harmonized.text
    if not message.content_blocks and harmonized.content_blocks:
        content_blocks = [
            parsed_block
            for block in harmonized.content_blocks
            if (parsed_block := parsed_block_from_harmonized(block)) is not None
        ]
        if content_blocks:
            updates["content_blocks"] = content_blocks
    if not updates:
        return message
    return message.model_copy(update=updates)


def canonicalize_conversation_content(convo: ParsedConversation) -> ParsedConversation:
    messages = [canonicalize_message_content(str(convo.provider_name), message) for message in convo.messages]
    if all(original == updated for original, updated in zip(convo.messages, messages, strict=True)):
        return convo
    return convo.model_copy(update={"messages": messages})


__all__ = [
    "canonicalize_conversation_content",
    "canonicalize_message_content",
    "parsed_block_from_harmonized",
]
