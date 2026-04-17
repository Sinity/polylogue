"""Claude AI conversation parser helpers."""

from __future__ import annotations

from pydantic import ValidationError

from polylogue.sources.providers.claude_ai import ClaudeAIConversation
from polylogue.types import ContentBlockType, Provider

from .base import (
    ParsedContentBlock,
    ParsedConversation,
    ParsedMessage,
    attachment_from_meta,
    content_blocks_from_segments,
)
from .claude_common import extract_messages_from_chat_messages, normalize_timestamp


def looks_like_ai(payload: object) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("chat_messages"), list)


def parse_ai(payload: dict[str, object], fallback_id: str) -> ParsedConversation:
    try:
        conv = ClaudeAIConversation.model_validate(payload)
    except ValidationError:
        chat_msgs = payload.get("chat_messages") or []
        if not isinstance(chat_msgs, list):
            chat_msgs = []
        messages, attachments = extract_messages_from_chat_messages(chat_msgs)
        title = payload.get("title") or payload.get("name") or fallback_id
        conv_id = payload.get("id") or payload.get("uuid") or payload.get("conversation_id")
        return ParsedConversation(
            provider_name=Provider.CLAUDE_AI,
            provider_conversation_id=str(conv_id or fallback_id),
            title=str(title),
            created_at=str(payload.get("created_at")) if payload.get("created_at") else None,
            updated_at=str(payload.get("updated_at")) if payload.get("updated_at") else None,
            messages=messages,
            attachments=attachments,
        )

    messages = []
    attachments = []
    for msg in conv.chat_messages:
        timestamp = normalize_timestamp(msg.created_at)
        if msg.text:
            raw_content = msg.model_dump().get("content")
            content_blocks = content_blocks_from_segments(raw_content) if isinstance(raw_content, list) else []
            if not content_blocks and msg.text:
                content_blocks = [ParsedContentBlock(type=ContentBlockType.TEXT, text=msg.text)]
            messages.append(
                ParsedMessage(
                    provider_message_id=msg.uuid,
                    role=msg.role_normalized,
                    text=msg.text,
                    timestamp=timestamp,
                    content_blocks=content_blocks,
                )
            )
        for att_idx, meta in enumerate(msg.attachments + msg.files, start=1):
            attachment = attachment_from_meta(meta, msg.uuid, att_idx)
            if attachment:
                attachments.append(attachment)

    return ParsedConversation(
        provider_name=Provider.CLAUDE_AI,
        provider_conversation_id=conv.uuid,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=messages,
        attachments=attachments,
    )


__all__ = ["looks_like_ai", "parse_ai"]
