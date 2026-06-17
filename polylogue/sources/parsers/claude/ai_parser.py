"""Claude AI session parser helpers."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ValidationError

from polylogue.core.enums import BlockType, Provider
from polylogue.sources.providers.claude_ai import ClaudeAISession

from ..base import (
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    attachment_from_meta,
    content_blocks_from_segments,
)
from .common import (
    _message_duration_ms,
    _message_model_effort,
    _message_model_name,
    extract_messages_from_chat_messages,
    normalize_timestamp,
)


def looks_like_ai(payload: object) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("chat_messages"), list)


def parse_ai(payload: Mapping[str, object], fallback_id: str) -> ParsedSession:
    try:
        conv = ClaudeAISession.model_validate(payload)
    except ValidationError:
        chat_msgs = payload.get("chat_messages") or []
        if not isinstance(chat_msgs, list):
            chat_msgs = []
        messages, attachments = extract_messages_from_chat_messages(chat_msgs)
        title = payload.get("title") or payload.get("name") or fallback_id
        conv_id = payload.get("id") or payload.get("uuid") or payload.get("conversation_id")
        return ParsedSession(
            source_name=Provider.CLAUDE_AI,
            provider_session_id=str(conv_id or fallback_id),
            title=str(title),
            created_at=str(payload.get("created_at")) if payload.get("created_at") else None,
            updated_at=str(payload.get("updated_at")) if payload.get("updated_at") else None,
            messages=messages,
            active_leaf_message_provider_id=next(
                (message.provider_message_id for message in messages if message.is_active_leaf),
                None,
            ),
            attachments=attachments,
        )

    messages = []
    attachments = []
    message_position = 0
    for msg in conv.chat_messages:
        timestamp = normalize_timestamp(msg.created_at)
        if msg.text:
            raw_message = msg.model_dump()
            raw_content = raw_message.get("content")
            content_blocks = content_blocks_from_segments(raw_content) if isinstance(raw_content, list) else []
            if not content_blocks and msg.text:
                content_blocks = [ParsedContentBlock(type=BlockType.TEXT, text=msg.text)]
            messages.append(
                ParsedMessage(
                    provider_message_id=msg.uuid,
                    role=msg.role_normalized,
                    text=msg.text,
                    timestamp=timestamp,
                    blocks=content_blocks,
                    position=message_position,
                    variant_index=0,
                    is_active_path=True,
                    model_name=_message_model_name(raw_message),
                    model_effort=_message_model_effort(raw_message),
                    duration_ms=_message_duration_ms(raw_message),
                )
            )
            message_position += 1
        for att_idx, meta in enumerate(msg.attachments + msg.files, start=1):
            attachment = attachment_from_meta(meta, msg.uuid, att_idx)
            if attachment:
                attachments.append(attachment)
    active_leaf_message_provider_id = messages[-1].provider_message_id if messages else None
    if active_leaf_message_provider_id is not None:
        messages = [
            message.model_copy(
                update={"is_active_leaf": message.provider_message_id == active_leaf_message_provider_id}
            )
            for message in messages
        ]

    return ParsedSession(
        source_name=Provider.CLAUDE_AI,
        provider_session_id=conv.uuid,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=messages,
        active_leaf_message_provider_id=active_leaf_message_provider_id,
        attachments=attachments,
    )


__all__ = ["looks_like_ai", "parse_ai"]
