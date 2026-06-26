"""Claude AI session parser helpers."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ValidationError

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider, SessionKind, WebConstructType
from polylogue.sources.providers.claude_ai import ClaudeAISession

from ..base import (
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedWebConstruct,
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

CLAUDE_DESIGN_CHAT_INGEST_FLAG = "capture:claude-design-chat"
CLAUDE_TEMPORARY_CHAT_INGEST_FLAG = "capture:temporary-chat"


def looks_like_ai(payload: object) -> bool:
    return isinstance(payload, dict) and (
        isinstance(payload.get("chat_messages"), list) or _looks_like_design_chat(payload)
    )


def _looks_like_design_chat(payload: object) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("messages"), list)
        and ("project" in payload or "design_chats" in str(payload.get("source_path", "")))
        and not isinstance(payload.get("chat_messages"), list)
    )


def _session_ingest_flags(payload: Mapping[str, object], *, design_chat: bool = False) -> list[str]:
    flags: list[str] = []
    if design_chat:
        flags.append(CLAUDE_DESIGN_CHAT_INGEST_FLAG)
    if payload.get("is_temporary") is True:
        flags.append(CLAUDE_TEMPORARY_CHAT_INGEST_FLAG)
    return flags


def _session_kind(payload: Mapping[str, object]) -> SessionKind:
    return SessionKind.TEMPORARY if payload.get("is_temporary") is True else SessionKind.STANDARD


def _design_content_payload(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    nested = value.get("content")
    if isinstance(nested, Mapping):
        return nested
    return value


def _design_project_title(value: object) -> str | None:
    if isinstance(value, Mapping):
        name = value.get("name")
        return str(name) if name is not None else None
    return str(value) if value is not None else None


def _design_project_id(value: object) -> str | None:
    if isinstance(value, Mapping):
        project_id = value.get("uuid") or value.get("id")
        return str(project_id) if project_id is not None else None
    return None


def _parse_design_chat(payload: Mapping[str, object], fallback_id: str) -> ParsedSession:
    messages: list[ParsedMessage] = []
    attachments = []
    raw_messages = payload.get("messages")
    design_messages = raw_messages if isinstance(raw_messages, list) else []
    for position, raw_message in enumerate(design_messages):
        if not isinstance(raw_message, Mapping):
            continue
        content_payload = _design_content_payload(raw_message.get("content"))
        text = content_payload.get("content") or content_payload.get("text") or raw_message.get("text")
        if not isinstance(text, str) or not text:
            continue
        message_id = (
            raw_message.get("uuid") or content_payload.get("id") or raw_message.get("id") or f"design-{position}"
        )
        raw_role = raw_message.get("role") or content_payload.get("role")
        messages.append(
            ParsedMessage(
                provider_message_id=str(message_id),
                role=Role.normalize(str(raw_role or "unknown")),
                text=text,
                timestamp=str(content_payload.get("timestamp") or raw_message.get("created_at"))
                if content_payload.get("timestamp") or raw_message.get("created_at")
                else None,
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TEXT,
                        text=text,
                        web_constructs=[
                            ParsedWebConstruct(
                                construct_type=WebConstructType.CANVAS,
                                provider_key="claude_design_chat",
                                title=_design_project_title(payload.get("project")),
                                source_id=(
                                    str(content_payload.get("id"))
                                    if content_payload.get("id") is not None
                                    else _design_project_id(payload.get("project"))
                                ),
                            )
                        ],
                    )
                ],
                position=position,
                variant_index=0,
                is_active_path=True,
            )
        )
        raw_attachments = content_payload.get("attachments")
        design_attachments = raw_attachments if isinstance(raw_attachments, list) else []
        for att_idx, meta in enumerate(design_attachments, start=1):
            attachment = attachment_from_meta(meta, str(message_id), att_idx)
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
        provider_session_id=str(payload.get("uuid") or payload.get("id") or fallback_id),
        title=str(payload.get("title") or payload.get("name") or fallback_id),
        session_kind=_session_kind(payload),
        created_at=str(payload.get("created_at")) if payload.get("created_at") else None,
        updated_at=str(payload.get("updated_at")) if payload.get("updated_at") else None,
        messages=messages,
        active_leaf_message_provider_id=active_leaf_message_provider_id,
        attachments=attachments,
        ingest_flags=_session_ingest_flags(payload, design_chat=True),
    )


def parse_ai(payload: Mapping[str, object], fallback_id: str) -> ParsedSession:
    if _looks_like_design_chat(payload):
        return _parse_design_chat(payload, fallback_id)

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
            session_kind=_session_kind(payload),
            created_at=str(payload.get("created_at")) if payload.get("created_at") else None,
            updated_at=str(payload.get("updated_at")) if payload.get("updated_at") else None,
            messages=messages,
            active_leaf_message_provider_id=next(
                (message.provider_message_id for message in messages if message.is_active_leaf),
                None,
            ),
            attachments=attachments,
            ingest_flags=_session_ingest_flags(payload),
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
        session_kind=_session_kind(payload),
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=messages,
        active_leaf_message_provider_id=active_leaf_message_provider_id,
        attachments=attachments,
        ingest_flags=_session_ingest_flags(payload),
    )


__all__ = [
    "CLAUDE_DESIGN_CHAT_INGEST_FLAG",
    "CLAUDE_TEMPORARY_CHAT_INGEST_FLAG",
    "looks_like_ai",
    "parse_ai",
]
