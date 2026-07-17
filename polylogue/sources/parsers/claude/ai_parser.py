"""Claude AI session parser helpers."""

from __future__ import annotations

from collections.abc import Mapping

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider, SessionKind, WebConstructType

from ..base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
    ParsedWebConstruct,
    attachment_from_meta,
)
from .common import (
    _first_identity_field,
    _first_string_field,
    _message_model_effort,
    _message_model_name,
    _thinking_configuration,
    normalize_chat_messages,
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


def _session_timestamp(payload: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float, str)):
            normalized = normalize_timestamp(value)
            if normalized is not None:
                return normalized
    return None


def _merge_session_attachments(
    message_attachments: list[ParsedAttachment],
    payload: Mapping[str, object],
) -> list[ParsedAttachment]:
    attachments = list(message_attachments)
    top_level: list[object] = []
    for key in ("attachments", "files"):
        value = payload.get(key)
        if isinstance(value, list):
            top_level.extend(value)
    for index, meta in enumerate(top_level, start=1):
        attachment = attachment_from_meta(meta, None, index)
        if attachment is not None:
            attachments.append(attachment)

    merged: dict[str, ParsedAttachment] = {}
    for candidate in attachments:
        existing = merged.get(candidate.provider_attachment_id)
        if existing is None:
            merged[candidate.provider_attachment_id] = candidate
            continue
        preferred = candidate if candidate.inline_bytes is not None and existing.inline_bytes is None else existing
        other = existing if preferred is candidate else candidate
        merged[candidate.provider_attachment_id] = preferred.model_copy(
            update={
                "message_provider_id": preferred.message_provider_id or other.message_provider_id,
                "name": preferred.name or other.name,
                "mime_type": preferred.mime_type or other.mime_type,
                "size_bytes": preferred.size_bytes if preferred.size_bytes is not None else other.size_bytes,
                "provider_file_id": preferred.provider_file_id or other.provider_file_id,
                "provider_drive_id": preferred.provider_drive_id or other.provider_drive_id,
            }
        )
    return list(merged.values())


def parse_ai(payload: Mapping[str, object], fallback_id: str) -> ParsedSession:
    if _looks_like_design_chat(payload):
        return _parse_design_chat(payload, fallback_id)

    raw_messages = payload.get("chat_messages")
    chat_messages = raw_messages if isinstance(raw_messages, list) else []
    created_at = _session_timestamp(payload, "created_at", "create_time", "timestamp")
    updated_at = _session_timestamp(payload, "updated_at", "update_time")
    session_model = _message_model_name(payload)
    session_effort = _message_model_effort(payload)
    active_leaf_message_provider_id = _first_identity_field(
        payload,
        "current_leaf_message_uuid",
        "current_leaf_message_id",
        "active_leaf_message_uuid",
        "active_leaf_message_id",
        "current_message_uuid",
        "current_message_id",
        "current_node",
    )
    normalized = normalize_chat_messages(
        chat_messages,
        session_model=session_model,
        session_effort=session_effort,
        session_thinking_configuration=_thinking_configuration(payload),
        session_created_at=created_at,
        session_updated_at=updated_at,
        active_leaf_message_provider_id=active_leaf_message_provider_id,
    )

    session_events = list(normalized.session_events)
    provider_status = _first_string_field(payload, "status", "conversation_status")
    if provider_status is not None:
        session_events.append(
            ParsedSessionEvent(
                event_type="provider_session_status",
                timestamp=updated_at or created_at,
                payload={"status": provider_status},
            )
        )

    conversation_id = _first_identity_field(payload, "uuid", "id", "conversation_id", "conversationId")
    title = payload.get("title") or payload.get("name") or fallback_id
    return ParsedSession(
        source_name=Provider.CLAUDE_AI,
        provider_session_id=conversation_id or fallback_id,
        title=str(title),
        session_kind=_session_kind(payload),
        created_at=created_at,
        updated_at=updated_at,
        messages=normalized.messages,
        active_leaf_message_provider_id=normalized.active_leaf_message_provider_id,
        attachments=_merge_session_attachments(normalized.attachments, payload),
        session_events=session_events,
        reported_duration_ms=normalized.reported_duration_ms,
        models_used=normalized.models_used,
        ingest_flags=list(
            dict.fromkeys(
                [
                    *_session_ingest_flags(payload),
                    *normalized.ingest_flags,
                ]
            )
        ),
    )


__all__ = [
    "CLAUDE_DESIGN_CHAT_INGEST_FLAG",
    "CLAUDE_TEMPORARY_CHAT_INGEST_FLAG",
    "looks_like_ai",
    "parse_ai",
]
