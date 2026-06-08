"""ID generation and content hashing logic for pipeline items."""

from __future__ import annotations

import unicodedata
from typing import TypeAlias

from polylogue.core.hashing import hash_payload
from polylogue.core.json import JSONValue
from polylogue.core.sources import origin_from_provider
from polylogue.sources import ParsedMessage, ParsedSession
from polylogue.sources.parsers.base import ParsedContentBlock
from polylogue.types import ContentHash, MessageId, Provider, SessionEventId, SessionId

# Sentinel values to distinguish None from empty in hash computations
_NULL_SENTINEL = "__POLYLOGUE_NULL__"
_EMPTY_SENTINEL = "__POLYLOGUE_EMPTY__"
HashScalar: TypeAlias = str | int | float | bool | None


def _normalize_for_hash(value: HashScalar) -> JSONValue:
    """Normalize a value for hashing, distinguishing None from empty.

    Args:
        value: Hash-compatible scalar value to normalize.

    Returns:
        Normalized JSON value with None → _NULL_SENTINEL and "" → _EMPTY_SENTINEL.
    """
    if value is None:
        return _NULL_SENTINEL
    if value == "":
        return _EMPTY_SENTINEL
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    return value


def session_id(source_name: str, provider_session_id: str) -> SessionId:
    """Generate the archive session ID from source/provider input.

    Args:
        provider_session_id: Provider's session identifier.

    Returns:
        Formatted session ID.

    Raises:
        ValueError: If source_name or provider_session_id is empty.
    """
    if not source_name or not source_name.strip():
        raise ValueError("source_name cannot be empty")
    if not provider_session_id or not provider_session_id.strip():
        raise ValueError("provider_session_id cannot be empty")
    origin = origin_from_provider(Provider.from_string(source_name))
    return SessionId(f"{origin.value}:{provider_session_id}")


def message_id(session_id: SessionId, provider_message_id: str) -> MessageId:
    return MessageId(f"{session_id}:{provider_message_id}")


def session_event_id(session_id: SessionId, event_index: int) -> SessionEventId:
    return SessionEventId(f"{session_id}:session-event:{event_index:06d}")


def _content_block_payload(block: ParsedContentBlock) -> dict[str, JSONValue]:
    """Build a hash-stable payload for a single content block."""
    payload: dict[str, JSONValue] = {
        "type": str(block.type),
        "text": _normalize_for_hash(block.text),
    }
    if block.tool_name:
        payload["tool_name"] = _normalize_for_hash(block.tool_name)
    if block.tool_id:
        payload["tool_id"] = _normalize_for_hash(block.tool_id)
    if block.tool_input is not None:
        payload["tool_input"] = hash_payload(dict(block.tool_input))
    if block.media_type:
        payload["media_type"] = _normalize_for_hash(block.media_type)
    return payload


def _message_hash_payload(message: ParsedMessage, message_id: str) -> dict[str, JSONValue]:
    """Build the hash-stable payload for a single message."""
    payload: dict[str, JSONValue] = {
        "id": message_id,
        "role": str(message.role),
        "text": _normalize_for_hash(message.text),
        "timestamp": _normalize_for_hash(message.timestamp),
    }
    if message.content_blocks:
        payload["content_blocks"] = [_content_block_payload(b) for b in message.content_blocks]
    return payload


def _session_hash_payload(
    *,
    title: str | None,
    created_at: str | None,
    updated_at: str | None,
    messages: list[dict[str, JSONValue]],
    attachments: list[dict[str, JSONValue]],
    session_events: list[dict[str, JSONValue]],
) -> dict[str, object]:
    """Build the content-hash payload dict shared by pipeline and async write paths."""
    return {
        "title": _normalize_for_hash(title),
        "created_at": _normalize_for_hash(created_at),
        "updated_at": _normalize_for_hash(updated_at),
        "messages": messages,
        "session_events": session_events,
        "attachments": sorted(
            attachments,
            key=lambda item: (
                str(item.get("message_id") or ""),
                str(item.get("id") or ""),
                str(item.get("name") or ""),
            ),
        ),
    }


def session_content_hash(convo: ParsedSession) -> ContentHash:
    """Generate the content hash for a session.

    Uses sentinel values to distinguish None from empty/missing fields. The
    hash incorporates the per-message payload (id, role, text, timestamp,
    content blocks), attachments, and session events, so any change to a
    message also changes the session hash.
    """
    messages_payload = [
        _message_hash_payload(msg, msg.provider_message_id or f"msg-{idx}")
        for idx, msg in enumerate(convo.messages, start=1)
    ]
    attachments_payload = [
        {
            "id": _normalize_for_hash(att.provider_attachment_id),
            "message_id": _normalize_for_hash(att.message_provider_id),
            "name": _normalize_for_hash(att.name),
            "mime_type": _normalize_for_hash(att.mime_type),
            "size_bytes": _normalize_for_hash(att.size_bytes),
            "digest": _normalize_for_hash(
                str(att.provider_meta["sha256"]) if att.provider_meta and att.provider_meta.get("sha256") else None
            ),
        }
        for att in convo.attachments
    ]
    session_events_payload = [
        {
            "event_index": event_index,
            "event_type": _normalize_for_hash(event.event_type),
            "timestamp": _normalize_for_hash(event.timestamp),
            "source_message_provider_id": _normalize_for_hash(event.source_message_provider_id),
            "payload": hash_payload(event.payload),
        }
        for event_index, event in enumerate(convo.session_events)
    ]
    return ContentHash(
        hash_payload(
            _session_hash_payload(
                title=convo.title,
                created_at=convo.created_at,
                updated_at=convo.updated_at,
                messages=messages_payload,
                attachments=attachments_payload,
                session_events=session_events_payload,
            )
        )
    )
