"""Hydrators: translate storage records into domain models.

This module owns the knowledge of how to map storage records (MessageRecord,
SessionRecord, AttachmentRecord) to domain models (Message,
SessionSummary, Session).

Keeping this logic here preserves the dependency direction:
  storage → domain  (correct)
  domain  → storage (WRONG — what this module prevents)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from polylogue.archive.attachment.models import Attachment
from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.archive.provider.events import ProviderEvent
from polylogue.archive.session.domain_models import Session, SessionSummary
from polylogue.core.json import loads
from polylogue.core.sources import origin_from_provider
from polylogue.core.timestamps import parse_timestamp
from polylogue.storage.runtime import (
    AttachmentRecord,
    MessageRecord,
    ProviderEventRecord,
    SessionRecord,
)
from polylogue.types import MessageId, Provider


def _parse_json_blob(raw: object) -> object | None:
    """Parse persisted JSON payloads used in content block fields.

    Canonical storage keeps tool_input/metadata as JSON strings. Domain-model
    hydration should preserve that structure instead of discarding it.
    """
    if raw in {None, ""}:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str):
        return raw
    try:
        return loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def attachment_from_record(record: AttachmentRecord) -> Attachment:
    """Hydrate an Attachment domain model from an AttachmentRecord."""
    name = record.provider_meta.get("name") if record.provider_meta else None
    return Attachment(
        id=record.attachment_id,
        name=name if isinstance(name, str) else record.attachment_id,
        mime_type=record.mime_type,
        size_bytes=record.size_bytes,
        path=record.path,
        provider_meta=record.provider_meta,
    )


def message_from_record(
    record: MessageRecord,
    attachments: list[AttachmentRecord],
    *,
    provider: Provider | str | None = None,
) -> Message:
    """Hydrate a Message domain model from a MessageRecord and attachment records."""
    # Reconstruct timestamp from sort_key (numeric epoch seconds)
    ts = None
    if record.sort_key is not None:
        try:
            ts = datetime.fromtimestamp(record.sort_key, tz=timezone.utc)
        except (ValueError, OSError):
            ts = None

    # Domain messages expose semantic content blocks, not storage row identity.
    # #1240: media_type is stored inside the block-metadata JSON (image/document
    # blocks). Lift it back to the top-level for callers that still expect it.
    blocks = []
    for b in record.content_blocks:
        block_metadata = _parse_json_blob(b.metadata)
        media_type: object = None
        if isinstance(block_metadata, dict):
            media_type = block_metadata.get("media_type")
        blocks.append(
            {
                "type": str(b.type),
                "text": b.text,
                "tool_name": b.tool_name,
                "tool_id": b.tool_id,
                "tool_input": _parse_json_blob(b.tool_input),
                "media_type": media_type,
                "metadata": block_metadata,
                "semantic_type": str(b.semantic_type) if b.semantic_type is not None else None,
            }
        )

    normalized_provider = None
    if provider is not None:
        normalized_provider = provider if isinstance(provider, Provider) else Provider.from_string(provider)

    return Message(
        id=record.message_id,
        role=Role.normalize((record.role or "").strip() or "unknown"),
        text=record.text,
        timestamp=ts,
        provider=normalized_provider,
        attachments=[attachment_from_record(a) for a in attachments],
        content_blocks=blocks,
        message_type=record.message_type,
        parent_id=record.parent_message_id,
        branch_index=record.branch_index,
        has_tool_use=bool(record.has_tool_use),
        has_thinking=bool(record.has_thinking),
        has_paste=bool(record.has_paste),
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        cache_read_tokens=record.cache_read_tokens,
        cache_write_tokens=record.cache_write_tokens,
        model_name=record.model_name,
    )


def provider_event_from_record(record: ProviderEventRecord) -> ProviderEvent:
    return ProviderEvent(
        id=record.event_id,
        session_id=record.session_id,
        provider=Provider.from_string(record.source_name),
        event_index=record.event_index,
        event_type=record.event_type,
        timestamp=parse_timestamp(record.timestamp),
        sort_key=record.sort_key,
        payload=record.payload,
        source_message_id=record.source_message_id,
        raw_id=record.raw_id,
        materializer_version=record.materializer_version,
    )


def session_summary_from_record(
    record: SessionRecord,
    *,
    tags: tuple[str, ...] = (),
    message_count: int | None = None,
) -> SessionSummary:
    """Hydrate a SessionSummary domain model from a SessionRecord."""
    return SessionSummary(
        id=record.session_id,
        origin=origin_from_provider(Provider.from_string(record.source_name)),
        title=record.title,
        created_at=parse_timestamp(record.created_at),
        updated_at=parse_timestamp(record.updated_at),
        provider_meta=record.provider_meta,
        metadata=record.metadata or {},
        parent_id=record.parent_session_id,
        branch_type=record.branch_type,
        message_count=message_count,
        tags_m2m=tags,
    )


def session_from_records(
    session: SessionRecord,
    messages: list[MessageRecord],
    attachments: list[AttachmentRecord],
    provider_events: list[ProviderEventRecord] | None = None,
    *,
    tags: tuple[str, ...] = (),
) -> Session:
    """Hydrate a Session domain model from records.

    This is the canonical constructor that loads all messages into memory.
    Used for filtered views, tests, and when full message access is needed.

    Args:
        session: Session metadata record
        messages: List of message records
        attachments: List of attachment records

    Returns:
        Session with messages in eager mode
    """
    att_map: dict[MessageId, list[AttachmentRecord]] = {}
    for att in attachments:
        if att.message_id:
            att_map.setdefault(att.message_id, []).append(att)

    conv_provider = Provider.from_string(session.source_name)
    rich_messages = [
        message_from_record(
            msg,
            att_map.get(msg.message_id, []),
            provider=conv_provider,
        )
        for msg in messages
    ]

    return Session(
        id=session.session_id,
        origin=origin_from_provider(conv_provider),
        title=session.title,
        messages=MessageCollection(messages=rich_messages),
        created_at=parse_timestamp(session.created_at),
        updated_at=parse_timestamp(session.updated_at),
        provider_meta=session.provider_meta,
        metadata=session.metadata or {},
        provider_events=tuple(provider_event_from_record(event) for event in (provider_events or [])),
        parent_id=session.parent_session_id,
        branch_type=session.branch_type,
        tags_m2m=tags,
    )


__all__ = [
    "attachment_from_record",
    "session_from_records",
    "session_summary_from_record",
    "message_from_record",
    "provider_event_from_record",
]
