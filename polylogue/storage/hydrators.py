"""Hydrators: translate storage records into domain models.

This module owns the knowledge of how to map storage records (MessageRecord,
ConversationRecord, AttachmentRecord) to domain models (Message,
ConversationSummary, Conversation).

Keeping this logic here preserves the dependency direction:
  storage → domain  (correct)
  domain  → storage (WRONG — what this module prevents)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from polylogue.archive.attachment.models import Attachment
from polylogue.archive.conversation.models import Conversation, ConversationSummary
from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.core.json import loads
from polylogue.core.timestamps import parse_timestamp
from polylogue.lib.roles import Role
from polylogue.storage.runtime import (
    AttachmentRecord,
    ConversationRecord,
    MessageRecord,
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
    blocks = [
        {
            "type": str(b.type),
            "text": b.text,
            "tool_name": b.tool_name,
            "tool_id": b.tool_id,
            "tool_input": _parse_json_blob(b.tool_input),
            "media_type": b.media_type,
            "metadata": _parse_json_blob(b.metadata),
            "semantic_type": str(b.semantic_type) if b.semantic_type is not None else None,
        }
        for b in record.content_blocks
    ]

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
        provider_meta=None,  # Canonical storage keeps message semantics in content_blocks, not message-level provider_meta.
        content_blocks=blocks,
        message_type=record.message_type,
        parent_id=record.parent_message_id,
        branch_index=record.branch_index,
    )


def conversation_summary_from_record(record: ConversationRecord) -> ConversationSummary:
    """Hydrate a ConversationSummary domain model from a ConversationRecord."""
    return ConversationSummary(
        id=record.conversation_id,
        provider=Provider.from_string(record.provider_name),
        title=record.title,
        created_at=parse_timestamp(record.created_at),
        updated_at=parse_timestamp(record.updated_at),
        provider_meta=record.provider_meta,
        metadata=record.metadata or {},
        parent_id=record.parent_conversation_id,
        branch_type=record.branch_type,
    )


def conversation_from_records(
    conversation: ConversationRecord,
    messages: list[MessageRecord],
    attachments: list[AttachmentRecord],
) -> Conversation:
    """Hydrate a Conversation domain model from records.

    This is the canonical constructor that loads all messages into memory.
    Used for filtered views, tests, and when full message access is needed.

    Args:
        conversation: Conversation metadata record
        messages: List of message records
        attachments: List of attachment records

    Returns:
        Conversation with messages in eager mode
    """
    att_map: dict[MessageId, list[AttachmentRecord]] = {}
    for att in attachments:
        if att.message_id:
            att_map.setdefault(att.message_id, []).append(att)

    conv_provider = Provider.from_string(conversation.provider_name)
    rich_messages = [
        message_from_record(
            msg,
            att_map.get(msg.message_id, []),
            provider=conv_provider,
        )
        for msg in messages
    ]

    return Conversation(
        id=conversation.conversation_id,
        provider=conv_provider,
        title=conversation.title,
        messages=MessageCollection(messages=rich_messages),
        created_at=parse_timestamp(conversation.created_at),
        updated_at=parse_timestamp(conversation.updated_at),
        provider_meta=conversation.provider_meta,
        metadata=conversation.metadata or {},
        parent_id=conversation.parent_conversation_id,
        branch_type=conversation.branch_type,
    )


__all__ = [
    "attachment_from_record",
    "conversation_from_records",
    "conversation_summary_from_record",
    "message_from_record",
]
