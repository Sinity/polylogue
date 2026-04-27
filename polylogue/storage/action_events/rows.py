"""Canonical materialization and hydration for durable action-event rows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone

from polylogue.lib.action_event.action_events import (
    ActionEvent,
    build_action_events,
    build_tool_calls_from_content_blocks,
)
from polylogue.lib.viewport.viewports import ToolCategory
from polylogue.storage.hydrators import message_from_record
from polylogue.storage.runtime import (
    ACTION_EVENT_MATERIALIZER_VERSION,
    ActionEventRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
)
from polylogue.types import Provider


def _timestamp_to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _record_message_provider(conversation: ConversationRecord) -> Provider:
    return Provider.from_string(conversation.provider_name)


def hydrate_action_event(record: ActionEventRecord) -> ActionEvent:
    """Hydrate an in-memory canonical ActionEvent from a durable row."""
    timestamp = None
    if record.timestamp:
        normalized = record.timestamp.replace("Z", "+00:00")
        try:
            timestamp = datetime.fromisoformat(normalized)
        except ValueError:
            timestamp = None
    provider = Provider.from_string(record.provider_name) if record.provider_name else None
    return ActionEvent(
        event_id=record.event_id,
        message_id=str(record.message_id),
        timestamp=timestamp,
        sequence_index=record.sequence_index,
        kind=ToolCategory(record.action_kind),
        tool_name=record.tool_name or record.normalized_tool_name,
        tool_id=record.tool_id,
        provider=provider,
        affected_paths=tuple(record.affected_paths),
        cwd_path=record.cwd_path,
        branch_names=tuple(record.branch_names),
        command=record.command,
        query=record.query_text,
        url=record.url,
        output_text=record.output_text,
        search_text=record.search_text,
        raw={
            "source_block_id": record.source_block_id,
            "provider_name": record.provider_name,
        },
    )


def hydrate_action_events(records: Iterable[ActionEventRecord]) -> tuple[ActionEvent, ...]:
    return tuple(hydrate_action_event(record) for record in records)


def build_action_event_records(
    conversation: ConversationRecord,
    messages: Sequence[MessageRecord],
) -> list[ActionEventRecord]:
    """Build canonical durable action-event rows for one conversation."""
    records: list[ActionEventRecord] = []
    provider = _record_message_provider(conversation)

    for message in messages:
        domain_message = message_from_record(message, attachments=[], provider=provider)
        tool_calls = build_tool_calls_from_content_blocks(
            provider=provider,
            content_blocks=domain_message.content_blocks,
        )
        for event in build_action_events(domain_message, tool_calls):
            records.append(
                ActionEventRecord(
                    event_id=event.event_id,
                    conversation_id=conversation.conversation_id,
                    message_id=message.message_id,
                    materializer_version=ACTION_EVENT_MATERIALIZER_VERSION,
                    source_block_id=event.raw.get("block_id") if isinstance(event.raw, dict) else None,
                    timestamp=_timestamp_to_iso(event.timestamp),
                    sort_key=message.sort_key,
                    sequence_index=event.sequence_index,
                    provider_name=conversation.provider_name,
                    action_kind=event.kind.value,
                    tool_name=event.tool_name,
                    normalized_tool_name=event.normalized_tool_name,
                    tool_id=event.tool_id,
                    affected_paths=tuple(event.affected_paths),
                    cwd_path=event.cwd_path,
                    branch_names=tuple(event.branch_names),
                    command=event.command,
                    query_text=event.query,
                    url=event.url,
                    output_text=event.output_text,
                    search_text=event.search_text,
                )
            )
    return records


def attach_blocks_to_messages(
    messages: Sequence[MessageRecord],
    content_blocks: Sequence[ContentBlockRecord],
) -> list[MessageRecord]:
    """Attach content blocks to message records for action-event derivation."""
    grouped: dict[str, list[ContentBlockRecord]] = defaultdict(list)
    for block in content_blocks:
        grouped[str(block.message_id)].append(block)
    return [
        message.model_copy(update={"content_blocks": list(grouped.get(str(message.message_id), []))})
        for message in messages
    ]


__all__ = [
    "attach_blocks_to_messages",
    "build_action_event_records",
    "hydrate_action_event",
    "hydrate_action_events",
]
