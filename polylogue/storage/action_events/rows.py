"""Canonical materialization for durable action-event rows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from polylogue.archive.action_event.action_events import (
    build_action_events,
    build_tool_calls_from_content_blocks,
)
from polylogue.storage.runtime import (
    ACTION_EVENT_MATERIALIZER_VERSION,
    ActionEventRecord,
    ContentBlockRecord,
    MessageRecord,
    SessionRecord,
)
from polylogue.types import Provider


@dataclass(frozen=True, slots=True)
class _ActionEventMessage:
    id: str
    timestamp: datetime | None = None


def _timestamp_to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _record_message_provider(session: SessionRecord) -> Provider:
    return Provider.from_string(session.source_name)


def build_action_event_records(
    session: SessionRecord,
    messages: Sequence[MessageRecord],
) -> list[ActionEventRecord]:
    """Build canonical durable action-event rows for one session."""
    records: list[ActionEventRecord] = []
    provider = _record_message_provider(session)

    for message in messages:
        if not message.content_blocks:
            continue
        tool_calls = build_tool_calls_from_content_blocks(
            provider=provider,
            content_blocks=[_content_block_mapping(block) for block in message.content_blocks],
        )
        if not tool_calls:
            continue
        event_message = _ActionEventMessage(id=str(message.message_id))
        for event in build_action_events(event_message, tool_calls):
            records.append(
                ActionEventRecord(
                    event_id=event.event_id,
                    session_id=session.session_id,
                    message_id=message.message_id,
                    materializer_version=ACTION_EVENT_MATERIALIZER_VERSION,
                    source_block_id=event.raw.get("block_id") if isinstance(event.raw, dict) else None,
                    timestamp=_timestamp_to_iso(event.timestamp),
                    sort_key=message.sort_key,
                    sequence_index=event.sequence_index,
                    source_name=session.source_name,
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


def _content_block_mapping(block: ContentBlockRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "block_id": block.block_id,
        "message_id": str(block.message_id),
        "session_id": str(block.session_id),
        "block_index": block.block_index,
        "type": block.type.value,
    }
    if block.text is not None:
        payload["text"] = block.text
    if block.tool_name is not None:
        payload["tool_name"] = block.tool_name
    if block.tool_id is not None:
        payload["tool_id"] = block.tool_id
    if block.tool_input is not None:
        payload["tool_input"] = block.tool_input
    if block.metadata is not None:
        payload["metadata"] = block.metadata
    if block.semantic_type is not None:
        payload["semantic_type"] = block.semantic_type.value
    return payload


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
]
