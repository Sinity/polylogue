"""Materialization helpers for action-event read-model rebuilds."""

from __future__ import annotations

from collections import defaultdict

from polylogue.storage.action_event_rows import attach_blocks_to_messages, build_action_event_records
from polylogue.storage.store import (
    ActionEventRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
)


def materialize_batch(
    conversations: list[ConversationRecord],
    messages: list[MessageRecord],
    blocks: list[ContentBlockRecord],
) -> dict[str, list[ActionEventRecord]]:
    messages_by_conversation: dict[str, list[MessageRecord]] = defaultdict(list)
    for message in messages:
        messages_by_conversation[str(message.conversation_id)].append(message)
    blocks_by_conversation: dict[str, list[ContentBlockRecord]] = defaultdict(list)
    for block in blocks:
        blocks_by_conversation[str(block.conversation_id)].append(block)

    materialized: dict[str, list[ActionEventRecord]] = {}
    for conversation in conversations:
        conversation_id = str(conversation.conversation_id)
        attached_messages = attach_blocks_to_messages(
            messages_by_conversation.get(conversation_id, []),
            blocks_by_conversation.get(conversation_id, []),
        )
        materialized[conversation_id] = build_action_event_records(conversation, attached_messages)
    return materialized
