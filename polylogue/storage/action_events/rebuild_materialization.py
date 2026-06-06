"""Materialization helpers for action-event read-model rebuilds."""

from __future__ import annotations

from collections import defaultdict

from polylogue.storage.action_events.rows import attach_blocks_to_messages, build_action_event_records
from polylogue.storage.runtime import (
    ActionEventRecord,
    ContentBlockRecord,
    MessageRecord,
    SessionRecord,
)


def materialize_batch(
    sessions: list[SessionRecord],
    messages: list[MessageRecord],
    blocks: list[ContentBlockRecord],
) -> dict[str, list[ActionEventRecord]]:
    messages_by_session: dict[str, list[MessageRecord]] = defaultdict(list)
    for message in messages:
        messages_by_session[str(message.session_id)].append(message)
    blocks_by_session: dict[str, list[ContentBlockRecord]] = defaultdict(list)
    for block in blocks:
        blocks_by_session[str(block.session_id)].append(block)

    materialized: dict[str, list[ActionEventRecord]] = {}
    for session in sessions:
        session_id = str(session.session_id)
        attached_messages = attach_blocks_to_messages(
            messages_by_session.get(session_id, []),
            blocks_by_session.get(session_id, []),
        )
        materialized[session_id] = build_action_event_records(session, attached_messages)
    return materialized
