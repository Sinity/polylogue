"""Projection/message semantic fact builders."""

from __future__ import annotations

from collections import Counter

from polylogue.lib.action_events import build_action_events
from polylogue.lib.semantic_fact_models import MessageSemanticFacts, ProjectionSemanticFacts
from polylogue.lib.semantic_fact_support import (
    message_has_text,
    message_reasoning_traces,
    message_tool_calls,
    normalized_role_label,
    sorted_counts,
)


def build_projection_semantic_facts(projection) -> ProjectionSemanticFacts:
    attachment_counts: Counter[str] = Counter(
        attachment.message_id for attachment in projection.attachments if attachment.message_id
    )
    renderable_messages = 0
    timestamped_renderable_messages = 0
    empty_messages = 0
    thinking_messages = 0
    tool_messages = 0
    role_counts: Counter[str] = Counter()

    for message in projection.messages:
        has_attachments = attachment_counts.get(message.message_id, 0) > 0
        has_text = message_has_text(message)
        if has_text or has_attachments:
            renderable_messages += 1
            role_counts[normalized_role_label(message.role)] += 1
            if message.sort_key is not None:
                timestamped_renderable_messages += 1
        else:
            empty_messages += 1
        if int(message.has_thinking or 0) > 0:
            thinking_messages += 1
        if int(message.has_tool_use or 0) > 0:
            tool_messages += 1

    return ProjectionSemanticFacts(
        total_messages=len(projection.messages),
        renderable_messages=renderable_messages,
        timestamped_renderable_messages=timestamped_renderable_messages,
        attachment_count=len(projection.attachments),
        empty_messages=empty_messages,
        thinking_messages=thinking_messages,
        tool_messages=tool_messages,
        renderable_role_counts=sorted_counts(dict(role_counts)),
    )


def build_message_semantic_facts(message) -> MessageSemanticFacts:
    tool_calls = message_tool_calls(message)
    return MessageSemanticFacts(
        message_id=str(message.id),
        role=normalized_role_label(message.role),
        text=message.text or "",
        timestamp=message.timestamp,
        branch_index=message.branch_index,
        attachment_count=len(message.attachments),
        word_count=message.word_count,
        is_user=message.is_user,
        is_assistant=message.is_assistant,
        is_dialogue=message.is_dialogue,
        is_context_dump=message.is_context_dump,
        is_thinking=message.is_thinking,
        is_tool_use=message.is_tool_use,
        is_substantive=message.is_substantive,
        tool_calls=tool_calls,
        action_events=build_action_events(message, tool_calls),
        reasoning_traces=message_reasoning_traces(message),
    )


__all__ = ["build_message_semantic_facts", "build_projection_semantic_facts"]
