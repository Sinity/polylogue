"""Canonical semantic fact API — models, builders, and support helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any

from polylogue.lib.action_events import ActionEvent, build_action_events
from polylogue.lib.semantic_fact_models import (
    ConversationSemanticFacts,
    MCPDetailSemanticFacts,
    MCPSummarySemanticFacts,
    MessageSemanticFacts,
    ProjectionSemanticFacts,
    StreamSemanticFacts,
    SummarySemanticFacts,
)
from polylogue.lib.semantic_fact_support import (
    message_has_text,
    message_model_name,
    message_reasoning_traces,
    message_tokens,
    message_tool_calls,
    normalized_role_label,
    sorted_counts,
)

# ---------------------------------------------------------------------------
# Projection / message builders
# ---------------------------------------------------------------------------


def build_projection_semantic_facts(projection: Any) -> ProjectionSemanticFacts:
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


def build_message_semantic_facts(message: Any) -> MessageSemanticFacts:
    tool_calls = message_tool_calls(message)
    action_events = build_action_events(message, tool_calls)
    tool_category_counts = Counter(action.kind.value for action in action_events)
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
        action_events=action_events,
        tool_category_counts=sorted_counts(dict(tool_category_counts)),
        reasoning_traces=message_reasoning_traces(message),
    )


# ---------------------------------------------------------------------------
# Conversation / detail builders
# ---------------------------------------------------------------------------


def build_conversation_semantic_facts(conversation: Any) -> ConversationSemanticFacts:
    role_counts: Counter[str] = Counter()
    tool_categories: Counter[str] = Counter()
    message_facts = tuple(build_message_semantic_facts(message) for message in conversation.messages)
    message_ids: list[str] = []
    text_message_ids: list[str] = []
    timestamped_text_messages = 0
    attachment_count = 0
    thinking_messages = 0
    tool_messages = 0
    branch_messages = 0
    substantive_messages = 0
    word_count = 0
    timestamps: list[Any] = []
    action_events: list[ActionEvent] = []

    for message_fact in message_facts:
        message_ids.append(message_fact.message_id)
        attachment_count += message_fact.attachment_count
        if message_fact.is_thinking:
            thinking_messages += 1
        if message_fact.is_tool_use:
            tool_messages += 1
        if message_fact.branch_index > 0:
            branch_messages += 1
        if message_fact.is_substantive:
            substantive_messages += 1
        word_count += message_fact.word_count
        action_events.extend(message_fact.action_events)
        if message_fact.timestamp is not None:
            timestamps.append(message_fact.timestamp)
        if message_fact.text.strip():
            text_message_ids.append(message_fact.message_id)
            role_counts[message_fact.role] += 1
            if message_fact.timestamp is not None:
                timestamped_text_messages += 1
        for category, count in message_fact.tool_category_counts.items():
            tool_categories[category] += count

    first_message_at = min(timestamps) if timestamps else None
    last_message_at = max(timestamps) if timestamps else None
    wall_duration_ms = 0
    if first_message_at and last_message_at:
        wall_duration_ms = max(int((last_message_at - first_message_at).total_seconds() * 1000), 0)

    return ConversationSemanticFacts(
        conversation_id=str(conversation.id),
        provider=str(conversation.provider),
        title=conversation.display_title,
        date=conversation.display_date.isoformat() if conversation.display_date else None,
        total_messages=len(conversation.messages),
        substantive_messages=substantive_messages,
        text_messages=len(text_message_ids),
        message_ids=tuple(message_ids),
        text_message_ids=tuple(text_message_ids),
        text_role_counts=sorted_counts(dict(role_counts)),
        timestamped_text_messages=timestamped_text_messages,
        attachment_count=attachment_count,
        thinking_messages=thinking_messages,
        tool_messages=tool_messages,
        branch_messages=branch_messages,
        word_count=word_count,
        tool_category_counts=sorted_counts(dict(tool_categories)),
        action_events=tuple(action_events),
        first_message_at=first_message_at,
        last_message_at=last_message_at,
        wall_duration_ms=wall_duration_ms,
        message_facts=message_facts,
    )


def build_mcp_detail_semantic_facts(conversation: Any) -> MCPDetailSemanticFacts:
    facts = build_conversation_semantic_facts(conversation)
    return MCPDetailSemanticFacts(
        conversation_id=facts.conversation_id,
        provider=facts.provider,
        title=facts.title,
        created_at=conversation.created_at.isoformat() if conversation.created_at else None,
        updated_at=conversation.updated_at.isoformat() if conversation.updated_at else None,
        messages=facts.total_messages,
        message_ids=facts.message_ids,
        role_counts=facts.text_role_counts,
        timestamped_messages=facts.timestamped_text_messages,
        attachment_count=facts.attachment_count,
        thinking_messages=facts.thinking_messages,
        tool_messages=facts.tool_messages,
        branch_messages=facts.branch_messages,
    )


# ---------------------------------------------------------------------------
# Summary / stream builders
# ---------------------------------------------------------------------------


def build_summary_semantic_facts(summary: Any, *, message_count: int) -> SummarySemanticFacts:
    return SummarySemanticFacts(
        conversation_id=str(summary.id),
        provider=str(summary.provider),
        title=summary.display_title,
        date=summary.display_date.isoformat() if summary.display_date else None,
        messages=message_count,
        tags=tuple(summary.tags),
        summary=summary.summary,
    )


def build_mcp_summary_semantic_facts(
    summary: Any,
    *,
    message_count: int,
) -> MCPSummarySemanticFacts:
    return MCPSummarySemanticFacts(
        conversation_id=str(summary.id),
        provider=str(summary.provider),
        title=summary.display_title,
        messages=message_count,
        created_at=summary.created_at.isoformat() if summary.created_at else None,
        updated_at=summary.updated_at.isoformat() if summary.updated_at else None,
        tags=tuple(summary.tags),
        summary=summary.summary,
    )


def build_stream_semantic_facts(
    conversation: Any,
    *,
    dialogue_only: bool = False,
    message_limit: int | None = None,
) -> StreamSemanticFacts:
    filtered_messages = [message for message in conversation.messages if not dialogue_only or message.is_dialogue]
    if message_limit is not None:
        filtered_messages = filtered_messages[:message_limit]

    visible_messages = [message for message in filtered_messages if message_has_text(message)]
    role_counts: Counter[str] = Counter(normalized_role_label(message.role) for message in visible_messages)
    timestamped_messages = sum(1 for message in visible_messages if message.timestamp is not None)

    return StreamSemanticFacts(
        conversation_id=str(conversation.id),
        provider=str(conversation.provider),
        title=conversation.display_title,
        date=conversation.display_date.isoformat() if conversation.display_date else None,
        text_messages=len(visible_messages),
        text_message_ids=tuple(str(message.id) for message in visible_messages),
        text_role_counts=sorted_counts(dict(role_counts)),
        timestamped_text_messages=timestamped_messages,
        attachment_count=sum(len(message.attachments) for message in filtered_messages),
        thinking_messages=sum(1 for message in filtered_messages if message.is_thinking),
        tool_messages=sum(1 for message in filtered_messages if message.is_tool_use),
        branch_messages=sum(1 for message in filtered_messages if message.branch_index > 0),
        dialogue_only=dialogue_only,
        message_limit=message_limit,
    )


__all__ = [
    "ConversationSemanticFacts",
    "MCPDetailSemanticFacts",
    "MCPSummarySemanticFacts",
    "MessageSemanticFacts",
    "ProjectionSemanticFacts",
    "StreamSemanticFacts",
    "SummarySemanticFacts",
    "build_conversation_semantic_facts",
    "build_mcp_detail_semantic_facts",
    "build_mcp_summary_semantic_facts",
    "build_message_semantic_facts",
    "build_projection_semantic_facts",
    "build_stream_semantic_facts",
    "build_summary_semantic_facts",
    "message_has_text",
    "message_model_name",
    "message_reasoning_traces",
    "message_tokens",
    "message_tool_calls",
    "normalized_role_label",
    "sorted_counts",
]
