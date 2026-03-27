"""Conversation/detail semantic fact builders."""

from __future__ import annotations

from collections import Counter

from polylogue.lib.action_events import ActionEvent
from polylogue.lib.semantic_fact_models import ConversationSemanticFacts, MCPDetailSemanticFacts
from polylogue.lib.semantic_fact_support import sorted_counts

from .semantic_fact_projection_builders import build_message_semantic_facts


def build_conversation_semantic_facts(conversation) -> ConversationSemanticFacts:
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
    timestamps: list = []
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


def build_mcp_detail_semantic_facts(conversation) -> MCPDetailSemanticFacts:
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


__all__ = ["build_conversation_semantic_facts", "build_mcp_detail_semantic_facts"]
