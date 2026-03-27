"""Summary/stream semantic fact builders."""

from __future__ import annotations

from collections import Counter

from polylogue.lib.semantic_fact_models import (
    MCPSummarySemanticFacts,
    StreamSemanticFacts,
    SummarySemanticFacts,
)
from polylogue.lib.semantic_fact_support import message_has_text, normalized_role_label, sorted_counts


def build_summary_semantic_facts(summary, *, message_count: int) -> SummarySemanticFacts:
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
    summary,
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
    conversation,
    *,
    dialogue_only: bool = False,
    message_limit: int | None = None,
) -> StreamSemanticFacts:
    filtered_messages = [
        message
        for message in conversation.messages
        if not dialogue_only or message.is_dialogue
    ]
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
    "build_mcp_summary_semantic_facts",
    "build_stream_semantic_facts",
    "build_summary_semantic_facts",
]
