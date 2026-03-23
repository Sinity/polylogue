"""Canonical semantic facts derived from conversations and render projections."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from polylogue.lib.roles import Role
from polylogue.lib.viewports import (
    ReasoningTrace,
    ToolCall,
    ToolCategory,
    classify_tool,
)

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, ConversationSummary, Message
    from polylogue.lib.viewports import TokenUsage
    from polylogue.storage.state_views import ConversationRenderProjection
    from polylogue.storage.store import MessageRecord


def normalized_role_label(value: object) -> str:
    if isinstance(value, Role):
        return str(value)
    if value:
        return str(Role.normalize(str(value)))
    return "message"


def sorted_counts(counts: dict[str, int]) -> dict[str, int]:
    return dict(sorted(counts.items()))


def message_has_text(message: Message | MessageRecord) -> bool:
    return bool((message.text or "").strip())


def message_tool_calls(message: Message) -> tuple[ToolCall, ...]:
    harmonized = message.harmonized
    if harmonized is not None:
        calls = getattr(harmonized, "tool_calls", None)
        if calls:
            return tuple(calls)
    return _message_content_block_tool_calls(message)


def message_reasoning_traces(message: Message) -> tuple[ReasoningTrace, ...]:
    harmonized = message.harmonized
    if harmonized is not None:
        traces = getattr(harmonized, "reasoning_traces", None)
        if traces:
            return tuple(traces)
    return _message_content_block_reasoning_traces(message)


def message_tokens(message: Message) -> TokenUsage | None:
    harmonized = message.harmonized
    if harmonized is None:
        return None
    tokens = getattr(harmonized, "tokens", None)
    return tokens if tokens is not None else None


def message_model_name(message: Message) -> str | None:
    harmonized = message.harmonized
    if harmonized is None:
        return None
    model = getattr(harmonized, "model", None)
    return str(model) if model else None


def _tool_category_from_semantic(value: object) -> ToolCategory | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return ToolCategory(value)
    except ValueError:
        return None


def _message_content_block_tool_calls(message: Message) -> tuple[ToolCall, ...]:
    tool_result_outputs: dict[str, str] = {}
    for block in message.content_blocks:
        if str(block.get("type")) != "tool_result":
            continue
        tool_id = block.get("tool_id")
        text = block.get("text")
        if isinstance(tool_id, str) and tool_id and isinstance(text, str) and text:
            tool_result_outputs.setdefault(tool_id, text)

    calls: list[ToolCall] = []
    for block in message.content_blocks:
        if str(block.get("type")) != "tool_use":
            continue
        name = block.get("tool_name")
        if not isinstance(name, str) or not name:
            continue
        tool_id = block.get("tool_id")
        tool_input = block.get("tool_input")
        normalized_input = tool_input if isinstance(tool_input, dict) else {}
        semantic_category = _tool_category_from_semantic(block.get("semantic_type"))
        classified_category = classify_tool(name, normalized_input)
        category = classified_category if semantic_category in (None, ToolCategory.OTHER) else semantic_category
        raw = {
            "type": block.get("type"),
            "tool_name": name,
            "tool_id": tool_id,
            "tool_input": normalized_input,
            "media_type": block.get("media_type"),
            "metadata": block.get("metadata"),
            "semantic_type": block.get("semantic_type"),
            "text": block.get("text"),
        }
        calls.append(
            ToolCall(
                name=name,
                id=tool_id if isinstance(tool_id, str) and tool_id else None,
                input=normalized_input,
                output=tool_result_outputs.get(tool_id) if isinstance(tool_id, str) else None,
                category=category,
                provider=message.provider,
                raw=raw,
            )
        )
    return tuple(calls)


def _message_content_block_reasoning_traces(message: Message) -> tuple[ReasoningTrace, ...]:
    traces: list[ReasoningTrace] = []
    for block in message.content_blocks:
        if str(block.get("type")) != "thinking":
            continue
        text = block.get("text")
        if not isinstance(text, str) or not text:
            continue
        traces.append(
            ReasoningTrace(
                text=text,
                provider=message.provider,
                raw={
                    "type": block.get("type"),
                    "media_type": block.get("media_type"),
                    "metadata": block.get("metadata"),
                    "semantic_type": block.get("semantic_type"),
                },
            )
        )
    return tuple(traces)


@dataclass(frozen=True, slots=True)
class MessageSemanticFacts:
    message_id: str
    role: str
    text: str
    timestamp: datetime | None
    branch_index: int
    attachment_count: int
    word_count: int
    is_user: bool
    is_assistant: bool
    is_dialogue: bool
    is_context_dump: bool
    is_thinking: bool
    is_tool_use: bool
    is_substantive: bool
    tool_calls: tuple[ToolCall, ...]
    reasoning_traces: tuple[ReasoningTrace, ...]

    @property
    def tool_category_counts(self) -> dict[str, int]:
        counts = Counter(call.category.value for call in self.tool_calls)
        return sorted_counts(dict(counts))

    @property
    def affected_paths(self) -> tuple[str, ...]:
        paths: list[str] = []
        for call in self.tool_calls:
            paths.extend(call.affected_paths)
        return tuple(dict.fromkeys(paths))


@dataclass(frozen=True, slots=True)
class ProjectionSemanticFacts:
    total_messages: int
    renderable_messages: int
    timestamped_renderable_messages: int
    attachment_count: int
    empty_messages: int
    thinking_messages: int
    tool_messages: int
    renderable_role_counts: dict[str, int]

    def to_proof_input(self) -> dict[str, Any]:
        return {
            "total_messages": self.total_messages,
            "renderable_messages": self.renderable_messages,
            "timestamped_renderable_messages": self.timestamped_renderable_messages,
            "attachment_count": self.attachment_count,
            "empty_messages": self.empty_messages,
            "thinking_messages": self.thinking_messages,
            "tool_messages": self.tool_messages,
            "renderable_role_counts": self.renderable_role_counts,
        }


@dataclass(frozen=True, slots=True)
class ConversationSemanticFacts:
    conversation_id: str
    provider: str
    title: str
    date: str | None
    total_messages: int
    substantive_messages: int
    text_messages: int
    message_ids: tuple[str, ...]
    text_message_ids: tuple[str, ...]
    text_role_counts: dict[str, int]
    timestamped_text_messages: int
    attachment_count: int
    thinking_messages: int
    tool_messages: int
    branch_messages: int
    word_count: int
    tool_category_counts: dict[str, int]
    first_message_at: datetime | None
    last_message_at: datetime | None
    wall_duration_ms: int
    message_facts: tuple[MessageSemanticFacts, ...]

    def to_proof_input(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "provider": self.provider,
            "title": self.title,
            "date": self.date,
            "total_messages": self.total_messages,
            "text_messages": self.text_messages,
            "message_ids": list(self.message_ids),
            "text_message_ids": list(self.text_message_ids),
            "text_role_counts": self.text_role_counts,
            "timestamped_text_messages": self.timestamped_text_messages,
            "attachment_count": self.attachment_count,
            "thinking_messages": self.thinking_messages,
            "tool_messages": self.tool_messages,
            "branch_messages": self.branch_messages,
        }


@dataclass(frozen=True, slots=True)
class SummarySemanticFacts:
    conversation_id: str
    provider: str
    title: str
    date: str | None
    messages: int
    tags: tuple[str, ...]
    summary: str | None

    def to_proof_input(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "provider": self.provider,
            "title": self.title,
            "date": self.date,
            "messages": self.messages,
            "tags": list(self.tags),
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class StreamSemanticFacts:
    conversation_id: str
    provider: str
    title: str
    date: str | None
    text_messages: int
    text_message_ids: tuple[str, ...]
    text_role_counts: dict[str, int]
    timestamped_text_messages: int
    attachment_count: int
    thinking_messages: int
    tool_messages: int
    branch_messages: int
    dialogue_only: bool
    message_limit: int | None

    def to_proof_input(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "provider": self.provider,
            "title": self.title,
            "date": self.date,
            "text_messages": self.text_messages,
            "text_message_ids": list(self.text_message_ids),
            "text_role_counts": self.text_role_counts,
            "timestamped_text_messages": self.timestamped_text_messages,
            "attachment_count": self.attachment_count,
            "thinking_messages": self.thinking_messages,
            "tool_messages": self.tool_messages,
            "branch_messages": self.branch_messages,
            "dialogue_only": self.dialogue_only,
            "message_limit": self.message_limit,
        }


@dataclass(frozen=True, slots=True)
class MCPDetailSemanticFacts:
    conversation_id: str
    provider: str
    title: str
    created_at: str | None
    updated_at: str | None
    messages: int
    message_ids: tuple[str, ...]
    role_counts: dict[str, int]
    timestamped_messages: int
    attachment_count: int
    thinking_messages: int
    tool_messages: int
    branch_messages: int

    def to_proof_input(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "provider": self.provider,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages,
            "message_ids": list(self.message_ids),
            "role_counts": self.role_counts,
            "timestamped_messages": self.timestamped_messages,
            "attachment_count": self.attachment_count,
            "thinking_messages": self.thinking_messages,
            "tool_messages": self.tool_messages,
            "branch_messages": self.branch_messages,
        }


@dataclass(frozen=True, slots=True)
class MCPSummarySemanticFacts:
    conversation_id: str
    provider: str
    title: str
    messages: int
    created_at: str | None
    updated_at: str | None
    tags: tuple[str, ...]
    summary: str | None

    def to_proof_input(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "provider": self.provider,
            "title": self.title,
            "messages": self.messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": list(self.tags),
            "summary": self.summary,
        }


def build_projection_semantic_facts(projection: ConversationRenderProjection) -> ProjectionSemanticFacts:
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


def build_message_semantic_facts(message: Message) -> MessageSemanticFacts:
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
        tool_calls=message_tool_calls(message),
        reasoning_traces=message_reasoning_traces(message),
    )


def build_conversation_semantic_facts(conversation: Conversation) -> ConversationSemanticFacts:
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
    timestamps: list[datetime] = []

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
        first_message_at=first_message_at,
        last_message_at=last_message_at,
        wall_duration_ms=wall_duration_ms,
        message_facts=message_facts,
    )


def build_summary_semantic_facts(summary: ConversationSummary, *, message_count: int) -> SummarySemanticFacts:
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
    summary: ConversationSummary,
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
    conversation: Conversation,
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


def build_mcp_detail_semantic_facts(conversation: Conversation) -> MCPDetailSemanticFacts:
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


__all__ = [
    "ConversationSemanticFacts",
    "MessageSemanticFacts",
    "MCPDetailSemanticFacts",
    "MCPSummarySemanticFacts",
    "ProjectionSemanticFacts",
    "StreamSemanticFacts",
    "SummarySemanticFacts",
    "build_conversation_semantic_facts",
    "build_message_semantic_facts",
    "build_mcp_detail_semantic_facts",
    "build_mcp_summary_semantic_facts",
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
