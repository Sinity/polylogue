"""Typed semantic fact models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from polylogue.archive.action_event.action_events import ActionEvent
from polylogue.archive.viewport.viewports import ReasoningTrace, ToolCall
from polylogue.core.json import JSONDocument, json_document


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
    action_events: tuple[ActionEvent, ...]
    tool_category_counts: dict[str, int]
    reasoning_traces: tuple[ReasoningTrace, ...]

    @property
    def affected_paths(self) -> tuple[str, ...]:
        paths: list[str] = []
        for action in self.action_events:
            paths.extend(action.affected_paths)
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

    def to_proof_input(self) -> JSONDocument:
        return json_document(
            {
                "total_messages": self.total_messages,
                "renderable_messages": self.renderable_messages,
                "timestamped_renderable_messages": self.timestamped_renderable_messages,
                "attachment_count": self.attachment_count,
                "empty_messages": self.empty_messages,
                "thinking_messages": self.thinking_messages,
                "tool_messages": self.tool_messages,
                "renderable_role_counts": self.renderable_role_counts,
            }
        )


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
    timestamped_messages: int
    untimestamped_messages: int
    timestamp_coverage: str
    attachment_count: int
    thinking_messages: int
    tool_messages: int
    branch_messages: int
    word_count: int
    tool_category_counts: dict[str, int]
    action_events: tuple[ActionEvent, ...]
    first_message_at: datetime | None
    last_message_at: datetime | None
    wall_duration_ms: int
    message_facts: tuple[MessageSemanticFacts, ...]

    def to_proof_input(self) -> JSONDocument:
        return json_document(
            {
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
                "timestamped_messages": self.timestamped_messages,
                "untimestamped_messages": self.untimestamped_messages,
                "timestamp_coverage": self.timestamp_coverage,
                "attachment_count": self.attachment_count,
                "thinking_messages": self.thinking_messages,
                "tool_messages": self.tool_messages,
                "branch_messages": self.branch_messages,
            }
        )


@dataclass(frozen=True, slots=True)
class SummarySemanticFacts:
    conversation_id: str
    provider: str
    title: str
    date: str | None
    messages: int
    tags: tuple[str, ...]
    summary: str | None

    def to_proof_input(self) -> JSONDocument:
        return json_document(
            {
                "conversation_id": self.conversation_id,
                "provider": self.provider,
                "title": self.title,
                "date": self.date,
                "messages": self.messages,
                "tags": list(self.tags),
                "summary": self.summary,
            }
        )


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
    message_roles: tuple[str, ...]
    message_limit: int | None

    def to_proof_input(self) -> JSONDocument:
        return json_document(
            {
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
                "message_roles": list(self.message_roles),
                "message_limit": self.message_limit,
            }
        )


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

    def to_proof_input(self) -> JSONDocument:
        return json_document(
            {
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
        )


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

    def to_proof_input(self) -> JSONDocument:
        return json_document(
            {
                "conversation_id": self.conversation_id,
                "provider": self.provider,
                "title": self.title,
                "messages": self.messages,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "tags": list(self.tags),
                "summary": self.summary,
            }
        )


__all__ = [
    "ConversationSemanticFacts",
    "MCPDetailSemanticFacts",
    "MCPSummarySemanticFacts",
    "MessageSemanticFacts",
    "ProjectionSemanticFacts",
    "StreamSemanticFacts",
    "SummarySemanticFacts",
]
