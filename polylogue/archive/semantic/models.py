"""Typed semantic fact models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from polylogue.archive.actions.actions import Action
from polylogue.archive.viewport.viewports import ReasoningTrace, ToolCall


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
    is_human_authored: bool
    is_candidate_human_authored: bool
    is_assistant: bool
    is_dialogue: bool
    is_context_dump: bool
    is_protocol_artifact: bool
    is_noise: bool
    is_thinking: bool
    is_tool_use: bool
    is_substantive: bool
    tool_calls: tuple[ToolCall, ...]
    actions: tuple[Action, ...]
    tool_category_counts: dict[str, int]
    reasoning_traces: tuple[ReasoningTrace, ...]

    @property
    def affected_paths(self) -> tuple[str, ...]:
        paths: list[str] = []
        for action in self.actions:
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


@dataclass(frozen=True, slots=True)
class SessionSemanticFacts:
    session_id: str
    origin: str
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
    actions: tuple[Action, ...]
    first_message_at: datetime | None
    last_message_at: datetime | None
    wall_duration_ms: int
    message_facts: tuple[MessageSemanticFacts, ...]


@dataclass(frozen=True, slots=True)
class SummarySemanticFacts:
    session_id: str
    origin: str
    title: str
    date: str | None
    messages: int
    tags: tuple[str, ...]
    summary: str | None


@dataclass(frozen=True, slots=True)
class StreamSemanticFacts:
    session_id: str
    origin: str
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
    message_roles: tuple[str, ...]
    message_limit: int | None


@dataclass(frozen=True, slots=True)
class MCPDetailSemanticFacts:
    session_id: str
    origin: str
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


@dataclass(frozen=True, slots=True)
class MCPSummarySemanticFacts:
    session_id: str
    origin: str
    title: str
    messages: int
    created_at: str | None
    updated_at: str | None
    tags: tuple[str, ...]
    summary: str | None


__all__ = [
    "SessionSemanticFacts",
    "MCPDetailSemanticFacts",
    "MCPSummarySemanticFacts",
    "MessageSemanticFacts",
    "ProjectionSemanticFacts",
    "StreamSemanticFacts",
    "SummarySemanticFacts",
]
