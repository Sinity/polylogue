"""Canonical semantic fact API — models, builders, and support helpers."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Collection, Iterator, Sequence
from datetime import datetime
from typing import Protocol

from polylogue.archive.actions.actions import Action, build_actions
from polylogue.archive.message.roles import MessageRoleFilter, Role, message_role_labels
from polylogue.archive.semantic.models import (
    MCPDetailSemanticFacts,
    MCPSummarySemanticFacts,
    MessageSemanticFacts,
    ProjectionSemanticFacts,
    SessionSemanticFacts,
    StreamSemanticFacts,
    SummarySemanticFacts,
)
from polylogue.archive.semantic.support import (
    SemanticMessageLike,
    TextMessageLike,
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


class ProjectionAttachmentLike(Protocol):
    @property
    def message_id(self) -> str | None: ...


class ProjectionMessageLike(TextMessageLike, Protocol):
    @property
    def message_id(self) -> str: ...

    @property
    def role(self) -> object: ...

    @property
    def sort_key(self) -> float | None: ...

    @property
    def has_thinking(self) -> int | bool: ...

    @property
    def has_tool_use(self) -> int | bool: ...


class ProjectionLike(Protocol):
    @property
    def messages(self) -> Collection[ProjectionMessageLike]: ...

    @property
    def attachments(self) -> Collection[ProjectionAttachmentLike]: ...


class SemanticSessionMessagesLike(Protocol):
    def __iter__(self) -> Iterator[SemanticSessionMessageLike]: ...

    def __len__(self) -> int: ...


class SemanticSessionLike(Protocol):
    @property
    def id(self) -> object: ...

    @property
    def origin(self) -> object: ...

    @property
    def display_title(self) -> str: ...

    @property
    def display_date(self) -> datetime | None: ...

    @property
    def created_at(self) -> datetime | None: ...

    @property
    def updated_at(self) -> datetime | None: ...

    @property
    def messages(self) -> SemanticSessionMessagesLike: ...


class SemanticSessionMessageLike(SemanticMessageLike, Protocol):
    @property
    def id(self) -> object: ...

    @property
    def role(self) -> object: ...

    @property
    def timestamp(self) -> datetime | None: ...

    @property
    def branch_index(self) -> int: ...

    @property
    def attachments(self) -> Sequence[object]: ...

    @property
    def word_count(self) -> int: ...

    @property
    def is_user(self) -> bool: ...

    @property
    def is_human_authored(self) -> bool: ...

    @property
    def is_candidate_human_authored(self) -> bool: ...

    @property
    def is_assistant(self) -> bool: ...

    @property
    def is_dialogue(self) -> bool: ...

    @property
    def is_context_dump(self) -> bool: ...

    @property
    def is_protocol_artifact(self) -> bool: ...

    @property
    def is_noise(self) -> bool: ...

    @property
    def is_thinking(self) -> bool: ...

    @property
    def is_tool_use(self) -> bool: ...

    @property
    def is_substantive(self) -> bool: ...


class SemanticSummaryLike(Protocol):
    @property
    def id(self) -> object: ...

    @property
    def origin(self) -> object: ...

    @property
    def display_title(self) -> str: ...

    @property
    def display_date(self) -> datetime | None: ...

    @property
    def created_at(self) -> datetime | None: ...

    @property
    def updated_at(self) -> datetime | None: ...

    @property
    def tags(self) -> Sequence[str]: ...

    @property
    def summary(self) -> str | None: ...


def build_projection_semantic_facts(projection: ProjectionLike) -> ProjectionSemanticFacts:
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


def build_message_semantic_facts(
    message: SemanticSessionMessageLike,
    *,
    stage_timing_add: Callable[[str, float], None] | None = None,
) -> MessageSemanticFacts:
    def add_timing(name: str, started_at: float) -> None:
        if stage_timing_add is not None:
            stage_timing_add(name, started_at)

    t0 = time.perf_counter()
    is_tool_use = message.is_tool_use
    is_thinking = message.is_thinking
    add_timing("facts.message_flags", t0)
    t0 = time.perf_counter()
    tool_calls = message_tool_calls(message) if is_tool_use else ()
    add_timing("facts.message_tool_calls", t0)
    t0 = time.perf_counter()
    actions = build_actions(message, tool_calls)
    add_timing("facts.message_actions", t0)
    t0 = time.perf_counter()
    tool_category_counts = Counter(action.kind.value for action in actions)
    add_timing("facts.message_tool_category_counts", t0)
    t0 = time.perf_counter()
    reasoning_traces = message_reasoning_traces(message) if is_thinking else ()
    add_timing("facts.message_reasoning_traces", t0)
    t0 = time.perf_counter()
    facts = MessageSemanticFacts(
        message_id=str(message.id),
        role=normalized_role_label(message.role),
        text=message.text or "",
        timestamp=message.timestamp,
        branch_index=message.branch_index,
        attachment_count=len(message.attachments),
        word_count=message.word_count,
        is_user=message.is_user,
        is_human_authored=message.is_human_authored,
        is_candidate_human_authored=message.is_candidate_human_authored,
        is_assistant=message.is_assistant,
        is_dialogue=message.is_dialogue,
        is_context_dump=message.is_context_dump,
        is_protocol_artifact=message.is_protocol_artifact,
        is_noise=message.is_noise,
        is_thinking=is_thinking,
        is_tool_use=is_tool_use,
        is_substantive=message.is_substantive,
        tool_calls=tool_calls,
        actions=actions,
        tool_category_counts=sorted_counts(dict(tool_category_counts)),
        reasoning_traces=reasoning_traces,
    )
    add_timing("facts.message_construct", t0)
    return facts


def build_session_semantic_facts(
    session: SemanticSessionLike,
    *,
    stage_timing_add: Callable[[str, float], None] | None = None,
) -> SessionSemanticFacts:
    def add_timing(name: str, started_at: float) -> None:
        if stage_timing_add is not None:
            stage_timing_add(name, started_at)

    role_counts: Counter[str] = Counter()
    tool_categories: Counter[str] = Counter()
    t0 = time.perf_counter()
    message_facts = tuple(
        build_message_semantic_facts(message, stage_timing_add=stage_timing_add) for message in session.messages
    )
    add_timing("facts.message_facts", t0)
    t0 = time.perf_counter()
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
    actions: list[Action] = []
    timestamped_messages = 0

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
        actions.extend(message_fact.actions)
        if message_fact.timestamp is not None:
            timestamps.append(message_fact.timestamp)
            timestamped_messages += 1
        if message_fact.text.strip():
            text_message_ids.append(message_fact.message_id)
            role_counts[message_fact.role] += 1
            if message_fact.timestamp is not None:
                timestamped_text_messages += 1
        for category, count in message_fact.tool_category_counts.items():
            tool_categories[category] += count
    add_timing("facts.aggregate_messages", t0)

    t0 = time.perf_counter()
    first_message_at = min(timestamps) if timestamps else None
    last_message_at = max(timestamps) if timestamps else None
    wall_duration_ms = 0
    if first_message_at and last_message_at:
        wall_duration_ms = max(int((last_message_at - first_message_at).total_seconds() * 1000), 0)
    untimestamped_messages = max(len(message_facts) - timestamped_messages, 0)
    add_timing("facts.timestamp_bounds", t0)

    t0 = time.perf_counter()
    facts = SessionSemanticFacts(
        session_id=str(session.id),
        origin=str(session.origin),
        title=session.display_title,
        date=session.display_date.isoformat() if session.display_date else None,
        total_messages=len(session.messages),
        substantive_messages=substantive_messages,
        text_messages=len(text_message_ids),
        message_ids=tuple(message_ids),
        text_message_ids=tuple(text_message_ids),
        text_role_counts=sorted_counts(dict(role_counts)),
        timestamped_text_messages=timestamped_text_messages,
        timestamped_messages=timestamped_messages,
        untimestamped_messages=untimestamped_messages,
        timestamp_coverage=_timestamp_coverage(
            total_messages=len(message_facts),
            timestamped_messages=timestamped_messages,
        ),
        attachment_count=attachment_count,
        thinking_messages=thinking_messages,
        tool_messages=tool_messages,
        branch_messages=branch_messages,
        word_count=word_count,
        tool_category_counts=sorted_counts(dict(tool_categories)),
        actions=tuple(actions),
        first_message_at=first_message_at,
        last_message_at=last_message_at,
        wall_duration_ms=wall_duration_ms,
        message_facts=message_facts,
    )
    add_timing("facts.construct", t0)
    return facts


# ---------------------------------------------------------------------------
# Session / detail builders
# ---------------------------------------------------------------------------


def _timestamp_coverage(*, total_messages: int, timestamped_messages: int) -> str:
    if total_messages <= 0 or timestamped_messages <= 0:
        return "none"
    if timestamped_messages >= total_messages:
        return "complete"
    return "partial"


def build_mcp_detail_semantic_facts(session: SemanticSessionLike) -> MCPDetailSemanticFacts:
    facts = build_session_semantic_facts(session)
    return MCPDetailSemanticFacts(
        session_id=facts.session_id,
        origin=facts.origin,
        title=facts.title,
        created_at=session.created_at.isoformat() if session.created_at else None,
        updated_at=session.updated_at.isoformat() if session.updated_at else None,
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


def build_summary_semantic_facts(summary: SemanticSummaryLike, *, message_count: int) -> SummarySemanticFacts:
    return SummarySemanticFacts(
        session_id=str(summary.id),
        origin=str(summary.origin),
        title=summary.display_title,
        date=summary.display_date.isoformat() if summary.display_date else None,
        messages=message_count,
        tags=tuple(summary.tags),
        summary=summary.summary,
    )


def build_mcp_summary_semantic_facts(
    summary: SemanticSummaryLike,
    *,
    message_count: int,
) -> MCPSummarySemanticFacts:
    return MCPSummarySemanticFacts(
        session_id=str(summary.id),
        origin=str(summary.origin),
        title=summary.display_title,
        messages=message_count,
        created_at=summary.created_at.isoformat() if summary.created_at else None,
        updated_at=summary.updated_at.isoformat() if summary.updated_at else None,
        tags=tuple(summary.tags),
        summary=summary.summary,
    )


def build_stream_semantic_facts(
    session: SemanticSessionLike,
    *,
    message_roles: MessageRoleFilter = (),
    message_limit: int | None = None,
) -> StreamSemanticFacts:
    effective_roles = message_roles

    def _passes_role_filter(message: SemanticSessionMessageLike) -> bool:
        return not effective_roles or Role.normalize(str(message.role)) in effective_roles

    filtered_messages = [message for message in session.messages if _passes_role_filter(message)]
    if message_limit is not None:
        filtered_messages = filtered_messages[:message_limit]

    visible_messages = [message for message in filtered_messages if message_has_text(message)]
    role_counts: Counter[str] = Counter(normalized_role_label(message.role) for message in visible_messages)
    timestamped_messages = sum(1 for message in visible_messages if message.timestamp is not None)

    return StreamSemanticFacts(
        session_id=str(session.id),
        origin=str(session.origin),
        title=session.display_title,
        date=session.display_date.isoformat() if session.display_date else None,
        text_messages=len(visible_messages),
        text_message_ids=tuple(str(message.id) for message in visible_messages),
        text_role_counts=sorted_counts(dict(role_counts)),
        timestamped_text_messages=timestamped_messages,
        attachment_count=sum(len(message.attachments) for message in filtered_messages),
        thinking_messages=sum(1 for message in filtered_messages if message.is_thinking),
        tool_messages=sum(1 for message in filtered_messages if message.is_tool_use),
        branch_messages=sum(1 for message in filtered_messages if message.branch_index > 0),
        message_roles=message_role_labels(effective_roles),
        message_limit=message_limit,
    )


__all__ = [
    "ProjectionAttachmentLike",
    "ProjectionLike",
    "ProjectionMessageLike",
    "SemanticSessionLike",
    "SemanticSessionMessageLike",
    "SemanticSummaryLike",
    "SessionSemanticFacts",
    "MCPDetailSemanticFacts",
    "MCPSummarySemanticFacts",
    "MessageSemanticFacts",
    "ProjectionSemanticFacts",
    "StreamSemanticFacts",
    "SummarySemanticFacts",
    "build_session_semantic_facts",
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
