"""Context pack builder — typed models for agent-facing context bundles."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from polylogue.archive.action_event.action_events import ActionEvent
    from polylogue.archive.query.spec import ConversationQuerySpec


class ContextPackProject(BaseModel):
    repo_url: str | None = None
    repo_name: str | None = None
    branch: str | None = None
    cwd_paths: list[str] = Field(default_factory=list)
    affected_paths: list[str] = Field(default_factory=list)


class ContextPackDateRange(BaseModel):
    since: str | None = None
    until: str | None = None
    earliest: str | None = None
    latest: str | None = None
    sessions_in_range: int = 0
    conversation_count_in_range: int = 0
    has_gaps: bool = False


class ContextPackQueryContext(BaseModel):
    query: str | None = None
    project_path: str | None = None
    project_repo: str | None = None
    provider: str | None = None
    query_matched: int = 0
    query_total: int = 0
    total_matching_conversations: int = 0
    conversations_included: int = 0
    match_strategy: str = "strict"
    relaxed_filters: list[str] = Field(default_factory=list)


class ContextPackMessage(BaseModel):
    role: str
    text: str
    sort_key: float | None = None
    has_tool_use: bool = False
    has_thinking: bool = False


class ContextPackConversation(BaseModel):
    conversation_id: str
    title: str | None = None
    provider: str
    created_at: str | None = None
    updated_at: str | None = None
    message_count: int = 0
    tool_use_count: int | None = None
    messages: list[ContextPackMessage] = Field(default_factory=list)
    cwd_paths: list[str] = Field(default_factory=list)
    branch_names: list[str] = Field(default_factory=list)
    affected_paths: list[str] = Field(default_factory=list)


class ContextPackActionSummary(BaseModel):
    tool_name: str
    count: int
    cwd_paths: list[str] = Field(default_factory=list)
    affected_paths: list[str] = Field(default_factory=list)


class ContextPackUnresolvedWork(BaseModel):
    conversation_id: str | None = None
    provider: str | None = None
    title: str | None = None
    description: str | None = None
    last_activity: str | None = None
    tool_use_count: int | None = None
    reason: str | None = None
    source: str = "action-event"


class ContextPackProvenance(BaseModel):
    generated_at: str = ""
    source: str = "polylogue"
    redacted: bool = True


class ContextPackIntent(BaseModel):
    """What the user was trying to accomplish (from session title + first messages)."""

    summary: str = ""
    goals: list[str] = Field(default_factory=list)


class ContextPackDecisions(BaseModel):
    """Key decisions and their context (from action events + message analysis)."""

    items: list[str] = Field(default_factory=list)


class ContextPackPayload(BaseModel):
    intent: ContextPackIntent = Field(default_factory=ContextPackIntent)
    decisions: ContextPackDecisions = Field(default_factory=ContextPackDecisions)
    project: ContextPackProject = Field(default_factory=ContextPackProject)
    date_range: ContextPackDateRange = Field(default_factory=ContextPackDateRange)
    query_context: ContextPackQueryContext = Field(default_factory=ContextPackQueryContext)
    conversations: list[ContextPackConversation] = Field(default_factory=list)
    action_summaries: list[ContextPackActionSummary] = Field(default_factory=list)
    unresolved_work: list[ContextPackUnresolvedWork] = Field(default_factory=list)
    provenance: ContextPackProvenance = Field(default_factory=ContextPackProvenance)
    total_conversations: int = 0
    total_messages: int = 0
    total_tool_calls: int = 0
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class ContextPackSelection:
    conversations: list[Any]
    match_strategy: str
    relaxed_filters: tuple[str, ...] = ()
    query_total: int = 0


@dataclass(frozen=True, slots=True)
class _ContextPackQueryAttempt:
    query: str | None
    project_path: str | None
    project_repo: str | None
    strategy: str
    relaxed_filters: tuple[str, ...] = ()


def _context_pack_recall_terms(query: str | None) -> tuple[str, ...]:
    if not query:
        return ()
    from polylogue.storage.search.query_support import extract_match_terms

    terms = extract_match_terms(query)
    # Single-letter FTS terms produce noisy archaeology packs and pure boolean
    # operators are already stripped by extract_match_terms().
    return tuple(term for term in terms if len(term) > 1)


def _context_pack_query_attempts(
    *,
    query: str | None,
    project_path: str | None,
    project_repo: str | None,
) -> tuple[_ContextPackQueryAttempt, ...]:
    attempts = [
        _ContextPackQueryAttempt(
            query=query,
            project_path=project_path,
            project_repo=project_repo,
            strategy="strict",
        )
    ]
    terms = _context_pack_recall_terms(query)
    if len(terms) <= 1:
        return tuple(attempts)

    attempts.extend(
        _ContextPackQueryAttempt(
            query=term,
            project_path=project_path,
            project_repo=project_repo,
            strategy="term_recall",
        )
        for term in terms
    )
    if project_path or project_repo:
        relaxed = tuple(
            name for name, value in (("project_path", project_path), ("project_repo", project_repo)) if value
        )
        attempts.extend(
            _ContextPackQueryAttempt(
                query=term,
                project_path=None,
                project_repo=None,
                strategy="relaxed_project_term_recall",
                relaxed_filters=relaxed,
            )
            for term in terms
        )
    return tuple(attempts)


async def select_context_pack_conversations(
    query_conversations: Callable[[ConversationQuerySpec], Awaitable[Sequence[Any]]],
    clamp_limit: Callable[[int | object], int],
    *,
    project_path: str | None,
    project_repo: str | None,
    since: str | None,
    until: str | None,
    provider: str | None,
    query: str | None,
    limit: int,
) -> ContextPackSelection:
    """Select conversations for a context pack with recall-oriented fallback.

    The context-pack surface is an archaeology/reorientation tool. A pasted
    investigative query often contains many alternative identifiers; treating
    it as one strict FTS conjunction produces false "no history" answers. We
    still run the strict request first, then fall back to single-term recall
    only when strict selection returns no conversations.
    """
    from polylogue.mcp.query_contracts import MCPConversationQueryRequest

    def _spec(attempt: _ContextPackQueryAttempt) -> ConversationQuerySpec:
        return MCPConversationQueryRequest(
            query=attempt.query,
            provider=provider,
            since=since,
            until=until,
            cwd_prefix=attempt.project_path,
            repo=attempt.project_repo,
            sort="date",
            reverse=True,
            limit=limit,
        ).build_spec(clamp_limit)

    attempts = _context_pack_query_attempts(
        query=query,
        project_path=project_path,
        project_repo=project_repo,
    )
    strict = list(await query_conversations(_spec(attempts[0])))
    if strict:
        return ContextPackSelection(conversations=strict[:limit], match_strategy="strict", query_total=len(strict))

    for strategy in ("term_recall", "relaxed_project_term_recall"):
        merged: list[Any] = []
        seen: set[str] = set()
        relaxed_filters: tuple[str, ...] = ()
        for attempt in attempts:
            if attempt.strategy != strategy:
                continue
            for conversation in await query_conversations(_spec(attempt)):
                conv_id = str(getattr(conversation, "id", ""))
                if conv_id and conv_id in seen:
                    continue
                if conv_id:
                    seen.add(conv_id)
                merged.append(conversation)
                if len(merged) >= limit:
                    break
            relaxed_filters = attempt.relaxed_filters
            if len(merged) >= limit:
                break
        if merged:
            return ContextPackSelection(
                conversations=merged,
                match_strategy=strategy,
                relaxed_filters=relaxed_filters,
                query_total=len(merged),
            )

    return ContextPackSelection(conversations=[], match_strategy="strict", query_total=0)


def redact_path(path: str) -> str:
    import os

    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home) :]
    return path


def _build_project_context(
    action_events: Sequence[ActionEvent],
    *,
    redact: bool = True,
) -> ContextPackProject:
    """Build project context from action events across matched conversations."""
    cwd_set: list[str] = []
    branch_set: list[str] = []
    affected_set: list[str] = []

    for event in action_events:
        if event.cwd_path:
            cwd_display = redact_path(event.cwd_path) if redact else event.cwd_path
            if cwd_display not in cwd_set:
                cwd_set.append(cwd_display)
        for branch in event.branch_names:
            if branch and branch not in branch_set:
                branch_set.append(branch)
        for path in event.affected_paths:
            affected = redact_path(path) if redact else path
            if affected not in affected_set:
                affected_set.append(affected)

    # Use first non-empty branch as primary
    primary_branch = branch_set[0] if branch_set else None

    return ContextPackProject(
        branch=primary_branch,
        cwd_paths=cwd_set,
        affected_paths=affected_set,
    )


def _summarize_action_events(
    action_events: Sequence[ActionEvent],
    *,
    redact: bool = True,
) -> list[ContextPackActionSummary]:
    """Aggregate action events into tool-level summaries."""
    counts: dict[str, int] = {}
    cwds: dict[str, set[str]] = {}
    paths: dict[str, set[str]] = {}
    for event in action_events:
        tool = event.normalized_tool_name
        counts[tool] = counts.get(tool, 0) + 1
        if event.cwd_path:
            cwds.setdefault(tool, set()).add(redact_path(event.cwd_path) if redact else event.cwd_path)
        for path in event.affected_paths:
            paths.setdefault(tool, set()).add(redact_path(path) if redact else path)
    return [
        ContextPackActionSummary(
            tool_name=tool,
            count=cnt,
            cwd_paths=sorted(cwds.get(tool, set())),
            affected_paths=sorted(paths.get(tool, set())),
        )
        for tool, cnt in sorted(counts.items(), key=lambda x: -x[1])
    ]
