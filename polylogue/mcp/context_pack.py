"""Context pack builder — typed models for agent-facing context bundles."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from polylogue.archive.action_event.action_events import ActionEvent


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


class ContextPackPayload(BaseModel):
    project: ContextPackProject = Field(default_factory=ContextPackProject)
    date_range: ContextPackDateRange = Field(default_factory=ContextPackDateRange)
    query_context: ContextPackQueryContext = Field(default_factory=ContextPackQueryContext)
    conversations: list[ContextPackConversation] = Field(default_factory=list)
    action_summaries: list[ContextPackActionSummary] = Field(default_factory=list)
    unresolved_work: list[ContextPackUnresolvedWork] = Field(default_factory=list)
    provenance: ContextPackProvenance = Field(default_factory=ContextPackProvenance)


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
