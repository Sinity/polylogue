"""Context pack builder — typed models for agent-facing context bundles."""

from __future__ import annotations

from pydantic import BaseModel, Field


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
    message_count: int = 0
    messages: list[ContextPackMessage] = Field(default_factory=list)


class ContextPackActionSummary(BaseModel):
    tool_name: str
    count: int
    cwd_paths: list[str] = Field(default_factory=list)
    affected_paths: list[str] = Field(default_factory=list)


class ContextPackUnresolvedWork(BaseModel):
    description: str
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
    conversations: list[dict[str, object]],
    *,
    redact: bool = True,
) -> ContextPackProject:
    return ContextPackProject()


def _summarize_action_events(
    action_events: list[dict[str, object]],
    *,
    redact: bool = True,
) -> list[ContextPackActionSummary]:
    counts: dict[str, int] = {}
    cwds: dict[str, set[str]] = {}
    for evt in action_events:
        tool = str(evt.get("normalized_tool_name", evt.get("action_kind", "unknown")))
        counts[tool] = counts.get(tool, 0) + 1
        cwd = evt.get("cwd")
        if cwd and isinstance(cwd, str):
            cwds.setdefault(tool, set()).add(cwd)
    return [
        ContextPackActionSummary(
            tool_name=tool,
            count=cnt,
            cwd_paths=sorted(cwds.get(tool, set())),
        )
        for tool, cnt in sorted(counts.items(), key=lambda x: -x[1])
    ]
