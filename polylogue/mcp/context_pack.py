"""Context pack builder — typed models for agent-facing context bundles."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TypedDict

from pydantic import BaseModel, Field

from polylogue.mcp.archive_support import archive_index_active_paths, archive_query_filters
from polylogue.storage.sqlite.archive_tiers.archive import (
    ArchiveSessionSearchHit,
    ArchiveSessionSummary,
    ArchiveStore,
)

if TYPE_CHECKING:
    from polylogue.archive.actions.actions import Action
    from polylogue.archive.query.spec import SessionQuerySpec


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
    session_count_in_range: int = 0
    has_gaps: bool = False


class ContextPackQueryContext(BaseModel):
    query: str | None = None
    project_path: str | None = None
    project_repo: str | None = None
    origin: str | None = None
    query_matched: int = 0
    query_total: int = 0
    total_matching_sessions: int = 0
    sessions_included: int = 0
    match_strategy: str = "strict"
    relaxed_filters: list[str] = Field(default_factory=list)


class ContextPackMessage(BaseModel):
    role: str
    text: str
    sort_key: float | None = None
    has_tool_use: bool = False
    has_thinking: bool = False


class ContextPackSession(BaseModel):
    session_id: str
    title: str | None = None
    origin: str
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
    session_id: str | None = None
    origin: str | None = None
    title: str | None = None
    description: str | None = None
    last_activity: str | None = None
    tool_use_count: int | None = None
    reason: str | None = None
    source: str = "action"


class ContextPackProvenance(BaseModel):
    generated_at: str = ""
    source: str = "polylogue"
    redacted: bool = True
    archive_runtime: str = "archive_file_set"
    archive_root: str | None = None
    active_db_path: str | None = None
    redaction_policy: str = "public_refs_and_redacted_paths"


class ContextPackScope(BaseModel):
    """What a context-pack selected and bounded for handoff."""

    seed_refs: list[str] = Field(default_factory=list)
    read_views: list[str] = Field(default_factory=lambda: ["context-pack"])
    project_path: str | None = None
    project_repo: str | None = None
    origin: str | None = None
    since: str | None = None
    until: str | None = None
    query: str | None = None
    include_messages: bool = True
    limits: dict[str, int] = Field(default_factory=dict)


class ContextPackOmission(BaseModel):
    """Material intentionally omitted or unavailable in a context-pack."""

    ref: str | None = None
    query: str | None = None
    view: str | None = None
    reason: str
    detail: str
    evidence_refs: list[str] = Field(default_factory=list)


class ContextPackSizeEstimate(BaseModel):
    """Approximate byte/token posture for context-pack consumers."""

    json_bytes: int = 0
    message_text_bytes: int = 0
    session_count: int = 0
    message_count: int = 0
    token_estimate: int = 0


class ContextPackIntent(BaseModel):
    """What the user was trying to accomplish (from session title + first messages)."""

    summary: str = ""
    goals: list[str] = Field(default_factory=list)


class ContextPackDecisions(BaseModel):
    """Key decisions and their context (from actions + message analysis)."""

    items: list[str] = Field(default_factory=list)


class ContextPackPayload(BaseModel):
    selection_strategy: str = "strict"
    scope: ContextPackScope = Field(default_factory=ContextPackScope)
    omissions: list[ContextPackOmission] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    redaction_policy: str = "public_refs_and_redacted_paths"
    token_estimate: int = 0
    size_estimate: ContextPackSizeEstimate = Field(default_factory=ContextPackSizeEstimate)
    intent: ContextPackIntent = Field(default_factory=ContextPackIntent)
    decisions: ContextPackDecisions = Field(default_factory=ContextPackDecisions)
    project: ContextPackProject = Field(default_factory=ContextPackProject)
    date_range: ContextPackDateRange = Field(default_factory=ContextPackDateRange)
    query_context: ContextPackQueryContext = Field(default_factory=ContextPackQueryContext)
    sessions: list[ContextPackSession] = Field(default_factory=list)
    action_summaries: list[ContextPackActionSummary] = Field(default_factory=list)
    unresolved_work: list[ContextPackUnresolvedWork] = Field(default_factory=list)
    provenance: ContextPackProvenance = Field(default_factory=ContextPackProvenance)
    total_sessions: int = 0
    total_messages: int = 0
    total_tool_calls: int = 0
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class ContextPackSelection:
    sessions: list[Any]
    match_strategy: str
    relaxed_filters: tuple[str, ...] = ()
    query_total: int = 0


class ArchiveContextPackFilters(TypedDict):
    origins: tuple[str, ...]
    excluded_origins: tuple[str, ...]
    tags: tuple[str, ...]
    excluded_tags: tuple[str, ...]
    repo_names: tuple[str, ...]
    cwd_prefix: str | None
    since_ms: int | None
    until_ms: int | None


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


async def select_context_pack_sessions(
    query_sessions: Callable[[SessionQuerySpec], Awaitable[Sequence[Any]]],
    clamp_limit: Callable[[int | object], int],
    *,
    project_path: str | None,
    project_repo: str | None,
    since: str | None,
    until: str | None,
    origin: str | None,
    query: str | None,
    limit: int,
) -> ContextPackSelection:
    """Select sessions for a context pack with recall-oriented fallback.

    The context-pack surface is an archaeology/reorientation tool. A pasted
    investigative query often contains many alternative identifiers; treating
    it as one strict FTS conjunction produces false "no history" answers. We
    still run the strict request first, then fall back to single-term recall
    only when strict selection returns no sessions.
    """
    from polylogue.mcp.query_contracts import MCPSessionQueryRequest

    def _spec(attempt: _ContextPackQueryAttempt) -> SessionQuerySpec:
        return MCPSessionQueryRequest(
            query=attempt.query,
            origin=origin,
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
    strict = list(await query_sessions(_spec(attempts[0])))
    if strict:
        return ContextPackSelection(sessions=strict[:limit], match_strategy="strict", query_total=len(strict))

    for strategy in ("term_recall", "relaxed_project_term_recall"):
        merged: list[Any] = []
        seen: set[str] = set()
        relaxed_filters: tuple[str, ...] = ()
        for attempt in attempts:
            if attempt.strategy != strategy:
                continue
            for session in await query_sessions(_spec(attempt)):
                conv_id = str(getattr(session, "id", ""))
                if conv_id and conv_id in seen:
                    continue
                if conv_id:
                    seen.add(conv_id)
                merged.append(session)
                if len(merged) >= limit:
                    break
            relaxed_filters = attempt.relaxed_filters
            if len(merged) >= limit:
                break
        if merged:
            return ContextPackSelection(
                sessions=merged,
                match_strategy=strategy,
                relaxed_filters=relaxed_filters,
                query_total=len(merged),
            )

    return ContextPackSelection(sessions=[], match_strategy="strict", query_total=0)


def archive_context_pack_active(
    *,
    archive_root: Path,
    db_anchor_path: Path,
) -> bool:
    """Return whether context-pack should read archive index data."""
    return archive_index_active_paths(
        archive_root=archive_root,
        db_anchor_path=db_anchor_path,
    )


def query_archive_context_pack(
    archive: ArchiveStore,
    spec: SessionQuerySpec,
    *,
    default_limit: int,
) -> list[SimpleNamespace]:
    """Project archive sessions into the context-pack summary surface."""
    query = " ".join(spec.query_terms).strip()
    kwargs = archive_context_pack_filters(spec)
    if query:
        rows: list[ArchiveSessionSummary | ArchiveSessionSearchHit] = list(
            archive.search_summaries(
                query,
                limit=spec.limit or default_limit,
                offset=spec.offset,
                sort="date",
                reverse=spec.reverse,
                **kwargs,
            )
        )
    else:
        rows = list(
            archive.list_summaries(
                limit=spec.limit or default_limit,
                offset=spec.offset,
                sort="date",
                reverse=spec.reverse,
                **kwargs,
            )
        )

    summaries: list[ArchiveSessionSummary] = []
    for row in dedupe_archive_context_pack_rows(rows):
        if isinstance(row, ArchiveSessionSearchHit):
            try:
                summaries.append(archive.read_summary(row.session_id))
            except KeyError:
                continue
        else:
            summaries.append(row)
    return [archive_context_pack_summary(row) for row in summaries]


def archive_context_pack_filters(spec: SessionQuerySpec) -> ArchiveContextPackFilters:
    filters = archive_query_filters(spec)
    return {
        "origins": filters["origins"],
        "excluded_origins": filters["excluded_origins"],
        "tags": filters["tags"],
        "excluded_tags": filters["excluded_tags"],
        "repo_names": filters["repo_names"],
        "cwd_prefix": filters["cwd_prefix"],
        "since_ms": filters["since_ms"],
        "until_ms": filters["until_ms"],
    }


def archive_context_pack_summary(row: ArchiveSessionSummary) -> SimpleNamespace:
    return SimpleNamespace(
        id=row.session_id,
        origin=row.origin,
        title=row.title,
        display_title=row.title,
        created_at=_parse_archive_datetime(row.created_at),
        updated_at=_parse_archive_datetime(row.updated_at),
        message_count=row.message_count,
        messages=(),
        tool_use_count=0,
    )


def dedupe_archive_context_pack_rows(
    rows: list[ArchiveSessionSummary | ArchiveSessionSearchHit],
) -> list[ArchiveSessionSummary | ArchiveSessionSearchHit]:
    deduped: list[ArchiveSessionSummary | ArchiveSessionSearchHit] = []
    seen: set[str] = set()
    for row in rows:
        session_id = row.session_id
        if session_id in seen:
            continue
        seen.add(session_id)
        deduped.append(row)
    return deduped


def _parse_archive_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def redact_path(path: str) -> str:
    import os

    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home) :]
    return path


def _build_project_context(
    actions: Sequence[Action],
    *,
    redact: bool = True,
) -> ContextPackProject:
    """Build project context from actions across matched sessions."""
    cwd_set: list[str] = []
    branch_set: list[str] = []
    affected_set: list[str] = []

    for event in actions:
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


def _summarize_actions(
    actions: Sequence[Action],
    *,
    redact: bool = True,
) -> list[ContextPackActionSummary]:
    """Aggregate actions into tool-level summaries."""
    counts: dict[str, int] = {}
    cwds: dict[str, set[str]] = {}
    paths: dict[str, set[str]] = {}
    for action in actions:
        tool = action.normalized_tool_name
        counts[tool] = counts.get(tool, 0) + 1
        if action.cwd_path:
            cwds.setdefault(tool, set()).add(redact_path(action.cwd_path) if redact else action.cwd_path)
        for path in action.affected_paths:
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
