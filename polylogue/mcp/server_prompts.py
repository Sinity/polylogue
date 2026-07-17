"""Prompt registration for the MCP server."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from typing_extensions import TypedDict

from polylogue.archive.query.transaction import run_archive_read
from polylogue.mcp.archive_support import archive_query_filters, mcp_archive_root
from polylogue.mcp.payloads import MCPFencedCodeBlock
from polylogue.mcp.query_contracts import MCPSessionQueryRequest

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.archive.message.models import Message
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.mcp.server_support import ServerCallbacks
    from polylogue.storage.sqlite.archive_tiers.archive import (
        ArchiveSessionSearchHit,
        ArchiveSessionSummary,
        ArchiveStore,
    )
    from polylogue.storage.sqlite.archive_tiers.write import ArchiveSessionEnvelope


class ErrorContextPayload(TypedDict):
    session_id: str
    origin: str
    timestamp: str | None
    snippet: str


class ExtractedCodeSnippetPayload(TypedDict):
    language: str
    code: str
    session: str


class ComparedMessagePayload(TypedDict):
    role: str
    text: str


class MissingSessionPayload(TypedDict):
    error: str


class SessionSummaryPayload(TypedDict):
    id: str
    origin: str
    title: str
    message_count: int
    messages: list[ComparedMessagePayload]


class SessionPatternPayload(TypedDict):
    id: str
    origin: str
    title: str
    opening: list[str]


CompareSessionPayload: TypeAlias = SessionSummaryPayload | MissingSessionPayload


@dataclass(frozen=True, slots=True)
class PromptMessage:
    role: str
    text: str
    timestamp: str | None = None


@dataclass(frozen=True, slots=True)
class PromptSession:
    id: str
    origin: str
    display_title: str
    messages: tuple[PromptMessage, ...]


def _message_role(message: Message) -> str:
    role = message.role
    return role.value if hasattr(role, "value") else str(role)


def _prompt_message_role(message: Message | PromptMessage) -> str:
    if isinstance(message, PromptMessage):
        return message.role
    return _message_role(message) if message.role else "unknown"


def _prompt_message_text(message: Message | PromptMessage) -> str:
    return message.text or ""


def _prompt_message_timestamp(message: Message | PromptMessage) -> str | None:
    if isinstance(message, PromptMessage):
        return message.timestamp
    return message.timestamp.isoformat() if message.timestamp else None


def _prompt_messages(session: Any) -> list[Message | PromptMessage]:
    messages = session.messages
    to_list = getattr(messages, "to_list", None)
    if callable(to_list):
        return list(to_list())
    return list(messages)


def _summarize_session(conv: Any | None) -> CompareSessionPayload:
    if conv is None:
        return {"error": "not found"}
    messages = _prompt_messages(conv)
    return {
        "id": str(conv.id),
        "origin": str(conv.origin),
        "title": conv.display_title,
        "message_count": len(messages),
        "messages": [
            {
                "role": _prompt_message_role(message),
                "text": _prompt_message_text(message)[:200],
            }
            for message in islice(messages, 10)
        ],
    }


def _code_snippet_payload(block: MCPFencedCodeBlock, session_id: str) -> ExtractedCodeSnippetPayload:
    return {
        "language": block.get("language", ""),
        "code": block.get("code", ""),
        "session": session_id[:20],
    }


def _archive_prompt_session(session: ArchiveSessionEnvelope) -> PromptSession:
    return PromptSession(
        id=session.session_id,
        origin=session.origin,
        display_title=session.title or "(untitled)",
        messages=tuple(
            PromptMessage(
                role=message.role,
                text="\n\n".join(block.text for block in message.blocks if block.text),
                timestamp=message.occurred_at,
            )
            for message in session.messages
        ),
    )


def _archive_prompt_session_page(archive: ArchiveStore, session_id: str, *, limit: int = 20) -> PromptSession:
    """Build prompt context from a bounded message projection."""
    summary = archive.read_summary(session_id)
    rows = archive.query_session_messages((session_id,), limit=limit, offset=0)
    return PromptSession(
        id=summary.session_id,
        origin=summary.origin,
        display_title=summary.title or "(untitled)",
        messages=tuple(
            PromptMessage(
                role=row.role,
                text=row.text[:4000],
                timestamp=(
                    datetime.fromtimestamp(row.occurred_at_ms / 1000.0, UTC).isoformat()
                    if row.occurred_at_ms is not None
                    else None
                ),
            )
            for row in rows
        ),
    )


def _archive_prompt_sessions(archive: ArchiveStore, spec: SessionQuerySpec) -> list[PromptSession]:
    filters = archive_query_filters(spec)
    limit = spec.limit or 10
    query = " ".join(spec.query_terms).strip()
    if query:
        rows: list[ArchiveSessionSummary | ArchiveSessionSearchHit] = list(
            archive.search_summaries(
                query,
                limit=limit,
                offset=spec.offset,
                sort=None,
                reverse=spec.reverse,
                **filters,
            )
        )
    else:
        rows = list(
            archive.list_summaries(
                limit=limit,
                offset=spec.offset,
                sort=None,
                reverse=spec.reverse,
                sample=bool(spec.sample),
                **filters,
            )
        )
    sessions: list[PromptSession] = []
    seen: set[str] = set()
    for row in rows:
        session_id = row.session_id
        if session_id in seen:
            continue
        seen.add(session_id)
        sessions.append(_archive_prompt_session_page(archive, session_id))
    return sessions


def _archive_prompt_session_by_id(archive: ArchiveStore, token: str) -> PromptSession | None:
    try:
        session_id = archive.resolve_session_id(token)
        return _archive_prompt_session_page(archive, session_id)
    except (KeyError, ValueError):
        return None


async def _prompt_sessions_from_config(hooks: Any, spec: SessionQuerySpec) -> list[PromptSession]:
    config = hooks.get_config()
    try:
        return await run_archive_read(
            mcp_archive_root(config),
            operation="prompt.sessions",
            arguments={
                "query": spec.query_terms,
                "limit": spec.limit,
                "offset": spec.offset,
                "origins": spec.origins,
                "since": spec.since,
            },
            work=lambda archive: _archive_prompt_sessions(archive, spec),
            page_size=spec.limit,
            offset=spec.offset,
            projection="prompt-session",
            stable_order="date,session_id",
        )
    except sqlite3.OperationalError:
        return []


async def _prompt_session_by_id_from_config(hooks: Any, token: str) -> PromptSession | None:
    config = hooks.get_config()
    try:
        return await run_archive_read(
            mcp_archive_root(config),
            operation="prompt.session",
            arguments={"session_id": token},
            work=lambda archive: _archive_prompt_session_by_id(archive, token),
            projection="prompt-session",
        )
    except sqlite3.OperationalError:
        return None


def register_prompts(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    """Register MCP prompts on the given server."""

    @mcp.prompt()
    async def agent_coordination_brief(view: str = "status", limit: int = 10, detail: bool = False) -> str:
        from polylogue.coordination import CoordinationView, build_coordination_envelope

        normalized_view: CoordinationView
        if view in {"status", "self", "work-item", "conflicts", "handoff"}:
            normalized_view = cast(CoordinationView, view)
        else:
            normalized_view = "status"
        payload = build_coordination_envelope(
            view=normalized_view,
            limit=hooks.clamp_limit(limit),
            detail=detail,
        )
        return f"""Use this bounded coordination envelope to decide the next agent action.

Rules:
1. Treat overlaps as awareness unless the payload marks them blocking.
2. Prefer the work_item source with the highest confidence.
3. Use handoff refs and advisories as navigation, not as a separate source of truth.

Envelope:
{payload.to_json(exclude_none=True)}
"""

    @mcp.prompt()
    async def analyze_errors(
        origin: str | None = None,
        since: str | None = None,
        limit: int = 50,
    ) -> str:
        spec = MCPSessionQueryRequest(
            query="error",
            origin=origin,
            since=since,
            limit=limit,
        ).build_spec(hooks.clamp_limit)
        convs = await _prompt_sessions_from_config(hooks, spec)

        error_contexts: list[ErrorContextPayload] = []
        for conv in convs:
            for msg in _prompt_messages(conv):
                text = _prompt_message_text(msg)
                if text and ("error" in text.lower() or "exception" in text.lower()):
                    error_contexts.append(
                        {
                            "session_id": str(conv.id),
                            "origin": str(conv.origin),
                            "timestamp": _prompt_message_timestamp(msg),
                            "snippet": text[:200],
                        }
                    )
                    if len(error_contexts) >= 20:
                        break
            if len(error_contexts) >= 20:
                break

        return f"""Analyze error patterns from {len(convs)} sessions.

Context: {len(error_contexts)} error instances found.

Your task:
1. Identify common error patterns and root causes
2. Note which errors have known solutions in the sessions
3. Suggest preventive measures based on successful resolutions
4. Highlight any recurring pain points

Error contexts:
{json.dumps(error_contexts, indent=2)}
"""

    @mcp.prompt()
    async def summarize_week(limit: int = 100) -> str:
        week_ago = (datetime.now(tz=timezone.utc) - timedelta(days=7)).isoformat()
        spec = MCPSessionQueryRequest(
            since=week_ago,
            limit=limit,
        ).build_spec(hooks.clamp_limit)
        convs = await _prompt_sessions_from_config(hooks, spec)

        by_origin: dict[str, int] = {}
        total_messages = 0
        for conv in convs:
            origin = str(conv.origin)
            by_origin[origin] = by_origin.get(origin, 0) + 1
            total_messages += len(_prompt_messages(conv))

        return f"""Summarize key insights from the past week's AI sessions.

Statistics:
- {len(convs)} sessions
- {total_messages} messages
- Origins: {", ".join(f"{k}({v})" for k, v in by_origin.items())}

Your task:
1. Identify main topics and themes discussed
2. Highlight key decisions or insights
3. Note any unresolved questions or ongoing work
4. Suggest follow-up actions based on the sessions

Focus on actionable insights and patterns, not exhaustive summaries.
"""

    @mcp.prompt()
    async def extract_code(language: str = "", limit: int = 50) -> str:
        spec = MCPSessionQueryRequest(limit=limit).build_spec(hooks.clamp_limit)
        convs = await _prompt_sessions_from_config(hooks, spec)

        code_snippets: list[ExtractedCodeSnippetPayload] = []
        for conv in convs:
            for msg in _prompt_messages(conv):
                text = _prompt_message_text(msg)
                if not text:
                    continue
                for block in hooks.extract_fenced_code(text, language):
                    code_snippets.append(_code_snippet_payload(block, str(conv.id)))
                if len(code_snippets) >= 15:
                    break

        lang_filter = f" (language: {language})" if language else ""
        return f"""Extract and organize code snippets from sessions{lang_filter}.

Found {len(code_snippets)} code blocks.

Your task:
1. Categorize code snippets by purpose/functionality
2. Identify reusable patterns or utilities
3. Note any incomplete or problematic code
4. Suggest organization into a knowledge base

Code snippets:
{json.dumps(code_snippets, indent=2)}
"""

    @mcp.prompt()
    async def compare_sessions(id1: str, id2: str) -> str:
        conv1 = await _prompt_session_by_id_from_config(hooks, id1)
        conv2 = await _prompt_session_by_id_from_config(hooks, id2)

        return f"""Compare these two sessions and analyze:

1. What topics/problems are discussed in each?
2. How do the approaches differ?
3. Which session had better outcomes?
4. What can be learned from the differences?

Session 1:
{json.dumps(_summarize_session(conv1), indent=2)}

Session 2:
{json.dumps(_summarize_session(conv2), indent=2)}
"""

    @mcp.prompt()
    async def extract_patterns(origin: str | None = None, limit: int = 30) -> str:
        spec = MCPSessionQueryRequest(
            origin=origin,
            limit=limit,
        ).build_spec(hooks.clamp_limit)
        convs = await _prompt_sessions_from_config(hooks, spec)

        summaries: list[SessionPatternPayload] = []
        for conv in convs:
            first_msgs = [_prompt_message_text(message)[:150] for message in _prompt_messages(conv)[:3]]
            summaries.append(
                {
                    "id": str(conv.id)[:20],
                    "origin": str(conv.origin),
                    "title": conv.display_title,
                    "opening": first_msgs,
                }
            )

        origin_filter = f" (origin: {origin})" if origin else ""
        return f"""Analyze {len(convs)} sessions{origin_filter} for recurring patterns.

Your task:
1. Identify common topics and themes
2. Find recurring questions or problems
3. Note patterns in how AI assistants are used
4. Suggest workflow improvements based on patterns

Session summaries:
{json.dumps(summaries, indent=2)}
"""

    @mcp.prompt()
    async def resume_context(repo: str = "", limit: int = 5) -> str:
        """Rebuild working context for the current repo from the archive."""
        repo_name, cwd = _repo_context(repo)
        return f"""Rebuild working context for repo '{repo_name}' from the Polylogue archive.

Call sequence:
1. find_resume_candidates(repo_path="{cwd}", limit={limit}) — ranked resumable logical sessions for this checkout.
2. get_resume_brief(session_id=<top candidate>) — typed brief: goals, open threads, next actions, provenance refs.
3. agent_coordination_brief(view="self") — check concurrent agents/worktrees before claiming work.
4. blackboard_list(scope_repo="{repo_name}", unresolved=True) — unresolved notes/handoffs addressed to agents here.

Rules:
- Cite refs (session_id, message_id) instead of pasting transcripts; fetch full text only for messages you will act on.
- Empty results usually mean a wrong archive root: the harness must point POLYLOGUE_ARCHIVE_ROOT at the live archive, not a sandbox default.
"""

    @mcp.prompt()
    async def postmortem_last(repo: str = "", since: str = "14d") -> str:
        """Postmortem the most recent failed or abandoned session for a repo."""
        repo_name, cwd = _repo_context(repo)
        return f"""Postmortem the most recent failed or abandoned session for repo '{repo_name}'.

Call sequence:
1. find_abandoned_sessions(repo_path="{cwd}", since="{since}") and find_stuck_sessions(since="{since}") — candidates with dangling work or stuck tool calls.
2. Pick the most recent relevant session; orient with get_session_summary(id=<session_id>).
3. get_postmortem_bundle(repo="{repo_name}", since="{since}") — forensic bundle: timeline, decisions, tool errors.
4. get_pathologies(repo="{repo_name}", since="{since}") — detected anti-patterns in the same window.

Report: what was attempted, where it failed (cite tool_result errors by ref), what remained undone, and the smallest next action.
"""

    @mcp.prompt()
    async def decisions_about(topic: str, limit: int = 20) -> str:
        """Find what was decided about a topic, recorded and inferred."""
        return f"""Find what was decided about '{topic}'.

Call sequence:
1. list_assertion_claims(kinds="decision,judgment,lesson", statuses="active,candidate", limit={limit}) — recorded decisions (authoritative when user-authored).
2. query_units(expression='assertions where kind:decision AND text:"{topic}"') — targeted assertion search.
3. search(query='near:"{topic}"', limit=10) — decision discussions never recorded as assertions.

Rules:
- Recorded assertions outrank inferred prose; label each finding as recorded vs inferred.
- status:candidate rows are agent-proposed and unreviewed — never present them as settled.
"""

    @mcp.prompt()
    async def unacknowledged_failures(repo: str = "", since: str = "7d") -> str:
        """Surface recent failures that nobody acknowledged."""
        repo_name, _cwd = _repo_context(repo)
        return f"""Surface failures in '{repo_name}' since {since} that were never acknowledged.

Call sequence:
1. query_units(expression='actions where session.repo:{repo_name} since:{since} AND output:failed', limit=20) — action rows for sessions containing failed tool outcomes.
2. find_stuck_sessions(since="{since}") — sessions whose provider tool calls are bounded as stuck.
3. For each hit: list_marks(session_id=<session_id>) and list_annotations for that session — an existing mark/annotation means acknowledged.

Report only sessions with failures and no acknowledgment; cite the failing action refs (tool, path, error).
"""

    @mcp.prompt()
    async def sessions_touching_file(path: str, repo: str = "") -> str:
        """Find sessions that edited or referenced a file path."""
        repo_name, _cwd = _repo_context(repo)
        repo_clause = f"repo:{repo_name} AND " if repo_name else ""
        return f"""Find sessions that touched file path '{path}'.

Call sequence:
1. query_units(expression='files where {repo_clause}path:{path}', limit=20) — file/action rows for sessions that touched the path.
2. query_units(expression='files where path:{path}', limit=20) — per-file action rows (edits, reads, shell references).
3. search(query='"{path}"', limit=10) — mentions in prose that never became edits.
4. get_session_summary(id=<session_id>) on each hit for orientation.

Rules: path matching is substring — prefer repo-relative fragments (e.g. polylogue/mcp/server_prompts.py) over bare filenames.
"""

    @mcp.prompt()
    async def cost_of(since: str = "30d", limit: int = 10) -> str:
        """Report cost/usage for a time window with honest accounting."""
        return f"""Report cost/usage for the last {since}, with honest accounting.

Call sequence:
1. cost_rollups(since="{since}") — aggregate spend by model.
2. session_costs(since="{since}", limit={limit}) — top sessions by cost.
3. provider_usage(detail="summary") — usage accounting diagnostics without billing estimates.
4. For one repo's sessions: search(query='repo:<repo> since:{since}') then session_costs(session_id=<id>) per hit — cost tools have no repo filter.

Rules:
- cost_usd is API-list-equivalent; on subscription plans report the subscription-credit view separately (cache reads are ~free on Claude Max/Pro).
- Codex 'input' includes cached tokens and 'output' includes reasoning — treat lanes as disjoint, never sum naively.
"""


def _repo_context(repo: str) -> tuple[str, str]:
    cwd = Path.cwd()
    if repo:
        return repo, str(cwd)
    return cwd.name, str(cwd)


__all__ = ["register_prompts"]
