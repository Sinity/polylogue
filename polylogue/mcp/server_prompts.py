"""Prompt registration for the MCP server."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import islice
from typing import TYPE_CHECKING, Any, TypeAlias

from typing_extensions import TypedDict

from polylogue.mcp.archive_support import active_archive_root, archive_query_filters
from polylogue.mcp.payloads import MCPFencedCodeBlock
from polylogue.mcp.query_contracts import MCPSessionQueryRequest
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.archive.message.models import Message
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.mcp.server_support import ServerCallbacks
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSearchHit, ArchiveSessionSummary
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
        sessions.append(_archive_prompt_session(archive.read_session(session_id)))
    return sessions


def _archive_prompt_session_by_id(archive: ArchiveStore, token: str) -> PromptSession | None:
    try:
        session_id = archive.resolve_session_id(token)
        return _archive_prompt_session(archive.read_session(session_id))
    except (KeyError, ValueError):
        return None


def register_prompts(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    """Register MCP prompts on the given server."""

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
        config = hooks.get_config()
        convs: list[Any]
        with ArchiveStore.open_existing(active_archive_root(config) or config.archive_root) as archive:
            convs = _archive_prompt_sessions(archive, spec)

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
        config = hooks.get_config()
        convs: list[Any]
        with ArchiveStore.open_existing(active_archive_root(config) or config.archive_root) as archive:
            convs = _archive_prompt_sessions(archive, spec)

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
        config = hooks.get_config()
        convs: list[Any]
        with ArchiveStore.open_existing(active_archive_root(config) or config.archive_root) as archive:
            convs = _archive_prompt_sessions(archive, spec)

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
        config = hooks.get_config()
        conv1: Any | None
        conv2: Any | None
        with ArchiveStore.open_existing(active_archive_root(config) or config.archive_root) as archive:
            conv1 = _archive_prompt_session_by_id(archive, id1)
            conv2 = _archive_prompt_session_by_id(archive, id2)

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
        config = hooks.get_config()
        convs: list[Any]
        with ArchiveStore.open_existing(active_archive_root(config) or config.archive_root) as archive:
            convs = _archive_prompt_sessions(archive, spec)

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


__all__ = ["register_prompts"]
