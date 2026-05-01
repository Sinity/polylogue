"""Prompt registration for the MCP server."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from itertools import islice
from typing import TYPE_CHECKING, TypeAlias

from typing_extensions import TypedDict

from polylogue.mcp.payloads import MCPFencedCodeBlock
from polylogue.mcp.query_contracts import MCPConversationQueryRequest
from polylogue.types import Provider

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.lib.conversation.models import Conversation
    from polylogue.lib.message.models import Message
    from polylogue.mcp.server_support import ServerCallbacks


class ErrorContextPayload(TypedDict):
    conversation_id: str
    provider: Provider
    timestamp: str | None
    snippet: str


class ExtractedCodeSnippetPayload(TypedDict):
    language: str
    code: str
    conversation: str


class ComparedMessagePayload(TypedDict):
    role: str
    text: str


class MissingConversationPayload(TypedDict):
    error: str


class ConversationSummaryPayload(TypedDict):
    id: str
    provider: Provider
    title: str
    message_count: int
    messages: list[ComparedMessagePayload]


class ConversationPatternPayload(TypedDict):
    id: str
    provider: Provider
    title: str
    opening: list[str]


CompareConversationPayload: TypeAlias = ConversationSummaryPayload | MissingConversationPayload


def _message_role(message: Message) -> str:
    role = message.role
    return role.value if hasattr(role, "value") else str(role)


def _summarize_conversation(conv: Conversation | None) -> CompareConversationPayload:
    if conv is None:
        return {"error": "not found"}
    return {
        "id": str(conv.id),
        "provider": conv.provider,
        "title": conv.display_title,
        "message_count": len(conv.messages),
        "messages": [
            {
                "role": _message_role(message) if message.role else "unknown",
                "text": (message.text or "")[:200],
            }
            for message in islice(conv.messages, 10)
        ],
    }


def _code_snippet_payload(block: MCPFencedCodeBlock, conversation_id: str) -> ExtractedCodeSnippetPayload:
    return {
        "language": block.get("language", ""),
        "code": block.get("code", ""),
        "conversation": conversation_id[:20],
    }


def register_prompts(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    """Register MCP prompts on the given server."""

    @mcp.prompt()
    async def analyze_errors(
        provider: str | None = None,
        since: str | None = None,
        limit: int = 50,
    ) -> str:
        store = hooks.get_query_store()
        spec = MCPConversationQueryRequest(
            query="error",
            provider=provider,
            since=since,
            limit=limit,
        ).build_spec(hooks.clamp_limit)
        convs = await spec.list(store)

        error_contexts: list[ErrorContextPayload] = []
        for conv in convs:
            for msg in conv.messages:
                if msg.text and ("error" in msg.text.lower() or "exception" in msg.text.lower()):
                    error_contexts.append(
                        {
                            "conversation_id": str(conv.id),
                            "provider": conv.provider,
                            "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                            "snippet": msg.text[:200],
                        }
                    )
                    if len(error_contexts) >= 20:
                        break
            if len(error_contexts) >= 20:
                break

        return f"""Analyze error patterns from {len(convs)} conversations.

Context: {len(error_contexts)} error instances found.

Your task:
1. Identify common error patterns and root causes
2. Note which errors have known solutions in the conversations
3. Suggest preventive measures based on successful resolutions
4. Highlight any recurring pain points

Error contexts:
{json.dumps(error_contexts, indent=2)}
"""

    @mcp.prompt()
    async def summarize_week(limit: int = 100) -> str:
        store = hooks.get_query_store()
        week_ago = (datetime.now(tz=timezone.utc) - timedelta(days=7)).isoformat()
        spec = MCPConversationQueryRequest(
            since=week_ago,
            limit=limit,
        ).build_spec(hooks.clamp_limit)
        convs = await spec.list(store)

        by_provider: dict[str, int] = {}
        total_messages = 0
        for conv in convs:
            by_provider[conv.provider] = by_provider.get(conv.provider, 0) + 1
            total_messages += len(conv.messages)

        return f"""Summarize key insights from the past week's AI conversations.

Statistics:
- {len(convs)} conversations
- {total_messages} messages
- Providers: {", ".join(f"{k}({v})" for k, v in by_provider.items())}

Your task:
1. Identify main topics and themes discussed
2. Highlight key decisions or insights
3. Note any unresolved questions or ongoing work
4. Suggest follow-up actions based on the conversations

Focus on actionable insights and patterns, not exhaustive summaries.
"""

    @mcp.prompt()
    async def extract_code(language: str = "", limit: int = 50) -> str:
        store = hooks.get_query_store()
        convs = await MCPConversationQueryRequest(limit=limit).build_spec(hooks.clamp_limit).list(store)

        code_snippets: list[ExtractedCodeSnippetPayload] = []
        for conv in convs:
            for msg in conv.messages:
                if not msg.text:
                    continue
                for block in hooks.extract_fenced_code(msg.text, language):
                    code_snippets.append(_code_snippet_payload(block, str(conv.id)))
                if len(code_snippets) >= 15:
                    break

        lang_filter = f" (language: {language})" if language else ""
        return f"""Extract and organize code snippets from conversations{lang_filter}.

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
    async def compare_conversations(id1: str, id2: str) -> str:
        store = hooks.get_query_store()
        conv1 = await store.get_eager(id1)
        conv2 = await store.get_eager(id2)

        return f"""Compare these two conversations and analyze:

1. What topics/problems are discussed in each?
2. How do the approaches differ?
3. Which conversation had better outcomes?
4. What can be learned from the differences?

Conversation 1:
{json.dumps(_summarize_conversation(conv1), indent=2)}

Conversation 2:
{json.dumps(_summarize_conversation(conv2), indent=2)}
"""

    @mcp.prompt()
    async def extract_patterns(provider: str | None = None, limit: int = 30) -> str:
        store = hooks.get_query_store()
        spec = MCPConversationQueryRequest(
            provider=provider,
            limit=limit,
        ).build_spec(hooks.clamp_limit)
        convs = await spec.list(store)

        summaries: list[ConversationPatternPayload] = []
        for conv in convs:
            first_msgs = [m.text[:150] for m in conv.messages.to_list()[:3] if m.text]
            summaries.append(
                {
                    "id": str(conv.id)[:20],
                    "provider": conv.provider,
                    "title": conv.display_title,
                    "opening": first_msgs,
                }
            )

        provider_filter = f" (provider: {provider})" if provider else ""
        return f"""Analyze {len(convs)} conversations{provider_filter} for recurring patterns.

Your task:
1. Identify common topics and themes
2. Find recurring questions or problems
3. Note patterns in how AI assistants are used
4. Suggest workflow improvements based on patterns

Conversation summaries:
{json.dumps(summaries, indent=2)}
"""


__all__ = ["register_prompts"]
