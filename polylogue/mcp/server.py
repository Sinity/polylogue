"""MCP server implementation for Polylogue using the official MCP SDK.

Provides conversation search and retrieval capabilities via the Model Context Protocol.
Uses stdio transport for communication with AI assistants (Claude Desktop, Cursor, etc.).

Rewritten from hand-rolled JSON-RPC to the official ``mcp`` SDK (FastMCP) for
proper protocol compliance, typed responses, and automatic schema generation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.lib.models import Conversation
    from polylogue.storage.repository import ConversationRepository

logger = logging.getLogger(__name__)

_MAX_LIMIT = 10000

# ---------------------------------------------------------------------------
# Lazy repository singleton
# ---------------------------------------------------------------------------

_repo_instance: ConversationRepository | None = None


def _get_repo() -> ConversationRepository:
    """Lazily initialize and return the ConversationRepository singleton."""
    global _repo_instance
    if _repo_instance is None:
        from polylogue.storage.backends.sqlite import create_default_backend
        from polylogue.storage.repository import ConversationRepository

        backend = create_default_backend()
        _repo_instance = ConversationRepository(backend=backend)
    return _repo_instance


# ---------------------------------------------------------------------------
# Serialization helpers (preserved from previous implementation)
# ---------------------------------------------------------------------------


def _conversation_to_dict(conv: Conversation) -> dict[str, Any]:
    """Convert Conversation to a JSON-serializable dict."""
    return {
        "id": str(conv.id),
        "provider": conv.provider,
        "title": conv.display_title,
        "message_count": len(conv.messages),
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
    }


def _conversation_to_full_dict(conv: Conversation) -> dict[str, Any]:
    """Convert Conversation with messages to a JSON-serializable dict."""
    result = _conversation_to_dict(conv)
    result["messages"] = [
        {
            "id": str(msg.id),
            "role": (msg.role.value if hasattr(msg.role, "value") else str(msg.role)) if msg.role else "unknown",
            "text": msg.text or "",
            "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
        }
        for msg in conv.messages
    ]
    return result


def _clamp_limit(limit: int | Any) -> int:
    """Clamp a limit value to [1, _MAX_LIMIT]."""
    try:
        return max(1, min(int(limit), _MAX_LIMIT))
    except (TypeError, ValueError):
        return 10


# ---------------------------------------------------------------------------
# Build the FastMCP server
# ---------------------------------------------------------------------------


def _build_server() -> FastMCP:
    """Construct the FastMCP server with all tools, resources, and prompts."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "polylogue",
        instructions=(
            "Polylogue is an AI conversation archive. Use the tools to search, "
            "list, and retrieve conversations from ChatGPT, Claude, Codex, and "
            "other providers. Conversations include full message history."
        ),
    )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @mcp.tool()
    def search(
        query: str,
        limit: int = 10,
        provider: str | None = None,
        since: str | None = None,
    ) -> str:
        """Search conversations by text query. Returns matching conversations with metadata.

        Args:
            query: Full-text search query
            limit: Max results (default: 10)
            provider: Filter by provider (claude, chatgpt, claude-code, etc.)
            since: Only conversations updated after this date (ISO format or natural language like 'last week')
        """
        from polylogue.lib.filters import ConversationFilter

        repo = _get_repo()
        limit = _clamp_limit(limit)

        filter_chain = ConversationFilter(repo).contains(query).limit(limit)
        if provider:
            filter_chain = filter_chain.provider(provider)
        if since:
            filter_chain = filter_chain.since(since)

        results = filter_chain.list()
        return json.dumps([_conversation_to_dict(r) for r in results], indent=2)

    @mcp.tool()
    def list_conversations(
        limit: int = 10,
        provider: str | None = None,
        since: str | None = None,
    ) -> str:
        """List recent conversations, optionally filtered by provider or date.

        Args:
            limit: Max results (default: 10)
            provider: Filter by provider (claude, chatgpt, claude-code, etc.)
            since: Only conversations updated after this date
        """
        from polylogue.lib.filters import ConversationFilter

        repo = _get_repo()
        limit = _clamp_limit(limit)

        filter_chain = ConversationFilter(repo).limit(limit)
        if provider:
            filter_chain = filter_chain.provider(provider)
        if since:
            filter_chain = filter_chain.since(since)

        conversations = filter_chain.list()
        return json.dumps([_conversation_to_dict(c) for c in conversations], indent=2)

    @mcp.tool()
    def get_conversation(id: str) -> str:
        """Get a conversation by ID (supports prefix matching). Returns full message content.

        Args:
            id: Conversation ID or unique prefix
        """
        repo = _get_repo()
        conv = repo.view(id)
        if conv is None:
            return json.dumps({"error": f"Conversation not found: {id}"})
        return json.dumps(_conversation_to_full_dict(conv), indent=2)

    @mcp.tool()
    def stats() -> str:
        """Get archive statistics: total conversations, messages, provider breakdown, database size."""
        repo = _get_repo()
        archive_stats = repo.get_archive_stats()
        data = {
            "total_conversations": archive_stats.total_conversations,
            "total_messages": archive_stats.total_messages,
            "providers": archive_stats.providers,
            "embedded_conversations": archive_stats.embedded_conversations,
            "embedded_messages": archive_stats.embedded_messages,
            "db_size_mb": round(archive_stats.db_size_bytes / 1_048_576, 1) if archive_stats.db_size_bytes else 0,
        }
        return json.dumps(data, indent=2)

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    @mcp.resource("polylogue://stats")
    def stats_resource() -> str:
        """Overall statistics about the conversation archive."""
        repo = _get_repo()
        archive_stats = repo.get_archive_stats()
        return json.dumps(
            {
                "total_conversations": archive_stats.total_conversations,
                "total_messages": archive_stats.total_messages,
                "providers": archive_stats.providers,
            },
            indent=2,
        )

    @mcp.resource("polylogue://conversations")
    def conversations_resource() -> str:
        """List of all conversations in the archive (up to 1000)."""
        from polylogue.lib.filters import ConversationFilter

        repo = _get_repo()
        convs = ConversationFilter(repo).limit(1000).list()
        return json.dumps([_conversation_to_dict(c) for c in convs], indent=2)

    @mcp.resource("polylogue://conversation/{conv_id}")
    def conversation_resource(conv_id: str) -> str:
        """Get a single conversation with full message content."""
        repo = _get_repo()
        conv = repo.get(conv_id)
        if not conv:
            return json.dumps({"error": f"Conversation not found: {conv_id}"})
        return json.dumps(_conversation_to_full_dict(conv), indent=2)

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    @mcp.prompt()
    def analyze_errors(
        provider: str | None = None,
        since: str | None = None,
    ) -> str:
        """Analyze error patterns and solutions across conversations.

        Args:
            provider: Filter by provider (claude, chatgpt, etc.)
            since: Only analyze conversations since this date
        """
        from polylogue.lib.filters import ConversationFilter

        repo = _get_repo()
        filter_chain = ConversationFilter(repo).contains("error")
        if provider:
            filter_chain = filter_chain.provider(provider)
        if since:
            filter_chain = filter_chain.since(since)

        convs = filter_chain.limit(50).list()

        error_contexts = []
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
    def summarize_week() -> str:
        """Summarize key insights from the past week's conversations."""
        from polylogue.lib.filters import ConversationFilter

        repo = _get_repo()
        week_ago = (datetime.now(tz=timezone.utc) - timedelta(days=7)).isoformat()
        convs = ConversationFilter(repo).since(week_ago).limit(100).list()

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
    def extract_code(language: str = "") -> str:
        """Extract and organize code snippets from conversations.

        Args:
            language: Programming language to focus on (optional)
        """
        from polylogue.lib.filters import ConversationFilter

        repo = _get_repo()
        convs = ConversationFilter(repo).limit(50).list()

        code_snippets: list[dict[str, str]] = []
        for conv in convs:
            for msg in conv.messages:
                if not msg.text or "```" not in msg.text:
                    continue
                blocks = msg.text.split("```")
                for i in range(1, len(blocks), 2):
                    block = blocks[i]
                    lines = block.split("\n", 1)
                    block_lang = lines[0].strip() if lines else ""
                    code = lines[1] if len(lines) > 1 else block
                    if not language or block_lang == language:
                        code_snippets.append(
                            {
                                "language": block_lang,
                                "code": code[:300],
                                "conversation": str(conv.id)[:20],
                            }
                        )
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

    return mcp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Module-level server instance (lazy-built on first access)
_server_instance: FastMCP | None = None


def _get_server() -> FastMCP:
    global _server_instance
    if _server_instance is None:
        _server_instance = _build_server()
    return _server_instance


def serve_stdio() -> None:
    """Start MCP server with stdio transport.

    This is the main entry point called from ``polylogue mcp`` CLI command.
    Uses the official MCP SDK for proper protocol compliance.
    """
    server = _get_server()
    server.run(transport="stdio")
