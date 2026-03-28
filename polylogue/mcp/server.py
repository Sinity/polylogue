"""MCP server implementation for Polylogue using the official MCP SDK.

Provides conversation search, retrieval, tagging, metadata management, indexing,
and export capabilities via the Model Context Protocol.
Uses stdio transport for communication with AI assistants (Claude Desktop, Cursor, etc.).

Tools are organized into tiers:
  - Tier 1: Read-only queries (search, list, get, stats)
  - Tier 2: Mutation (tags, metadata, delete)
  - Tier 3: Enhanced reads (summaries, session trees, stats-by)
  - Tier 4: Pipeline triggers (rebuild_index, render)
  - Tier 5: Export (format conversion)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from polylogue.lib.log import get_logger
from polylogue.services import get_repository as _get_repo
from polylogue.services import get_service_config as _get_config

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.lib.models import Conversation

logger = get_logger(__name__)

_MAX_LIMIT = 10000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_fenced_code(text: str, language: str = "") -> list[dict[str, str]]:
    """Extract fenced code blocks from markdown text.

    Parses triple-backtick fenced blocks, extracting the language tag and code.
    Optionally filters to a specific language.

    Returns:
        List of dicts with 'language' and 'code' keys.
    """
    if "```" not in text:
        return []
    blocks = text.split("```")
    results = []
    for i in range(1, len(blocks), 2):
        block = blocks[i]
        lines = block.split("\n", 1)
        block_lang = lines[0].strip() if lines else ""
        code = lines[1] if len(lines) > 1 else block
        if not language or block_lang == language:
            results.append({"language": block_lang, "code": code[:300]})
    return results


# ---------------------------------------------------------------------------
# Serialization helpers
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


def _safe_call(fn_name: str, fn: Any) -> str:
    """Call fn() and return its result, or a JSON error dict on exception.

    Tracebacks are logged server-side at ERROR level but never sent to
    MCP clients, preventing leakage of internal paths and code structure.
    """
    try:
        return fn()
    except Exception as exc:
        logger.exception("MCP tool %s failed", fn_name)
        return json.dumps({
            "error": str(exc),
            "tool": fn_name,
        })


async def _async_safe_call(fn_name: str, fn: Any) -> str:
    """Async version of _safe_call for async tool handlers."""
    try:
        return await fn()
    except Exception as exc:
        logger.exception("MCP tool %s failed", fn_name)
        return json.dumps({
            "error": str(exc),
            "tool": fn_name,
        })


def _error_json(message: str, **extra: Any) -> str:
    """Return a JSON-encoded error dict."""
    return json.dumps({"error": message, **extra})


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
            "other providers. Conversations include full message history. "
            "You can also manage tags, metadata, trigger indexing, and export conversations."
        ),
    )

    # ==================================================================
    # Tier 1: Read-only query tools (existing)
    # ==================================================================

    @mcp.tool()
    async def search(
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
        async def _run() -> str:
            from polylogue.lib.filters import ConversationFilter

            repo = _get_repo()
            clamped = _clamp_limit(limit)

            filter_chain = ConversationFilter(repo).contains(query).limit(clamped)
            if provider:
                filter_chain = filter_chain.provider(provider)
            if since:
                filter_chain = filter_chain.since(since)

            results = await filter_chain.list()
            return json.dumps([_conversation_to_dict(r) for r in results], indent=2)
        return await _async_safe_call("search", _run)

    @mcp.tool()
    async def list_conversations(
        limit: int = 10,
        provider: str | None = None,
        since: str | None = None,
        tag: str | None = None,
        title: str | None = None,
        sort: str = "updated",
    ) -> str:
        """List recent conversations, optionally filtered by provider, date, tag, or title.

        Args:
            limit: Max results (default: 10)
            provider: Filter by provider (claude, chatgpt, claude-code, etc.)
            since: Only conversations updated after this date
            tag: Filter by tag
            title: Filter by title substring
            sort: Sort order ('updated' or 'created', default: 'updated')
        """
        async def _run() -> str:
            from polylogue.lib.filters import ConversationFilter

            repo = _get_repo()
            clamped = _clamp_limit(limit)

            filter_chain = ConversationFilter(repo).limit(clamped)
            if provider:
                filter_chain = filter_chain.provider(provider)
            if since:
                filter_chain = filter_chain.since(since)
            if tag:
                filter_chain = filter_chain.tag(tag)
            if title:
                filter_chain = filter_chain.title(title)

            conversations = await filter_chain.list()
            return json.dumps([_conversation_to_dict(c) for c in conversations], indent=2)
        return await _async_safe_call("list_conversations", _run)

    @mcp.tool()
    def get_conversation(id: str) -> str:
        """Get a conversation by ID (supports prefix matching). Returns full message content.

        Args:
            id: Conversation ID or unique prefix
        """
        def _run() -> str:
            repo = _get_repo()
            conv = repo.view(id)
            if conv is None:
                return _error_json(f"Conversation not found: {id}")
            return json.dumps(_conversation_to_full_dict(conv), indent=2)
        return _safe_call("get_conversation", _run)

    @mcp.tool()
    def stats() -> str:
        """Get archive statistics: total conversations, messages, provider breakdown, database size."""
        def _run() -> str:
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
        return _safe_call("stats", _run)

    # ==================================================================
    # Tier 2: Mutation tools (Phase 3A — wire existing repo methods)
    # ==================================================================

    @mcp.tool()
    def add_tag(conversation_id: str, tag: str) -> str:
        """Add a tag to a conversation. Idempotent — adding an existing tag is a no-op.

        Args:
            conversation_id: Conversation ID
            tag: Tag to add
        """
        def _run() -> str:
            repo = _get_repo()
            repo.add_tag(conversation_id, tag)
            return json.dumps({"status": "ok", "conversation_id": conversation_id, "tag": tag})
        return _safe_call("add_tag", _run)

    @mcp.tool()
    def remove_tag(conversation_id: str, tag: str) -> str:
        """Remove a tag from a conversation. Idempotent — removing a missing tag is a no-op.

        Args:
            conversation_id: Conversation ID
            tag: Tag to remove
        """
        def _run() -> str:
            repo = _get_repo()
            repo.remove_tag(conversation_id, tag)
            return json.dumps({"status": "ok", "conversation_id": conversation_id, "tag": tag})
        return _safe_call("remove_tag", _run)

    @mcp.tool()
    def list_tags(provider: str | None = None) -> str:
        """List all tags with their counts, optionally filtered by provider.

        Args:
            provider: Filter tags by provider (optional)
        """
        def _run() -> str:
            repo = _get_repo()
            tags = repo.list_tags(provider=provider)
            return json.dumps(tags, indent=2)
        return _safe_call("list_tags", _run)

    @mcp.tool()
    def get_metadata(conversation_id: str) -> str:
        """Get all metadata key-value pairs for a conversation.

        Args:
            conversation_id: Conversation ID
        """
        def _run() -> str:
            repo = _get_repo()
            metadata = repo.get_metadata(conversation_id)
            return json.dumps(metadata, indent=2, default=str)
        return _safe_call("get_metadata", _run)

    @mcp.tool()
    def set_metadata(conversation_id: str, key: str, value: str) -> str:
        """Set a metadata key-value pair on a conversation. JSON-serializable values accepted.

        Args:
            conversation_id: Conversation ID
            key: Metadata key
            value: Metadata value (JSON string for complex values)
        """
        def _run() -> str:
            repo = _get_repo()
            # Try to parse as JSON for complex values, fall back to string
            try:
                parsed_value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                parsed_value = value
            repo.update_metadata(conversation_id, key, parsed_value)
            return json.dumps({"status": "ok", "conversation_id": conversation_id, "key": key})
        return _safe_call("set_metadata", _run)

    @mcp.tool()
    def delete_metadata(conversation_id: str, key: str) -> str:
        """Delete a metadata key from a conversation.

        Args:
            conversation_id: Conversation ID
            key: Metadata key to delete
        """
        def _run() -> str:
            repo = _get_repo()
            repo.delete_metadata(conversation_id, key)
            return json.dumps({"status": "ok", "conversation_id": conversation_id, "key": key})
        return _safe_call("delete_metadata", _run)

    @mcp.tool()
    def delete_conversation(conversation_id: str, confirm: bool = False) -> str:
        """Delete a conversation permanently. Requires confirm=true as a safety guard.

        Args:
            conversation_id: Conversation ID to delete
            confirm: Must be true to actually delete (safety guard)
        """
        def _run() -> str:
            if not confirm:
                return _error_json(
                    "Safety guard: set confirm=true to delete",
                    conversation_id=conversation_id,
                )
            repo = _get_repo()
            deleted = repo.delete_conversation(conversation_id)
            return json.dumps({
                "status": "deleted" if deleted else "not_found",
                "conversation_id": conversation_id,
            })
        return _safe_call("delete_conversation", _run)

    # ==================================================================
    # Tier 3: Enhanced read tools (Phase 3B)
    # ==================================================================

    @mcp.tool()
    def get_conversation_summary(id: str) -> str:
        """Get a lightweight conversation summary without messages. Faster than get_conversation.

        Args:
            id: Conversation ID or unique prefix
        """
        def _run() -> str:
            repo = _get_repo()
            full_id = repo.resolve_id(id) or id
            summary = repo.get_summary(full_id)
            if summary is None:
                return _error_json(f"Conversation not found: {id}")
            return json.dumps({
                "id": str(summary.id),
                "provider": summary.provider,
                "title": summary.title,
                "message_count": summary.message_count,
                "created_at": summary.created_at.isoformat() if summary.created_at else None,
                "updated_at": summary.updated_at.isoformat() if summary.updated_at else None,
            }, indent=2)
        return _safe_call("get_conversation_summary", _run)

    @mcp.tool()
    def get_session_tree(conversation_id: str) -> str:
        """Get the full session tree for a conversation (parent, siblings, children).

        Args:
            conversation_id: Any conversation ID in the session tree
        """
        def _run() -> str:
            repo = _get_repo()
            tree = repo.get_session_tree(conversation_id)
            return json.dumps([_conversation_to_dict(c) for c in tree], indent=2)
        return _safe_call("get_session_tree", _run)

    @mcp.tool()
    def get_stats_by(group_by: str = "provider") -> str:
        """Get conversation counts grouped by provider, month, or year.

        Args:
            group_by: Grouping dimension ('provider', 'month', or 'year')
        """
        def _run() -> str:
            repo = _get_repo()
            conn = repo.backend._get_connection()

            if group_by == "month":
                rows = conn.execute("""
                    SELECT strftime('%Y-%m', updated_at) as period, COUNT(*) as count
                    FROM conversations
                    WHERE updated_at IS NOT NULL
                    GROUP BY period ORDER BY period DESC
                """).fetchall()
            elif group_by == "year":
                rows = conn.execute("""
                    SELECT strftime('%Y', updated_at) as period, COUNT(*) as count
                    FROM conversations
                    WHERE updated_at IS NOT NULL
                    GROUP BY period ORDER BY period DESC
                """).fetchall()
            else:  # provider
                rows = conn.execute("""
                    SELECT provider_name as period, COUNT(*) as count
                    FROM conversations
                    GROUP BY provider_name ORDER BY count DESC
                """).fetchall()

            return json.dumps({row[0]: row[1] for row in rows}, indent=2)
        return _safe_call("get_stats_by", _run)

    @mcp.tool()
    def health_check() -> str:
        """Run a health check on the Polylogue archive. Returns status of database, indexes, and configuration."""
        def _run() -> str:
            from polylogue.health import get_health

            config = _get_config()
            report = get_health(config)
            return json.dumps({
                "checks": [
                    {
                        "name": c.name,
                        "status": c.status.value if hasattr(c.status, "value") else str(c.status),
                        "count": c.count,
                        "detail": c.detail,
                    }
                    for c in report.checks
                ],
                "summary": report.summary,
                "cached": getattr(report, "cached", False),
            }, indent=2)
        return _safe_call("health_check", _run)

    # ==================================================================
    # Tier 4: Pipeline trigger tools (Phase 3C)
    # ==================================================================

    @mcp.tool()
    def rebuild_index() -> str:
        """Rebuild the full-text search index from scratch. Use after bulk imports or if search seems stale."""
        def _run() -> str:
            from polylogue.pipeline.services.indexing import IndexService

            config = _get_config()
            conn = _get_repo().backend._get_connection()
            service = IndexService(config=config, conn=conn)
            success = service.rebuild_index()
            status_info = service.get_index_status()
            return json.dumps({
                "status": "ok" if success else "failed",
                "index_exists": status_info.get("exists", False),
                "indexed_messages": status_info.get("count", 0),
            }, indent=2)
        return _safe_call("rebuild_index", _run)

    @mcp.tool()
    def update_index(conversation_ids: list[str]) -> str:
        """Update the search index for specific conversations (incremental).

        Args:
            conversation_ids: List of conversation IDs to re-index
        """
        def _run() -> str:
            from polylogue.pipeline.services.indexing import IndexService

            config = _get_config()
            conn = _get_repo().backend._get_connection()
            service = IndexService(config=config, conn=conn)
            success = service.update_index(conversation_ids)
            return json.dumps({
                "status": "ok" if success else "failed",
                "conversation_count": len(conversation_ids),
            }, indent=2)
        return _safe_call("update_index", _run)

    # ==================================================================
    # Tier 5: Export tool (Phase 3D)
    # ==================================================================

    @mcp.tool()
    def export_conversation(
        id: str,
        format: str = "markdown",
    ) -> str:
        """Export a conversation in various formats.

        Args:
            id: Conversation ID or unique prefix
            format: Output format ('markdown', 'json', 'html', 'yaml', 'plaintext', 'csv', 'obsidian', 'org')
        """
        def _run() -> str:
            from polylogue.lib.formatting import format_conversation

            repo = _get_repo()
            conv = repo.view(id)
            if conv is None:
                return _error_json(f"Conversation not found: {id}")

            valid_formats = {"markdown", "json", "html", "yaml", "plaintext", "csv", "obsidian", "org"}
            fmt = format if format in valid_formats else "markdown"
            return format_conversation(conv, fmt, None)
        return _safe_call("export_conversation", _run)

    # ==================================================================
    # Resources
    # ==================================================================

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
    async def conversations_resource() -> str:
        """List of all conversations in the archive (up to 1000)."""
        from polylogue.lib.filters import ConversationFilter

        repo = _get_repo()
        convs = await ConversationFilter(repo).limit(1000).list()
        return json.dumps([_conversation_to_dict(c) for c in convs], indent=2)

    @mcp.resource("polylogue://conversation/{conv_id}")
    def conversation_resource(conv_id: str) -> str:
        """Get a single conversation with full message content."""
        repo = _get_repo()
        conv = repo.get(conv_id)
        if not conv:
            return json.dumps({"error": f"Conversation not found: {conv_id}"})
        return json.dumps(_conversation_to_full_dict(conv), indent=2)

    @mcp.resource("polylogue://tags")
    def tags_resource() -> str:
        """All tags with their counts."""
        repo = _get_repo()
        tags = repo.list_tags()
        return json.dumps(tags, indent=2)

    @mcp.resource("polylogue://health")
    def health_resource() -> str:
        """Health check status of the archive."""
        try:
            from polylogue.health import get_health

            config = _get_config()
            report = get_health(config)
            return json.dumps({
                "checks": [
                    {"name": c.name, "status": c.status.value if hasattr(c.status, "value") else str(c.status)}
                    for c in report.checks
                ],
                "summary": report.summary,
            }, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    # ==================================================================
    # Prompts
    # ==================================================================

    @mcp.prompt()
    async def analyze_errors(
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

        convs = await filter_chain.limit(50).list()

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
    async def summarize_week() -> str:
        """Summarize key insights from the past week's conversations."""
        from polylogue.lib.filters import ConversationFilter

        repo = _get_repo()
        week_ago = (datetime.now(tz=timezone.utc) - timedelta(days=7)).isoformat()
        convs = await ConversationFilter(repo).since(week_ago).limit(100).list()

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
    async def extract_code(language: str = "") -> str:
        """Extract and organize code snippets from conversations.

        Args:
            language: Programming language to focus on (optional)
        """
        from polylogue.lib.filters import ConversationFilter

        repo = _get_repo()
        convs = await ConversationFilter(repo).limit(50).list()

        code_snippets: list[dict[str, str]] = []
        for conv in convs:
            for msg in conv.messages:
                if not msg.text:
                    continue
                for block in _extract_fenced_code(msg.text, language):
                    block["conversation"] = str(conv.id)[:20]
                    code_snippets.append(block)
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
    def compare_conversations(id1: str, id2: str) -> str:
        """Compare two conversations side by side, analyzing differences in approach and outcomes.

        Args:
            id1: First conversation ID
            id2: Second conversation ID
        """
        repo = _get_repo()
        conv1 = repo.view(id1)
        conv2 = repo.view(id2)

        def _summarize(conv: Any) -> dict[str, Any]:
            if conv is None:
                return {"error": "not found"}
            return {
                "id": str(conv.id),
                "provider": conv.provider,
                "title": conv.display_title,
                "message_count": len(conv.messages),
                "messages": [
                    {"role": (m.role.value if hasattr(m.role, "value") else str(m.role)) if m.role else "unknown",
                     "text": (m.text or "")[:200]}
                    for m in conv.messages[:10]  # First 10 messages
                ],
            }

        return f"""Compare these two conversations and analyze:

1. What topics/problems are discussed in each?
2. How do the approaches differ?
3. Which conversation had better outcomes?
4. What can be learned from the differences?

Conversation 1:
{json.dumps(_summarize(conv1), indent=2)}

Conversation 2:
{json.dumps(_summarize(conv2), indent=2)}
"""

    @mcp.prompt()
    async def extract_patterns(
        provider: str | None = None,
        limit: int = 30,
    ) -> str:
        """Find recurring patterns, topics, and themes across conversations.

        Args:
            provider: Filter by provider (optional)
            limit: Number of conversations to analyze (default: 30)
        """
        from polylogue.lib.filters import ConversationFilter

        repo = _get_repo()
        clamped = _clamp_limit(limit)
        filter_chain = ConversationFilter(repo).limit(clamped)
        if provider:
            filter_chain = filter_chain.provider(provider)

        convs = await filter_chain.list()

        summaries = []
        for conv in convs:
            first_msgs = [m.text[:150] for m in conv.messages[:3] if m.text]
            summaries.append({
                "id": str(conv.id)[:20],
                "provider": conv.provider,
                "title": conv.display_title,
                "opening": first_msgs,
            })

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
