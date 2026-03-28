"""MCP server implementation for Polylogue.

Provides conversation search and retrieval capabilities via the Model Context Protocol.
Uses stdio transport for communication with AI assistants.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

_MAX_LIMIT = 10000

if TYPE_CHECKING:
    from polylogue.storage.repository import ConversationRepository


def serve_stdio() -> None:
    """Start MCP server with stdio transport.

    Implements the Model Context Protocol for AI assistant integration.
    Reads JSON-RPC requests from stdin and writes responses to stdout.
    """
    from polylogue.storage.backends.sqlite import create_default_backend
    from polylogue.storage.repository import ConversationRepository

    backend = create_default_backend()
    repo = ConversationRepository(backend=backend)

    # MCP protocol handler
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            request = json.loads(line)
            response = _handle_request(request, repo)
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        except json.JSONDecodeError:
            _write_error(-32700, "Parse error")
        except KeyboardInterrupt:
            break
        except Exception:
            logger.exception("MCP request handler error")
            _write_error(-32603, "Internal server error")


def _handle_request(request: dict[str, Any], repo: ConversationRepository) -> dict[str, Any]:
    """Handle a single MCP request."""
    method = request.get("method", "")
    params = request.get("params", {})
    request_id = request.get("id")

    if method == "initialize":
        return _success(
            request_id,
            {
                "protocolVersion": "0.1.0",
                "serverInfo": {"name": "polylogue", "version": "0.1.0"},
                "capabilities": {
                    "tools": {
                        "search": {"description": "Search conversations by text"},
                        "list": {"description": "List conversations"},
                        "get": {"description": "Get conversation by ID"},
                    },
                    "resources": {},
                    "prompts": {},
                },
            },
        )

    elif method == "tools/list":
        return _success(
            request_id,
            {
                "tools": [
                    {
                        "name": "search",
                        "description": "Search conversations by text query. Returns matching conversations with metadata.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Full-text search query"},
                                "limit": {"type": "integer", "description": "Max results (default: 10)", "default": 10},
                                "provider": {"type": "string", "description": "Filter by provider (claude, chatgpt, claude-code, etc.)"},
                                "since": {"type": "string", "description": "Only conversations updated after this date (ISO format or natural language like 'last week')"},
                            },
                            "required": ["query"],
                        },
                    },
                    {
                        "name": "list",
                        "description": "List recent conversations, optionally filtered by provider or date.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "limit": {"type": "integer", "description": "Max results (default: 10)", "default": 10},
                                "provider": {"type": "string", "description": "Filter by provider (claude, chatgpt, claude-code, etc.)"},
                                "since": {"type": "string", "description": "Only conversations updated after this date"},
                            },
                        },
                    },
                    {
                        "name": "get",
                        "description": "Get a conversation by ID (supports prefix matching). Returns full message content.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Conversation ID or unique prefix"},
                            },
                            "required": ["id"],
                        },
                    },
                    {
                        "name": "stats",
                        "description": "Get archive statistics: total conversations, messages, provider breakdown, database size.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                ],
            },
        )

    elif method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        if tool_name == "search":
            return _handle_search(request_id, tool_args, repo)
        elif tool_name == "list":
            return _handle_list(request_id, tool_args, repo)
        elif tool_name == "get":
            return _handle_get(request_id, tool_args, repo)
        elif tool_name == "stats":
            return _handle_stats_tool(request_id, repo)
        else:
            return _error(request_id, -32601, f"Unknown tool: {tool_name}")

    elif method == "resources/list":
        return _success(
            request_id,
            {
                "resources": [
                    {
                        "uri": "polylogue://stats",
                        "name": "Archive Statistics",
                        "mimeType": "application/json",
                        "description": "Overall statistics about the conversation archive",
                    },
                    {
                        "uri": "polylogue://conversations",
                        "name": "All Conversations",
                        "mimeType": "application/json",
                        "description": "List of all conversations in the archive",
                    },
                    {
                        "uri": "polylogue://conversation/{id}",
                        "name": "Conversation by ID",
                        "mimeType": "application/json",
                        "description": "Get a single conversation with full message content",
                    },
                ],
            },
        )

    elif method == "resources/read":
        uri = params.get("uri", "")

        # Parse URI and query parameters
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(uri)
        base_uri = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        query_params = parse_qs(parsed.query)

        if base_uri == "polylogue://stats":
            return _handle_stats_resource(request_id, repo)
        elif base_uri == "polylogue://conversations":
            return _handle_conversations_resource(request_id, repo, query_params)
        elif base_uri.startswith("polylogue://conversation/"):
            conv_id = base_uri.replace("polylogue://conversation/", "")
            return _handle_conversation_resource(request_id, conv_id, repo)
        else:
            return _error(request_id, -32602, f"Unknown resource URI: {uri}")

    elif method == "prompts/list":
        return _success(
            request_id,
            {
                "prompts": [
                    {
                        "name": "analyze-errors",
                        "description": "Analyze error patterns and solutions across conversations",
                        "arguments": [
                            {
                                "name": "provider",
                                "description": "Filter by provider (claude, chatgpt, etc.)",
                                "required": False,
                            },
                            {
                                "name": "since",
                                "description": "Only analyze conversations since this date",
                                "required": False,
                            },
                        ],
                    },
                    {
                        "name": "summarize-week",
                        "description": "Summarize key insights from the past week's conversations",
                    },
                    {
                        "name": "extract-code",
                        "description": "Extract and organize code snippets from conversations",
                        "arguments": [
                            {"name": "language", "description": "Programming language to focus on", "required": False},
                        ],
                    },
                ],
            },
        )

    elif method == "prompts/get":
        prompt_name = params.get("name", "")
        prompt_args = params.get("arguments", {})

        if prompt_name == "analyze-errors":
            return _handle_analyze_errors_prompt(request_id, prompt_args, repo)
        elif prompt_name == "summarize-week":
            return _handle_summarize_week_prompt(request_id, prompt_args, repo)
        elif prompt_name == "extract-code":
            return _handle_extract_code_prompt(request_id, prompt_args, repo)
        else:
            return _error(request_id, -32602, f"Unknown prompt: {prompt_name}")

    else:
        return _error(request_id, -32601, f"Method not found: {method}")


def _conversation_to_dict(conv: Any) -> dict[str, Any]:
    """Convert Conversation to a JSON-serializable dict."""
    return {
        "id": str(conv.id),
        "provider": conv.provider,
        "title": conv.display_title,
        "message_count": len(conv.messages),
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
    }


def _conversation_to_full_dict(conv: Any) -> dict[str, Any]:
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


def _handle_search(request_id: Any, args: dict[str, Any], repo: ConversationRepository) -> dict[str, Any]:
    """Handle search tool call with optional provider/since filters."""
    query = args.get("query", "")
    limit = args.get("limit", 10)
    provider = args.get("provider")
    since = args.get("since")

    if not query:
        return _error(request_id, -32602, "Missing required parameter: query")

    try:
        limit = max(1, min(int(limit), _MAX_LIMIT))
    except (TypeError, ValueError):
        return _error(request_id, -32602, "limit must be an integer")

    # Use filter chain for composable filtering
    from polylogue.lib.filters import ConversationFilter

    filter_chain = ConversationFilter(repo).contains(query).limit(limit)
    if provider:
        filter_chain = filter_chain.provider(provider)
    if since:
        try:
            filter_chain = filter_chain.since(since)
        except ValueError as exc:
            return _error(request_id, -32602, f"Invalid date: {exc}")

    results = filter_chain.list()
    return _success(
        request_id,
        {
            "content": [
                {"type": "text", "text": json.dumps([_conversation_to_dict(r) for r in results], indent=2)},
            ],
        },
    )


def _handle_list(request_id: Any, args: dict[str, Any], repo: ConversationRepository) -> dict[str, Any]:
    """Handle list tool call with optional filters."""
    limit = args.get("limit", 10)
    provider = args.get("provider")
    since = args.get("since")

    try:
        limit = max(1, min(int(limit), _MAX_LIMIT))
    except (TypeError, ValueError):
        return _error(request_id, -32602, "limit must be an integer")

    from polylogue.lib.filters import ConversationFilter

    filter_chain = ConversationFilter(repo).limit(limit)
    if provider:
        filter_chain = filter_chain.provider(provider)
    if since:
        try:
            filter_chain = filter_chain.since(since)
        except ValueError as exc:
            return _error(request_id, -32602, f"Invalid date: {exc}")

    conversations = filter_chain.list()
    return _success(
        request_id,
        {
            "content": [
                {"type": "text", "text": json.dumps([_conversation_to_dict(c) for c in conversations], indent=2)},
            ],
        },
    )


def _handle_get(request_id: Any, args: dict[str, Any], repo: ConversationRepository) -> dict[str, Any]:
    """Handle get tool call with ID prefix support."""
    conv_id = args.get("id", "")

    if not conv_id:
        return _error(request_id, -32602, "Missing required parameter: id")

    # Use view() for ID prefix resolution
    conv = repo.view(conv_id)
    if conv is None:
        return _error(request_id, -32602, f"Conversation not found: {conv_id}")

    return _success(
        request_id,
        {
            "content": [
                {"type": "text", "text": json.dumps(_conversation_to_full_dict(conv), indent=2)},
            ],
        },
    )


def _handle_stats_tool(request_id: Any, repo: ConversationRepository) -> dict[str, Any]:
    """Handle stats tool call."""
    stats = repo.get_archive_stats()
    data = {
        "total_conversations": stats.total_conversations,
        "total_messages": stats.total_messages,
        "providers": stats.providers,
        "embedded_conversations": stats.embedded_conversations,
        "embedded_messages": stats.embedded_messages,
        "db_size_mb": round(stats.db_size_bytes / 1_048_576, 1) if stats.db_size_bytes else 0,
    }
    return _success(
        request_id,
        {
            "content": [
                {"type": "text", "text": json.dumps(data, indent=2)},
            ],
        },
    )


def _success(request_id: Any, result: Any) -> dict[str, Any]:
    """Create a successful JSON-RPC response."""
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    """Create an error JSON-RPC response."""
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _write_error(code: int, message: str) -> None:
    """Write an error response to stdout."""
    response = {"jsonrpc": "2.0", "id": None, "error": {"code": code, "message": message}}
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def _handle_stats_resource(request_id: Any, repo: ConversationRepository) -> dict[str, Any]:
    """Handle polylogue://stats resource."""
    archive_stats = repo.get_archive_stats()

    stats = {
        "total_conversations": archive_stats.total_conversations,
        "total_messages": archive_stats.total_messages,
        "providers": archive_stats.providers,
    }

    return _success(
        request_id,
        {
            "contents": [
                {"uri": "polylogue://stats", "mimeType": "application/json", "text": json.dumps(stats, indent=2)},
            ],
        },
    )


def _handle_conversations_resource(
    request_id: Any,
    repo: ConversationRepository,
    query_params: dict[str, list[str]],
) -> dict[str, Any]:
    """Handle polylogue://conversations resource with query parameters.

    Supports query parameters:
    - provider: Filter by provider name
    - since: Filter by date (YYYY-MM-DD)
    - tag: Filter by tag
    - limit: Max results (default: 1000)
    """
    # Extract parameters (parse_qs returns lists)
    provider = query_params.get("provider", [None])[0]
    since = query_params.get("since", [None])[0]
    tag = query_params.get("tag", [None])[0]
    limit_str = query_params.get("limit", ["1000"])[0]
    try:
        limit = max(1, min(int(limit_str), _MAX_LIMIT)) if limit_str else 1000
    except (TypeError, ValueError):
        return _error(request_id, -32602, "limit must be an integer")

    # Build filter chain
    from polylogue.lib.filters import ConversationFilter

    filter_chain = ConversationFilter(repo)

    if provider:
        filter_chain = filter_chain.provider(provider)
    if since:
        try:
            filter_chain = filter_chain.since(since)
        except ValueError as exc:
            return _error(request_id, -32602, f"Invalid date: {exc}")
    if tag:
        filter_chain = filter_chain.tag(tag)

    filter_chain = filter_chain.limit(limit)

    # Execute query
    convs = filter_chain.list()
    convs_data = [_conversation_to_dict(c) for c in convs]

    return _success(
        request_id,
        {
            "contents": [
                {
                    "uri": "polylogue://conversations",
                    "mimeType": "application/json",
                    "text": json.dumps(convs_data, indent=2),
                },
            ],
        },
    )


def _handle_conversation_resource(request_id: Any, conv_id: str, repo: ConversationRepository) -> dict[str, Any]:
    """Handle polylogue://conversation/{id} resource."""
    conv = repo.get(conv_id)
    if not conv:
        return _error(request_id, -32602, f"Conversation not found: {conv_id}")

    conv_data = _conversation_to_full_dict(conv)
    uri = f"polylogue://conversation/{conv_id}"

    return _success(
        request_id,
        {
            "contents": [
                {"uri": uri, "mimeType": "application/json", "text": json.dumps(conv_data, indent=2)},
            ],
        },
    )


def _handle_analyze_errors_prompt(
    request_id: Any,
    args: dict[str, Any],
    repo: ConversationRepository,
) -> dict[str, Any]:
    """Generate a prompt for analyzing error patterns."""
    provider = args.get("provider")
    since = args.get("since")

    # Build filter to find conversations with errors
    from polylogue.lib.filters import ConversationFilter

    filter_chain = ConversationFilter(repo)
    filter_chain = filter_chain.contains("error")

    if provider:
        filter_chain = filter_chain.provider(provider)
    if since:
        try:
            filter_chain = filter_chain.since(since)
        except ValueError as exc:
            return _error(request_id, -32602, f"Invalid date: {exc}")

    convs = filter_chain.limit(50).list()

    # Build prompt with context
    error_contexts = []
    for conv in convs:
        # Extract error-related messages
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

    prompt_text = f"""Analyze error patterns from {len(convs)} conversations.

Context: {len(error_contexts)} error instances found.

Your task:
1. Identify common error patterns and root causes
2. Note which errors have known solutions in the conversations
3. Suggest preventive measures based on successful resolutions
4. Highlight any recurring pain points

Error contexts:
{json.dumps(error_contexts, indent=2)}
"""

    return _success(
        request_id,
        {
            "description": "Analyze error patterns and solutions",
            "messages": [
                {"role": "user", "content": {"type": "text", "text": prompt_text}},
            ],
        },
    )


def _handle_summarize_week_prompt(
    request_id: Any,
    args: dict[str, Any],
    repo: ConversationRepository,
) -> dict[str, Any]:
    """Generate a prompt for summarizing the past week."""
    # Get conversations from the past 7 days
    from datetime import datetime, timedelta, timezone

    week_ago = (datetime.now(tz=timezone.utc) - timedelta(days=7)).isoformat()

    from polylogue.lib.filters import ConversationFilter

    convs = ConversationFilter(repo).since(week_ago).limit(100).list()

    # Group by provider and topic
    by_provider: dict[str, int] = {}
    total_messages = 0
    for conv in convs:
        by_provider[conv.provider] = by_provider.get(conv.provider, 0) + 1
        total_messages += len(conv.messages)

    prompt_text = f"""Summarize key insights from the past week's AI conversations.

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

    return _success(
        request_id,
        {
            "description": "Summarize the past week's conversations",
            "messages": [
                {"role": "user", "content": {"type": "text", "text": prompt_text}},
            ],
        },
    )


def _handle_extract_code_prompt(
    request_id: Any,
    args: dict[str, Any],
    repo: ConversationRepository,
) -> dict[str, Any]:
    """Generate a prompt for extracting code snippets."""
    language = args.get("language", "")

    # Find conversations with code blocks
    from polylogue.lib.filters import ConversationFilter

    convs = ConversationFilter(repo).limit(50).list()

    # Extract code blocks (simplified - looks for ```language markers)
    code_snippets = []
    for conv in convs:
        for msg in conv.messages:
            if not msg.text:
                continue

            # Simple code block detection
            if "```" in msg.text:
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
    prompt_text = f"""Extract and organize code snippets from conversations{lang_filter}.

Found {len(code_snippets)} code blocks.

Your task:
1. Categorize code snippets by purpose/functionality
2. Identify reusable patterns or utilities
3. Note any incomplete or problematic code
4. Suggest organization into a knowledge base

Code snippets:
{json.dumps(code_snippets, indent=2)}
"""

    return _success(
        request_id,
        {
            "description": f"Extract code snippets{lang_filter}",
            "messages": [
                {"role": "user", "content": {"type": "text", "text": prompt_text}},
            ],
        },
    )
