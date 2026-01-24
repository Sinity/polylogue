"""MCP server implementation for Polylogue.

Provides conversation search and retrieval capabilities via the Model Context Protocol.
Uses stdio transport for communication with AI assistants.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polylogue.lib.repository import ConversationRepository


def serve_stdio() -> None:
    """Start MCP server with stdio transport.

    Implements the Model Context Protocol for AI assistant integration.
    Reads JSON-RPC requests from stdin and writes responses to stdout.
    """
    from polylogue.lib.repository import ConversationRepository
    from polylogue.storage.backends.sqlite import create_default_backend

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
        except Exception as exc:
            _write_error(-32603, f"Internal error: {exc}")


def _handle_request(request: dict[str, Any], repo: ConversationRepository) -> dict[str, Any]:
    """Handle a single MCP request."""
    method = request.get("method", "")
    params = request.get("params", {})
    request_id = request.get("id")

    if method == "initialize":
        return _success(request_id, {
            "protocolVersion": "0.1.0",
            "serverInfo": {"name": "polylogue", "version": "0.1.0"},
            "capabilities": {
                "tools": {
                    "search": {"description": "Search conversations by text"},
                    "list": {"description": "List conversations"},
                    "get": {"description": "Get conversation by ID"},
                },
                "resources": {},
            },
        })

    elif method == "tools/list":
        return _success(request_id, {
            "tools": [
                {
                    "name": "search",
                    "description": "Search conversations by text query",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "limit": {"type": "integer", "description": "Max results", "default": 10},
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "list",
                    "description": "List recent conversations",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "description": "Max results", "default": 10},
                            "provider": {"type": "string", "description": "Filter by provider"},
                        },
                    },
                },
                {
                    "name": "get",
                    "description": "Get a conversation by ID",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Conversation ID"},
                        },
                        "required": ["id"],
                    },
                },
            ],
        })

    elif method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        if tool_name == "search":
            return _handle_search(request_id, tool_args, repo)
        elif tool_name == "list":
            return _handle_list(request_id, tool_args, repo)
        elif tool_name == "get":
            return _handle_get(request_id, tool_args, repo)
        else:
            return _error(request_id, -32601, f"Unknown tool: {tool_name}")

    elif method == "resources/list":
        return _success(request_id, {
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
            ],
        })

    elif method == "resources/read":
        uri = params.get("uri", "")

        if uri == "polylogue://stats":
            return _handle_stats_resource(request_id, repo)
        elif uri == "polylogue://conversations":
            return _handle_conversations_resource(request_id, repo)
        elif uri.startswith("polylogue://conversation/"):
            conv_id = uri.replace("polylogue://conversation/", "")
            return _handle_conversation_resource(request_id, conv_id, repo)
        else:
            return _error(request_id, -32602, f"Unknown resource URI: {uri}")

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
            "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
            "text": msg.text[:1000] + "..." if len(msg.text) > 1000 else msg.text,
            "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
        }
        for msg in conv.messages
    ]
    return result


def _handle_search(request_id: Any, args: dict[str, Any], repo: ConversationRepository) -> dict[str, Any]:
    """Handle search tool call."""
    query = args.get("query", "")
    limit = args.get("limit", 10)

    if not query:
        return _error(request_id, -32602, "Missing required parameter: query")

    results = repo.search(query, limit=limit)
    return _success(request_id, {
        "content": [
            {"type": "text", "text": json.dumps([_conversation_to_dict(r) for r in results], indent=2)},
        ],
    })


def _handle_list(request_id: Any, args: dict[str, Any], repo: ConversationRepository) -> dict[str, Any]:
    """Handle list tool call."""
    limit = args.get("limit", 10)
    provider = args.get("provider")

    conversations = repo.list(limit=limit, provider=provider)
    return _success(request_id, {
        "content": [
            {"type": "text", "text": json.dumps([_conversation_to_dict(c) for c in conversations], indent=2)},
        ],
    })


def _handle_get(request_id: Any, args: dict[str, Any], repo: ConversationRepository) -> dict[str, Any]:
    """Handle get tool call."""
    conv_id = args.get("id", "")

    if not conv_id:
        return _error(request_id, -32602, "Missing required parameter: id")

    conv = repo.get(conv_id)
    if conv is None:
        return _error(request_id, -32602, f"Conversation not found: {conv_id}")

    return _success(request_id, {
        "content": [
            {"type": "text", "text": json.dumps(_conversation_to_full_dict(conv), indent=2)},
        ],
    })


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
    convs = repo.list(limit=10000)

    providers_count: dict[str, int] = {}
    for conv in convs:
        providers_count[conv.provider] = providers_count.get(conv.provider, 0) + 1

    stats = {
        "total_conversations": len(convs),
        "total_messages": sum(len(c.messages) for c in convs),
        "providers": providers_count,
    }

    return _success(request_id, {
        "contents": [
            {"uri": "polylogue://stats", "mimeType": "application/json", "text": json.dumps(stats, indent=2)},
        ],
    })


def _handle_conversations_resource(request_id: Any, repo: ConversationRepository) -> dict[str, Any]:
    """Handle polylogue://conversations resource."""
    convs = repo.list(limit=1000)
    convs_data = [_conversation_to_dict(c) for c in convs]

    return _success(request_id, {
        "contents": [
            {"uri": "polylogue://conversations", "mimeType": "application/json", "text": json.dumps(convs_data, indent=2)},
        ],
    })


def _handle_conversation_resource(request_id: Any, conv_id: str, repo: ConversationRepository) -> dict[str, Any]:
    """Handle polylogue://conversation/{id} resource."""
    conv = repo.get(conv_id)
    if not conv:
        return _error(request_id, -32602, f"Conversation not found: {conv_id}")

    conv_data = _conversation_to_full_dict(conv)
    uri = f"polylogue://conversation/{conv_id}"

    return _success(request_id, {
        "contents": [
            {"uri": uri, "mimeType": "application/json", "text": json.dumps(conv_data, indent=2)},
        ],
    })
