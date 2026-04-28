"""Resource registration for the MCP server."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.mcp.payloads import (
    MCPArchiveStatsPayload,
    MCPConversationDetailPayload,
    MCPErrorPayload,
    MCPReadinessReportPayload,
    MCPTagCountsPayload,
    conversation_summary_list_payload,
)
from polylogue.mcp.query_contracts import MCPConversationQueryRequest

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


def register_resources(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    """Register MCP resources on the given server."""

    @mcp.resource("polylogue://stats")
    async def stats_resource() -> str:
        archive_stats = await hooks.get_archive_ops().storage_stats()
        return hooks.json_payload(
            MCPArchiveStatsPayload.from_archive_stats(
                archive_stats,
                include_embedded=False,
                include_db_size=False,
            ),
            exclude_none=True,
        )

    @mcp.resource("polylogue://conversations")
    async def conversations_resource() -> str:
        convs = await MCPConversationQueryRequest().build_spec(hooks.clamp_limit).list(hooks.get_query_store())
        return hooks.json_payload(conversation_summary_list_payload(convs))

    @mcp.resource("polylogue://conversation/{conv_id}")
    async def conversation_resource(conv_id: str) -> str:
        conv = await hooks.get_archive_ops().get_conversation(conv_id)
        if not conv:
            return hooks.error_json(f"Conversation not found: {conv_id}")
        return hooks.json_payload(MCPConversationDetailPayload.from_conversation(conv))

    @mcp.resource("polylogue://tags")
    async def tags_resource() -> str:
        tags = await hooks.get_tag_store().list_tags()
        return hooks.json_payload(MCPTagCountsPayload(root=tags))

    @mcp.resource("polylogue://messages/{conv_id}")
    async def messages_resource(conv_id: str) -> str:
        ops = hooks.get_archive_ops()
        try:
            messages, total = await ops.get_messages_paginated(conv_id, limit=20, offset=0)
        except Exception:
            return hooks.error_json(f"Conversation not found: {conv_id}")
        import json as _json

        return _json.dumps(
            {
                "conversation_id": conv_id,
                "messages": [
                    {
                        "id": str(getattr(m, "id", "")),
                        "role": str(getattr(m, "role", "")),
                        "text": (getattr(m, "text", None) or "")[:200],
                    }
                    for m in messages
                ],
                "total": total,
            },
            indent=2,
        )

    @mcp.resource("polylogue://session-tree/{conv_id}")
    async def session_tree_resource(conv_id: str) -> str:
        tree = await hooks.get_archive_ops().get_session_tree(conv_id)
        return hooks.json_payload(conversation_summary_list_payload(tree))

    @mcp.resource("polylogue://provider/{name}/recent")
    async def provider_recent_resource(name: str) -> str:
        spec = MCPConversationQueryRequest(provider=name, sort="date", limit=10).build_spec(hooks.clamp_limit)
        convs = await spec.list(hooks.get_query_store())
        return hooks.json_payload(conversation_summary_list_payload(convs))

    @mcp.resource("polylogue://readiness")
    def readiness_resource() -> str:
        try:
            from polylogue.readiness import get_readiness

            report = get_readiness(hooks.get_config())
            return hooks.json_payload(
                MCPReadinessReportPayload.from_report(
                    report,
                    include_counts=False,
                    include_detail=False,
                    include_cached=False,
                ),
                exclude_none=True,
            )
        except Exception as exc:
            return hooks.json_payload(MCPErrorPayload(error=str(exc)), exclude_none=True)


__all__ = ["register_resources"]
