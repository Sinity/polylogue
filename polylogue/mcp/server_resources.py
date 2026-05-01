"""Resource registration for the MCP server."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.mcp.payloads import (
    MCPArchiveStatsPayload,
    MCPConversationSummaryPayload,
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
        ops = hooks.get_archive_ops()
        summary = await ops.get_conversation_summary(conv_id)
        if not summary:
            return hooks.error_json(f"Conversation not found: {conv_id}")
        stats = await ops.get_conversation_stats(conv_id)
        return hooks.json_payload(
            MCPConversationSummaryPayload.from_summary(
                summary,
                message_count=stats["total_messages"] if stats else 0,
            )
        )

    @mcp.resource("polylogue://tags")
    async def tags_resource() -> str:
        tags = await hooks.get_tag_store().list_tags()
        return hooks.json_payload(MCPTagCountsPayload(root=tags))

    @mcp.resource("polylogue://messages/{conv_id}")
    async def messages_resource(conv_id: str) -> str:
        ops = hooks.get_archive_ops()
        summary = await ops.get_conversation_summary(conv_id)
        if summary is None:
            return hooks.error_json(f"Conversation not found: {conv_id}")
        canonical_id = str(summary.id)
        messages, total = await ops.get_messages_paginated(canonical_id, limit=20, offset=0)
        from polylogue.mcp.payloads import MCPMessagePayload, MCPMessagesListPayload

        return hooks.json_payload(
            MCPMessagesListPayload(
                conversation_id=canonical_id,
                messages=tuple(MCPMessagePayload.from_message(message) for message in messages),
                total=total,
                limit=20,
                offset=0,
            )
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
            return hooks.json_payload(
                MCPErrorPayload(error="internal MCP resource error", code="internal_error", detail=type(exc).__name__),
                exclude_none=True,
            )


__all__ = ["register_resources"]
