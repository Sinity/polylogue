"""Resource registration for the MCP server."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.mcp.payloads import (
    MCPArchiveStatsPayload,
    MCPConversationSummaryPayload,
    MCPErrorPayload,
    MCPReadinessReportPayload,
    MCPTagCountsPayload,
    conversation_query_result_payload,
)
from polylogue.mcp.query_contracts import MCPConversationQueryRequest

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


def register_resources(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    """Register MCP resources on the given server."""

    @mcp.resource("polylogue://stats")
    async def stats_resource() -> str:
        try:
            archive_stats = await hooks.get_archive_ops().storage_stats()
        except Exception as exc:
            return hooks.error_json(
                f"Failed to retrieve archive stats: {exc}",
                code="internal_error",
                detail=type(exc).__name__,
            )
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
        try:
            spec = MCPConversationQueryRequest().build_spec(hooks.clamp_limit)
            convs = await spec.list(hooks.get_query_store())
            total = await spec.count(hooks.get_query_store())
        except Exception as exc:
            return hooks.error_json(
                f"Failed to list conversations: {exc}",
                code="internal_error",
                detail=type(exc).__name__,
            )
        return hooks.json_payload(
            conversation_query_result_payload(convs, total=total, limit=spec.limit or 0, offset=spec.offset)
        )

    @mcp.resource("polylogue://conversation/{conv_id}")
    async def conversation_resource(conv_id: str) -> str:
        ops = hooks.get_archive_ops()
        summary = await ops.get_conversation_summary(conv_id)
        if not summary:
            return hooks.error_json(f"Conversation not found: {conv_id}", code="not_found")
        stats = await ops.get_conversation_stats(conv_id)
        return hooks.json_payload(
            MCPConversationSummaryPayload.from_summary(
                summary,
                message_count=stats["total_messages"] if stats else 0,
            )
        )

    @mcp.resource("polylogue://tags")
    async def tags_resource() -> str:
        try:
            tags = await hooks.get_tag_store().list_tags()
        except Exception as exc:
            return hooks.error_json(
                f"Failed to list tags: {exc}",
                code="internal_error",
                detail=type(exc).__name__,
            )
        return hooks.json_payload(MCPTagCountsPayload(root=tags))

    @mcp.resource("polylogue://messages/{conv_id}")
    async def messages_resource(conv_id: str) -> str:
        ops = hooks.get_archive_ops()
        summary = await ops.get_conversation_summary(conv_id)
        if summary is None:
            return hooks.error_json(f"Conversation not found: {conv_id}", code="not_found")
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
        try:
            tree = await hooks.get_archive_ops().get_session_tree(conv_id)
        except Exception as exc:
            return hooks.error_json(
                f"Failed to get session tree for {conv_id}: {exc}",
                code="internal_error",
                detail=type(exc).__name__,
            )
        return hooks.json_payload(conversation_query_result_payload(tree, total=len(tree), limit=len(tree), offset=0))

    @mcp.resource("polylogue://provider/{name}/recent")
    async def provider_recent_resource(name: str) -> str:
        try:
            spec = MCPConversationQueryRequest(provider=name, sort="date", limit=10).build_spec(hooks.clamp_limit)
            convs = await spec.list(hooks.get_query_store())
            total = await spec.count(hooks.get_query_store())
        except Exception as exc:
            return hooks.error_json(
                f"Failed to list recent conversations for provider {name}: {exc}",
                code="internal_error",
                detail=type(exc).__name__,
            )
        return hooks.json_payload(
            conversation_query_result_payload(convs, total=total, limit=spec.limit or 0, offset=spec.offset)
        )

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
