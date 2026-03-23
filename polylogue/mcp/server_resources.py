"""Resource registration for the MCP server."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.lib.query_spec import ConversationQuerySpec
from polylogue.mcp.payloads import (
    MCPArchiveStatsPayload,
    MCPConversationDetailPayload,
    MCPConversationSummaryListPayload,
    MCPConversationSummaryPayload,
    MCPErrorPayload,
    MCPHealthReportPayload,
    MCPTagCountsPayload,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


def register_resources(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    """Register MCP resources on the given server."""

    @mcp.resource("polylogue://stats")
    async def stats_resource() -> str:
        archive_stats = await hooks.get_repo().get_archive_stats()
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
        convs = await ConversationQuerySpec().list(hooks.get_repo())
        return hooks.json_payload(
            MCPConversationSummaryListPayload(
                root=[
                    MCPConversationSummaryPayload.from_conversation(conv)
                    for conv in convs
                ]
            )
        )

    @mcp.resource("polylogue://conversation/{conv_id}")
    async def conversation_resource(conv_id: str) -> str:
        conv = await hooks.get_repo().get(conv_id)
        if not conv:
            return hooks.error_json(f"Conversation not found: {conv_id}")
        return hooks.json_payload(MCPConversationDetailPayload.from_conversation(conv))

    @mcp.resource("polylogue://tags")
    async def tags_resource() -> str:
        tags = await hooks.get_repo().list_tags()
        return hooks.json_payload(MCPTagCountsPayload(root=tags))

    @mcp.resource("polylogue://health")
    def health_resource() -> str:
        try:
            from polylogue.health_archive import get_health

            report = get_health(hooks.get_config())
            return hooks.json_payload(
                MCPHealthReportPayload.from_report(
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
