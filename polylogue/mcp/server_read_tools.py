"""Extended read-only MCP tool registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.mcp.payloads import (
    MCPConversationSummaryListPayload,
    MCPConversationSummaryPayload,
    MCPHealthReportPayload,
    MCPStatsByPayload,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


def register_read_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def get_conversation_summary(id: str) -> str:
        async def run() -> str:
            repo = hooks.get_repo()
            full_id = await repo.resolve_id(id) or id
            summary = await repo.get_summary(full_id)
            if summary is None:
                return hooks.error_json(f"Conversation not found: {id}")
            stats = await repo.queries.get_conversation_stats(str(full_id))
            return hooks.json_payload(
                MCPConversationSummaryPayload.from_summary(
                    summary,
                    message_count=stats["total_messages"] if stats else 0,
                )
            )

        return await hooks.async_safe_call("get_conversation_summary", run)

    @mcp.tool()
    async def get_session_tree(conversation_id: str) -> str:
        async def run() -> str:
            tree = await hooks.get_repo().get_session_tree(conversation_id)
            return hooks.json_payload(
                MCPConversationSummaryListPayload(
                    root=[
                        MCPConversationSummaryPayload.from_conversation(conv)
                        for conv in tree
                    ]
                )
            )

        return await hooks.async_safe_call("get_session_tree", run)

    @mcp.tool()
    async def get_stats_by(group_by: str = "provider") -> str:
        async def run() -> str:
            root = await hooks.get_repo().queries.get_stats_by(group_by)
            return hooks.json_payload(MCPStatsByPayload(root=root))

        return await hooks.async_safe_call("get_stats_by", run)

    @mcp.tool()
    def health_check() -> str:
        def run() -> str:
            from polylogue.health_archive import get_health

            report = get_health(hooks.get_config())
            return hooks.json_payload(
                MCPHealthReportPayload.from_report(
                    report,
                    include_counts=True,
                    include_detail=True,
                    include_cached=True,
                ),
                exclude_none=True,
            )

        return hooks.safe_call("health_check", run)


__all__ = ["register_read_tools"]
