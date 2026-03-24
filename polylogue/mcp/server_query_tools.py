"""Query-oriented MCP tool registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.mcp.payloads import (
    MCPArchiveStatsPayload,
    MCPConversationDetailPayload,
    MCPConversationSummaryListPayload,
    MCPConversationSummaryPayload,
)
from polylogue.mcp.query_support import build_query_spec

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


def register_query_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def search(
        query: str,
        limit: int = 10,
        retrieval_lane: str | None = None,
        provider: str | None = None,
        since: str | None = None,
        path: str | None = None,
        action: str | None = None,
        exclude_action: str | None = None,
        action_sequence: str | None = None,
        action_text: str | None = None,
        tool: str | None = None,
        exclude_tool: str | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        min_words: int | None = None,
    ) -> str:
        async def run() -> str:
            ops = hooks.get_archive_ops()
            spec = build_query_spec(
                query=query,
                retrieval_lane=retrieval_lane or "auto",
                provider=provider,
                since=since,
                path=path,
                action=action,
                exclude_action=exclude_action,
                action_sequence=action_sequence,
                action_text=action_text,
                tool=tool,
                exclude_tool=exclude_tool,
                limit=hooks.clamp_limit(limit),
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                min_messages=min_messages,
                min_words=min_words,
            )
            results = await ops.query_conversations(spec)
            return hooks.json_payload(
                MCPConversationSummaryListPayload(
                    root=[
                        MCPConversationSummaryPayload.from_conversation(result)
                        for result in results
                    ]
                )
            )

        return await hooks.async_safe_call("search", run)

    @mcp.tool()
    async def list_conversations(
        limit: int = 10,
        retrieval_lane: str | None = None,
        provider: str | None = None,
        since: str | None = None,
        tag: str | None = None,
        title: str | None = None,
        path: str | None = None,
        action: str | None = None,
        exclude_action: str | None = None,
        action_sequence: str | None = None,
        action_text: str | None = None,
        tool: str | None = None,
        exclude_tool: str | None = None,
        sort: str = "updated",
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        min_words: int | None = None,
    ) -> str:
        async def run() -> str:
            ops = hooks.get_archive_ops()
            spec = build_query_spec(
                provider=provider,
                retrieval_lane=retrieval_lane or "auto",
                tag=tag,
                title=title,
                since=since,
                path=path,
                action=action,
                exclude_action=exclude_action,
                action_sequence=action_sequence,
                action_text=action_text,
                tool=tool,
                exclude_tool=exclude_tool,
                sort=sort,
                limit=hooks.clamp_limit(limit),
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                min_messages=min_messages,
                min_words=min_words,
            )
            conversations = await ops.query_conversations(spec)
            return hooks.json_payload(
                MCPConversationSummaryListPayload(
                    root=[
                        MCPConversationSummaryPayload.from_conversation(conv)
                        for conv in conversations
                    ]
                )
            )

        return await hooks.async_safe_call("list_conversations", run)

    @mcp.tool()
    async def get_conversation(id: str) -> str:
        async def run() -> str:
            conv = await hooks.get_archive_ops().get_conversation(id)
            if conv is None:
                return hooks.error_json(f"Conversation not found: {id}")
            return hooks.json_payload(MCPConversationDetailPayload.from_conversation(conv))

        return await hooks.async_safe_call("get_conversation", run)

    @mcp.tool()
    async def stats() -> str:
        async def run() -> str:
            archive_stats = await hooks.get_archive_ops().storage_stats()
            return hooks.json_payload(
                MCPArchiveStatsPayload.from_archive_stats(
                    archive_stats,
                    include_embedded=True,
                    include_db_size=True,
                )
            )

        return await hooks.async_safe_call("stats", run)


__all__ = ["register_query_tools"]
