"""Stable MCP tool registration surface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from polylogue.lib.query_spec import ConversationQuerySpec
from polylogue.mcp.payloads import (
    MCPArchiveStatsPayload,
    MCPConversationDetailPayload,
    MCPConversationSummaryListPayload,
    MCPConversationSummaryPayload,
    MCPHealthReportPayload,
    MCPStatsByPayload,
)
from polylogue.mcp.server_maintenance_tools import register_maintenance_tools
from polylogue.mcp.server_mutation_tools import register_mutation_tools
from polylogue.mcp.server_product_tools import register_product_tools

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


def build_query_spec(**params: Any) -> ConversationQuerySpec:
    normalized = dict(params)
    if "has_tool_use" in normalized:
        normalized["filter_has_tool_use"] = normalized.pop("has_tool_use")
    if "has_thinking" in normalized:
        normalized["filter_has_thinking"] = normalized.pop("has_thinking")
    return ConversationQuerySpec.from_params(normalized)


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
            from polylogue.health import get_health

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


def register_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    register_query_tools(mcp, hooks)
    register_mutation_tools(mcp, hooks)
    register_read_tools(mcp, hooks)
    register_maintenance_tools(mcp, hooks)
    register_product_tools(mcp, hooks)


__all__ = ["register_query_tools", "register_read_tools", "register_tools"]
