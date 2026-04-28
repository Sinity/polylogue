"""Stable MCP tool registration surface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from polylogue.mcp.payloads import (
    MCPArchiveStatsPayload,
    MCPConversationDetailPayload,
    MCPConversationSummaryPayload,
    MCPMessagePayload,
    MCPMessagesListPayload,
    MCPRawRecordPayload,
    MCPRawRecordsListPayload,
    MCPReadinessReportPayload,
    MCPStatsByPayload,
    conversation_neighbor_candidate_list_payload,
    conversation_query_result_payload,
    conversation_search_result_payload,
    conversation_summary_list_payload,
)
from polylogue.mcp.query_contracts import MCPContentProjectionRequest, MCPConversationQueryRequest
from polylogue.mcp.server_maintenance_tools import register_maintenance_tools
from polylogue.mcp.server_mutation_tools import register_mutation_tools
from polylogue.mcp.server_product_tools import register_product_tools

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
        referenced_path: str | None = None,
        cwd_prefix: str | None = None,
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
        since_session: str | None = None,
        message_type: str | None = None,
        offset: int = 0,
    ) -> str:
        async def run() -> str:
            ops = hooks.get_archive_ops()
            spec = MCPConversationQueryRequest(
                query=query,
                retrieval_lane=retrieval_lane,
                provider=provider,
                since=since,
                referenced_path=referenced_path,
                cwd_prefix=cwd_prefix,
                action=action,
                exclude_action=exclude_action,
                action_sequence=action_sequence,
                action_text=action_text,
                tool=tool,
                exclude_tool=exclude_tool,
                limit=limit,
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                min_messages=min_messages,
                min_words=min_words,
                since_session=since_session,
                message_type=message_type,
                offset=offset,
            ).build_spec(hooks.clamp_limit)
            results = await ops.search_conversation_hits(spec)
            diagnostics = await ops.diagnose_query_miss(spec) if not results else None
            return hooks.json_payload(conversation_search_result_payload(results, diagnostics=diagnostics))

        return await hooks.async_safe_call("search", run)

    @mcp.tool()
    async def list_conversations(
        limit: int = 10,
        retrieval_lane: str | None = None,
        provider: str | None = None,
        since: str | None = None,
        tag: str | None = None,
        title: str | None = None,
        referenced_path: str | None = None,
        cwd_prefix: str | None = None,
        action: str | None = None,
        exclude_action: str | None = None,
        action_sequence: str | None = None,
        action_text: str | None = None,
        tool: str | None = None,
        exclude_tool: str | None = None,
        sort: str | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        min_words: int | None = None,
        since_session: str | None = None,
        message_type: str | None = None,
        offset: int = 0,
    ) -> str:
        async def run() -> str:
            ops = hooks.get_archive_ops()
            spec = MCPConversationQueryRequest(
                provider=provider,
                retrieval_lane=retrieval_lane,
                tag=tag,
                title=title,
                since=since,
                referenced_path=referenced_path,
                cwd_prefix=cwd_prefix,
                action=action,
                exclude_action=exclude_action,
                action_sequence=action_sequence,
                action_text=action_text,
                tool=tool,
                exclude_tool=exclude_tool,
                sort=sort,
                limit=limit,
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                min_messages=min_messages,
                min_words=min_words,
                since_session=since_session,
                message_type=message_type,
                offset=offset,
            ).build_spec(hooks.clamp_limit)
            conversations = await ops.query_conversations(spec)
            diagnostics = await ops.diagnose_query_miss(spec) if not conversations else None
            return hooks.json_payload(conversation_query_result_payload(conversations, diagnostics=diagnostics))

        return await hooks.async_safe_call("list_conversations", run)

    @mcp.tool()
    async def get_conversation(
        id: str,
        no_code_blocks: bool = False,
        no_tool_calls: bool = False,
        no_tool_outputs: bool = False,
        no_file_reads: bool = False,
        prose_only: bool = False,
    ) -> str:
        async def run() -> str:
            projection = MCPContentProjectionRequest(
                no_code_blocks=no_code_blocks,
                no_tool_calls=no_tool_calls,
                no_tool_outputs=no_tool_outputs,
                no_file_reads=no_file_reads,
                prose_only=prose_only,
            ).build_projection()
            conv = await hooks.get_archive_ops().get_conversation(id, content_projection=projection)
            if conv is None:
                return hooks.error_json(f"Conversation not found: {id}")
            return hooks.json_payload(
                MCPConversationDetailPayload.from_conversation(conv, content_projection=projection)
            )

        return await hooks.async_safe_call("get_conversation", run)

    @mcp.tool()
    async def neighbor_candidates(
        id: str | None = None,
        query: str | None = None,
        provider: str | None = None,
        limit: int = 10,
        window_hours: int = 24,
    ) -> str:
        async def run() -> str:
            if not id and not (query and query.strip()):
                return hooks.error_json("neighbor_candidates requires id or query")
            candidates = await hooks.get_archive_ops().neighbor_candidates(
                conversation_id=id,
                query=query,
                provider=provider,
                limit=hooks.clamp_limit(limit),
                window_hours=max(1, window_hours),
            )
            return hooks.json_payload(conversation_neighbor_candidate_list_payload(candidates), exclude_none=True)

        return await hooks.async_safe_call("neighbor_candidates", run)

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
            ops = hooks.get_archive_ops()
            summary = await ops.get_conversation_summary(id)
            if summary is None:
                return hooks.error_json(f"Conversation not found: {id}")
            stats = await ops.get_conversation_stats(id)
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
            tree = await hooks.get_archive_ops().get_session_tree(conversation_id)
            return hooks.json_payload(conversation_summary_list_payload(tree))

        return await hooks.async_safe_call("get_session_tree", run)

    @mcp.tool()
    async def get_stats_by(group_by: Literal["provider", "month", "year"] = "provider") -> str:
        async def run() -> str:
            root = await hooks.get_archive_ops().get_stats_by(group_by)
            return hooks.json_payload(MCPStatsByPayload(root=root))

        return await hooks.async_safe_call("get_stats_by", run)

    @mcp.tool()
    async def get_messages(
        conversation_id: str,
        message_role: str | None = None,
        message_type: str | None = None,
        no_code_blocks: bool = False,
        no_tool_calls: bool = False,
        no_tool_outputs: bool = False,
        no_file_reads: bool = False,
        prose_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        async def run() -> str:
            projection = MCPContentProjectionRequest(
                no_code_blocks=no_code_blocks,
                no_tool_calls=no_tool_calls,
                no_tool_outputs=no_tool_outputs,
                no_file_reads=no_file_reads,
                prose_only=prose_only,
            ).build_projection()

            ops = hooks.get_archive_ops()
            conv = await ops.get_conversation(conversation_id, content_projection=projection)
            if conv is None:
                return hooks.error_json(f"Conversation not found: {conversation_id}")

            # Filter by message_role
            messages = list(conv.messages)
            if message_role:
                from polylogue.lib.message.roles import normalize_message_roles

                roles = normalize_message_roles(message_role)
                messages = [m for m in messages if m.role in roles]

            # Filter by message_type (post-filter on content blocks)
            if message_type:
                if message_type == "tool_use":
                    messages = [m for m in messages if any(b.get("type") == "tool_use" for b in m.content_blocks)]
                elif message_type == "tool_result":
                    messages = [m for m in messages if any(b.get("type") == "tool_result" for b in m.content_blocks)]
                elif message_type == "thinking":
                    messages = [m for m in messages if any(b.get("type") == "thinking" for b in m.content_blocks)]
                elif message_type == "summary":
                    messages = [m for m in messages if str(m.role) == "system"]

            total = len(messages)
            paginated = messages[offset : offset + limit]

            return hooks.json_payload(
                MCPMessagesListPayload(
                    conversation_id=conversation_id,
                    messages=tuple(MCPMessagePayload.from_message(msg) for msg in paginated),
                    total=total,
                    limit=limit,
                    offset=offset,
                )
            )

        return await hooks.async_safe_call("get_messages", run)

    @mcp.tool()
    async def raw_records(
        conversation_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        async def run() -> str:
            ops = hooks.get_archive_ops()
            conv_check = await ops.get_conversation_summary(conversation_id)
            if conv_check is None:
                return hooks.error_json(f"Conversation not found: {conversation_id}")
            records, total = await ops.get_raw_records_for_conversation(
                conversation_id,
                limit=limit,
                offset=offset,
            )
            return hooks.json_payload(
                MCPRawRecordsListPayload(
                    conversation_id=conversation_id,
                    raw_records=tuple(MCPRawRecordPayload.from_record(r) for r in records),
                    total=total,
                    limit=limit,
                    offset=offset,
                )
            )

        return await hooks.async_safe_call("raw_records", run)

    @mcp.tool()
    def readiness_check() -> str:
        def run() -> str:
            from polylogue.readiness import get_readiness

            report = get_readiness(hooks.get_config())
            return hooks.json_payload(
                MCPReadinessReportPayload.from_report(
                    report,
                    include_counts=True,
                    include_detail=True,
                    include_cached=True,
                ),
                exclude_none=True,
            )

        return hooks.safe_call("readiness_check", run)


def register_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    register_query_tools(mcp, hooks)
    register_mutation_tools(mcp, hooks)
    register_read_tools(mcp, hooks)
    register_maintenance_tools(mcp, hooks)
    register_product_tools(mcp, hooks)


__all__ = ["register_query_tools", "register_read_tools", "register_tools"]
