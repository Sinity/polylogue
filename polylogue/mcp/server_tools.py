"""Stable MCP tool registration surface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from polylogue.config import ConfigError
from polylogue.mcp.payloads import (
    MCPArchiveStatsPayload,
    MCPConversationSummaryPayload,
    MCPMessagePayload,
    MCPMessagesListPayload,
    MCPRawArtifactPayload,
    MCPRawArtifactsListPayload,
    MCPReadinessReportPayload,
    MCPStatsByPayload,
    conversation_query_result_payload,
    conversation_search_result_payload,
    neighbor_candidates_payload,
    session_tree_payload,
)
from polylogue.mcp.query_contracts import (
    MCPContentProjectionRequest,
    MCPConversationQueryRequest,
    MCPToolLimit,
    MCPToolOffset,
)
from polylogue.mcp.server_context_tools import register_context_tools
from polylogue.mcp.server_insight_tools import register_insight_tools
from polylogue.mcp.server_maintenance_tools import register_maintenance_tools
from polylogue.mcp.server_mutation_tools import register_mutation_tools
from polylogue.mcp.server_support import role_allows

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks
    from polylogue.storage.sqlite.queries.message_query_reads import MessageTypeName


def register_query_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def search(
        query: str,
        limit: MCPToolLimit = 10,
        retrieval_lane: str | None = None,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        tag: str | None = None,
        repo: str | None = None,
        title: str | None = None,
        contains: str | None = None,
        exclude_text: str | None = None,
        exclude_provider: str | None = None,
        exclude_tag: str | None = None,
        has_type: str | None = None,
        conv_id: str | None = None,
        referenced_path: str | None = None,
        cwd_prefix: str | None = None,
        action: str | None = None,
        exclude_action: str | None = None,
        action_sequence: str | None = None,
        action_text: str | None = None,
        tool: str | None = None,
        exclude_tool: str | None = None,
        sort: str | None = None,
        reverse: bool = False,
        latest: str | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        typed_only: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        sample: int | None = None,
        similar_text: str | None = None,
        since_session: str | None = None,
        since_session_id: str | None = None,
        message_type: str | None = None,
        offset: MCPToolOffset = 0,
    ) -> str:
        async def run() -> str:
            ops = hooks.get_archive_ops()
            clamped_limit = hooks.clamp_limit(limit)
            clamped_offset = max(0, offset)
            spec = MCPConversationQueryRequest(
                query=query,
                retrieval_lane=retrieval_lane,
                provider=provider,
                since=since,
                until=until,
                tag=tag,
                repo=repo,
                title=title,
                contains=contains,
                exclude_text=exclude_text,
                exclude_provider=exclude_provider,
                exclude_tag=exclude_tag,
                has_type=has_type,
                conv_id=conv_id,
                referenced_path=referenced_path,
                cwd_prefix=cwd_prefix,
                action=action,
                exclude_action=exclude_action,
                action_sequence=action_sequence,
                action_text=action_text,
                tool=tool,
                exclude_tool=exclude_tool,
                sort=sort,
                reverse=reverse,
                latest=latest,
                limit=clamped_limit,
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                has_paste=has_paste,
                typed_only=typed_only,
                min_messages=min_messages,
                max_messages=max_messages,
                min_words=min_words,
                sample=sample,
                similar_text=similar_text,
                since_session=since_session,
                since_session_id=since_session_id,
                message_type=message_type,
                offset=clamped_offset,
            ).build_spec(hooks.clamp_limit)
            results = await ops.search_conversation_hits(spec)
            total = await spec.count(hooks.get_query_store())
            diagnostics = await ops.diagnose_query_miss(spec) if not results else None
            return hooks.json_payload(
                conversation_search_result_payload(
                    results,
                    total=total,
                    limit=clamped_limit,
                    offset=clamped_offset,
                    diagnostics=diagnostics,
                )
            )

        return await hooks.async_safe_call("search", run)

    @mcp.tool()
    async def list_conversations(
        limit: MCPToolLimit = 10,
        retrieval_lane: str | None = None,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        tag: str | None = None,
        repo: str | None = None,
        title: str | None = None,
        contains: str | None = None,
        exclude_text: str | None = None,
        exclude_provider: str | None = None,
        exclude_tag: str | None = None,
        has_type: str | None = None,
        conv_id: str | None = None,
        referenced_path: str | None = None,
        cwd_prefix: str | None = None,
        action: str | None = None,
        exclude_action: str | None = None,
        action_sequence: str | None = None,
        action_text: str | None = None,
        tool: str | None = None,
        exclude_tool: str | None = None,
        sort: str | None = None,
        reverse: bool = False,
        latest: str | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        typed_only: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        sample: int | None = None,
        similar_text: str | None = None,
        since_session: str | None = None,
        since_session_id: str | None = None,
        message_type: str | None = None,
        offset: MCPToolOffset = 0,
    ) -> str:
        async def run() -> str:
            ops = hooks.get_archive_ops()
            clamped_limit = hooks.clamp_limit(limit)
            clamped_offset = max(0, offset)
            spec = MCPConversationQueryRequest(
                provider=provider,
                retrieval_lane=retrieval_lane,
                since=since,
                until=until,
                tag=tag,
                repo=repo,
                title=title,
                contains=contains,
                exclude_text=exclude_text,
                exclude_provider=exclude_provider,
                exclude_tag=exclude_tag,
                has_type=has_type,
                conv_id=conv_id,
                referenced_path=referenced_path,
                cwd_prefix=cwd_prefix,
                action=action,
                exclude_action=exclude_action,
                action_sequence=action_sequence,
                action_text=action_text,
                tool=tool,
                exclude_tool=exclude_tool,
                sort=sort,
                reverse=reverse,
                latest=latest,
                limit=clamped_limit,
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                has_paste=has_paste,
                typed_only=typed_only,
                min_messages=min_messages,
                max_messages=max_messages,
                min_words=min_words,
                sample=sample,
                similar_text=similar_text,
                since_session=since_session,
                since_session_id=since_session_id,
                message_type=message_type,
                offset=clamped_offset,
            ).build_spec(hooks.clamp_limit)
            conversations = await ops.query_conversations(spec)
            total = await spec.count(hooks.get_query_store())
            diagnostics = None
            if not conversations:
                try:
                    diagnostics = await ops.diagnose_query_miss(spec)
                except ConfigError:
                    diagnostics = None
            return hooks.json_payload(
                conversation_query_result_payload(
                    conversations,
                    total=total,
                    limit=clamped_limit,
                    offset=clamped_offset,
                    diagnostics=diagnostics,
                )
            )

        return await hooks.async_safe_call("list_conversations", run)

    @mcp.tool()
    async def get_conversation(
        id: str,
    ) -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            summary = await poly.get_conversation_summary(id)
            if summary is None:
                return hooks.error_json(f"Conversation not found: {id}", code="not_found")
            stats = await poly.get_conversation_stats(id)
            return hooks.json_payload(
                MCPConversationSummaryPayload.from_summary(
                    summary,
                    message_count=stats["total_messages"] if stats else 0,
                )
            )

        return await hooks.async_safe_call("get_conversation", run)

    @mcp.tool()
    async def neighbor_candidates(
        id: str | None = None,
        query: str | None = None,
        provider: str | None = None,
        limit: MCPToolLimit = 10,
        window_hours: int = 24,
    ) -> str:
        async def run() -> str:
            if not id and not (query and query.strip()):
                return hooks.error_json("neighbor_candidates requires id or query")

            poly = hooks.get_polylogue()
            clamped_limit = hooks.clamp_limit(limit)
            candidates = await poly.neighbor_candidates(
                conversation_id=id,
                query=query,
                provider=provider,
                limit=clamped_limit,
                window_hours=max(1, window_hours),
            )
            return hooks.json_payload(
                neighbor_candidates_payload(candidates, limit=clamped_limit),
                exclude_none=True,
            )

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
            poly = hooks.get_polylogue()
            summary = await poly.get_conversation_summary(id)
            if summary is None:
                return hooks.error_json(f"Conversation not found: {id}", code="not_found")
            stats = await poly.get_conversation_stats(id)
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
            poly = hooks.get_polylogue()
            tree = await poly.get_session_tree(conversation_id)
            return hooks.json_payload(session_tree_payload(tree))

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
        limit: MCPToolLimit = 50,
        offset: MCPToolOffset = 0,
    ) -> str:
        async def run() -> str:
            projection = MCPContentProjectionRequest(
                no_code_blocks=no_code_blocks,
                no_tool_calls=no_tool_calls,
                no_tool_outputs=no_tool_outputs,
                no_file_reads=no_file_reads,
                prose_only=prose_only,
            ).build_projection()

            from polylogue.api.archive import ConversationNotFoundError

            poly = hooks.get_polylogue()
            from polylogue.archive.message.roles import normalize_message_roles
            from polylogue.archive.message.types import validate_message_type_filter

            roles = normalize_message_roles(message_role) if message_role else ()
            normalized_message_type = (
                validate_message_type_filter(message_type).value if message_type is not None else None
            )
            try:
                paginated, total = await poly.get_messages_paginated(
                    conversation_id,
                    message_role=roles,
                    message_type=cast("MessageTypeName | None", normalized_message_type),
                    limit=hooks.clamp_limit(limit),
                    offset=max(0, offset),
                    content_projection=projection,
                )
            except ConversationNotFoundError:
                return hooks.error_json(f"Conversation not found: {conversation_id}", code="not_found")

            return hooks.json_payload(
                MCPMessagesListPayload(
                    conversation_id=conversation_id,
                    messages=tuple(MCPMessagePayload.from_message(msg) for msg in paginated),
                    total=total,
                    limit=hooks.clamp_limit(limit),
                    offset=max(0, offset),
                )
            )

        return await hooks.async_safe_call("get_messages", run)

    @mcp.tool()
    async def raw_artifacts(
        conversation_id: str,
        limit: MCPToolLimit = 50,
        offset: MCPToolOffset = 0,
    ) -> str:
        async def run() -> str:
            ops = hooks.get_archive_ops()
            conv_check = await ops.get_conversation_summary(conversation_id)
            if conv_check is None:
                return hooks.error_json(f"Conversation not found: {conversation_id}", code="not_found")
            canonical_id = str(conv_check.id)
            artifacts, total = await ops.get_raw_artifacts_for_conversation(
                canonical_id,
                limit=hooks.clamp_limit(limit),
                offset=max(0, offset),
            )
            return hooks.json_payload(
                MCPRawArtifactsListPayload(
                    conversation_id=canonical_id,
                    raw_artifacts=tuple(MCPRawArtifactPayload.from_record(r) for r in artifacts),
                    total=total,
                    limit=hooks.clamp_limit(limit),
                    offset=max(0, offset),
                )
            )

        return await hooks.async_safe_call("raw_artifacts", run)

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
    register_read_tools(mcp, hooks)
    register_context_tools(mcp, hooks)
    register_insight_tools(mcp, hooks)
    if role_allows(hooks.role, "write"):
        register_mutation_tools(mcp, hooks)
    if role_allows(hooks.role, "admin"):
        register_maintenance_tools(mcp, hooks)


__all__ = ["register_query_tools", "register_read_tools", "register_tools"]
