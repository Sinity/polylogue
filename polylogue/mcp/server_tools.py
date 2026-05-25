"""Stable MCP tool registration surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from polylogue.config import ConfigError
from polylogue.mcp.payloads import (
    MCPArchiveStatsPayload,
    MCPConversationSummaryPayload,
    MCPEmbeddingStatusPayload,
    MCPMessagePayload,
    MCPMessagesListPayload,
    MCPRawArtifactPayload,
    MCPRawArtifactsListPayload,
    MCPReadinessReportPayload,
    MCPStatsByPayload,
    conversation_query_result_payload,
    conversation_search_result_payload,
    logical_session_payload,
    neighbor_candidates_payload,
    session_topology_payload,
    session_tree_payload,
)
from polylogue.mcp.query_contracts import (
    MCPContentProjectionRequest,
    MCPToolLimit,
    MCPToolOffset,
    build_conversation_query_request,
    conversation_query_request_signature,
)
from polylogue.mcp.server_context_tools import register_context_tools
from polylogue.mcp.server_insight_tools import register_insight_tools
from polylogue.mcp.server_maintenance_tools import register_maintenance_tools
from polylogue.mcp.server_mutation_tools import register_mutation_tools
from polylogue.mcp.server_support import role_allows

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.config import Config
    from polylogue.mcp.server_support import ServerCallbacks
    from polylogue.storage.sqlite.queries.message_query_reads import MessageTypeName


@dataclass(frozen=True)
class _MCPEmbeddingStatusEnv:
    config: Config


def register_query_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    async def _search(**kwargs: object) -> str:
        if "query" not in kwargs:
            raise TypeError("search() missing required keyword argument: 'query'")
        request = build_conversation_query_request(**kwargs)

        async def run() -> str:
            from dataclasses import replace as dc_replace

            from polylogue.surfaces.payloads import InvalidSearchCursorError, decode_search_cursor

            ops = hooks.get_archive_ops()
            clamped_limit = hooks.clamp_limit(request.limit)
            clamped_offset = max(0, request.offset)
            decoded_cursor = None
            if request.cursor:
                try:
                    decoded_cursor = decode_search_cursor(request.cursor)
                except InvalidSearchCursorError as exc:
                    return hooks.error_json(str(exc), code="invalid_cursor")
            spec = request.build_spec(hooks.clamp_limit)
            if decoded_cursor is not None:
                # Advance fetch past the anchor and buffer the window so
                # post-fetch trim cannot starve the response.
                spec = dc_replace(
                    spec,
                    offset=decoded_cursor.r,
                    limit=(spec.limit or clamped_limit) + clamped_limit,
                )
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
                    query=request.query or "",
                    retrieval_lane=request.retrieval_lane or "auto",
                    sort=request.sort,
                    cursor=decoded_cursor,
                )
            )

        return await hooks.async_safe_call("search", run)

    cast(Any, _search).__signature__ = conversation_query_request_signature(include_query=True, query_required=True)
    _search.__name__ = "search"
    _search.__doc__ = (
        "Search the archive for conversations matching ``query``. "
        "Filter parameters mirror ``MCPConversationQueryRequest`` fields."
    )
    mcp.tool()(_search)

    async def _list_conversations(**kwargs: object) -> str:
        request = build_conversation_query_request(**kwargs)

        async def run() -> str:
            ops = hooks.get_archive_ops()
            clamped_limit = hooks.clamp_limit(request.limit)
            clamped_offset = max(0, request.offset)
            spec = request.build_spec(hooks.clamp_limit)
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

    cast(Any, _list_conversations).__signature__ = conversation_query_request_signature(
        include_query=False,
    )
    _list_conversations.__name__ = "list_conversations"
    _list_conversations.__doc__ = (
        "List conversations matching the supplied filters. "
        "Filter parameters mirror ``MCPConversationQueryRequest`` fields."
    )
    mcp.tool()(_list_conversations)

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

    @mcp.tool()
    def embedding_status(detail: bool = False) -> str:
        def run() -> str:
            from polylogue.storage.embeddings.status_payload import embedding_status_payload

            payload = embedding_status_payload(
                _MCPEmbeddingStatusEnv(hooks.get_config()),
                include_retrieval_bands=detail,
                include_detail=detail,
            )
            return hooks.json_payload(MCPEmbeddingStatusPayload.from_payload(dict(payload)))

        return hooks.safe_call("embedding_status", run)

    async def _facets(**kwargs: object) -> str:
        # Reuse the conversation query request so MCP facets share the
        # same filter vocabulary as ``search``/``list_conversations``
        # (#1269). When called with no filters the response carries
        # ``scoped_to_query=False`` and the global archive view.
        request = build_conversation_query_request(**kwargs)

        async def run() -> str:
            built = request.build_spec(hooks.clamp_limit)
            spec = built if built.has_filters() else None
            poly = hooks.get_polylogue()
            response = await poly.facets(spec)
            return hooks.json_payload(response)

        return await hooks.async_safe_call("facets", run)

    cast(Any, _facets).__signature__ = conversation_query_request_signature(include_query=True)
    _facets.__name__ = "facets"
    _facets.__doc__ = (
        "Compute scoped and global facet aggregates over the archive. "
        "Filter parameters mirror ``MCPConversationQueryRequest`` fields; "
        "when any filter narrows the result set the response carries "
        "``scoped_to_query=true`` with both ``scoped`` and ``global`` bucket "
        "counts plus inverse-document-frequency weights for each value."
    )
    mcp.tool()(_facets)


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
    async def get_session_topology(conversation_id: str) -> str:
        """Return typed lineage topology (ancestors/descendants/siblings/thread).

        Returns the resolved :class:`SessionTopology` graph (#1261 /
        #866 slice D) plus pre-projected ref lists so external agents can
        navigate session lineage without re-walking parent pointers.
        """

        async def run() -> str:
            poly = hooks.get_polylogue()
            topology = await poly.get_session_topology(conversation_id)
            if topology is None:
                return hooks.error_json(
                    f"Conversation not found: {conversation_id}",
                    code="not_found",
                )
            return hooks.json_payload(session_topology_payload(topology, conversation_id=str(topology.target_id)))

        return await hooks.async_safe_call("get_session_topology", run)

    @mcp.tool()
    async def get_logical_session(conversation_id: str) -> str:
        """Return compact logical-session lineage for read-pull consumers."""

        async def run() -> str:
            poly = hooks.get_polylogue()
            logical_session = await poly.get_logical_session(conversation_id)
            if logical_session is None:
                return hooks.error_json(
                    f"Conversation not found: {conversation_id}",
                    code="not_found",
                )
            return hooks.json_payload(logical_session_payload(logical_session))

        return await hooks.async_safe_call("get_logical_session", run)

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
