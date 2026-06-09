"""Stable MCP tool registration surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from polylogue.core.enums import Origin
from polylogue.core.sources import provider_from_origin
from polylogue.mcp.archive_support import (
    active_archive_root,
    archive_messages_payload,
    archive_search_payload,
    archive_session_list_payload,
    archive_summary_payload,
)
from polylogue.mcp.payloads import (
    MCPArchiveSearchHitPayload,
    MCPArchiveSearchPayload,
    MCPArchiveSessionListPayload,
    MCPArchiveSessionPayload,
    MCPArchiveSessionSummaryPayload,
    MCPArchiveStatsPayload,
    MCPEmbeddingPreflightPayload,
    MCPEmbeddingStatusPayload,
    MCPRawArtifactPayload,
    MCPRawArtifactsListPayload,
    MCPReadinessReportPayload,
    MCPStatsByPayload,
    logical_session_payload,
    neighbor_candidates_payload,
    session_topology_payload,
    session_tree_payload,
)
from polylogue.mcp.query_contracts import (
    MCPToolLimit,
    MCPToolOffset,
    build_session_query_request,
    session_query_request_signature,
)
from polylogue.mcp.server_context_tools import register_context_tools
from polylogue.mcp.server_insight_tools import register_insight_tools
from polylogue.mcp.server_maintenance_tools import register_maintenance_tools
from polylogue.mcp.server_mutation_tools import register_mutation_tools
from polylogue.mcp.server_support import role_allows
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.config import Config
    from polylogue.mcp.server_support import ServerCallbacks


@dataclass(frozen=True)
class _MCPEmbeddingStatusEnv:
    config: Config


def _split_archive_csv(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(token.strip() for token in value.split(",") if token.strip())


def register_query_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    async def _search(**kwargs: object) -> str:
        if "query" not in kwargs:
            raise TypeError("search() missing required keyword argument: 'query'")
        request = build_session_query_request(**kwargs)

        async def run() -> str:
            from dataclasses import replace as dc_replace

            from polylogue.surfaces.payloads import InvalidSearchCursorError, decode_search_cursor

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
            config = hooks.get_config()
            with ArchiveStore.open_existing(active_archive_root(config) or config.archive_root) as archive:
                return hooks.json_payload(
                    archive_search_payload(
                        archive,
                        spec,
                        query=request.query or "",
                        limit=clamped_limit,
                        offset=clamped_offset,
                        retrieval_lane=request.retrieval_lane or "dialogue",
                        sort=request.sort,
                    )
                )

        return await hooks.async_safe_call("search", run)

    cast(Any, _search).__signature__ = session_query_request_signature(include_query=True, query_required=True)
    _search.__name__ = "search"
    _search.__doc__ = (
        "Search the archive for sessions matching ``query``. "
        "Filter parameters mirror ``MCPSessionQueryRequest`` fields."
    )
    mcp.tool()(_search)

    async def _list_sessions(**kwargs: object) -> str:
        request = build_session_query_request(**kwargs)

        async def run() -> str:
            spec = request.build_spec(hooks.clamp_limit)
            config = hooks.get_config()
            with ArchiveStore.open_existing(active_archive_root(config) or config.archive_root) as archive:
                return hooks.json_payload(archive_session_list_payload(archive, spec))

        return await hooks.async_safe_call("list_sessions", run)

    cast(Any, _list_sessions).__signature__ = session_query_request_signature(
        include_query=False,
    )
    _list_sessions.__name__ = "list_sessions"
    _list_sessions.__doc__ = (
        "List sessions matching the supplied filters. Filter parameters mirror ``MCPSessionQueryRequest`` fields."
    )
    mcp.tool()(_list_sessions)

    @mcp.tool()
    async def get_session(
        id: str,
    ) -> str:
        async def run() -> str:
            config = hooks.get_config()
            with ArchiveStore.open_existing(active_archive_root(config) or config.archive_root) as archive:
                try:
                    session_id = archive.resolve_session_id(id)
                    archive_summary = archive.read_summary(session_id)
                except (KeyError, ValueError):
                    return hooks.error_json(f"Session not found: {id}", code="not_found")
                return hooks.json_payload(archive_summary_payload(archive_summary))

        return await hooks.async_safe_call("get_session", run)

    @mcp.tool()
    async def archive_list_sessions(
        origin: str | None = None,
        exclude_origin: str | None = None,
        tag: str | None = None,
        exclude_tag: str | None = None,
        repo: str | None = None,
        has_type: str | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool: str | None = None,
        exclude_tool: str | None = None,
        action: str | None = None,
        exclude_action: str | None = None,
        action_sequence: str | None = None,
        action_text: str | None = None,
        referenced_path: str | None = None,
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: MCPToolLimit = 10,
        offset: MCPToolOffset = 0,
        sample: int | None = None,
    ) -> str:
        """List sessions from the split archive store."""

        async def run() -> str:
            poly = hooks.get_polylogue()
            clamped_limit = hooks.clamp_limit(sample if sample is not None else limit)
            clamped_offset = 0 if sample is not None else max(0, offset)
            excluded_origins = _split_archive_csv(exclude_origin)
            tags = _split_archive_csv(tag)
            excluded_tags = _split_archive_csv(exclude_tag)
            repo_names = _split_archive_csv(repo)
            has_types = _split_archive_csv(has_type)
            tool_terms = _split_archive_csv(tool)
            excluded_tool_terms = _split_archive_csv(exclude_tool)
            action_terms = _split_archive_csv(action)
            excluded_action_terms = _split_archive_csv(exclude_action)
            action_sequence_terms = _split_archive_csv(action_sequence)
            action_text_terms = _split_archive_csv(action_text)
            referenced_paths = _split_archive_csv(referenced_path)
            summaries = await poly.archive_list_sessions(
                origin=origin,
                excluded_origins=excluded_origins,
                tags=tags,
                excluded_tags=excluded_tags,
                repo_names=repo_names,
                has_types=has_types,
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                has_paste=has_paste,
                tool_terms=tool_terms,
                excluded_tool_terms=excluded_tool_terms,
                action_terms=action_terms,
                excluded_action_terms=excluded_action_terms,
                action_sequence=action_sequence_terms,
                action_text_terms=action_text_terms,
                referenced_paths=referenced_paths,
                cwd_prefix=cwd_prefix,
                typed_only=typed_only,
                message_type=message_type,
                title=title,
                min_messages=min_messages,
                max_messages=max_messages,
                min_words=min_words,
                since=since,
                until=until,
                limit=clamped_limit,
                offset=clamped_offset,
                sample=sample is not None,
            )
            total = await poly.archive_count_sessions(
                origin=origin,
                excluded_origins=excluded_origins,
                tags=tags,
                excluded_tags=excluded_tags,
                repo_names=repo_names,
                has_types=has_types,
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                has_paste=has_paste,
                tool_terms=tool_terms,
                excluded_tool_terms=excluded_tool_terms,
                action_terms=action_terms,
                excluded_action_terms=excluded_action_terms,
                action_sequence=action_sequence_terms,
                action_text_terms=action_text_terms,
                referenced_paths=referenced_paths,
                cwd_prefix=cwd_prefix,
                typed_only=typed_only,
                message_type=message_type,
                title=title,
                min_messages=min_messages,
                max_messages=max_messages,
                min_words=min_words,
                since=since,
                until=until,
            )
            return hooks.json_payload(
                MCPArchiveSessionListPayload(
                    items=tuple(MCPArchiveSessionSummaryPayload.from_summary(summary) for summary in summaries),
                    total=total,
                    limit=clamped_limit,
                    offset=clamped_offset,
                    origin=origin,
                )
            )

        return await hooks.async_safe_call("archive_list_sessions", run)

    @mcp.tool()
    async def archive_search_sessions(
        query: str,
        origin: str | None = None,
        exclude_origin: str | None = None,
        tag: str | None = None,
        exclude_tag: str | None = None,
        repo: str | None = None,
        has_type: str | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool: str | None = None,
        exclude_tool: str | None = None,
        action: str | None = None,
        exclude_action: str | None = None,
        action_sequence: str | None = None,
        action_text: str | None = None,
        referenced_path: str | None = None,
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: MCPToolLimit = 10,
    ) -> str:
        """Search session block text."""

        async def run() -> str:
            poly = hooks.get_polylogue()
            clamped_limit = hooks.clamp_limit(limit)
            hits = await poly.archive_search_sessions(
                query,
                origin=origin,
                excluded_origins=_split_archive_csv(exclude_origin),
                tags=_split_archive_csv(tag),
                excluded_tags=_split_archive_csv(exclude_tag),
                repo_names=_split_archive_csv(repo),
                has_types=_split_archive_csv(has_type),
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                has_paste=has_paste,
                tool_terms=_split_archive_csv(tool),
                excluded_tool_terms=_split_archive_csv(exclude_tool),
                action_terms=_split_archive_csv(action),
                excluded_action_terms=_split_archive_csv(exclude_action),
                action_sequence=_split_archive_csv(action_sequence),
                action_text_terms=_split_archive_csv(action_text),
                referenced_paths=_split_archive_csv(referenced_path),
                cwd_prefix=cwd_prefix,
                typed_only=typed_only,
                message_type=message_type,
                title=title,
                min_messages=min_messages,
                max_messages=max_messages,
                min_words=min_words,
                since=since,
                until=until,
                limit=clamped_limit,
            )
            return hooks.json_payload(
                MCPArchiveSearchPayload(
                    items=tuple(MCPArchiveSearchHitPayload.from_hit(hit) for hit in hits),
                    total=len(hits),
                    limit=clamped_limit,
                    query=query,
                    origin=origin,
                )
            )

        return await hooks.async_safe_call("archive_search_sessions", run)

    @mcp.tool()
    async def archive_get_session(session_id: str) -> str:
        """Read a full session by exact id or prefix."""

        async def run() -> str:
            poly = hooks.get_polylogue()
            session = await poly.archive_get_session(session_id)
            if session is None:
                return hooks.error_json(f"archive session not found: {session_id}", code="not_found")
            return hooks.json_payload(MCPArchiveSessionPayload.from_session(session))

        return await hooks.async_safe_call("archive_get_session", run)

    @mcp.tool()
    async def neighbor_candidates(
        id: str | None = None,
        query: str | None = None,
        origin: str | None = None,
        limit: MCPToolLimit = 10,
        window_hours: int = 24,
    ) -> str:
        async def run() -> str:
            if not id and not (query and query.strip()):
                return hooks.error_json("neighbor_candidates requires id or query")

            poly = hooks.get_polylogue()
            clamped_limit = hooks.clamp_limit(limit)
            candidates = await poly.neighbor_candidates(
                session_id=id,
                query=query,
                provider=provider_from_origin(Origin(origin)).value if origin is not None else None,
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
            config = hooks.get_config()
            with ArchiveStore.open_existing(active_archive_root(config) or config.archive_root) as archive:
                archive_stats = archive.stats()
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

    @mcp.tool()
    def embedding_preflight(
        rebuild: bool = False,
        max_sessions: int | None = None,
        max_messages: int | None = None,
        max_cost_usd: float | None = None,
    ) -> str:
        def run() -> str:
            from polylogue.storage.embeddings.preflight import build_preflight_report, preflight_payload

            payload = preflight_payload(
                build_preflight_report(
                    hooks.get_config().db_path,
                    rebuild=rebuild,
                    max_sessions=max_sessions,
                    max_messages=max_messages,
                    max_cost_usd=max_cost_usd,
                )
            )
            return hooks.json_payload(MCPEmbeddingPreflightPayload.from_payload(payload))

        return hooks.safe_call("embedding_preflight", run)

    async def _facets(**kwargs: object) -> str:
        # Reuse the session query request so MCP facets share the
        # same filter vocabulary as ``search``/``list_sessions``
        # (#1269). When called with no filters the response carries
        # ``scoped_to_query=False`` and the global archive view.
        request = build_session_query_request(**kwargs)

        async def run() -> str:
            built = request.build_spec(hooks.clamp_limit)
            spec = built if built.has_filters() else None
            poly = hooks.get_polylogue()
            response = await poly.facets(spec)
            return hooks.json_payload(response)

        return await hooks.async_safe_call("facets", run)

    cast(Any, _facets).__signature__ = session_query_request_signature(include_query=True)
    _facets.__name__ = "facets"
    _facets.__doc__ = (
        "Compute scoped and global facet aggregates over the archive. "
        "Filter parameters mirror ``MCPSessionQueryRequest`` fields; "
        "when any filter narrows the result set the response carries "
        "``scoped_to_query=true`` with both ``scoped`` and ``global`` bucket "
        "counts plus inverse-document-frequency weights for each value."
    )
    mcp.tool()(_facets)


def register_read_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def get_session_summary(id: str) -> str:
        async def run() -> str:
            config = hooks.get_config()
            with ArchiveStore.open_existing(active_archive_root(config) or config.archive_root) as archive:
                try:
                    session_id = archive.resolve_session_id(id)
                    archive_summary = archive.read_summary(session_id)
                except (KeyError, ValueError):
                    return hooks.error_json(f"Session not found: {id}", code="not_found")
                return hooks.json_payload(archive_summary_payload(archive_summary))

        return await hooks.async_safe_call("get_session_summary", run)

    @mcp.tool()
    async def get_session_tree(session_id: str) -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            tree = await poly.get_session_tree(session_id)
            return hooks.json_payload(session_tree_payload(tree))

        return await hooks.async_safe_call("get_session_tree", run)

    @mcp.tool()
    async def get_session_topology(session_id: str) -> str:
        """Return typed lineage topology (ancestors/descendants/siblings/thread).

        Returns the resolved :class:`SessionTopology` graph (#1261 /
        #866 slice D) plus pre-projected ref lists so external agents can
        navigate session lineage without re-walking parent pointers.
        """

        async def run() -> str:
            poly = hooks.get_polylogue()
            topology = await poly.get_session_topology(session_id)
            if topology is None:
                return hooks.error_json(
                    f"Session not found: {session_id}",
                    code="not_found",
                )
            return hooks.json_payload(session_topology_payload(topology, session_id=str(topology.target_id)))

        return await hooks.async_safe_call("get_session_topology", run)

    @mcp.tool()
    async def get_logical_session(session_id: str) -> str:
        """Return compact logical-session lineage for read-pull consumers."""

        async def run() -> str:
            poly = hooks.get_polylogue()
            logical_session = await poly.get_logical_session(session_id)
            if logical_session is None:
                return hooks.error_json(
                    f"Session not found: {session_id}",
                    code="not_found",
                )
            return hooks.json_payload(logical_session_payload(logical_session))

        return await hooks.async_safe_call("get_logical_session", run)

    @mcp.tool()
    async def get_stats_by(group_by: Literal["origin", "day", "month", "year"] = "origin") -> str:
        """Session counts grouped by a calendar/origin dimension.

        These are the sessions-table count dimensions. The CLI ``stats
        --by`` surface additionally supports ``action``/``tool``/``repo``/
        ``work-kind``, which are insight-summary aggregations over a different
        code path, not plain session counts (#1749).
        """

        async def run() -> str:
            root = await hooks.get_polylogue().get_stats_by(group_by)
            return hooks.json_payload(MCPStatsByPayload(root=root))

        return await hooks.async_safe_call("get_stats_by", run)

    @mcp.tool()
    async def get_messages(
        session_id: str,
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
            from polylogue.archive.message.roles import normalize_message_roles
            from polylogue.archive.message.types import validate_message_type_filter

            roles = normalize_message_roles(message_role) if message_role else ()
            normalized_message_type = (
                validate_message_type_filter(message_type).value if message_type is not None else None
            )
            config = hooks.get_config()
            with ArchiveStore.open_existing(active_archive_root(config) or config.archive_root) as archive:
                try:
                    resolved_session_id = archive.resolve_session_id(session_id)
                    session = archive.read_session(resolved_session_id)
                except (KeyError, ValueError):
                    return hooks.error_json(f"Session not found: {session_id}", code="not_found")
                return hooks.json_payload(
                    archive_messages_payload(
                        session,
                        roles=tuple(str(role) for role in roles),
                        message_type=normalized_message_type,
                        limit=hooks.clamp_limit(limit),
                        offset=max(0, offset),
                    )
                )

        return await hooks.async_safe_call("get_messages", run)

    @mcp.tool()
    async def raw_artifacts(
        session_id: str,
        limit: MCPToolLimit = 50,
        offset: MCPToolOffset = 0,
    ) -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            conv_check = await poly.get_session_summary(session_id)
            if conv_check is None:
                return hooks.error_json(f"Session not found: {session_id}", code="not_found")
            canonical_id = str(conv_check.id)
            artifacts, total = await poly.get_raw_artifacts_for_session(
                canonical_id,
                limit=hooks.clamp_limit(limit),
                offset=max(0, offset),
            )
            return hooks.json_payload(
                MCPRawArtifactsListPayload(
                    session_id=canonical_id,
                    raw_artifacts=tuple(MCPRawArtifactPayload.model_validate(r) for r in artifacts),
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
