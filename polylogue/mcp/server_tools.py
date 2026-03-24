"""Tool registration for the MCP server."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from polylogue.logging import get_logger
from polylogue.mcp.payloads import (
    MCPArchiveStatsPayload,
    MCPConversationDetailPayload,
    MCPConversationSummaryListPayload,
    MCPConversationSummaryPayload,
    MCPHealthReportPayload,
    MCPMetadataPayload,
    MCPMutationStatusPayload,
    MCPRootPayload,
    MCPStatsByPayload,
    MCPTagCountsPayload,
)
from polylogue.mcp.query_support import build_query_spec

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks

logger = get_logger(__name__)


def register_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    """Register all MCP tools on the given server."""
    _register_query_tools(mcp, hooks)
    _register_mutation_tools(mcp, hooks)
    _register_extended_read_tools(mcp, hooks)
    _register_maintenance_tools(mcp, hooks)
    _register_profile_tools(mcp, hooks)


def _register_query_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
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
        async def _run() -> str:
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

        return await hooks.async_safe_call("search", _run)

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
        async def _run() -> str:
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

        return await hooks.async_safe_call("list_conversations", _run)

    @mcp.tool()
    async def get_conversation(id: str) -> str:
        async def _run() -> str:
            conv = await hooks.get_archive_ops().get_conversation(id)
            if conv is None:
                return hooks.error_json(f"Conversation not found: {id}")
            return hooks.json_payload(MCPConversationDetailPayload.from_conversation(conv))

        return await hooks.async_safe_call("get_conversation", _run)

    @mcp.tool()
    async def stats() -> str:
        async def _run() -> str:
            archive_stats = await hooks.get_archive_ops().storage_stats()
            return hooks.json_payload(
                MCPArchiveStatsPayload.from_archive_stats(
                    archive_stats,
                    include_embedded=True,
                    include_db_size=True,
                )
            )

        return await hooks.async_safe_call("stats", _run)


def _register_mutation_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def add_tag(conversation_id: str, tag: str) -> str:
        async def _run() -> str:
            repo = hooks.get_repo()
            await repo.add_tag(conversation_id, tag)
            return hooks.json_payload(
                MCPMutationStatusPayload(
                    status="ok",
                    conversation_id=conversation_id,
                    tag=tag,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("add_tag", _run)

    @mcp.tool()
    async def remove_tag(conversation_id: str, tag: str) -> str:
        async def _run() -> str:
            repo = hooks.get_repo()
            await repo.remove_tag(conversation_id, tag)
            return hooks.json_payload(
                MCPMutationStatusPayload(
                    status="ok",
                    conversation_id=conversation_id,
                    tag=tag,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("remove_tag", _run)

    @mcp.tool()
    async def list_tags(provider: str | None = None) -> str:
        async def _run() -> str:
            tags = await hooks.get_repo().list_tags(provider=provider)
            return hooks.json_payload(MCPTagCountsPayload(root=tags))

        return await hooks.async_safe_call("list_tags", _run)

    @mcp.tool()
    async def get_metadata(conversation_id: str) -> str:
        async def _run() -> str:
            metadata = await hooks.get_repo().get_metadata(conversation_id)
            return hooks.json_payload(MCPMetadataPayload(root=metadata))

        return await hooks.async_safe_call("get_metadata", _run)

    @mcp.tool()
    async def set_metadata(conversation_id: str, key: str, value: str) -> str:
        async def _run() -> str:
            repo = hooks.get_repo()
            try:
                parsed_value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                parsed_value = value
            await repo.update_metadata(conversation_id, key, parsed_value)
            return hooks.json_payload(
                MCPMutationStatusPayload(
                    status="ok",
                    conversation_id=conversation_id,
                    key=key,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("set_metadata", _run)

    @mcp.tool()
    async def delete_metadata(conversation_id: str, key: str) -> str:
        async def _run() -> str:
            await hooks.get_repo().delete_metadata(conversation_id, key)
            return hooks.json_payload(
                MCPMutationStatusPayload(
                    status="ok",
                    conversation_id=conversation_id,
                    key=key,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("delete_metadata", _run)

    @mcp.tool()
    async def delete_conversation(conversation_id: str, confirm: bool = False) -> str:
        async def _run() -> str:
            if not confirm:
                return hooks.error_json(
                    "Safety guard: set confirm=true to delete",
                    conversation_id=conversation_id,
                )
            deleted = await hooks.get_repo().delete_conversation(conversation_id)
            return hooks.json_payload(
                MCPMutationStatusPayload(
                    status="deleted" if deleted else "not_found",
                    conversation_id=conversation_id,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("delete_conversation", _run)


def _register_extended_read_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def get_conversation_summary(id: str) -> str:
        async def _run() -> str:
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

        return await hooks.async_safe_call("get_conversation_summary", _run)

    @mcp.tool()
    async def get_session_tree(conversation_id: str) -> str:
        async def _run() -> str:
            tree = await hooks.get_repo().get_session_tree(conversation_id)
            return hooks.json_payload(
                MCPConversationSummaryListPayload(
                    root=[
                        MCPConversationSummaryPayload.from_conversation(conv)
                        for conv in tree
                    ]
                )
            )

        return await hooks.async_safe_call("get_session_tree", _run)

    @mcp.tool()
    async def get_stats_by(group_by: str = "provider") -> str:
        async def _run() -> str:
            root = await hooks.get_repo().queries.get_stats_by(group_by)
            return hooks.json_payload(MCPStatsByPayload(root=root))

        return await hooks.async_safe_call("get_stats_by", _run)

    @mcp.tool()
    def health_check() -> str:
        def _run() -> str:
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

        return hooks.safe_call("health_check", _run)


def _register_maintenance_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def rebuild_index() -> str:
        async def _run() -> str:
            from polylogue.pipeline.services.indexing import IndexService

            repo = hooks.get_repo()
            service = IndexService(config=hooks.get_config(), backend=repo.backend)
            success = await service.rebuild_index()
            status_info = await service.get_index_status()
            return hooks.json_payload(
                MCPMutationStatusPayload(
                    status="ok" if success else "failed",
                    index_exists=status_info.get("exists", False),
                    indexed_messages=status_info.get("count", 0),
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("rebuild_index", _run)

    @mcp.tool()
    async def update_index(conversation_ids: list[str]) -> str:
        async def _run() -> str:
            from polylogue.pipeline.services.indexing import IndexService

            repo = hooks.get_repo()
            service = IndexService(config=hooks.get_config(), backend=repo.backend)
            success = await service.update_index(conversation_ids)
            return hooks.json_payload(
                MCPMutationStatusPayload(
                    status="ok" if success else "failed",
                    conversation_count=len(conversation_ids),
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("update_index", _run)

    @mcp.tool()
    async def export_conversation(id: str, format: str = "markdown") -> str:
        async def _run() -> str:
            from polylogue.rendering.formatting import format_conversation

            conv = await hooks.get_repo().view(id)
            if conv is None:
                return hooks.error_json(f"Conversation not found: {id}")
            valid_formats = {
                "markdown",
                "json",
                "html",
                "yaml",
                "plaintext",
                "csv",
                "obsidian",
                "org",
            }
            fmt = format if format in valid_formats else "markdown"
            return format_conversation(conv, fmt, None)

        return await hooks.async_safe_call("export_conversation", _run)


def _register_profile_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def session_profile(conversation_id: str) -> str:
        async def _run() -> str:
            product = await hooks.get_archive_ops().get_session_profile_product(conversation_id)
            if product is None:
                return hooks.error_json("Conversation not found", conversation_id=conversation_id)
            return hooks.json_payload(product, exclude_none=True)

        return await hooks.async_safe_call("session_profile", _run)

    @mcp.tool()
    async def session_profiles(
        since: str | None = None,
        until: str | None = None,
        provider: str | None = None,
        query: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        async def _run() -> str:
            from polylogue.archive_products import SessionProfileProductQuery

            products = await hooks.get_archive_ops().list_session_profile_products(
                SessionProfileProductQuery(
                    provider=provider,
                    since=since,
                    until=until,
                    query=query,
                    limit=hooks.clamp_limit(limit),
                    offset=max(0, int(offset)),
                )
            )
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "count": len(products),
                        "items": [product.model_dump(mode="json") for product in products],
                    }
                )
            )

        return await hooks.async_safe_call("session_profiles", _run)

    @mcp.tool()
    async def session_work_events(
        conversation_id: str | None = None,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        kind: str | None = None,
        query: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        async def _run() -> str:
            from polylogue.archive_products import SessionWorkEventProductQuery

            ops = hooks.get_archive_ops()
            if conversation_id:
                products = await ops.get_session_work_event_products(conversation_id)
            else:
                products = await ops.list_session_work_event_products(
                    SessionWorkEventProductQuery(
                        provider=provider,
                        since=since,
                        until=until,
                        kind=kind,
                        query=query,
                        limit=hooks.clamp_limit(limit),
                        offset=max(0, int(offset)),
                    )
                )
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "count": len(products),
                        "items": [product.model_dump(mode="json") for product in products],
                    }
                )
            )

        return await hooks.async_safe_call("session_work_events", _run)

    @mcp.tool()
    async def session_tag_rollups(
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        query: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> str:
        async def _run() -> str:
            from polylogue.archive_products import SessionTagRollupQuery

            products = await hooks.get_archive_ops().list_session_tag_rollup_products(
                SessionTagRollupQuery(
                    provider=provider,
                    since=since,
                    until=until,
                    query=query,
                    limit=hooks.clamp_limit(limit),
                    offset=max(0, int(offset)),
                )
            )
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "count": len(products),
                        "items": [product.model_dump(mode="json") for product in products],
                    }
                )
            )

        return await hooks.async_safe_call("session_tag_rollups", _run)

    @mcp.tool()
    async def work_threads(
        since: str | None = None,
        until: str | None = None,
        query: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        async def _run() -> str:
            from polylogue.archive_products import WorkThreadProductQuery

            products = await hooks.get_archive_ops().list_work_thread_products(
                WorkThreadProductQuery(
                    since=since,
                    until=until,
                    query=query,
                    limit=hooks.clamp_limit(limit),
                    offset=max(0, int(offset)),
                )
            )
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "count": len(products),
                        "items": [product.model_dump(mode="json") for product in products],
                    }
                )
            )

        return await hooks.async_safe_call("work_threads", _run)

    @mcp.tool()
    async def day_session_summaries(
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 90,
        offset: int = 0,
    ) -> str:
        async def _run() -> str:
            from polylogue.archive_products import DaySessionSummaryProductQuery

            products = await hooks.get_archive_ops().list_day_session_summary_products(
                DaySessionSummaryProductQuery(
                    provider=provider,
                    since=since,
                    until=until,
                    limit=hooks.clamp_limit(limit),
                    offset=max(0, int(offset)),
                )
            )
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "count": len(products),
                        "items": [product.model_dump(mode="json") for product in products],
                    }
                )
            )

        return await hooks.async_safe_call("day_session_summaries", _run)

    @mcp.tool()
    async def week_session_summaries(
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 52,
        offset: int = 0,
    ) -> str:
        async def _run() -> str:
            from polylogue.archive_products import WeekSessionSummaryProductQuery

            products = await hooks.get_archive_ops().list_week_session_summary_products(
                WeekSessionSummaryProductQuery(
                    provider=provider,
                    since=since,
                    until=until,
                    limit=hooks.clamp_limit(limit),
                    offset=max(0, int(offset)),
                )
            )
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "count": len(products),
                        "items": [product.model_dump(mode="json") for product in products],
                    }
                )
            )

        return await hooks.async_safe_call("week_session_summaries", _run)

    @mcp.tool()
    async def maintenance_runs(limit: int = 20) -> str:
        async def _run() -> str:
            from polylogue.archive_products import MaintenanceRunProductQuery

            products = await hooks.get_archive_ops().list_maintenance_run_products(
                MaintenanceRunProductQuery(limit=hooks.clamp_limit(limit))
            )
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "count": len(products),
                        "items": [product.model_dump(mode="json") for product in products],
                    }
                )
            )

        return await hooks.async_safe_call("maintenance_runs", _run)

    @mcp.tool()
    async def archive_coverage() -> str:
        async def _run() -> str:
            from polylogue.lib.coverage import analyze_coverage

            summaries = await hooks.get_repo().list_summaries()
            coverage = analyze_coverage(summaries)
            return json.dumps(
                {
                    "total_conversations": coverage.total_conversations,
                    "total_messages": coverage.total_messages,
                    "provider_counts": coverage.provider_counts,
                    "provider_ranges": [
                        {
                            "provider": r.provider,
                            "first_date": r.first_date.isoformat(),
                            "last_date": r.last_date.isoformat(),
                            "count": r.count,
                        }
                        for r in coverage.provider_ranges
                    ],
                    "gaps": [
                        {
                            "start_date": g.start_date.isoformat(),
                            "end_date": g.end_date.isoformat(),
                            "days": g.days,
                        }
                        for g in coverage.gaps
                    ],
                    "truncated_sessions": coverage.truncated_sessions,
                    "date_range": [d.isoformat() if d else None for d in coverage.date_range],
                },
                indent=2,
            )

        return await hooks.async_safe_call("archive_coverage", _run)


__all__ = ["register_tools"]
