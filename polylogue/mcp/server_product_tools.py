"""Archive-product MCP tool registration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from polylogue.mcp.payloads import MCPRootPayload

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


def register_product_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def session_profile(conversation_id: str, tier: str = "merged") -> str:
        async def run() -> str:
            product = await hooks.get_archive_ops().get_session_profile_product(
                conversation_id,
                tier=tier,
            )
            if product is None:
                return hooks.error_json("Conversation not found", conversation_id=conversation_id)
            return hooks.json_payload(product, exclude_none=True)

        return await hooks.async_safe_call("session_profile", run)

    @mcp.tool()
    async def provider_analytics(
        provider: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        async def run() -> str:
            from polylogue.archive_products import ProviderAnalyticsProductQuery

            products = await hooks.get_archive_ops().list_provider_analytics_products(
                ProviderAnalyticsProductQuery(
                    provider=provider,
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

        return await hooks.async_safe_call("provider_analytics", run)

    @mcp.tool()
    async def archive_debt(
        category: str | None = None,
        actionable_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        async def run() -> str:
            from polylogue.archive_products import ArchiveDebtProductQuery

            products = await hooks.get_archive_ops().list_archive_debt_products(
                ArchiveDebtProductQuery(
                    category=category,
                    only_actionable=bool(actionable_only),
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

        return await hooks.async_safe_call("archive_debt", run)

    @mcp.tool()
    async def session_profiles(
        since: str | None = None,
        until: str | None = None,
        first_message_since: str | None = None,
        first_message_until: str | None = None,
        session_date_since: str | None = None,
        session_date_until: str | None = None,
        tier: str = "merged",
        provider: str | None = None,
        query: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        async def run() -> str:
            from polylogue.archive_products import SessionProfileProductQuery

            products = await hooks.get_archive_ops().list_session_profile_products(
                SessionProfileProductQuery(
                    provider=provider,
                    since=since,
                    until=until,
                    first_message_since=first_message_since,
                    first_message_until=first_message_until,
                    session_date_since=session_date_since,
                    session_date_until=session_date_until,
                    tier=tier,
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

        return await hooks.async_safe_call("session_profiles", run)

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
        async def run() -> str:
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

        return await hooks.async_safe_call("session_work_events", run)

    @mcp.tool()
    async def session_phases(
        conversation_id: str | None = None,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        kind: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        async def run() -> str:
            from polylogue.archive_products import SessionPhaseProductQuery

            ops = hooks.get_archive_ops()
            if conversation_id:
                products = await ops.get_session_phase_products(conversation_id)
            else:
                products = await ops.list_session_phase_products(
                    SessionPhaseProductQuery(
                        provider=provider,
                        since=since,
                        until=until,
                        kind=kind,
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

        return await hooks.async_safe_call("session_phases", run)

    @mcp.tool()
    async def session_tag_rollups(
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        query: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> str:
        async def run() -> str:
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

        return await hooks.async_safe_call("session_tag_rollups", run)

    @mcp.tool()
    async def work_threads(
        since: str | None = None,
        until: str | None = None,
        query: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        async def run() -> str:
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

        return await hooks.async_safe_call("work_threads", run)

    @mcp.tool()
    async def day_session_summaries(
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 90,
        offset: int = 0,
    ) -> str:
        async def run() -> str:
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

        return await hooks.async_safe_call("day_session_summaries", run)

    @mcp.tool()
    async def week_session_summaries(
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 52,
        offset: int = 0,
    ) -> str:
        async def run() -> str:
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

        return await hooks.async_safe_call("week_session_summaries", run)

    @mcp.tool()
    async def maintenance_runs(limit: int = 20) -> str:
        async def run() -> str:
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

        return await hooks.async_safe_call("maintenance_runs", run)

    @mcp.tool()
    async def archive_coverage() -> str:
        async def run() -> str:
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

        return await hooks.async_safe_call("archive_coverage", run)


__all__ = ["register_product_tools"]
