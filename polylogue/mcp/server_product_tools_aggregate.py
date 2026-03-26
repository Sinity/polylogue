"""Aggregate and analytics MCP archive-product tools."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from polylogue.mcp.payloads import MCPRootPayload

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


def register_aggregate_product_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
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
                MCPRootPayload(root={"count": len(products), "items": [product.model_dump(mode="json") for product in products]})
            )

        return await hooks.async_safe_call("provider_analytics", run)

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
                MCPRootPayload(root={"count": len(products), "items": [product.model_dump(mode="json") for product in products]})
            )

        return await hooks.async_safe_call("session_tag_rollups", run)

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
                MCPRootPayload(root={"count": len(products), "items": [product.model_dump(mode="json") for product in products]})
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
                MCPRootPayload(root={"count": len(products), "items": [product.model_dump(mode="json") for product in products]})
            )

        return await hooks.async_safe_call("week_session_summaries", run)

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


__all__ = ["register_aggregate_product_tools"]
