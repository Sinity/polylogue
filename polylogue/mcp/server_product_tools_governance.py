"""Maintenance and debt MCP archive-product tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.mcp.payloads import MCPRootPayload

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


def register_governance_product_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
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
                MCPRootPayload(root={"count": len(products), "items": [product.model_dump(mode="json") for product in products]})
            )

        return await hooks.async_safe_call("archive_debt", run)

    @mcp.tool()
    async def maintenance_runs(limit: int = 20) -> str:
        async def run() -> str:
            from polylogue.archive_products import MaintenanceRunProductQuery

            products = await hooks.get_archive_ops().list_maintenance_run_products(
                MaintenanceRunProductQuery(limit=hooks.clamp_limit(limit))
            )
            return hooks.json_payload(
                MCPRootPayload(root={"count": len(products), "items": [product.model_dump(mode="json") for product in products]})
            )

        return await hooks.async_safe_call("maintenance_runs", run)


__all__ = ["register_governance_product_tools"]
