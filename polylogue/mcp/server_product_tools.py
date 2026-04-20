"""Registry-driven MCP archive-product tool registration.

Iterates PRODUCT_REGISTRY and registers a ``list_<name>`` MCP tool for
each product type. Special one-off tools (archive_coverage, single-item
lookups) are registered directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.mcp.payloads import MCPRootPayload
from polylogue.mcp.product_tool_contracts import ProductListToolSpec
from polylogue.products.registry import (
    PRODUCT_REGISTRY,
    ProductType,
    fetch_products_async,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


def _register_list_tool(
    mcp: FastMCP,
    hooks: ServerCallbacks,
    pt: ProductType,
) -> None:
    """Register one list-style MCP tool for a product type."""
    spec = ProductListToolSpec.from_product_type(pt)

    async def tool_fn(**kwargs: object) -> str:
        async def run() -> str:
            ops = hooks.get_archive_ops()
            normalized_kwargs = spec.normalize_kwargs(hooks.clamp_limit, kwargs)
            products = await fetch_products_async(pt, ops, **normalized_kwargs)
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "count": len(products),
                        "items": [product.model_dump(mode="json") for product in products],
                    }
                )
            )

        return await hooks.async_safe_call(pt.name, run)

    async def wrapper(**kw: object) -> str:
        return await tool_fn(**kw)

    wrapper.__name__ = spec.name
    wrapper.__qualname__ = spec.name
    wrapper.__doc__ = spec.doc

    wrapper.__annotations__ = spec.signature.annotations
    wrapper.__kwdefaults__ = spec.signature.kwdefaults

    mcp.tool()(wrapper)


def register_product_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    """Register all product-type list tools plus special tools."""

    # Register generic list tools from the registry
    for pt in PRODUCT_REGISTRY.values():
        if pt.query_model is not None and pt.operations_method_name:
            _register_list_tool(mcp, hooks, pt)

    # --- Special tools ---

    @mcp.tool()
    async def session_profile(conversation_id: str, tier: str = "merged") -> str:
        """Get a single session profile by conversation ID."""

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
    async def archive_coverage() -> str:
        """Show archive coverage statistics."""

        async def run() -> str:
            from polylogue.lib.coverage import analyze_coverage

            summaries = await hooks.get_query_store().list_summaries()
            coverage = analyze_coverage(summaries)
            return hooks.json_payload(
                MCPRootPayload(
                    root={
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
                    }
                )
            )

        return await hooks.async_safe_call("archive_coverage", run)


__all__ = ["register_product_tools"]
