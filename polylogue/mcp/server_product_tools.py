"""Registry-driven MCP archive-product tool registration.

Iterates PRODUCT_REGISTRY and registers a ``list_<name>`` MCP tool for
each product type. Special one-off tools (archive_coverage, single-item
lookups) are registered directly.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import TYPE_CHECKING

from polylogue.mcp.payloads import MCPRootPayload
from polylogue.products.registry import (
    PRODUCT_REGISTRY,
    ProductType,
    fetch_products_async,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


def _query_field_defaults(pt: ProductType) -> tuple[dict[str, object | None], dict[str, object]]:
    query_model = pt.query_model
    if query_model is None:
        raise ValueError(f"Product type {pt.name} does not declare a query model")

    defaults: dict[str, object | None] = {}
    annotations: dict[str, object] = {}
    for field_name, field_info in query_model.model_fields.items():
        defaults[field_name] = None if field_info.is_required() else field_info.get_default(call_default_factory=True)
        annotations[field_name] = field_info.annotation if field_info.annotation is not None else object
    return defaults, annotations


def _sanitize_offset(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, str, bytes, bytearray)):
        return max(0, int(value))
    return 0


def _normalize_list_tool_kwargs(
    hooks: ServerCallbacks,
    kwargs: Mapping[str, object],
) -> dict[str, object]:
    normalized = dict(kwargs)
    if "limit" in normalized:
        normalized["limit"] = hooks.clamp_limit(normalized["limit"])
    if "offset" in normalized:
        normalized["offset"] = _sanitize_offset(normalized["offset"])
    return normalized


def _build_list_tool_parameters(
    pt: ProductType,
    *,
    field_defaults: dict[str, object | None],
    field_annotations: dict[str, object],
) -> list[inspect.Parameter]:
    params: list[inspect.Parameter] = []
    for field_name in sorted(field_annotations):
        if field_name == "limit":
            params.append(
                inspect.Parameter(
                    field_name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=pt.mcp_default_limit,
                    annotation=int,
                )
            )
            continue
        if field_name == "offset":
            params.append(
                inspect.Parameter(
                    field_name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=0,
                    annotation=int,
                )
            )
            continue
        params.append(
            inspect.Parameter(
                field_name,
                inspect.Parameter.KEYWORD_ONLY,
                default=field_defaults.get(field_name),
                annotation=field_annotations[field_name],
            )
        )
    return params


def _wrapper_annotations(params: list[inspect.Parameter]) -> dict[str, object]:
    annotations: dict[str, object] = {parameter.name: parameter.annotation for parameter in params}
    annotations["return"] = str
    return annotations


def _register_list_tool(
    mcp: FastMCP,
    hooks: ServerCallbacks,
    pt: ProductType,
) -> None:
    """Register one list-style MCP tool for a product type."""

    query_model = pt.query_model
    if query_model is None:
        raise ValueError(f"Product type {pt.name} does not declare a query model")
    field_defaults, field_annotations = _query_field_defaults(pt)
    params = _build_list_tool_parameters(
        pt,
        field_defaults=field_defaults,
        field_annotations=field_annotations,
    )

    async def tool_fn(**kwargs: object) -> str:
        async def run() -> str:
            ops = hooks.get_archive_ops()
            normalized_kwargs = _normalize_list_tool_kwargs(hooks, kwargs)
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

    wrapper.__name__ = pt.name
    wrapper.__qualname__ = pt.name
    wrapper.__doc__ = f"List {pt.display_name.lower()} from the archive."

    wrapper.__annotations__ = _wrapper_annotations(params)
    wrapper.__kwdefaults__ = {parameter.name: parameter.default for parameter in params}

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
