"""Registry-driven MCP archive-product tool registration.

Iterates PRODUCT_REGISTRY and registers a ``list_<name>`` MCP tool for
each product type. Special one-off tools (archive_coverage, single-item
lookups) are registered directly.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from polylogue.mcp.payloads import MCPRootPayload
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

    # Build the set of accepted query field names from the query class
    from polylogue.products.registry import _resolve_query_class

    query_cls = _resolve_query_class(pt.query_class_path)
    query_fields = set(query_cls.model_fields)

    # Determine parameter defaults from the query class
    field_defaults: dict[str, Any] = {}
    for fname, finfo in query_cls.model_fields.items():
        if finfo.default is not None:
            field_defaults[fname] = finfo.default
        else:
            field_defaults[fname] = None

    async def tool_fn(**kwargs: Any) -> str:
        async def run() -> str:
            ops = hooks.get_archive_ops()
            # Apply limit clamping and offset sanitization
            if "limit" in kwargs:
                kwargs["limit"] = hooks.clamp_limit(kwargs["limit"])
            if "offset" in kwargs:
                kwargs["offset"] = max(0, int(kwargs["offset"]))
            products = await fetch_products_async(pt, ops, **kwargs)
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "count": len(products),
                        "items": [p.model_dump(mode="json") if hasattr(p, "model_dump") else p for p in products],
                    }
                )
            )

        return await hooks.async_safe_call(pt.name, run)

    # Build a dynamic function signature so FastMCP can introspect parameters.
    # We create a wrapper with explicit keyword arguments matching the query class.
    import inspect

    params = [
        inspect.Parameter("self_placeholder", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
    ]
    for fname in sorted(query_fields):
        default = field_defaults.get(fname)
        # FastMCP reads annotations from __annotations__, so we build them
        if fname == "limit":
            params.append(
                inspect.Parameter(fname, inspect.Parameter.KEYWORD_ONLY, default=pt.mcp_default_limit, annotation=int)
            )
        elif fname == "offset":
            params.append(inspect.Parameter(fname, inspect.Parameter.KEYWORD_ONLY, default=0, annotation=int))
        elif isinstance(default, str):
            params.append(inspect.Parameter(fname, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=str))
        elif isinstance(default, int):
            params.append(inspect.Parameter(fname, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=int))
        else:
            params.append(
                inspect.Parameter(fname, inspect.Parameter.KEYWORD_ONLY, default=None, annotation="str | None")
            )

    # Remove the placeholder — FastMCP doesn't need it
    params = params[1:]

    # Create a properly-signed wrapper
    async def wrapper(**kw: Any) -> str:
        return await tool_fn(**kw)

    wrapper.__name__ = pt.name
    wrapper.__qualname__ = pt.name
    wrapper.__doc__ = f"List {pt.display_name.lower()} from the archive."

    # Set annotations for FastMCP introspection
    annotations: dict[str, Any] = {}
    for p in params:
        annotations[p.name] = p.annotation
    annotations["return"] = str
    wrapper.__annotations__ = annotations

    # Set defaults
    defaults_dict = {p.name: p.default for p in params}
    wrapper.__kwdefaults__ = defaults_dict

    mcp.tool()(wrapper)


def register_product_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    """Register all product-type list tools plus special tools."""

    # Register generic list tools from the registry
    for pt in PRODUCT_REGISTRY.values():
        if pt.query_class_path and pt.operations_method:
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
