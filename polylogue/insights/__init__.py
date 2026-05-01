"""Data-driven derived product system.

Insight types are registered descriptors. CLI rendering, MCP exposure,
and library API are generic — they take a product type name and produce
appropriate output. No per-product-type rendering/workflow/command files.
"""

from polylogue.insights.registry import (
    INSIGHT_REGISTRY,
    CliOption,
    InsightField,
    InsightType,
    fetch_insights,
    fetch_insights_async,
    get_insight_type,
    list_insight_types,
    render_insight_items,
)

__all__ = [
    "CliOption",
    "INSIGHT_REGISTRY",
    "InsightField",
    "InsightType",
    "fetch_insights",
    "fetch_insights_async",
    "get_insight_type",
    "list_insight_types",
    "render_insight_items",
]
