"""Data-driven derived insight system.

Insight types are registered descriptors. CLI rendering, MCP exposure,
and library API are generic: they take an insight type name and produce
appropriate output. No per-insight-type rendering/workflow/command files.

Re-exports are lazy (PEP 562 module ``__getattr__``, polylogue-8s70): a caller
that only needs a specific submodule -- e.g. ``polylogue.insights.archive``
for its ``date_from_iso`` helper -- reaches it by importing a submodule of
this package, which Python resolves by running this ``__init__`` first. The
eager form here forced ``insights.registry`` (which builds the full
``INSIGHT_REGISTRY`` -- ~20 pydantic insight models plus their CLI/MCP
metadata) to load in full for that single unrelated helper.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


def __getattr__(name: str) -> object:
    lazy_exports = {
        "INSIGHT_REGISTRY",
        "CliOption",
        "InsightField",
        "InsightType",
        "fetch_insights",
        "fetch_insights_async",
        "get_insight_type",
        "list_insight_types",
        "render_insight_items",
    }
    if name in lazy_exports:
        module = importlib.import_module("polylogue.insights.registry")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
