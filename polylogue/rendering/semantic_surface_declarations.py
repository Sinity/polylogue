"""Declared semantic-proof surface catalog."""

from __future__ import annotations

from polylogue.rendering.semantic_surface_aliases import build_surface_aliases
from polylogue.rendering.semantic_surface_canonical_declarations import (
    CANONICAL_SEMANTIC_SURFACE_SPECS,
    EXPORT_SEMANTIC_SURFACE_SPECS,
)
from polylogue.rendering.semantic_surface_mcp_declarations import (
    MCP_SEMANTIC_SURFACE_SPECS,
)
from polylogue.rendering.semantic_surface_models import SemanticMetricContract, SemanticSurfaceSpec
from polylogue.rendering.semantic_surface_query_declarations import (
    QUERY_SEMANTIC_SURFACE_SPECS,
)

SEMANTIC_SURFACE_SPECS: tuple[SemanticSurfaceSpec, ...] = (
    *CANONICAL_SEMANTIC_SURFACE_SPECS,
    *EXPORT_SEMANTIC_SURFACE_SPECS,
    *QUERY_SEMANTIC_SURFACE_SPECS,
    *MCP_SEMANTIC_SURFACE_SPECS,
)

(
    DEFAULT_SEMANTIC_SURFACES,
    SURFACE_ALIASES,
    EXPORT_SURFACE_FORMATS,
    STREAM_SURFACE_FORMATS,
) = build_surface_aliases(SEMANTIC_SURFACE_SPECS)


__all__ = [
    "DEFAULT_SEMANTIC_SURFACES",
    "EXPORT_SURFACE_FORMATS",
    "SEMANTIC_SURFACE_SPECS",
    "STREAM_SURFACE_FORMATS",
    "SURFACE_ALIASES",
    "SemanticMetricContract",
    "SemanticSurfaceSpec",
]
