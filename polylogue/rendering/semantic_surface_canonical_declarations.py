"""Canonical and export-surface semantic proof declarations."""

from __future__ import annotations

from polylogue.rendering.semantic_surface_canonical_contracts import CANONICAL_MARKDOWN_CONTRACTS
from polylogue.rendering.semantic_surface_export_document_contracts import (
    EXPORT_HTML_CONTRACTS,
    EXPORT_MARKDOWN_CONTRACTS,
    EXPORT_OBSIDIAN_CONTRACTS,
    EXPORT_ORG_CONTRACTS,
)
from polylogue.rendering.semantic_surface_export_structured_contracts import (
    EXPORT_CSV_CONTRACTS,
    EXPORT_JSON_LIKE_CONTRACTS,
)
from polylogue.rendering.semantic_surface_models import SemanticSurfaceSpec

CANONICAL_SEMANTIC_SURFACE_SPECS: tuple[SemanticSurfaceSpec, ...] = (
    SemanticSurfaceSpec(
        "canonical_markdown_v1",
        "canonical",
        aliases=("canonical", "canonical_markdown"),
        contracts=CANONICAL_MARKDOWN_CONTRACTS,
    ),
)

EXPORT_SEMANTIC_SURFACE_SPECS: tuple[SemanticSurfaceSpec, ...] = (
    SemanticSurfaceSpec("export_json_v1", "export", aliases=("json",), export_format="json", contracts=EXPORT_JSON_LIKE_CONTRACTS),
    SemanticSurfaceSpec("export_yaml_v1", "export", aliases=("yaml",), export_format="yaml", contracts=EXPORT_JSON_LIKE_CONTRACTS),
    SemanticSurfaceSpec("export_csv_v1", "export", aliases=("csv",), export_format="csv", contracts=EXPORT_CSV_CONTRACTS),
    SemanticSurfaceSpec("export_markdown_v1", "export", aliases=("markdown",), export_format="markdown", contracts=EXPORT_MARKDOWN_CONTRACTS),
    SemanticSurfaceSpec("export_html_v1", "export", aliases=("html",), export_format="html", contracts=EXPORT_HTML_CONTRACTS),
    SemanticSurfaceSpec("export_obsidian_v1", "export", aliases=("obsidian",), export_format="obsidian", contracts=EXPORT_OBSIDIAN_CONTRACTS),
    SemanticSurfaceSpec("export_org_v1", "export", aliases=("org",), export_format="org", contracts=EXPORT_ORG_CONTRACTS),
)

__all__ = [
    "CANONICAL_SEMANTIC_SURFACE_SPECS",
    "EXPORT_SEMANTIC_SURFACE_SPECS",
]
