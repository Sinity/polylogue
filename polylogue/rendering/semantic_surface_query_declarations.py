"""Query-summary and query-stream semantic proof declarations."""

from __future__ import annotations

from polylogue.rendering.semantic_surface_models import SemanticSurfaceSpec
from polylogue.rendering.semantic_surface_query_stream_contracts import (
    QUERY_STREAM_JSON_LINES_CONTRACTS,
    QUERY_STREAM_MARKDOWN_CONTRACTS,
    QUERY_STREAM_PLAINTEXT_CONTRACTS,
)
from polylogue.rendering.semantic_surface_query_summary_contracts import (
    QUERY_SUMMARY_CSV_CONTRACTS,
    QUERY_SUMMARY_JSON_LIKE_CONTRACTS,
    QUERY_SUMMARY_TEXT_CONTRACTS,
)

QUERY_SEMANTIC_SURFACE_SPECS: tuple[SemanticSurfaceSpec, ...] = (
    SemanticSurfaceSpec("query_summary_json_v1", "query_summary", aliases=("query_summary_json",), contracts=QUERY_SUMMARY_JSON_LIKE_CONTRACTS),
    SemanticSurfaceSpec("query_summary_yaml_v1", "query_summary", aliases=("query_summary_yaml",), contracts=QUERY_SUMMARY_JSON_LIKE_CONTRACTS),
    SemanticSurfaceSpec("query_summary_csv_v1", "query_summary", aliases=("query_summary_csv",), contracts=QUERY_SUMMARY_CSV_CONTRACTS),
    SemanticSurfaceSpec("query_summary_text_v1", "query_summary", aliases=("query_summary_text",), contracts=QUERY_SUMMARY_TEXT_CONTRACTS),
    SemanticSurfaceSpec(
        "query_stream_plaintext_v1",
        "query_stream",
        aliases=("stream_plaintext",),
        stream_format="plaintext",
        contracts=QUERY_STREAM_PLAINTEXT_CONTRACTS,
    ),
    SemanticSurfaceSpec(
        "query_stream_markdown_v1",
        "query_stream",
        aliases=("stream_markdown",),
        stream_format="markdown",
        contracts=QUERY_STREAM_MARKDOWN_CONTRACTS,
    ),
    SemanticSurfaceSpec(
        "query_stream_json_lines_v1",
        "query_stream",
        aliases=("stream_json_lines",),
        stream_format="json-lines",
        contracts=QUERY_STREAM_JSON_LINES_CONTRACTS,
    ),
)

__all__ = ["QUERY_SEMANTIC_SURFACE_SPECS"]
