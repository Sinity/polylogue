"""Surface-specific semantic-proof functions."""

from __future__ import annotations

from polylogue.rendering.semantic_proof_surface_exports import (
    prove_export_surface_semantics,
    prove_markdown_projection_semantics,
)
from polylogue.rendering.semantic_proof_surface_reads import (
    _prove_mcp_detail_surface,
    _prove_mcp_summary_surface,
    _prove_query_stream_json_lines_surface,
    _prove_query_stream_markdown_surface,
    _prove_query_stream_plaintext_surface,
    _prove_query_summary_csv_surface,
    _prove_query_summary_json_like_surface,
    _prove_query_summary_text_surface,
)

__all__ = [
    "_prove_mcp_detail_surface",
    "_prove_mcp_summary_surface",
    "_prove_query_stream_json_lines_surface",
    "_prove_query_stream_markdown_surface",
    "_prove_query_stream_plaintext_surface",
    "_prove_query_summary_csv_surface",
    "_prove_query_summary_json_like_surface",
    "_prove_query_summary_text_surface",
    "prove_export_surface_semantics",
    "prove_markdown_projection_semantics",
]
