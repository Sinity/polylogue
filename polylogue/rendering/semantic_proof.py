"""Semantic preservation proofing for render, export, query, stream, and MCP surfaces."""

from __future__ import annotations

from polylogue.rendering.semantic_proof_models import (
    ProviderSemanticProof,
    SemanticConversationProof,
    SemanticMetricCheck,
    SemanticProofReport,
    SemanticProofSuiteReport,
)
from polylogue.rendering.semantic_proof_suite import (
    prove_markdown_render_semantics,
    prove_semantic_surface_suite,
)
from polylogue.rendering.semantic_proof_surface_exports import (
    prove_export_surface_semantics,
    prove_markdown_projection_semantics,
)
from polylogue.rendering.semantic_surface_registry import (
    DEFAULT_SEMANTIC_SURFACES,
    list_semantic_surface_specs,
    resolve_semantic_surfaces,
)

__all__ = [
    "DEFAULT_SEMANTIC_SURFACES",
    "ProviderSemanticProof",
    "SemanticConversationProof",
    "SemanticMetricCheck",
    "SemanticProofReport",
    "SemanticProofSuiteReport",
    "list_semantic_surface_specs",
    "prove_export_surface_semantics",
    "prove_markdown_projection_semantics",
    "prove_markdown_render_semantics",
    "prove_semantic_surface_suite",
    "resolve_semantic_surfaces",
]
