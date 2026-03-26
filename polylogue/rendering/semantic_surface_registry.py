"""Semantic surface registry: declaration lookup, alias expansion, and evaluation."""

from __future__ import annotations

from polylogue.rendering.semantic_surface_declarations import (
    DEFAULT_SEMANTIC_SURFACES,
    EXPORT_SURFACE_FORMATS,
    SEMANTIC_SURFACE_SPECS,
    STREAM_SURFACE_FORMATS,
    SURFACE_ALIASES,
    SemanticMetricContract,
    SemanticSurfaceSpec,
)
from polylogue.rendering.semantic_surface_evaluation import evaluate_contracts

_SPECS_BY_NAME = {spec.name: spec for spec in SEMANTIC_SURFACE_SPECS}


def list_semantic_surface_specs() -> tuple[SemanticSurfaceSpec, ...]:
    return SEMANTIC_SURFACE_SPECS


def semantic_surface_spec(name: str) -> SemanticSurfaceSpec:
    return _SPECS_BY_NAME[name]


def evaluate_semantic_contracts(
    surface: str,
    input_facts: dict[str, object],
    output_facts: dict[str, object],
):
    return evaluate_contracts(
        semantic_surface_spec(surface).contracts,
        input_facts,
        output_facts,
    )


def resolve_semantic_surfaces(surfaces: list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize semantic-proof surface filters to canonical surface names."""
    if not surfaces:
        return list(DEFAULT_SEMANTIC_SURFACES)

    resolved: list[str] = []
    seen: set[str] = set()
    for surface in surfaces:
        token = str(surface).strip().lower().replace("-", "_")
        aliases = SURFACE_ALIASES.get(token)
        if aliases is None:
            raise ValueError(
                "Unknown semantic surface "
                f"{surface!r}. Valid values: {', '.join(sorted(SURFACE_ALIASES))}"
            )
        for alias in aliases:
            if alias not in seen:
                seen.add(alias)
                resolved.append(alias)
    return resolved


__all__ = [
    "DEFAULT_SEMANTIC_SURFACES",
    "EXPORT_SURFACE_FORMATS",
    "SEMANTIC_SURFACE_SPECS",
    "STREAM_SURFACE_FORMATS",
    "SURFACE_ALIASES",
    "SemanticMetricContract",
    "SemanticSurfaceSpec",
    "evaluate_semantic_contracts",
    "list_semantic_surface_specs",
    "resolve_semantic_surfaces",
    "semantic_surface_spec",
]
