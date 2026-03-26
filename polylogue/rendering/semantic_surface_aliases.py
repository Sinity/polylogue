"""Alias and format-map helpers for semantic surface declarations."""

from __future__ import annotations

from polylogue.rendering.semantic_surface_models import SemanticSurfaceSpec


def build_surface_aliases(
    specs: tuple[SemanticSurfaceSpec, ...],
) -> tuple[
    tuple[str, ...],
    dict[str, tuple[str, ...]],
    dict[str, str],
    dict[str, str],
]:
    default_surfaces = tuple(spec.name for spec in specs)
    surface_aliases: dict[str, tuple[str, ...]] = {
        "all": default_surfaces,
        "canonical": ("canonical_markdown_v1",),
        "canonical_markdown": ("canonical_markdown_v1",),
        "canonical_markdown_v1": ("canonical_markdown_v1",),
        "query_summary_all": tuple(spec.name for spec in specs if spec.category == "query_summary"),
        "stream_all": tuple(spec.name for spec in specs if spec.category == "query_stream"),
        "query_all": tuple(
            spec.name for spec in specs if spec.category in {"query_summary", "query_stream"}
        ),
        "mcp_all": tuple(spec.name for spec in specs if spec.category == "mcp"),
        "read_all": tuple(
            spec.name for spec in specs if spec.category in {"query_summary", "query_stream", "mcp"}
        ),
        "export_all": tuple(spec.name for spec in specs if spec.category == "export"),
    }

    for spec in specs:
        surface_aliases[spec.name] = (spec.name,)
        for alias in spec.aliases:
            surface_aliases[alias] = (spec.name,)

    export_surface_formats = {
        spec.name: spec.export_format
        for spec in specs
        if spec.export_format is not None
    }
    stream_surface_formats = {
        spec.name: spec.stream_format
        for spec in specs
        if spec.stream_format is not None
    }
    return default_surfaces, surface_aliases, export_surface_formats, stream_surface_formats


__all__ = ["build_surface_aliases"]
