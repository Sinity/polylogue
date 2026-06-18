"""Registry for generated repository surfaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from devtools import (
    render_agents,
    render_cli_output_schemas,
    render_cli_reference,
    render_devtools_reference,
    render_docs_surface,
    render_openapi,
    render_pages,
    render_quality_reference,
    render_topology_status,
)
from devtools.command_catalog import control_plane_argv

SurfaceMain = Callable[[list[str] | None], int]


@dataclass(frozen=True, slots=True)
class GeneratedSurface:
    name: str
    label: str
    description: str
    command: tuple[str, ...]
    main: SurfaceMain
    inputs: tuple[str, ...] = ()  # glob patterns whose content hash invalidates the render


GENERATED_SURFACES: tuple[GeneratedSurface, ...] = (
    GeneratedSurface(
        name="agents",
        label="AGENTS",
        description="Render AGENTS.md from the root CLAUDE transclusion surface.",
        command=control_plane_argv("render-agents"),
        main=render_agents.main,
        inputs=(
            "devtools/render_agents.py",
            "CLAUDE.md",
            "CONTRIBUTING.md",
            "TESTING.md",
            "docs/architecture.md",
            "docs/internals.md",
            "docs/devtools.md",
        ),
    ),
    GeneratedSurface(
        name="cli-reference",
        label="CLI docs",
        description="Render docs/cli-reference.md from live CLI help and action-contract metadata.",
        command=control_plane_argv("render-cli-reference"),
        main=render_cli_reference.main,
        inputs=(
            "polylogue/cli/",
            "polylogue/surfaces/payloads.py",
            "devtools/render_cli_reference.py",
            "devtools/action_contract_report.py",
            "devtools/render_cli_output_schemas.py",
        ),
    ),
    GeneratedSurface(
        name="cli-output-schemas",
        label="CLI output schemas",
        description="Render JSON Schema artifacts for stable CLI output payloads (#1272).",
        command=control_plane_argv("render-cli-output-schemas"),
        main=render_cli_output_schemas.main,
        inputs=("polylogue/surfaces/payloads.py", "devtools/render_cli_output_schemas.py"),
    ),
    GeneratedSurface(
        name="openapi",
        label="OpenAPI schema",
        description="Render docs/openapi/search.yaml from typed daemon query payload models.",
        command=control_plane_argv("render-openapi"),
        main=render_openapi.main,
        inputs=("polylogue/surfaces/payloads.py", "devtools/render_openapi.py"),
    ),
    GeneratedSurface(
        name="devtools-reference",
        label="Devtools docs",
        description="Render the generated command catalog inside docs/devtools.md.",
        command=control_plane_argv("render-devtools-reference"),
        main=render_devtools_reference.main,
        inputs=("devtools/command_catalog.py", "devtools/render_devtools_reference.py"),
    ),
    GeneratedSurface(
        name="quality-reference",
        label="Quality docs",
        description="Render docs/test-quality-workflows.md from quality registries.",
        command=control_plane_argv("render-quality-reference"),
        main=render_quality_reference.main,
        inputs=("devtools/render_quality_reference.py", "devtools/run_validation_lanes.py", "pyproject.toml"),
    ),
    GeneratedSurface(
        name="docs-surface",
        label="Docs surface",
        description="Render docs/README.md and the generated docs table in README.md.",
        command=control_plane_argv("render-docs-surface"),
        main=render_docs_surface.main,
        inputs=("devtools/render_docs_surface.py", "docs/", "README.md"),
    ),
    GeneratedSurface(
        name="topology-status",
        label="Topology status",
        description="Render docs/topology-status.md from the topology projection and realized tree.",
        command=control_plane_argv("render-topology-status"),
        main=render_topology_status.main,
        inputs=("devtools/render_topology_status.py", "docs/plans/topology-target.yaml"),
    ),
    GeneratedSurface(
        name="pages",
        label="GitHub Pages",
        description="Build the GitHub Pages documentation site into .cache/site/.",
        command=control_plane_argv("render-pages"),
        main=render_pages.main,
        inputs=("devtools/render_pages.py", "docs/", "README.md", "polylogue/", "pyproject.toml"),
    ),
)

GENERATED_SURFACE_BY_NAME = {surface.name: surface for surface in GENERATED_SURFACES}


__all__ = ["GENERATED_SURFACES", "GENERATED_SURFACE_BY_NAME", "GeneratedSurface"]
