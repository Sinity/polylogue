"""Registry for generated repository surfaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from devtools import (
    render_agents,
    render_cli_reference,
    render_devtools_reference,
    render_docs_surface,
    render_quality_reference,
    render_topology_status,
    render_verification_catalog,
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


GENERATED_SURFACES: tuple[GeneratedSurface, ...] = (
    GeneratedSurface(
        name="agents",
        label="AGENTS",
        description="Render AGENTS.md from the root CLAUDE transclusion surface.",
        command=control_plane_argv("render-agents"),
        main=render_agents.main,
    ),
    GeneratedSurface(
        name="cli-reference",
        label="CLI docs",
        description="Render docs/cli-reference.md from live CLI help.",
        command=control_plane_argv("render-cli-reference"),
        main=render_cli_reference.main,
    ),
    GeneratedSurface(
        name="devtools-reference",
        label="Devtools docs",
        description="Render the generated command catalog inside docs/devtools.md.",
        command=control_plane_argv("render-devtools-reference"),
        main=render_devtools_reference.main,
    ),
    GeneratedSurface(
        name="quality-reference",
        label="Quality docs",
        description="Render docs/test-quality-workflows.md from quality registries.",
        command=control_plane_argv("render-quality-reference"),
        main=render_quality_reference.main,
    ),
    GeneratedSurface(
        name="verification-catalog",
        label="Verification catalog",
        description="Render docs/verification-catalog.md from proof-obligation registries.",
        command=control_plane_argv("render-verification-catalog"),
        main=render_verification_catalog.main,
    ),
    GeneratedSurface(
        name="docs-surface",
        label="Docs surface",
        description="Render docs/README.md and the generated docs table in README.md.",
        command=control_plane_argv("render-docs-surface"),
        main=render_docs_surface.main,
    ),
    GeneratedSurface(
        name="topology-status",
        label="Topology status",
        description="Render docs/topology-status.md from the topology projection and realized tree.",
        command=control_plane_argv("render-topology-status"),
        main=render_topology_status.main,
    ),
)

GENERATED_SURFACE_BY_NAME = {surface.name: surface for surface in GENERATED_SURFACES}


__all__ = ["GENERATED_SURFACES", "GENERATED_SURFACE_BY_NAME", "GeneratedSurface"]
