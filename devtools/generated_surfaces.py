"""Registry for generated repository surfaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from devtools import (
    render_agents,
    render_cli_output_schemas,
    render_cli_reference,
    render_demo_corpus_datasheet,
    render_devtools_reference,
    render_docs_surface,
    render_openapi,
    render_pages,
    render_product_workflows,
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
        command=control_plane_argv("render agents"),
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
        command=control_plane_argv("render cli-reference"),
        main=render_cli_reference.main,
        inputs=(
            "polylogue/cli/",
            "polylogue/cli/click_command_registration.py",
            "polylogue/cli/command_inventory.py",
            "polylogue/cli/query_group.py",
            "polylogue/archive/query/",
            "polylogue/archive/query/metadata.py",
            "polylogue/archive/query/fields.py",
            "polylogue/archive/query/unit_results.py",
            "polylogue/archive/viewport/",
            "polylogue/archive/viewport/profiles.py",
            "polylogue/operations/action_contracts.py",
            "polylogue/surfaces/action_affordances.py",
            "polylogue/surfaces/payloads.py",
            "polylogue/sources/provider_completeness.py",
            "devtools/render_cli_reference.py",
            "devtools/action_contract_report.py",
            "devtools/render_cli_output_schemas.py",
            "devtools/provider_completeness.py",
        ),
    ),
    GeneratedSurface(
        name="cli-output-schemas",
        label="CLI output schemas",
        description="Render JSON Schema artifacts for stable CLI output payloads (#1272).",
        command=control_plane_argv("render cli-output-schemas"),
        main=render_cli_output_schemas.main,
        inputs=(
            "polylogue/archive/query/",
            "polylogue/archive/query/metadata.py",
            "polylogue/archive/query/unit_results.py",
            "polylogue/context/compiler.py",
            "polylogue/insights/transforms.py",
            "polylogue/surfaces/action_affordances.py",
            "polylogue/surfaces/payloads.py",
            "devtools/render_cli_output_schemas.py",
        ),
    ),
    GeneratedSurface(
        name="openapi",
        label="OpenAPI schema",
        description="Render docs/openapi/search.yaml from typed daemon query payload models.",
        command=control_plane_argv("render openapi"),
        main=render_openapi.main,
        inputs=(
            "polylogue/archive/query/",
            "polylogue/archive/query/metadata.py",
            "polylogue/archive/query/unit_results.py",
            "polylogue/archive/viewport/",
            "polylogue/archive/viewport/profiles.py",
            "polylogue/browser_capture/models.py",
            "polylogue/browser_capture/route_contracts.py",
            "polylogue/daemon/",
            "polylogue/daemon/http.py",
            "polylogue/daemon/route_contracts.py",
            "polylogue/context/compiler.py",
            "polylogue/insights/transforms.py",
            "polylogue/surfaces/action_affordances.py",
            "polylogue/surfaces/payloads.py",
            "polylogue/sources/provider_completeness.py",
            "devtools/render_openapi.py",
        ),
    ),
    GeneratedSurface(
        name="devtools-reference",
        label="Devtools docs",
        description="Render the generated command catalog inside docs/devtools.md.",
        command=control_plane_argv("render devtools-reference"),
        main=render_devtools_reference.main,
        inputs=(
            "devtools/command_catalog.py",
            "devtools/provider_completeness.py",
            "polylogue/sources/provider_completeness.py",
            "devtools/render_devtools_reference.py",
        ),
    ),
    GeneratedSurface(
        name="demo-corpus-datasheet",
        label="Demo corpus datasheet",
        description="Render docs/plans/demo-corpus-construct-audit.md from declared demo families and measured seed rows.",
        command=control_plane_argv("render demo-corpus-datasheet"),
        main=render_demo_corpus_datasheet.main,
        inputs=(
            "devtools/render_demo_corpus_datasheet.py",
            "polylogue/demo/",
            "polylogue/scenarios/",
        ),
    ),
    GeneratedSurface(
        name="quality-reference",
        label="Quality docs",
        description="Render docs/test-quality-workflows.md from quality registries.",
        command=control_plane_argv("render quality-reference"),
        main=render_quality_reference.main,
        inputs=(
            "devtools/render_quality_reference.py",
            "devtools/benchmark_catalog.py",
            "devtools/mutation_catalog.py",
            "devtools/quality_registry.py",
            "devtools/run_validation_lanes.py",
            "devtools/scenario_coverage.py",
            "devtools/scenario_projection_catalog.py",
            "devtools/validation_lane_catalog_contracts.py",
            "devtools/validation_lane_catalog_live.py",
            "polylogue/operations/specs.py",
            "polylogue/scenarios/",
            "pyproject.toml",
        ),
    ),
    GeneratedSurface(
        name="product-workflows",
        label="Product workflows",
        description="Render docs/product/workflows.md from query-action workflow registries (#2305).",
        command=control_plane_argv("render product-workflows"),
        main=render_product_workflows.main,
        inputs=(
            "devtools/render_product_workflows.py",
            "polylogue/product/workflows.py",
            "polylogue/operations/action_contracts.py",
            "polylogue/surfaces/action_affordances.py",
            "polylogue/archive/viewport/profiles.py",
        ),
    ),
    GeneratedSurface(
        name="docs-surface",
        label="Docs surface",
        description="Render docs/README.md and the generated docs table in README.md.",
        command=control_plane_argv("render docs-surface"),
        main=render_docs_surface.main,
        inputs=(
            "devtools/render_docs_surface.py",
            "devtools/docs_surface.py",
            "polylogue/cli/click_command_registration.py",
            "polylogue/operations/action_contracts.py",
            "polylogue/archive/query/metadata.py",
            "polylogue/archive/viewport/profiles.py",
            "polylogue/daemon/route_contracts.py",
            "polylogue/sources/provider_completeness.py",
            "docs/",
            "README.md",
        ),
    ),
    GeneratedSurface(
        name="topology-status",
        label="Topology status",
        description="Render docs/topology-status.md from the topology projection and realized tree.",
        command=control_plane_argv("render topology-status"),
        main=render_topology_status.main,
        inputs=("devtools/render_topology_status.py", "docs/plans/topology-target.yaml"),
    ),
    GeneratedSurface(
        name="pages",
        label="GitHub Pages",
        description="Build the GitHub Pages documentation site into .cache/site/.",
        command=control_plane_argv("render pages"),
        main=render_pages.main,
        inputs=(
            "devtools/render_pages.py",
            "docs/",
            "README.md",
            "polylogue/",
            "polylogue/daemon/route_contracts.py",
            "polylogue/archive/query/metadata.py",
            "polylogue/archive/viewport/profiles.py",
            "polylogue/surfaces/payloads.py",
            "pyproject.toml",
        ),
    ),
)

GENERATED_SURFACE_BY_NAME = {surface.name: surface for surface in GENERATED_SURFACES}


__all__ = ["GENERATED_SURFACES", "GENERATED_SURFACE_BY_NAME", "GeneratedSurface"]
