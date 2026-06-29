"""Authored insight-query surface families built on shared CLI surface specs."""

from __future__ import annotations

from .cli_surfaces import (
    CliSurfaceFamily,
    CliSurfaceVariant,
    CompiledCliSurface,
    build_cli_surface_contracts,
    build_cli_surface_live_variants,
)

INSIGHT_SURFACE_FAMILIES: tuple[CliSurfaceFamily, ...] = (
    CliSurfaceFamily(
        slug="profiles",
        command_args=("analyze", "insights", "profiles"),
        tags=("insights", "session-profiles"),
        contract_variant=CliSurfaceVariant(
            name="json-insights-profiles",
            description="insights profiles JSON contract",
        ),
        live_variants=(
            CliSurfaceVariant(
                name="live-insights-profiles-evidence",
                description="Live archive evidence-tier session-profile insight surface",
                suffix_args=("--tier", "evidence", "--limit", "3", "--format", "json"),
                env="any",
            ),
            CliSurfaceVariant(
                name="live-insights-profiles-inference",
                description="Live archive inference-tier session-profile insight surface",
                suffix_args=("--tier", "inference", "--limit", "3", "--format", "json"),
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="work-events",
        command_args=("analyze", "insights", "work-events"),
        tags=("insights", "work-events"),
        contract_variant=CliSurfaceVariant(
            name="json-insights-work-events",
            description="insights work-events JSON contract",
        ),
        live_variants=(
            CliSurfaceVariant(
                name="live-insights-work-events",
                description="Live archive inferred work-event insight surface",
                suffix_args=("--limit", "3", "--format", "json"),
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="phases",
        command_args=("analyze", "insights", "phases"),
        tags=("insights", "phases"),
        contract_variant=CliSurfaceVariant(
            name="json-insights-phases",
            description="insights phases JSON contract",
        ),
        live_variants=(
            CliSurfaceVariant(
                name="live-insights-phases",
                description="Live archive phase interval evidence surface",
                suffix_args=("--limit", "3", "--format", "json"),
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="threads",
        command_args=("analyze", "insights", "threads"),
        tags=("insights", "threads"),
        contract_variant=CliSurfaceVariant(
            name="json-insights-threads",
            description="insights threads JSON contract",
        ),
    ),
    CliSurfaceFamily(
        slug="tags",
        command_args=("analyze", "insights", "tags"),
        tags=("insights", "tags"),
        contract_variant=CliSurfaceVariant(
            name="json-insights-tags",
            description="insights tags JSON contract",
        ),
        live_variants=(
            CliSurfaceVariant(
                name="live-insights-tags",
                description="Live archive tag-rollup insight view",
                suffix_args=("--limit", "20", "--format", "json"),
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="coverage",
        command_args=("analyze", "insights", "coverage"),
        tags=("insights", "coverage"),
        contract_variant=CliSurfaceVariant(
            name="json-insights-coverage",
            description="insights coverage JSON contract",
        ),
        live_variants=(
            CliSurfaceVariant(
                name="live-insights-coverage-provider",
                description="Live archive provider coverage insight surface",
                suffix_args=("--limit", "20", "--format", "json"),
                env="any",
            ),
            CliSurfaceVariant(
                name="live-insights-coverage-day",
                description="Live archive day coverage insight surface over the recent semantic slice",
                prefix_args=("--provider", "claude-code", "--since", "2026-03-01"),
                suffix_args=("--group-by", "day", "--limit", "14", "--format", "json"),
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="status",
        command_args=("ops", "insights", "status"),
        tags=("insights", "status"),
        live_variants=(
            CliSurfaceVariant(
                name="live-insights-status",
                description="Live archive insight status view",
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="debt",
        command_args=("analyze", "insights", "debt"),
        tags=("insights", "debt"),
        live_variants=(
            CliSurfaceVariant(
                name="live-insights-debt",
                description="Live archive debt and cleanup insight view",
                suffix_args=("--limit", "20", "--format", "json"),
                env="any",
            ),
        ),
    ),
)


def build_insight_contract_surfaces() -> tuple[CompiledCliSurface, ...]:
    return build_cli_surface_contracts(INSIGHT_SURFACE_FAMILIES)


def build_live_insight_surface_lanes() -> tuple[CompiledCliSurface, ...]:
    return build_cli_surface_live_variants(INSIGHT_SURFACE_FAMILIES)


__all__ = [
    "INSIGHT_SURFACE_FAMILIES",
    "build_insight_contract_surfaces",
    "build_live_insight_surface_lanes",
]
