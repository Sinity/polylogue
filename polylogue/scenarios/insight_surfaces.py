"""Authored insight-query surface families built on shared CLI surface specs."""

from __future__ import annotations

from .cli_surfaces import (
    CliSurfaceFamily,
    CliSurfaceVariant,
    CompiledCliSurface,
    build_cli_surface_exercises,
    build_cli_surface_live_variants,
)

INSIGHT_SURFACE_FAMILIES: tuple[CliSurfaceFamily, ...] = (
    CliSurfaceFamily(
        slug="profiles",
        command_args=("insights", "profiles"),
        tags=("insights", "session-profiles"),
        exercise=CliSurfaceVariant(
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
        slug="enrichments",
        command_args=("insights", "enrichments"),
        tags=("insights", "enrichments"),
        live_variants=(
            CliSurfaceVariant(
                name="live-insights-enrichments",
                description="Live archive probabilistic session-enrichment insight surface",
                suffix_args=("--limit", "5", "--format", "json"),
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="work-events",
        command_args=("insights", "work-events"),
        tags=("insights", "work-events"),
        exercise=CliSurfaceVariant(
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
        command_args=("insights", "phases"),
        tags=("insights", "phases"),
        exercise=CliSurfaceVariant(
            name="json-insights-phases",
            description="insights phases JSON contract",
        ),
        live_variants=(
            CliSurfaceVariant(
                name="live-insights-phases",
                description="Live archive inferred phase insight surface",
                suffix_args=("--limit", "3", "--format", "json"),
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="threads",
        command_args=("insights", "threads"),
        tags=("insights", "threads"),
        exercise=CliSurfaceVariant(
            name="json-insights-threads",
            description="insights threads JSON contract",
        ),
    ),
    CliSurfaceFamily(
        slug="tags",
        command_args=("insights", "tags"),
        tags=("insights", "tags"),
        exercise=CliSurfaceVariant(
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
        slug="day-summaries",
        command_args=("insights", "day-summaries"),
        tags=("insights", "day-summaries"),
        exercise=CliSurfaceVariant(
            name="json-insights-day-summaries",
            description="insights day-summaries JSON contract",
        ),
        live_variants=(
            CliSurfaceVariant(
                name="live-insights-day-summaries",
                description="Live archive day-summary insight surface over the recent semantic slice",
                prefix_args=("--provider", "claude-code", "--since", "2026-03-01"),
                suffix_args=("--limit", "14", "--format", "json"),
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="week-summaries",
        command_args=("insights", "week-summaries"),
        tags=("insights", "week-summaries"),
        exercise=CliSurfaceVariant(
            name="json-insights-week-summaries",
            description="insights week-summaries JSON contract",
        ),
    ),
    CliSurfaceFamily(
        slug="analytics",
        command_args=("insights", "analytics"),
        tags=("insights", "analytics"),
        exercise=CliSurfaceVariant(
            name="json-insights-analytics",
            description="insights analytics JSON contract",
        ),
        live_variants=(
            CliSurfaceVariant(
                name="live-insights-analytics",
                description="Live archive provider-analytics insight surface",
                suffix_args=("--limit", "20", "--format", "json"),
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="status",
        command_args=("insights", "status"),
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
        command_args=("insights", "debt"),
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
    return build_cli_surface_exercises(INSIGHT_SURFACE_FAMILIES)


def build_live_insight_surface_lanes() -> tuple[CompiledCliSurface, ...]:
    return build_cli_surface_live_variants(INSIGHT_SURFACE_FAMILIES)


__all__ = [
    "INSIGHT_SURFACE_FAMILIES",
    "build_insight_contract_surfaces",
    "build_live_insight_surface_lanes",
]
