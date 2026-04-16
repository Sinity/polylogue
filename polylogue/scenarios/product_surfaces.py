"""Authored product-query surface families built on shared CLI surface specs."""

from __future__ import annotations

from .cli_surfaces import (
    CliSurfaceFamily,
    CliSurfaceVariant,
    CompiledCliSurface,
    build_cli_surface_exercises,
    build_cli_surface_live_variants,
)

PRODUCT_SURFACE_FAMILIES: tuple[CliSurfaceFamily, ...] = (
    CliSurfaceFamily(
        slug="profiles",
        command_args=("products", "profiles"),
        tags=("products", "session-profiles"),
        exercise=CliSurfaceVariant(
            name="json-products-profiles",
            description="products profiles JSON contract",
        ),
        live_variants=(
            CliSurfaceVariant(
                name="live-products-profiles-evidence",
                description="Live archive evidence-tier session-profile product surface",
                suffix_args=("--tier", "evidence", "--limit", "3", "--json"),
                env="any",
            ),
            CliSurfaceVariant(
                name="live-products-profiles-inference",
                description="Live archive inference-tier session-profile product surface",
                suffix_args=("--tier", "inference", "--limit", "3", "--json"),
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="enrichments",
        command_args=("products", "enrichments"),
        tags=("products", "enrichments"),
        live_variants=(
            CliSurfaceVariant(
                name="live-products-enrichments",
                description="Live archive probabilistic session-enrichment product surface",
                suffix_args=("--limit", "5", "--json"),
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="work-events",
        command_args=("products", "work-events"),
        tags=("products", "work-events"),
        exercise=CliSurfaceVariant(
            name="json-products-work-events",
            description="products work-events JSON contract",
        ),
        live_variants=(
            CliSurfaceVariant(
                name="live-products-work-events",
                description="Live archive inferred work-event product surface",
                suffix_args=("--limit", "3", "--json"),
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="phases",
        command_args=("products", "phases"),
        tags=("products", "phases"),
        exercise=CliSurfaceVariant(
            name="json-products-phases",
            description="products phases JSON contract",
        ),
        live_variants=(
            CliSurfaceVariant(
                name="live-products-phases",
                description="Live archive inferred phase product surface",
                suffix_args=("--limit", "3", "--json"),
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="threads",
        command_args=("products", "threads"),
        tags=("products", "threads"),
        exercise=CliSurfaceVariant(
            name="json-products-threads",
            description="products threads JSON contract",
        ),
    ),
    CliSurfaceFamily(
        slug="tags",
        command_args=("products", "tags"),
        tags=("products", "tags"),
        exercise=CliSurfaceVariant(
            name="json-products-tags",
            description="products tags JSON contract",
        ),
        live_variants=(
            CliSurfaceVariant(
                name="live-products-tags",
                description="Live archive tag-rollup product view",
                suffix_args=("--limit", "20", "--json"),
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="day-summaries",
        command_args=("products", "day-summaries"),
        tags=("products", "day-summaries"),
        exercise=CliSurfaceVariant(
            name="json-products-day-summaries",
            description="products day-summaries JSON contract",
        ),
        live_variants=(
            CliSurfaceVariant(
                name="live-products-day-summaries",
                description="Live archive day-summary product surface over the recent semantic slice",
                prefix_args=("--provider", "claude-code", "--since", "2026-03-01"),
                suffix_args=("--limit", "14", "--json"),
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="week-summaries",
        command_args=("products", "week-summaries"),
        tags=("products", "week-summaries"),
        exercise=CliSurfaceVariant(
            name="json-products-week-summaries",
            description="products week-summaries JSON contract",
        ),
    ),
    CliSurfaceFamily(
        slug="analytics",
        command_args=("products", "analytics"),
        tags=("products", "analytics"),
        exercise=CliSurfaceVariant(
            name="json-products-analytics",
            description="products analytics JSON contract",
        ),
        live_variants=(
            CliSurfaceVariant(
                name="live-products-analytics",
                description="Live archive provider-analytics product surface",
                suffix_args=("--limit", "20", "--json"),
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="status",
        command_args=("products", "status"),
        tags=("products", "status"),
        live_variants=(
            CliSurfaceVariant(
                name="live-products-status",
                description="Live archive product status view",
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="debt",
        command_args=("products", "debt"),
        tags=("products", "debt"),
        live_variants=(
            CliSurfaceVariant(
                name="live-products-debt",
                description="Live archive debt and cleanup product view",
                suffix_args=("--limit", "20", "--json"),
                env="any",
            ),
        ),
    ),
)


def build_product_contract_surfaces() -> tuple[CompiledCliSurface, ...]:
    return build_cli_surface_exercises(PRODUCT_SURFACE_FAMILIES)


def build_live_product_surface_lanes() -> tuple[CompiledCliSurface, ...]:
    return build_cli_surface_live_variants(PRODUCT_SURFACE_FAMILIES)


__all__ = [
    "PRODUCT_SURFACE_FAMILIES",
    "build_live_product_surface_lanes",
    "build_product_contract_surfaces",
]
