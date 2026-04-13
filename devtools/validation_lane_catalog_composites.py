"""Composite validation lane declarations."""

from __future__ import annotations

from functools import partial

from devtools.validation_family_models import (
    ValidationLaneFamily,
    ValidationLaneStageSpec,
    compile_validation_lane_families,
)
from devtools.validation_lane_base import composite_lane as _composite_lane

composite_lane = partial(_composite_lane, category="composite")

VALIDATION_FAMILIES: tuple[ValidationLaneFamily, ...] = (
    ValidationLaneFamily.from_stages(
        name="domain-read-model",
        description="Validation family for read-model contracts, live checks, and hardening lanes.",
        stages=(
            ValidationLaneStageSpec(
                suffix="contracts",
                description="Local domain read-model lane for analytics/products, consumer contracts, and debt views",
                timeout_s=2400,
                members=("archive-data-products", "maintenance-workflows"),
            ),
            ValidationLaneStageSpec(
                suffix="live",
                description="Bounded live archive lane for products, analytics/debt views, and maintenance checks",
                timeout_s=1800,
                members=(
                    "live-products-small",
                    "live-products-analytics",
                    "live-products-debt",
                    "live-maintenance-small",
                ),
            ),
            ValidationLaneStageSpec(
                suffix="hardening",
                description="Full domain read-model lane with local contracts and bounded live checks",
                timeout_s=3600,
                member_stages=("contracts", "live"),
            ),
        ),
    ),
    ValidationLaneFamily.from_stages(
        name="runtime-substrate",
        description="Validation family for runtime-substrate contract, live, and hardening lanes.",
        stages=(
            ValidationLaneStageSpec(
                suffix="contracts",
                description="Local runtime-substrate lane across query, semantic checks, archive products, and maintenance workflows",
                timeout_s=2400,
                members=("query-routing", "semantic-stack", "maintenance-workflows", "archive-data-products"),
            ),
            ValidationLaneStageSpec(
                suffix="live",
                description="Bounded live archive lane for runtime-substrate checks, maintenance checks, and memory budgets",
                timeout_s=1800,
                members=("live-archive-small", "live-maintenance-small", "memory-budget"),
            ),
            ValidationLaneStageSpec(
                suffix="hardening",
                description="Full runtime-substrate validation lane covering local contracts plus bounded live archive checks",
                timeout_s=3600,
                member_stages=("contracts", "live"),
            ),
        ),
    ),
    ValidationLaneFamily.from_stages(
        name="evidence",
        description="Validation family for evidence-tier contracts, live checks, and hardening lanes.",
        stages=(
            ValidationLaneStageSpec(
                suffix="contracts",
                description="Evidence/inference contract lane across explicit evidence, inferred semantics, consumer parity, and retrieval readiness",
                timeout_s=2400,
                members=(
                    "evidence-tier-contracts",
                    "inference-tier-contracts",
                    "mixed-consumer-contracts",
                    "retrieval-band-readiness",
                ),
            ),
            ValidationLaneStageSpec(
                suffix="live",
                description="Bounded live archive lane for tiered product views, live migration, health, and retrieval-band budgets",
                timeout_s=2400,
                members=(
                    "live-session-product-repair",
                    "live-products-status",
                    "live-products-profiles-evidence",
                    "live-products-profiles-inference",
                    "live-products-work-events",
                    "live-products-phases",
                    "live-embed-stats",
                    "live-health-json",
                    "maintenance-memory-budget",
                ),
            ),
            ValidationLaneStageSpec(
                suffix="hardening",
                description="Full evidence lane with contracts and bounded live checks",
                timeout_s=4800,
                member_stages=("contracts", "live"),
            ),
        ),
    ),
    ValidationLaneFamily.from_stages(
        name="semantic-product",
        description="Validation family for semantic-product live and hardening lanes.",
        stages=(
            ValidationLaneStageSpec(
                suffix="live",
                description="Bounded live archive lane for normalized products, maintenance preview, and memory budgets",
                timeout_s=1800,
                members=(
                    "live-products-status",
                    "live-products-tags",
                    "live-products-day-summaries",
                    "live-products-debt",
                    "live-maintenance-small",
                ),
            ),
            ValidationLaneStageSpec(
                suffix="hardening",
                description="Full semantic-product normalization and toolchain convergence lane",
                timeout_s=3600,
                members=("semantic-product-normalization", "semantic-product-live"),
            ),
        ),
    ),
    ValidationLaneFamily.from_stages(
        name="probabilistic-enrichment",
        description="Validation family for probabilistic-enrichment, cleanup, and hardening lanes.",
        stages=(
            ValidationLaneStageSpec(
                suffix="live",
                description="Bounded live archive lane for enrichment products, retrieval bands, and health surfaces",
                timeout_s=2400,
                members=(
                    "live-session-product-repair",
                    "live-products-status",
                    "live-products-profiles-inference",
                    "live-products-enrichments",
                    "live-embed-stats",
                    "live-health-json",
                    "memory-budget",
                ),
            ),
            ValidationLaneStageSpec(
                suffix="cleanup-live",
                description="Bounded live archive lane for cleanup/debt preview and maintenance budgets",
                timeout_s=2400,
                members=("live-products-debt", "live-maintenance-preview", "maintenance-memory-budget"),
            ),
            ValidationLaneStageSpec(
                suffix="hardening",
                description="Full probabilistic-enrichment and cleanup lane",
                timeout_s=5400,
                members=(
                    "heuristic-inference-contracts",
                    "probabilistic-enrichment-contracts",
                    "cleanup-contracts",
                ),
                member_stages=("live", "cleanup-live"),
            ),
        ),
    ),
)

FAMILY_COMPOSITE_LANES = compile_validation_lane_families(VALIDATION_FAMILIES)

STANDALONE_COMPOSITE_LANES = {
    "source-runtime-alignment": composite_lane(
        "source-runtime-alignment",
        "Local source/provider fidelity plus runtime maintenance alignment",
        1800,
        "source-provider-fidelity",
        "maintenance-workflows",
    ),
    "live-archive-small": composite_lane(
        "live-archive-small",
        "Bounded live archive retrieval/readiness/health dogfood lane",
        480,
        "live-embed-stats",
        "live-retrieval-checks",
        "live-products-status",
        "live-health-json",
    ),
    "live-products-small": composite_lane(
        "live-products-small",
        "Bounded live archive product and grouped-stats lane",
        480,
        "live-products-status",
        "live-products-tags",
        "live-project-stats",
    ),
    "live-archive-slow": composite_lane(
        "live-archive-slow",
        "Broader live archive dogfood lane including retrieval/readiness and live QA exercises",
        2400,
        "live-archive-small",
        "live-exercises",
    ),
    "archive-intelligence": composite_lane(
        "archive-intelligence",
        "Local archive-intelligence closure lane for retrieval and embedding readiness",
        1800,
        "retrieval-checks",
        "embeddings-coverage",
    ),
    "archive-data-products-live": composite_lane(
        "archive-data-products-live",
        "Local product-contract lane plus bounded live archive product checks",
        1800,
        "archive-data-products",
        "live-products-small",
    ),
    "live-maintenance-small": composite_lane(
        "live-maintenance-small",
        "Bounded live archive lane for health, maintenance preview, and maintenance memory budget",
        720,
        "live-health-json",
        "live-maintenance-preview",
        "maintenance-memory-budget",
    ),
    "scale-stretch": composite_lane(
        "scale-stretch",
        "Combined fast and slow storage scale budgets",
        600,
        "scale-fast",
        "scale-slow",
    ),
    "frontier-local": composite_lane(
        "frontier-local",
        "Non-live local closure lane for machine/query/semantic/TUI/chaos validation",
        1500,
        "machine-contract",
        "query-routing",
        "showcase-baselines",
        "semantic-stack",
        "tui",
        "chaos",
    ),
    "frontier-extended": composite_lane(
        "frontier-extended",
        "Local closure lane plus fast scale and small long-haul campaign",
        3600,
        "frontier-local",
        "pipeline-probe-chatgpt",
        "scale-fast",
        "long-haul-small",
    ),
}

COMPOSITE_LANES = {
    **FAMILY_COMPOSITE_LANES,
    **STANDALONE_COMPOSITE_LANES,
}


__all__ = ["COMPOSITE_LANES", "VALIDATION_FAMILIES"]
