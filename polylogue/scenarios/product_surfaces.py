"""Shared authored families for product-query verification surfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProductSurfaceVariant:
    """One compiled verification surface for a product-query family."""

    name: str
    description: str
    prefix_args: tuple[str, ...] = ()
    suffix_args: tuple[str, ...] = ("--json",)
    tags: tuple[str, ...] = ()
    timeout_s: int = 180
    needs_data: bool = True
    tier: int = 1
    env: str = "seeded"

    def compile_args(self, product_args: tuple[str, ...]) -> tuple[str, ...]:
        return self.prefix_args + product_args + self.suffix_args


@dataclass(frozen=True, slots=True)
class ProductSurfaceFamily:
    """One product-query family compiled into showcase and validation variants."""

    slug: str
    product_args: tuple[str, ...]
    tags: tuple[str, ...]
    exercise: ProductSurfaceVariant | None = None
    live_variants: tuple[ProductSurfaceVariant, ...] = ()


@dataclass(frozen=True, slots=True)
class CompiledProductSurface:
    """A concrete command surface compiled from a product family variant."""

    name: str
    description: str
    args: tuple[str, ...]
    tags: tuple[str, ...]
    timeout_s: int
    needs_data: bool
    tier: int
    env: str


PRODUCT_SURFACE_FAMILIES: tuple[ProductSurfaceFamily, ...] = (
    ProductSurfaceFamily(
        slug="profiles",
        product_args=("products", "profiles"),
        tags=("products", "session-profiles"),
        exercise=ProductSurfaceVariant(
            name="json-products-profiles",
            description="products profiles JSON contract",
        ),
        live_variants=(
            ProductSurfaceVariant(
                name="live-products-profiles-evidence",
                description="Live archive evidence-tier session-profile product surface",
                suffix_args=("--tier", "evidence", "--limit", "3", "--json"),
                env="any",
            ),
            ProductSurfaceVariant(
                name="live-products-profiles-inference",
                description="Live archive inference-tier session-profile product surface",
                suffix_args=("--tier", "inference", "--limit", "3", "--json"),
                env="any",
            ),
        ),
    ),
    ProductSurfaceFamily(
        slug="enrichments",
        product_args=("products", "enrichments"),
        tags=("products", "enrichments"),
        live_variants=(
            ProductSurfaceVariant(
                name="live-products-enrichments",
                description="Live archive probabilistic session-enrichment product surface",
                suffix_args=("--limit", "5", "--json"),
                env="any",
            ),
        ),
    ),
    ProductSurfaceFamily(
        slug="work-events",
        product_args=("products", "work-events"),
        tags=("products", "work-events"),
        exercise=ProductSurfaceVariant(
            name="json-products-work-events",
            description="products work-events JSON contract",
        ),
        live_variants=(
            ProductSurfaceVariant(
                name="live-products-work-events",
                description="Live archive inferred work-event product surface",
                suffix_args=("--limit", "3", "--json"),
                env="any",
            ),
        ),
    ),
    ProductSurfaceFamily(
        slug="phases",
        product_args=("products", "phases"),
        tags=("products", "phases"),
        exercise=ProductSurfaceVariant(
            name="json-products-phases",
            description="products phases JSON contract",
        ),
        live_variants=(
            ProductSurfaceVariant(
                name="live-products-phases",
                description="Live archive inferred phase product surface",
                suffix_args=("--limit", "3", "--json"),
                env="any",
            ),
        ),
    ),
    ProductSurfaceFamily(
        slug="threads",
        product_args=("products", "threads"),
        tags=("products", "threads"),
        exercise=ProductSurfaceVariant(
            name="json-products-threads",
            description="products threads JSON contract",
        ),
    ),
    ProductSurfaceFamily(
        slug="tags",
        product_args=("products", "tags"),
        tags=("products", "tags"),
        exercise=ProductSurfaceVariant(
            name="json-products-tags",
            description="products tags JSON contract",
        ),
        live_variants=(
            ProductSurfaceVariant(
                name="live-products-tags",
                description="Live archive tag-rollup product view",
                suffix_args=("--limit", "20", "--json"),
                env="any",
            ),
        ),
    ),
    ProductSurfaceFamily(
        slug="day-summaries",
        product_args=("products", "day-summaries"),
        tags=("products", "day-summaries"),
        exercise=ProductSurfaceVariant(
            name="json-products-day-summaries",
            description="products day-summaries JSON contract",
        ),
        live_variants=(
            ProductSurfaceVariant(
                name="live-products-day-summaries",
                description="Live archive day-summary product surface over the recent semantic slice",
                prefix_args=("--provider", "claude-code", "--since", "2026-03-01"),
                suffix_args=("--limit", "14", "--json"),
                env="any",
            ),
        ),
    ),
    ProductSurfaceFamily(
        slug="week-summaries",
        product_args=("products", "week-summaries"),
        tags=("products", "week-summaries"),
        exercise=ProductSurfaceVariant(
            name="json-products-week-summaries",
            description="products week-summaries JSON contract",
        ),
    ),
    ProductSurfaceFamily(
        slug="analytics",
        product_args=("products", "analytics"),
        tags=("products", "analytics"),
        exercise=ProductSurfaceVariant(
            name="json-products-analytics",
            description="products analytics JSON contract",
        ),
        live_variants=(
            ProductSurfaceVariant(
                name="live-products-analytics",
                description="Live archive provider-analytics product surface",
                suffix_args=("--limit", "20", "--json"),
                env="any",
            ),
        ),
    ),
    ProductSurfaceFamily(
        slug="status",
        product_args=("products", "status"),
        tags=("products", "status"),
        live_variants=(
            ProductSurfaceVariant(
                name="live-products-status",
                description="Live archive product status view",
                env="any",
            ),
        ),
    ),
    ProductSurfaceFamily(
        slug="debt",
        product_args=("products", "debt"),
        tags=("products", "debt"),
        live_variants=(
            ProductSurfaceVariant(
                name="live-products-debt",
                description="Live archive debt and cleanup product view",
                suffix_args=("--limit", "20", "--json"),
                env="any",
            ),
        ),
    ),
)


def _merge_tags(*groups: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for item in group:
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return tuple(merged)


def _compile_variant(family: ProductSurfaceFamily, variant: ProductSurfaceVariant) -> CompiledProductSurface:
    return CompiledProductSurface(
        name=variant.name,
        description=variant.description,
        args=variant.compile_args(family.product_args),
        tags=_merge_tags(family.tags, variant.tags),
        timeout_s=variant.timeout_s,
        needs_data=variant.needs_data,
        tier=variant.tier,
        env=variant.env,
    )


def build_product_contract_surfaces() -> tuple[CompiledProductSurface, ...]:
    return tuple(
        _compile_variant(family, family.exercise)
        for family in PRODUCT_SURFACE_FAMILIES
        if family.exercise is not None
    )


def build_live_product_surface_lanes() -> tuple[CompiledProductSurface, ...]:
    return tuple(
        _compile_variant(family, variant)
        for family in PRODUCT_SURFACE_FAMILIES
        for variant in family.live_variants
    )


__all__ = [
    "CompiledProductSurface",
    "PRODUCT_SURFACE_FAMILIES",
    "ProductSurfaceFamily",
    "ProductSurfaceVariant",
    "build_live_product_surface_lanes",
    "build_product_contract_surfaces",
]
