"""Shared authored CLI surface families for scenario compilation."""

from __future__ import annotations

from dataclasses import dataclass

from .payloads import merge_unique_string_tuples


@dataclass(frozen=True, slots=True)
class CliSurfaceVariant:
    """One compiled verification surface for a CLI family."""

    name: str
    description: str
    prefix_args: tuple[str, ...] = ()
    suffix_args: tuple[str, ...] = ("--json",)
    tags: tuple[str, ...] = ()
    timeout_s: int = 180
    needs_data: bool = True
    tier: int = 1
    env: str = "seeded"
    max_rss_mb: int | None = None

    def compile_args(self, command_args: tuple[str, ...]) -> tuple[str, ...]:
        return self.prefix_args + command_args + self.suffix_args


@dataclass(frozen=True, slots=True)
class CliSurfaceFamily:
    """One authored CLI command family compiled into multiple projections."""

    slug: str
    command_args: tuple[str, ...]
    tags: tuple[str, ...]
    exercise: CliSurfaceVariant | None = None
    live_variants: tuple[CliSurfaceVariant, ...] = ()
    memory_budget_variants: tuple[CliSurfaceVariant, ...] = ()


@dataclass(frozen=True, slots=True)
class CompiledCliSurface:
    """A concrete command surface compiled from a CLI family variant."""

    name: str
    description: str
    args: tuple[str, ...]
    tags: tuple[str, ...]
    timeout_s: int
    needs_data: bool
    tier: int
    env: str
    max_rss_mb: int | None = None


def merge_cli_surface_tags(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """Merge tag groups while preserving first-seen order."""
    return merge_unique_string_tuples(*groups, skip_empty=True)


def compile_cli_surface_variant(family: CliSurfaceFamily, variant: CliSurfaceVariant) -> CompiledCliSurface:
    """Compile one authored family variant into a concrete surface."""

    return CompiledCliSurface(
        name=variant.name,
        description=variant.description,
        args=variant.compile_args(family.command_args),
        tags=merge_cli_surface_tags(family.tags, variant.tags),
        timeout_s=variant.timeout_s,
        needs_data=variant.needs_data,
        tier=variant.tier,
        env=variant.env,
        max_rss_mb=variant.max_rss_mb,
    )


def build_cli_surface_exercises(families: tuple[CliSurfaceFamily, ...]) -> tuple[CompiledCliSurface, ...]:
    """Compile exercise projections from authored families."""

    return tuple(
        compile_cli_surface_variant(family, family.exercise) for family in families if family.exercise is not None
    )


def build_cli_surface_live_variants(families: tuple[CliSurfaceFamily, ...]) -> tuple[CompiledCliSurface, ...]:
    """Compile live-lane projections from authored families."""

    return tuple(
        compile_cli_surface_variant(family, variant) for family in families for variant in family.live_variants
    )


def build_cli_surface_memory_budget_variants(
    families: tuple[CliSurfaceFamily, ...],
) -> tuple[CompiledCliSurface, ...]:
    """Compile memory-budget projections from authored families."""

    return tuple(
        compile_cli_surface_variant(family, variant) for family in families for variant in family.memory_budget_variants
    )


__all__ = [
    "CliSurfaceFamily",
    "CliSurfaceVariant",
    "CompiledCliSurface",
    "build_cli_surface_exercises",
    "build_cli_surface_live_variants",
    "build_cli_surface_memory_budget_variants",
    "compile_cli_surface_variant",
    "merge_cli_surface_tags",
]
