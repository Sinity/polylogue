"""Authored operational CLI surface families built on shared CLI surface specs."""

from __future__ import annotations

from .cli_surfaces import (
    CliSurfaceFamily,
    CliSurfaceVariant,
    CompiledCliSurface,
    build_cli_surface_exercises,
    build_cli_surface_live_variants,
    build_cli_surface_memory_budget_variants,
)

OPERATIONAL_SURFACE_FAMILIES: tuple[CliSurfaceFamily, ...] = (
    CliSurfaceFamily(
        slug="doctor-readiness",
        command_args=("doctor", "--format", "json"),
        tags=("maintenance", "readiness"),
        exercise=CliSurfaceVariant(
            name="json-doctor",
            description="doctor JSON contract",
            suffix_args=(),
            needs_data=False,
            tier=0,
            env="any",
        ),
        live_variants=(
            CliSurfaceVariant(
                name="live-readiness-json",
                description="Live archive machine-readable readiness report",
                suffix_args=(),
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="doctor-action-event-preview",
        command_args=("doctor", "--format", "json", "--repair", "--preview", "--target", "action_event_read_model"),
        tags=("maintenance", "action-events"),
        exercise=CliSurfaceVariant(
            name="json-doctor-action-event-preview",
            description="doctor JSON contract",
            suffix_args=(),
            needs_data=False,
            tier=0,
            env="any",
        ),
    ),
    CliSurfaceFamily(
        slug="doctor-session-insights-preview",
        command_args=("doctor", "--format", "json", "--repair", "--preview", "--target", "session_products"),
        tags=("maintenance", "session-insights"),
        exercise=CliSurfaceVariant(
            name="json-doctor-session-insights-preview",
            description="doctor JSON contract",
            suffix_args=(),
            needs_data=False,
            tier=0,
            env="any",
        ),
    ),
    CliSurfaceFamily(
        slug="doctor-session-insights-repair",
        command_args=("doctor", "--format", "json", "--repair", "--target", "session_products"),
        tags=("live", "repair", "session-insights"),
        live_variants=(
            CliSurfaceVariant(
                name="live-session-insight-repair",
                description="Live archive evidence/inference session-insight rebuild and migration surface",
                suffix_args=(),
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="maintenance-preview",
        command_args=("doctor", "--format", "json", "--repair", "--cleanup", "--preview"),
        tags=("live", "maintenance", "preview"),
        live_variants=(
            CliSurfaceVariant(
                name="live-maintenance-preview",
                description="Live archive machine-readable maintenance preview for safe repairs and destructive cleanup",
                suffix_args=(),
                timeout_s=240,
                env="any",
            ),
        ),
        memory_budget_variants=(
            CliSurfaceVariant(
                name="maintenance-memory-budget",
                description="Live archive maintenance preview under an explicit RSS budget",
                suffix_args=(),
                timeout_s=240,
                max_rss_mb=1024,
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="embed-stats",
        command_args=("embed", "--stats", "--format", "json"),
        tags=("live", "embeddings", "readiness"),
        live_variants=(
            CliSurfaceVariant(
                name="live-embed-stats",
                description="Live archive embedding status JSON view",
                suffix_args=(),
                timeout_s=120,
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="action-retrieval-stats",
        command_args=(
            "--provider",
            "claude-code",
            "--since",
            "2026-01-01",
            "--stats-by",
            "action",
            "--format",
            "json",
            "--limit",
            "50",
        ),
        tags=("live", "retrieval", "readiness"),
        live_variants=(
            CliSurfaceVariant(
                name="live-retrieval-checks",
                description="Live archive action-aware grouped retrieval stats on a bounded semantic slice",
                suffix_args=(),
                env="any",
            ),
        ),
        memory_budget_variants=(
            CliSurfaceVariant(
                name="memory-budget",
                description="Live archive grouped retrieval command under an explicit RSS budget",
                suffix_args=(),
                timeout_s=240,
                max_rss_mb=1536,
                env="any",
            ),
        ),
    ),
    CliSurfaceFamily(
        slug="project-stats",
        command_args=(
            "--provider",
            "claude-code",
            "--since",
            "2026-01-01",
            "--stats-by",
            "project",
            "--format",
            "json",
            "--limit",
            "50",
        ),
        tags=("live", "insights", "project-stats"),
        live_variants=(
            CliSurfaceVariant(
                name="live-project-stats",
                description="Live archive project-grouped stats over session insights",
                suffix_args=(),
                env="any",
            ),
        ),
    ),
)


def build_operational_contract_surfaces() -> tuple[CompiledCliSurface, ...]:
    return build_cli_surface_exercises(OPERATIONAL_SURFACE_FAMILIES)


def build_live_operational_surface_lanes() -> tuple[CompiledCliSurface, ...]:
    return build_cli_surface_live_variants(OPERATIONAL_SURFACE_FAMILIES)


def build_memory_budget_operational_surface_lanes() -> tuple[CompiledCliSurface, ...]:
    return build_cli_surface_memory_budget_variants(OPERATIONAL_SURFACE_FAMILIES)


__all__ = [
    "OPERATIONAL_SURFACE_FAMILIES",
    "build_live_operational_surface_lanes",
    "build_memory_budget_operational_surface_lanes",
    "build_operational_contract_surfaces",
]
