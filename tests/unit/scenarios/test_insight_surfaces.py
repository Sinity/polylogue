from __future__ import annotations

import click
import pytest

from polylogue.cli.click_app import cli
from polylogue.scenarios import (
    INSIGHT_SURFACE_FAMILIES,
    CliSurfaceFamily,
    build_insight_contract_surfaces,
    build_live_insight_surface_lanes,
)
from polylogue.scenarios.cli_surfaces import validate_cli_surface_families


def _cli_command_exists(command_args: tuple[str, ...]) -> bool:
    command: click.Command = cli
    context = click.Context(cli)
    for part in command_args:
        if not isinstance(command, click.Group):
            return False
        resolved = command.get_command(context, part)
        if resolved is None:
            return False
        command = resolved
    return True


def test_insight_surface_families_reference_registered_cli_commands() -> None:
    # Families are a curated scenario subset, including operational commands,
    # so validate them against Click instead of deriving them from insights.
    # Archive-readiness status surfaces are independent of retired insight types.
    validate_cli_surface_families(INSIGHT_SURFACE_FAMILIES, _cli_command_exists)


def test_insight_surface_command_validation_rejects_retired_command() -> None:
    retired_family = CliSurfaceFamily(
        slug="phases",
        command_args=("analyze", "insights", "phases"),
        tags=("insights", "phases"),
    )

    with pytest.raises(ValueError, match="phases: analyze insights phases"):
        validate_cli_surface_families((retired_family,), _cli_command_exists)


def test_build_insight_contract_surfaces_compiles_canonical_json_contract_entries() -> None:
    surfaces = {surface.name: surface for surface in build_insight_contract_surfaces()}

    assert surfaces["json-insights-profiles"].args == ("analyze", "insights", "profiles", "--format", "json")
    assert surfaces["json-insights-profiles"].tags == ("insights", "session-profiles")
    assert surfaces["json-insights-coverage"].args == ("analyze", "insights", "coverage", "--format", "json")


def test_build_live_insight_surface_lanes_compiles_live_variants() -> None:
    surfaces = {surface.name: surface for surface in build_live_insight_surface_lanes()}

    assert surfaces["live-insights-status"].args == ("ops", "insights", "status", "--format", "json")
    assert surfaces["live-insights-profiles-evidence"].args == (
        "analyze",
        "insights",
        "profiles",
        "--tier",
        "evidence",
        "--limit",
        "3",
        "--format",
        "json",
    )
    assert surfaces["live-insights-coverage-day"].args == (
        "--provider",
        "claude-code",
        "--since",
        "2026-03-01",
        "analyze",
        "insights",
        "coverage",
        "--group-by",
        "day",
        "--limit",
        "14",
        "--format",
        "json",
    )
    assert surfaces["live-insights-debt"].tags == ("insights", "debt")
