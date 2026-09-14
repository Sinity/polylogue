from __future__ import annotations

from polylogue.scenarios import (
    build_live_operational_surface_lanes,
    build_memory_budget_operational_surface_lanes,
    build_operational_contract_surfaces,
)


def test_build_operational_contract_surfaces_compiles_runtime_aligned_json_contracts() -> None:
    surfaces = {surface.name: surface for surface in build_operational_contract_surfaces()}

    assert surfaces["json-doctor"].args == ("ops", "doctor", "--format", "json")
    assert surfaces["json-doctor"].tags == ("maintenance", "readiness")
    assert "json-doctor-action-preview" not in surfaces
    # `polylogue ops doctor` is a read-only report. No declared surface may
    # name a mutating repair/cleanup projection of it.
    assert "json-doctor-session-insights-preview" not in surfaces


def test_build_live_operational_surface_lanes_compiles_live_variants() -> None:
    surfaces = {surface.name: surface for surface in build_live_operational_surface_lanes()}

    assert surfaces["live-readiness-json"].args == ("ops", "doctor", "--format", "json")
    assert surfaces["live-readiness-json"].tags == ("maintenance", "readiness")
    assert surfaces["live-retrieval-checks"].args == (
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
    )
    assert surfaces["live-retrieval-checks"].tags == ("live", "retrieval", "readiness")
    assert surfaces["live-project-stats"].args == (
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
    )


def test_build_memory_budget_operational_surface_lanes_compiles_budget_variants() -> None:
    surfaces = {surface.name: surface for surface in build_memory_budget_operational_surface_lanes()}

    assert surfaces["memory-budget"].args == (
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
    )
    assert surfaces["memory-budget"].max_rss_mb == 1536
    assert "maintenance-memory-budget" not in surfaces


def test_no_declared_surface_invokes_doctor_with_an_undeclared_option() -> None:
    """`polylogue ops doctor` is a read-only report; no declared surface may flag it otherwise.

    Three families used to compile `ops doctor --repair [--cleanup] [--target ...]`
    long after those options were deleted from the command, and nothing caught it
    because the scenario declarations were never resolved against the real Click app.

    Anti-vacuity: re-adding ``--repair`` to any ``ops doctor`` family's
    ``command_args`` makes this red, because ``get_params`` on the resolved
    command reports no such option. Verified by reverting the deletion.
    """
    import click

    from polylogue.cli.click_app import cli
    from polylogue.scenarios import OPERATIONAL_SURFACE_FAMILIES

    root_ctx = click.Context(cli)
    doctor_families = [
        family for family in OPERATIONAL_SURFACE_FAMILIES if family.command_args[:2] == ("ops", "doctor")
    ]
    assert doctor_families, "expected at least one declared `ops doctor` surface family"

    ops = cli.get_command(root_ctx, "ops")
    assert isinstance(ops, click.Group)
    ops_ctx = click.Context(ops, parent=root_ctx)
    doctor = ops.get_command(ops_ctx, "doctor")
    assert doctor is not None
    doctor_ctx = click.Context(doctor, parent=ops_ctx)
    declared = {opt for param in doctor.get_params(doctor_ctx) for opt in param.opts if opt.startswith("--")}

    unknown = [
        f"{family.slug}: {token} is not an option of `polylogue ops doctor`"
        for family in doctor_families
        for token in family.command_args[2:]
        if token.startswith("--") and token.split("=", 1)[0] not in declared
    ]
    assert not unknown, unknown
