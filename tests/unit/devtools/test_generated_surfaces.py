from __future__ import annotations

from pathlib import Path

from devtools.generated_surfaces import GENERATED_SURFACES


def _surface_inputs(name: str) -> set[str]:
    for surface in GENERATED_SURFACES:
        if surface.name == name:
            return set(surface.inputs)
    raise AssertionError(f"unknown generated surface: {name}")


def test_generated_surfaces_use_public_devtools_commands() -> None:
    assert GENERATED_SURFACES
    for surface in GENERATED_SURFACES:
        assert surface.command[0] == "devtools"
        assert len(surface.command) >= 2
        assert all(part and " " not in part for part in surface.command)
        assert callable(surface.main)


def test_generated_surface_names_and_labels_are_unique() -> None:
    assert len({surface.name for surface in GENERATED_SURFACES}) == len(GENERATED_SURFACES)
    assert len({surface.label for surface in GENERATED_SURFACES}) == len(GENERATED_SURFACES)


def test_generated_surface_cache_inputs_include_renderer_module() -> None:
    """Renderer edits must invalidate normal render all stamps, not only --check."""
    for surface in GENERATED_SURFACES:
        renderer_path = Path(*surface.main.__module__.split(".")).with_suffix(".py").as_posix()
        assert renderer_path in surface.inputs, surface.name


def test_generated_surface_cache_inputs_include_contract_owners() -> None:
    """Contract-owner edits must invalidate generated surfaces that publish them."""
    assert {
        "polylogue/archive/query/",
        "polylogue/archive/viewport/",
        "polylogue/operations/action_contracts.py",
        "polylogue/surfaces/payloads.py",
    }.issubset(_surface_inputs("cli-reference"))

    assert {
        "polylogue/archive/query/",
        "polylogue/surfaces/payloads.py",
    }.issubset(_surface_inputs("cli-output-schemas"))

    assert {
        "polylogue/archive/query/",
        "polylogue/archive/viewport/",
        "polylogue/daemon/",
        "polylogue/surfaces/payloads.py",
    }.issubset(_surface_inputs("openapi"))

    assert {
        "devtools/benchmark_catalog.py",
        "devtools/mutation_catalog.py",
        "devtools/quality_registry.py",
        "devtools/scenario_coverage.py",
        "devtools/scenario_projection_catalog.py",
        "devtools/validation_lane_catalog_contracts.py",
        "devtools/validation_lane_catalog_live.py",
        "polylogue/operations/specs.py",
        "polylogue/scenarios/",
    }.issubset(_surface_inputs("quality-reference"))
