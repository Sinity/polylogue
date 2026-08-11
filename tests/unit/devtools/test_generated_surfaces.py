from __future__ import annotations

from pathlib import Path

from devtools.command_catalog import COMMAND_SPECS
from devtools.generated_surfaces import GENERATED_SURFACES, GENERATED_SURFACES_CATALOG_EXEMPTIONS


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


def test_generated_surfaces_catalog_category_is_gated() -> None:
    """Every ``devtools/command_catalog.py`` command in the "generated
    surfaces" category must either be a registered ``GeneratedSurface`` (so
    ``render all`` / ``render all --check`` actually gates it) or carry an
    explicit, commented exemption in ``GENERATED_SURFACES_CATALOG_EXEMPTIONS``
    naming its bespoke gate.

    Without this test, a new "generated surfaces"-category CommandSpec can
    join the catalog without joining any gate at all -- polylogue-bfc7a found
    exactly this for `render topology-projection` and `render visual-tapes`,
    which rode the category label but were wired (or, for visual-tapes, not
    wired at all) outside `devtools/generated_surfaces.py`.
    """
    registered_commands = {surface.command for surface in GENERATED_SURFACES}
    ungated = [
        spec.name
        for spec in COMMAND_SPECS
        if spec.category == "generated surfaces"
        and spec.name != "render all"  # the orchestrator over the registry, not a member of it
        and spec.argv not in registered_commands
        and spec.name not in GENERATED_SURFACES_CATALOG_EXEMPTIONS
    ]
    assert not ungated, (
        f"generated-surfaces category command(s) with no gate: {ungated} -- "
        "register in GENERATED_SURFACES or add a commented exemption to "
        "GENERATED_SURFACES_CATALOG_EXEMPTIONS naming the bespoke gate"
    )


def test_generated_surfaces_catalog_exemptions_reference_real_commands() -> None:
    """Keep the exemption dict itself honest: a stale exemption for a command
    that no longer exists (renamed/removed) should not silently linger."""
    catalog_names = {spec.name for spec in COMMAND_SPECS if spec.category == "generated surfaces"}
    stale = set(GENERATED_SURFACES_CATALOG_EXEMPTIONS) - catalog_names
    assert not stale, f"stale generated-surfaces exemption(s), no matching command: {stale}"


def test_generated_surface_cache_inputs_include_contract_owners() -> None:
    """Contract-owner edits must invalidate generated surfaces that publish them."""
    assert {
        "polylogue/cli/click_command_registration.py",
        "polylogue/cli/command_inventory.py",
        "polylogue/cli/query_group.py",
        "polylogue/archive/query/",
        "polylogue/archive/query/metadata.py",
        "polylogue/archive/query/fields.py",
        "polylogue/archive/query/unit_results.py",
        "polylogue/archive/viewport/",
        "polylogue/archive/viewport/profiles.py",
        "polylogue/operations/action_contracts.py",
        "polylogue/surfaces/action_affordances.py",
        "polylogue/surfaces/payloads.py",
    }.issubset(_surface_inputs("cli-reference"))

    assert {
        "polylogue/archive/query/",
        "polylogue/archive/query/metadata.py",
        "polylogue/archive/query/unit_results.py",
        "polylogue/context/compiler.py",
        "polylogue/insights/transforms.py",
        "polylogue/surfaces/action_affordances.py",
        "polylogue/surfaces/payloads.py",
    }.issubset(_surface_inputs("cli-output-schemas"))

    assert {
        "polylogue/archive/query/",
        "polylogue/archive/query/metadata.py",
        "polylogue/archive/query/unit_results.py",
        "polylogue/archive/viewport/",
        "polylogue/archive/viewport/profiles.py",
        "polylogue/browser_capture/models.py",
        "polylogue/browser_capture/route_contracts.py",
        "polylogue/daemon/",
        "polylogue/daemon/http.py",
        "polylogue/daemon/route_contracts.py",
        "polylogue/context/compiler.py",
        "polylogue/insights/transforms.py",
        "polylogue/surfaces/action_affordances.py",
        "polylogue/surfaces/payloads.py",
    }.issubset(_surface_inputs("openapi"))

    assert {
        "docs/openapi/search.yaml",
        "devtools/render_webui_client.py",
    }.issubset(_surface_inputs("webui-client"))

    assert {
        "polylogue/cli/click_command_registration.py",
        "polylogue/operations/action_contracts.py",
        "polylogue/archive/query/metadata.py",
        "polylogue/archive/viewport/profiles.py",
        "polylogue/daemon/route_contracts.py",
        "polylogue/sources/provider_completeness.py",
    }.issubset(_surface_inputs("docs-surface"))

    assert {
        "polylogue/daemon/route_contracts.py",
        "polylogue/archive/query/metadata.py",
        "polylogue/archive/viewport/profiles.py",
        "polylogue/surfaces/payloads.py",
    }.issubset(_surface_inputs("pages"))

    assert {
        "polylogue/demo/",
        "polylogue/scenarios/",
    }.issubset(_surface_inputs("demo-corpus-datasheet"))

    assert {
        "devtools/authored_scenario_catalog.py",
        "devtools/benchmark_catalog.py",
        "devtools/mutation_catalog.py",
        "devtools/validation_lane_catalog_contracts.py",
        "devtools/validation_lane_catalog_live.py",
        "polylogue/operations/specs.py",
        "polylogue/scenarios/",
    }.issubset(_surface_inputs("quality-reference"))
