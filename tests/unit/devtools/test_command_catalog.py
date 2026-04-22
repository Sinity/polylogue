from __future__ import annotations

from devtools.command_catalog import (
    CATEGORY_ORDER,
    COMMAND_SPECS,
    VERIFICATION_LAB_COMMAND_NAMES,
    control_plane_argv,
    control_plane_command,
    featured_command_specs,
    grouped_command_specs,
    verification_lab_command_specs,
)


def test_control_plane_helpers_render_consistent_invocations() -> None:
    assert control_plane_command("status", "--json") == "devtools status --json"
    assert control_plane_argv("status", "--json") == ("devtools", "status", "--json")


def test_command_specs_have_unique_names_and_known_categories() -> None:
    names = [spec.name for spec in COMMAND_SPECS]
    assert len(names) == len(set(names))
    assert {spec.category for spec in COMMAND_SPECS}.issubset(set(CATEGORY_ORDER))


def test_grouped_command_specs_preserves_declared_category_order() -> None:
    grouped = grouped_command_specs()
    assert tuple(grouped) == tuple(category for category in CATEGORY_ORDER if grouped.get(category))
    for specs in grouped.values():
        assert specs == sorted(specs, key=lambda item: item.name)


def test_featured_command_specs_are_actionable() -> None:
    featured = featured_command_specs()
    assert featured
    for spec in featured:
        assert spec.use_when
        assert spec.examples
        assert spec.to_dict()["argv"] == list(spec.argv)


def test_verification_lab_surface_is_explicit_and_implemented() -> None:
    specs = verification_lab_command_specs()

    assert tuple(spec.name for spec in specs) == VERIFICATION_LAB_COMMAND_NAMES
    assert {spec.category for spec in specs} == {"generated surfaces", "verification"}
    assert len({spec.module for spec in specs}) == len(specs)

    for spec in specs:
        assert spec.module.startswith("devtools.")
        assert "Alias" not in spec.description
        assert spec.use_when
        assert spec.examples
        assert callable(spec.resolve_main())
