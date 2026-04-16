from __future__ import annotations

from devtools.command_catalog import (
    CATEGORY_ORDER,
    COMMAND_SPECS,
    control_plane_argv,
    control_plane_command,
    featured_command_specs,
    grouped_command_specs,
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
