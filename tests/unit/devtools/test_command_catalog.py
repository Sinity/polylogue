from __future__ import annotations

from devtools.command_catalog import (
    CATEGORY_ORDER,
    COMMAND_SPECS,
    COMMANDS,
    command_name_from_tokens,
    control_plane_argv,
    control_plane_command,
    featured_command_specs,
    grouped_command_specs,
)


def test_control_plane_helpers_render_consistent_invocations() -> None:
    assert control_plane_command("status", "--json") == "devtools status --json"
    assert control_plane_argv("status", "--json") == ("devtools", "status", "--json")
    assert control_plane_command("render all", "--check") == "devtools render all --check"
    assert control_plane_argv("render all", "--check") == ("devtools", "render", "all", "--check")
    assert control_plane_command("verify schema-roundtrip", "--all") == "devtools verify schema-roundtrip --all"
    assert control_plane_argv("verify schema-roundtrip", "--all") == ("devtools", "verify", "schema-roundtrip", "--all")
    assert command_name_from_tokens(["render", "all", "--check"]) == "render all"
    assert command_name_from_tokens(["verify", "schema-roundtrip", "--all"]) == "verify schema-roundtrip"


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


def test_catalog_uses_command_ownership_categories() -> None:
    assert "verification lab" not in {spec.category for spec in COMMAND_SPECS}
    assert {"verify schema-roundtrip", "bench pipeline", "workspace schema commit"} <= set(COMMANDS)


def test_bead_graph_catalog_exposes_json_report() -> None:
    graph = COMMANDS["verify bead-graph"]

    assert any("--json" in example for example in graph.examples)
