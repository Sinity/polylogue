from __future__ import annotations

from pathlib import Path

from devtools.command_catalog import (
    CATALOG_BYPASS_SITES,
    CATEGORY_ORDER,
    COMMAND_SPECS,
    COMMANDS,
    VERIFICATION_LAB_COMMAND_NAMES,
    WORKSPACE_COMMAND_DISPOSITIONS,
    command_name_from_tokens,
    control_plane_argv,
    control_plane_command,
    featured_command_specs,
    grouped_command_specs,
    verification_lab_command_specs,
)


def test_control_plane_helpers_render_consistent_invocations() -> None:
    assert control_plane_command("status", "--json") == "devtools status --json"
    assert control_plane_argv("status", "--json") == ("devtools", "status", "--json")
    assert control_plane_command("render all", "--check") == "devtools render all --check"
    assert control_plane_argv("render all", "--check") == ("devtools", "render", "all", "--check")
    assert control_plane_command("lab schema roundtrip", "--all") == "devtools lab schema roundtrip --all"
    assert control_plane_argv("lab schema roundtrip", "--all") == ("devtools", "lab", "schema", "roundtrip", "--all")
    assert command_name_from_tokens(["render", "all", "--check"]) == "render all"
    assert command_name_from_tokens(["lab", "schema", "roundtrip", "--all"]) == "lab schema roundtrip"


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
    assert {spec.category for spec in specs} == {"verification lab"}
    assert len({(spec.module, spec.entrypoint) for spec in specs}) == len(specs)

    for spec in specs:
        assert spec.module.startswith("devtools.")
        assert "Alias" not in spec.description
        assert spec.use_when
        assert spec.examples
        assert callable(spec.resolve_main())


def test_bead_graph_and_frontier_catalogs_expose_complete_json_reports() -> None:
    graph = COMMANDS["lab policy bead-graph"]
    frontier = COMMANDS["workspace frontier"]

    assert "--json" in graph.examples[-1]
    assert "missing acceptance criteria" in graph.use_when
    assert any("--json" in example for example in frontier.examples)
    assert "execution focus" in frontier.description


def test_workspace_dispositions_cover_the_named_utf_entries() -> None:
    expected = {
        "workspace index-fast-forward",
        "workspace archive-schema-fast-forward",
        "workspace degraded-archive-proof",
        "workspace frontier",
        "workspace temporal-read-profile",
        "workspace temporal-devloop",
        "workspace temporal-archive-aggregates",
        "workspace lineage-validation",
        "workspace cli-surface-audit",
        "demo real-slice-screen",
    }
    dispositions = {item.name: item for item in WORKSPACE_COMMAND_DISPOSITIONS}

    assert set(dispositions) == expected
    archived = dispositions["workspace archive-schema-fast-forward"]
    assert archived.disposition == "remove"
    assert archived.replacement_command == "workspace index-fast-forward"
    assert archived.replacement_command in COMMANDS
    for name, item in dispositions.items():
        assert item.evidence
        assert item.replacement
        if item.disposition == "retain":
            assert name in COMMANDS
            assert item.replacement_command is None
        else:
            assert name not in COMMANDS
            assert item.replacement_command in COMMANDS


def test_named_catalog_bypass_sites_are_registered_or_sanctioned() -> None:
    root = Path(__file__).resolve().parents[3]

    assert CATALOG_BYPASS_SITES
    for site in CATALOG_BYPASS_SITES:
        source = (root / site.path).read_text(encoding="utf-8")
        assert site.marker in source
        assert site.disposition in {"registered", "sanctioned-bypass"}
        if site.command_name is not None:
            assert site.command_name in COMMANDS
        else:
            assert site.disposition == "sanctioned-bypass"
        if site.occurrence_line is not None:
            assert site.disposition == "sanctioned-bypass"
            assert site.expected_occurrences == 1
            assert source.splitlines()[site.occurrence_line - 1].strip().startswith(site.marker)
        assert site.reason
