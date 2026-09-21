from __future__ import annotations

import argparse
import importlib
import shlex

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
    assert control_plane_command("render", "all", "--check") == "devtools render all --check"
    assert control_plane_argv("render", "all", "--check") == ("devtools", "render", "all", "--check")
    assert control_plane_command("gate", "schema-roundtrip") == "devtools gate schema-roundtrip"
    assert control_plane_argv("gate", "schema-roundtrip") == ("devtools", "gate", "schema-roundtrip")
    assert command_name_from_tokens(["render", "all", "--check"]) == "render"
    assert command_name_from_tokens(["schema", "parser-diff"]) == "schema parser-diff"


def test_command_specs_have_unique_names_and_known_categories() -> None:
    names = [spec.name for spec in COMMAND_SPECS]
    assert len(names) == len(set(names))
    assert {spec.category for spec in COMMAND_SPECS}.issubset(set(CATEGORY_ORDER))


def test_every_declared_command_resolves_to_a_callable_entrypoint() -> None:
    for spec in COMMAND_SPECS:
        assert callable(spec.resolve_main())


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
    assert {"gate", "bench pipeline", "schema commit"} <= set(COMMANDS)


def test_top_level_command_surface_is_the_folded_twelve() -> None:
    """The fold is only real while nothing re-grows a thirteenth root verb."""
    roots = {spec.command_path[0] for spec in COMMAND_SPECS}
    assert roots == {
        "archive",
        "bench",
        "cache",
        "gate",
        "render",
        "scenario",
        "schema",
        "smoke",
        "status",
        "test",
        "verify",
        "why",
    }


def _documented_examples_with_a_module_parser() -> list[tuple[str, tuple[str, ...], str, argparse.ArgumentParser]]:
    """Every documented example whose command exposes an argparse parser."""
    rows: list[tuple[str, tuple[str, ...], str, argparse.ArgumentParser]] = []
    for spec in COMMAND_SPECS:
        maker = getattr(importlib.import_module(spec.module), "_parser", None)
        if not callable(maker):
            continue
        parser = maker()
        if not isinstance(parser, argparse.ArgumentParser):
            continue
        for example in spec.examples:
            rows.append((spec.name, tuple(spec.argv), example, parser))
    return rows


def test_documented_examples_parse_against_their_own_command_parser() -> None:
    """A documented example that cannot run is worse than no example.

    ``devtools archive tool-outcome-census`` and its ``--json`` sibling were
    both documented without the parser's required ``--archive-root``, so the
    two commands the catalog teaches exited 2 on a usage error.

    Anti-vacuity: drop ``--archive-root`` from either tool-outcome-census
    example and this reddens with ``SystemExit 2``. Nine examples across four
    commands currently carry a module parser, so the loop is not empty.
    """
    rows = _documented_examples_with_a_module_parser()
    assert len(rows) >= 9, f"the example denominator collapsed to {len(rows)}"
    failures: list[tuple[str, str, str]] = []
    for name, argv, example, parser in rows:
        tokens = shlex.split(example)
        assert tokens[: len(argv)] == list(argv), f"{name}: example {example!r} does not invoke {' '.join(argv)!r}"
        try:
            parser.parse_args(tokens[len(argv) :])
        except SystemExit as exit_code:
            failures.append((name, example, f"SystemExit {exit_code.code}"))
    assert not failures, f"documented examples their own parser refuses: {failures}"
