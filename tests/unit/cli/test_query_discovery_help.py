"""CLI help examples must be rendered from parser-gated query declarations."""

from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner

from polylogue.archive.query import discovery
from polylogue.cli.click_app import _render_query_help_examples, cli


def test_root_help_resolves_query_discovery_markers() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "@@query:" not in result.output
    for key in ("actions-shell-pytest", "actions-file-edits", "ranked-semantic-text"):
        assert discovery.query_discovery_example(key).expression in result.output


def test_a_new_help_marker_uses_the_discovery_declaration_without_a_renderer_edit(
    monkeypatch,
) -> None:
    """A marker key is resolved through the declaration lookup, not a copied map."""

    marker = "synthetic-declared-example"
    monkeypatch.setattr(
        discovery,
        "query_discovery_example",
        lambda key: SimpleNamespace(expression=f"declared:{key}"),
    )

    assert _render_query_help_examples(f"polylogue find @@query:{marker}@@") == f"polylogue find declared:{marker}"
