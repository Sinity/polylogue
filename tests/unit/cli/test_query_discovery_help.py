"""CLI help examples must be rendered from parser-gated query declarations."""

from __future__ import annotations

from click.testing import CliRunner

from polylogue.archive.query.discovery import query_discovery_example
from polylogue.cli.click_app import cli


def test_root_help_resolves_query_discovery_markers() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "@@query:" not in result.output
    for key in ("actions-shell-pytest", "actions-file-edits", "ranked-semantic-text"):
        assert query_discovery_example(key).expression in result.output
