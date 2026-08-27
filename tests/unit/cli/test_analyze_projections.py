"""Proof that aggregate analysis modes have named command routes."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from polylogue.cli.click_app import cli


def test_named_count_and_grouped_projection_dispatch_to_query_executor() -> None:
    """A seeded query scope reaches the canonical executor for both projections."""
    runner = CliRunner()
    with patch("polylogue.cli.query_verbs._execute_query_verb") as execute:
        count = runner.invoke(cli, ["find", "repo:polylogue", "then", "analyze", "count"])
        grouped = runner.invoke(cli, ["find", "repo:polylogue", "then", "analyze", "by", "origin"])

    assert count.exit_code == 0, count.output
    assert grouped.exit_code == 0, grouped.output
    assert execute.call_count == 2
    assert execute.call_args_list[0].args[1].params["count_only"] is True
    assert execute.call_args_list[1].args[1].params["stats_by"] == "origin"


def test_empty_projection_scope_still_dispatches_without_special_case() -> None:
    """An explicitly empty query scope uses the same projection route."""
    runner = CliRunner()
    with patch("polylogue.cli.query_verbs._execute_query_verb") as execute:
        result = runner.invoke(cli, ["find", "repo:does-not-exist", "then", "analyze", "count"])

    assert result.exit_code == 0, result.output
    request = execute.call_args.args[1]
    assert request.query_terms == ("repo:does-not-exist",)
    assert request.params["count_only"] is True
