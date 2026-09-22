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


def test_named_projections_reach_the_shared_analyze_view() -> None:
    """cost-outlook/postmortem/portfolio dispatch without a context collision.

    Anti-vacuity: restoring ``analyze_verb.callback(parent, **kwargs)`` makes
    Click's ``@pass_context`` wrapper inject the current context first and bind
    ``parent`` to the second positional (``count_only``), so every invocation
    below dies with ``TypeError: analyze_verb() got multiple values for
    argument 'count_only'`` and exits non-zero before ``explain`` is reached.
    """
    runner = CliRunner()
    projections = (["postmortem"], ["portfolio"], ["cost-outlook", "--plan", "claude-pro"])
    with patch("polylogue.cli.query.explain_query_request") as explain:
        for projection in projections:
            result = runner.invoke(cli, ["--explain", "find", "repo:polylogue", "then", "analyze", *projection])
            assert result.exit_code == 0, result.output

    assert explain.call_count == len(projections)
    for call in explain.call_args_list:
        action = call.kwargs["terminal_action"]
        assert action["action"] == "analyze"
        # ``parent`` bound to ``count_only`` would make this a Context.
        assert action["count"] is False


def test_root_format_reaches_a_named_analyze_projection() -> None:
    """The root ``--format`` is the contract a named projection must honour."""
    runner = CliRunner()
    with patch("polylogue.cli.query.explain_query_request") as explain:
        result = runner.invoke(
            cli, ["--explain", "--format", "json", "find", "repo:polylogue", "then", "analyze", "postmortem"]
        )

    assert result.exit_code == 0, result.output
    assert explain.call_args.kwargs["terminal_action"]["format"] == "json"


def test_analyze_facets_subcommand_honours_the_root_format() -> None:
    """``--format json ... then analyze facets`` emits JSON, not text.

    Anti-vacuity: restoring ``normalize_output_dialect(output_format) or
    "text"`` makes the asserted format ``text`` while the root request still
    carries ``json``.
    """
    runner = CliRunner()
    with (
        patch("polylogue.cli.query_verbs.run_coroutine_sync", return_value=object()),
        patch("polylogue.cli.query_verbs.emit_facets_response") as emit,
    ):
        result = runner.invoke(cli, ["--format", "json", "find", "repo:polylogue", "then", "analyze", "facets"])

    assert result.exit_code == 0, result.output
    assert emit.call_args.kwargs["output_format"] == "json"


def test_analyze_facets_local_format_still_wins_over_root() -> None:
    """An explicit subcommand ``--format`` is not overridden by the root."""
    runner = CliRunner()
    with (
        patch("polylogue.cli.query_verbs.run_coroutine_sync", return_value=object()),
        patch("polylogue.cli.query_verbs.emit_facets_response") as emit,
    ):
        result = runner.invoke(
            cli, ["--format", "json", "find", "repo:polylogue", "then", "analyze", "facets", "--format", "text"]
        )

    assert result.exit_code == 0, result.output
    assert emit.call_args.kwargs["output_format"] == "text"
