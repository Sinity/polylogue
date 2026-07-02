"""Focused tests for click_app routing and setup internals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli as click_cli
from tests.infra.cli_subprocess import run_cli

QueryParams = dict[str, object]
CliWorkspace = dict[str, Path]


class TestHandleQueryMode:
    def _make_params(self, **overrides: object) -> QueryParams:
        defaults: QueryParams = {
            "conv_id": None,
            "contains": (),
            "exclude_text": (),
            "retrieval_lane": None,
            "origin": None,
            "exclude_origin": None,
            "tag": None,
            "exclude_tag": None,
            "has_type": (),
            "since": None,
            "until": None,
            "title": None,
            "referenced_path": (),
            "action": (),
            "exclude_action": (),
            "action_sequence": None,
            "action_text": (),
            "tool": (),
            "exclude_tool": (),
            "similar_text": None,
            "latest": False,
            "limit": None,
            "sort": None,
            "reverse": False,
            "sample": None,
            "output": None,
            "output_format": None,
            "stream": False,
            "set_meta": (),
            "add_tag": (),
            "plain": False,
            "verbose": False,
            "filter_has_tool_use": False,
            "filter_has_thinking": False,
            "min_messages": None,
            "max_messages": None,
            "min_words": None,
        }
        defaults.update(overrides)
        return defaults

    def _call(self, params: QueryParams) -> tuple[MagicMock, MagicMock]:
        from polylogue.cli.click_app import _handle_query_mode

        mock_ctx = MagicMock()
        mock_ctx.params = params
        mock_ctx.obj = MagicMock()
        mock_ctx.meta = {}

        with (
            patch("polylogue.cli.query.execute_query_request") as mock_execute,
            patch("polylogue.cli.click_app._show_stats") as mock_stats,
        ):
            _handle_query_mode(mock_ctx)
            return mock_execute, mock_stats

    def test_no_args_routes_to_archive_executor(self) -> None:
        # Bare root invocation shows fast status / archive summary instead of
        # constructing a query request.
        mock_execute, mock_stats = self._call(self._make_params())
        mock_execute.assert_not_called()
        mock_stats.assert_called_once()

    def test_verbose_no_args_routes_to_archive_executor(self) -> None:
        mock_execute, mock_stats = self._call(self._make_params(verbose=True))
        mock_execute.assert_not_called()
        mock_stats.assert_called_once()

    def test_query_terms_trigger_query(self) -> None:
        mock_ctx = MagicMock()
        mock_ctx.params = self._make_params()
        mock_ctx.obj = MagicMock()
        mock_ctx.meta = {"polylogue_query_terms": ("error", "handling")}

        with (
            patch("polylogue.cli.query.execute_query_request") as mock_execute,
            patch("polylogue.cli.click_app._show_stats") as mock_stats,
        ):
            from polylogue.cli.click_app import _handle_query_mode

            _handle_query_mode(mock_ctx)

        mock_execute.assert_called_once()
        mock_stats.assert_not_called()

    def test_filter_flags_trigger_query(self) -> None:
        for params in (
            self._make_params(conv_id="abc123"),
            self._make_params(origin="claude-ai-export"),
            self._make_params(tag="important"),
            self._make_params(contains=("error",)),
            self._make_params(has_type=("thinking",)),
            self._make_params(since="2025-01-01"),
            self._make_params(until="2025-12-31"),
            self._make_params(latest=True),
            self._make_params(title="test"),
            self._make_params(referenced_path=("/workspace/polylogue/README.md",)),
            self._make_params(action=("search",)),
            self._make_params(exclude_action=("git",)),
            self._make_params(action_sequence="file_read,file_edit,shell"),
            self._make_params(action_text=("pytest -q",)),
            self._make_params(retrieval_lane="actions", contains=("pytest",)),
            self._make_params(tool=("grep",)),
            self._make_params(exclude_tool=("bash",)),
            self._make_params(similar_text="sqlite locking bug"),
            self._make_params(exclude_text=("noise",)),
            self._make_params(exclude_origin="chatgpt-export"),
            self._make_params(exclude_tag="deprecated"),
            self._make_params(filter_has_tool_use=True),
            self._make_params(min_messages=10),
        ):
            mock_execute, mock_stats = self._call(params)
            mock_execute.assert_called_once()
            mock_stats.assert_not_called()

    def test_output_mode_flags_trigger_query(self) -> None:
        for params in (
            self._make_params(limit=10),
            self._make_params(stream=True),
        ):
            mock_execute, _ = self._call(params)
            mock_execute.assert_called_once()

    def test_modifier_flags_trigger_query(self) -> None:
        for params in (
            self._make_params(add_tag=("review",)),
            self._make_params(set_meta=(("status", "done"),)),
        ):
            mock_execute, mock_stats = self._call(params)
            mock_execute.assert_called_once()
            mock_stats.assert_not_called()

    def test_query_terms_forwarded(self) -> None:
        mock_ctx = MagicMock()
        mock_ctx.params = self._make_params()
        mock_ctx.obj = MagicMock()
        mock_ctx.meta = {"polylogue_query_terms": ("python", "error")}

        with (
            patch("polylogue.cli.query.execute_query_request") as mock_execute,
            patch("polylogue.cli.click_app._show_stats"),
        ):
            from polylogue.cli.click_app import _handle_query_mode

            _handle_query_mode(mock_ctx)

        request = mock_execute.call_args[0][1]
        assert request.query_params()["query"] == ("python", "error")

    def test_root_json_flag_reaches_query_request(self, cli_runner: CliRunner) -> None:
        with (
            patch("polylogue.cli.query.execute_query_request") as mock_execute,
            patch("polylogue.cli.click_app.should_use_plain", return_value=True),
        ):
            result = cli_runner.invoke(click_cli, ["find", "FTS", "--json"], catch_exceptions=False)

        assert result.exit_code == 0
        request = mock_execute.call_args[0][1]
        assert request.query_terms == ("FTS",)
        assert request.params["output_format"] == "json"
        assert request.params["plain"] is True


def test_read_verb_messages_view_forwards_options(cli_runner: CliRunner) -> None:
    """read --view messages routes pagination and projection flags to run_messages."""
    with patch("polylogue.cli.messages.run_messages") as mock_run_messages:
        result = cli_runner.invoke(
            click_cli,
            [
                "--plain",
                "--id",
                "conv-1",
                "read",
                "--view",
                "messages",
                "--limit",
                "2",
                "--offset",
                "1",
                "-f",
                "json",
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    assert mock_run_messages.call_args.kwargs["session_id"] == "conv-1"
    assert mock_run_messages.call_args.kwargs["output_format"] == "json"
    assert mock_run_messages.call_args.kwargs["limit"] == 2


def test_read_verb_messages_requires_id(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(click_cli, ["--plain", "read", "--view", "messages"])

    assert result.exit_code != 0
    assert "requires a session ID" in result.output


def test_root_query_explain_json_outputs_ast_payload(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(
        click_cli,
        [
            "--plain",
            "--format",
            "json",
            "find",
            "sessions where repo:polylogue OR origin:chatgpt-export",
            "--explain",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["source_text"] == "sessions where repo:polylogue OR origin:chatgpt-export"
    assert payload["lowerer"] == "lark-query-expression-to-session-query-spec"
    assert payload["predicate"]["kind"] == "or"
    assert payload["ast"]["entry"] == "boolean"
    assert payload["ast"]["predicate"]["kind"] == "or"
    assert payload["lowering_plan"]["lowerer"] == "lark-query-expression-to-session-query-spec"
    assert payload["selected_units"] == ["session"]
    assert payload["execution_legs"] == ["sql"]
    assert payload["unsupported_nodes"] == []


def test_root_query_explain_json_outputs_terminal_unit_payload(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(
        click_cli,
        [
            "--plain",
            "--format",
            "json",
            "find",
            "messages where role:assistant AND text:timeout",
            "--explain",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["lowerer"] == "lark-query-unit-source-to-terminal-unit"
    assert payload["selected_units"] == ["message"]
    assert payload["execution_legs"] == ["sql", "terminal-message-rows"]
    assert payload["plan_description"] == [
        "terminal unit source: message",
        "compatibility session selector: exists message(...)",
    ]
    assert payload["ast"]["entry"] == "unit_source"
    assert payload["ast"]["unit_source"]["unit"] == "message"
    assert payload["lowering_plan"]["compatibility_selector"] == "exists message(...)"
    assert payload["predicate"]["kind"] == "and"


def test_root_query_explain_json_outputs_terminal_pipeline_stages(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(
        click_cli,
        [
            "--plain",
            "--format",
            "json",
            "find",
            "messages where role:assistant | sort by time desc | limit 2 | offset 3",
            "--explain",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = cast(dict[str, Any], json.loads(result.output))
    ast_payload = cast(dict[str, Any], payload["ast"])
    unit_source = cast(dict[str, Any], ast_payload["unit_source"])
    expected_stages = [
        {"kind": "sort", "sort": {"field": "time", "direction": "desc"}},
        {"kind": "limit", "value": 2},
        {"kind": "offset", "value": 3},
    ]
    assert unit_source["pipeline_stages"] == expected_stages
    lowering_plan = cast(dict[str, Any], payload["lowering_plan"])
    assert lowering_plan["pipeline_stages"] == expected_stages


def test_query_action_read_explain_json_outputs_terminal_stage(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(
        click_cli,
        [
            "--plain",
            "--format",
            "json",
            "--explain",
            "find",
            "repo:polylogue",
            "then",
            "read",
            "--view",
            "messages",
            "--limit",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = cast(dict[str, Any], json.loads(result.output))
    assert "terminal_action" not in payload
    lowering_plan = cast(dict[str, Any], payload["lowering_plan"])
    assert "terminal_action" not in lowering_plan
    pipeline = cast(dict[str, Any], payload["pipeline"])
    assert pipeline["source"]["unit"] == "sessions"
    assert pipeline["stages"] == [
        {
            "kind": "terminal",
            "action": "read",
            "args": {
                "all": False,
                "destination": "terminal",
                "first": False,
                "format": "default",
                "view": "messages",
            },
        }
    ]
    assert lowering_plan["pipeline"] == pipeline
    assert payload["plan_description"][-1] == "terminal stage: read"


def test_query_action_read_explain_uses_default_read_format(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(
        click_cli,
        [
            "--plain",
            "--format",
            "json",
            "--explain",
            "find",
            "repo:polylogue",
            "then",
            "read",
            "--view",
            "messages",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = cast(dict[str, Any], json.loads(result.output))
    pipeline = cast(dict[str, Any], payload["pipeline"])
    stage = cast(dict[str, Any], pipeline["stages"][-1])
    assert cast(dict[str, Any], stage["args"])["format"] == "default"


def test_query_action_read_explain_uses_local_read_format(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(
        click_cli,
        [
            "--plain",
            "--format",
            "json",
            "--explain",
            "find",
            "repo:polylogue",
            "then",
            "read",
            "--view",
            "messages",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = cast(dict[str, Any], json.loads(result.output))
    pipeline = cast(dict[str, Any], payload["pipeline"])
    stage = cast(dict[str, Any], pipeline["stages"][-1])
    assert cast(dict[str, Any], stage["args"])["format"] == "json"


@pytest.mark.parametrize(
    "action_args,expected",
    [
        (
            ["select", "--limit", "5", "--print", "title"],
            {"action": "select", "limit": 5, "print_field": "json", "format": "json"},
        ),
        (
            ["continue"],
            {"action": "continue", "candidates": False, "destination": "terminal", "format": "json"},
        ),
        (
            ["delete", "--dry-run"],
            {"action": "delete", "all": False, "dry_run": True, "format": "json", "yes": False},
        ),
        (
            ["mark", "--tag-add", "reviewed"],
            {
                "action": "mark",
                "all": False,
                "archive": False,
                "first": False,
                "format": "default",
                "note": False,
                "pin": False,
                "star": False,
                "tag_add": ["reviewed"],
                "tag_remove": [],
                "unarchive": False,
                "unpin": False,
                "unstar": False,
            },
        ),
        (
            ["analyze", "--count"],
            {
                "action": "analyze",
                "by": None,
                "cost_outlook": False,
                "count": True,
                "facets": False,
                "format": "json",
                "include_deferred": False,
                "limit": None,
            },
        ),
    ],
)
def test_query_action_explain_json_outputs_terminal_stage_floor(
    cli_runner: CliRunner,
    action_args: list[str],
    expected: dict[str, object],
) -> None:
    result = cli_runner.invoke(
        click_cli,
        [
            "--plain",
            "--format",
            "json",
            "find",
            "repo:polylogue",
            "--explain",
            "then",
            *action_args,
        ],
    )

    assert result.exit_code == 0, result.output
    payload = cast(dict[str, Any], json.loads(result.output))
    assert "terminal_action" not in payload
    lowering_plan = cast(dict[str, Any], payload["lowering_plan"])
    assert "terminal_action" not in lowering_plan
    pipeline = cast(dict[str, Any], payload["pipeline"])
    stage = cast(dict[str, Any], pipeline["stages"][-1])
    assert stage["action"] == expected["action"]
    assert stage["args"] == {key: value for key, value in expected.items() if key != "action"}
    assert lowering_plan["pipeline"] == pipeline
    assert payload["plan_description"][-1] == f"terminal stage: {expected['action']}"


def test_root_query_explain_json_marks_sql_unit_payload(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(
        click_cli,
        [
            "--plain",
            "--format",
            "json",
            "find",
            "runs where session.repo:polylogue AND status:completed",
            "--explain",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["lowerer"] == "lark-query-unit-source-to-terminal-unit"
    assert payload["selected_units"] == ["run"]
    assert payload["execution_legs"] == ["sql", "terminal-run-rows"]
    assert payload["plan_description"] == [
        "terminal unit source: run",
        "compatibility session selector: exists run(...)",
    ]
    assert "compatibility_selector" in payload["lowering_plan"]
    assert payload["ast"]["unit_source"]["unit"] == "run"


def test_root_query_explain_plain_outputs_plan(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(click_cli, ["--plain", "find", 'repo:polylogue "json envelope"', "--explain"])

    assert result.exit_code == 0, result.output
    assert 'query: repo:polylogue "json envelope"' in result.output
    assert "lowerer: lark-query-expression-to-session-query-spec" in result.output
    assert "units: session" in result.output
    assert "execution legs: fts, sql" in result.output
    assert "clauses:" in result.output
    assert "lowering plan:" in result.output
    assert "plan:" in result.output


def test_root_query_explain_accepts_negated_dsl_tokens(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(
        click_cli,
        ["--plain", "--format", "json", "--explain", "find", "repo:polylogue", "--", "-tag:stale"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["source_text"] == "repo:polylogue -tag:stale"
    assert payload["clauses"][1] == {
        "field": "tag",
        "kind": "field",
        "negated": True,
        "value": "stale",
    }


def test_read_views_plain_lists_profile_metadata(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(click_cli, ["--plain", "read", "--views"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "Read views:" in result.output
    assert "recovery" not in result.output
    assert "evidence=required" in result.output
    assert "handoff" in result.output
    assert "options=--limit, --offset" in result.output
    assert "options=--confidence-threshold, --github-api, --otlp, --repo-path, --since-hours" in result.output
    assert "scope=query-set" in result.output
    assert (
        "projection=context,messages; body=authored-dialogue; render=standard; timestamps=include-available"
        in result.output
    )


def test_read_help_groups_options_by_ownership(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(click_cli, ["--plain", "read", "--help"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "Options:" not in result.output
    assert "Projection:" in result.output
    assert "Delivery and format:" in result.output
    assert "Cardinality and pagination:" in result.output
    assert "Context-image projection:" in result.output
    assert "Context and neighbor views:" in result.output
    assert "Correlation view:" in result.output
    assert "Other options:" in result.output
    assert "--views" in result.output
    assert "--render" in result.output
    assert "--projection" in result.output
    assert "--render-layout" in result.output
    assert "--timestamps" in result.output
    assert "--max-tokens" in result.output
    assert "--repo-path" in result.output


def test_read_views_json_outputs_profile_payload(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(click_cli, ["--plain", "read", "--views", "--format", "json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = json.loads(result.output)
    views = {item["view_id"]: item for item in payload["result"]["read_views"]}
    assert payload["status"] == "ok"
    assert views["raw"]["lossiness"] == "raw"
    assert "recovery" not in views
    assert views["context-image"]["successor_handoff"] is True
    assert views["raw"]["cli_options"] == ["limit", "offset"]
    assert views["raw"]["session_policy"] == "required"
    assert views["dialogue"]["accepts_query_set"] is True
    assert views["chronicle"]["accepts_query_set"] is True
    assert views["context-image"]["projection_contract"] == {
        "families": ["context", "messages"],
        "body_policy": "authored-dialogue",
        "render_layout": "standard",
        "timestamp_policy": "include-available",
    }
    assert views["raw"]["projection_contract"] == {
        "families": ["raw"],
        "body_policy": "full",
        "render_layout": "standard",
        "timestamp_policy": "renderer-default",
    }


def test_read_verb_raw_view_forwards_options(cli_runner: CliRunner) -> None:
    """read --view raw routes pagination and format to run_raw."""
    with patch("polylogue.cli.messages.run_raw") as mock_run_raw:
        result = cli_runner.invoke(
            click_cli,
            ["--plain", "--id", "conv-1", "read", "--view", "raw", "--limit", "3", "--offset", "2", "-f", "json"],
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    assert mock_run_raw.call_args.kwargs["session_id"] == "conv-1"
    assert mock_run_raw.call_args.kwargs["limit"] == 3
    assert mock_run_raw.call_args.kwargs["offset"] == 2


def test_read_verb_raw_requires_id(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(click_cli, ["--plain", "read", "--view", "raw"])

    assert result.exit_code != 0
    assert "requires a session ID" in result.output


class TestQueryFirstGroupParseArgs:
    def test_subcommand_dispatches_normally(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["ops", "doctor", "--help"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "health" in result.output.lower() or "repair" in result.output.lower()

    def test_positional_args_become_query_terms(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query_request") as mock_execute:
            cli_runner.invoke(cli, ["find", "hello", "world", "--plain"], catch_exceptions=False)
        request = mock_execute.call_args[0][1]
        assert set(request.query_params().get("query", ())) == {"hello", "world"}

    def test_query_option_before_find_query_stays_query_mode(self, cli_runner: CliRunner) -> None:
        """Filter options followed by `find` + a term stay in query mode (#1842 strict floor)."""
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query_request") as mock_execute:
            cli_runner.invoke(
                cli, ["--origin", "claude-ai-export", "find", "my_search", "--plain"], catch_exceptions=False
            )
        params = mock_execute.call_args[0][1].query_params()
        assert params.get("origin") == "claude-ai-export"
        assert params.get("query") == ("my_search",)

    def test_filter_option_before_subcommand_routes_to_subcommand(self, cli_runner: CliRunner) -> None:
        """Filter options followed by a known subcommand route to that subcommand."""
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(
            cli,
            ["--plain", "--origin", "claude-ai-export", "ops", "insights", "--help"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "insights" in result.output.lower()

    def test_option_args_preserved(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query_request") as mock_execute:
            cli_runner.invoke(
                cli, ["--origin", "claude-ai-export", "find", "search_term", "--plain"], catch_exceptions=False
            )
        params = mock_execute.call_args[0][1].query_params()
        assert params.get("origin") == "claude-ai-export"
        assert "search_term" in params.get("query", ())

    def test_mixed_options_and_positionals(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query_request") as mock_execute:
            cli_runner.invoke(
                cli,
                ["find", "error", "--origin", "claude-ai-export", "handling", "--latest", "--plain"],
                catch_exceptions=False,
            )
        params = mock_execute.call_args[0][1].query_params()
        assert params.get("origin") == "claude-ai-export"
        assert params.get("latest") is True
        assert set(params.get("query", ())) == {"error", "handling"}

    def test_no_args_routes_to_archive_executor(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with (
            patch("polylogue.cli.query.execute_query_request") as mock_execute,
            patch("polylogue.cli.click_app._show_stats") as mock_stats,
        ):
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        mock_execute.assert_not_called()
        mock_stats.assert_called_once()

    def test_help_flag(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["--help"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "polylogue" in result.output.lower()
        assert "ops" in result.output
        assert "--origin" in result.output
        assert "--latest" in result.output
        assert "Subcommands:" not in result.output
        assert (
            "polylogue --origin claude-code-session --since 2026-01-01 find 'repo:polylogue' then analyze --by repo --format json"
            in result.output
        )
        assert (
            "polylogue stats --by repo --origin claude-code-session --since 2026-01-01 --format json"
            not in result.output
        )

    def test_find_help_renders_query_workflow_help(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["find", "--help"], catch_exceptions=False)

        assert result.exit_code == 0
        assert "Search the archive, then optionally run an action." in result.output
        assert "polylogue find QUERY then ACTION" in result.output
        assert "Subcommands:" not in result.output

    def test_query_verb_help_renders_with_root_options(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["--plain", "read", "--help"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "Read matched sessions." in result.output

    def test_root_query_option_after_verb_gets_specific_usage_error(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["analyze", "--by", "origin", "--since", "2026-01-01"], catch_exceptions=False)
        assert result.exit_code == 2
        assert "Query filters and root output flags must appear before the verb." in result.output
        assert "Move --since before `analyze`." in result.output

    def test_root_filter_after_verb_gets_specific_usage_error(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(
            cli, ["analyze", "--by", "origin", "--origin", "claude-ai-export"], catch_exceptions=False
        )
        assert result.exit_code == 2
        assert "Query filters and root output flags must appear before the verb." in result.output
        assert "Move --origin before `analyze`." in result.output


class TestQueryFirstGroupInvoke:
    def test_subcommand_invokes_super(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["ops", "doctor", "--help"])
        assert result.exit_code == 0

    def test_no_subcommand_routes_to_archive_executor(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with (
            patch("polylogue.cli.query.execute_query_request") as mock_execute,
            patch("polylogue.cli.click_app._show_stats") as mock_stats,
        ):
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        mock_execute.assert_not_called()
        mock_stats.assert_called_once()

    def test_analyze_by_subcommand_preserves_grouped_stats_mode(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query_request") as mock_execute:
            result = cli_runner.invoke(cli, ["--plain", "analyze", "--by", "origin"], catch_exceptions=False)

        assert result.exit_code == 0
        params = mock_execute.call_args[0][1].query_params()
        assert params["stats_by"] == "origin"
        assert params["stats_only"] is False

    def test_query_mode_with_positional_args(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query_request") as mock_exec:
            cli_runner.invoke(cli, ["find", "hello", "--plain"], catch_exceptions=False)
        mock_exec.assert_called_once()


class TestStrictCommandFloor:
    """#1842: bare unquoted/unsignalled roots hint instead of silently searching."""

    def test_find_keyword_is_stripped_from_query(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query_request") as mock_exec:
            cli_runner.invoke(cli, ["find", "hello", "world", "--plain"], catch_exceptions=False)
        assert set(mock_exec.call_args[0][1].query_params().get("query", ())) == {"hello", "world"}

    def test_find_then_read_dispatches_read_verb(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["find", "id:abc", "then", "read", "--help"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "read" in result.output.lower()

    def test_bare_multi_token_root_hints_and_does_not_search(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query_request") as mock_exec:
            result = cli_runner.invoke(cli, ["foo", "bar", "--plain"])
        assert result.exit_code == 2
        assert "polylogue find foo bar" in result.output
        mock_exec.assert_not_called()

    def test_bare_single_plain_word_hints(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query_request") as mock_exec:
            result = cli_runner.invoke(cli, ["foo", "--plain"])
        assert result.exit_code == 2
        assert "find" in result.output
        mock_exec.assert_not_called()

    def test_field_expression_root_still_searches(self, cli_runner: CliRunner) -> None:
        """A bare token with field syntax (`repo:x`) is unambiguously a query."""
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query_request") as mock_exec:
            cli_runner.invoke(cli, ["repo:polylogue", "--plain"], catch_exceptions=False)
        mock_exec.assert_called_once()

    def test_quoted_free_text_root_still_searches(self, cli_runner: CliRunner) -> None:
        """A single argv token with internal whitespace can only have come from quoting."""
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query_request") as mock_exec:
            cli_runner.invoke(cli, ["machine learning", "--plain"], catch_exceptions=False)
        mock_exec.assert_called_once()

    def test_double_dash_escape_forces_query(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query_request") as mock_exec:
            cli_runner.invoke(cli, ["--plain", "--", "foo", "bar"], catch_exceptions=False)
        mock_exec.assert_called_once()

    def test_verb_word_after_find_is_a_query_term(self, cli_runner: CliRunner) -> None:
        """`find read` searches for "read"; it does not dispatch the read action (Codex P2)."""
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query_request") as mock_exec:
            cli_runner.invoke(cli, ["find", "read", "--plain"], catch_exceptions=False)
        mock_exec.assert_called_once()
        assert mock_exec.call_args[0][1].query_params().get("query") == ("read",)

    def test_non_leading_find_stays_a_literal_query_term(self, cli_runner: CliRunner) -> None:
        """Only the FIRST positional `find` is the keyword; a later `find` is searchable."""
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query_request") as mock_exec:
            cli_runner.invoke(cli, ["find", "alpha", "find", "--plain"], catch_exceptions=False)
        assert set(mock_exec.call_args[0][1].query_params().get("query", ())) == {"alpha", "find"}


class TestCliSetup:
    def test_verbose_configures_debug_logging(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with (
            patch("polylogue.cli.click_app.configure_logging") as mock_log,
            patch("polylogue.cli.click_app._show_stats"),
        ):
            cli_runner.invoke(cli, ["--verbose", "--plain"], catch_exceptions=False)
        mock_log.assert_called_once_with(verbose=True)

    def test_no_verbose_configures_info_logging(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with (
            patch("polylogue.cli.click_app.configure_logging") as mock_log,
            patch("polylogue.cli.click_app._show_stats"),
        ):
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        mock_log.assert_called_once_with(verbose=False)

    def test_plain_flag_configures_lazy_app_env(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli
        from polylogue.cli.shared.types import AppEnv

        captured_env: dict[str, object] = {}

        def capture_env(env: object, **_: object) -> None:
            captured_env["env"] = env

        with patch("polylogue.cli.click_app._show_stats", side_effect=capture_env):
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        env = captured_env.get("env")
        assert isinstance(env, AppEnv)
        assert env._plain is True

    def test_plain_mode_auto_detection_does_not_announce(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with (
            patch("polylogue.cli.click_app.should_use_plain", return_value=True),
            patch("polylogue.cli.click_app.create_ui", return_value=MagicMock()),
            patch("polylogue.cli.click_app._show_stats"),
        ):
            result = cli_runner.invoke(cli, [], catch_exceptions=False, env={"POLYLOGUE_FORCE_PLAIN": ""})
        assert "Plain output active" not in result.output

    def test_no_announcement_when_plain_flag_explicit(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with (
            patch("polylogue.cli.click_app.should_use_plain", return_value=True),
            patch("polylogue.cli.click_app.create_ui", return_value=MagicMock()),
            patch("polylogue.cli.click_app._show_stats"),
        ):
            result = cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        assert "Plain output active" not in result.output

    def test_no_announcement_when_env_force_plain(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with (
            patch("polylogue.cli.click_app.should_use_plain", return_value=True),
            patch("polylogue.cli.click_app.create_ui", return_value=MagicMock()),
            patch("polylogue.cli.click_app._show_stats"),
        ):
            result = cli_runner.invoke(cli, [], catch_exceptions=False, env={"POLYLOGUE_FORCE_PLAIN": "1"})
        assert "Plain output active" not in result.output

    def test_no_announcement_when_json_requested(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        with (
            patch("polylogue.cli.click_app.should_use_plain", return_value=True),
            patch("polylogue.cli.click_app.create_ui", return_value=MagicMock()),
            patch("polylogue.cli.click_app._show_stats"),
        ):
            result = cli_runner.invoke(cli, ["read", "--all", "--format", "json"], catch_exceptions=False)
        assert "Plain output active" not in result.output

    def test_env_force_plain_false_values_still_do_not_announce(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        for value in ("0", "false", "no"):
            with (
                patch("polylogue.cli.click_app.should_use_plain", return_value=True),
                patch("polylogue.cli.click_app.create_ui", return_value=MagicMock()),
                patch("polylogue.cli.click_app._show_stats"),
            ):
                result = cli_runner.invoke(cli, [], catch_exceptions=False, env={"POLYLOGUE_FORCE_PLAIN": value})
            assert "Plain output active" not in result.output

    def test_ctx_obj_set_to_appenv(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli
        from polylogue.cli.shared.types import AppEnv

        captured_env: dict[str, object] = {}

        def capture_env(env: object, **_: object) -> None:
            captured_env["env"] = env

        with patch("polylogue.cli.click_app._show_stats", side_effect=capture_env):
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        assert isinstance(captured_env.get("env"), AppEnv)


class TestShowStats:
    def test_calls_print_summary_verbose(self) -> None:
        from polylogue.cli.click_app import _show_stats

        env = MagicMock()
        with patch("polylogue.cli.shared.helpers.print_summary") as mock_print:
            _show_stats(env, verbose=True)
        mock_print.assert_called_once_with(env, verbose=True)

    def test_calls_print_summary_not_verbose(self) -> None:
        from polylogue.cli.click_app import _show_stats

        env = MagicMock()
        with (
            patch("polylogue.cli.commands.status.show_fast_status", side_effect=RuntimeError("daemon offline")),
            patch("polylogue.cli.shared.helpers.print_summary") as mock_print,
        ):
            _show_stats(env, verbose=False)
        mock_print.assert_called_once_with(env, verbose=False)


class TestCliMetadata:
    def test_version_flag(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "polylogue" in result.output.lower()

    def test_help_flag_lists_subcommands(self, cli_runner: CliRunner) -> None:
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        for command in ("ops", "import", "continue", "read", "analyze"):
            assert command in result.output
        assert result.output.count("Commands:") == 1

    def test_retained_command_owners_are_registered(self) -> None:
        from polylogue.cli.click_app import cli

        required = {
            "import",
            "init",
            "config",
            "continue",
            "ops",
            # Query verbs
            "read",
            "select",
            "delete",
            "mark",
            "analyze",
        }
        assert required <= set(cli.commands.keys())


# ---------------------------------------------------------------------------
# Merged from test_cli_subprocess.py (2026-03-15)
# ---------------------------------------------------------------------------


def test_run_cli_honors_explicit_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MUTANT_UNDER_TEST", raising=False)
    monkeypatch.delenv("PY_IGNORE_IMPORTMISMATCH", raising=False)
    completed = MagicMock(returncode=0, stdout="ok", stderr="")

    with patch("tests.infra.cli_subprocess.subprocess.run", return_value=completed) as mock_run:
        run_cli(["--help"], cwd=tmp_path)

    command = mock_run.call_args.args[0]
    kwargs = mock_run.call_args.kwargs

    assert kwargs["cwd"] == tmp_path
    assert command[:4] == ["uv", "run", "--project", str(Path(__file__).parents[3])]


def test_run_cli_defaults_cwd_to_project_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MUTANT_UNDER_TEST", raising=False)
    monkeypatch.delenv("PY_IGNORE_IMPORTMISMATCH", raising=False)
    completed = MagicMock(returncode=0, stdout="ok", stderr="")

    with patch("tests.infra.cli_subprocess.subprocess.run", return_value=completed) as mock_run:
        run_cli(["--help"])

    kwargs = mock_run.call_args.kwargs
    assert kwargs["cwd"] == Path(__file__).parents[3]


def test_run_cli_uses_python_bootstrap_under_mutmut(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUTANT_UNDER_TEST", "stats")
    monkeypatch.setenv("PY_IGNORE_IMPORTMISMATCH", "1")
    completed = MagicMock(returncode=0, stdout="ok", stderr="")

    with patch("tests.infra.cli_subprocess.subprocess.run", return_value=completed) as mock_run:
        run_cli(["--help"], cwd=tmp_path)

    command = mock_run.call_args.args[0]
    kwargs = mock_run.call_args.kwargs
    env = kwargs["env"]

    assert kwargs["cwd"] == tmp_path
    assert "python" in Path(command[0]).name
    assert command[1] == "-c"
    assert "ensure_config_loaded" in command[2]
    assert command[3:] == ["--help"]
    assert env["MUTANT_UNDER_TEST"] == "stats"
    assert env["PY_IGNORE_IMPORTMISMATCH"] == "1"


# ---------------------------------------------------------------------------
# Merged from test_command_surfaces.py (2026-03-15)
# ---------------------------------------------------------------------------


class TestDashboardCommand:
    def test_dashboard_launches_app(self, cli_runner: CliRunner, cli_workspace: CliWorkspace) -> None:
        with patch("polylogue.ui.tui.app.PolylogueApp") as mock_app_cls:
            mock_app = MagicMock()
            mock_app_cls.return_value = mock_app
            result = cli_runner.invoke(click_cli, ["--plain", "dashboard"])
        assert result.exit_code == 0
        mock_app.run.assert_called_once()

    def test_dashboard_creates_app_with_facade(
        self,
        cli_runner: CliRunner,
        cli_workspace: CliWorkspace,
    ) -> None:
        with patch("polylogue.ui.tui.app.PolylogueApp") as mock_app_cls:
            mock_app = MagicMock()
            mock_app_cls.return_value = mock_app
            result = cli_runner.invoke(click_cli, ["--plain", "dashboard"])
        assert result.exit_code == 0
        kwargs = mock_app_cls.call_args.kwargs
        assert kwargs["polylogue"] is not None


class TestCompletionsCommand:
    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
    def test_completion_generates_script(self, cli_runner: CliRunner, shell: str) -> None:
        result = cli_runner.invoke(click_cli, ["config", "completions", "--shell", shell])
        assert result.exit_code == 0
        assert "polylogue" in result.output.lower() or "complete" in result.output.lower()

    def test_shell_option_is_required(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(click_cli, ["config", "completions"])
        assert result.exit_code != 0
        assert "missing option" in result.output.lower() or "required" in result.output.lower()

    def test_invalid_shell_rejected(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(click_cli, ["config", "completions", "--shell", "powershell"])
        assert result.exit_code != 0
        assert "invalid value" in result.output.lower() or "choice" in result.output.lower()


class TestMcpServerImport:
    def test_serve_stdio_can_be_imported(self) -> None:
        try:
            from polylogue.mcp.server import serve_stdio

            assert callable(serve_stdio)
        except ImportError:
            pytest.skip("MCP dependencies not installed")
