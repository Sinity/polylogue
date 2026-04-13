"""Focused tests for click_app routing and setup internals."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.cli.click_app import cli as click_cli
from polylogue.cli.click_app import mcp_command
from polylogue.scenarios import CorpusSpec
from tests.infra.cli_subprocess import run_cli


class TestHandleQueryMode:
    def _make_params(self, **overrides):
        defaults = {
            "conv_id": None,
            "contains": (),
            "exclude_text": (),
            "retrieval_lane": None,
            "provider": None,
            "exclude_provider": None,
            "tag": None,
            "exclude_tag": None,
            "has_type": (),
            "since": None,
            "until": None,
            "title": None,
            "path_terms": (),
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
            "transform": None,
            "stream": False,
            "dialogue_only": False,
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

    def _call(self, params):
        from polylogue.cli.click_app import _handle_query_mode

        mock_ctx = MagicMock()
        mock_ctx.params = params
        mock_ctx.obj = MagicMock()
        mock_ctx.meta = {}

        with (
            patch("polylogue.cli.query.execute_query") as mock_execute,
            patch("polylogue.cli.click_app._show_stats") as mock_stats,
        ):
            _handle_query_mode(mock_ctx)
            return mock_execute, mock_stats

    def test_no_args_shows_stats(self):
        mock_execute, mock_stats = self._call(self._make_params())
        mock_stats.assert_called_once()
        mock_execute.assert_not_called()

    def test_verbose_stats(self):
        mock_execute, mock_stats = self._call(self._make_params(verbose=True))
        mock_stats.assert_called_once()
        mock_execute.assert_not_called()

    def test_query_terms_trigger_query(self):
        mock_ctx = MagicMock()
        mock_ctx.params = self._make_params()
        mock_ctx.obj = MagicMock()
        mock_ctx.meta = {"polylogue_query_terms": ("error", "handling")}

        with (
            patch("polylogue.cli.query.execute_query") as mock_execute,
            patch("polylogue.cli.click_app._show_stats") as mock_stats,
        ):
            from polylogue.cli.click_app import _handle_query_mode

            _handle_query_mode(mock_ctx)

        mock_execute.assert_called_once()
        mock_stats.assert_not_called()

    def test_filter_flags_trigger_query(self):
        for params in (
            self._make_params(conv_id="abc123"),
            self._make_params(provider="claude-ai"),
            self._make_params(tag="important"),
            self._make_params(contains=("error",)),
            self._make_params(has_type=("thinking",)),
            self._make_params(since="2025-01-01"),
            self._make_params(until="2025-12-31"),
            self._make_params(latest=True),
            self._make_params(title="test"),
            self._make_params(path_terms=("/workspace/polylogue/README.md",)),
            self._make_params(action=("search",)),
            self._make_params(exclude_action=("git",)),
            self._make_params(action_sequence="file_read,file_edit,shell"),
            self._make_params(action_text=("pytest -q",)),
            self._make_params(retrieval_lane="actions", contains=("pytest",)),
            self._make_params(tool=("grep",)),
            self._make_params(exclude_tool=("bash",)),
            self._make_params(similar_text="sqlite locking bug"),
            self._make_params(exclude_text=("noise",)),
            self._make_params(exclude_provider="chatgpt"),
            self._make_params(exclude_tag="deprecated"),
            self._make_params(filter_has_tool_use=True),
            self._make_params(min_messages=10),
        ):
            mock_execute, mock_stats = self._call(params)
            mock_execute.assert_called_once()
            mock_stats.assert_not_called()

    def test_output_mode_flags_trigger_query(self):
        for params in (
            self._make_params(limit=10),
            self._make_params(stream=True),
            self._make_params(dialogue_only=True),
        ):
            mock_execute, _ = self._call(params)
            mock_execute.assert_called_once()

    def test_modifier_flags_trigger_query(self):
        for params in (
            self._make_params(add_tag=("review",)),
            self._make_params(set_meta=(("status", "done"),)),
        ):
            mock_execute, mock_stats = self._call(params)
            mock_execute.assert_called_once()
            mock_stats.assert_not_called()

    def test_query_terms_forwarded(self):
        mock_ctx = MagicMock()
        mock_ctx.params = self._make_params()
        mock_ctx.obj = MagicMock()
        mock_ctx.meta = {"polylogue_query_terms": ("python", "error")}

        with patch("polylogue.cli.query.execute_query") as mock_execute, patch("polylogue.cli.click_app._show_stats"):
            from polylogue.cli.click_app import _handle_query_mode

            _handle_query_mode(mock_ctx)

        params = mock_execute.call_args[0][1]
        assert params["query"] == ("python", "error")


class TestQueryFirstGroupParseArgs:
    def test_subcommand_dispatches_normally(self, cli_runner):
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["doctor", "--help"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "health" in result.output.lower() or "repair" in result.output.lower()

    def test_positional_args_become_query_terms(self, cli_runner):
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query") as mock_execute:
            cli_runner.invoke(cli, ["hello", "world", "--plain"], catch_exceptions=False)
        _, params = mock_execute.call_args[0]
        assert set(params.get("query", ())) == {"hello", "world"}

    def test_query_option_before_bare_word_stays_query_mode(self, cli_runner):
        """Filter options followed by a bare word (not a subcommand name) stay in query mode."""
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query") as mock_execute:
            cli_runner.invoke(cli, ["-p", "claude-ai", "my_search", "--plain"], catch_exceptions=False)
        _, params = mock_execute.call_args[0]
        assert params.get("provider") == "claude-ai"
        assert params.get("query") == ("my_search",)

    def test_filter_option_before_subcommand_routes_to_subcommand(self, cli_runner):
        """Filter options followed by a known subcommand route to that subcommand."""
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["--plain", "-p", "claude-ai", "products", "--help"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "products" in result.output.lower()

    def test_option_args_preserved(self, cli_runner):
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query") as mock_execute:
            cli_runner.invoke(cli, ["-p", "claude-ai", "search_term", "--plain"], catch_exceptions=False)
        _, params = mock_execute.call_args[0]
        assert params.get("provider") == "claude-ai"
        assert "search_term" in params.get("query", ())

    def test_mixed_options_and_positionals(self, cli_runner):
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query") as mock_execute:
            cli_runner.invoke(
                cli,
                ["error", "-p", "claude-ai", "handling", "--latest", "--plain"],
                catch_exceptions=False,
            )
        _, params = mock_execute.call_args[0]
        assert params.get("provider") == "claude-ai"
        assert params.get("latest") is True
        assert set(params.get("query", ())) == {"error", "handling"}

    def test_no_args_shows_stats(self, cli_runner):
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app._show_stats") as mock_stats:
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        mock_stats.assert_called_once()

    def test_help_flag(self, cli_runner):
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["--help"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "polylogue" in result.output.lower()
        assert "products" in result.output
        assert "--provider" in result.output
        assert "--latest" in result.output
        assert "Subcommands:" not in result.output
        assert "polylogue --provider claude-code --since 2026-01-01 stats --by repo --format json" in result.output
        assert "polylogue stats --by repo --provider claude-code --since 2026-01-01 --format json" not in result.output

    def test_root_query_option_after_verb_gets_specific_usage_error(self, cli_runner):
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["stats", "--by", "provider", "--since", "2026-01-01"], catch_exceptions=False)
        assert result.exit_code == 2
        assert "Query filters and root output flags must appear before the verb." in result.output
        assert "Move --since before `stats`." in result.output

    def test_root_filter_after_verb_gets_specific_usage_error(self, cli_runner):
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(
            cli, ["stats", "--by", "provider", "--provider", "claude-ai"], catch_exceptions=False
        )
        assert result.exit_code == 2
        assert "Query filters and root output flags must appear before the verb." in result.output
        assert "Move --provider before `stats`." in result.output


class TestQueryFirstGroupInvoke:
    def test_subcommand_invokes_super(self, cli_runner):
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["doctor", "--help"])
        assert result.exit_code == 0

    def test_no_subcommand_calls_stats_path(self, cli_runner):
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app._show_stats") as mock_stats:
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        mock_stats.assert_called_once()

    def test_stats_by_subcommand_preserves_grouped_stats_mode(self, cli_runner):
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query") as mock_execute:
            result = cli_runner.invoke(cli, ["--plain", "stats", "--by", "provider"], catch_exceptions=False)

        assert result.exit_code == 0
        _, params = mock_execute.call_args[0]
        assert params["stats_by"] == "provider"
        assert params["stats_only"] is False

    def test_query_mode_with_positional_args(self, cli_runner):
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.query.execute_query") as mock_exec:
            cli_runner.invoke(cli, ["hello", "--plain"], catch_exceptions=False)
        mock_exec.assert_called_once()


class TestCliSetup:
    def test_verbose_configures_debug_logging(self, cli_runner):
        from polylogue.cli.click_app import cli

        with (
            patch("polylogue.cli.click_app.configure_logging") as mock_log,
            patch("polylogue.cli.click_app._show_stats"),
        ):
            cli_runner.invoke(cli, ["--verbose", "--plain"], catch_exceptions=False)
        mock_log.assert_called_once_with(verbose=True)

    def test_no_verbose_configures_info_logging(self, cli_runner):
        from polylogue.cli.click_app import cli

        with (
            patch("polylogue.cli.click_app.configure_logging") as mock_log,
            patch("polylogue.cli.click_app._show_stats"),
        ):
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        mock_log.assert_called_once_with(verbose=False)

    def test_plain_flag_creates_plain_ui(self, cli_runner):
        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.click_app.create_ui") as mock_ui, patch("polylogue.cli.click_app._show_stats"):
            mock_ui.return_value = MagicMock()
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        mock_ui.assert_called_once_with(True)

    def test_plain_mode_auto_detection_does_not_announce(self, cli_runner):
        from polylogue.cli.click_app import cli

        with (
            patch("polylogue.cli.click_app.should_use_plain", return_value=True),
            patch("polylogue.cli.click_app.create_ui", return_value=MagicMock()),
            patch("polylogue.cli.click_app._show_stats"),
        ):
            result = cli_runner.invoke(cli, [], catch_exceptions=False, env={"POLYLOGUE_FORCE_PLAIN": ""})
        assert "Plain output active" not in result.output

    def test_no_announcement_when_plain_flag_explicit(self, cli_runner):
        from polylogue.cli.click_app import cli

        with (
            patch("polylogue.cli.click_app.should_use_plain", return_value=True),
            patch("polylogue.cli.click_app.create_ui", return_value=MagicMock()),
            patch("polylogue.cli.click_app._show_stats"),
        ):
            result = cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        assert "Plain output active" not in result.output

    def test_no_announcement_when_env_force_plain(self, cli_runner):
        from polylogue.cli.click_app import cli

        with (
            patch("polylogue.cli.click_app.should_use_plain", return_value=True),
            patch("polylogue.cli.click_app.create_ui", return_value=MagicMock()),
            patch("polylogue.cli.click_app._show_stats"),
        ):
            result = cli_runner.invoke(cli, [], catch_exceptions=False, env={"POLYLOGUE_FORCE_PLAIN": "1"})
        assert "Plain output active" not in result.output

    def test_no_announcement_when_json_requested(self, cli_runner):
        from polylogue.cli.click_app import cli

        with (
            patch("polylogue.cli.click_app.should_use_plain", return_value=True),
            patch("polylogue.cli.click_app.create_ui", return_value=MagicMock()),
            patch("polylogue.cli.click_app._show_stats"),
        ):
            result = cli_runner.invoke(cli, ["list", "--format", "json"], catch_exceptions=False)
        assert "Plain output active" not in result.output

    def test_env_force_plain_false_values_still_do_not_announce(self, cli_runner):
        from polylogue.cli.click_app import cli

        for value in ("0", "false", "no"):
            with (
                patch("polylogue.cli.click_app.should_use_plain", return_value=True),
                patch("polylogue.cli.click_app.create_ui", return_value=MagicMock()),
                patch("polylogue.cli.click_app._show_stats"),
            ):
                result = cli_runner.invoke(cli, [], catch_exceptions=False, env={"POLYLOGUE_FORCE_PLAIN": value})
            assert "Plain output active" not in result.output

    def test_ctx_obj_set_to_appenv(self, cli_runner):
        from polylogue.cli.click_app import cli
        from polylogue.cli.types import AppEnv

        captured_env = {}

        def capture_stats(env, *, verbose=False):
            captured_env["env"] = env

        with patch("polylogue.cli.click_app._show_stats", side_effect=capture_stats):
            cli_runner.invoke(cli, ["--plain"], catch_exceptions=False)
        assert isinstance(captured_env.get("env"), AppEnv)


class TestShowStats:
    def test_calls_print_summary_verbose(self):
        from polylogue.cli.click_app import _show_stats

        env = MagicMock()
        with patch("polylogue.cli.helpers.print_summary") as mock_print:
            _show_stats(env, verbose=True)
        mock_print.assert_called_once_with(env, verbose=True)

    def test_calls_print_summary_not_verbose(self):
        from polylogue.cli.click_app import _show_stats

        env = MagicMock()
        with patch("polylogue.cli.helpers.print_summary") as mock_print:
            _show_stats(env, verbose=False)
        mock_print.assert_called_once_with(env, verbose=False)


class TestCliMetadata:
    def test_version_flag(self, cli_runner):
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "polylogue" in result.output.lower()

    def test_help_flag_lists_subcommands(self, cli_runner):
        from polylogue.cli.click_app import cli

        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        for command in ("run", "doctor", "mcp", "tags", "list", "count", "stats"):
            assert command in result.output
        assert result.output.count("Commands:") == 1

    def test_all_subcommands_registered(self):
        from polylogue.cli.click_app import cli

        expected = {
            "run",
            "doctor",
            "reset",
            "mcp",
            "auth",
            "completions",
            "dashboard",
            "products",
            "audit",
            "schema",
            "tags",
            # Query verbs
            "list",
            "count",
            "stats",
            "open",
            "delete",
        }
        assert set(cli.commands.keys()) == expected


# ---------------------------------------------------------------------------
# Generate command tests (replaces test_demo.py)
# ---------------------------------------------------------------------------


class TestGenerateSeed:
    """``polylogue generate --seed`` creates a full demo environment."""

    def test_seed_creates_database(self, cli_runner, tmp_path):
        result = cli_runner.invoke(
            click_cli,
            [
                "audit",
                "generate",
                "--seed",
                "-o",
                str(tmp_path),
                "-n",
                "1",
                "-p",
                "chatgpt",
            ],
        )
        assert result.exit_code == 0
        db_path = tmp_path / "data" / "polylogue" / "polylogue.db"
        assert db_path.exists()
        assert (tmp_path / "home").exists()
        assert (tmp_path / "data" / "polylogue" / "inbox" / "chatgpt").exists()

    def test_seed_restores_environment(self, cli_runner, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", "/tmp/original-data")
        monkeypatch.setenv("HOME", "/tmp/original-home")
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", "/tmp/original-archive")

        with patch(
            "polylogue.pipeline.runner.run_sources",
            new=AsyncMock(
                return_value=type(
                    "_Result",
                    (),
                    {"counts": {"conversations": 0, "messages": 0}},
                )()
            ),
        ):
            result = cli_runner.invoke(
                click_cli,
                ["audit", "generate", "--seed", "-o", str(tmp_path), "-n", "1", "-p", "chatgpt"],
            )

        assert result.exit_code == 0
        assert os.environ["HOME"] == "/tmp/original-home"
        assert os.environ["XDG_DATA_HOME"] == "/tmp/original-data"
        assert os.environ["POLYLOGUE_ARCHIVE_ROOT"] == "/tmp/original-archive"

    def test_env_only_requires_seed(self, cli_runner):
        result = cli_runner.invoke(click_cli, ["audit", "generate", "--env-only"])
        assert result.exit_code != 0
        assert "requires --seed" in result.output.lower() or "error" in result.output.lower()

    def test_seed_env_only_exports_isolated_workspace(self, cli_runner, tmp_path):
        result = cli_runner.invoke(
            click_cli,
            ["audit", "generate", "--seed", "--env-only", "-o", str(tmp_path), "-n", "1", "-p", "chatgpt"],
        )

        assert result.exit_code == 0
        assert f'export HOME="{tmp_path / "home"}"' in result.output
        assert f'export XDG_CONFIG_HOME="{tmp_path / "config"}"' in result.output
        assert f'export XDG_CACHE_HOME="{tmp_path / "cache"}"' in result.output

    def test_inferred_corpus_generation_uses_unique_prefixes_per_same_provider_spec(self, cli_runner, tmp_path):
        inferred_specs = (
            CorpusSpec(
                provider="chatgpt",
                package_version="v1",
                count=1,
                messages_min=4,
                messages_max=4,
                seed=7,
                profile_family_ids=("cluster-a",),
            ),
            CorpusSpec(
                provider="chatgpt",
                package_version="v1",
                count=1,
                messages_min=4,
                messages_max=4,
                seed=8,
                profile_family_ids=("cluster-b",),
            ),
        )

        with patch(
            "polylogue.schemas.operator_inference.list_inferred_corpus_specs",
            return_value=inferred_specs,
        ):
            result = cli_runner.invoke(
                click_cli,
                ["audit", "generate", "--corpus-source", "inferred", "-o", str(tmp_path), "-n", "1", "-p", "chatgpt"],
            )

        assert result.exit_code == 0
        provider_dir = tmp_path / "chatgpt"
        assert (provider_dir / "sample-v1-cluster-a-00.json").exists()
        assert (provider_dir / "sample-v1-cluster-b-00.json").exists()

    def test_inferred_corpus_generation_fails_when_no_specs_match(self, cli_runner, tmp_path):
        with patch(
            "polylogue.schemas.operator_inference.list_inferred_corpus_specs",
            return_value=(),
        ):
            result = cli_runner.invoke(
                click_cli,
                ["audit", "generate", "--corpus-source", "inferred", "-o", str(tmp_path), "-p", "chatgpt"],
            )

        assert result.exit_code != 0
        assert "No corpus scenarios matched" in result.output


# ---------------------------------------------------------------------------
# QA command tests
# ---------------------------------------------------------------------------


class TestQaCommand:
    """``polylogue qa`` flag validation and wiring."""

    def test_only_and_skip_mutually_exclusive(self, cli_runner):
        result = cli_runner.invoke(
            click_cli,
            ["audit", "--only", "audit", "--skip", "exercises"],
        )
        assert result.exit_code != 0

    def test_snapshot_from_skips_qa(self, cli_runner, tmp_path):
        """--snapshot-from archives a directory without running QA."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "report.json").write_text("{}")

        output_root = tmp_path / "snapshots"

        result = cli_runner.invoke(
            click_cli,
            ["audit", "--snapshot-from", str(source), "--report-dir", str(output_root)],
        )
        assert result.exit_code == 0
        # A snapshot directory should have been created
        assert any(output_root.iterdir())

    def test_qa_help_shows_key_flags(self, cli_runner):
        result = cli_runner.invoke(click_cli, ["audit", "--help"])
        assert result.exit_code == 0
        assert "--live" in result.output
        assert "--only" in result.output
        assert "--skip" in result.output
        assert "--snapshot" in result.output

    def test_json_output_uses_composed_qa_session_payload(self, cli_runner):
        from polylogue.lib.outcomes import OutcomeCheck, OutcomeStatus
        from polylogue.schemas.audit_models import AuditReport
        from polylogue.schemas.verification_models import ArtifactProofReport, ProviderArtifactProof
        from polylogue.showcase.qa_runner import QAResult

        qa_result = QAResult(
            audit_report=AuditReport(
                checks=[
                    OutcomeCheck(name="privacy", status=OutcomeStatus.OK, summary="ok"),
                ]
            ),
            proof_report=ArtifactProofReport(
                providers={
                    "chatgpt": ProviderArtifactProof(
                        provider="chatgpt",
                        total_records=1,
                        contract_backed_records=1,
                    )
                },
                total_records=1,
            ),
            exercises_skipped=True,
            invariants_skipped=True,
        )

        with patch("polylogue.showcase.qa_runner.run_qa_session", return_value=qa_result):
            result = cli_runner.invoke(click_cli, ["audit", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["audit"]["status"] == "ok"
        assert payload["showcase"]["status"] == "skip"
        assert payload["overall_status"] == "ok"

    def test_audit_only_skips_artifact_proof(self, cli_runner):
        from polylogue.lib.outcomes import OutcomeCheck, OutcomeStatus
        from polylogue.schemas.audit_models import AuditReport
        from polylogue.showcase.qa_runner import QAResult

        qa_result = QAResult(
            audit_report=AuditReport(
                checks=[
                    OutcomeCheck(name="privacy", status=OutcomeStatus.OK, summary="ok"),
                ]
            ),
            proof_skipped=True,
            exercises_skipped=True,
            invariants_skipped=True,
        )

        with patch("polylogue.showcase.qa_runner.run_qa_session", return_value=qa_result) as mock_run:
            result = cli_runner.invoke(click_cli, ["audit", "--only", "audit", "--json"])

        assert result.exit_code == 0
        assert mock_run.call_args.args[0].skip_proof is True
        payload = json.loads(result.output)
        assert payload["proof"]["status"] == "skip"
        assert payload["proof"]["skipped"] is True
        assert payload["overall_status"] == "ok"

    def test_exercises_only_skips_artifact_proof(self, cli_runner):
        from polylogue.scenarios import polylogue_execution
        from polylogue.showcase.exercises import Exercise, Validation
        from polylogue.showcase.qa_runner import QAResult
        from polylogue.showcase.runner import ExerciseResult, ShowcaseResult

        qa_result = QAResult(
            proof_skipped=True,
            audit_skipped=True,
            showcase_result=ShowcaseResult(
                results=[
                    ExerciseResult(
                        exercise=Exercise(
                            name="smoke",
                            group="structural",
                            description="smoke",
                            execution=polylogue_execution("--help"),
                            validation=Validation(),
                        ),
                        passed=True,
                        exit_code=0,
                        output="ok",
                        duration_ms=1.0,
                    )
                ],
                total_duration_ms=1.0,
            ),
            invariants_skipped=True,
        )

        with patch("polylogue.showcase.qa_runner.run_qa_session", return_value=qa_result) as mock_run:
            result = cli_runner.invoke(click_cli, ["audit", "--only", "exercises", "--json"])

        assert result.exit_code == 0
        assert mock_run.call_args.args[0].skip_proof is True
        payload = json.loads(result.output)
        assert payload["proof"]["status"] == "skip"
        assert payload["proof"]["skipped"] is True


# ---------------------------------------------------------------------------
# Merged from test_cli_subprocess.py (2026-03-15)
# ---------------------------------------------------------------------------


def test_run_cli_honors_explicit_cwd(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MUTANT_UNDER_TEST", raising=False)
    monkeypatch.delenv("PY_IGNORE_IMPORTMISMATCH", raising=False)
    completed = MagicMock(returncode=0, stdout="ok", stderr="")

    with patch("tests.infra.cli_subprocess.subprocess.run", return_value=completed) as mock_run:
        run_cli(["--help"], cwd=tmp_path)

    command = mock_run.call_args.args[0]
    kwargs = mock_run.call_args.kwargs

    assert kwargs["cwd"] == tmp_path
    assert command[:4] == ["uv", "run", "--project", str(Path(__file__).parents[3])]


def test_run_cli_defaults_cwd_to_project_root(monkeypatch) -> None:
    monkeypatch.delenv("MUTANT_UNDER_TEST", raising=False)
    monkeypatch.delenv("PY_IGNORE_IMPORTMISMATCH", raising=False)
    completed = MagicMock(returncode=0, stdout="ok", stderr="")

    with patch("tests.infra.cli_subprocess.subprocess.run", return_value=completed) as mock_run:
        run_cli(["--help"])

    kwargs = mock_run.call_args.kwargs
    assert kwargs["cwd"] == Path(__file__).parents[3]


def test_run_cli_uses_python_bootstrap_under_mutmut(tmp_path: Path, monkeypatch) -> None:
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
    def test_dashboard_launches_app(self, cli_runner, cli_workspace) -> None:
        with patch("polylogue.ui.tui.app.PolylogueApp") as mock_app_cls:
            mock_app = MagicMock()
            mock_app_cls.return_value = mock_app
            result = cli_runner.invoke(click_cli, ["--plain", "dashboard"])
        assert result.exit_code == 0
        mock_app.run.assert_called_once()

    def test_dashboard_creates_app_with_repository(self, cli_runner, cli_workspace) -> None:
        with patch("polylogue.ui.tui.app.PolylogueApp") as mock_app_cls:
            mock_app = MagicMock()
            mock_app_cls.return_value = mock_app
            result = cli_runner.invoke(click_cli, ["--plain", "dashboard"])
        assert result.exit_code == 0
        kwargs = mock_app_cls.call_args.kwargs
        assert kwargs["repository"] is not None


class TestCompletionsCommand:
    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
    def test_completion_generates_script(self, cli_runner, shell: str) -> None:
        result = cli_runner.invoke(click_cli, ["completions", "--shell", shell])
        assert result.exit_code == 0
        assert "polylogue" in result.output.lower() or "complete" in result.output.lower()

    def test_shell_option_is_required(self, cli_runner) -> None:
        result = cli_runner.invoke(click_cli, ["completions"])
        assert result.exit_code != 0
        assert "missing option" in result.output.lower() or "required" in result.output.lower()

    def test_invalid_shell_rejected(self, cli_runner) -> None:
        result = cli_runner.invoke(click_cli, ["completions", "--shell", "powershell"])
        assert result.exit_code != 0
        assert "invalid value" in result.output.lower() or "choice" in result.output.lower()


class TestMcpCommandUnit:
    @pytest.fixture
    def mock_env(self):
        mock_ui = MagicMock()
        mock_ui.plain = True
        mock_ui.console = MagicMock()
        env = MagicMock()
        env.ui = mock_ui
        return env

    def test_default_transport_is_stdio(self, cli_runner, mock_env) -> None:
        with patch("polylogue.mcp.server.serve_stdio") as mock_serve:
            result = cli_runner.invoke(mcp_command, [], obj=mock_env)
        mock_serve.assert_called_once()
        assert result.exit_code == 0

    def test_explicit_stdio_transport_works(self, cli_runner, mock_env) -> None:
        with patch("polylogue.mcp.server.serve_stdio") as mock_serve:
            result = cli_runner.invoke(mcp_command, ["--transport", "stdio"], obj=mock_env)
        mock_serve.assert_called_once()
        assert result.exit_code == 0

    def test_missing_mcp_dependencies_error(self, cli_runner, mock_env) -> None:
        with patch.dict(sys.modules, {"polylogue.mcp.server": None}):

            def mock_import(*args, **kwargs):
                raise ImportError("No module named 'mcp'")

            with patch("builtins.__import__", side_effect=mock_import):
                result = cli_runner.invoke(mcp_command, [], obj=mock_env)
        assert result.exit_code != 0 or mock_env.ui.console.print.called

    def test_unsupported_transport_error(self, cli_runner, mock_env) -> None:
        result = cli_runner.invoke(click_cli, ["mcp", "--transport", "http"])
        assert result.exit_code != 0

    def test_mcp_help_shows_description(self, cli_runner) -> None:
        result = cli_runner.invoke(click_cli, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "mcp" in result.output.lower()
        assert "server" in result.output.lower() or "protocol" in result.output.lower()


class TestMcpServerImport:
    def test_serve_stdio_can_be_imported(self) -> None:
        try:
            from polylogue.mcp.server import serve_stdio

            assert callable(serve_stdio)
        except ImportError:
            pytest.skip("MCP dependencies not installed")
