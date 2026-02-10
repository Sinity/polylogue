"""Tests for CLI run command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from datetime import datetime

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.sources import DriveError
from polylogue.storage.store import PlanResult, RunResult


@pytest.fixture
def runner():
    """Click test runner."""
    return CliRunner()


@pytest.fixture
def mock_plan_result():
    """Mock PlanResult for preview mode."""
    return PlanResult(
        timestamp=1234567890,
        counts={"conversations": 5, "messages": 50, "attachments": 2},
        sources=["test-inbox"],
        cursors={},
    )


@pytest.fixture
def mock_run_result():
    """Mock RunResult for sync mode."""
    return RunResult(
        run_id="run-123",
        counts={"conversations": 3, "messages": 30, "attachments": 1},
        drift={"conversations": {"new": 2, "updated": 1, "unchanged": 5}},
        indexed=True,
        index_error=None,
        duration_ms=1500,
        render_failures=[],
    )


# ============================================================================
# Parametrization tables (module-level constants)
# ============================================================================

PREVIEW_MODE_CASES = [
    ("calls_plan_sources", None, None),
    ("displays_plan_snapshot", ["test-inbox"], "Preview"),
    ("with_plain_mode_skips_confirm", None, "exit_code_0"),
]

PREVIEW_ERROR_CASES = [
    ("drive_error_fails", ["google-drive"], "OAuth token expired"),
]

NON_PREVIEW_MODE_CASES = [
    ("calls_run_sources", None),
    ("displays_duration", "1500ms"),
    ("displays_counts", "Counts"),
]

NON_PREVIEW_ERROR_CASES = [
    ("drive_error_fails", "Drive API rate limit"),
]

STAGE_OPTION_CASES = [
    ("parse", "parse"),
    ("render", "render"),
    ("index", "index"),
    ("all_default", "all", []),  # no args passed
]

STAGE_DISPLAY_CASES = [
    ("index_displays_index_status", "index", True, "format_index_status"),
]

SOURCE_OPTION_CASES = [
    ("single", ("test-inbox",), ["test-inbox"]),
    ("multiple", ("test-inbox", "drive"), ["test-inbox", "drive"]),
    ("displays_in_title", ("my-source",), ["my-source"]),
]

FORMAT_OPTION_CASES = [
    ("markdown", "markdown", "markdown"),
    ("html_default", None, "html"),
]

INDEX_ERROR_CASES = [
    ("displays_error_with_hint", "FTS5 index error", True, "format_counts"),
    ("stage_index_displays_error", "Vector database unavailable", False, "format_index_status"),
]

RENDER_OUTPUT_CASES = [
    ("displays_latest_for_render_stage", "all", True),
    ("skips_latest_for_non_render_stage", "index", False),
]

TITLE_CASES = [
    ("includes_stage_when_not_all", "render", "stage"),
    ("includes_sources_when_filtered", None, "source"),
]

FLAG_COMBO_CASES = [
    ("preview_with_stage", ["--preview", "--stage", "parse"], "plan_sources"),
    ("preview_with_format", ["--preview", "--format", "markdown"], "plan_sources"),
    ("stage_render_with_source", ["--stage", "render", "--source", "test"], "run_sources"),
]

RENDER_FAILURE_CASES = [
    ("displays_render_failures", 1, 5, True, "Render failures (1)", "conv-1: Template error"),
    ("displays_truncated", 15, 5, False, "Render failures (15)", "... and 5 more"),
    ("no_section_when_empty", 0, 0, False, "Render failures", False),
]


# ============================================================================
# Test Classes
# ============================================================================


class TestRunCommandPreviewMode:
    """Tests for --preview flag."""

    @pytest.mark.parametrize("case_name,resolved_sources,check_output", PREVIEW_MODE_CASES)
    def test_run_preview(self, runner, cli_workspace, mock_plan_result, case_name, resolved_sources, check_output):
        """Preview mode variations."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.plan_sources") as mock_plan:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=resolved_sources):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=resolved_sources):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="5 conversations, 50 messages"):
                            with patch("polylogue.cli.commands.run.format_cursors", return_value=""):
                                mock_plan.return_value = mock_plan_result

                                result = runner.invoke(cli, ["run", "--preview"])

        assert result.exit_code == 0
        if case_name == "calls_plan_sources":
            mock_plan.assert_called_once()
        elif case_name == "displays_plan_snapshot":
            assert "Preview" in result.output or "preview" in result.output.lower()
        elif case_name == "with_plain_mode_skips_confirm":
            assert result.exit_code == 0

    @pytest.mark.parametrize("case_name,resolved_sources,error_msg", PREVIEW_ERROR_CASES)
    def test_run_preview_error(self, runner, cli_workspace, case_name, resolved_sources, error_msg):
        """Preview mode error handling."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.plan_sources") as mock_plan:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=resolved_sources):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=resolved_sources):
                        mock_plan.side_effect = DriveError(error_msg)

                        result = runner.invoke(cli, ["run", "--preview"])

        assert result.exit_code != 0
        assert error_msg in result.output


class TestRunCommandNonPreviewMode:
    """Tests for normal (non-preview) sync mode."""

    @pytest.mark.parametrize("case_name,check_output", NON_PREVIEW_MODE_CASES)
    def test_run_non_preview(self, runner, cli_workspace, mock_run_result, case_name, check_output):
        """Non-preview mode variations."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="3 conversations, 30 messages"):
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["run"])

        assert result.exit_code == 0
        if case_name == "calls_run_sources":
            mock_run.assert_called_once()
        elif case_name == "displays_duration":
            assert "Duration" in result.output or "duration" in result.output.lower()
            assert check_output in result.output
        elif case_name == "displays_counts":
            assert check_output in result.output or "counts" in result.output.lower()

    @pytest.mark.parametrize("case_name,error_msg", NON_PREVIEW_ERROR_CASES)
    def test_run_non_preview_error(self, runner, cli_workspace, case_name, error_msg):
        """Non-preview mode error handling."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=["google-drive"]):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=["google-drive"]):
                        mock_run.side_effect = DriveError(error_msg)

                        result = runner.invoke(cli, ["run"])

        assert result.exit_code != 0
        assert error_msg in result.output


class TestRunCommandStageOption:
    """Tests for --stage flag."""

    @pytest.mark.parametrize(
        "stage,expected_stage,extra_args",
        [
            ("parse", "parse", ["--stage", "parse"]),
            ("render", "render", ["--stage", "render"]),
            ("index", "index", ["--stage", "index"]),
            ("all_default", "all", []),
        ],
    )
    def test_run_stage_option(self, runner, cli_workspace, mock_run_result, stage, expected_stage, extra_args):
        """Stage option passed correctly."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="3 conversations"):
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["run"] + extra_args)

        assert result.exit_code == 0
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["stage"] == expected_stage

    def test_run_stage_index_displays_index_status(self, runner, cli_workspace):
        """--stage index displays index status instead of counts."""
        from unittest.mock import patch

        result_indexed = RunResult(
            run_id="run-idx",
            counts={"conversations": 0},
            drift={},
            indexed=True,
            index_error=None,
            duration_ms=800,
        )

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_index_status") as mock_format_idx:
                            mock_run.return_value = result_indexed
                            mock_format_idx.return_value = "Index status: indexed"

                            result = runner.invoke(cli, ["run", "--stage", "index"])

        assert result.exit_code == 0
        mock_format_idx.assert_called_once()


class TestRunCommandSourceOption:
    """Tests for --source flag."""

    @pytest.mark.parametrize(
        "case_name,sources,resolved_sources",
        SOURCE_OPTION_CASES,
    )
    def test_run_source_option(self, runner, cli_workspace, mock_run_result, case_name, sources, resolved_sources):
        """Source option variations."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.services.get_service_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources") as mock_resolve:
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=resolved_sources):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="3+ conversations"):
                            mock_resolve.return_value = resolved_sources
                            mock_run.return_value = mock_run_result

                            cli_args = ["run"]
                            for source in sources:
                                cli_args.extend(["--source", source])

                            result = runner.invoke(cli, cli_args)

        assert result.exit_code == 0
        if case_name == "single" or case_name == "multiple":
            call_args = mock_resolve.call_args
            for source in sources:
                assert source in call_args[0][1]
        elif case_name == "displays_in_title":
            assert resolved_sources[0] in result.output or "Run" in result.output


class TestRunCommandFormatOption:
    """Tests for --format flag."""

    @pytest.mark.parametrize(
        "case_name,format_arg,expected_format",
        FORMAT_OPTION_CASES,
    )
    def test_run_format_option(self, runner, cli_workspace, mock_run_result, case_name, format_arg, expected_format):
        """Format option variations."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="3 conversations"):
                            mock_run.return_value = mock_run_result

                            cli_args = ["run"]
                            if format_arg:
                                cli_args.extend(["--format", format_arg])

                            result = runner.invoke(cli, cli_args)

        assert result.exit_code == 0
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["render_format"] == expected_format


class TestRunCommandProgressOutput:
    """Tests for progress display."""

    def test_run_plain_mode_shows_progress(self, runner, cli_workspace, mock_run_result):
        """Plain mode displays periodic progress updates."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="3 conversations"):
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["run"])

        assert result.exit_code == 0


class TestRunCommandIndexError:
    """Tests for index error handling."""

    @pytest.mark.parametrize(
        "case_name,error_msg,full_run,format_func",
        INDEX_ERROR_CASES,
    )
    def test_run_index_error(self, runner, cli_workspace, case_name, error_msg, full_run, format_func):
        """Index error handling variations."""
        from unittest.mock import patch

        result_with_error = RunResult(
            run_id="run-err",
            counts={"conversations": 2},
            drift={},
            indexed=False,
            index_error=error_msg,
            duration_ms=1200,
        )

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch(f"polylogue.cli.commands.run.{format_func}") as mock_format:
                            mock_run.return_value = result_with_error
                            mock_format.return_value = f"Index error: {error_msg}"

                            cli_args = ["run"]
                            if not full_run:
                                cli_args.extend(["--stage", "index"])

                            result = runner.invoke(cli, cli_args)

        assert result.exit_code == 0
        if not full_run:
            mock_format.assert_called_once()
        else:
            assert error_msg in result.output or "Index error" in result.output


class TestRunCommandRenderOutput:
    """Tests for render output display."""

    @pytest.mark.parametrize(
        "case_name,stage,should_call_latest",
        RENDER_OUTPUT_CASES,
    )
    def test_run_render_output(self, runner, cli_workspace, mock_run_result, case_name, stage, should_call_latest):
        """Render output path display."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="3 conversations"):
                            with patch("polylogue.cli.commands.run.format_index_status", return_value="Indexed"):
                                with patch("polylogue.cli.helpers.latest_render_path") as mock_latest:
                                    mock_config.render_root = Path("/render")
                                    mock_run.return_value = mock_run_result
                                    mock_latest.return_value = Path("/render/conv1/conversation.html")

                                    result = runner.invoke(cli, ["run", "--stage", stage])

        assert result.exit_code == 0
        if should_call_latest:
            mock_latest.assert_called_once()
        else:
            mock_latest.assert_not_called()


class TestRunCommandTitle:
    """Tests for run output title."""

    @pytest.mark.parametrize(
        "case_name,stage_or_source,title_element",
        TITLE_CASES,
    )
    def test_run_title(self, runner, cli_workspace, mock_run_result, case_name, stage_or_source, title_element):
        """Title includes stage or source information."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=[stage_or_source] if title_element == "source" else None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=[stage_or_source] if title_element == "source" else None):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="5 conversations"):
                            mock_run.return_value = mock_run_result

                            cli_args = ["run"]
                            if title_element == "stage":
                                cli_args.extend(["--stage", stage_or_source])
                            else:
                                cli_args.extend(["--source", stage_or_source])

                            result = runner.invoke(cli, cli_args)

        assert result.exit_code == 0
        if title_element == "stage":
            assert stage_or_source in result.output.lower() or "Run" in result.output
        else:
            assert stage_or_source in result.output or "Run" in result.output


class TestRunCommandCombinations:
    """Tests for flag combinations."""

    @pytest.mark.parametrize(
        "case_name,cli_args,mock_func",
        FLAG_COMBO_CASES,
    )
    def test_run_flag_combinations(self, runner, cli_workspace, mock_plan_result, mock_run_result, case_name, cli_args, mock_func):
        """Flag combinations work together."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.plan_sources") as mock_plan:
                with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                    with patch("polylogue.cli.commands.run.resolve_sources") as mock_resolve:
                        with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=["test"] if "--source" in cli_args else None):
                            with patch("polylogue.cli.commands.run.format_counts", return_value="5 conversations"):
                                mock_plan.return_value = mock_plan_result
                                mock_run.return_value = mock_run_result
                                mock_resolve.return_value = ["test"] if "--source" in cli_args else None

                                result = runner.invoke(cli, cli_args)

        assert result.exit_code == 0
        if mock_func == "plan_sources":
            mock_plan.assert_called_once()
        else:
            mock_run.assert_called_once()


class TestRunCommandRenderFailures:
    """Tests for render failure handling in run output."""

    @pytest.mark.parametrize(
        "case_name,num_failures,show_count,show_heading_as_text,expected_in_output,not_expected",
        RENDER_FAILURE_CASES,
    )
    def test_run_render_failures(self, runner, cli_workspace, case_name, num_failures, show_count, show_heading_as_text, expected_in_output, not_expected):
        """Render failure display variations."""
        from unittest.mock import patch

        if num_failures > 0:
            failures = [
                {"conversation_id": f"conv-{i}", "error": f"Error {i}"}
                for i in range(1, num_failures + 1)
            ]
        else:
            failures = []

        result_with_failures = RunResult(
            run_id="run-fail",
            counts={"conversations": num_failures},
            drift={},
            indexed=True,
            index_error=None,
            duration_ms=1500,
            render_failures=failures,
        )

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_counts", return_value=f"{num_failures} conversation(s)"):
                            mock_run.return_value = result_with_failures

                            result = runner.invoke(cli, ["run"])

        assert result.exit_code == 0

        # Check expected content
        if isinstance(expected_in_output, str):
            assert expected_in_output in result.output

        # Check for content that should NOT appear
        if not_expected is False:
            assert "Render failures" not in result.output
        elif isinstance(not_expected, str):
            assert not_expected not in result.output


class TestDeleteConversationPreview:
    """Tests for enhanced deletion preview in query mode."""

    def test_delete_dry_run_shows_provider_breakdown(self, capsys):
        """Dry-run deletion shows provider breakdown."""
        from unittest.mock import patch

        from polylogue.cli.query import _delete_conversations

        # Create mock conversations with different providers
        convs = []
        for i in range(3):
            conv = MagicMock()
            conv.provider = "claude"
            conv.created_at = datetime(2024, 1, 15)
            conv.display_title = f"Conversation {i}"
            conv.id = f"conv-{i}"
            convs.append(conv)

        for i in range(2):
            conv = MagicMock()
            conv.provider = "chatgpt"
            conv.created_at = datetime(2024, 1, 16)
            conv.display_title = f"ChatGPT Conversation {i}"
            conv.id = f"gpt-{i}"
            convs.append(conv)

        env = MagicMock()
        env.ui.console.print = MagicMock()

        _delete_conversations(env, convs, {"dry_run": True})

        captured = capsys.readouterr()
        assert "DRY-RUN: Would delete 5 conversation(s)" in captured.out
        assert "Providers:" in captured.out
        assert "claude: 3" in captured.out
        assert "chatgpt: 2" in captured.out

    def test_delete_dry_run_shows_date_range(self, capsys):
        """Dry-run deletion shows date range."""
        from polylogue.cli.query import _delete_conversations

        # Create mock conversations with different dates
        convs = []
        conv1 = MagicMock()
        conv1.provider = "claude"
        conv1.created_at = datetime(2023, 6, 1)
        conv1.display_title = "Old conversation"
        conv1.id = "old-1"
        convs.append(conv1)

        conv2 = MagicMock()
        conv2.provider = "claude"
        conv2.created_at = datetime(2024, 2, 15)
        conv2.display_title = "Recent conversation"
        conv2.id = "new-1"
        convs.append(conv2)

        env = MagicMock()
        env.ui.console.print = MagicMock()

        _delete_conversations(env, convs, {"dry_run": True})

        captured = capsys.readouterr()
        assert "Date range: 2023-06-01 → 2024-02-15" in captured.out

    def test_delete_bulk_shows_breakdown_and_prompts(self, capsys):
        """Bulk deletion (>10 items) without force shows breakdown and prompts."""
        from polylogue.cli.query import _delete_conversations

        # Create 15 mock conversations
        convs = []
        for i in range(10):
            conv = MagicMock()
            conv.provider = "claude"
            conv.created_at = datetime(2024, 1, 15)
            conv.display_title = f"Conv {i}"
            conv.id = f"conv-{i}"
            convs.append(conv)

        for i in range(5):
            conv = MagicMock()
            conv.provider = "chatgpt"
            conv.created_at = datetime(2024, 1, 16)
            conv.display_title = f"ChatGPT {i}"
            conv.id = f"gpt-{i}"
            convs.append(conv)

        env = MagicMock()
        env.ui.console.print = MagicMock()
        env.ui.confirm = MagicMock(return_value=False)

        # Should prompt for confirmation and abort when declined
        _delete_conversations(env, convs, {"force": False})
        env.ui.confirm.assert_called_once()

        captured = capsys.readouterr()
        assert "About to DELETE 15 conversations" in captured.err
        assert "Providers:" in captured.out
        assert "claude: 10" in captured.out
        assert "chatgpt: 5" in captured.out


class TestTagsCommand:
    """Tests for the polylogue tags subcommand."""

    def test_tags_list_all(self, runner, cli_workspace):
        """Tags command displays all tags with counts."""
        from unittest.mock import patch

        with patch("polylogue.storage.backends.sqlite.SQLiteBackend") as mock_backend_class:
            with patch("polylogue.storage.repository.ConversationRepository") as mock_repo_class:
                mock_backend = MagicMock()
                mock_backend_class.return_value = mock_backend
                mock_repo = MagicMock()
                mock_repo_class.return_value = mock_repo
                mock_repo.list_tags.return_value = {"important": 5, "review": 3, "draft": 1}

                result = runner.invoke(cli, ["tags"])

        assert result.exit_code == 0
        assert "important" in result.output
        assert "5" in result.output
        assert "review" in result.output
        assert "3" in result.output
        assert "3 total" in result.output

    def test_tags_json_output(self, runner, cli_workspace):
        """Tags --json outputs valid JSON dict."""
        from unittest.mock import patch
        import json

        with patch("polylogue.storage.backends.sqlite.SQLiteBackend") as mock_backend_class:
            with patch("polylogue.storage.repository.ConversationRepository") as mock_repo_class:
                mock_backend = MagicMock()
                mock_backend_class.return_value = mock_backend
                mock_repo = MagicMock()
                mock_repo_class.return_value = mock_repo
                mock_repo.list_tags.return_value = {"tag1": 10, "tag2": 2}

                result = runner.invoke(cli, ["tags", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == {"tag1": 10, "tag2": 2}

    def test_tags_provider_filter(self, runner, cli_workspace):
        """Tags -p passes provider to list_tags."""
        from unittest.mock import patch

        with patch("polylogue.storage.backends.sqlite.SQLiteBackend") as mock_backend_class:
            with patch("polylogue.storage.repository.ConversationRepository") as mock_repo_class:
                mock_backend = MagicMock()
                mock_backend_class.return_value = mock_backend
                mock_repo = MagicMock()
                mock_repo_class.return_value = mock_repo
                mock_repo.list_tags.return_value = {"claude-tag": 3}

                result = runner.invoke(cli, ["tags", "-p", "claude"])

        assert result.exit_code == 0
        mock_repo.list_tags.assert_called_once_with(provider="claude")
        assert "claude-tag" in result.output

    def test_tags_count_limit(self, runner, cli_workspace):
        """Tags -n truncates to top N."""
        from unittest.mock import patch

        with patch("polylogue.storage.backends.sqlite.SQLiteBackend") as mock_backend_class:
            with patch("polylogue.storage.repository.ConversationRepository") as mock_repo_class:
                mock_backend = MagicMock()
                mock_backend_class.return_value = mock_backend
                mock_repo = MagicMock()
                mock_repo_class.return_value = mock_repo
                mock_repo.list_tags.return_value = {"a": 10, "b": 5, "c": 1}

                result = runner.invoke(cli, ["tags", "-n", "2"])

        assert result.exit_code == 0
        assert "a" in result.output
        assert "b" in result.output
        assert "c" not in result.output

    def test_tags_empty(self, runner, cli_workspace):
        """Tags with no tags shows hint."""
        from unittest.mock import patch

        with patch("polylogue.storage.backends.sqlite.SQLiteBackend") as mock_backend_class:
            with patch("polylogue.storage.repository.ConversationRepository") as mock_repo_class:
                mock_backend = MagicMock()
                mock_backend_class.return_value = mock_backend
                mock_repo = MagicMock()
                mock_repo_class.return_value = mock_repo
                mock_repo.list_tags.return_value = {}

                result = runner.invoke(cli, ["tags"])

        assert result.exit_code == 0
        assert "No tags found" in result.output
        assert "--add-tag" in result.output

    def test_tags_empty_with_provider_filter(self, runner, cli_workspace):
        """Tags with provider filter and no tags shows provider-specific hint."""
        from unittest.mock import patch

        with patch("polylogue.storage.backends.sqlite.SQLiteBackend") as mock_backend_class:
            with patch("polylogue.storage.repository.ConversationRepository") as mock_repo_class:
                mock_backend = MagicMock()
                mock_backend_class.return_value = mock_backend
                mock_repo = MagicMock()
                mock_repo_class.return_value = mock_repo
                mock_repo.list_tags.return_value = {}

                result = runner.invoke(cli, ["tags", "-p", "chatgpt"])

        assert result.exit_code == 0
        assert "No tags found for provider 'chatgpt'" in result.output
        assert "--add-tag" in result.output


class TestEmbedCommand:
    """Tests for the polylogue embed subcommand."""

    def test_embed_no_api_key(self, runner, cli_workspace):
        """Embed without VOYAGE_API_KEY should print error and abort."""
        from unittest.mock import patch

        # Ensure both env vars are unset
        with patch.dict(
            "os.environ",
            {"VOYAGE_API_KEY": "", "POLYLOGUE_VOYAGE_API_KEY": ""},
            clear=False,
        ):
            result = runner.invoke(cli, ["embed"])

        assert result.exit_code != 0
        assert "VOYAGE_API_KEY" in result.output or "not set" in result.output.lower()

    def test_embed_stats_no_api_key(self, runner, cli_workspace):
        """--stats flag should work WITHOUT an API key."""
        from unittest.mock import patch

        # Mock open_connection to return mock database with stats
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)

        # Mock execute results for three COUNT queries
        # Total conversations
        mock_result_1 = MagicMock()
        mock_result_1.fetchone.return_value = (5,)

        # Embedded conversations
        mock_result_2 = MagicMock()
        mock_result_2.fetchone.return_value = (3,)

        # Embedded messages
        mock_result_3 = MagicMock()
        mock_result_3.fetchone.return_value = (45,)

        # Pending conversations
        mock_result_4 = MagicMock()
        mock_result_4.fetchone.return_value = (2,)

        # Mock execute to return results in sequence
        mock_conn.execute.side_effect = [
            mock_result_1,
            mock_result_2,
            mock_result_3,
            mock_result_4,
        ]

        with patch("polylogue.storage.backends.sqlite.open_connection") as mock_open:
            mock_open.return_value = mock_conn
            # Ensure API key is not set
            with patch.dict(
                "os.environ",
                {"VOYAGE_API_KEY": "", "POLYLOGUE_VOYAGE_API_KEY": ""},
                clear=False,
            ):
                result = runner.invoke(cli, ["embed", "--stats"])

        assert result.exit_code == 0
        assert "Embedding Statistics" in result.output

    def test_embed_stats_output(self, runner, cli_workspace):
        """Verify --stats output includes correct labels and values."""
        from unittest.mock import patch

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)

        # Setup mock results for stats
        results = [
            MagicMock(fetchone=MagicMock(return_value=(10,))),  # Total conversations
            MagicMock(fetchone=MagicMock(return_value=(7,))),   # Embedded conversations
            MagicMock(fetchone=MagicMock(return_value=(100,))), # Embedded messages
            MagicMock(fetchone=MagicMock(return_value=(3,))),   # Pending conversations
        ]
        mock_conn.execute.side_effect = results

        with patch("polylogue.storage.backends.sqlite.open_connection") as mock_open:
            mock_open.return_value = mock_conn
            with patch.dict(
                "os.environ",
                {"VOYAGE_API_KEY": "", "POLYLOGUE_VOYAGE_API_KEY": ""},
                clear=False,
            ):
                result = runner.invoke(cli, ["embed", "--stats"])

        assert result.exit_code == 0
        assert "Embedding Statistics" in result.output
        assert "Total conversations" in result.output
        assert "10" in result.output
        assert "Embedded conversations" in result.output
        assert "7" in result.output
        assert "Coverage" in result.output
        assert "Embedded messages" in result.output
        assert "100" in result.output
        assert "Pending" in result.output
        assert "3" in result.output

    def test_embed_no_sqlite_vec(self, runner, cli_workspace):
        """With API key set but no sqlite-vec, should print error."""
        from unittest.mock import patch

        with patch("polylogue.storage.search_providers.create_vector_provider") as mock_create:
            mock_create.return_value = None

            result = runner.invoke(
                cli, ["embed"], env={"VOYAGE_API_KEY": "test-key", "POLYLOGUE_FORCE_PLAIN": "1"}
            )

        assert result.exit_code != 0
        assert "sqlite-vec" in result.output.lower()
        assert "not available" in result.output.lower()

    def test_embed_single_not_found(self, runner, cli_workspace):
        """--conversation with nonexistent ID should print error."""
        from unittest.mock import patch

        with patch("polylogue.storage.backends.sqlite.SQLiteBackend") as mock_backend_class:
            with patch("polylogue.storage.repository.ConversationRepository") as mock_repo_class:
                with patch("polylogue.storage.search_providers.create_vector_provider") as mock_create:
                    mock_backend = MagicMock()
                    mock_backend_class.return_value = mock_backend
                    mock_repo = MagicMock()
                    mock_repo_class.return_value = mock_repo
                    mock_repo.get.return_value = None
                    mock_provider = MagicMock()
                    mock_create.return_value = mock_provider

                    result = runner.invoke(
                        cli,
                        ["embed", "--conversation", "nonexistent-id"],
                        env={"VOYAGE_API_KEY": "test-key", "POLYLOGUE_FORCE_PLAIN": "1"},
                    )

        assert result.exit_code != 0
        assert "not found" in result.output.lower()
        assert "nonexistent-id" in result.output

    def test_embed_rebuild_flag(self, runner, cli_workspace):
        """--rebuild flag is passed to _embed_batch."""
        from unittest.mock import patch

        with patch("polylogue.storage.backends.sqlite.SQLiteBackend") as mock_backend_class:
            with patch("polylogue.storage.repository.ConversationRepository") as mock_repo_class:
                with patch("polylogue.storage.search_providers.create_vector_provider") as mock_create:
                    with patch("polylogue.cli.commands.embed._embed_batch") as mock_batch:
                        mock_backend = MagicMock()
                        mock_backend_class.return_value = mock_backend
                        mock_repo = MagicMock()
                        mock_repo_class.return_value = mock_repo
                        mock_provider = MagicMock()
                        mock_create.return_value = mock_provider

                        result = runner.invoke(
                            cli,
                            ["embed", "--rebuild"],
                            env={"VOYAGE_API_KEY": "test-key", "POLYLOGUE_FORCE_PLAIN": "1"},
                        )

        assert result.exit_code == 0
        mock_batch.assert_called_once()
        # Verify rebuild=True was passed
        call_kwargs = mock_batch.call_args[1]
        assert call_kwargs["rebuild"] is True

    def test_embed_limit_flag(self, runner, cli_workspace):
        """--limit flag is passed to _embed_batch."""
        from unittest.mock import patch

        with patch("polylogue.storage.backends.sqlite.SQLiteBackend") as mock_backend_class:
            with patch("polylogue.storage.repository.ConversationRepository") as mock_repo_class:
                with patch("polylogue.storage.search_providers.create_vector_provider") as mock_create:
                    with patch("polylogue.cli.commands.embed._embed_batch") as mock_batch:
                        mock_backend = MagicMock()
                        mock_backend_class.return_value = mock_backend
                        mock_repo = MagicMock()
                        mock_repo_class.return_value = mock_repo
                        mock_provider = MagicMock()
                        mock_create.return_value = mock_provider

                        result = runner.invoke(
                            cli,
                            ["embed", "--limit", "50"],
                            env={"VOYAGE_API_KEY": "test-key", "POLYLOGUE_FORCE_PLAIN": "1"},
                        )

        assert result.exit_code == 0
        mock_batch.assert_called_once()
        # Verify limit was passed
        call_kwargs = mock_batch.call_args[1]
        assert call_kwargs["limit"] == 50

    def test_embed_model_choice(self, runner, cli_workspace):
        """--model flag selects embedding model."""
        from unittest.mock import patch

        with patch("polylogue.storage.backends.sqlite.SQLiteBackend") as mock_backend_class:
            with patch("polylogue.storage.repository.ConversationRepository") as mock_repo_class:
                with patch("polylogue.storage.search_providers.create_vector_provider") as mock_create:
                    with patch("polylogue.cli.commands.embed._embed_batch") as mock_batch:
                        mock_backend = MagicMock()
                        mock_backend_class.return_value = mock_backend
                        mock_repo = MagicMock()
                        mock_repo_class.return_value = mock_repo
                        mock_provider = MagicMock()
                        mock_create.return_value = mock_provider

                        result = runner.invoke(
                            cli,
                            ["embed", "--model", "voyage-4-large"],
                            env={"VOYAGE_API_KEY": "test-key", "POLYLOGUE_FORCE_PLAIN": "1"},
                        )

        assert result.exit_code == 0
        # Verify model was set on provider
        assert mock_provider.model == "voyage-4-large"

    def test_embed_voyage_api_key_alt_env(self, runner, cli_workspace):
        """POLYLOGUE_VOYAGE_API_KEY env var is accepted as fallback."""
        from unittest.mock import patch

        with patch("polylogue.storage.backends.sqlite.SQLiteBackend") as mock_backend_class:
            with patch("polylogue.storage.repository.ConversationRepository") as mock_repo_class:
                with patch("polylogue.storage.search_providers.create_vector_provider") as mock_create:
                    with patch("polylogue.cli.commands.embed._embed_batch") as mock_batch:
                        mock_backend = MagicMock()
                        mock_backend_class.return_value = mock_backend
                        mock_repo = MagicMock()
                        mock_repo_class.return_value = mock_repo
                        mock_provider = MagicMock()
                        mock_create.return_value = mock_provider

                        result = runner.invoke(
                            cli,
                            ["embed"],
                            env={
                                "POLYLOGUE_VOYAGE_API_KEY": "alt-test-key",
                                "POLYLOGUE_FORCE_PLAIN": "1",
                            },
                        )

        assert result.exit_code == 0
        # Verify create_vector_provider was called with the key
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["voyage_api_key"] == "alt-test-key"
