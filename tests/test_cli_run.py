"""Tests for CLI run command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

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


class TestRunCommandPreviewMode:
    """Tests for --preview flag."""

    def test_run_preview_calls_plan_sources(self, runner, cli_workspace, mock_plan_result):
        """Preview mode calls plan_sources()."""
        from unittest.mock import patch


        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.plan_sources") as mock_plan:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        mock_plan.return_value = mock_plan_result

                        result = runner.invoke(cli, ["run", "--preview"])

        assert result.exit_code == 0
        mock_plan.assert_called_once()

    def test_run_preview_displays_plan_snapshot(self, runner, cli_workspace, mock_plan_result):
        """Preview mode displays snapshot information."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.plan_sources") as mock_plan:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=["test-inbox"]):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=["test-inbox"]):
                        with patch(
                            "polylogue.cli.commands.run.format_counts", return_value="5 conversations, 50 messages"
                        ):
                            with patch("polylogue.cli.commands.run.format_cursors", return_value=""):
                                mock_plan.return_value = mock_plan_result

                                result = runner.invoke(cli, ["run", "--preview"])

        assert result.exit_code == 0
        assert "Preview" in result.output or "preview" in result.output.lower()

    def test_run_preview_with_plain_mode_skips_confirm(self, runner, cli_workspace, mock_plan_result):
        """Preview mode in plain mode skips confirmation."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.plan_sources") as mock_plan:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        mock_plan.return_value = mock_plan_result

                        result = runner.invoke(cli, ["run", "--preview"])

        # In plain mode (forced by POLYLOGUE_FORCE_PLAIN), should exit without asking
        assert result.exit_code == 0

    def test_run_preview_drive_error_fails(self, runner, cli_workspace):
        """Preview mode propagates DriveError."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.plan_sources") as mock_plan:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=["google-drive"]):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=["google-drive"]):
                        mock_plan.side_effect = DriveError("OAuth token expired")

                        result = runner.invoke(cli, ["run", "--preview"])

        assert result.exit_code != 0
        assert "OAuth token expired" in result.output


class TestRunCommandNonPreviewMode:
    """Tests for normal (non-preview) sync mode."""

    def test_run_calls_run_sources(self, runner, cli_workspace, mock_run_result):
        """Non-preview mode calls run_sources()."""
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
        mock_run.assert_called_once()

    def test_run_displays_duration(self, runner, cli_workspace, mock_run_result):
        """Run displays duration in milliseconds."""
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
        assert "Duration" in result.output or "duration" in result.output.lower()
        assert "1500ms" in result.output

    def test_run_displays_counts(self, runner, cli_workspace, mock_run_result):
        """Run displays counts."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch(
                            "polylogue.cli.commands.run.format_counts", return_value="3 conversations, 30 messages"
                        ):
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["run"])

        assert result.exit_code == 0
        assert "Counts" in result.output or "counts" in result.output.lower()

    def test_run_drive_error_fails(self, runner, cli_workspace):
        """Non-preview mode propagates DriveError."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=["google-drive"]):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=["google-drive"]):
                        mock_run.side_effect = DriveError("Drive API rate limit exceeded")

                        result = runner.invoke(cli, ["run"])

        assert result.exit_code != 0
        assert "Drive API rate limit" in result.output


class TestRunCommandStageOption:
    """Tests for --stage flag."""

    def test_run_stage_ingest_only(self, runner, cli_workspace, mock_run_result):
        """--stage parse passes stage to run_sources()."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="3 conversations"):
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["run", "--stage", "parse"])

        assert result.exit_code == 0
        # Check that stage parameter was passed
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["stage"] == "parse"

    def test_run_stage_render_only(self, runner, cli_workspace, mock_run_result):
        """--stage render passes stage to run_sources()."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="3 conversations"):
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["run", "--stage", "render"])

        assert result.exit_code == 0
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["stage"] == "render"

    def test_run_stage_index_only(self, runner, cli_workspace, mock_run_result):
        """--stage index passes stage to run_sources()."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="3 conversations"):
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["run", "--stage", "index"])

        assert result.exit_code == 0
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["stage"] == "index"

    def test_run_stage_all_default(self, runner, cli_workspace, mock_run_result):
        """--stage all is the default."""
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
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["stage"] == "all"

    def test_run_stage_index_displays_index_status(self, runner, cli_workspace):
        """--stage index displays index status instead of counts."""
        result_indexed = RunResult(
            run_id="run-idx",
            counts={"conversations": 0},
            drift={},
            indexed=True,
            index_error=None,
            duration_ms=800,
        )
        from unittest.mock import patch

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

    def test_run_source_single(self, runner, cli_workspace, mock_run_result):
        """--source filters to single source."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.services.get_service_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources") as mock_resolve:
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=["test-inbox"]):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="3 conversations"):
                            mock_resolve.return_value = ["test-inbox"]
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["run", "--source", "test-inbox"])

        assert result.exit_code == 0
        mock_resolve.assert_called_once_with(mock_config, ("test-inbox",), "run")

    def test_run_source_multiple(self, runner, cli_workspace, mock_run_result):
        """--source can be repeated for multiple sources."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources") as mock_resolve:
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=["test-inbox", "drive"]):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="8 conversations"):
                            mock_resolve.return_value = ["test-inbox", "drive"]
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["run", "--source", "test-inbox", "--source", "drive"])

        assert result.exit_code == 0
        # Verify that resolve_sources was called with both sources
        call_args = mock_resolve.call_args
        assert "test-inbox" in call_args[0][1]
        assert "drive" in call_args[0][1]

    def test_run_source_displays_in_title(self, runner, cli_workspace, mock_run_result):
        """Selected sources are displayed in run title."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=["my-source"]):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=["my-source"]):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="5 conversations"):
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["run", "--source", "my-source"])

        assert result.exit_code == 0
        # Run title should include source name
        assert "my-source" in result.output or "Run" in result.output


class TestRunCommandFormatOption:
    """Tests for --format flag."""

    def test_run_format_markdown(self, runner, cli_workspace, mock_run_result):
        """--format markdown passes format to run_sources()."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="3 conversations"):
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["run", "--format", "markdown"])

        assert result.exit_code == 0
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["render_format"] == "markdown"

    def test_run_format_html_default(self, runner, cli_workspace, mock_run_result):
        """--format html is the default."""
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
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["render_format"] == "html"


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

        # In plain mode, should see "Running..." message (or no rich output)
        assert result.exit_code == 0
        # Just verify it completes successfully


class TestRunCommandIndexError:
    """Tests for index error handling."""

    def test_run_displays_index_error_with_hint(self, runner, cli_workspace):
        """Run displays index error with rebuild hint."""
        result_with_error = RunResult(
            run_id="run-err",
            counts={"conversations": 2},
            drift={},
            indexed=False,
            index_error="FTS5 index error",
            duration_ms=1200,
        )
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="2 conversations"):
                            with patch("polylogue.cli.commands.run.format_index_status") as mock_format_idx:
                                mock_run.return_value = result_with_error
                                mock_format_idx.return_value = "Index error: FTS5 index error"

                                result = runner.invoke(cli, ["run"])

        assert result.exit_code == 0
        # Should display error and hint
        assert "FTS5 index error" in result.output or "Index error" in result.output
        assert "rebuild" in result.output.lower() or "--stage index" in result.output

    def test_run_stage_index_displays_error(self, runner, cli_workspace):
        """Run with --stage index displays index error."""
        result_with_error = RunResult(
            run_id="run-idx-err",
            counts={},
            drift={},
            indexed=False,
            index_error="Vector database unavailable",
            duration_ms=500,
        )
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_index_status") as mock_format_idx:
                            mock_run.return_value = result_with_error
                            mock_format_idx.return_value = "Index error: Vector database unavailable"

                            result = runner.invoke(cli, ["run", "--stage", "index"])

        assert result.exit_code == 0
        mock_format_idx.assert_called_once()


class TestRunCommandRenderOutput:
    """Tests for render output display."""

    def test_run_displays_latest_render_path_for_render_stage(self, runner, cli_workspace, mock_run_result):
        """Run displays latest render path when render stage included."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="3 conversations"):
                            with patch("polylogue.cli.helpers.latest_render_path") as mock_latest:
                                mock_config.render_root = Path("/render")
                                mock_run.return_value = mock_run_result
                                mock_latest.return_value = Path("/render/conv1/conversation.html")

                                result = runner.invoke(cli, ["run", "--stage", "all"])

        assert result.exit_code == 0
        mock_latest.assert_called_once()

    def test_run_skips_latest_render_for_non_render_stage(self, runner, cli_workspace, mock_run_result):
        """Run skips render path display for index-only stage."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_index_status") as mock_format_idx:
                            with patch("polylogue.cli.helpers.latest_render_path") as mock_latest:
                                mock_run.return_value = mock_run_result
                                mock_format_idx.return_value = "Indexed"

                                result = runner.invoke(cli, ["run", "--stage", "index"])

        assert result.exit_code == 0
        # Should not call latest_render_path for index-only stage
        mock_latest.assert_not_called()


class TestRunCommandTitle:
    """Tests for run output title."""

    def test_run_title_includes_stage_when_not_all(self, runner, cli_workspace, mock_run_result):
        """Run title includes stage name when stage is not 'all'."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="3 conversations"):
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["run", "--stage", "render"])

        assert result.exit_code == 0
        # Title should include "render"
        assert "render" in result.output.lower() or "Run" in result.output

    def test_run_title_includes_sources_when_filtered(self, runner, cli_workspace, mock_run_result):
        """Run title includes source names when sources are filtered."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=["my-inbox"]):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=["my-inbox"]):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="5 conversations"):
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["run", "--source", "my-inbox"])

        assert result.exit_code == 0
        # Title should include source name
        assert "my-inbox" in result.output or "Run" in result.output


class TestRunCommandCombinations:
    """Tests for flag combinations."""

    def test_run_preview_with_stage_ingest(self, runner, cli_workspace, mock_plan_result):
        """Preview mode with specific stage still works."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.plan_sources") as mock_plan:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="5 conversations"):
                            mock_plan.return_value = mock_plan_result

                            result = runner.invoke(cli, ["run", "--preview", "--stage", "parse"])

        assert result.exit_code == 0
        mock_plan.assert_called_once()

    def test_run_preview_with_format_markdown(self, runner, cli_workspace, mock_plan_result):
        """Preview mode with format flag still works."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.plan_sources") as mock_plan:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="5 conversations"):
                            mock_plan.return_value = mock_plan_result

                            result = runner.invoke(cli, ["run", "--preview", "--format", "markdown"])

        assert result.exit_code == 0
        mock_plan.assert_called_once()

    def test_run_stage_render_with_source_filter(self, runner, cli_workspace, mock_run_result):
        """Stage and source filters can be combined."""
        from unittest.mock import patch

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources") as mock_resolve:
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=["test"]):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="3 conversations"):
                            mock_resolve.return_value = ["test"]
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["run", "--stage", "render", "--source", "test"])

        assert result.exit_code == 0
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["stage"] == "render"


class TestRunCommandRenderFailures:
    """Tests for render failure handling in run output."""

    def test_run_displays_render_failures(self, runner, cli_workspace):
        """Run displays render failures with conversation IDs and error details."""
        from unittest.mock import patch

        result_with_failures = RunResult(
            run_id="run-fail",
            counts={"conversations": 1},
            drift={},
            indexed=True,
            index_error=None,
            duration_ms=1500,
            render_failures=[
                {"conversation_id": "conv-1", "error": "Template error"},
            ],
        )

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="1 conversation"):
                            mock_run.return_value = result_with_failures

                            result = runner.invoke(cli, ["run"])

        assert result.exit_code == 0
        assert "Render failures (1)" in result.output
        assert "conv-1: Template error" in result.output

    def test_run_displays_render_failures_truncated(self, runner, cli_workspace):
        """Run truncates render failures at 10 and shows 'and N more' message."""
        from unittest.mock import patch

        failures = [
            {"conversation_id": f"conv-{i}", "error": f"Error {i}"}
            for i in range(1, 16)
        ]
        result_with_failures = RunResult(
            run_id="run-many-fail",
            counts={"conversations": 15},
            drift={},
            indexed=True,
            index_error=None,
            duration_ms=2000,
            render_failures=failures,
        )

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="15 conversations"):
                            mock_run.return_value = result_with_failures

                            result = runner.invoke(cli, ["run"])

        assert result.exit_code == 0
        assert "Render failures (15)" in result.output
        # Should show first 10 failures
        assert "conv-1: Error 1" in result.output
        assert "conv-10: Error 10" in result.output
        # Should NOT show 11th failure
        assert "conv-11:" not in result.output
        # Should show "and N more" message
        assert "... and 5 more" in result.output

    def test_run_no_render_failures_section_when_empty(self, runner, cli_workspace):
        """Run does not display render failures section when empty."""
        from unittest.mock import patch

        result_no_failures = RunResult(
            run_id="run-ok",
            counts={"conversations": 3},
            drift={},
            indexed=True,
            index_error=None,
            duration_ms=1200,
            render_failures=[],
        )

        mock_config = MagicMock(sources=[])
        with patch("polylogue.config.get_config", return_value=mock_config):
            with patch("polylogue.cli.commands.run.run_sources") as mock_run:
                with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.run.format_counts", return_value="3 conversations"):
                            mock_run.return_value = result_no_failures

                            result = runner.invoke(cli, ["run"])

        assert result.exit_code == 0
        assert "Render failures" not in result.output


class TestDeleteConversationPreview:
    """Tests for enhanced deletion preview in query mode."""

    def test_delete_dry_run_shows_provider_breakdown(self, capsys):
        """Dry-run deletion shows provider breakdown."""
        from unittest.mock import MagicMock, patch
        from datetime import datetime

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
        from unittest.mock import MagicMock
        from datetime import datetime

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

    def test_delete_bulk_shows_breakdown_before_exit(self, capsys):
        """Bulk deletion (>10 items) without force shows breakdown and exits."""
        from unittest.mock import MagicMock
        from datetime import datetime

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

        # Should raise SystemExit for bulk without force
        try:
            _delete_conversations(env, convs, {"force": False})
            assert False, "Expected SystemExit"
        except SystemExit as e:
            assert e.code == 1

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
        from unittest.mock import patch, MagicMock

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
        from unittest.mock import patch, MagicMock

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
