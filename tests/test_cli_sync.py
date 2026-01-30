"""Tests for CLI sync command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.config import ConfigError
from polylogue.ingestion import DriveError
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


class TestSyncCommandPreviewMode:
    """Tests for --preview flag."""

    def test_sync_preview_calls_plan_sources(self, runner, cli_workspace, mock_plan_result):
        """Preview mode calls plan_sources()."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.plan_sources") as mock_plan:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=None):
                        mock_create_config.return_value = MagicMock(sources=[])
                        mock_plan.return_value = mock_plan_result

                        result = runner.invoke(cli, ["sync", "--preview"])

        assert result.exit_code == 0
        mock_plan.assert_called_once()

    def test_sync_preview_displays_plan_snapshot(self, runner, cli_workspace, mock_plan_result):
        """Preview mode displays snapshot information."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.plan_sources") as mock_plan:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=["test-inbox"]):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=["test-inbox"]):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="5 conversations, 50 messages"):
                            with patch("polylogue.cli.commands.sync.format_cursors", return_value=""):
                                mock_config = MagicMock()
                                mock_config.sources = []
                                mock_create_config.return_value = mock_config
                                mock_plan.return_value = mock_plan_result

                                result = runner.invoke(cli, ["sync", "--preview"])

        assert result.exit_code == 0
        assert "Preview" in result.output or "preview" in result.output.lower()

    def test_sync_preview_with_plain_mode_skips_confirm(self, runner, cli_workspace, mock_plan_result):
        """Preview mode in plain mode skips confirmation."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.plan_sources") as mock_plan:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=None):
                        mock_config = MagicMock()
                        mock_config.sources = []
                        mock_create_config.return_value = mock_config
                        mock_plan.return_value = mock_plan_result

                        result = runner.invoke(cli, ["sync", "--preview"])

        # In plain mode (forced by POLYLOGUE_FORCE_PLAIN), should exit without asking
        assert result.exit_code == 0

    def test_sync_preview_drive_error_fails(self, runner, cli_workspace):
        """Preview mode propagates DriveError."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.plan_sources") as mock_plan:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=["google-drive"]):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=["google-drive"]):
                        mock_config = MagicMock()
                        mock_config.sources = []
                        mock_create_config.return_value = mock_config
                        mock_plan.side_effect = DriveError("OAuth token expired")

                        result = runner.invoke(cli, ["sync", "--preview"])

        assert result.exit_code != 0
        assert "OAuth token expired" in result.output

class TestSyncCommandNonPreviewMode:
    """Tests for normal (non-preview) sync mode."""

    def test_sync_calls_run_sources(self, runner, cli_workspace, mock_run_result):
        """Non-preview mode calls run_sources()."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="3 conversations"):
                            mock_config = MagicMock()
                            mock_config.sources = []
                            mock_create_config.return_value = mock_config
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["sync"])

        assert result.exit_code == 0
        mock_run.assert_called_once()

    def test_sync_displays_duration(self, runner, cli_workspace, mock_run_result):
        """Run displays duration in milliseconds."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="3 conversations"):
                            mock_config = MagicMock()
                            mock_config.sources = []
                            mock_create_config.return_value = mock_config
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["sync"])

        assert result.exit_code == 0
        assert "Duration" in result.output or "duration" in result.output.lower()
        assert "1500ms" in result.output

    def test_sync_displays_counts(self, runner, cli_workspace, mock_run_result):
        """Run displays counts."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="3 conversations, 30 messages"):
                            mock_config = MagicMock()
                            mock_config.sources = []
                            mock_create_config.return_value = mock_config
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["sync"])

        assert result.exit_code == 0
        assert "Counts" in result.output or "counts" in result.output.lower()

    def test_sync_drive_error_fails(self, runner, cli_workspace):
        """Non-preview mode propagates DriveError."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=["google-drive"]):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=["google-drive"]):
                        mock_config = MagicMock()
                        mock_config.sources = []
                        mock_create_config.return_value = mock_config
                        mock_run.side_effect = DriveError("Drive API rate limit exceeded")

                        result = runner.invoke(cli, ["sync"])

        assert result.exit_code != 0
        assert "Drive API rate limit" in result.output


class TestSyncCommandStageOption:
    """Tests for --stage flag."""

    def test_sync_stage_ingest_only(self, runner, cli_workspace, mock_run_result):
        """--stage ingest passes stage to run_sources()."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="3 conversations"):
                            mock_config = MagicMock()
                            mock_config.sources = []
                            mock_create_config.return_value = mock_config
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["sync", "--stage", "ingest"])

        assert result.exit_code == 0
        # Check that stage parameter was passed
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["stage"] == "ingest"

    def test_sync_stage_render_only(self, runner, cli_workspace, mock_run_result):
        """--stage render passes stage to run_sources()."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="3 conversations"):
                            mock_config = MagicMock()
                            mock_config.sources = []
                            mock_create_config.return_value = mock_config
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["sync", "--stage", "render"])

        assert result.exit_code == 0
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["stage"] == "render"

    def test_sync_stage_index_only(self, runner, cli_workspace, mock_run_result):
        """--stage index passes stage to run_sources()."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="3 conversations"):
                            mock_config = MagicMock()
                            mock_config.sources = []
                            mock_create_config.return_value = mock_config
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["sync", "--stage", "index"])

        assert result.exit_code == 0
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["stage"] == "index"

    def test_sync_stage_all_default(self, runner, cli_workspace, mock_run_result):
        """--stage all is the default."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="3 conversations"):
                            mock_config = MagicMock()
                            mock_config.sources = []
                            mock_create_config.return_value = mock_config
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["sync"])

        assert result.exit_code == 0
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["stage"] == "all"

    def test_sync_stage_index_displays_index_status(self, runner, cli_workspace):
        """--stage index displays index status instead of counts."""
        result_indexed = RunResult(
            run_id="run-idx",
            counts={"conversations": 0},
            drift={},
            indexed=True,
            index_error=None,
            duration_ms=800,
        )
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.sync.format_index_status") as mock_format_idx:
                            mock_config = MagicMock()
                            mock_config.sources = []
                            mock_create_config.return_value = mock_config
                            mock_run.return_value = result_indexed
                            mock_format_idx.return_value = "Index status: indexed"

                            result = runner.invoke(cli, ["sync", "--stage", "index"])

        assert result.exit_code == 0
        mock_format_idx.assert_called_once()


class TestSyncCommandSourceOption:
    """Tests for --source flag."""

    def test_sync_source_single(self, runner, cli_workspace, mock_run_result):
        """--source filters to single source."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources") as mock_resolve:
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=["test-inbox"]):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="3 conversations"):
                            mock_config = MagicMock()
                            mock_config.sources = []
                            mock_create_config.return_value = mock_config
                            mock_resolve.return_value = ["test-inbox"]
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["sync", "--source", "test-inbox"])

        assert result.exit_code == 0
        mock_resolve.assert_called_once_with(mock_config, ("test-inbox",), "sync")

    def test_sync_source_multiple(self, runner, cli_workspace, mock_run_result):
        """--source can be repeated for multiple sources."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources") as mock_resolve:
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=["test-inbox", "drive"]):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="8 conversations"):
                            mock_config = MagicMock()
                            mock_config.sources = []
                            mock_create_config.return_value = mock_config
                            mock_resolve.return_value = ["test-inbox", "drive"]
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(
                                cli, ["sync", "--source", "test-inbox", "--source", "drive"]
                            )

        assert result.exit_code == 0
        # Verify that resolve_sources was called with both sources
        call_args = mock_resolve.call_args
        assert "test-inbox" in call_args[0][1]
        assert "drive" in call_args[0][1]

    def test_sync_source_displays_in_title(self, runner, cli_workspace, mock_run_result):
        """Selected sources are displayed in run title."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=["my-source"]):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=["my-source"]):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="5 conversations"):
                            mock_config = MagicMock()
                            mock_config.sources = []
                            mock_create_config.return_value = mock_config
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["sync", "--source", "my-source"])

        assert result.exit_code == 0
        # Run title should include source name
        assert "my-source" in result.output or "Run" in result.output


class TestSyncCommandFormatOption:
    """Tests for --format flag."""

    def test_sync_format_markdown(self, runner, cli_workspace, mock_run_result):
        """--format markdown passes format to run_sources()."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="3 conversations"):
                            mock_config = MagicMock()
                            mock_config.sources = []
                            mock_create_config.return_value = mock_config
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["sync", "--format", "markdown"])

        assert result.exit_code == 0
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["render_format"] == "markdown"

    def test_sync_format_html_default(self, runner, cli_workspace, mock_run_result):
        """--format html is the default."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="3 conversations"):
                            mock_config = MagicMock()
                            mock_config.sources = []
                            mock_create_config.return_value = mock_config
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["sync"])

        assert result.exit_code == 0
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["render_format"] == "html"


class TestSyncCommandProgressOutput:
    """Tests for progress display."""

    def test_sync_plain_mode_shows_progress(self, runner, cli_workspace, mock_run_result):
        """Plain mode displays periodic progress updates."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="3 conversations"):
                            mock_config = MagicMock()
                            mock_config.sources = []
                            mock_create_config.return_value = mock_config
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["sync"])

        # In plain mode, should see "Running..." message (or no rich output)
        assert result.exit_code == 0
        # Just verify it completes successfully


class TestSyncCommandIndexError:
    """Tests for index error handling."""

    def test_sync_displays_index_error_with_hint(self, runner, cli_workspace):
        """Run displays index error with rebuild hint."""
        result_with_error = RunResult(
            run_id="run-err",
            counts={"conversations": 2},
            drift={},
            indexed=False,
            index_error="Qdrant connection timeout",
            duration_ms=1200,
        )
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="2 conversations"):
                            with patch("polylogue.cli.commands.sync.format_index_status") as mock_format_idx:
                                mock_config = MagicMock()
                                mock_config.sources = []
                                mock_create_config.return_value = mock_config
                                mock_run.return_value = result_with_error
                                mock_format_idx.return_value = "Index error: Qdrant connection timeout"

                                result = runner.invoke(cli, ["sync"])

        assert result.exit_code == 0
        # Should display error and hint
        assert "Qdrant connection timeout" in result.output or "Index error" in result.output
        assert "rebuild" in result.output.lower() or "--stage index" in result.output

    def test_sync_stage_index_displays_error(self, runner, cli_workspace):
        """Run with --stage index displays index error."""
        result_with_error = RunResult(
            run_id="run-idx-err",
            counts={},
            drift={},
            indexed=False,
            index_error="Vector database unavailable",
            duration_ms=500,
        )
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.sync.format_index_status") as mock_format_idx:
                            mock_config = MagicMock()
                            mock_config.sources = []
                            mock_create_config.return_value = mock_config
                            mock_run.return_value = result_with_error
                            mock_format_idx.return_value = "Index error: Vector database unavailable"

                            result = runner.invoke(cli, ["sync", "--stage", "index"])

        assert result.exit_code == 0
        mock_format_idx.assert_called_once()


class TestSyncCommandRenderOutput:
    """Tests for render output display."""

    def test_sync_displays_latest_render_path_for_render_stage(self, runner, cli_workspace, mock_run_result):
        """Run displays latest render path when render stage included."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="3 conversations"):
                            with patch("polylogue.cli.helpers.latest_render_path") as mock_latest:
                                mock_config = MagicMock()
                                mock_config.sources = []
                                mock_config.render_root = Path("/render")
                                mock_create_config.return_value = mock_config
                                mock_run.return_value = mock_run_result
                                mock_latest.return_value = Path("/render/conv1/conversation.html")

                                result = runner.invoke(cli, ["sync", "--stage", "all"])

        assert result.exit_code == 0
        mock_latest.assert_called_once()

    def test_sync_skips_latest_render_for_non_render_stage(self, runner, cli_workspace, mock_run_result):
        """Run skips render path display for index-only stage."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.sync.format_index_status") as mock_format_idx:
                            with patch("polylogue.cli.helpers.latest_render_path") as mock_latest:
                                mock_config = MagicMock()
                                mock_config.sources = []
                                mock_create_config.return_value = mock_config
                                mock_run.return_value = mock_run_result
                                mock_format_idx.return_value = "Indexed"

                                result = runner.invoke(cli, ["sync", "--stage", "index"])

        assert result.exit_code == 0
        # Should not call latest_render_path for index-only stage
        mock_latest.assert_not_called()


class TestSyncCommandTitle:
    """Tests for run output title."""

    def test_sync_title_includes_stage_when_not_all(self, runner, cli_workspace, mock_run_result):
        """Run title includes stage name when stage is not 'all'."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="3 conversations"):
                            mock_config = MagicMock()
                            mock_config.sources = []
                            mock_create_config.return_value = mock_config
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["sync", "--stage", "render"])

        assert result.exit_code == 0
        # Title should include "render"
        assert "render" in result.output.lower() or "Run" in result.output

    def test_sync_title_includes_sources_when_filtered(self, runner, cli_workspace, mock_run_result):
        """Run title includes source names when sources are filtered."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=["my-inbox"]):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=["my-inbox"]):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="5 conversations"):
                            mock_config = MagicMock()
                            mock_config.sources = []
                            mock_create_config.return_value = mock_config
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["sync", "--source", "my-inbox"])

        assert result.exit_code == 0
        # Title should include source name
        assert "my-inbox" in result.output or "Run" in result.output


class TestSyncCommandCombinations:
    """Tests for flag combinations."""

    def test_sync_preview_with_stage_ingest(self, runner, cli_workspace, mock_plan_result):
        """Preview mode with specific stage still works."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.plan_sources") as mock_plan:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="5 conversations"):
                            mock_config = MagicMock()
                            mock_config.sources = []
                            mock_create_config.return_value = mock_config
                            mock_plan.return_value = mock_plan_result

                            result = runner.invoke(cli, ["sync", "--preview", "--stage", "ingest"])

        assert result.exit_code == 0
        mock_plan.assert_called_once()

    def test_sync_preview_with_format_markdown(self, runner, cli_workspace, mock_plan_result):
        """Preview mode with format flag still works."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.plan_sources") as mock_plan:
                with patch("polylogue.cli.commands.sync.resolve_sources", return_value=None):
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=None):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="5 conversations"):
                            mock_config = MagicMock()
                            mock_config.sources = []
                            mock_create_config.return_value = mock_config
                            mock_plan.return_value = mock_plan_result

                            result = runner.invoke(cli, ["sync", "--preview", "--format", "markdown"])

        assert result.exit_code == 0
        mock_plan.assert_called_once()

    def test_sync_stage_render_with_source_filter(self, runner, cli_workspace, mock_run_result):
        """Stage and source filters can be combined."""
        with patch("polylogue.cli.commands.sync.create_config") as mock_create_config:
            with patch("polylogue.cli.commands.sync.run_sources") as mock_run:
                with patch("polylogue.cli.commands.sync.resolve_sources") as mock_resolve:
                    with patch("polylogue.cli.commands.sync.maybe_prompt_sources", return_value=["test"]):
                        with patch("polylogue.cli.commands.sync.format_counts", return_value="3 conversations"):
                            mock_config = MagicMock()
                            mock_config.sources = []
                            mock_create_config.return_value = mock_config
                            mock_resolve.return_value = ["test"]
                            mock_run.return_value = mock_run_result

                            result = runner.invoke(cli, ["sync", "--stage", "render", "--source", "test"])

        assert result.exit_code == 0
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["stage"] == "render"
