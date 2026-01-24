"""Tests for the analytics CLI command."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.config import ConfigError


@dataclass
class MockProviderMetrics:
    """Mock provider metrics for testing."""

    provider_name: str
    conversation_count: int
    message_count: int
    user_message_count: int
    assistant_message_count: int
    avg_messages_per_conversation: float
    avg_user_words: float
    avg_assistant_words: float
    tool_use_count: int
    thinking_count: int
    tool_use_percentage: float
    thinking_percentage: float


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def mock_metrics():
    """Sample provider metrics for testing."""
    return [
        MockProviderMetrics(
            provider_name="chatgpt",
            conversation_count=10,
            message_count=100,
            user_message_count=50,
            assistant_message_count=50,
            avg_messages_per_conversation=10.0,
            avg_user_words=25.5,
            avg_assistant_words=100.3,
            tool_use_count=0,
            thinking_count=0,
            tool_use_percentage=0.0,
            thinking_percentage=0.0,
        ),
        MockProviderMetrics(
            provider_name="claude",
            conversation_count=5,
            message_count=60,
            user_message_count=30,
            assistant_message_count=30,
            avg_messages_per_conversation=12.0,
            avg_user_words=30.0,
            avg_assistant_words=150.0,
            tool_use_count=10,
            thinking_count=5,
            tool_use_percentage=16.7,
            thinking_percentage=8.3,
        ),
    ]


class TestAnalyticsBasic:
    """Tests for basic analytics display."""

    def test_analytics_default_output(self, runner, cli_workspace, mock_metrics):
        """Analytics command shows provider comparison by default."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = mock_metrics

                result = runner.invoke(cli, ["analytics"])

        assert result.exit_code == 0
        # Rich tables don't render to text in CliRunner, but total is displayed
        assert "Total" in result.output

    def test_analytics_shows_conversation_counts(self, runner, cli_workspace, mock_metrics):
        """Analytics shows conversation counts in total summary."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = mock_metrics

                result = runner.invoke(cli, ["analytics"])

        assert result.exit_code == 0
        # Total: 15 conversations shown in summary
        assert "15" in result.output

    def test_analytics_shows_message_counts(self, runner, cli_workspace, mock_metrics):
        """Analytics shows message counts in total summary."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = mock_metrics

                result = runner.invoke(cli, ["analytics"])

        assert result.exit_code == 0
        # Total: 160 messages shown in summary
        assert "160" in result.output

    def test_analytics_shows_summary(self, runner, cli_workspace, mock_metrics):
        """Analytics shows total summary."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = mock_metrics

                result = runner.invoke(cli, ["analytics"])

        assert result.exit_code == 0
        assert "Total" in result.output
        # Total: 15 conversations, 160 messages
        assert "15" in result.output
        assert "160" in result.output

    def test_analytics_empty_database(self, runner, cli_workspace):
        """Analytics handles empty database gracefully."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = []  # Empty metrics

                result = runner.invoke(cli, ["analytics"])

        assert result.exit_code == 0


class TestAnalyticsJson:
    """Tests for JSON output mode."""

    def test_analytics_json_output(self, runner, cli_workspace, mock_metrics):
        """--json flag outputs valid JSON."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = mock_metrics

                result = runner.invoke(cli, ["analytics", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_analytics_json_structure(self, runner, cli_workspace, mock_metrics):
        """JSON output has correct structure."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = mock_metrics

                result = runner.invoke(cli, ["analytics", "--json"])

        data = json.loads(result.output)
        # Check required fields
        assert "provider" in data[0]
        assert "conversations" in data[0]
        assert "messages" in data[0]
        assert "user_messages" in data[0]
        assert "assistant_messages" in data[0]

    def test_analytics_json_provider_names(self, runner, cli_workspace, mock_metrics):
        """JSON output includes correct provider names."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = mock_metrics

                result = runner.invoke(cli, ["analytics", "--json"])

        data = json.loads(result.output)
        providers = [d["provider"] for d in data]
        assert "chatgpt" in providers
        assert "claude" in providers

    def test_analytics_json_conversation_counts(self, runner, cli_workspace, mock_metrics):
        """JSON output has correct conversation counts."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = mock_metrics

                result = runner.invoke(cli, ["analytics", "--json"])

        data = json.loads(result.output)
        chatgpt = next(d for d in data if d["provider"] == "chatgpt")
        assert chatgpt["conversations"] == 10

    def test_analytics_json_message_counts(self, runner, cli_workspace, mock_metrics):
        """JSON output has correct message counts."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = mock_metrics

                result = runner.invoke(cli, ["analytics", "--json"])

        data = json.loads(result.output)
        chatgpt = next(d for d in data if d["provider"] == "chatgpt")
        assert chatgpt["messages"] == 100
        assert chatgpt["user_messages"] == 50
        assert chatgpt["assistant_messages"] == 50

    def test_analytics_json_averages(self, runner, cli_workspace, mock_metrics):
        """JSON output has correct average calculations."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = mock_metrics

                result = runner.invoke(cli, ["analytics", "--json"])

        data = json.loads(result.output)
        chatgpt = next(d for d in data if d["provider"] == "chatgpt")
        assert chatgpt["avg_messages_per_conversation"] == 10.0

    def test_analytics_json_numeric_values(self, runner, cli_workspace, mock_metrics):
        """JSON output values are numeric, not strings."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = mock_metrics

                result = runner.invoke(cli, ["analytics", "--json"])

        data = json.loads(result.output)
        assert isinstance(data[0]["conversations"], int)
        assert isinstance(data[0]["messages"], int)
        assert isinstance(data[0]["avg_messages_per_conversation"], float)


class TestAnalyticsStatistics:
    """Tests for tool use and thinking statistics."""

    def test_analytics_tool_use_detection(self, runner, cli_workspace, mock_metrics):
        """Analytics detects tool use in conversations."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = mock_metrics

                result = runner.invoke(cli, ["analytics", "--json"])

        data = json.loads(result.output)
        claude = next(d for d in data if d["provider"] == "claude")
        assert claude["tool_use_count"] == 10

    def test_analytics_thinking_detection(self, runner, cli_workspace, mock_metrics):
        """Analytics detects thinking blocks in conversations."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = mock_metrics

                result = runner.invoke(cli, ["analytics", "--json"])

        data = json.loads(result.output)
        claude = next(d for d in data if d["provider"] == "claude")
        assert claude["thinking_count"] == 5

    def test_analytics_tool_use_percentage(self, runner, cli_workspace, mock_metrics):
        """Analytics calculates tool use percentage."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = mock_metrics

                result = runner.invoke(cli, ["analytics", "--json"])

        data = json.loads(result.output)
        claude = next(d for d in data if d["provider"] == "claude")
        assert claude["tool_use_percentage"] == 16.7

    def test_analytics_thinking_percentage(self, runner, cli_workspace, mock_metrics):
        """Analytics calculates thinking percentage."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = mock_metrics

                result = runner.invoke(cli, ["analytics", "--json"])

        data = json.loads(result.output)
        claude = next(d for d in data if d["provider"] == "claude")
        assert claude["thinking_percentage"] == 8.3

    def test_analytics_word_count_averages(self, runner, cli_workspace, mock_metrics):
        """Analytics calculates word count averages."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = mock_metrics

                result = runner.invoke(cli, ["analytics", "--json"])

        data = json.loads(result.output)
        chatgpt = next(d for d in data if d["provider"] == "chatgpt")
        assert chatgpt["avg_user_words"] == 25.5
        assert chatgpt["avg_assistant_words"] == 100.3


class TestAnalyticsFlags:
    """Tests for command flags."""

    def test_analytics_provider_comparison_flag(self, runner, cli_workspace, mock_metrics):
        """--provider-comparison flag enables provider comparison."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = mock_metrics

                result = runner.invoke(cli, ["analytics", "--provider-comparison"])

        assert result.exit_code == 0
        mock_compute.assert_called_once()

    def test_analytics_provider_comparison_with_json(self, runner, cli_workspace, mock_metrics):
        """--provider-comparison and --json work together."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = mock_metrics

                result = runner.invoke(cli, ["analytics", "--provider-comparison", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2

    def test_analytics_custom_config_path(self, runner, cli_workspace, mock_metrics):
        """--config flag uses custom config path."""
        custom_config = cli_workspace["config_path"]

        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = mock_metrics

                result = runner.invoke(cli, ["analytics", "--config", str(custom_config)])

        assert result.exit_code == 0
        mock_config.assert_called_once_with(custom_config)


class TestAnalyticsOrdering:
    """Tests for result ordering."""

    def test_analytics_json_preserves_order(self, runner, cli_workspace, mock_metrics):
        """JSON output preserves provider order from compute function."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = mock_metrics

                result = runner.invoke(cli, ["analytics", "--json"])

        data = json.loads(result.output)
        # First in mock_metrics is chatgpt
        assert data[0]["provider"] == "chatgpt"
        assert data[1]["provider"] == "claude"


class TestAnalyticsEdgeCases:
    """Tests for edge cases."""

    def test_analytics_config_error(self, runner, cli_workspace):
        """Analytics handles ConfigError gracefully."""
        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            mock_config.side_effect = ConfigError("Invalid config file")

            result = runner.invoke(cli, ["analytics"])

        assert result.exit_code != 0
        assert "Invalid config file" in result.output

    def test_analytics_single_provider(self, runner, cli_workspace):
        """Analytics works with single provider."""
        single_metric = [
            MockProviderMetrics(
                provider_name="chatgpt",
                conversation_count=5,
                message_count=50,
                user_message_count=25,
                assistant_message_count=25,
                avg_messages_per_conversation=10.0,
                avg_user_words=20.0,
                avg_assistant_words=80.0,
                tool_use_count=0,
                thinking_count=0,
                tool_use_percentage=0.0,
                thinking_percentage=0.0,
            )
        ]

        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = single_metric

                result = runner.invoke(cli, ["analytics"])

        assert result.exit_code == 0
        # Single provider total: 5 conversations, 50 messages
        assert "5 conversations" in result.output

    def test_analytics_zero_tool_use_json(self, runner, cli_workspace):
        """Analytics JSON output shows 0 for zero tool use."""
        zero_tool_metric = [
            MockProviderMetrics(
                provider_name="chatgpt",
                conversation_count=5,
                message_count=50,
                user_message_count=25,
                assistant_message_count=25,
                avg_messages_per_conversation=10.0,
                avg_user_words=20.0,
                avg_assistant_words=80.0,
                tool_use_count=0,  # No tool use
                thinking_count=0,
                tool_use_percentage=0.0,
                thinking_percentage=0.0,
            )
        ]

        with patch("polylogue.cli.commands.analytics.load_effective_config") as mock_config:
            with patch("polylogue.cli.commands.analytics.compute_provider_comparison") as mock_compute:
                mock_config.return_value = MagicMock(archive_root=cli_workspace["archive_root"])
                mock_compute.return_value = zero_tool_metric

                result = runner.invoke(cli, ["analytics", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]["tool_use_count"] == 0
        assert data[0]["tool_use_percentage"] == 0.0
