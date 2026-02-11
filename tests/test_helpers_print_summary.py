"""Comprehensive tests for print_summary function in polylogue/cli/helpers.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polylogue.cli.helpers import print_summary
from polylogue.cli.types import AppEnv
from polylogue.config import Config


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_ui():
    """Create a mocked UI object."""
    ui = MagicMock()
    ui.plain = False
    ui.console = MagicMock()
    return ui


@pytest.fixture
def mock_env(mock_ui):
    """Create a mocked AppEnv."""
    env = AppEnv(ui=mock_ui)
    return env


@pytest.fixture
def mock_config():
    """Create a real Config object for testing."""
    return Config(
        archive_root=Path("/data/archive"),
        render_root=Path("/data/archive/rendered"),
        sources=[],
    )


@pytest.fixture
def mock_run_data():
    """Create a mocked run data object."""
    run_data = MagicMock()
    run_data.run_id = "run-123"
    run_data.timestamp = "2025-01-15T12:30:45Z"
    return run_data


def _patch_all_dependencies(mock_config, **kwargs):
    """Create a context manager that patches all required dependencies."""
    patches = [
        patch("polylogue.config.get_config", return_value=mock_config),
        patch("polylogue.cli.helpers.latest_run", return_value=kwargs.get("latest_run", None)),
        patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"),
        patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"),
        patch("polylogue.cli.analytics.compute_provider_comparison", return_value=kwargs.get("analytics", None)),
    ]

    # Handle get_health patches
    if "get_health" in kwargs:
        patches.insert(3, patch("polylogue.cli.helpers.get_health", return_value=kwargs["get_health"]))

    # Chain the patches together
    import contextlib
    @contextlib.contextmanager
    def chained():
        with patches[0]:
            with patches[1]:
                with patches[2]:
                    if len(patches) > 4:
                        for p in patches[3:-1]:
                            with p:
                                with patches[-1]:
                                    yield
                    else:
                        with patches[3]:
                            yield

    return chained()


# ============================================================================
# TEST DATA: Health check states for parametrization
# ============================================================================

HEALTH_CHECK_PARAMS = [
    # (status, is_plain, expected_indicator_or_text)
    ("ok", False, "[green]✓[/green]"),
    ("warning", False, "[yellow]![/yellow]"),
    ("error", False, "[red]✗[/red]"),
    ("ok", True, "OK"),
    ("warning", True, "WARN"),
    ("error", True, "ERR"),
]

PROVIDER_COLOR_PARAMS = [
    # (provider_name, expected_color)
    ("claude", "#d97757"),
    ("chatgpt", "#10a37f"),
    ("gemini", "#4285f4"),
    ("codex", "cyan"),
    ("unknown-ai", "white"),
]

TOOL_USE_PARAMS = [
    # (tool_use_count, tool_use_percentage, should_be_present)
    (25, 25.0, True),
    (0, 0.0, False),
]

THINKING_PARAMS = [
    # (thinking_count, thinking_percentage, should_be_present)
    (15, 15.0, True),
    (0, 0.0, False),
]


# ============================================================================
# TestPrintSummaryBasic: Non-verbose mode, basic lines
# ============================================================================


class TestPrintSummaryBasic:
    """Test basic non-verbose print_summary behavior."""

    def test_print_summary_no_last_run(self, mock_env, mock_config):
        """Test summary when no last run data exists."""
        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox (2)"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=None):

            print_summary(mock_env, verbose=False)

            # Verify summary was called with expected lines
            mock_env.ui.summary.assert_called_once()
            call_args = mock_env.ui.summary.call_args
            lines = call_args[0][1]
            assert any("Last run: none" in str(line) for line in lines)

    def test_print_summary_with_last_run(self, mock_env, mock_config, mock_run_data):
        """Test summary with last run data."""
        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=mock_run_data), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=None):

            print_summary(mock_env, verbose=False)

            mock_env.ui.summary.assert_called_once()
            call_args = mock_env.ui.summary.call_args
            lines = call_args[0][1]
            assert any("run-123" in str(line) and "2025-01-15T12:30:45Z" in str(line) for line in lines)

    def test_print_summary_calls_cached_health_in_normal_mode(self, mock_env, mock_config):
        """Test that non-verbose mode calls cached_health_summary, not get_health."""
        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK") as mock_cached, \
             patch("polylogue.cli.helpers.get_health") as mock_get_health, \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=None):

            print_summary(mock_env, verbose=False)

            mock_cached.assert_called_once()
            mock_get_health.assert_not_called()

    def test_print_summary_includes_archive_path(self, mock_env, mock_config):
        """Test that summary includes archive root path."""
        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=None):

            print_summary(mock_env, verbose=False)

            call_args = mock_env.ui.summary.call_args
            lines = call_args[0][1]
            assert any("/data/archive" in str(line) for line in lines)

    def test_print_summary_includes_render_path(self, mock_env, mock_config):
        """Test that summary includes render root path."""
        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=None):

            print_summary(mock_env, verbose=False)

            call_args = mock_env.ui.summary.call_args
            lines = call_args[0][1]
            assert any("/data/archive/rendered" in str(line) for line in lines)

    def test_print_summary_includes_sources(self, mock_env, mock_config):
        """Test that summary includes formatted sources."""
        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox, backup (2 total)") as mock_format, \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=None):

            print_summary(mock_env, verbose=False)

            mock_format.assert_called_once()
            call_args = mock_env.ui.summary.call_args
            lines = call_args[0][1]
            assert any("inbox, backup (2 total)" in str(line) for line in lines)

    def test_print_summary_title_is_polylogue(self, mock_env, mock_config):
        """Test that summary title is 'Polylogue'."""
        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=None):

            print_summary(mock_env, verbose=False)

            call_args = mock_env.ui.summary.call_args
            title = call_args[0][0]
            assert title == "Polylogue"


# ============================================================================
# TestPrintSummaryVerbose: Verbose mode with detailed health checks
# ============================================================================


class TestPrintSummaryVerbose:
    """Test verbose print_summary with detailed health checks."""

    def test_print_summary_verbose_calls_get_health(self, mock_env, mock_config):
        """Test that verbose mode calls get_health, not cached_health_summary."""
        mock_health = MagicMock()
        mock_health.cached = True
        mock_health.age_seconds = 30
        mock_health.checks = []

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary") as mock_cached, \
             patch("polylogue.cli.helpers.get_health", return_value=mock_health) as mock_get_health, \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=None):

            print_summary(mock_env, verbose=True)

            mock_get_health.assert_called_once()
            mock_cached.assert_not_called()

    def test_print_summary_verbose_health_header_with_metadata(self, mock_env, mock_config):
        """Test health header includes cached and age metadata."""
        mock_health = MagicMock()
        mock_health.cached = True
        mock_health.age_seconds = 30
        mock_health.checks = []

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.get_health", return_value=mock_health), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=None):

            print_summary(mock_env, verbose=True)

            call_args = mock_env.ui.summary.call_args
            lines = call_args[0][1]
            assert any("Health" in str(line) and "cached=True" in str(line) and "age=30s" in str(line) for line in lines)

    def test_print_summary_verbose_health_header_without_metadata(self, mock_env, mock_config):
        """Test health header when cached is None."""
        mock_health = MagicMock()
        mock_health.cached = None
        mock_health.age_seconds = None
        mock_health.checks = []

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.get_health", return_value=mock_health), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=None):

            print_summary(mock_env, verbose=True)

            call_args = mock_env.ui.summary.call_args
            lines = call_args[0][1]
            assert any(str(line) == "Health" for line in lines)

    @pytest.mark.parametrize("status,is_plain,expected_indicator", HEALTH_CHECK_PARAMS)
    def test_print_summary_verbose_health_status_by_mode(
        self, mock_env, mock_config, status, is_plain, expected_indicator
    ):
        """Test health check with various statuses in rich and plain modes (parametrized)."""
        check1 = MagicMock()
        check1.name = "database"
        check1.status = status
        check1.detail = "Test detail"

        mock_health = MagicMock()
        mock_health.cached = True
        mock_health.age_seconds = 30
        mock_health.checks = [check1]

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.get_health", return_value=mock_health), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=None):

            mock_env.ui.plain = is_plain
            print_summary(mock_env, verbose=True)

            call_args = mock_env.ui.summary.call_args
            lines = call_args[0][1]
            assert any(expected_indicator in str(line) and "database" in str(line) for line in lines)

    def test_print_summary_verbose_multiple_health_checks(self, mock_env, mock_config):
        """Test summary with multiple health checks."""
        check1 = MagicMock()
        check1.name = "database"
        check1.status = "ok"
        check1.detail = "17.5 GB"

        check2 = MagicMock()
        check2.name = "indexes"
        check2.status = "warning"
        check2.detail = "Needs rebuild"

        check3 = MagicMock()
        check3.name = "storage"
        check3.status = "error"
        check3.detail = "Disk full"

        mock_health = MagicMock()
        mock_health.cached = True
        mock_health.age_seconds = 30
        mock_health.checks = [check1, check2, check3]

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.get_health", return_value=mock_health), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=None):

            mock_env.ui.plain = False
            print_summary(mock_env, verbose=True)

            call_args = mock_env.ui.summary.call_args
            lines = call_args[0][1]
            assert any("database" in str(line) for line in lines)
            assert any("indexes" in str(line) for line in lines)
            assert any("storage" in str(line) for line in lines)

    def test_print_summary_verbose_no_checks(self, mock_env, mock_config):
        """Test verbose mode when no health checks are available."""
        mock_health = MagicMock()
        mock_health.cached = True
        mock_health.age_seconds = 30
        mock_health.checks = []

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.get_health", return_value=mock_health), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=None):

            mock_env.ui.plain = False
            print_summary(mock_env, verbose=True)

            # Should not raise, just no check lines
            mock_env.ui.summary.assert_called_once()


# ============================================================================
# TestPrintSummaryAnalyticsBasic: Analytics visualization
# ============================================================================


class TestPrintSummaryAnalyticsBasic:
    """Test analytics visualization in print_summary."""

    def _create_metric(
        self,
        provider_name: str = "claude",
        conversation_count: int = 100,
        message_count: int = 5000,
        user_message_count: int = 2000,
        assistant_message_count: int = 3000,
        avg_messages_per_conversation: float = 50.0,
        avg_user_words: float = 20.0,
        avg_assistant_words: float = 100.0,
        tool_use_count: int = 10,
        tool_use_percentage: float = 10.0,
        thinking_count: int = 5,
        thinking_percentage: float = 5.0,
    ) -> MagicMock:
        """Create a mock ProviderMetrics object."""
        metric = MagicMock()
        metric.provider_name = provider_name
        metric.conversation_count = conversation_count
        metric.message_count = message_count
        metric.user_message_count = user_message_count
        metric.assistant_message_count = assistant_message_count
        metric.avg_messages_per_conversation = avg_messages_per_conversation
        metric.avg_user_words = avg_user_words
        metric.avg_assistant_words = avg_assistant_words
        metric.tool_use_count = tool_use_count
        metric.tool_use_percentage = tool_use_percentage
        metric.thinking_count = thinking_count
        metric.thinking_percentage = thinking_percentage
        return metric

    def test_print_summary_no_analytics(self, mock_env, mock_config):
        """Test summary when analytics returns None."""
        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=None):

            print_summary(mock_env, verbose=False)

            # Should not crash, just no analytics output
            mock_env.ui.summary.assert_called_once()
            console_calls = mock_env.ui.console.print.call_args_list
            archive_calls = [c for c in console_calls if "Archive" in str(c)]
            assert len(archive_calls) == 0

    def test_print_summary_empty_analytics(self, mock_env, mock_config):
        """Test summary when analytics returns empty list."""
        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=[]):

            print_summary(mock_env, verbose=False)

            mock_env.ui.summary.assert_called_once()
            console_calls = mock_env.ui.console.print.call_args_list
            archive_calls = [c for c in console_calls if "Archive" in str(c)]
            assert len(archive_calls) == 0

    def test_print_summary_single_provider_analytics(self, mock_env, mock_config):
        """Test analytics visualization with single provider."""
        metric = self._create_metric(provider_name="claude", conversation_count=100)

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=[metric]):

            print_summary(mock_env, verbose=False)

            console_calls = mock_env.ui.console.print.call_args_list
            assert any("Archive:" in str(c) and "100" in str(c) for c in console_calls)

    def test_print_summary_multiple_providers_analytics(self, mock_env, mock_config):
        """Test analytics with multiple providers."""
        metric1 = self._create_metric(provider_name="claude", conversation_count=100)
        metric2 = self._create_metric(provider_name="chatgpt", conversation_count=50)

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=[metric1, metric2]):

            print_summary(mock_env, verbose=False)

            console_calls = mock_env.ui.console.print.call_args_list
            assert any("Archive:" in str(c) and "150" in str(c) for c in console_calls)

    @pytest.mark.parametrize("provider_name,expected_color", PROVIDER_COLOR_PARAMS)
    def test_print_summary_analytics_provider_color(
        self, mock_env, mock_config, provider_name, expected_color
    ):
        """Test that providers use correct colors in analytics (parametrized)."""
        metric = self._create_metric(provider_name=provider_name, conversation_count=100)

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=[metric]):

            print_summary(mock_env, verbose=False)

            console_calls = mock_env.ui.console.print.call_args_list
            assert any(expected_color in str(c) for c in console_calls)

    def test_print_summary_analytics_bar_chart_rendering(self, mock_env, mock_config):
        """Test that bar chart is rendered with correct proportions."""
        metric1 = self._create_metric(provider_name="claude", conversation_count=100)
        metric2 = self._create_metric(provider_name="chatgpt", conversation_count=50)

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=[metric1, metric2]):

            print_summary(mock_env, verbose=False)

            console_calls = mock_env.ui.console.print.call_args_list
            all_output = str(console_calls)
            assert "█" in all_output

    def test_print_summary_analytics_percentage_calculation(self, mock_env, mock_config):
        """Test that percentages are calculated correctly."""
        metric = self._create_metric(provider_name="claude", conversation_count=33)

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=[metric]):

            print_summary(mock_env, verbose=False)

            console_calls = mock_env.ui.console.print.call_args_list
            assert any("(100%)" in str(c) for c in console_calls)

    def test_print_summary_analytics_zero_conversations(self, mock_env, mock_config):
        """Test analytics with zero total conversations."""
        metric = self._create_metric(provider_name="claude", conversation_count=0)

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=[metric]):

            print_summary(mock_env, verbose=False)

            console_calls = mock_env.ui.console.print.call_args_list
            assert any("0" in str(c) for c in console_calls)


# ============================================================================
# TestPrintSummaryAnalyticsVerbose: Verbose analytics with deep dive
# ============================================================================


class TestPrintSummaryAnalyticsVerbose:
    """Test verbose analytics output with deep dive statistics."""

    def _create_metric(
        self,
        provider_name: str = "claude",
        conversation_count: int = 100,
        message_count: int = 5000,
        user_message_count: int = 2000,
        assistant_message_count: int = 3000,
        avg_messages_per_conversation: float = 50.0,
        avg_user_words: float = 20.0,
        avg_assistant_words: float = 100.0,
        tool_use_count: int = 10,
        tool_use_percentage: float = 10.0,
        thinking_count: int = 5,
        thinking_percentage: float = 5.0,
    ) -> MagicMock:
        """Create a mock ProviderMetrics object."""
        metric = MagicMock()
        metric.provider_name = provider_name
        metric.conversation_count = conversation_count
        metric.message_count = message_count
        metric.user_message_count = user_message_count
        metric.assistant_message_count = assistant_message_count
        metric.avg_messages_per_conversation = avg_messages_per_conversation
        metric.avg_user_words = avg_user_words
        metric.avg_assistant_words = avg_assistant_words
        metric.tool_use_count = tool_use_count
        metric.tool_use_percentage = tool_use_percentage
        metric.thinking_count = thinking_count
        metric.thinking_percentage = thinking_percentage
        return metric

    def test_print_summary_verbose_analytics_deep_dive_header(self, mock_env, mock_config):
        """Test that verbose mode shows 'Deep Dive:' header."""
        metric = self._create_metric(provider_name="claude")

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=[metric]):

            print_summary(mock_env, verbose=True)

            console_calls = mock_env.ui.console.print.call_args_list
            assert any("Deep Dive:" in str(c) for c in console_calls)

    def test_print_summary_verbose_analytics_messages_count(self, mock_env, mock_config):
        """Test that deep dive shows message count and average."""
        metric = self._create_metric(
            provider_name="claude",
            message_count=5000,
            avg_messages_per_conversation=50.0,
        )

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=[metric]):

            print_summary(mock_env, verbose=True)

            console_calls = mock_env.ui.console.print.call_args_list
            assert any("Messages:" in str(c) and ("5000" in str(c) or "5,000" in str(c)) and "50.0" in str(c) for c in console_calls)

    def test_print_summary_verbose_analytics_words_average(self, mock_env, mock_config):
        """Test that deep dive shows user and assistant word averages."""
        metric = self._create_metric(
            provider_name="claude",
            avg_user_words=20.0,
            avg_assistant_words=100.0,
        )

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=[metric]):

            print_summary(mock_env, verbose=True)

            console_calls = mock_env.ui.console.print.call_args_list
            assert any("Words:" in str(c) and "20" in str(c) and "100" in str(c) for c in console_calls)

    @pytest.mark.parametrize("tool_use_count,tool_use_percentage,should_be_present", TOOL_USE_PARAMS)
    def test_print_summary_verbose_analytics_tool_use(
        self, mock_env, mock_config, tool_use_count, tool_use_percentage, should_be_present
    ):
        """Test that tool use line appears/disappears based on count (parametrized)."""
        metric = self._create_metric(
            provider_name="claude",
            tool_use_count=tool_use_count,
            tool_use_percentage=tool_use_percentage,
        )

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=[metric]):

            print_summary(mock_env, verbose=True)

            console_calls = mock_env.ui.console.print.call_args_list
            has_tool_use = any("Tool Use:" in str(c) for c in console_calls)
            assert has_tool_use == should_be_present

    @pytest.mark.parametrize("thinking_count,thinking_percentage,should_be_present", THINKING_PARAMS)
    def test_print_summary_verbose_analytics_thinking(
        self, mock_env, mock_config, thinking_count, thinking_percentage, should_be_present
    ):
        """Test that thinking line appears/disappears based on count (parametrized)."""
        metric = self._create_metric(
            provider_name="claude",
            thinking_count=thinking_count,
            thinking_percentage=thinking_percentage,
        )

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=[metric]):

            print_summary(mock_env, verbose=True)

            console_calls = mock_env.ui.console.print.call_args_list
            has_thinking = any("Thinking:" in str(c) for c in console_calls)
            assert has_thinking == should_be_present

    def test_print_summary_verbose_analytics_multiple_providers_deep_dive(self, mock_env, mock_config):
        """Test deep dive with multiple providers."""
        metric1 = self._create_metric(provider_name="claude", conversation_count=100)
        metric2 = self._create_metric(provider_name="chatgpt", conversation_count=50)

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=[metric1, metric2]):

            print_summary(mock_env, verbose=True)

            console_calls = mock_env.ui.console.print.call_args_list
            output = str(console_calls)
            assert "claude" in output
            assert "chatgpt" in output
            assert "Messages:" in output

    def test_print_summary_normal_no_deep_dive(self, mock_env, mock_config):
        """Test that non-verbose mode does not show deep dive."""
        metric = self._create_metric(provider_name="claude")

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=[metric]):

            print_summary(mock_env, verbose=False)

            console_calls = mock_env.ui.console.print.call_args_list
            assert not any("Deep Dive:" in str(c) for c in console_calls)


# ============================================================================
# TestPrintSummaryAnalyticsError: Error handling for analytics
# ============================================================================


class TestPrintSummaryAnalyticsError:
    """Test error handling in analytics computation."""

    def test_print_summary_analytics_exception_verbose(self, mock_env, mock_config):
        """Test that analytics exception is shown in verbose mode."""
        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", side_effect=RuntimeError("DB error")):

            print_summary(mock_env, verbose=True)

            console_calls = mock_env.ui.console.print.call_args_list
            assert any("Analytics computation failed" in str(c) or "DB error" in str(c) for c in console_calls)

    def test_print_summary_analytics_exception_silent_in_normal_mode(self, mock_env, mock_config):
        """Test that analytics exception is silent in non-verbose mode."""
        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", side_effect=RuntimeError("DB error")):

            print_summary(mock_env, verbose=False)

            # Should not raise exception
            mock_env.ui.summary.assert_called_once()

    def test_print_summary_analytics_import_error(self, mock_env, mock_config):
        """Test handling when analytics module import fails."""
        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", side_effect=ImportError("No module")):

            print_summary(mock_env, verbose=False)
            mock_env.ui.summary.assert_called_once()


# ============================================================================
# TestPrintSummaryIntegration: Full integration scenarios
# ============================================================================


class TestPrintSummaryIntegration:
    """Test complete print_summary scenarios."""

    def _create_metric(self, provider_name: str, conversation_count: int) -> MagicMock:
        """Create a metric."""
        metric = MagicMock()
        metric.provider_name = provider_name
        metric.conversation_count = conversation_count
        metric.message_count = conversation_count * 50
        metric.avg_messages_per_conversation = 50.0
        metric.avg_user_words = 20.0
        metric.avg_assistant_words = 100.0
        metric.tool_use_count = 0
        metric.tool_use_percentage = 0.0
        metric.thinking_count = 0
        metric.thinking_percentage = 0.0
        return metric

    def test_print_summary_complete_scenario_normal_mode(self, mock_env, mock_config, mock_run_data):
        """Test complete summary output in normal mode."""
        metric = self._create_metric("claude", 100)

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=mock_run_data), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=[metric]):

            print_summary(mock_env, verbose=False)

            call_args = mock_env.ui.summary.call_args
            lines = call_args[0][1]
            output = str(lines)
            assert "run-123" in output
            assert "Health:" in output

    def test_print_summary_complete_scenario_verbose_mode(self, mock_env, mock_config, mock_run_data):
        """Test complete summary output in verbose mode."""
        metric = self._create_metric("claude", 100)
        check1 = MagicMock()
        check1.name = "database"
        check1.status = "ok"
        check1.detail = "17.5 GB"

        mock_health = MagicMock()
        mock_health.cached = True
        mock_health.age_seconds = 30
        mock_health.checks = [check1]

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=mock_run_data), \
             patch("polylogue.cli.helpers.get_health", return_value=mock_health), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=[metric]):

            mock_env.ui.plain = False
            print_summary(mock_env, verbose=True)

            mock_env.ui.summary.assert_called_once()
            assert mock_env.ui.console.print.called

    def test_print_summary_empty_sources_list(self, mock_env, mock_config):
        """Test summary with no configured sources."""
        mock_config.sources = []

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="none"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=None):

            print_summary(mock_env, verbose=False)

            call_args = mock_env.ui.summary.call_args
            lines = call_args[0][1]
            assert any("Sources:" in str(line) for line in lines)

    def test_print_summary_with_special_characters_in_paths(self, mock_env, mock_config):
        """Test summary with special characters in paths."""
        mock_config.archive_root = Path("/data/archive with spaces/stuff")
        mock_config.render_root = Path("/data/archive with spaces/stuff/rendered")

        with patch("polylogue.cli.helpers.get_config", return_value=mock_config), \
             patch("polylogue.cli.helpers.latest_run", return_value=None), \
             patch("polylogue.cli.helpers.cached_health_summary", return_value="OK"), \
             patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"), \
             patch("polylogue.cli.analytics.compute_provider_comparison", return_value=None):

            print_summary(mock_env, verbose=False)

            mock_env.ui.summary.assert_called_once()
