"""Tests for ConversationFilter SQL pushdown and execution plan logic.

Production code under test: polylogue/lib/filters.py
Methods: _sql_pushdown_params, _needs_content_loading, can_use_summaries, _has_post_filters
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.lib.filters import ConversationFilter
from tests.infra.strategies.filters import filter_chain_strategy


def _make_filter() -> ConversationFilter:
    """Create a ConversationFilter with a mock repository."""
    mock_repo = MagicMock()
    return ConversationFilter(repository=mock_repo)


# =============================================================================
# _sql_pushdown_params
# =============================================================================


class TestSqlPushdownParams:
    """Tests for SQL pushdown parameter generation."""

    def test_empty_filter_produces_empty_params(self):
        """No filters applied -> empty pushdown dict."""
        f = _make_filter()
        params = f._sql_pushdown_params()
        assert params == {}

    def test_single_provider_pushdown(self):
        """Single provider uses 'provider' key (not 'providers')."""
        f = _make_filter().provider("claude")
        params = f._sql_pushdown_params()
        assert params == {"provider": "claude"}

    def test_multi_provider_pushdown(self):
        """Multiple providers use 'providers' key as a list."""
        f = _make_filter().provider("claude", "chatgpt")
        params = f._sql_pushdown_params()
        assert "providers" in params
        assert params["providers"] == ["claude", "chatgpt"]

    def test_date_pushdown_since(self):
        """since date is pushed down as ISO string."""
        dt = datetime(2024, 6, 1, tzinfo=timezone.utc)
        f = _make_filter().since(dt)
        params = f._sql_pushdown_params()
        assert "since" in params
        assert params["since"] == dt.isoformat()

    def test_date_pushdown_until(self):
        """until date is pushed down as ISO string."""
        dt = datetime(2024, 12, 31, tzinfo=timezone.utc)
        f = _make_filter().until(dt)
        params = f._sql_pushdown_params()
        assert "until" in params
        assert params["until"] == dt.isoformat()

    def test_title_pattern_pushdown(self):
        """Title pattern is pushed down as title_contains."""
        f = _make_filter().title("important")
        params = f._sql_pushdown_params()
        assert params == {"title_contains": "important"}

    def test_combined_pushdown(self):
        """Multiple pushdown-eligible filters produce combined params."""
        dt_since = datetime(2024, 1, 1, tzinfo=timezone.utc)
        dt_until = datetime(2024, 12, 31, tzinfo=timezone.utc)
        f = _make_filter().provider("claude").since(dt_since).until(dt_until).title("test")
        params = f._sql_pushdown_params()
        assert params["provider"] == "claude"
        assert params["since"] == dt_since.isoformat()
        assert params["until"] == dt_until.isoformat()
        assert params["title_contains"] == "test"


# =============================================================================
# Execution plan (can_use_summaries / _needs_content_loading)
# =============================================================================


class TestExecutionPlan:
    """Tests for execution plan determination."""

    def test_summary_compatible_for_simple_filters(self):
        """SQL-only filters are summary-compatible (don't need message content)."""
        f = _make_filter().provider("claude").title("test")
        assert f.can_use_summaries() is True

    def test_content_required_for_has_thinking(self):
        """has('thinking') requires content loading (messages needed)."""
        f = _make_filter().has("thinking")
        assert f.can_use_summaries() is False
        assert f._needs_content_loading() is True

    def test_content_required_for_has_tools(self):
        """has('tools') requires content loading."""
        f = _make_filter().has("tools")
        assert f.can_use_summaries() is False

    def test_content_required_for_negative_fts(self):
        """exclude_text() requires content loading."""
        f = _make_filter().exclude_text("secret")
        assert f.can_use_summaries() is False

    def test_content_required_for_custom_predicates(self):
        """Custom where() predicates require content loading."""
        f = _make_filter().where(lambda c: len(c.messages) > 5)
        assert f.can_use_summaries() is False

    def test_content_required_for_sort_by_messages(self):
        """Sorting by messages requires content loading."""
        f = _make_filter().sort("messages")
        assert f.can_use_summaries() is False

    def test_content_required_for_sort_by_words(self):
        """Sorting by words requires content loading."""
        f = _make_filter().sort("words")
        assert f.can_use_summaries() is False

    def test_summary_ok_for_date_sort(self):
        """Sorting by date doesn't need content loading."""
        f = _make_filter().sort("date")
        assert f.can_use_summaries() is True

    def test_has_summary_is_summary_compatible(self):
        """has('summary') doesn't require message content."""
        f = _make_filter().has("summary")
        assert f.can_use_summaries() is True


# =============================================================================
# _has_post_filters
# =============================================================================


class TestHasPostFilters:
    """Tests for post-filter detection."""

    def test_no_post_filters_for_pushdown_only(self):
        """Pushdown-only filters have no post-filters."""
        f = _make_filter().provider("claude").since(datetime(2024, 1, 1, tzinfo=timezone.utc))
        assert f._has_post_filters() is False

    def test_excluded_providers_require_post_filter(self):
        """Excluded providers need in-memory post-filtering."""
        f = _make_filter().exclude_provider("chatgpt")
        assert f._has_post_filters() is True

    def test_tags_require_post_filter(self):
        """Tag filters need in-memory post-filtering."""
        f = _make_filter().tag("important")
        assert f._has_post_filters() is True

    def test_has_types_require_post_filter(self):
        """Content type filters need post-filtering."""
        f = _make_filter().has("thinking")
        assert f._has_post_filters() is True


# =============================================================================
# Property-based tests
# =============================================================================


class TestFilterChainProperties:
    """Property-based tests for filter chain safety."""

    @given(filter_chain_strategy(min_filters=1, max_filters=5))
    @settings(max_examples=50)
    def test_sql_pushdown_params_never_crashes(self, chain: list[dict]):
        """Building pushdown params from any filter chain never crashes."""
        f = _make_filter()
        # Apply filters from the chain
        for spec in chain:
            filter_type = spec.get("type")
            value = spec.get("value")
            if filter_type == "provider" and isinstance(value, str):
                f = f.provider(value)
            elif filter_type == "contains" and isinstance(value, str):
                f = f.contains(value)
            elif filter_type == "since" and isinstance(value, str):
                try:
                    f = f.since(value)
                except ValueError:
                    pass  # Invalid dates are OK to skip
            elif filter_type == "until" and isinstance(value, str):
                try:
                    f = f.until(value)
                except ValueError:
                    pass
            elif filter_type == "limit" and isinstance(value, int):
                f = f.limit(value)

        # Must not crash
        params = f._sql_pushdown_params()
        assert isinstance(params, dict)
        # can_use_summaries must not crash
        result = f.can_use_summaries()
        assert isinstance(result, bool)
