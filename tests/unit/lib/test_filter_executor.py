"""Tests for ConversationFilter SQL pushdown and execution plan logic.

Production code under test: polylogue/lib/filter/filters.py
Methods: _sql_pushdown_params, _needs_content_loading, can_use_summaries, _has_post_filters
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from hypothesis import given, settings

from polylogue.lib.filter.filters import ConversationFilter
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

    def test_empty_filter_produces_empty_params(self) -> None:
        """No filters applied -> empty pushdown dict."""
        f = _make_filter()
        params = f._sql_pushdown_params()
        assert params == {}

    def test_single_provider_pushdown(self) -> None:
        """Single provider uses 'provider' key (not 'providers')."""
        f = _make_filter().provider("claude-ai")
        params = f._sql_pushdown_params()
        assert params == {"provider": "claude-ai"}

    def test_multi_provider_pushdown(self) -> None:
        """Multiple providers use 'providers' key as a list."""
        f = _make_filter().provider("claude-ai", "chatgpt")
        params = f._sql_pushdown_params()
        assert "providers" in params
        assert params["providers"] == ["claude-ai", "chatgpt"]

    def test_date_pushdown_since(self) -> None:
        """since date is pushed down as ISO string."""
        dt = datetime(2024, 6, 1, tzinfo=timezone.utc)
        f = _make_filter().since(dt)
        params = f._sql_pushdown_params()
        assert "since" in params
        assert params["since"] == dt.isoformat()

    def test_date_pushdown_until(self) -> None:
        """until date is pushed down as ISO string."""
        dt = datetime(2024, 12, 31, tzinfo=timezone.utc)
        f = _make_filter().until(dt)
        params = f._sql_pushdown_params()
        assert "until" in params
        assert params["until"] == dt.isoformat()

    def test_title_pattern_pushdown(self) -> None:
        """Title pattern is pushed down as title_contains."""
        f = _make_filter().title("important")
        params = f._sql_pushdown_params()
        assert params == {"title_contains": "important"}

    def test_combined_pushdown(self) -> None:
        """Multiple pushdown-eligible filters produce combined params."""
        dt_since = datetime(2024, 1, 1, tzinfo=timezone.utc)
        dt_until = datetime(2024, 12, 31, tzinfo=timezone.utc)
        f = _make_filter().provider("claude-ai").since(dt_since).until(dt_until).title("test")
        params = f._sql_pushdown_params()
        assert params["provider"] == "claude-ai"
        assert params["since"] == dt_since.isoformat()
        assert params["until"] == dt_until.isoformat()
        assert params["title_contains"] == "test"


# =============================================================================
# Execution plan (can_use_summaries / _needs_content_loading)
# =============================================================================


class TestExecutionPlan:
    """Tests for execution plan determination."""

    def test_summary_compatible_for_simple_filters(self) -> None:
        """SQL-only filters are summary-compatible (don't need message content)."""
        f = _make_filter().provider("claude-ai").title("test")
        assert f.can_use_summaries() is True

    def test_content_required_for_has_thinking(self) -> None:
        """has('thinking') requires content loading (messages needed)."""
        f = _make_filter().has("thinking")
        assert f.can_use_summaries() is False
        assert f._needs_content_loading() is True

    def test_content_required_for_has_tools(self) -> None:
        """has('tools') requires content loading."""
        f = _make_filter().has("tools")
        assert f.can_use_summaries() is False

    def test_content_required_for_negative_fts(self) -> None:
        """exclude_text() requires content loading."""
        f = _make_filter().exclude_text("secret")
        assert f.can_use_summaries() is False

    def test_content_required_for_path_filters(self) -> None:
        """Path filters reconcile against runtime semantic facts."""
        f = _make_filter().referenced_path("/workspace/polylogue/README.md")
        assert f.can_use_summaries() is False
        assert f._needs_content_loading() is True

    def test_content_required_for_action_filters(self) -> None:
        """Action filters reconcile against runtime semantic facts."""
        f = _make_filter().action("agent")
        assert f.can_use_summaries() is False
        assert f._needs_content_loading() is True

    def test_content_required_for_tool_filters(self) -> None:
        """Tool filters reconcile against runtime semantic facts."""
        f = _make_filter().tool("bash")
        assert f.can_use_summaries() is False
        assert f._needs_content_loading() is True

    def test_content_required_for_custom_predicates(self) -> None:
        """Custom where() predicates require content loading."""
        f = _make_filter().where(lambda c: len(c.messages) > 5)
        assert f.can_use_summaries() is False

    def test_content_required_for_sort_by_messages(self) -> None:
        """Sorting by messages requires content loading."""
        f = _make_filter().sort("messages")
        assert f.can_use_summaries() is False

    def test_content_required_for_sort_by_words(self) -> None:
        """Sorting by words requires content loading."""
        f = _make_filter().sort("words")
        assert f.can_use_summaries() is False

    def test_summary_ok_for_date_sort(self) -> None:
        """Sorting by date doesn't need content loading."""
        f = _make_filter().sort("date")
        assert f.can_use_summaries() is True

    def test_has_summary_is_summary_compatible(self) -> None:
        """has('summary') doesn't require message content."""
        f = _make_filter().has("summary")
        assert f.can_use_summaries() is True


# =============================================================================
# _has_post_filters
# =============================================================================


class TestHasPostFilters:
    """Tests for post-filter detection."""

    def test_no_post_filters_for_pushdown_only(self) -> None:
        """Pushdown-only filters have no post-filters."""
        f = _make_filter().provider("claude-ai").since(datetime(2024, 1, 1, tzinfo=timezone.utc))
        assert f._has_post_filters() is False

    def test_excluded_providers_require_post_filter(self) -> None:
        """Excluded providers need in-memory post-filtering."""
        f = _make_filter().exclude_provider("chatgpt")
        assert f._has_post_filters() is True

    def test_tags_require_post_filter(self) -> None:
        """Tag filters need in-memory post-filtering."""
        f = _make_filter().tag("important")
        assert f._has_post_filters() is True

    def test_has_types_require_post_filter(self) -> None:
        """Content type filters need post-filtering."""
        f = _make_filter().has("thinking")
        assert f._has_post_filters() is True

    def test_semantic_filters_require_post_filter(self) -> None:
        """Path/action/tool filters need runtime semantic post-filtering."""
        f = _make_filter().referenced_path("/workspace/polylogue/README.md").action("agent").tool("bash")
        assert f._has_post_filters() is True
        assert f.build_query_plan().can_count_in_sql() is False

    def test_candidate_query_keeps_stable_tool_filter_but_clears_unstable_action_filters(self) -> None:
        """Candidate fetch may keep raw tool-name narrowing while clearing stale action/path semantics."""
        plan = (
            _make_filter()
            .referenced_path("/workspace/polylogue/README.md")
            .action("agent")
            .tool("bash")
            .build_query_plan()
        )

        candidate = plan.fetch_record_query()
        assert candidate.path_terms == ()
        assert candidate.action_terms == ()
        assert candidate.tool_terms == ("bash",)


# =============================================================================
# Property-based tests
# =============================================================================


class TestFilterChainProperties:
    """Property-based tests for filter chain safety."""

    @given(filter_chain_strategy(min_filters=1, max_filters=5))
    @settings(max_examples=50)
    def test_sql_pushdown_params_never_crashes(self, chain: list[dict[str, object]]) -> None:
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
