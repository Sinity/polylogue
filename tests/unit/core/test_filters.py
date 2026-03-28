"""Tests for ConversationFilter fluent API."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from polylogue.lib.filters import ConversationFilter
from polylogue.lib.models import Conversation, ConversationSummary, Message
from polylogue.schemas.unified import (
    HarmonizedMessage,
    bulk_harmonize,
    extract_chatgpt_text,
    extract_claude_code_text,
    extract_codex_text,
    extract_content_blocks,
    extract_from_provider_meta,
    extract_harmonized_message,
    extract_reasoning_traces,
    extract_token_usage,
    harmonize_parsed_message,
    is_message_record,
)
from polylogue.schemas.validator import (
    SchemaValidator,
    ValidationResult,
    validate_provider_export,
)
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.index import rebuild_index
from polylogue.storage.repository import ConversationRepository
from tests.infra.helpers import ConversationBuilder

# =============================================================================
# TEST DATA: PARAMETRIZATION CONSTANTS (SCREAMING_CASE)
# =============================================================================

PROVIDER_FILTER_CASES = [
    ("claude", 2, "Filter by single provider"),
    (("claude", "chatgpt"), 3, "Filter by multiple providers"),
    ("chatgpt", 1, "Exclude specific provider (exclude_provider)"),
]

SORT_OPERATION_CASES = [
    ("date", "Sort by date"),
    ("messages", "Sort by message count"),
    ("random", "Random sort"),
]

SORT_VARIANTS_CASES = [
    ("tokens", "tokens"),
    ("words", "words"),
    ("longest", "longest"),
    ("messages", "messages"),
]

DATE_METHOD_CASES = [
    ("since", "_since_date", "yesterday"),
    ("since", "_since_date", "2025-01-15"),
    ("since", "_since_date", "last week"),
    ("until", "_until_date", "today"),
]

BRANCH_PREDICATE_CASES = [
    ("is_continuation", True, 1, "is_continuation(True)"),
    ("is_continuation", False, 1, "is_continuation(False)"),
    ("is_sidechain", True, 1, "is_sidechain(True)"),
    ("is_sidechain", False, 1, "is_sidechain(False)"),
    ("has_branches", True, 1, "has_branches(True)"),
    ("has_branches", False, 1, "has_branches(False)"),
]

FTS_PROVIDER_CASES = [
    ("errors", "claude", True, "FTS + provider match"),
    ("async", "chatgpt", True, "FTS + provider match (chatgpt)"),
    ("schema", "chatgpt", False, "FTS + provider mismatch"),
]

DRIFT_DETECTION_CASES = [
    ("unexpected_field", "Unexpected field detection"),
    ("additional_true", "additionalProperties: true"),
    ("additional_schema", "additionalProperties with schema"),
    ("nested_object", "Nested object drift"),
    ("list_items", "Array items drift"),
]

PICK_OPERATION_CASES = [
    (False, None, None, None, "No results case"),
    (True, False, None, "", "Non-TTY returns first"),
    (True, True, "builtins.input", "", "Empty input returns first"),
    (True, True, "builtins.input", "1", "Valid numeric choice"),
    (True, True, "builtins.input", "999", "Out-of-range choice"),
    (True, True, "builtins.input", "not a number", "Non-numeric input"),
    (True, True, "builtins.input", EOFError, "EOFError handling"),
    (True, True, "builtins.input", KeyboardInterrupt, "KeyboardInterrupt handling"),
]

REASONING_TRACES_CASES = [
    (None, "claude", 0, None, "empty_none_content"),
    (["string", 123], "claude", 0, None, "non_dict_block"),
    ([{"type": "thinking", "thinking": "Let me think..."}], "claude", 1, "Let me think...", "thinking_block"),
    ([{"isThought": True, "text": "Gemini thinking"}], "gemini", 1, "Gemini thinking", "gemini_thought"),
    ([{"type": "thinking", "text": "Fallback text"}], "claude", 1, "Fallback text", "thinking_fallback"),
]

CONTENT_BLOCKS_CASES = [
    (None, 0, [], "empty_none"),
    (["string", 123, None], 0, [], "non_dict_items"),
    ([{"type": "text", "text": "Hello"}], 1, ["text"], "text_block"),
    ([{"type": "thinking", "thinking": "Thought"}], 1, ["thinking"], "thinking_block"),
    ([{"type": "tool_use", "name": "bash", "id": "tool1", "input": {"command": "ls"}}], 1, ["tool_use"], "tool_use_block"),
    ([{"type": "tool_result", "content": "result data"}], 1, ["tool_result"], "tool_result_block"),
    ([{"type": "tool_result"}], 1, ["tool_result"], "tool_result_no_content"),
    ([{"type": "code", "text": "print('hello')", "language": "python"}], 1, ["code"], "code_block_text"),
    ([{"type": "code", "code": "def test(): pass"}], 1, ["code"], "code_block_code"),
    ([{"type": "unknown", "data": "something"}], 0, [], "unknown_block_type"),
]

EXTRACT_TEXT_CASES = [
    (extract_claude_code_text, None, "", "claude_code_none"),
    (extract_claude_code_text, ["string", 123], "", "claude_code_non_dict"),
    (extract_chatgpt_text, None, "", "chatgpt_none"),
    (extract_chatgpt_text, {}, "", "chatgpt_no_parts"),
    (extract_chatgpt_text, {"parts": "string"}, "string", "chatgpt_parts_as_string"),
    (extract_chatgpt_text, {"parts": [123, "text", {"key": "val"}]}, "text", "chatgpt_non_string_parts"),
    (extract_codex_text, {"data": "not a list"}, "", "codex_non_list"),
]

HARMONIZED_MESSAGE_PROVIDER_CASES = [
    ("claude-code", "claude_code_msg", "Claude Code extraction"),
    ("claude-ai", "claude_ai_msg", "Claude AI extraction"),
    ("chatgpt", "chatgpt_msg", "ChatGPT extraction"),
    ("gemini", "gemini_msg", "Gemini extraction"),
    ("codex", "codex_msg", "Codex extraction"),
]

MESSAGE_RECORD_TYPE_CASES = [
    ("claude-code", "user", True, "Claude Code user"),
    ("claude-code", "assistant", True, "Claude Code assistant"),
    ("claude-code", "metadata", False, "Claude Code metadata"),
    ("chatgpt", "anything", True, "Other providers always True"),
]

TOKEN_USAGE_CASES = [
    (None, None, "None usage"),
    ({}, None, "Empty dict (no tokens)"),
    ({"input_tokens": 100}, "partial", "Partial token fields"),
]


@pytest.fixture
def filter_db(tmp_path):
    """Create database with test conversations for filter tests."""
    db_path = tmp_path / "filter_test.db"

    (ConversationBuilder(db_path, "claude-1")
     .provider("claude")
     .title("Python Error Handling")
     .add_message("m1", role="user", text="How do I handle errors in Python?")
     .add_message("m2", role="assistant", text="You can use try-except blocks.")
     .metadata({"tags": ["python", "errors"]})
     .save())

    (ConversationBuilder(db_path, "chatgpt-1")
     .provider("chatgpt")
     .title("JavaScript Async")
     .add_message("m3", role="user", text="How do async functions work?")
     .add_message("m4", role="assistant", text="Async functions return promises.")
     .metadata({"tags": ["javascript"]})
     .save())

    (ConversationBuilder(db_path, "claude-2")
     .provider("claude")
     .title("Database Design")
     .add_message("m5", role="user", text="How to design a database schema?")
     .add_message("m6", role="assistant", text="Start with identifying entities.")
     .metadata({"tags": ["database", "design"]})
     .save())

    with open_connection(db_path) as conn:
        rebuild_index(conn)

    return db_path


@pytest.fixture
def filter_repo(filter_db):
    """Create repository for filter tests."""
    backend = SQLiteBackend(db_path=filter_db)
    return ConversationRepository(backend=backend)


class TestConversationFilterChaining:
    """Tests for filter method chaining."""

    def test_filter_returns_self(self, filter_repo):
        """Every fluent filter method must return self."""
        CHAINABLE_METHODS = [
            lambda f: f.provider("claude"),
            lambda f: f.since("2024-01-01"),
            lambda f: f.until("2025-01-01"),
            lambda f: f.limit(10),
            lambda f: f.sort("date"),
            lambda f: f.reverse(),
            lambda f: f.tag("test"),
            lambda f: f.contains("hello"),
            lambda f: f.title("test"),
            lambda f: f.similar("query"),
        ]
        for method_fn in CHAINABLE_METHODS:
            fresh = ConversationFilter(filter_repo)
            assert method_fn(fresh) is fresh, f"Method not chainable: {method_fn}"

    @pytest.mark.asyncio
    async def test_filter_chain_multiple_methods(self, filter_repo):
        """Chain must apply ALL filters — provider, limit both take effect."""
        result = await (
            ConversationFilter(filter_repo)
            .provider("claude")
            .limit(1)
            .sort("date")
            .list()
        )
        assert isinstance(result, list)
        assert len(result) <= 1  # limit applied
        assert all(c.provider == "claude" for c in result)  # provider applied


class TestConversationFilterMethods:
    """Consolidated tests for filter methods (provider, tag, text, title, id, limit)."""

    FILTER_METHOD_CASES = [
        ("provider_single", lambda f: f.provider("claude"), 2, "claude", "Filter by single provider"),
        ("provider_multi", lambda f: f.provider("claude", "chatgpt"), 3, "multi", "Filter by multiple providers"),
        ("exclude_provider", lambda f: f.exclude_provider("claude"), 1, "not_claude", "Exclude specific provider"),
        ("tag_python", lambda f: f.tag("python"), None, None, "Filter by tag (or empty)"),
        ("exclude_tag", lambda f: f.exclude_tag("nonexistent-tag"), 3, None, "Exclude nonexistent tag"),
        ("contains", lambda f: f.contains("Python"), None, None, "Filter contains text"),
        ("exclude_text", lambda f: f.exclude_text("database"), None, None, "Exclude text"),
        ("limit_1", lambda f: f.limit(1), 1, None, "Limit to 1 result"),
        ("limit_0", lambda f: f.limit(0), 0, None, "Limit of zero"),
        ("title_Python", lambda f: f.title("Python"), 1, "title", "Filter by title"),
        ("title_python_case", lambda f: f.title("python"), 1, "title", "Title case insensitive"),
        ("id_prefix", lambda f: f.id("claude"), 2, "id_prefix", "Filter by ID prefix"),
    ]

    @pytest.mark.parametrize("method_name,filter_fn,expected_count,check_type,description", FILTER_METHOD_CASES)
    @pytest.mark.asyncio
    async def test_filter_method(self, filter_repo, method_name, filter_fn, expected_count, check_type, description):
        """Test individual filter methods."""
        result = await filter_fn(ConversationFilter(filter_repo)).list()

        if expected_count is not None:
            assert len(result) == expected_count, f"Failed {description}: expected {expected_count}, got {len(result)}"
        else:
            assert isinstance(result, list), f"Failed {description}: should return list"

        # Type-specific assertions
        if check_type == "claude" and result:
            assert all(c.provider == "claude" for c in result)
        elif check_type == "not_claude" and result:
            assert all(c.provider != "claude" for c in result)
        elif check_type == "multi" and result:
            assert all(c.provider in ("claude", "chatgpt") for c in result)
        elif check_type == "title" and result:
            assert "Python" in result[0].display_title or "python" in result[0].display_title
        elif check_type == "id_prefix" and result:
            assert all(c.id.startswith("claude") for c in result)


class TestConversationFilterTerminal:
    """Tests for terminal methods."""

    @pytest.mark.asyncio
    async def test_filter_first(self, filter_repo):
        """first() returns single conversation."""
        result = await ConversationFilter(filter_repo).first()
        assert result is not None
        assert hasattr(result, "id")

    @pytest.mark.asyncio
    async def test_filter_first_empty(self, filter_repo):
        """first() returns None when no matches."""
        result = await ConversationFilter(filter_repo).provider("nonexistent").first()
        assert result is None

    @pytest.mark.asyncio
    async def test_filter_count(self, filter_repo):
        """count() returns number of matches."""
        count = await ConversationFilter(filter_repo).count()
        assert count == 3

    @pytest.mark.asyncio
    async def test_filter_count_with_filter(self, filter_repo):
        """count() respects filters."""
        count = await ConversationFilter(filter_repo).provider("claude").count()
        assert count == 2

    @pytest.mark.asyncio
    async def test_filter_delete_removes_conversations(self, filter_repo):
        """delete() removes matched conversations."""
        initial_count = await ConversationFilter(filter_repo).count()
        assert initial_count > 0

        deleted = await ConversationFilter(filter_repo).limit(1).delete()
        assert deleted == 1

        final_count = await ConversationFilter(filter_repo).count()
        assert final_count == initial_count - 1

    @pytest.mark.asyncio
    async def test_filter_delete_counts_multiple_deleted_conversations(self, filter_repo):
        """delete() should count every successful deletion, not just the last one."""
        deleted = await ConversationFilter(filter_repo).provider("claude").delete()

        assert deleted == 2
        assert await ConversationFilter(filter_repo).count() == 1

    @pytest.mark.asyncio
    async def test_filter_delete_uses_summaries_when_possible(self, filter_repo):
        """delete() uses summary-only loading for content-independent filters."""
        filter_obj = ConversationFilter(filter_repo).provider("claude").limit(1)
        filter_obj.list_summaries = AsyncMock(  # type: ignore[method-assign]
            return_value=[ConversationSummary(id="claude-1", provider="claude")]
        )
        filter_obj.list = AsyncMock(side_effect=AssertionError("full conversations should not be loaded"))  # type: ignore[method-assign]
        delete_mock = AsyncMock(return_value=True)
        filter_repo.backend.delete_conversation = delete_mock  # type: ignore[method-assign]

        deleted = await filter_obj.delete()

        assert deleted == 1
        filter_obj.list_summaries.assert_awaited_once()
        delete_mock.assert_awaited_once_with("claude-1")

    @pytest.mark.asyncio
    async def test_filter_delete_uses_full_conversations_for_content_filters(self, filter_repo):
        """delete() should use list() when summaries cannot satisfy the filter."""
        filter_obj = ConversationFilter(filter_repo).exclude_text("errors")
        filter_obj.list_summaries = AsyncMock(side_effect=AssertionError("summary path should not run"))  # type: ignore[method-assign]
        filter_obj.list = AsyncMock(  # type: ignore[method-assign]
            return_value=[Conversation(id="chatgpt-1", provider="chatgpt", messages=[])]
        )
        delete_mock = AsyncMock(return_value=True)
        filter_repo.backend.delete_conversation = delete_mock  # type: ignore[method-assign]

        deleted = await filter_obj.delete()

        assert deleted == 1
        filter_obj.list.assert_awaited_once()
        delete_mock.assert_awaited_once_with("chatgpt-1")


class TestConversationFilterSort:
    """Tests for sorting."""

    @pytest.mark.parametrize("sort_key,description", SORT_OPERATION_CASES)
    @pytest.mark.asyncio
    async def test_filter_sort(self, filter_repo, sort_key, description):
        """Test sorting by various keys."""
        result = await ConversationFilter(filter_repo).sort(sort_key).list()
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_filter_sort_reverse(self, filter_repo):
        """Reverse sort order."""
        normal = await ConversationFilter(filter_repo).sort("date").list()
        reversed_list = await ConversationFilter(filter_repo).sort("date").reverse().list()
        if len(normal) > 1:
            assert normal[0].id == reversed_list[-1].id
            assert normal[-1].id == reversed_list[0].id


class TestConversationFilterCustom:
    """Tests for custom predicates."""

    @pytest.mark.asyncio
    async def test_filter_where_predicate(self, filter_repo):
        """Filter with custom predicate."""
        result = await (
            ConversationFilter(filter_repo)
            .where(lambda c: len(c.messages) >= 2)
            .list()
        )
        assert all(len(c.messages) >= 2 for c in result)


class TestConversationFilterInternalContracts:
    """Fast contracts for internal helper methods used by mutmut."""

    @pytest.fixture
    def mock_repo(self):
        repo = Mock()
        repo.resolve_id = AsyncMock(return_value=None)
        return repo

    def test_sql_pushdown_params_emit_all_sql_safe_filters(self, mock_repo):
        """SQL-pushdown helper should emit every active SQL-safe filter."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        filter_obj = (
            ConversationFilter(mock_repo)
            .provider("claude")
            .since(start)
            .until(end)
            .title("Alpha")
            .has_tool_use()
            .has_thinking()
            .min_messages(2)
            .max_messages(8)
            .min_words(5)
            .has_file_operations()
            .has_git_operations()
            .has_subagent_spawns()
        )

        assert filter_obj._sql_pushdown_params() == {
            "provider": filter_obj._providers[0],
            "since": "2024-01-01T00:00:00+00:00",
            "until": "2024-12-31T00:00:00+00:00",
            "title_contains": "Alpha",
            "has_tool_use": True,
            "has_thinking": True,
            "min_messages": 2,
            "max_messages": 8,
            "min_words": 5,
            "has_file_ops": True,
            "has_git_ops": True,
            "has_subagent": True,
        }

    def test_apply_common_filters_matches_metadata_contract(self, mock_repo):
        """Metadata filtering should preserve exactly the matching summaries."""
        filter_obj = (
            ConversationFilter(mock_repo)
            .provider("claude")
            .exclude_provider("codex")
            .tag("science")
            .exclude_tag("drop")
            .title("Python")
            .since(datetime(2024, 1, 1, tzinfo=timezone.utc))
            .until(datetime(2024, 6, 1, tzinfo=timezone.utc))
            .id("conv-a")
            .has("summary")
        )
        summaries = [
            ConversationSummary(
                id="conv-alpha",
                provider="claude",
                title="Alpha Python",
                updated_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
                metadata={"tags": ["science"], "summary": "alpha"},
            ),
            ConversationSummary(
                id="conv-beta",
                provider="claude",
                title="Alpha Python",
                updated_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
                metadata={"tags": ["science", "drop"], "summary": "beta"},
            ),
            ConversationSummary(
                id="conv-codex",
                provider="codex",
                title="Alpha Python",
                updated_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
                metadata={"tags": ["science"], "summary": "gamma"},
            ),
        ]

        result = filter_obj._apply_common_filters(summaries, sql_pushed=False)

        assert [summary.id for summary in result] == ["conv-alpha"]

    def test_has_post_filters_tracks_predicates_and_negative_fts_independently(self, mock_repo):
        """Post-filter detection must treat predicates and negative FTS as separate triggers."""
        assert ConversationFilter(mock_repo)._has_post_filters() is False
        assert ConversationFilter(mock_repo).exclude_tag("drop")._has_post_filters() is True
        assert ConversationFilter(mock_repo).has("summary")._has_post_filters() is True
        assert ConversationFilter(mock_repo).exclude_text("error")._has_post_filters() is True
        assert ConversationFilter(mock_repo).where(lambda c: True)._has_post_filters() is True

    def test_needs_content_loading_only_for_message_dependent_filters(self, mock_repo):
        """Summary compatibility should only close over message-dependent filters."""
        assert ConversationFilter(mock_repo).can_use_summaries() is True
        assert ConversationFilter(mock_repo).has("thinking")._needs_content_loading() is True
        assert ConversationFilter(mock_repo).exclude_text("error")._needs_content_loading() is True
        assert ConversationFilter(mock_repo).where(lambda c: True)._needs_content_loading() is True
        assert ConversationFilter(mock_repo).sort("words")._needs_content_loading() is True
        assert (
            ConversationFilter(mock_repo)
            .has_tool_use()
            .has_thinking()
            .min_messages(1)
            .max_messages(10)
            .min_words(1)
            .can_use_summaries()
            is True
        )

    def test_effective_fetch_limit_tracks_post_filters_and_sampling(self, mock_repo):
        """Backend fetch limit should expand only when post-processing requires it."""
        assert ConversationFilter(mock_repo)._effective_fetch_limit() is None
        assert ConversationFilter(mock_repo).limit(10)._effective_fetch_limit() == 20
        assert ConversationFilter(mock_repo).limit(10).tag("science")._effective_fetch_limit() == 500
        assert ConversationFilter(mock_repo).limit(10).sample(5)._effective_fetch_limit() == 200

    @pytest.mark.asyncio
    async def test_fetch_generic_prefers_resolved_id(self, mock_repo):
        """ID prefix resolution should short-circuit search/list fallbacks."""
        mock_repo.resolve_id = AsyncMock(return_value="conv-alpha")
        filter_obj = ConversationFilter(mock_repo).id("conv-a")
        get_by_id = AsyncMock(return_value="resolved")
        search = AsyncMock()
        list_all = AsyncMock()

        result = await filter_obj._fetch_generic(get_by_id, search, list_all)

        assert result == ["resolved"]
        get_by_id.assert_awaited_once_with("conv-alpha")
        search.assert_not_called()
        list_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_generic_falls_back_to_list_on_search_error(self, mock_repo):
        """FTS exceptions should fall back to the list backend path."""
        filter_obj = ConversationFilter(mock_repo).contains("needle")
        get_by_id = AsyncMock()
        search = AsyncMock(side_effect=Exception("fts boom"))
        list_all = AsyncMock(return_value=["fallback"])

        result = await filter_obj._fetch_generic(get_by_id, search, list_all)

        assert result == ["fallback"]
        search.assert_awaited_once_with("needle", 10000, None)
        list_all.assert_awaited_once_with(limit=None)

    @pytest.mark.asyncio
    async def test_fetch_generic_uses_search_limit_floor_and_provider_scope(self, mock_repo):
        """FTS fetches should use the search path with the enforced minimum limit."""
        filter_obj = ConversationFilter(mock_repo).contains("needle").provider("claude").limit(10)
        get_by_id = AsyncMock()
        search = AsyncMock(return_value=["hit"])
        list_all = AsyncMock()

        result = await filter_obj._fetch_generic(get_by_id, search, list_all)

        assert result == ["hit"]
        search.assert_awaited_once_with("needle", 100, filter_obj._providers)
        list_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_summary_candidates_uses_summary_search_backend(self, mock_repo):
        """Summary candidate fetches should use search_summaries for FTS queries."""
        mock_repo.get_summary = AsyncMock()
        mock_repo.search_summaries = AsyncMock(return_value=["summary-hit"])
        mock_repo.list_summaries = AsyncMock(return_value=[])
        filter_obj = ConversationFilter(mock_repo).contains("needle").provider("claude").limit(10)

        result = await filter_obj._fetch_summary_candidates()

        assert result == ["summary-hit"]
        mock_repo.search_summaries.assert_awaited_once_with(
            "needle",
            limit=100,
            providers=filter_obj._providers,
        )
        mock_repo.list_summaries.assert_not_called()

    def test_execute_pipeline_applies_filter_sort_sample_and_limit(self, mock_repo):
        """Execution pipeline should respect the full transformation order."""
        filter_obj = ConversationFilter(mock_repo).sample(2).limit(1)
        with patch("random.sample", return_value=[4, 2]) as random_sample:
            result = filter_obj._execute_pipeline(
                [6, 5, 4, 3, 2, 1],
                lambda items, _sql_pushed: [item for item in items if item % 2 == 0],
                lambda items: sorted(items, reverse=True),
            )

        random_sample.assert_called_once_with([6, 4, 2], 2)
        assert result == [4]

    @pytest.mark.asyncio
    async def test_list_and_list_summaries_use_fast_helper_paths(self, mock_repo):
        """Public list methods should use the helper fetch/pipeline surfaces directly."""
        full_filter = ConversationFilter(mock_repo)
        summary_filter = ConversationFilter(mock_repo)
        conversations = [
            Conversation(
                id="conv-1",
                provider="claude",
                messages=[Message(id="m1", role="user", text="substantive user question")],
            )
        ]
        summaries = [ConversationSummary(id="conv-1", provider="claude")]
        full_filter._fetch_candidates = AsyncMock(return_value=conversations)  # type: ignore[method-assign]
        summary_filter._fetch_summary_candidates = AsyncMock(return_value=summaries)  # type: ignore[method-assign]

        assert [conv.id for conv in await full_filter.list()] == ["conv-1"]
        assert [summary.id for summary in await summary_filter.list_summaries()] == ["conv-1"]
        full_filter._fetch_candidates.assert_awaited_once()
        summary_filter._fetch_summary_candidates.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_count_uses_sql_fast_path_and_ignores_display_limit(self, mock_repo):
        """count() should use repository COUNT(*) when all filters are SQL-safe."""
        mock_repo.count = AsyncMock(return_value=7)
        filter_obj = (
            ConversationFilter(mock_repo)
            .provider("claude")
            .title("Alpha")
            .min_messages(2)
            .limit(1)
        )
        filter_obj.list_summaries = AsyncMock(side_effect=AssertionError("summary path should not run"))  # type: ignore[method-assign]
        filter_obj.list = AsyncMock(side_effect=AssertionError("full path should not run"))  # type: ignore[method-assign]

        count = await filter_obj.count()

        assert count == 7
        mock_repo.count.assert_awaited_once_with(
            provider=filter_obj._providers[0],
            title_contains="Alpha",
            min_messages=2,
        )

    @pytest.mark.asyncio
    async def test_count_uses_summary_path_and_restores_limit(self, mock_repo):
        """Summary-compatible post-filters should count via list_summaries() without keeping limit()."""
        filter_obj = ConversationFilter(mock_repo).tag("science").limit(1)

        async def fake_list_summaries() -> list[ConversationSummary]:
            assert filter_obj._limit_count is None
            return [
                ConversationSummary(id="conv-1", provider="claude"),
                ConversationSummary(id="conv-2", provider="claude"),
                ConversationSummary(id="conv-3", provider="chatgpt"),
            ]

        filter_obj.list_summaries = AsyncMock(side_effect=fake_list_summaries)  # type: ignore[method-assign]
        filter_obj.list = AsyncMock(side_effect=AssertionError("full path should not run"))  # type: ignore[method-assign]

        count = await filter_obj.count()

        assert count == 3
        assert filter_obj._limit_count == 1
        filter_obj.list_summaries.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_count_with_fts_uses_summary_path_not_sql_count(self, mock_repo):
        """FTS count() should never use the SQL fast path."""
        mock_repo.count = AsyncMock(side_effect=AssertionError("SQL count should not run for FTS"))
        filter_obj = ConversationFilter(mock_repo).contains("needle").limit(1)

        async def fake_list_summaries() -> list[ConversationSummary]:
            assert filter_obj._limit_count is None
            return [ConversationSummary(id="conv-1", provider="claude")]

        filter_obj.list_summaries = AsyncMock(side_effect=fake_list_summaries)  # type: ignore[method-assign]
        filter_obj.list = AsyncMock(side_effect=AssertionError("full path should not run"))  # type: ignore[method-assign]

        count = await filter_obj.count()

        assert count == 1
        assert filter_obj._limit_count == 1
        filter_obj.list_summaries.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_count_uses_full_path_for_content_filters_and_restores_limit(self, mock_repo):
        """Content-dependent filters should force count() onto full conversations."""
        filter_obj = ConversationFilter(mock_repo).has("thinking").limit(1)

        async def fake_list() -> list[Conversation]:
            assert filter_obj._limit_count is None
            return [
                Conversation(
                    id="conv-1",
                    provider="claude",
                    messages=[Message(id="m1", role="assistant", text="thought")],
                ),
                Conversation(
                    id="conv-2",
                    provider="chatgpt",
                    messages=[Message(id="m2", role="assistant", text="thought 2")],
                ),
            ]

        filter_obj.list = AsyncMock(side_effect=fake_list)  # type: ignore[method-assign]
        filter_obj.list_summaries = AsyncMock(side_effect=AssertionError("summary path should not run"))  # type: ignore[method-assign]

        count = await filter_obj.count()

        assert count == 2
        assert filter_obj._limit_count == 1
        filter_obj.list.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_uses_similarity_search_then_post_filters(self, mock_repo):
        """similar() should use search_similar() and still honor metadata filters."""
        mock_repo.search_similar = AsyncMock(
            return_value=[
                Conversation(id="claude-1", provider="claude", messages=[]),
                Conversation(id="chatgpt-1", provider="chatgpt", messages=[]),
            ]
        )
        filter_obj = ConversationFilter(mock_repo).similar("query").provider("claude").limit(1)

        result = await filter_obj.list()

        assert [conversation.id for conversation in result] == ["claude-1"]
        mock_repo.search_similar.assert_awaited_once_with(
            "query",
            limit=1,
            vector_provider=None,
        )

    @pytest.mark.asyncio
    async def test_first_sets_limit_one_before_loading(self, mock_repo):
        """first() should narrow list() to a single result before executing."""
        filter_obj = ConversationFilter(mock_repo)

        async def fake_list() -> list[Conversation]:
            assert filter_obj._limit_count == 1
            return [Conversation(id="conv-1", provider="claude", messages=[])]

        filter_obj.list = AsyncMock(side_effect=fake_list)  # type: ignore[method-assign]

        result = await filter_obj.first()

        assert result is not None
        assert result.id == "conv-1"
        assert filter_obj._limit_count == 1
        filter_obj.list.assert_awaited_once()

    def test_apply_summary_sort_handles_missing_dates(self, mock_repo):
        """Summary date sorting should keep undated rows stable without TypeError."""
        filter_obj = ConversationFilter(mock_repo).sort("date")
        summaries = [
            ConversationSummary(
                id="dated",
                provider="claude",
                updated_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
            ),
            ConversationSummary(id="undated", provider="claude", updated_at=None),
        ]

        result = filter_obj._apply_summary_sort(summaries)

        assert [summary.id for summary in result] == ["dated", "undated"]


class TestFilterDateParsing:
    """Tests for date parsing in ConversationFilter.since() and until()."""

    @pytest.mark.parametrize("method_name,field_name", [("since", "_since_date"), ("until", "_until_date")])
    def test_date_method_raises_on_invalid(self, filter_repo, method_name, field_name):
        """Calling .since() or .until() with unparseable string raises ValueError."""
        f = ConversationFilter(filter_repo)
        with pytest.raises(ValueError, match="Cannot parse date"):
            getattr(f, method_name)("not-a-date")

    @pytest.mark.parametrize("method_name,field_name,date_str", DATE_METHOD_CASES)
    def test_date_method_accepts_string_formats(self, filter_repo, method_name, field_name, date_str):
        """since() and until() both accept natural-language and ISO date strings."""
        f = ConversationFilter(filter_repo)
        getattr(f, method_name)(date_str)
        assert getattr(f, field_name) is not None
        assert isinstance(getattr(f, field_name), datetime)


class TestFtsWithProviderFilter:
    """Tests for combined FTS search + provider filter."""

    @pytest.mark.parametrize("search_term,provider,should_find,description", FTS_PROVIDER_CASES)
    @pytest.mark.asyncio
    async def test_fts_with_provider(self, filter_repo, search_term, provider, should_find, description):
        """Test FTS search combined with provider filter."""
        result = await ConversationFilter(filter_repo).contains(search_term).provider(provider).list()
        if should_find:
            assert len(result) > 0, f"Should find '{search_term}' in {provider} conversations"
            assert all(c.provider == provider for c in result)
        else:
            assert len(result) == 0, f"Should not find '{search_term}' in {provider} conversations"


# MERGED FROM test_filters_schemas_coverage.py

# =============================================================================
# FILTERS.PY TESTS
# =============================================================================


class TestFiltersDateTimeHandling:
    """Test date handling with datetime objects (lines 193, 216)."""

    @pytest.fixture
    def filter_repo(self, tmp_path):
        """Create repository for filter tests."""
        db_path = tmp_path / "filter_test.db"
        with open_connection(db_path) as conn:
            rebuild_index(conn)
        now = datetime.now(timezone.utc)
        (ConversationBuilder(db_path, "conv1")
         .provider("claude")
         .created_at(now.isoformat())
         .save())
        backend = SQLiteBackend(db_path)
        return ConversationRepository(backend)

    @pytest.mark.parametrize("method_name,field_name", [("since", "_since_date"), ("until", "_until_date")])
    def test_date_method_accepts_datetime_object(self, filter_repo, method_name, field_name):
        """since() and until() both accept a datetime object directly."""
        dt = datetime(2024, 6, 15, tzinfo=timezone.utc)
        f = ConversationFilter(filter_repo)
        result = getattr(f, method_name)(dt)
        assert result is f
        assert getattr(f, field_name) == dt

    def test_since_and_until_together_datetime(self, filter_repo):
        """Test both since/until with datetime objects."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        filter_obj = ConversationFilter(filter_repo).since(start).until(end)
        assert filter_obj._since_date == start
        assert filter_obj._until_date == end


class TestFiltersSimilarAndBranches:
    """Test similar() and branch-related predicates."""

    @pytest.fixture
    def filter_repo(self, tmp_path):
        db_path = tmp_path / "filter_branches.db"
        with open_connection(db_path) as conn:
            rebuild_index(conn)

        (ConversationBuilder(db_path, "root")
         .provider("claude")
         .save())

        (ConversationBuilder(db_path, "cont")
         .provider("claude")
         .parent_conversation("root")
         .branch_type("continuation")
         .save())

        (ConversationBuilder(db_path, "side")
         .provider("claude")
         .parent_conversation("root")
         .branch_type("sidechain")
         .save())

        (ConversationBuilder(db_path, "branched")
         .provider("claude")
         .add_message("m1", role="user", text="test")
         .add_message("m2", role="assistant", text="resp1", branch_index=0)
         .add_message("m3", role="assistant", text="resp2", branch_index=1)
         .save())

        backend = SQLiteBackend(db_path)
        return ConversationRepository(backend)

    def test_similar_text(self, filter_repo):
        """Test .similar() stores text for vector search."""
        filter_obj = ConversationFilter(filter_repo).similar("test query")
        assert filter_obj._similar_text == "test query"

    @pytest.mark.parametrize("method,value,expected_predicates,description", BRANCH_PREDICATE_CASES)
    def test_branch_predicates(self, filter_repo, method, value, expected_predicates, description):
        """Test branch-related predicate methods."""
        filter_obj = ConversationFilter(filter_repo)
        getattr(filter_obj, method)(value)
        assert len(filter_obj._predicates) == expected_predicates

    @pytest.mark.asyncio
    async def test_branch_filters_match_expected_conversation_ids(self, filter_repo):
        """Branch-related filters should return the exact expected IDs."""
        assert sorted(c.id for c in await ConversationFilter(filter_repo).is_root().list()) == ["branched", "root"]
        assert sorted(c.id for c in await ConversationFilter(filter_repo).is_root(False).list()) == ["cont", "side"]
        assert [c.id for c in await ConversationFilter(filter_repo).is_continuation().list()] == ["cont"]
        assert sorted(c.id for c in await ConversationFilter(filter_repo).is_continuation(False).list()) == ["branched", "root", "side"]
        assert [c.id for c in await ConversationFilter(filter_repo).is_sidechain().list()] == ["side"]
        assert sorted(c.id for c in await ConversationFilter(filter_repo).is_sidechain(False).list()) == ["branched", "cont", "root"]
        assert sorted(c.id for c in await ConversationFilter(filter_repo).parent("root").list()) == ["cont", "side"]
        assert [c.id for c in await ConversationFilter(filter_repo).has_branches().list()] == ["branched"]
        assert sorted(c.id for c in await ConversationFilter(filter_repo).has_branches(False).list()) == ["cont", "root", "side"]


class TestFiltersApplyFiltersLogic:
    """Test _apply_filters with all branches."""

    @pytest.fixture
    def filter_repo_populated(self, tmp_path):
        db_path = tmp_path / "filter_populated.db"
        with open_connection(db_path) as conn:
            rebuild_index(conn)

        (ConversationBuilder(db_path, "claude1")
         .provider("claude")
         .title("Short")
         .add_message("m1", text="a")
         .save())

        (ConversationBuilder(db_path, "gpt1")
         .provider("chatgpt")
         .title("Long conversation")
         .add_message("m1", text="word " * 50)
         .add_message("m2", text="more " * 100)
         .save())

        (ConversationBuilder(db_path, "claude2")
         .provider("claude")
         .title("Medium")
         .add_message("m1", text="test " * 10)
         .add_message("m2", text="data " * 15)
         .save())

        backend = SQLiteBackend(db_path)
        return ConversationRepository(backend)

    @pytest.mark.asyncio
    async def test_excluded_providers_filter(self, filter_repo_populated):
        """Test filtering out specific providers."""
        results = await (ConversationFilter(filter_repo_populated)
                   .exclude_provider("chatgpt")
                   .list())
        assert all(c.provider != "chatgpt" for c in results)

    @pytest.mark.parametrize("sort_key,description", SORT_VARIANTS_CASES)
    @pytest.mark.asyncio
    async def test_sort_by_variant(self, filter_repo_populated, sort_key, description):
        """Test sorting by various metrics."""
        results = await (ConversationFilter(filter_repo_populated)
                   .sort(sort_key)
                   .list())
        assert len(results) == 3, f"Sort '{sort_key}' should return all 3 conversations"


class TestFiltersIDPrefixResolution:
    """Test ID prefix resolution paths."""

    @pytest.fixture
    def filter_repo_with_id(self, tmp_path):
        db_path = tmp_path / "filter_id.db"
        with open_connection(db_path) as conn:
            rebuild_index(conn)

        (ConversationBuilder(db_path, "abc123def456")
         .provider("claude")
         .save())

        backend = SQLiteBackend(db_path)
        return ConversationRepository(backend)

    @pytest.mark.asyncio
    async def test_id_prefix_exact_match_fast_path(self, filter_repo_with_id):
        """Test ID prefix fast path when prefix resolves to single conversation."""
        results = await (ConversationFilter(filter_repo_with_id)
                   .id("abc123")
                   .list())
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_fts_search_exception_handling(self, filter_repo_with_id):
        """Test FTS search fallback on exception."""
        filter_obj = ConversationFilter(filter_repo_with_id)
        filter_obj.contains("test")
        with patch.object(filter_repo_with_id, 'search', side_effect=Exception("FTS error")):
            results = await filter_obj.list()
            assert isinstance(results, list)


class TestFiltersListSummariesPaths:
    """Test list_summaries() and all its branches."""

    @pytest.fixture
    def filter_repo_summaries(self, tmp_path):
        db_path = tmp_path / "filter_summaries.db"
        with open_connection(db_path) as conn:
            rebuild_index(conn)

        (ConversationBuilder(db_path, "with_summary")
         .provider("claude")
         .metadata({"summary": "This is a summary"})
         .add_message("m1", text="Message")
         .save())

        (ConversationBuilder(db_path, "without_summary")
         .provider("claude")
         .add_message("m1", text="Message")
         .save())

        backend = SQLiteBackend(db_path)
        return ConversationRepository(backend)

    @pytest.mark.asyncio
    async def test_list_summaries_with_provider_filter(self, filter_repo_summaries):
        """Test list_summaries with provider filter."""
        results = await (ConversationFilter(filter_repo_summaries)
                   .provider("claude")
                   .list_summaries())
        assert all(isinstance(s, ConversationSummary) for s in results)

    @pytest.mark.asyncio
    async def test_list_summaries_with_tag_filter(self, filter_repo_summaries):
        """Test list_summaries with tag filter."""
        results = await (ConversationFilter(filter_repo_summaries)
                   .tag("mytag")
                   .list_summaries())
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_list_summaries_with_summary_has_type(self, filter_repo_summaries):
        """Test list_summaries with has('summary') filter."""
        results = await (ConversationFilter(filter_repo_summaries)
                   .has("summary")
                   .list_summaries())
        assert all(s.summary for s in results)

    @pytest.mark.asyncio
    async def test_list_summaries_cannot_use_content_filters(self, filter_repo_summaries):
        """Test that list_summaries rejects content-dependent filters."""
        with pytest.raises(ValueError, match="content-dependent filters"):
            await (ConversationFilter(filter_repo_summaries)
             .has("thinking")
             .list_summaries())

    def test_can_use_summaries_check(self, filter_repo_summaries):
        """Test can_use_summaries() method."""
        simple = ConversationFilter(filter_repo_summaries).provider("claude")
        assert simple.can_use_summaries() is True

        with_content = ConversationFilter(filter_repo_summaries).has("thinking")
        assert with_content.can_use_summaries() is False


class TestFiltersPick:
    """Test pick() interactive picker."""

    @pytest.fixture
    def filter_repo_pick(self, tmp_path):
        db_path = tmp_path / "filter_pick.db"
        with open_connection(db_path) as conn:
            rebuild_index(conn)

        for i in range(5):
            (ConversationBuilder(db_path, f"conv{i}")
             .provider("claude")
             .title(f"Conversation {i}")
             .save())

        backend = SQLiteBackend(db_path)
        return ConversationRepository(backend)

    @pytest.mark.asyncio
    async def test_pick_no_results(self, filter_repo_pick):
        """Test pick() with no matching conversations."""
        result = await (ConversationFilter(filter_repo_pick)
                  .provider("nonexistent")
                  .pick())
        assert result is None

    @pytest.mark.parametrize("has_results,is_tty,patch_target,input_value,description", [
        (True, False, None, None, "Non-TTY returns first"),
        (True, True, "builtins.input", "", "Empty input returns first"),
        (True, True, "builtins.input", "1", "Valid numeric choice"),
        (True, True, "builtins.input", "999", "Out-of-range choice"),
        (True, True, "builtins.input", "not a number", "Non-numeric input"),
        (True, True, "builtins.input", "EOF", "EOFError handling"),
        (True, True, "builtins.input", "INTERRUPT", "KeyboardInterrupt handling"),
    ])
    @pytest.mark.asyncio
    async def test_pick_outcomes(self, filter_repo_pick, has_results, is_tty, patch_target, input_value, description):
        """Test pick() with various input scenarios."""
        if not has_results:
            result = await (ConversationFilter(filter_repo_pick)
                      .provider("nonexistent")
                      .pick())
            assert result is None
        else:
            with patch("sys.stdout.isatty", return_value=is_tty):
                if patch_target:
                    if input_value == "EOF":
                        with patch(patch_target, side_effect=EOFError):
                            result = await ConversationFilter(filter_repo_pick).pick()
                            assert result is None
                    elif input_value == "INTERRUPT":
                        with patch(patch_target, side_effect=KeyboardInterrupt):
                            result = await ConversationFilter(filter_repo_pick).pick()
                            assert result is None
                    else:
                        with patch(patch_target, return_value=input_value):
                            result = await ConversationFilter(filter_repo_pick).pick()
                            if input_value == "" or input_value == "1":
                                assert result is not None
                            elif input_value == "999" or input_value == "not a number":
                                assert result is None
                else:
                    result = await ConversationFilter(filter_repo_pick).pick()
                    assert result is not None



    @pytest.mark.asyncio
    async def test_pick_with_filter_respects_filter(self, filter_repo_pick):
        """pick() on filtered results returns from the filtered set."""
        filtered_convs = await ConversationFilter(filter_repo_pick).provider("claude").list()
        if filtered_convs:
            picked = await ConversationFilter(filter_repo_pick).provider("claude").pick()
            assert picked is not None
            assert picked.provider == "claude"

    @pytest.mark.asyncio
    async def test_pick_with_limit(self, filter_repo_pick):
        """pick() respects limit()."""
        picked = await ConversationFilter(filter_repo_pick).limit(1).pick()
        if picked:
            all_first = await ConversationFilter(filter_repo_pick).limit(1).list()
            assert picked.id == all_first[0].id

    @pytest.fixture
    def filter_repo_pick_many(self, tmp_path):
        """Repository with enough conversations to hit picker truncation."""
        db_path = tmp_path / "filter_pick_many.db"
        with open_connection(db_path) as conn:
            rebuild_index(conn)

        for i in range(25):
            (
                ConversationBuilder(db_path, f"conv-many-{i}")
                .provider("claude")
                .title(f"Conversation {i}")
                .created_at(f"2024-01-{i + 1:02d}T00:00:00+00:00")
                .save()
            )

        backend = SQLiteBackend(db_path)
        return ConversationRepository(backend)

    @pytest.mark.asyncio
    async def test_pick_prints_exactly_twenty_options(self, filter_repo_pick_many, capsys):
        """Interactive picker should render 20 rows and an exact remainder summary."""
        prompt = Mock(return_value="1")
        with (
            patch("sys.stdout.isatty", return_value=True),
            patch("builtins.input", prompt),
        ):
            result = await ConversationFilter(filter_repo_pick_many).pick()

        assert result is not None
        prompt.assert_called_once_with("\nSelect number (or Enter for first): ")
        output = capsys.readouterr().out
        numbered_lines = [
            line for line in output.splitlines() if re.match(r"\s+\d+\.\s", line)
        ]
        assert output.startswith("\n25 matching conversations:\n\n")
        assert len(numbered_lines) == 20
        assert "... and 5 more" in output
        assert "21." not in output
        assert numbered_lines[0].lstrip().startswith("1.")
        assert numbered_lines[-1].lstrip().startswith("20.")
        assert "(None)" not in output
        assert re.search(r"\(\d{4}-\d{2}-\d{2}\)", output)

    @pytest.mark.asyncio
    async def test_pick_formats_unknown_dates_and_truncates_titles_exactly(self, filter_repo_pick, capsys):
        """Interactive picker should show truncated titles and the literal unknown date sentinel."""
        filter_obj = ConversationFilter(filter_repo_pick)
        filter_obj.list = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                Mock(
                    provider="claude",
                    display_title="x" * 60,
                    display_date=None,
                )
            ]
        )
        prompt = Mock(return_value="")
        with (
            patch("sys.stdout.isatty", return_value=True),
            patch("builtins.input", prompt),
        ):
            result = await filter_obj.pick()

        assert result is not None
        output = capsys.readouterr().out
        assert "[claude] " + ("x" * 50) + " (unknown)" in output
        assert ("x" * 51) not in output

    @pytest.mark.asyncio
    async def test_pick_returns_expected_selection_indexes(self, filter_repo_pick):
        """Interactive picker should use 1-based input and accept only in-range indexes."""
        expected = await ConversationFilter(filter_repo_pick).sort("date").reverse().list()

        with (
            patch("sys.stdout.isatty", return_value=True),
            patch("builtins.input", return_value="2"),
        ):
            second = await ConversationFilter(filter_repo_pick).sort("date").reverse().pick()

        with (
            patch("sys.stdout.isatty", return_value=True),
            patch("builtins.input", return_value="6"),
        ):
            too_large = await ConversationFilter(filter_repo_pick).sort("date").reverse().pick()

        assert second is not None
        assert second.id == expected[1].id
        assert too_large is None

class TestFiltersNegativeFTSLogic:
    """Test negative FTS terms and has_post_filters."""

    @pytest.fixture
    def filter_repo_fts(self, tmp_path):
        db_path = tmp_path / "filter_fts.db"
        with open_connection(db_path) as conn:
            rebuild_index(conn)

        (ConversationBuilder(db_path, "conv1")
         .provider("claude")
         .add_message("m1", text="error in the system")
         .save())

        (ConversationBuilder(db_path, "conv2")
         .provider("claude")
         .add_message("m1", text="working perfectly")
         .save())

        backend = SQLiteBackend(db_path)
        return ConversationRepository(backend)

    @pytest.mark.asyncio
    async def test_negative_fts_excludes_conversations(self, filter_repo_fts):
        """Test exclude_text() excludes conversations with term."""
        results = await (ConversationFilter(filter_repo_fts)
                   .exclude_text("error")
                   .list())
        assert not any("error" in c.display_title or any("error" in (m.text or "").lower() for m in c.messages) for c in results)


# =============================================================================
# UNIFIED.PY TESTS
# =============================================================================


class TestUnifiedMissingRole:
    """Test _missing_role() error."""

    def test_missing_role_raises_error(self):
        """Test _missing_role() raises ValueError."""
        from polylogue.schemas.unified import _missing_role
        with pytest.raises(ValueError, match="Message has no role"):
            _missing_role()


class TestUnifiedExtractReasoningTraces:
    """Test extract_reasoning_traces with all branches."""

    @pytest.mark.parametrize("content,provider,expected_len,expected_text,description", REASONING_TRACES_CASES)
    def test_extract_reasoning_traces(self, content, provider, expected_len, expected_text, description):
        """Test reasoning trace extraction with various content types."""
        result = extract_reasoning_traces(content, provider)
        assert len(result) == expected_len
        if expected_text is not None:
            assert result[0].text == expected_text


class TestUnifiedExtractContentBlocks:
    """Test extract_content_blocks with all block types."""

    @pytest.mark.parametrize("content,expected_len,expected_types,description", CONTENT_BLOCKS_CASES)
    def test_extract_content_blocks(self, content, expected_len, expected_types, description):
        """Test content block extraction with various block types."""
        result = extract_content_blocks(content)
        assert len(result) == expected_len
        if expected_types:
            for block, expected_type in zip(result, expected_types, strict=True):
                assert block.type.value == expected_type


class TestUnifiedExtractTokenUsage:
    """Test extract_token_usage edge cases."""

    @pytest.mark.parametrize("usage,expected_type,description", TOKEN_USAGE_CASES)
    def test_extract_token_usage(self, usage, expected_type, description):
        """Test token usage extraction with various inputs."""
        result = extract_token_usage(usage)

        if expected_type is None:
            assert result is None
        elif expected_type == "partial":
            assert result.input_tokens == 100


class TestUnifiedExtractTextHelpers:
    """Test text extraction helpers."""

    @pytest.mark.parametrize("extract_fn,content,expected,description", EXTRACT_TEXT_CASES)
    def test_extract_text(self, extract_fn, content, expected, description):
        """Test text extraction from various content structures."""
        result = extract_fn(content)
        assert result == expected or expected in result


class TestUnifiedExtractHarmonizedMessage:
    """Test extract_harmonized_message with all providers."""

    def test_extract_harmonized_message_invalid_provider(self):
        """Test with unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            extract_harmonized_message("unknown_provider", {})

    @pytest.mark.parametrize("provider,msg_key,description", HARMONIZED_MESSAGE_PROVIDER_CASES)
    def test_extract_harmonized_message_by_provider(self, provider, msg_key, description):
        """Test extract_harmonized_message for each provider."""
        if provider == "claude-code":
            raw = {
                "uuid": "msg1",
                "timestamp": "2024-01-01T00:00:00Z",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}],
                    "model": "claude"
                }
            }
        elif provider == "claude-ai":
            raw = {
                "uuid": "msg1",
                "sender": "user",
                "text": "Hello",
                "created_at": "2024-01-01T00:00:00Z"
            }
        elif provider == "chatgpt":
            raw = {
                "id": "msg1",
                "author": {"role": "user"},
                "content": {"parts": ["Hello"]},
                "create_time": 1704067200
            }
        elif provider == "gemini":
            raw = {
                "role": "user",
                "text": "Hello"
            }
        elif provider == "codex":
            raw = {
                "id": "msg1",
                "role": "user",
                "content": [{"text": "Hello"}],
                "timestamp": "2024-01-01T00:00:00Z"
            }

        result = extract_harmonized_message(provider, raw)
        assert isinstance(result, HarmonizedMessage)


class TestUnifiedHarmonizeParsedMessage:
    """Test harmonize_parsed_message edge cases."""

    def test_harmonize_parsed_message_none_meta(self):
        """Test with None provider_meta."""
        result = harmonize_parsed_message("claude", None)
        assert result is None

    def test_harmonize_parsed_message_not_message_record(self):
        """Test with non-message record."""
        result = harmonize_parsed_message("claude-code", {"type": "metadata"})
        assert result is None

    def test_harmonize_parsed_message_valid(self):
        """Test valid harmonization."""
        meta = {
            "raw": {
                "uuid": "msg1",
                "sender": "user",
                "text": "Hello"
            }
        }
        result = harmonize_parsed_message("claude-ai", meta)
        assert isinstance(result, HarmonizedMessage)

    def test_harmonize_parsed_message_claude_code_raw_preserves_tool_fields(self):
        """Claude Code raw format should extract tool id/input/category correctly."""
        raw_record = {
            "type": "assistant",
            "uuid": "msg-1",
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Need to read file"},
                    {"type": "thinking", "thinking": "First inspect the repo"},
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "Read",
                        "input": {"file_path": "README.md"},
                    },
                ],
            },
        }
        result = harmonize_parsed_message("claude-code", {"raw": raw_record})
        assert result is not None
        assert result.id == "msg-1"
        assert result.role == "assistant"
        assert result.text == "Need to read file"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "tool-1"
        assert result.tool_calls[0].input == {"file_path": "README.md"}
        assert result.tool_calls[0].category.value == "file_read"
        assert len(result.reasoning_traces) == 1
        assert result.reasoning_traces[0].text == "First inspect the repo"

    def test_extract_from_provider_meta_claude_code_raw_extracts_text(self):
        """Raw format extraction should produce text from text-only blocks."""
        raw_record = {
            "type": "assistant",
            "uuid": "msg-2",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "line one"},
                    {"type": "text", "text": "line two"},
                    {"type": "thinking", "thinking": "some reasoning"},
                ],
            },
        }
        result = extract_from_provider_meta("claude-code", {"raw": raw_record})
        assert result.text == "line one\nline two"
        assert result.role == "assistant"
        assert len(result.reasoning_traces) == 1


class TestUnifiedBulkHarmonize:
    """Test bulk_harmonize edge cases."""

    def test_bulk_harmonize_no_provider_meta(self):
        """Test with parsed messages without provider_meta."""
        class MockParsedMessage:
            pass

        messages = [MockParsedMessage()]
        result = bulk_harmonize("claude", messages)
        assert result == []

    def test_bulk_harmonize_mixed_valid_invalid(self):
        """Test with mix of valid and invalid records."""
        class MockParsedMessage:
            def __init__(self, meta=None):
                self.provider_meta = meta

        messages = [
            MockParsedMessage({"raw": {"type": "metadata"}}),
            MockParsedMessage({"raw": {"type": "user", "uuid": "1", "message": {"role": "user", "content": [{"type": "text", "text": "Hi"}]}}}),
        ]
        result = bulk_harmonize("claude-code", messages)
        assert len(result) == 1


class TestUnifiedIsMessageRecord:
    """Test is_message_record for Claude Code type checking."""

    @pytest.mark.parametrize("provider,record_type,expected,description", MESSAGE_RECORD_TYPE_CASES)
    def test_is_message_record(self, provider, record_type, expected, description):
        """Test message record type checking for various providers."""
        result = is_message_record(provider, {"type": record_type})
        assert result == expected


# =============================================================================
# VALIDATOR.PY TESTS
# =============================================================================


class TestValidatorImportErrorHandling:
    """Test jsonschema ImportError handling."""

    def test_validator_jsonschema_not_installed(self):
        """Test SchemaValidator when jsonschema is not available."""
        with patch("polylogue.schemas.validator.jsonschema", None):
            with pytest.raises(ImportError, match="jsonschema not installed"):
                SchemaValidator({})


class TestValidatorAvailableProviders:
    """Test available_providers() method."""

    def test_available_providers_missing_schema_dir(self):
        """Test when SCHEMA_DIR doesn't exist."""
        with patch("polylogue.schemas.validator.SchemaRegistry.list_providers", return_value=[]):
            result = SchemaValidator.available_providers()
            assert result == []


class TestValidatorDetectDrift:
    """Test drift detection with all branches."""

    def test_validate_detects_unexpected_field(self):
        """Test detecting unexpected fields."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False
        }
        validator = SchemaValidator(schema, strict=True)
        result = validator.validate({"name": "test", "extra": "field"})
        assert result.has_drift
        assert any("Unexpected field" in w for w in result.drift_warnings)

    def test_validate_additional_properties_true(self):
        """Test schema with additionalProperties: true."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": True
        }
        validator = SchemaValidator(schema, strict=True)
        result = validator.validate({"name": "test", "extra": "field"})
        assert not result.has_drift

    def test_validate_additional_properties_schema(self):
        """Test additionalProperties with schema dict."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": {"type": "string"}
        }
        validator = SchemaValidator(schema, strict=True)
        validator.validate({"name": "test", "extra": "value"})

    def test_validate_nested_object_drift(self):
        """Test nested object drift detection."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}}
                }
            }
        }
        validator = SchemaValidator(schema, strict=True)
        data = {"user": {"name": "test", "extra": "field"}}
        result = validator.validate(data)
        assert isinstance(result, ValidationResult)

    def test_validate_list_items_drift(self):
        """Test array items drift detection."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}}
                    }
                }
            }
        }
        validator = SchemaValidator(schema, strict=True)
        data = {"items": [{"id": 1, "extra": "field"}]}
        result = validator.validate(data)
        assert isinstance(result, ValidationResult)


class TestValidatorFormatError:
    """Test _format_error method."""

    def test_validate_multiple_errors(self):
        """Test validation with multiple errors."""
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        validator = SchemaValidator(schema, strict=False)
        result = validator.validate({"age": "not an integer"})
        assert len(result.errors) >= 1


class TestValidatorConvenienceFunction:
    """Test validate_provider_export convenience function."""

    def test_validate_provider_export_raises_on_missing_schema(self):
        """Test that invalid provider raises error."""
        with pytest.raises(FileNotFoundError):
            validate_provider_export({}, "invalid_provider", strict=True)


class TestValidationResult:
    """Test ValidationResult class methods."""

    def test_validation_result_has_drift_property(self):
        """Test has_drift property."""
        result = ValidationResult(
            is_valid=True,
            drift_warnings=["Field X is new"]
        )
        assert result.has_drift is True

    def test_validation_result_no_drift(self):
        """Test has_drift when empty."""
        result = ValidationResult(is_valid=True)
        assert result.has_drift is False

    def test_validation_result_raise_if_invalid(self):
        """Test raise_if_invalid method."""
        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"]
        )
        with pytest.raises(ValueError, match="Schema validation failed"):
            result.raise_if_invalid()

    def test_validation_result_raise_if_valid(self):
        """Test raise_if_invalid when valid."""
        result = ValidationResult(is_valid=True)
        result.raise_if_invalid()
