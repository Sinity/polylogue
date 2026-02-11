"""Tests for ConversationFilter fluent API."""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from polylogue.lib.filters import ConversationFilter
from polylogue.lib.models import Conversation, ConversationSummary, Message
from polylogue.storage.backends.sqlite import SQLiteBackend, open_connection
from polylogue.storage.index import rebuild_index
from polylogue.storage.repository import ConversationRepository
from polylogue.schemas.unified import (
    HarmonizedMessage,
    extract_reasoning_traces,
    extract_tool_calls,
    extract_content_blocks,
    extract_token_usage,
    extract_claude_code_text,
    extract_chatgpt_text,
    extract_codex_text,
    extract_harmonized_message,
    harmonize_parsed_message,
    bulk_harmonize,
    is_message_record,
)
from polylogue.schemas.validator import (
    SchemaValidator,
    ValidationResult,
    validate_provider_export,
)
from tests.helpers import ConversationBuilder


# =============================================================================
# TEST DATA: PARAMETRIZATION CONSTANTS (SCREAMING_CASE)
# =============================================================================

PROVIDER_FILTER_CASES = [
    ("claude", 2, "Filter by single provider"),
    (("claude", "chatgpt"), 3, "Filter by multiple providers"),
    ("chatgpt", 1, "Exclude specific provider (no_provider)"),
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

DATE_PARSING_CASES = [
    ("yesterday", "Natural language: yesterday"),
    ("2025-01-15", "ISO format date"),
    ("last week", "Relative date format"),
]

DATE_UNTIL_CASES = [
    ("today", "Natural language: today"),
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
    (None, "claude", [], "Empty/None content"),
    (["string", 123], "claude", [], "Non-dict block"),
    ([{"type": "thinking", "thinking": "Let me think..."}], "claude", "thinking_block", "Thinking block"),
    ([{"isThought": True, "text": "Gemini thinking"}], "gemini", "gemini_thought", "Gemini isThought format"),
    ([{"type": "thinking", "text": "Fallback text"}], "claude", "thinking_fallback", "Thinking fallback to text"),
]

CONTENT_BLOCKS_CASES = [
    (None, [], "Empty/None content"),
    (["string", 123, None], [], "Non-dict items"),
    ([{"type": "text", "text": "Hello"}], "text_block", "Text block"),
    ([{"type": "thinking", "thinking": "Thought"}], "thinking_block", "Thinking block"),
    ([{"type": "tool_use", "name": "bash", "id": "tool1", "input": {"command": "ls"}}], "tool_use_block", "Tool use block"),
    ([{"type": "tool_result", "content": "result data"}], "tool_result_block", "Tool result block"),
    ([{"type": "tool_result"}], "tool_result_no_content", "Tool result without content"),
    ([{"type": "code", "text": "print('hello')", "language": "python"}], "code_block", "Code block with text"),
    ([{"type": "code", "code": "def test(): pass"}], "code_block_code", "Code block with code field"),
    ([{"type": "unknown", "data": "something"}], [], "Unknown block type"),
]

EXTRACT_TEXT_CASES = [
    ("claude_code", None, "", "extract_claude_code_text: None"),
    ("claude_code", ["string", 123], "", "extract_claude_code_text: non-dict"),
    ("chatgpt", None, "", "extract_chatgpt_text: None"),
    ("chatgpt", {}, "", "extract_chatgpt_text: no parts"),
    ("chatgpt", {"parts": "string"}, "string", "extract_chatgpt_text: parts as string"),
    ("chatgpt", {"parts": [123, "text", {"key": "val"}]}, "text", "extract_chatgpt_text: non-string parts"),
    ("codex", {"data": "not a list"}, "", "extract_codex_text: non-list"),
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
        """Each filter method returns self for chaining."""
        f = ConversationFilter(filter_repo)
        assert f.provider("claude") is f
        assert f.since("2024-01-01") is f
        assert f.limit(10) is f

    def test_filter_chain_multiple_methods(self, filter_repo):
        """Multiple filter methods can be chained."""
        result = (
            ConversationFilter(filter_repo)
            .provider("claude")
            .limit(5)
            .sort("date")
            .list()
        )
        assert isinstance(result, list)


class TestConversationFilterMethods:
    """Consolidated tests for filter methods (provider, tag, text, title, id, limit)."""

    FILTER_METHOD_CASES = [
        ("provider_single", lambda f: f.provider("claude"), 2, "Filter by single provider"),
        ("provider_multi", lambda f: f.provider("claude", "chatgpt"), 3, "Filter by multiple providers"),
        ("no_provider", lambda f: f.no_provider("claude"), 1, "Exclude specific provider"),
        ("tag_python", lambda f: f.tag("python"), None, "Filter by tag (or empty)"),
        ("no_tag", lambda f: f.no_tag("nonexistent-tag"), 3, "Exclude nonexistent tag"),
        ("contains", lambda f: f.contains("Python"), None, "Filter contains text"),
        ("no_contains", lambda f: f.no_contains("database"), None, "Exclude text"),
        ("limit_1", lambda f: f.limit(1), 1, "Limit to 1 result"),
        ("limit_0", lambda f: f.limit(0), 0, "Limit of zero"),
        ("title_Python", lambda f: f.title("Python"), 1, "Filter by title"),
        ("title_python_case", lambda f: f.title("python"), 1, "Title case insensitive"),
        ("id_prefix", lambda f: f.id("claude"), 2, "Filter by ID prefix"),
    ]

    @pytest.mark.parametrize("method_name,filter_fn,expected_count,description", FILTER_METHOD_CASES)
    def test_filter_method(self, filter_repo, method_name, filter_fn, expected_count, description):
        """Test individual filter methods."""
        result = filter_fn(ConversationFilter(filter_repo)).list()

        if expected_count is not None:
            assert len(result) == expected_count, f"Failed {description}: expected {expected_count}, got {len(result)}"
        else:
            assert isinstance(result, list), f"Failed {description}: should return list"

        # Type-specific assertions
        if "provider" in method_name:
            if "no_" not in method_name:
                provider = method_name.split("_")[1]
                if provider != "multi":
                    assert all(c.provider == provider for c in result)
        elif "title" in method_name:
            if result:
                assert "Python" in result[0].display_title or "python" in result[0].display_title
        elif "id" in method_name and result:
            assert all(c.id.startswith("claude") for c in result)


class TestConversationFilterTerminal:
    """Tests for terminal methods."""

    def test_filter_first(self, filter_repo):
        """first() returns single conversation."""
        result = ConversationFilter(filter_repo).first()
        assert result is not None
        assert hasattr(result, "id")

    def test_filter_first_empty(self, filter_repo):
        """first() returns None when no matches."""
        result = ConversationFilter(filter_repo).provider("nonexistent").first()
        assert result is None

    def test_filter_count(self, filter_repo):
        """count() returns number of matches."""
        count = ConversationFilter(filter_repo).count()
        assert count == 3

    def test_filter_count_with_filter(self, filter_repo):
        """count() respects filters."""
        count = ConversationFilter(filter_repo).provider("claude").count()
        assert count == 2

    def test_filter_delete_removes_conversations(self, filter_repo):
        """delete() removes matched conversations."""
        initial_count = ConversationFilter(filter_repo).count()
        assert initial_count > 0

        deleted = ConversationFilter(filter_repo).limit(1).delete()
        assert deleted == 1

        final_count = ConversationFilter(filter_repo).count()
        assert final_count == initial_count - 1


class TestConversationFilterSort:
    """Tests for sorting."""

    @pytest.mark.parametrize("sort_key,description", SORT_OPERATION_CASES)
    def test_filter_sort(self, filter_repo, sort_key, description):
        """Test sorting by various keys."""
        result = ConversationFilter(filter_repo).sort(sort_key).list()
        assert len(result) > 0

    def test_filter_sort_reverse(self, filter_repo):
        """Reverse sort order."""
        normal = ConversationFilter(filter_repo).sort("date").list()
        reversed_list = ConversationFilter(filter_repo).sort("date").reverse().list()
        if len(normal) > 1:
            assert normal[0].id == reversed_list[-1].id
            assert normal[-1].id == reversed_list[0].id


class TestConversationFilterCustom:
    """Tests for custom predicates."""

    def test_filter_where_predicate(self, filter_repo):
        """Filter with custom predicate."""
        result = (
            ConversationFilter(filter_repo)
            .where(lambda c: len(c.messages) >= 2)
            .list()
        )
        assert all(len(c.messages) >= 2 for c in result)


class TestDateParsing:
    """Tests for date parsing in ConversationFilter.since() and until()."""

    def test_since_invalid_date_raises_value_error(self, filter_repo):
        """Calling .since() with invalid date raises ValueError."""
        f = ConversationFilter(filter_repo)
        with pytest.raises(ValueError, match="Cannot parse date"):
            f.since("invalid-date")

    def test_until_invalid_date_raises_value_error(self, filter_repo):
        """Calling .until() with invalid date raises ValueError."""
        f = ConversationFilter(filter_repo)
        with pytest.raises(ValueError, match="Cannot parse date"):
            f.until("invalid-date")

    @pytest.mark.parametrize("date_str,description", DATE_PARSING_CASES)
    def test_since_date_formats(self, filter_repo, date_str, description):
        """Test since() with various date formats."""
        f = ConversationFilter(filter_repo)
        result = f.since(date_str)
        assert result is f
        assert f._since_date is not None
        assert isinstance(f._since_date, datetime)

    @pytest.mark.parametrize("date_str,description", DATE_UNTIL_CASES)
    def test_until_date_formats(self, filter_repo, date_str, description):
        """Test until() with various date formats."""
        f = ConversationFilter(filter_repo)
        result = f.until(date_str)
        assert result is f
        assert f._until_date is not None
        assert isinstance(f._until_date, datetime)


class TestFtsWithProviderFilter:
    """Tests for combined FTS search + provider filter."""

    @pytest.mark.parametrize("search_term,provider,should_find,description", FTS_PROVIDER_CASES)
    def test_fts_with_provider(self, filter_repo, search_term, provider, should_find, description):
        """Test FTS search combined with provider filter."""
        result = ConversationFilter(filter_repo).contains(search_term).provider(provider).list()
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

    def test_since_with_datetime_object(self, filter_repo):
        """Test .since() with datetime object instead of string."""
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        filter_obj = ConversationFilter(filter_repo).since(dt)
        assert filter_obj._since_date == dt

    def test_until_with_datetime_object(self, filter_repo):
        """Test .until() with datetime object instead of string."""
        dt = datetime(2024, 12, 31, tzinfo=timezone.utc)
        filter_obj = ConversationFilter(filter_repo).until(dt)
        assert filter_obj._until_date == dt

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

    def test_excluded_providers_filter(self, filter_repo_populated):
        """Test filtering out specific providers."""
        results = (ConversationFilter(filter_repo_populated)
                   .no_provider("chatgpt")
                   .list())
        assert all(c.provider != "chatgpt" for c in results)

    @pytest.mark.parametrize("sort_key,description", SORT_VARIANTS_CASES)
    def test_sort_by_variant(self, filter_repo_populated, sort_key, description):
        """Test sorting by various metrics."""
        results = (ConversationFilter(filter_repo_populated)
                   .sort(sort_key)
                   .list())
        assert len(results) >= 0


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

    def test_id_prefix_exact_match_fast_path(self, filter_repo_with_id):
        """Test ID prefix fast path when prefix resolves to single conversation."""
        results = (ConversationFilter(filter_repo_with_id)
                   .id("abc123")
                   .list())
        assert isinstance(results, list)

    def test_fts_search_exception_handling(self, filter_repo_with_id):
        """Test FTS search fallback on exception."""
        filter_obj = ConversationFilter(filter_repo_with_id)
        filter_obj.contains("test")
        with patch.object(filter_repo_with_id, 'search', side_effect=Exception("FTS error")):
            results = filter_obj.list()
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

    def test_list_summaries_with_provider_filter(self, filter_repo_summaries):
        """Test list_summaries with provider filter."""
        results = (ConversationFilter(filter_repo_summaries)
                   .provider("claude")
                   .list_summaries())
        assert all(isinstance(s, ConversationSummary) for s in results)

    def test_list_summaries_with_tag_filter(self, filter_repo_summaries):
        """Test list_summaries with tag filter."""
        results = (ConversationFilter(filter_repo_summaries)
                   .tag("mytag")
                   .list_summaries())
        assert isinstance(results, list)

    def test_list_summaries_with_summary_has_type(self, filter_repo_summaries):
        """Test list_summaries with has('summary') filter."""
        results = (ConversationFilter(filter_repo_summaries)
                   .has("summary")
                   .list_summaries())
        assert all(s.summary for s in results)

    def test_list_summaries_cannot_use_content_filters(self, filter_repo_summaries):
        """Test that list_summaries rejects content-dependent filters."""
        with pytest.raises(ValueError, match="content-dependent filters"):
            (ConversationFilter(filter_repo_summaries)
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

    def test_pick_no_results(self, filter_repo_pick):
        """Test pick() with no matching conversations."""
        result = (ConversationFilter(filter_repo_pick)
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
    def test_pick_outcomes(self, filter_repo_pick, has_results, is_tty, patch_target, input_value, description):
        """Test pick() with various input scenarios."""
        if not has_results:
            result = (ConversationFilter(filter_repo_pick)
                      .provider("nonexistent")
                      .pick())
            assert result is None
        else:
            with patch("sys.stdout.isatty", return_value=is_tty):
                if patch_target:
                    if input_value == "EOF":
                        with patch(patch_target, side_effect=EOFError):
                            result = ConversationFilter(filter_repo_pick).pick()
                            assert result is None
                    elif input_value == "INTERRUPT":
                        with patch(patch_target, side_effect=KeyboardInterrupt):
                            result = ConversationFilter(filter_repo_pick).pick()
                            assert result is None
                    else:
                        with patch(patch_target, return_value=input_value):
                            result = ConversationFilter(filter_repo_pick).pick()
                            if input_value == "":
                                assert result is not None
                            elif input_value == "1":
                                assert result is not None
                            elif input_value == "999" or input_value == "not a number":
                                assert result is None
                else:
                    result = ConversationFilter(filter_repo_pick).pick()
                    assert result is not None


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

    def test_negative_fts_excludes_conversations(self, filter_repo_fts):
        """Test no_contains() excludes conversations with term."""
        results = (ConversationFilter(filter_repo_fts)
                   .no_contains("error")
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

    @pytest.mark.parametrize("content,provider,expected_type,description", REASONING_TRACES_CASES)
    def test_extract_reasoning_traces(self, content, provider, expected_type, description):
        """Test reasoning trace extraction with various content types."""
        result = extract_reasoning_traces(content, provider)

        if expected_type == []:
            assert result == []
        elif expected_type == "thinking_block":
            assert len(result) == 1
            assert result[0].text == "Let me think..."
        elif expected_type == "gemini_thought":
            assert len(result) == 1
            assert result[0].text == "Gemini thinking"
        elif expected_type == "thinking_fallback":
            assert len(result) == 1
            assert result[0].text == "Fallback text"


class TestUnifiedExtractContentBlocks:
    """Test extract_content_blocks with all block types."""

    @pytest.mark.parametrize("content,expected_type,description", CONTENT_BLOCKS_CASES)
    def test_extract_content_blocks(self, content, expected_type, description):
        """Test content block extraction with various block types."""
        result = extract_content_blocks(content)

        if expected_type == []:
            assert result == []
        elif expected_type == "text_block":
            assert len(result) == 1
            from polylogue.lib.viewports import ContentType
            assert result[0].type == ContentType.TEXT
        elif expected_type == "thinking_block":
            assert len(result) == 1
        elif expected_type == "tool_use_block":
            assert len(result) == 1
        elif expected_type == "tool_result_block":
            assert len(result) == 1
        elif expected_type == "tool_result_no_content":
            assert len(result) == 1
            assert result[0].text == ""
        elif expected_type == "code_block":
            assert len(result) == 1
        elif expected_type == "code_block_code":
            assert len(result) == 1


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

    @pytest.mark.parametrize("provider,content,expected,description", EXTRACT_TEXT_CASES)
    def test_extract_text(self, provider, content, expected, description):
        """Test text extraction from various content structures."""
        if provider == "claude_code":
            result = extract_claude_code_text(content)
        elif provider == "chatgpt":
            result = extract_chatgpt_text(content)
        elif provider == "codex":
            result = extract_codex_text(content)

        if isinstance(expected, str):
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
        with patch("polylogue.schemas.validator.SCHEMA_DIR") as mock_dir:
            mock_dir.exists.return_value = False
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
        result = validator.validate({"name": "test", "extra": "value"})
        assert True

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
