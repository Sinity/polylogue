"""Tests for ConversationFilter fluent API."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.lib.filters import ConversationFilter
from polylogue.lib.models import ConversationSummary
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
from tests.infra.storage_records import ConversationBuilder

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
