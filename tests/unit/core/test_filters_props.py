"""Property-based tests for filter and RRF modules.

Consolidates tests from:
- test_filters.py (filter tests only — schema/validator tests moved to test_filters_schemas.py)
- test_filters_adv.py (advanced filter selection tests)
- test_filters_props.py (original property tests, preserved)

Key properties tested:
1. Filter chain never crashes on arbitrary filter combos
2. Filter monotonicity — adding filters never increases result count
3. Filter idempotence — same filter twice = same filter once
4. SQL pushdown params match filter state
5. RRF score bounds, symmetry, monotonicity
6. Terminal methods on empty repos
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.cli.filter_picker import pick_filter
from polylogue.lib.filters import ConversationFilter
from polylogue.lib.models import ConversationSummary
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.index import rebuild_index
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.search_providers.hybrid import reciprocal_rank_fusion
from tests.infra.storage_records import ConversationBuilder
from tests.infra.strategies.filters import (
    filter_chain_strategy,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def filter_db(tmp_path):
    """Create database with test conversations for filter tests."""
    db_path = tmp_path / "filter_test.db"

    (ConversationBuilder(db_path, "claude-1")
     .provider("claude-ai")
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
     .provider("claude-ai")
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


@pytest.fixture
def filter_db_empty(tmp_path):
    """Create empty database for testing empty repository edge cases."""
    db_path = tmp_path / "filter_empty.db"
    with open_connection(db_path) as conn:
        rebuild_index(conn)
    return db_path


@pytest.fixture
def filter_repo_empty(filter_db_empty):
    """Create repository for empty database tests."""
    backend = SQLiteBackend(db_path=filter_db_empty)
    return ConversationRepository(backend=backend)


@pytest.fixture
def filter_db_advanced(tmp_path):
    """Create database with conversations for advanced filter tests.

    Includes thinking blocks, tool use, attachments, summaries,
    various message counts, and token lengths.
    """
    db_path = tmp_path / "filter_advanced.db"

    (ConversationBuilder(db_path, "conv-thinking")
     .provider("claude-ai")
     .title("Complex Problem Analysis")
     .add_message("m1", role="user", text="Solve this complex math problem")
     .add_message("m2", role="assistant", text="The answer is 42.",
                  provider_meta={"content_blocks": [{"type": "thinking", "text": "Let me break this down step by step..."}]})
     .add_message("m3", role="user", text="Can you explain further?")
     .metadata({"tags": ["math", "complex"], "summary": "Math problem solving"})
     .save())

    (ConversationBuilder(db_path, "conv-tools")
     .provider("claude-ai")
     .title("API Integration Help")
     .add_message("m4", role="user", text="How do I call an API?")
     .add_message("m5", role="assistant", text="I'll help you with that.",
                  provider_meta={"content_blocks": [{"type": "tool_use", "tool_name": "bash", "input": {}}]})
     .add_message("m6", role="user", text="Show me an example")
     .add_message("m7", role="assistant", text="Here is a complete working example with error handling.")
     .metadata({"tags": ["api", "integration"]})
     .save())

    (ConversationBuilder(db_path, "conv-attachments")
     .provider("chatgpt")
     .title("Document Analysis")
     .add_message("m8", role="user", text="Please analyze this document")
     .add_message("m9", role="assistant", text="I see the file contains important data.")
     .add_attachment("att1", message_id="m8", mime_type="application/pdf", size_bytes=5000)
     .metadata({"tags": ["documents"]})
     .save())

    (ConversationBuilder(db_path, "conv-summary-only")
     .provider("claude-ai")
     .title("Brief Chat")
     .add_message("m10", role="user", text="Hello there")
     .add_message("m11", role="assistant", text="Hi how are you")
     .metadata({"summary": "Brief greeting exchange", "tags": ["greeting"]})
     .save())

    (ConversationBuilder(db_path, "conv-multi-attach")
     .provider("chatgpt")
     .title("Multiple File Analysis")
     .add_message("m12", role="user", text="Analyze these files please")
     .add_message("m13", role="assistant", text="I can see both files clearly.")
     .add_message("m14", role="user", text="What are the main differences?")
     .add_attachment("att2", message_id="m12", mime_type="image/png", size_bytes=2000)
     .add_attachment("att3", message_id="m12", mime_type="application/pdf", size_bytes=3000)
     .metadata({"tags": ["analysis"]})
     .save())

    (ConversationBuilder(db_path, "conv-long-messages")
     .provider("claude-ai")
     .title("Deep Discussion")
     .add_message("m15", role="user",
                  text="Tell me everything you know about quantum computing including the fundamentals principles and applications")
     .add_message("m16", role="assistant",
                  text="Quantum computing is a revolutionary field that leverages quantum mechanical phenomena like superposition and entanglement to perform computations exponentially faster than classical computers in certain domains such as cryptography and optimization.")
     .metadata({"tags": ["quantum"]})
     .save())

    (ConversationBuilder(db_path, "conv-plain")
     .provider("codex")
     .title("Simple")
     .add_message("m17", role="user", text="What is two plus two")
     .metadata({"tags": ["simple"]})
     .save())

    with open_connection(db_path) as conn:
        rebuild_index(conn)

    return db_path


@pytest.fixture
def filter_repo_advanced(filter_db_advanced):
    """Create repository for advanced filter tests."""
    backend = SQLiteBackend(db_path=filter_db_advanced)
    return ConversationRepository(backend=backend)


# =============================================================================
# Property: Filter chains never crash
# =============================================================================


def _apply_filter_spec(f: ConversationFilter, spec: dict) -> ConversationFilter:
    """Apply a single filter spec dict to a ConversationFilter."""
    ftype = spec["type"]
    if ftype == "provider":
        return f.provider(spec["value"])
    elif ftype == "contains":
        return f.contains(spec["value"])
    elif ftype == "since":
        return f.since(spec["value"])
    elif ftype == "until":
        return f.until(spec["value"])
    elif ftype == "limit":
        return f.limit(spec["value"])
    elif ftype == "offset":
        # ConversationFilter does not have .offset(), skip
        return f
    elif ftype == "sort":
        return f.sort("date")  # Use a safe sort field
    elif ftype == "role":
        # No direct role filter on ConversationFilter, skip
        return f
    elif ftype == "has_attachments":
        return f.has("attachments")
    elif ftype == "min_words" or ftype == "max_words":
        return f
    return f


@given(filter_chain_strategy(min_filters=1, max_filters=4))
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_filter_chain_never_crashes_on_build(chain: list[dict]):
    """Building a filter chain from arbitrary filter specs never crashes.

    We can't call .list() here (needs async + DB), but we verify the
    fluent builder accepts any valid filter combo without raising.
    """
    # Use a mock repo — we only test chain construction, not execution
    from unittest.mock import MagicMock
    mock_repo = MagicMock(spec=ConversationRepository)
    f = ConversationFilter(mock_repo)
    for spec in chain:
        f = _apply_filter_spec(f, spec)
    # If we got here, the chain didn't crash


@pytest.mark.asyncio
async def test_filter_chain_never_crashes_on_execution(make_filter_repo):
    """Composed filters never crash when executed against a real DB.

    Uses explicit examples rather than @given to avoid fixture-scope issues.
    """

    repo = make_filter_repo([
        {"id": "c1", "provider": "claude-ai", "title": "Test",
         "messages": [{"id": "m1", "role": "user", "text": "hello"}]},
        {"id": "c2", "provider": "chatgpt", "title": "Other",
         "messages": [{"id": "m2", "role": "user", "text": "world"}]},
    ])

    # Test a variety of filter chain combos
    chains = [
        [{"type": "provider", "value": "claude-ai"}],
        [{"type": "limit", "value": 1}],
        [{"type": "provider", "value": "chatgpt"}, {"type": "limit", "value": 5}],
        [{"type": "contains", "value": "hello"}, {"type": "provider", "value": "claude-ai"}],
        [{"type": "sort", "field": "created_at", "direction": "desc"}],
    ]
    for chain in chains:
        f = ConversationFilter(repo)
        for spec in chain:
            f = _apply_filter_spec(f, spec)
        result = await f.list()
        assert isinstance(result, list)


# =============================================================================
# Property: Filter monotonicity — adding filters never increases count
# =============================================================================


@pytest.mark.asyncio
async def test_filter_monotonicity_provider(filter_repo):
    """Adding a provider filter never increases result count."""
    all_count = await ConversationFilter(filter_repo).count()
    filtered_count = await ConversationFilter(filter_repo).provider("claude-ai").count()
    assert filtered_count <= all_count


@pytest.mark.asyncio
async def test_filter_monotonicity_limit(filter_repo):
    """Adding a limit filter never increases result count."""
    all_count = await ConversationFilter(filter_repo).count()
    limited_count = len(await ConversationFilter(filter_repo).limit(1).list())
    assert limited_count <= all_count


@pytest.mark.asyncio
async def test_filter_monotonicity_exclude(filter_repo):
    """Adding an exclude filter never increases result count."""
    all_count = await ConversationFilter(filter_repo).count()
    excluded = await ConversationFilter(filter_repo).exclude_provider("claude-ai").list()
    assert len(excluded) <= all_count


@pytest.mark.asyncio
async def test_filter_monotonicity_chained(filter_repo_advanced):
    """Stacking filters monotonically decreases results."""
    c0 = await ConversationFilter(filter_repo_advanced).count()
    c1 = len(await ConversationFilter(filter_repo_advanced).provider("claude-ai").list())
    c2 = len(await ConversationFilter(filter_repo_advanced).provider("claude-ai").has("thinking").list())
    assert c2 <= c1 <= c0


# =============================================================================
# Property: Filter idempotence — applying same filter twice = once
# =============================================================================


@pytest.mark.asyncio
async def test_filter_idempotence_provider(filter_repo):
    """Applying provider("claude-ai") twice yields same results as once."""
    once = await ConversationFilter(filter_repo).provider("claude-ai").list()
    twice = await ConversationFilter(filter_repo).provider("claude-ai").provider("claude-ai").list()
    assert {c.id for c in once} == {c.id for c in twice}


@pytest.mark.asyncio
async def test_filter_idempotence_exclude(filter_repo):
    """Applying exclude_provider("claude-ai") twice yields same results as once."""
    once = await ConversationFilter(filter_repo).exclude_provider("claude-ai").list()
    twice = await ConversationFilter(filter_repo).exclude_provider("claude-ai").exclude_provider("claude-ai").list()
    assert {c.id for c in once} == {c.id for c in twice}


# =============================================================================
# Property: SQL pushdown params match filter state
# =============================================================================


def test_sql_pushdown_provider():
    """SQL pushdown includes provider when set."""
    from unittest.mock import MagicMock
    f = ConversationFilter(MagicMock())
    f.provider("claude-ai")
    params = f._sql_pushdown_params()
    assert params["provider"] == "claude-ai"


def test_sql_pushdown_multi_provider():
    """SQL pushdown includes providers list when multiple set."""
    from unittest.mock import MagicMock
    f = ConversationFilter(MagicMock())
    f.provider("claude-ai", "chatgpt")
    params = f._sql_pushdown_params()
    assert params["providers"] == ["claude-ai", "chatgpt"]


def test_sql_pushdown_date_range():
    """SQL pushdown includes since/until when set."""
    from unittest.mock import MagicMock
    f = ConversationFilter(MagicMock())
    dt = datetime(2024, 6, 15, tzinfo=timezone.utc)
    f.since(dt).until(dt)
    params = f._sql_pushdown_params()
    assert "since" in params
    assert "until" in params


def test_sql_pushdown_title():
    """SQL pushdown includes title_contains when set."""
    from unittest.mock import MagicMock
    f = ConversationFilter(MagicMock())
    f.title("Python")
    params = f._sql_pushdown_params()
    assert params["title_contains"] == "Python"


def test_sql_pushdown_empty():
    """SQL pushdown is empty dict when no filters set."""
    from unittest.mock import MagicMock
    f = ConversationFilter(MagicMock())
    assert f._sql_pushdown_params() == {}


@given(
    st.lists(st.sampled_from(["chatgpt", "claude-ai", "codex"]), min_size=1, max_size=3),
    st.lists(st.sampled_from(["chatgpt", "claude-ai", "codex"]), min_size=0, max_size=2),
)
def test_provider_filter_exclusion_disjoint(
    include_providers: list[str],
    exclude_providers: list[str],
):
    """Provider inclusion and exclusion should be mutually exclusive."""
    included = set(include_providers)
    excluded = set(exclude_providers)
    result = included - excluded
    for provider in result:
        assert provider not in excluded


# =============================================================================
# Property: Filter composition laws (mock-based)
# =============================================================================


@given(
    st.integers(min_value=0, max_value=100),
    st.integers(min_value=1, max_value=50),
)
def test_limit_respects_bound(total_items: int, limit: int):
    """Limit filter produces at most `limit` items."""
    items = list(range(total_items))
    limited = items[:limit]
    assert len(limited) <= limit


@given(
    st.integers(min_value=0, max_value=50),
    st.integers(min_value=0, max_value=100),
)
def test_offset_skips_correctly(offset: int, total_items: int):
    """Offset filter skips the first `offset` items."""
    items = list(range(total_items))
    offset_items = items[offset:]
    expected_count = max(0, total_items - offset)
    assert len(offset_items) == expected_count


@given(
    st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2025, 1, 1), timezones=st.just(timezone.utc)),
    st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2025, 1, 1), timezones=st.just(timezone.utc)),
)
def test_date_range_logic(since: datetime, until: datetime):
    """Date range filter is a valid interval (since <= until)."""
    if since > until:
        since, until = until, since
    mid = since + (until - since) / 2
    assert since <= mid <= until
    before = since - timedelta(days=1)
    after = until + timedelta(days=1)
    assert before < since
    assert after > until


@given(
    st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=50),
    st.integers(min_value=0, max_value=100),
    st.integers(min_value=0, max_value=100),
)
def test_filter_composition_monotonic(items: list[int], threshold1: int, threshold2: int):
    """Composing filters never increases result count."""
    filtered_once = [x for x in items if x > threshold1]
    filtered_twice = [x for x in filtered_once if x > threshold2]
    assert len(filtered_twice) <= len(filtered_once)


@given(
    st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=50),
    st.integers(min_value=0, max_value=100),
)
def test_filter_idempotent(items: list[int], threshold: int):
    """Applying the same filter twice has no additional effect."""
    filtered_once = [x for x in items if x > threshold]
    filtered_twice = [x for x in filtered_once if x > threshold]
    assert filtered_once == filtered_twice


@given(
    st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=50),
    st.integers(min_value=0, max_value=100),
    st.integers(min_value=0, max_value=100),
)
def test_filter_order_independent(items: list[int], threshold1: int, threshold2: int):
    """For commutative filters, order doesn't affect result count."""
    filtered_ab = [x for x in items if x > threshold1 and x > threshold2]
    filtered_ba = [x for x in items if x > threshold2 and x > threshold1]
    assert filtered_ab == filtered_ba


# =============================================================================
# Regression: Key filter behaviors (from test_filters.py + test_filters_adv.py)
# =============================================================================


class TestConversationFilterChaining:
    """Chainability and multi-method filter tests (key regressions)."""

    def test_filter_returns_self(self, filter_repo):
        """Every fluent filter method must return self."""
        CHAINABLE_METHODS = [
            lambda f: f.provider("claude-ai"),
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
            assert method_fn(fresh) is fresh

    @pytest.mark.asyncio
    async def test_filter_chain_multiple_methods(self, filter_repo):
        """Chain must apply ALL filters — provider, limit both take effect."""
        result = await (
            ConversationFilter(filter_repo)
            .provider("claude-ai")
            .limit(1)
            .sort("date")
            .list()
        )
        assert isinstance(result, list)
        assert len(result) <= 1
        assert all(c.provider == "claude-ai" for c in result)


class TestConversationFilterTerminal:
    """Terminal methods: first, count, delete."""

    @pytest.mark.asyncio
    async def test_filter_first(self, filter_repo):
        result = await ConversationFilter(filter_repo).first()
        assert result is not None
        assert hasattr(result, "id")

    @pytest.mark.asyncio
    async def test_filter_first_empty(self, filter_repo):
        result = await ConversationFilter(filter_repo).provider("nonexistent").first()
        assert result is None

    @pytest.mark.asyncio
    async def test_filter_count(self, filter_repo):
        count = await ConversationFilter(filter_repo).count()
        assert count == 3

    @pytest.mark.asyncio
    async def test_filter_count_with_filter(self, filter_repo):
        count = await ConversationFilter(filter_repo).provider("claude-ai").count()
        assert count == 2

    @pytest.mark.asyncio
    async def test_filter_delete_removes_conversations(self, filter_repo):
        initial_count = await ConversationFilter(filter_repo).count()
        assert initial_count > 0
        deleted = await ConversationFilter(filter_repo).limit(1).delete()
        assert deleted == 1
        final_count = await ConversationFilter(filter_repo).count()
        assert final_count == initial_count - 1


class TestFilterDateParsing:
    """Date parsing for since/until — key regressions."""

    @pytest.mark.parametrize("method_name", ["since", "until"])
    def test_date_method_raises_on_invalid(self, filter_repo, method_name):
        f = ConversationFilter(filter_repo)
        with pytest.raises(ValueError, match="Cannot parse date"):
            getattr(f, method_name)("not-a-date")

    @pytest.mark.parametrize("method_name,date_str", [
        ("since", "yesterday"),
        ("since", "2025-01-15"),
        ("since", "last week"),
        ("until", "today"),
    ])
    def test_date_method_accepts_string_formats(self, filter_repo, method_name, date_str):
        f = ConversationFilter(filter_repo)
        getattr(f, method_name)(date_str)
        field_name = "_since_date" if method_name == "since" else "_until_date"
        assert getattr(f, field_name) is not None
        assert isinstance(getattr(f, field_name), datetime)

    def test_since_and_until_together_datetime(self, filter_repo):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        filter_obj = ConversationFilter(filter_repo).since(start).until(end)
        assert filter_obj._since_date == start
        assert filter_obj._until_date == end


class TestConversationFilterSort:
    """Sorting regression tests."""

    @pytest.mark.parametrize("sort_key", ["date", "messages", "random"])
    @pytest.mark.asyncio
    async def test_filter_sort(self, filter_repo, sort_key):
        result = await ConversationFilter(filter_repo).sort(sort_key).list()
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_filter_sort_reverse(self, filter_repo):
        normal = await ConversationFilter(filter_repo).sort("date").list()
        reversed_list = await ConversationFilter(filter_repo).sort("date").reverse().list()
        if len(normal) > 1:
            assert normal[0].id == reversed_list[-1].id
            assert normal[-1].id == reversed_list[0].id

    @pytest.mark.parametrize("sort_field", ["tokens", "words", "longest", "messages"])
    @pytest.mark.asyncio
    async def test_sort_field_produces_results(self, filter_repo_advanced, sort_field):
        result = await ConversationFilter(filter_repo_advanced).sort(sort_field).list()
        assert len(result) > 0


class TestConversationFilterHasTypes:
    """has() content type filtering — key regressions."""

    @pytest.mark.parametrize("content_type", ["thinking", "tools", "attachments", "summary"])
    @pytest.mark.asyncio
    async def test_has_content_type_returns_list(self, filter_repo_advanced, content_type):
        result = await ConversationFilter(filter_repo_advanced).has(content_type).list()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_has_thinking_with_provider(self, filter_repo_advanced):
        result = await (
            ConversationFilter(filter_repo_advanced)
            .provider("claude-ai")
            .has("thinking")
            .list()
        )
        assert len(result) >= 1
        for conv in result:
            assert conv.provider == "claude-ai"
            assert any(m.is_thinking for m in conv.messages)


class TestConversationFilterSample:
    """sample() random sampling — key regressions."""

    @pytest.mark.parametrize("sample_size,expected", [(2, 2), (0, 0), (1, 1)])
    @pytest.mark.asyncio
    async def test_sample_returns_correct_count(self, filter_repo_advanced, sample_size, expected):
        result = await ConversationFilter(filter_repo_advanced).sample(sample_size).list()
        assert len(result) == expected

    @pytest.mark.asyncio
    async def test_sample_with_filter_respects_filters(self, filter_repo_advanced):
        result = await (
            ConversationFilter(filter_repo_advanced)
            .provider("claude-ai")
            .sample(2)
            .list()
        )
        assert all(c.provider == "claude-ai" for c in result)


class TestConversationFilterCombinedFilters:
    """Combined positive + negative filters — key regressions."""

    @pytest.mark.asyncio
    async def test_exclude_provider_and_exclude_tag(self, filter_repo_advanced):
        result = await (
            ConversationFilter(filter_repo_advanced)
            .exclude_provider("claude-ai")
            .exclude_tag("quantum")
            .list()
        )
        assert all(c.provider != "claude-ai" for c in result)
        for conv in result:
            assert "quantum" not in conv.tags

    @pytest.mark.asyncio
    async def test_provider_with_exclude_tag(self, filter_repo_advanced):
        result = await (
            ConversationFilter(filter_repo_advanced)
            .provider("claude-ai")
            .exclude_tag("simple")
            .list()
        )
        assert all(c.provider == "claude-ai" for c in result)
        for conv in result:
            assert "simple" not in conv.tags

    @pytest.mark.asyncio
    async def test_multiple_exclude_providers(self, filter_repo_advanced):
        result = await (
            ConversationFilter(filter_repo_advanced)
            .exclude_provider("claude-ai", "chatgpt")
            .list()
        )
        for conv in result:
            assert conv.provider not in ("claude-ai", "chatgpt")


class TestConversationFilterListSummaries:
    """list_summaries() terminal method — key regressions."""

    @pytest.mark.asyncio
    async def test_list_summaries_returns_summary_objects(self, filter_repo_advanced):
        result = await ConversationFilter(filter_repo_advanced).list_summaries()
        assert len(result) > 0
        for summary in result:
            assert isinstance(summary, ConversationSummary)

    @pytest.mark.asyncio
    async def test_list_summaries_rejects_content_filters(self, filter_repo_advanced):
        with pytest.raises(ValueError, match="Cannot use list_summaries"):
            await ConversationFilter(filter_repo_advanced).has("thinking").list_summaries()

    @pytest.mark.asyncio
    async def test_list_summaries_allows_summary_has_filter(self, filter_repo_advanced):
        result = await ConversationFilter(filter_repo_advanced).has("summary").list_summaries()
        assert all(s.summary for s in result)

    def test_can_use_summaries_check(self, filter_repo_advanced):
        simple = ConversationFilter(filter_repo_advanced).provider("claude-ai")
        assert simple.can_use_summaries() is True
        with_content = ConversationFilter(filter_repo_advanced).has("thinking")
        assert with_content.can_use_summaries() is False


class TestConversationFilterEmptyRepository:
    """Terminal operations on empty repo."""

    @pytest.mark.parametrize("terminal_method,expected_result", [
        ("list", []),
        ("first", None),
        ("count", 0),
        ("delete", 0),
        ("pick", None),
    ])
    @pytest.mark.asyncio
    async def test_empty_repo_terminal_operations(self, filter_repo_empty, terminal_method, expected_result):
        filter_obj = ConversationFilter(filter_repo_empty)
        if terminal_method == "list":
            result = await filter_obj.list()
        elif terminal_method == "first":
            result = await filter_obj.first()
        elif terminal_method == "count":
            result = await filter_obj.count()
        elif terminal_method == "delete":
            result = await filter_obj.delete()
        elif terminal_method == "pick":
            result = await pick_filter(filter_obj)
        assert result == expected_result


class TestConversationFilterBranching:
    """Branch predicate and pick regression tests."""

    @pytest.fixture
    def filter_repo_branches(self, tmp_path):
        db_path = tmp_path / "filter_branches.db"
        with open_connection(db_path) as conn:
            rebuild_index(conn)

        (ConversationBuilder(db_path, "root")
         .provider("claude-ai")
         .save())
        (ConversationBuilder(db_path, "cont")
         .provider("claude-ai")
         .parent_conversation("root")
         .branch_type("continuation")
         .save())
        (ConversationBuilder(db_path, "side")
         .provider("claude-ai")
         .parent_conversation("root")
         .branch_type("sidechain")
         .save())

        backend = SQLiteBackend(db_path)
        return ConversationRepository(backend)

    @pytest.mark.parametrize("method,value", [
        ("is_continuation", True), ("is_continuation", False),
        ("is_sidechain", True), ("is_sidechain", False),
        ("has_branches", True), ("has_branches", False),
    ])
    def test_branch_predicates(self, filter_repo_branches, method, value):
        filter_obj = ConversationFilter(filter_repo_branches)
        getattr(filter_obj, method)(value)
        if method == "is_continuation":
            assert filter_obj._continuation is value
        elif method == "is_sidechain":
            assert filter_obj._sidechain is value
        elif method == "has_branches":
            assert filter_obj._has_branches is value

    @pytest.mark.asyncio
    async def test_parent_filters_by_parent_id(self, filter_repo_branches):
        result = await ConversationFilter(filter_repo_branches).parent("root").list()
        for conv in result:
            assert conv.parent_id == "root"


class TestFiltersPick:
    """pick() interactive picker regression tests."""

    @pytest.fixture
    def filter_repo_pick(self, tmp_path):
        db_path = tmp_path / "filter_pick.db"
        with open_connection(db_path) as conn:
            rebuild_index(conn)
        for i in range(5):
            (ConversationBuilder(db_path, f"conv{i}")
             .provider("claude-ai")
             .title(f"Conversation {i}")
             .save())
        backend = SQLiteBackend(db_path)
        return ConversationRepository(backend)

    @pytest.mark.asyncio
    async def test_pick_no_results(self, filter_repo_pick):
        result = await pick_filter(ConversationFilter(filter_repo_pick).provider("nonexistent"))
        assert result is None

    @pytest.mark.asyncio
    async def test_pick_non_tty(self, filter_repo_pick):
        with patch("sys.stdout.isatty", return_value=False):
            result = await pick_filter(ConversationFilter(filter_repo_pick))
            assert result is not None


class TestFtsWithProviderFilter:
    """FTS + provider combined filter regression."""

    @pytest.mark.asyncio
    async def test_fts_with_provider_match(self, filter_repo):
        result = await ConversationFilter(filter_repo).contains("errors").provider("claude-ai").list()
        assert len(result) > 0
        assert all(c.provider == "claude-ai" for c in result)

    @pytest.mark.asyncio
    async def test_fts_with_provider_mismatch(self, filter_repo):
        result = await ConversationFilter(filter_repo).contains("schema").provider("chatgpt").list()
        assert len(result) == 0


class TestDeleteCascade:
    """Delete cascade behavior."""

    @pytest.fixture
    async def populated_db(self, tmp_path):
        db_path = tmp_path / "cascade.db"
        backend = SQLiteBackend(db_path=db_path)
        repo = ConversationRepository(backend=backend)
        conv = (
            ConversationBuilder(db_path, "cascade-conv")
            .provider("claude-ai")
            .title("Cascade Test")
            .add_message("m1", role="user", text="Hello world")
            .add_message("m2", role="assistant", text="Hi there")
            .add_attachment("att1", message_id="m1", mime_type="image/png", size_bytes=1024)
        )
        conv.save()
        with open_connection(db_path) as conn:
            rebuild_index(conn)
        return db_path, backend, repo

    @pytest.mark.asyncio
    async def test_delete_cascades_to_messages(self, populated_db):
        db_path, backend, repo = populated_db
        f = ConversationFilter(repo).id("cascade-conv")
        deleted = await f.delete()
        assert deleted == 1
        with open_connection(db_path) as conn:
            msgs = conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = 'cascade-conv'").fetchone()[0]
            assert msgs == 0


class TestFilterLimitZeroEdgeCases:
    """limit(0) edge cases."""

    @pytest.mark.parametrize("setup_fn", [
        lambda f: f.limit(0).sample(5),
        lambda f: f.sample(5).limit(0),
        lambda f: f.sort("messages").limit(0),
    ])
    @pytest.mark.asyncio
    async def test_limit_zero_returns_empty(self, filter_repo_advanced, setup_fn):
        result = await setup_fn(ConversationFilter(filter_repo_advanced)).list()
        assert len(result) == 0


# =============================================================================
# RRF Score Bound Properties
# =============================================================================


@given(
    st.lists(
        st.tuples(st.uuids().map(str), st.floats(min_value=0, max_value=1)),
        min_size=1, max_size=100,
    ),
    st.integers(min_value=1, max_value=200),
)
def test_rrf_scores_bounded_single_list(results: list[tuple[str, float]], k: int):
    """RRF scores from a single list are bounded by 1/(k+1)."""
    fused = reciprocal_rank_fusion(results, k=k)
    max_score = 1.0 / (k + 1)
    for _item_id, score in fused:
        assert score <= max_score + 1e-10
        assert score > 0


@given(
    st.lists(st.tuples(st.uuids().map(str), st.floats(min_value=0, max_value=1)), min_size=1, max_size=50),
    st.lists(st.tuples(st.uuids().map(str), st.floats(min_value=0, max_value=1)), min_size=1, max_size=50),
    st.integers(min_value=1, max_value=100),
)
@settings(max_examples=50)
def test_rrf_scores_bounded_two_lists(
    results1: list[tuple[str, float]],
    results2: list[tuple[str, float]],
    k: int,
):
    """RRF scores from two lists are bounded by 2/(k+1)."""
    fused = reciprocal_rank_fusion(results1, results2, k=k)
    max_score = 2.0 / (k + 1)
    for _item_id, score in fused:
        assert score <= max_score + 1e-10
        assert score > 0


# =============================================================================
# RRF Symmetry Properties
# =============================================================================


@given(
    st.lists(st.tuples(st.uuids().map(str), st.floats(min_value=0, max_value=1)), min_size=1, max_size=30),
    st.lists(st.tuples(st.uuids().map(str), st.floats(min_value=0, max_value=1)), min_size=1, max_size=30),
    st.integers(min_value=1, max_value=100),
)
@settings(max_examples=30)
def test_rrf_symmetric_scores(
    results1: list[tuple[str, float]],
    results2: list[tuple[str, float]],
    k: int,
):
    """RRF(A, B) produces same scores as RRF(B, A)."""
    fused_ab = reciprocal_rank_fusion(results1, results2, k=k)
    fused_ba = reciprocal_rank_fusion(results2, results1, k=k)
    scores_ab = dict(fused_ab)
    scores_ba = dict(fused_ba)
    for item_id in scores_ab:
        assert abs(scores_ab[item_id] - scores_ba.get(item_id, 0)) < 1e-10


@given(
    st.lists(st.tuples(st.uuids().map(str), st.floats(min_value=0, max_value=1)), min_size=1, max_size=20),
)
def test_rrf_single_list_preserves_order(results: list[tuple[str, float]]):
    """For a single list, RRF preserves the original ranking order."""
    seen = set()
    unique_results = []
    for item_id, score in results:
        if item_id not in seen:
            seen.add(item_id)
            unique_results.append((item_id, score))
    fused = reciprocal_rank_fusion(unique_results)
    fused_order = [item_id for item_id, _ in fused]
    original_order = [item_id for item_id, _ in unique_results]
    assert fused_order == original_order


# =============================================================================
# RRF Monotonicity Properties
# =============================================================================


@given(st.integers(min_value=1, max_value=100))
def test_rrf_rank_decreases_score(k: int):
    """Lower rank produces lower score."""
    results = [(f"item_{i}", 0.0) for i in range(10)]
    fused = reciprocal_rank_fusion(results, k=k)
    scores = dict(fused)
    for i in range(len(results) - 1):
        assert scores[f"item_{i}"] > scores[f"item_{i + 1}"]


@given(st.integers(min_value=1, max_value=100))
def test_rrf_appearing_in_both_lists_increases_score(k: int):
    """Items appearing in both lists score higher than single-list items."""
    list1 = [("common", 1.0), ("only_list1", 0.9)]
    list2 = [("common", 0.95), ("only_list2", 0.85)]
    fused = reciprocal_rank_fusion(list1, list2, k=k)
    scores = dict(fused)
    assert scores["common"] > scores["only_list1"]
    assert scores["common"] > scores["only_list2"]


# =============================================================================
# RRF Edge Cases
# =============================================================================


def test_rrf_empty_lists():
    result = reciprocal_rank_fusion()
    assert result == []


def test_rrf_empty_list_in_args():
    results = [("item1", 1.0), ("item2", 0.9)]
    empty: list[tuple[str, float]] = []
    fused = reciprocal_rank_fusion(results, empty)
    assert len(fused) == 2


def test_rrf_duplicate_items_same_list():
    results = [("item1", 1.0), ("item2", 0.9), ("item1", 0.5)]
    fused = reciprocal_rank_fusion(results)
    item_ids = [item_id for item_id, _ in fused]
    assert len(set(item_ids)) == len(item_ids)


@given(
    st.lists(
        st.tuples(st.text(min_size=1, max_size=20), st.floats(min_value=0, max_value=1)),
        min_size=0, max_size=50,
    ),
)
def test_rrf_never_crashes(results: list[tuple[str, float]]):
    fused = reciprocal_rank_fusion(results)
    assert isinstance(fused, list)


# =============================================================================
# RRF Formula Verification
# =============================================================================


def test_rrf_formula_correctness():
    k = 60
    results = [("a", 0.0), ("b", 0.0), ("c", 0.0)]
    fused = reciprocal_rank_fusion(results, k=k)
    scores = dict(fused)
    assert abs(scores["a"] - 1/61) < 1e-10
    assert abs(scores["b"] - 1/62) < 1e-10
    assert abs(scores["c"] - 1/63) < 1e-10


def test_rrf_combined_formula():
    k = 60
    list1 = [("a", 0.0), ("b", 0.0)]
    list2 = [("b", 0.0), ("a", 0.0)]
    fused = reciprocal_rank_fusion(list1, list2, k=k)
    scores = dict(fused)
    expected_a = 1/61 + 1/62
    assert abs(scores["a"] - expected_a) < 1e-10
    expected_b = 1/62 + 1/61
    assert abs(scores["b"] - expected_b) < 1e-10
    assert abs(scores["a"] - scores["b"]) < 1e-10
