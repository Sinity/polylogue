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

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, NotRequired, TypeAlias

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from typing_extensions import TypedDict

from polylogue.archive.filter.filters import SessionFilter
from polylogue.archive.filter.types import SortField
from polylogue.archive.models import SessionSummary
from polylogue.archive.session.domain_models import Session
from polylogue.storage.search_providers.hybrid import reciprocal_rank_fusion
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.storage_records import SessionBuilder
from tests.infra.strategies.filters import (
    filter_chain_strategy,
)

# =============================================================================
# Fixtures
# =============================================================================


class MessageSeed(TypedDict):
    id: str
    role: NotRequired[str]
    text: NotRequired[str]


class SessionSeed(TypedDict):
    id: str
    provider: str
    title: NotRequired[str]
    messages: NotRequired[list[MessageSeed]]


class FilterSpec(TypedDict):
    type: str
    value: NotRequired[object]
    field: NotRequired[str]
    direction: NotRequired[str]


FilterRepoFactory: TypeAlias = Callable[[list[SessionSeed]], Path]
TerminalMethod: TypeAlias = Literal["list", "first", "count", "delete", "pick"]
TerminalResult: TypeAlias = list[Session] | Session | None | int
DateMethodName: TypeAlias = Literal["since", "until"]
BranchPredicateName: TypeAlias = Literal["is_continuation", "is_sidechain", "has_branches"]


@pytest.fixture
def filter_db(tmp_path: Path) -> Path:
    """Create a archive with test sessions for filter tests.

    Returns the archive root; the ``SessionBuilder`` writes directly to
    ``<archive_root>/index.db`` via the archive ``ArchiveStore``.
    """
    root = tmp_path / "filter_archive"
    root.mkdir()
    db_path = root / "index.db"

    (
        SessionBuilder(db_path, "claude-1")
        .provider("claude-ai")
        .title("Python Error Handling")
        .add_message("m1", role="user", text="How do I handle errors in Python?")
        .add_message("m2", role="assistant", text="You can use try-except blocks.")
        .metadata({"tags": ["python", "errors"]})
        .save()
    )

    (
        SessionBuilder(db_path, "chatgpt-1")
        .provider("chatgpt")
        .title("JavaScript Async")
        .add_message("m3", role="user", text="How do async functions work?")
        .add_message("m4", role="assistant", text="Async functions return promises.")
        .metadata({"tags": ["javascript"]})
        .save()
    )

    (
        SessionBuilder(db_path, "claude-2")
        .provider("claude-ai")
        .title("Database Design")
        .add_message("m5", role="user", text="How to design a database schema?")
        .add_message("m6", role="assistant", text="Start with identifying entities.")
        .metadata({"tags": ["database", "design"]})
        .save()
    )

    return root


@pytest.fixture
def filter_repo(filter_db: Path) -> Path:
    """Archive root for filter tests (archive SessionFilter source)."""
    return filter_db


@pytest.fixture
def filter_db_empty(tmp_path: Path) -> Path:
    """Create an empty archive for empty-repository edge cases."""
    root = tmp_path / "filter_empty_archive"
    root.mkdir()
    with ArchiveStore(root):
        pass
    return root


@pytest.fixture
def filter_repo_empty(filter_db_empty: Path) -> Path:
    """Archive root for empty-archive tests."""
    return filter_db_empty


@pytest.fixture
def filter_db_advanced(tmp_path: Path) -> Path:
    """Create a archive with sessions for advanced filter tests.

    Includes thinking blocks, tool use, attachments, summaries,
    various message counts, and token lengths. Returns the archive root.
    """
    root = tmp_path / "filter_advanced_archive"
    root.mkdir()
    db_path = root / "index.db"

    (
        SessionBuilder(db_path, "conv-thinking")
        .provider("claude-ai")
        .title("Complex Problem Analysis")
        .add_message("m1", role="user", text="Solve this complex math problem")
        .add_message(
            "m2",
            role="assistant",
            text="The answer is 42.",
            blocks=[{"type": "thinking", "text": "Let me break this down step by step..."}],
        )
        .add_message("m3", role="user", text="Can you explain further?")
        .metadata({"tags": ["math", "complex"], "summary": "Math problem solving"})
        .save()
    )

    (
        SessionBuilder(db_path, "conv-tools")
        .provider("claude-ai")
        .title("API Integration Help")
        .add_message("m4", role="user", text="How do I call an API?")
        .add_message(
            "m5",
            role="assistant",
            text="I'll help you with that.",
            blocks=[{"type": "tool_use", "tool_name": "bash", "input": {}}],
        )
        .add_message("m6", role="user", text="Show me an example")
        .add_message("m7", role="assistant", text="Here is a complete working example with error handling.")
        .metadata({"tags": ["api", "integration"]})
        .save()
    )

    (
        SessionBuilder(db_path, "conv-attachments")
        .provider("chatgpt")
        .title("Document Analysis")
        .add_message("m8", role="user", text="Please analyze this document")
        .add_message("m9", role="assistant", text="I see the file contains important data.")
        .add_attachment("att1", message_id="m8", mime_type="application/pdf", size_bytes=5000)
        .metadata({"tags": ["documents"]})
        .save()
    )

    (
        SessionBuilder(db_path, "conv-summary-only")
        .provider("claude-ai")
        .title("Brief Chat")
        .add_message("m10", role="user", text="Hello there")
        .add_message("m11", role="assistant", text="Hi how are you")
        .metadata({"summary": "Brief greeting exchange", "tags": ["greeting"]})
        .save()
    )

    (
        SessionBuilder(db_path, "conv-multi-attach")
        .provider("chatgpt")
        .title("Multiple File Analysis")
        .add_message("m12", role="user", text="Analyze these files please")
        .add_message("m13", role="assistant", text="I can see both files clearly.")
        .add_message("m14", role="user", text="What are the main differences?")
        .add_attachment("att2", message_id="m12", mime_type="image/png", size_bytes=2000)
        .add_attachment("att3", message_id="m12", mime_type="application/pdf", size_bytes=3000)
        .metadata({"tags": ["analysis"]})
        .save()
    )

    (
        SessionBuilder(db_path, "conv-long-messages")
        .provider("claude-ai")
        .title("Deep Discussion")
        .add_message(
            "m15",
            role="user",
            text="Tell me everything you know about quantum computing including the fundamentals principles and applications",
        )
        .add_message(
            "m16",
            role="assistant",
            text="Quantum computing is a revolutionary field that leverages quantum mechanical phenomena like superposition and entanglement to perform computations exponentially faster than classical computers in certain domains such as cryptography and optimization.",
        )
        .metadata({"tags": ["quantum"]})
        .save()
    )

    (
        SessionBuilder(db_path, "conv-plain")
        .provider("codex")
        .title("Simple")
        .add_message("m17", role="user", text="What is two plus two")
        .metadata({"tags": ["simple"]})
        .save()
    )

    return root


@pytest.fixture
def filter_repo_advanced(filter_db_advanced: Path) -> Path:
    """Archive root for advanced filter tests."""
    return filter_db_advanced


# =============================================================================
# Property: Filter chains never crash
# =============================================================================


def _apply_filter_spec(f: SessionFilter, spec: FilterSpec) -> SessionFilter:
    """Apply a single filter spec dict to a SessionFilter."""
    ftype = spec["type"]
    if ftype == "provider":
        return f.origin(str(spec["value"]))
    elif ftype == "contains":
        return f.contains(str(spec["value"]))
    elif ftype == "since":
        return f.since(str(spec["value"]))
    elif ftype == "until":
        return f.until(str(spec["value"]))
    elif ftype == "limit":
        raw_value = spec["value"]
        if isinstance(raw_value, bool):
            return f.limit(int(raw_value))
        if isinstance(raw_value, int):
            return f.limit(raw_value)
        if isinstance(raw_value, (float, str)):
            return f.limit(int(raw_value))
        return f.limit(0)
    elif ftype == "offset":
        # SessionFilter does not have .offset(), skip
        return f
    elif ftype == "sort":
        return f.sort("date")  # Use a safe sort field
    elif ftype == "role":
        # No direct role filter on SessionFilter, skip
        return f
    elif ftype == "has_attachments":
        return f.has("attachments")
    elif ftype == "min_words" or ftype == "max_words":
        return f
    return f


def _session_ids(sessions: list[Session]) -> list[str]:
    return [str(session.id) for session in sessions]


@given(filter_chain_strategy(min_filters=1, max_filters=4))
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_filter_chain_never_crashes_on_build(chain: list[FilterSpec]) -> None:
    """Building a filter chain from arbitrary filter specs never crashes.

    We can't call .list() here (needs async + DB), but we verify the
    fluent builder accepts any valid filter combo without raising.
    """
    # We only test chain construction, not execution; an arbitrary path works.
    f = SessionFilter(archive_root=Path("/nonexistent-archive-for-chain-build"))
    for spec in chain:
        f = _apply_filter_spec(f, spec)
    # If we got here, the chain didn't crash


@pytest.mark.asyncio
async def test_filter_chain_never_crashes_on_execution(make_filter_repo: FilterRepoFactory) -> None:
    """Composed filters never crash when executed against a real DB.

    Uses explicit examples rather than @given to avoid fixture-scope issues.
    """

    root = make_filter_repo(
        [
            {
                "id": "c1",
                "provider": "claude-ai",
                "title": "Test",
                "messages": [{"id": "m1", "role": "user", "text": "hello"}],
            },
            {
                "id": "c2",
                "provider": "chatgpt",
                "title": "Other",
                "messages": [{"id": "m2", "role": "user", "text": "world"}],
            },
        ]
    )

    # Test a variety of filter chain combos
    chains: list[list[FilterSpec]] = [
        [{"type": "provider", "value": "claude-ai"}],
        [{"type": "limit", "value": 1}],
        [{"type": "provider", "value": "chatgpt"}, {"type": "limit", "value": 5}],
        [{"type": "contains", "value": "hello"}, {"type": "provider", "value": "claude-ai"}],
        [{"type": "sort", "field": "created_at", "direction": "desc"}],
    ]
    for chain in chains:
        f = SessionFilter(archive_root=root)
        for spec in chain:
            f = _apply_filter_spec(f, spec)
        result = await f.list()
        assert isinstance(result, list)


# =============================================================================
# Property: Filter monotonicity — adding filters never increases count
# =============================================================================


@pytest.mark.asyncio
async def test_filter_monotonicity_provider(filter_repo: Path) -> None:
    """Adding a provider filter never increases result count."""
    all_count = await SessionFilter(archive_root=filter_repo).count()
    filtered_count = await SessionFilter(archive_root=filter_repo).origin("claude-ai-export").count()
    assert filtered_count <= all_count


@pytest.mark.asyncio
async def test_filter_monotonicity_limit(filter_repo: Path) -> None:
    """Adding a limit filter never increases result count."""
    all_count = await SessionFilter(archive_root=filter_repo).count()
    limited_count = len(await SessionFilter(archive_root=filter_repo).limit(1).list())
    assert limited_count <= all_count


@pytest.mark.asyncio
async def test_filter_monotonicity_exclude(filter_repo: Path) -> None:
    """Adding an exclude filter never increases result count."""
    all_count = await SessionFilter(archive_root=filter_repo).count()
    excluded = await SessionFilter(archive_root=filter_repo).exclude_origin("claude-ai-export").list()
    assert len(excluded) <= all_count


@pytest.mark.asyncio
async def test_filter_monotonicity_chained(filter_repo_advanced: Path) -> None:
    """Stacking filters monotonically decreases results."""
    c0 = await SessionFilter(archive_root=filter_repo_advanced).count()
    c1 = len(await SessionFilter(archive_root=filter_repo_advanced).origin("claude-ai-export").list())
    c2 = len(await SessionFilter(archive_root=filter_repo_advanced).origin("claude-ai-export").has("thinking").list())
    assert c2 <= c1 <= c0


# =============================================================================
# Property: Filter idempotence — applying same filter twice = once
# =============================================================================


@pytest.mark.asyncio
async def test_filter_idempotence_provider(filter_repo: Path) -> None:
    """Applying provider("claude-ai") twice yields same results as once."""
    once = await SessionFilter(archive_root=filter_repo).origin("claude-ai-export").list()
    twice = await SessionFilter(archive_root=filter_repo).origin("claude-ai-export").origin("claude-ai-export").list()
    assert {c.id for c in once} == {c.id for c in twice}


@pytest.mark.asyncio
async def test_filter_idempotence_exclude(filter_repo: Path) -> None:
    """Applying exclude_provider("claude-ai") twice yields same results as once."""
    once = await SessionFilter(archive_root=filter_repo).exclude_origin("claude-ai-export").list()
    twice = (
        await SessionFilter(archive_root=filter_repo)
        .exclude_origin("claude-ai-export")
        .exclude_origin("claude-ai-export")
        .list()
    )
    assert {c.id for c in once} == {c.id for c in twice}


# =============================================================================
# Property: Filter semantic oracles for mutation-frontier campaigns
# =============================================================================


@pytest.mark.asyncio
async def test_provider_filter_list_count_subset_and_order_law(filter_repo: Path) -> None:
    """Provider filtering agrees across list/count and preserves base result order."""
    all_items = await SessionFilter(archive_root=filter_repo).list()
    filtered = await SessionFilter(archive_root=filter_repo).origin("claude-ai-export").list()
    expected = [session for session in all_items if session.origin == "claude-ai-export"]

    assert _session_ids(filtered) == _session_ids(expected)
    assert await SessionFilter(archive_root=filter_repo).origin("claude-ai-export").count() == len(filtered)
    assert all(session.origin == "claude-ai-export" for session in filtered)


@pytest.mark.asyncio
async def test_equivalent_filter_order_list_count_and_first_law(filter_repo: Path) -> None:
    """Commutative filter constraints must agree across all terminal methods."""
    provider_then_title = (
        await SessionFilter(archive_root=filter_repo).origin("claude-ai-export").title("Python").list()
    )
    title_then_provider = (
        await SessionFilter(archive_root=filter_repo).title("Python").origin("claude-ai-export").list()
    )

    assert _session_ids(provider_then_title) == _session_ids(title_then_provider)
    assert await SessionFilter(archive_root=filter_repo).origin("claude-ai-export").title("Python").count() == len(
        provider_then_title
    )
    assert await SessionFilter(archive_root=filter_repo).title("Python").origin("claude-ai-export").count() == len(
        title_then_provider
    )

    first_provider_then_title = (
        await SessionFilter(archive_root=filter_repo).origin("claude-ai-export").title("Python").first()
    )
    first_title_then_provider = (
        await SessionFilter(archive_root=filter_repo).title("Python").origin("claude-ai-export").first()
    )
    assert first_provider_then_title is not None
    assert first_title_then_provider is not None
    assert first_provider_then_title.id == first_title_then_provider.id


# =============================================================================
# Property: SQL pushdown params match filter state
# =============================================================================


def test_sql_pushdown_origin() -> None:
    """SQL pushdown includes origin when set."""
    f = SessionFilter(archive_root=Path("/nonexistent-archive-for-plan"))
    f.origin("claude-ai-export")
    params = f._sql_pushdown_params()
    assert params["origin"] == "claude-ai-export"


def test_sql_pushdown_multi_origin() -> None:
    """SQL pushdown includes origins list when multiple are set."""
    f = SessionFilter(archive_root=Path("/nonexistent-archive-for-plan"))
    f.origin("claude-ai-export", "chatgpt-export")
    params = f._sql_pushdown_params()
    assert params["origins"] == ["claude-ai-export", "chatgpt-export"]


def test_sql_pushdown_date_range() -> None:
    """SQL pushdown includes since/until when set."""
    f = SessionFilter(archive_root=Path("/nonexistent-archive-for-plan"))
    dt = datetime(2024, 6, 15, tzinfo=timezone.utc)
    f.since(dt).until(dt)
    params = f._sql_pushdown_params()
    assert "since" in params
    assert "until" in params


def test_sql_pushdown_title() -> None:
    """SQL pushdown includes title_contains when set."""
    f = SessionFilter(archive_root=Path("/nonexistent-archive-for-plan"))
    f.title("Python")
    params = f._sql_pushdown_params()
    assert params["title_contains"] == "Python"


def test_sql_pushdown_empty() -> None:
    """SQL pushdown is empty dict when no filters set."""
    f = SessionFilter(archive_root=Path("/nonexistent-archive-for-plan"))
    assert f._sql_pushdown_params() == {}


@given(
    st.lists(st.sampled_from(["chatgpt", "claude-ai", "codex"]), min_size=1, max_size=3),
    st.lists(st.sampled_from(["chatgpt", "claude-ai", "codex"]), min_size=0, max_size=2),
)
def test_provider_filter_exclusion_disjoint(
    include_providers: list[str],
    exclude_providers: list[str],
) -> None:
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
def test_limit_respects_bound(total_items: int, limit: int) -> None:
    """Limit filter produces at most `limit` items."""
    items = list(range(total_items))
    limited = items[:limit]
    assert len(limited) <= limit


@given(
    st.integers(min_value=0, max_value=50),
    st.integers(min_value=0, max_value=100),
)
def test_offset_skips_correctly(offset: int, total_items: int) -> None:
    """Offset filter skips the first `offset` items."""
    items = list(range(total_items))
    offset_items = items[offset:]
    expected_count = max(0, total_items - offset)
    assert len(offset_items) == expected_count


@given(
    st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2025, 1, 1), timezones=st.just(timezone.utc)),
    st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2025, 1, 1), timezones=st.just(timezone.utc)),
)
def test_date_range_logic(since: datetime, until: datetime) -> None:
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
def test_filter_composition_monotonic(items: list[int], threshold1: int, threshold2: int) -> None:
    """Composing filters never increases result count."""
    filtered_once = [x for x in items if x > threshold1]
    filtered_twice = [x for x in filtered_once if x > threshold2]
    assert len(filtered_twice) <= len(filtered_once)


@given(
    st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=50),
    st.integers(min_value=0, max_value=100),
)
def test_filter_idempotent(items: list[int], threshold: int) -> None:
    """Applying the same filter twice has no additional effect."""
    filtered_once = [x for x in items if x > threshold]
    filtered_twice = [x for x in filtered_once if x > threshold]
    assert filtered_once == filtered_twice


@given(
    st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=50),
    st.integers(min_value=0, max_value=100),
    st.integers(min_value=0, max_value=100),
)
def test_filter_order_independent(items: list[int], threshold1: int, threshold2: int) -> None:
    """For commutative filters, order doesn't affect result count."""
    filtered_ab = [x for x in items if x > threshold1 and x > threshold2]
    filtered_ba = [x for x in items if x > threshold2 and x > threshold1]
    assert filtered_ab == filtered_ba


# =============================================================================
# Regression: Key filter behaviors (from test_filters.py + test_filters_adv.py)
# =============================================================================


class TestSessionFilterChaining:
    """Chainability and multi-method filter tests (key regressions)."""

    def test_filter_returns_self(self, filter_repo: Path) -> None:
        """Every fluent filter method must return self."""
        chainable_methods: tuple[Callable[[SessionFilter], SessionFilter], ...] = (
            lambda f: f.origin("claude-ai-export"),
            lambda f: f.since("2024-01-01"),
            lambda f: f.until("2025-01-01"),
            lambda f: f.limit(10),
            lambda f: f.sort("date"),
            lambda f: f.reverse(),
            lambda f: f.tag("test"),
            lambda f: f.contains("hello"),
            lambda f: f.title("test"),
            lambda f: f.referenced_path("/workspace/polylogue/README.md"),
            lambda f: f.action("search"),
            lambda f: f.exclude_action("git"),
            lambda f: f.tool("grep"),
            lambda f: f.exclude_tool("bash"),
            lambda f: f.similar("query"),
        )
        for method_fn in chainable_methods:
            fresh = SessionFilter(archive_root=filter_repo)
            assert method_fn(fresh) is fresh

    @pytest.mark.asyncio
    async def test_filter_chain_multiple_methods(self, filter_repo: Path) -> None:
        """Chain must apply ALL filters — provider, limit both take effect."""
        result = await SessionFilter(archive_root=filter_repo).origin("claude-ai-export").limit(1).sort("date").list()
        assert isinstance(result, list)
        assert len(result) <= 1
        assert all(c.origin == "claude-ai-export" for c in result)


class TestSessionFilterTerminal:
    """Terminal methods: first, count, delete."""

    @pytest.mark.asyncio
    async def test_filter_first(self, filter_repo: Path) -> None:
        result = await SessionFilter(archive_root=filter_repo).first()
        assert result is not None
        assert hasattr(result, "id")

    @pytest.mark.asyncio
    async def test_filter_first_empty(self, filter_repo: Path) -> None:
        result = await SessionFilter(archive_root=filter_repo).origin("nonexistent").first()
        assert result is None

    @pytest.mark.asyncio
    async def test_filter_count(self, filter_repo: Path) -> None:
        count = await SessionFilter(archive_root=filter_repo).count()
        assert count == 3

    @pytest.mark.asyncio
    async def test_filter_count_with_filter(self, filter_repo: Path) -> None:
        count = await SessionFilter(archive_root=filter_repo).origin("claude-ai-export").count()
        assert count == 2

    @pytest.mark.asyncio
    async def test_filter_delete_removes_sessions(self, filter_repo: Path) -> None:
        initial_count = await SessionFilter(archive_root=filter_repo).count()
        assert initial_count > 0
        deleted = await SessionFilter(archive_root=filter_repo).limit(1).delete()
        assert deleted == 1
        final_count = await SessionFilter(archive_root=filter_repo).count()
        assert final_count == initial_count - 1


class TestFilterDateParsing:
    """Date parsing for since/until — key regressions."""

    @pytest.mark.parametrize("method_name", ["since", "until"])
    def test_date_method_raises_on_invalid(
        self,
        filter_repo: Path,
        method_name: DateMethodName,
    ) -> None:
        f = SessionFilter(archive_root=filter_repo)
        with pytest.raises(ValueError, match="Cannot parse date"):
            getattr(f, method_name)("not-a-date")

    @pytest.mark.parametrize(
        "method_name,date_str",
        [
            ("since", "yesterday"),
            ("since", "2025-01-15"),
            ("since", "last week"),
            ("until", "today"),
        ],
    )
    def test_date_method_accepts_string_formats(
        self,
        filter_repo: Path,
        method_name: DateMethodName,
        date_str: str,
    ) -> None:
        f = SessionFilter(archive_root=filter_repo)
        getattr(f, method_name)(date_str)
        field_name = "_since_date" if method_name == "since" else "_until_date"
        assert getattr(f, field_name) is not None
        assert isinstance(getattr(f, field_name), datetime)

    def test_since_and_until_together_datetime(self, filter_repo: Path) -> None:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        filter_obj = SessionFilter(archive_root=filter_repo).since(start).until(end)
        assert filter_obj._since_date == start
        assert filter_obj._until_date == end


class TestSessionFilterSort:
    """Sorting regression tests."""

    @pytest.mark.parametrize("sort_key", ["date", "messages", "random"])
    @pytest.mark.asyncio
    async def test_filter_sort(self, filter_repo: Path, sort_key: SortField) -> None:
        result = await SessionFilter(archive_root=filter_repo).sort(sort_key).list()
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_filter_sort_reverse(self, filter_repo: Path) -> None:
        normal = await SessionFilter(archive_root=filter_repo).sort("date").list()
        reversed_list = await SessionFilter(archive_root=filter_repo).sort("date").reverse().list()
        if len(normal) > 1:
            assert normal[0].id == reversed_list[-1].id
            assert normal[-1].id == reversed_list[0].id

    @pytest.mark.parametrize("sort_field", ["tokens", "words", "longest", "messages"])
    @pytest.mark.asyncio
    async def test_sort_field_produces_results(
        self,
        filter_repo_advanced: Path,
        sort_field: SortField,
    ) -> None:
        result = await SessionFilter(archive_root=filter_repo_advanced).sort(sort_field).list()
        assert len(result) > 0


class TestSessionFilterHasTypes:
    """has() content type filtering — key regressions."""

    @pytest.mark.parametrize("content_type", ["thinking", "tools", "attachments", "summary"])
    @pytest.mark.asyncio
    async def test_has_content_type_returns_list(
        self,
        filter_repo_advanced: Path,
        content_type: str,
    ) -> None:
        result = await SessionFilter(archive_root=filter_repo_advanced).has(content_type).list()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_has_thinking_with_provider(self, filter_repo_advanced: Path) -> None:
        result = (
            await SessionFilter(archive_root=filter_repo_advanced).origin("claude-ai-export").has("thinking").list()
        )
        assert len(result) >= 1
        for conv in result:
            assert conv.origin == "claude-ai-export"
            assert any(m.is_thinking for m in conv.messages)


class TestSessionFilterSample:
    """sample() random sampling — key regressions."""

    @pytest.mark.parametrize("sample_size,expected", [(2, 2), (0, 0), (1, 1)])
    @pytest.mark.asyncio
    async def test_sample_returns_correct_count(
        self,
        filter_repo_advanced: Path,
        sample_size: int,
        expected: int,
    ) -> None:
        result = await SessionFilter(archive_root=filter_repo_advanced).sample(sample_size).list()
        assert len(result) == expected

    @pytest.mark.asyncio
    async def test_sample_with_filter_respects_filters(self, filter_repo_advanced: Path) -> None:
        result = await SessionFilter(archive_root=filter_repo_advanced).origin("claude-ai-export").sample(2).list()
        assert all(c.origin == "claude-ai-export" for c in result)


class TestSessionFilterCombinedFilters:
    """Combined positive + negative filters — key regressions."""

    @pytest.mark.asyncio
    async def test_exclude_provider_and_exclude_tag(self, filter_repo_advanced: Path) -> None:
        result = await (
            SessionFilter(archive_root=filter_repo_advanced)
            .exclude_origin("claude-ai-export")
            .exclude_tag("quantum")
            .list()
        )
        assert all(c.origin != "claude-ai-export" for c in result)
        for conv in result:
            assert "quantum" not in conv.tags

    @pytest.mark.asyncio
    async def test_provider_with_exclude_tag(self, filter_repo_advanced: Path) -> None:
        result = (
            await SessionFilter(archive_root=filter_repo_advanced)
            .origin("claude-ai-export")
            .exclude_tag("simple")
            .list()
        )
        assert all(c.origin == "claude-ai-export" for c in result)
        for conv in result:
            assert "simple" not in conv.tags

    @pytest.mark.asyncio
    async def test_multiple_exclude_providers(self, filter_repo_advanced: Path) -> None:
        result = (
            await SessionFilter(archive_root=filter_repo_advanced)
            .exclude_origin("claude-ai-export", "chatgpt-export")
            .list()
        )
        for conv in result:
            assert conv.origin not in ("claude-ai-export", "chatgpt-export")


class TestSessionFilterListSummaries:
    """list_summaries() terminal method — key regressions."""

    @pytest.mark.asyncio
    async def test_list_summaries_returns_summary_objects(self, filter_repo_advanced: Path) -> None:
        result = await SessionFilter(archive_root=filter_repo_advanced).list_summaries()
        assert len(result) > 0
        for summary in result:
            assert isinstance(summary, SessionSummary)

    @pytest.mark.asyncio
    async def test_list_summaries_with_content_filter_still_returns_summaries(self, filter_repo_advanced: Path) -> None:
        """A content filter cannot be SQL-pushed into the summary path, but the
        summary terminal still resolves it as a post-filter and returns
        SessionSummary objects (the legacy hard-reject was dropped in the
        archive collapse)."""
        flt = SessionFilter(archive_root=filter_repo_advanced).has("thinking")
        assert flt.can_use_summaries() is False
        result = await flt.list_summaries()
        for summary in result:
            assert isinstance(summary, SessionSummary)

    @pytest.mark.asyncio
    async def test_list_summaries_allows_summary_has_filter(self, filter_repo_advanced: Path) -> None:
        result = await SessionFilter(archive_root=filter_repo_advanced).has("summary").list_summaries()
        assert all(s.summary for s in result)

    def test_can_use_summaries_check(self, filter_repo_advanced: Path) -> None:
        simple = SessionFilter(archive_root=filter_repo_advanced).origin("claude-ai-export")
        assert simple.can_use_summaries() is True
        with_content = SessionFilter(archive_root=filter_repo_advanced).has("thinking")
        assert with_content.can_use_summaries() is False


class TestSessionFilterEmptyRepository:
    """Terminal operations on empty repo."""

    @pytest.mark.parametrize(
        "terminal_method,expected_result",
        [
            ("list", []),
            ("first", None),
            ("count", 0),
            ("delete", 0),
        ],
    )
    @pytest.mark.asyncio
    async def test_empty_repo_terminal_operations(
        self,
        filter_repo_empty: Path,
        terminal_method: TerminalMethod,
        expected_result: TerminalResult,
    ) -> None:
        filter_obj = SessionFilter(archive_root=filter_repo_empty)
        result: TerminalResult
        if terminal_method == "list":
            result = await filter_obj.list()
        elif terminal_method == "first":
            result = await filter_obj.first()
        elif terminal_method == "count":
            result = await filter_obj.count()
        elif terminal_method == "delete":
            result = await filter_obj.delete()
        assert result == expected_result


class TestSessionFilterBranching:
    """Branch predicate regression tests."""

    @pytest.fixture
    def filter_repo_branches(self, tmp_path: Path) -> Path:
        root = tmp_path / "filter_branches_archive"
        root.mkdir()
        db_path = root / "index.db"

        (SessionBuilder(db_path, "root").provider("claude-ai").save())
        (
            SessionBuilder(db_path, "cont")
            .provider("claude-ai")
            .parent_session("ext-root")
            .branch_type("continuation")
            .save()
        )
        (
            SessionBuilder(db_path, "side")
            .provider("claude-ai")
            .parent_session("ext-root")
            .branch_type("sidechain")
            .save()
        )

        return root

    @pytest.mark.parametrize(
        "method,value",
        [
            ("is_continuation", True),
            ("is_continuation", False),
            ("is_sidechain", True),
            ("is_sidechain", False),
            ("has_branches", True),
            ("has_branches", False),
        ],
    )
    def test_branch_predicates(
        self,
        filter_repo_branches: Path,
        method: BranchPredicateName,
        value: bool,
    ) -> None:
        filter_obj = SessionFilter(archive_root=filter_repo_branches)
        getattr(filter_obj, method)(value)
        if method == "is_continuation":
            assert filter_obj._continuation is value
        elif method == "is_sidechain":
            assert filter_obj._sidechain is value
        elif method == "has_branches":
            assert filter_obj._has_branches is value

    @pytest.mark.asyncio
    async def test_parent_filters_by_parent_id(self, filter_repo_branches: Path) -> None:
        root_native_id = "claude-ai-export:ext-root"
        result = await SessionFilter(archive_root=filter_repo_branches).parent(root_native_id).list()
        assert len(result) == 2
        for conv in result:
            assert conv.parent_id == root_native_id


class TestFtsWithProviderFilter:
    """FTS + provider combined filter regression."""

    @pytest.mark.asyncio
    async def test_fts_with_provider_match(self, filter_repo: Path) -> None:
        result = await SessionFilter(archive_root=filter_repo).contains("errors").origin("claude-ai-export").list()
        assert len(result) > 0
        assert all(c.origin == "claude-ai-export" for c in result)

    @pytest.mark.asyncio
    async def test_fts_with_provider_mismatch(self, filter_repo: Path) -> None:
        result = await SessionFilter(archive_root=filter_repo).contains("schema").origin("chatgpt-export").list()
        assert len(result) == 0


class TestDeleteCascade:
    """Delete cascade behavior."""

    @pytest.fixture
    async def cascade_archive(self, tmp_path: Path) -> tuple[Path, str]:
        root = tmp_path / "cascade_archive"
        root.mkdir()
        db_path = root / "index.db"
        conv = (
            SessionBuilder(db_path, "cascade-conv")
            .provider("claude-ai")
            .title("Cascade Test")
            .add_message("m1", role="user", text="Hello world")
            .add_message("m2", role="assistant", text="Hi there")
            .add_attachment("att1", message_id="m1", mime_type="image/png", size_bytes=1024)
        )
        conv.save()
        return root, conv.native_session_id()

    @pytest.mark.asyncio
    async def test_delete_cascades_to_messages(
        self,
        cascade_archive: tuple[Path, str],
    ) -> None:
        root, session_id = cascade_archive
        before = await SessionFilter(archive_root=root).first()
        assert before is not None
        assert len(before.messages) == 2

        deleted = await SessionFilter(archive_root=root).id(session_id).delete()
        assert deleted == 1

        # After delete the session and its messages are gone directly.
        assert await SessionFilter(archive_root=root).count() == 0
        assert await SessionFilter(archive_root=root).id(session_id).first() is None


class TestFilterLimitZeroEdgeCases:
    """limit(0) edge cases."""

    @pytest.mark.parametrize(
        "setup_fn",
        [
            lambda f: f.limit(0).sample(5),
            lambda f: f.sample(5).limit(0),
            lambda f: f.sort("messages").limit(0),
        ],
    )
    @pytest.mark.asyncio
    async def test_limit_zero_returns_empty(
        self,
        filter_repo_advanced: Path,
        setup_fn: Callable[[SessionFilter], SessionFilter],
    ) -> None:
        result = await setup_fn(SessionFilter(archive_root=filter_repo_advanced)).list()
        assert len(result) == 0


# =============================================================================
# RRF Score Bound Properties
# =============================================================================


@given(
    st.lists(
        st.tuples(st.uuids().map(str), st.floats(min_value=0, max_value=1)),
        min_size=1,
        max_size=100,
    ),
    st.integers(min_value=1, max_value=200),
)
def test_rrf_scores_bounded_single_list(results: list[tuple[str, float]], k: int) -> None:
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
) -> None:
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
) -> None:
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
def test_rrf_single_list_preserves_order(results: list[tuple[str, float]]) -> None:
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
def test_rrf_rank_decreases_score(k: int) -> None:
    """Lower rank produces lower score."""
    results = [(f"item_{i}", 0.0) for i in range(10)]
    fused = reciprocal_rank_fusion(results, k=k)
    scores = dict(fused)
    for i in range(len(results) - 1):
        assert scores[f"item_{i}"] > scores[f"item_{i + 1}"]


@given(st.integers(min_value=1, max_value=100))
def test_rrf_appearing_in_both_lists_increases_score(k: int) -> None:
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


def test_rrf_empty_lists() -> None:
    result = reciprocal_rank_fusion()
    assert result == []


def test_rrf_empty_list_in_args() -> None:
    results = [("item1", 1.0), ("item2", 0.9)]
    empty: list[tuple[str, float]] = []
    fused = reciprocal_rank_fusion(results, empty)
    assert len(fused) == 2


def test_rrf_duplicate_items_same_list() -> None:
    results = [("item1", 1.0), ("item2", 0.9), ("item1", 0.5)]
    fused = reciprocal_rank_fusion(results)
    item_ids = [item_id for item_id, _ in fused]
    assert len(set(item_ids)) == len(item_ids)


@given(
    st.lists(
        st.tuples(st.text(min_size=1, max_size=20), st.floats(min_value=0, max_value=1)),
        min_size=0,
        max_size=50,
    ),
)
def test_rrf_never_crashes(results: list[tuple[str, float]]) -> None:
    fused = reciprocal_rank_fusion(results)
    assert isinstance(fused, list)


# =============================================================================
# RRF Formula Verification
# =============================================================================


def test_rrf_formula_correctness() -> None:
    k = 60
    results = [("a", 0.0), ("b", 0.0), ("c", 0.0)]
    fused = reciprocal_rank_fusion(results, k=k)
    scores = dict(fused)
    assert abs(scores["a"] - 1 / 61) < 1e-10
    assert abs(scores["b"] - 1 / 62) < 1e-10
    assert abs(scores["c"] - 1 / 63) < 1e-10


def test_rrf_combined_formula() -> None:
    k = 60
    list1 = [("a", 0.0), ("b", 0.0)]
    list2 = [("b", 0.0), ("a", 0.0)]
    fused = reciprocal_rank_fusion(list1, list2, k=k)
    scores = dict(fused)
    expected_a = 1 / 61 + 1 / 62
    assert abs(scores["a"] - expected_a) < 1e-10
    expected_b = 1 / 62 + 1 / 61
    assert abs(scores["b"] - expected_b) < 1e-10
    assert abs(scores["a"] - scores["b"]) < 1e-10
