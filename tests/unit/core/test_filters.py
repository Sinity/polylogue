"""Focused foundational contracts for ConversationFilter.

This owner file covers fluent API persistence, basic selection semantics,
terminal methods, summary compatibility, and execution-planning helpers.
Advanced content-aware, branching, semantic, and summary-edge behavior lives
in ``test_filters_adv.py``.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from polylogue.lib.filters import ConversationFilter
from polylogue.lib.models import Conversation, ConversationSummary, Message
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.index import rebuild_index
from polylogue.storage.repository import ConversationRepository
from tests.infra.storage_records import ConversationBuilder


@pytest.fixture
def filter_db(tmp_path):
    db_path = tmp_path / "filter_test.db"

    (
        ConversationBuilder(db_path, "claude-1")
        .provider("claude")
        .title("Python Error Handling")
        .created_at("2024-01-10T12:00:00+00:00")
        .updated_at("2024-01-10T12:30:00+00:00")
        .add_message("m1", role="user", text="How do I handle errors in Python?")
        .add_message("m2", role="assistant", text="You can use try except blocks.")
        .metadata({"tags": ["python", "errors"], "summary": "Python error handling"})
        .save()
    )
    (
        ConversationBuilder(db_path, "chatgpt-1")
        .provider("chatgpt")
        .title("JavaScript Async")
        .created_at("2024-02-14T09:00:00+00:00")
        .updated_at("2024-02-14T09:30:00+00:00")
        .add_message("m3", role="user", text="How do async functions work?")
        .add_message("m4", role="assistant", text="Async functions return promises.")
        .metadata({"tags": ["javascript", "async"]})
        .save()
    )
    (
        ConversationBuilder(db_path, "claude-2")
        .provider("claude")
        .title("Database Design")
        .created_at("2024-03-05T15:00:00+00:00")
        .updated_at("2024-03-05T16:00:00+00:00")
        .add_message("m5", role="user", text="How to design a database schema?")
        .add_message("m6", role="assistant", text="Start with identifying entities and relationships.")
        .metadata({"tags": ["database", "design"], "summary": "Database schema design"})
        .save()
    )
    (
        ConversationBuilder(db_path, "codex-1")
        .provider("codex")
        .title("File Search Notes")
        .created_at("2024-04-01T08:00:00+00:00")
        .updated_at("2024-04-01T08:15:00+00:00")
        .add_message("m7", role="user", text="Show grep examples for searching logs")
        .metadata({"tags": ["search", "cli"]})
        .save()
    )

    with open_connection(db_path) as conn:
        rebuild_index(conn)

    return db_path


@pytest.fixture
def filter_repo(filter_db):
    return ConversationRepository(backend=SQLiteBackend(db_path=filter_db))


@pytest.fixture
def filter_repo_pick_many(tmp_path):
    db_path = tmp_path / "filter_pick_many.db"
    with open_connection(db_path) as conn:
        rebuild_index(conn)

    for index in range(25):
        (
            ConversationBuilder(db_path, f"pick-{index}")
            .provider("claude")
            .title(f"Conversation {index}")
            .created_at(f"2024-01-{index + 1:02d}T00:00:00+00:00")
            .save()
        )

    return ConversationRepository(SQLiteBackend(db_path=db_path))


CHAIN_CALLS = [
    lambda f: f.provider("claude"),
    lambda f: f.exclude_provider("codex"),
    lambda f: f.tag("python"),
    lambda f: f.exclude_tag("javascript"),
    lambda f: f.has("summary"),
    lambda f: f.since("2024-01-01"),
    lambda f: f.until("2024-12-31"),
    lambda f: f.title("python"),
    lambda f: f.id("claude"),
    lambda f: f.contains("error"),
    lambda f: f.exclude_text("async"),
    lambda f: f.sort("date"),
    lambda f: f.reverse(),
    lambda f: f.limit(10),
    lambda f: f.sample(2),
    lambda f: f.similar("vector query"),
    lambda f: f.where(lambda c: c.provider == "claude"),
    lambda f: f.is_root(),
    lambda f: f.is_continuation(False),
    lambda f: f.is_sidechain(False),
    lambda f: f.parent("root"),
    lambda f: f.has_branches(),
    lambda f: f.has_tool_use(),
    lambda f: f.has_thinking(),
    lambda f: f.min_messages(1),
    lambda f: f.max_messages(10),
    lambda f: f.min_words(5),
    lambda f: f.has_file_operations(),
    lambda f: f.has_git_operations(),
    lambda f: f.has_subagent_spawns(),
]


FILTER_CASES = [
    ("provider", lambda f: f.provider("claude"), {"claude-1", "claude-2"}),
    ("provider_multi", lambda f: f.provider("claude", "chatgpt"), {"claude-1", "claude-2", "chatgpt-1"}),
    ("exclude_provider", lambda f: f.exclude_provider("claude"), {"chatgpt-1", "codex-1"}),
    ("tag", lambda f: f.tag("python"), {"claude-1"}),
    ("exclude_tag", lambda f: f.exclude_tag("python"), {"chatgpt-1", "claude-2", "codex-1"}),
    ("title", lambda f: f.title("python"), {"claude-1"}),
    ("id_prefix", lambda f: f.id("claude"), {"claude-1", "claude-2"}),
    ("contains", lambda f: f.contains("async"), {"chatgpt-1"}),
    ("exclude_text", lambda f: f.exclude_text("database"), {"claude-1", "chatgpt-1", "codex-1"}),
    ("summary", lambda f: f.has("summary"), {"claude-1", "claude-2"}),
    ("since_datetime", lambda f: f.since(datetime(2024, 3, 1, tzinfo=timezone.utc)), {"claude-2", "codex-1"}),
    ("until_string", lambda f: f.until("2024-02-20"), {"claude-1", "chatgpt-1"}),
]


SUMMARY_COMPATIBILITY_CASES = [
    (lambda f: f.provider("claude"), True),
    (lambda f: f.tag("python"), True),
    (lambda f: f.has("summary"), True),
    (lambda f: f.has("thinking"), False),
    (lambda f: f.exclude_text("error"), False),
    (lambda f: f.where(lambda c: True), False),
    (lambda f: f.sort("words"), False),
]


PLANNING_CASES = [
    (
        "summary-safe provider filter",
        lambda repo: ConversationFilter(repo).provider("claude").limit(1),
        {
            "summary_safe": True,
            "needs_content": False,
            "has_post_filters": False,
            "fetch_limit": 2,
            "sql": {"provider": "claude"},
        },
    ),
    (
        "negative text requires content",
        lambda repo: ConversationFilter(repo).exclude_text("error").limit(10),
        {
            "summary_safe": False,
            "needs_content": True,
            "has_post_filters": True,
            "fetch_limit": 500,
            "sql": {},
        },
    ),
    (
        "post filtered tags over-fetch",
        lambda repo: ConversationFilter(repo).tag("science").limit(10),
        {
            "summary_safe": True,
            "needs_content": False,
            "has_post_filters": True,
            "fetch_limit": 500,
            "sql": {},
        },
    ),
    (
        "sample widens fetch budget",
        lambda repo: ConversationFilter(repo).provider("claude").limit(10).sample(5),
        {
            "summary_safe": True,
            "needs_content": False,
            "has_post_filters": False,
            "fetch_limit": 200,
            "sql": {"provider": "claude"},
        },
    ),
    (
        "stats filters push to sql",
        lambda repo: ConversationFilter(repo)
        .provider("claude")
        .since(datetime(2024, 1, 1, tzinfo=timezone.utc))
        .until(datetime(2024, 12, 31, tzinfo=timezone.utc))
        .title("Alpha")
        .has_tool_use()
        .has_thinking()
        .min_messages(2)
        .max_messages(8)
        .min_words(5)
        .has_file_operations()
        .has_git_operations()
        .has_subagent_spawns(),
        {
            "summary_safe": True,
            "needs_content": False,
            "has_post_filters": False,
            "fetch_limit": None,
            "sql": {
                "provider": "claude",
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
            },
        },
    ),
]


class TestConversationFilterFoundations:
    def test_chainable_methods_return_self(self, filter_repo):
        for call in CHAIN_CALLS:
            fresh = ConversationFilter(filter_repo)
            assert call(fresh) is fresh

    @pytest.mark.parametrize(("case_name", "apply_filter", "expected_ids"), FILTER_CASES)
    @pytest.mark.asyncio
    async def test_selection_matrix(self, filter_repo, case_name, apply_filter, expected_ids):
        result = await apply_filter(ConversationFilter(filter_repo)).list()
        assert {conversation.id for conversation in result} == expected_ids, case_name

    @pytest.mark.asyncio
    async def test_custom_predicate_contract(self, filter_repo):
        result = await ConversationFilter(filter_repo).where(lambda c: c.provider == "codex").list()
        assert [conversation.id for conversation in result] == ["codex-1"]

    def test_date_methods_parse_and_store_values(self, filter_repo):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        filter_obj = ConversationFilter(filter_repo).since(start).until(end)
        assert filter_obj._since_date == start
        assert filter_obj._until_date == end

    def test_invalid_date_string_raises_value_error(self, filter_repo):
        with pytest.raises(ValueError, match="Cannot parse date"):
            ConversationFilter(filter_repo).since("not-a-date")

    def test_pick_index_contract(self) -> None:
        assert ConversationFilter._pick_index("", 3) == 0
        assert ConversationFilter._pick_index("1", 3) == 0
        assert ConversationFilter._pick_index("3", 3) == 2
        assert ConversationFilter._pick_index("0", 3) is None
        assert ConversationFilter._pick_index("4", 3) is None
        assert ConversationFilter._pick_index("oops", 3) is None

    @pytest.mark.asyncio
    async def test_terminal_methods_contract(self, filter_repo):
        listed = await ConversationFilter(filter_repo).list()
        first = await ConversationFilter(filter_repo).sort("date").first()
        count = await ConversationFilter(filter_repo).count()

        assert len(listed) == 4
        assert first is not None
        assert count == 4

    @pytest.mark.asyncio
    async def test_first_returns_none_for_empty_result(self, filter_repo):
        assert await ConversationFilter(filter_repo).provider("nonexistent").first() is None

    @pytest.mark.asyncio
    async def test_sort_and_reverse_contract(self, filter_repo):
        default_order = await ConversationFilter(filter_repo).sort("date").list()
        reversed_order = await ConversationFilter(filter_repo).sort("date").reverse().list()
        assert [conversation.id for conversation in default_order] == ["codex-1", "claude-2", "chatgpt-1", "claude-1"]
        assert [conversation.id for conversation in reversed_order] == ["claude-1", "chatgpt-1", "claude-2", "codex-1"]

    @pytest.mark.asyncio
    async def test_delete_execution_paths(self, filter_repo):
        summary_filter = ConversationFilter(filter_repo).provider("claude").limit(1)
        summary_filter.list_summaries = AsyncMock(return_value=[ConversationSummary(id="claude-1", provider="claude")])  # type: ignore[method-assign]
        summary_filter.list = AsyncMock(side_effect=AssertionError("full conversations should not be loaded"))  # type: ignore[method-assign]
        filter_repo.backend.delete_conversation = AsyncMock(return_value=True)  # type: ignore[method-assign]

        deleted = await summary_filter.delete()

        assert deleted == 1
        summary_filter.list_summaries.assert_awaited_once()
        summary_filter.list.assert_not_called()
        filter_repo.backend.delete_conversation.assert_awaited_once_with("claude-1")

        content_filter = ConversationFilter(filter_repo).exclude_text("database")
        content_filter.list_summaries = AsyncMock(side_effect=AssertionError("summary path should not run"))  # type: ignore[method-assign]
        content_filter.list = AsyncMock(return_value=[Conversation(id="chatgpt-1", provider="chatgpt", messages=[])])  # type: ignore[method-assign]
        filter_repo.backend.delete_conversation = AsyncMock(return_value=True)  # type: ignore[method-assign]

        deleted = await content_filter.delete()

        assert deleted == 1
        content_filter.list.assert_awaited_once()
        content_filter.list_summaries.assert_not_called()
        filter_repo.backend.delete_conversation.assert_awaited_once_with("chatgpt-1")

    @pytest.mark.asyncio
    async def test_pick_contract_matrix(self, filter_repo_pick_many, monkeypatch, capsys):
        assert await ConversationFilter(filter_repo_pick_many).provider("nonexistent").pick() is None

        with patch("sys.stdout.isatty", return_value=False):
            selected = await ConversationFilter(filter_repo_pick_many).sort("date").pick()
        assert selected is not None and selected.id == "pick-24"

        with patch("sys.stdout.isatty", return_value=True), patch("builtins.input", return_value="2"):
            selected = await ConversationFilter(filter_repo_pick_many).sort("date").pick()
        assert selected is not None and selected.id == "pick-23"

        with patch("sys.stdout.isatty", return_value=True), patch("builtins.input", return_value=""):
            first = await ConversationFilter(filter_repo_pick_many).sort("date").pick()
        assert first is not None and first.id == "pick-24"

        with patch("sys.stdout.isatty", return_value=True), patch("builtins.input", return_value="999"):
            assert await ConversationFilter(filter_repo_pick_many).sort("date").pick() is None

        with patch("sys.stdout.isatty", return_value=True), patch("builtins.input", return_value="oops"):
            assert await ConversationFilter(filter_repo_pick_many).sort("date").pick() is None

        with patch("sys.stdout.isatty", return_value=True), patch("builtins.input", side_effect=EOFError):
            assert await ConversationFilter(filter_repo_pick_many).sort("date").pick() is None

        output = capsys.readouterr().out
        numbered_lines = [line for line in output.splitlines() if re.match(r"\s+\d+\.\s", line)]
        assert len(numbered_lines) >= 20
        assert "... and 5 more" in output


class TestConversationFilterExecutionContracts:
    @pytest.mark.parametrize(("apply_filter", "can_use_summaries"), SUMMARY_COMPATIBILITY_CASES)
    def test_summary_compatibility_matrix(self, filter_repo, apply_filter, can_use_summaries):
        assert apply_filter(ConversationFilter(filter_repo)).can_use_summaries() is can_use_summaries

    @pytest.mark.asyncio
    async def test_list_summaries_basic_contract(self, filter_repo):
        summaries = await ConversationFilter(filter_repo).provider("claude").list_summaries()
        assert {summary.id for summary in summaries} == {"claude-1", "claude-2"}

    @pytest.mark.asyncio
    async def test_list_summaries_sort_contract(self, filter_repo):
        summaries = await ConversationFilter(filter_repo).provider("claude").sort("date").list_summaries()
        reversed_summaries = await ConversationFilter(filter_repo).provider("claude").sort("date").reverse().list_summaries()
        assert [summary.id for summary in summaries] == ["claude-2", "claude-1"]
        assert [summary.id for summary in reversed_summaries] == ["claude-1", "claude-2"]

    @pytest.mark.asyncio
    async def test_list_summaries_rejects_content_filters(self, filter_repo):
        with pytest.raises(ValueError, match="Cannot use list_summaries"):
            await ConversationFilter(filter_repo).has("thinking").list_summaries()

    @pytest.fixture
    def mock_repo(self):
        repo = Mock()
        repo.resolve_id = AsyncMock(return_value=None)
        repo.count = AsyncMock(return_value=7)
        repo.get_summary = AsyncMock(return_value=None)
        repo.search_summaries = AsyncMock(return_value=[])
        repo.list_summaries = AsyncMock(return_value=[])
        return repo

    @pytest.mark.parametrize(("case_name", "build_filter", "expected"), PLANNING_CASES)
    def test_execution_planning_matrix(self, mock_repo, case_name, build_filter, expected):
        filter_obj = build_filter(mock_repo)
        plan = filter_obj._build_execution_plan()
        assert plan.can_use_summaries is expected["summary_safe"], case_name
        assert plan.needs_content_loading is expected["needs_content"], case_name
        assert plan.has_post_filters is expected["has_post_filters"], case_name
        assert plan.fetch_limit == expected["fetch_limit"], case_name
        assert plan.sql_params == expected["sql"], case_name

    def test_apply_common_filters_contract(self, mock_repo):
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

    def test_describe_tracks_branch_filter(self, mock_repo):
        filter_obj = ConversationFilter(mock_repo).has_branches()
        assert filter_obj.describe() == ["custom predicates: 1"]

    @pytest.mark.asyncio
    async def test_fetch_generic_contracts(self, mock_repo):
        resolved_filter = ConversationFilter(mock_repo).id("conv-a")
        mock_repo.resolve_id = AsyncMock(return_value="conv-alpha")
        get_by_id = AsyncMock(return_value="resolved")
        search = AsyncMock()
        list_all = AsyncMock()

        result = await resolved_filter._fetch_generic(get_by_id, search, list_all)
        assert result == ["resolved"]
        get_by_id.assert_awaited_once_with("conv-alpha")
        search.assert_not_called()
        list_all.assert_not_called()

        search_filter = ConversationFilter(mock_repo).contains("needle").provider("claude").limit(10)
        get_by_id = AsyncMock()
        search = AsyncMock(return_value=["hit"])
        list_all = AsyncMock()

        result = await search_filter._fetch_generic(get_by_id, search, list_all)
        assert result == ["hit"]
        search.assert_awaited_once_with("needle", 100, search_filter._providers)
        list_all.assert_not_called()

        fallback_filter = ConversationFilter(mock_repo).contains("needle")
        get_by_id = AsyncMock()
        search = AsyncMock(side_effect=Exception("fts boom"))
        list_all = AsyncMock(return_value=["fallback"])

        result = await fallback_filter._fetch_generic(get_by_id, search, list_all)
        assert result == ["fallback"]
        list_all.assert_awaited_once_with(limit=None)

    @pytest.mark.asyncio
    async def test_summary_fetch_and_count_contracts(self, mock_repo):
        filter_obj = ConversationFilter(mock_repo).contains("needle").provider("claude").limit(10)
        mock_repo.search_summaries = AsyncMock(return_value=["summary-hit"])

        result = await filter_obj._fetch_summary_candidates()
        assert result == ["summary-hit"]
        mock_repo.search_summaries.assert_awaited_once_with("needle", limit=100, providers=filter_obj._providers)

    def test_execute_pipeline_applies_filter_sort_sample_and_limit(self, mock_repo):
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
    async def test_list_and_count_helper_paths(self, mock_repo):
        full_filter = ConversationFilter(mock_repo)
        summary_filter = ConversationFilter(mock_repo)
        conversations = [Conversation(id="conv-1", provider="claude", messages=[Message(id="m1", role="user", text="question")])]
        summaries = [ConversationSummary(id="conv-1", provider="claude")]
        full_filter._fetch_candidates = AsyncMock(return_value=conversations)  # type: ignore[method-assign]
        summary_filter._fetch_summary_candidates = AsyncMock(return_value=summaries)  # type: ignore[method-assign]

        assert [conv.id for conv in await full_filter.list()] == ["conv-1"]
        assert [summary.id for summary in await summary_filter.list_summaries()] == ["conv-1"]

        fast = ConversationFilter(mock_repo).provider("claude")
        assert await fast.count() == 7
        assert fast._can_count_in_sql() is True
        mock_repo.count.assert_awaited_once_with(provider=fast._providers[0])

        summary_count = ConversationFilter(mock_repo).tag("python")
        summary_count.list_summaries = AsyncMock(return_value=[ConversationSummary(id="one", provider="claude")])  # type: ignore[method-assign]
        assert await summary_count.count() == 1
        assert summary_count._can_count_in_sql() is False
        summary_count.list_summaries.assert_awaited_once()

        content_count = ConversationFilter(mock_repo).exclude_text("secret")
        content_count.list = AsyncMock(return_value=[Conversation(id="one", provider="claude", messages=[])])  # type: ignore[method-assign]
        content_count.list_summaries = AsyncMock(side_effect=AssertionError("summary path should not run"))  # type: ignore[method-assign]
        assert await content_count.count() == 1
        content_count.list.assert_awaited_once()
