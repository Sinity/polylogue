"""Law-based contracts for ConversationRepository."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.lib.models import Conversation
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository, _records_to_conversation
from polylogue.storage.store import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
)
from tests.infra.helpers import make_message
from tests.infra.strategies import (
    ConversationSpec,
    MessageSpec,
    conversation_graph_strategy,
    expected_sorted_ids,
    expected_tree_ids,
    root_index,
    seed_conversation_graph,
    shortest_unique_prefix,
)


def _seed_repo(specs) -> tuple[TemporaryDirectory[str], ConversationRepository]:
    tempdir = TemporaryDirectory()
    db_path = Path(tempdir.name) / "repo.db"
    seed_conversation_graph(db_path, specs)
    backend = SQLiteBackend(db_path=db_path)
    return tempdir, ConversationRepository(backend=backend)


def _empty_repo() -> tuple[TemporaryDirectory[str], ConversationRepository]:
    tempdir = TemporaryDirectory()
    db_path = Path(tempdir.name) / "repo.db"
    backend = SQLiteBackend(db_path=db_path)
    return tempdir, ConversationRepository(backend=backend)


async def _collect_summary_pages(repo: ConversationRepository, page_size: int) -> list[str]:
    pages = [page async for page in repo.iter_summary_pages(page_size=page_size)]
    return [summary.id for page in pages for summary in page]


def _expected_archive_stats(specs) -> tuple[int, int, dict[str, int]]:
    provider_counts: dict[str, int] = {}
    total_messages = 0
    for spec in specs:
        provider_counts[spec.provider] = provider_counts.get(spec.provider, 0) + 1
        total_messages += len(spec.messages)
    return len(specs), total_messages, provider_counts


class _Cursor:
    def __init__(self, *, one=None, rows=None) -> None:
        self._one = one
        self._rows = rows or []

    async def fetchone(self):
        return self._one

    async def fetchall(self):
        return self._rows


class _Connection:
    def __init__(self, cursors: list[_Cursor]) -> None:
        self.calls: list[tuple[str, object]] = []
        self._cursors = cursors

    async def execute(self, query: str, params=()):
        self.calls.append((query, params))
        return self._cursors.pop(0)


@asynccontextmanager
async def _context(value):
    yield value


@settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(conversation_graph_strategy())
def test_repository_views_agree_on_generated_graph(specs) -> None:
    """count(), list_summaries(), and list() must describe the same visible archive."""
    tempdir, repo = _seed_repo(specs)
    try:
        expected_ids = expected_sorted_ids(specs)
        count = asyncio.run(repo.count())
        summaries = asyncio.run(repo.list_summaries(limit=None))
        conversations = asyncio.run(repo.list(limit=None))

        assert count == len(specs)
        assert [summary.id for summary in summaries] == expected_ids
        assert [str(conversation.id) for conversation in conversations] == expected_ids
        assert len(summaries) == len(conversations) == count
    finally:
        asyncio.run(repo.close())
        tempdir.cleanup()


@settings(
    deadline=None,
    max_examples=25,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(conversation_graph_strategy(), st.integers(min_value=1, max_value=4))
def test_repository_limited_summaries_are_prefix_of_full_order(specs, page_size: int) -> None:
    """Pagination and explicit limits must preserve backend ordering exactly."""
    tempdir, repo = _seed_repo(specs)
    try:
        full_ids = expected_sorted_ids(specs)
        page_ids = asyncio.run(_collect_summary_pages(repo, page_size))
        limit_ids = [summary.id for summary in asyncio.run(repo.list_summaries(limit=page_size))]

        assert page_ids == full_ids
        assert limit_ids == full_ids[:page_size]
    finally:
        asyncio.run(repo.close())
        tempdir.cleanup()


@settings(
    deadline=None,
    max_examples=25,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(conversation_graph_strategy())
def test_repository_provider_filters_are_subset_and_count_consistent(specs) -> None:
    """Filtering by provider must remain subset-preserving across repository views."""
    tempdir, repo = _seed_repo(specs)
    try:
        all_providers = sorted({spec.provider for spec in specs})
        for provider in all_providers:
            expected_ids = expected_sorted_ids(tuple(spec for spec in specs if spec.provider == provider))
            count = asyncio.run(repo.count(provider=provider))
            summaries = asyncio.run(repo.list_summaries(limit=None, provider=provider))
            conversations = asyncio.run(repo.list(limit=None, provider=provider))

            assert count == len(expected_ids)
            assert [summary.id for summary in summaries] == expected_ids
            assert [str(conversation.id) for conversation in conversations] == expected_ids
            assert set(expected_ids).issubset(set(expected_sorted_ids(specs)))
    finally:
        asyncio.run(repo.close())
        tempdir.cleanup()


@settings(
    deadline=None,
    max_examples=25,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(conversation_graph_strategy())
def test_repository_provider_id_lookup_matches_generated_graph(specs) -> None:
    """Provider-ID lookup must preserve archive sort order within each provider subset."""
    tempdir, repo = _seed_repo(specs)
    try:
        for provider in sorted({spec.provider for spec in specs}):
            expected_ids = expected_sorted_ids(tuple(spec for spec in specs if spec.provider == provider))
            assert asyncio.run(repo.get_provider_conversation_ids(provider)) == expected_ids
    finally:
        asyncio.run(repo.close())
        tempdir.cleanup()


@settings(
    deadline=None,
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(conversation_graph_strategy())
def test_repository_tree_methods_preserve_root_and_closure(specs) -> None:
    """get_root() and get_session_tree() must agree with the generated parent graph."""
    tempdir, repo = _seed_repo(specs)
    try:
        for index, spec in enumerate(specs):
            root = asyncio.run(repo.get_root(spec.conversation_id))
            tree = asyncio.run(repo.get_session_tree(spec.conversation_id))

            expected_root = specs[root_index(specs, index)].conversation_id
            assert str(root.id) == expected_root
            assert {str(conversation.id) for conversation in tree} == expected_tree_ids(specs, index)
    finally:
        asyncio.run(repo.close())
        tempdir.cleanup()


def test_repository_get_children_hydrates_loaded_child_records() -> None:
    """get_children() must hydrate directly from listed child records without refetching IDs."""
    backend = MagicMock()
    backend.list_conversations = AsyncMock(
        return_value=[
            ConversationRecord(
                conversation_id="conv-child",
                provider_name="claude",
                provider_conversation_id="child",
                title="Child",
                content_hash="hash-child",
                provider_meta={},
                metadata={},
            )
        ]
    )
    backend.get_messages_batch = AsyncMock(return_value={"conv-child": []})
    backend.get_attachments_batch = AsyncMock(return_value={"conv-child": []})
    backend.get_conversations_batch = AsyncMock(
        side_effect=AssertionError("get_children() should not refetch child records by ID")
    )
    repo = ConversationRepository(backend=backend)

    children = asyncio.run(repo.get_children("conv-root"))

    assert [str(child.id) for child in children] == ["conv-child"]
    backend.list_conversations.assert_awaited_once_with(parent_id="conv-root")
    backend.get_messages_batch.assert_awaited_once_with(["conv-child"])
    backend.get_attachments_batch.assert_awaited_once_with(["conv-child"])


@settings(
    deadline=None,
    max_examples=25,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(conversation_graph_strategy(), st.integers(min_value=0, max_value=5))
def test_repository_lookup_views_and_projection_agree_on_generated_id(specs, candidate_index: int) -> None:
    """All read-model entry points must agree on IDs, message counts, and projection shape."""
    tempdir, repo = _seed_repo(specs)
    try:
        index = candidate_index % len(specs)
        target = specs[index]
        ids = tuple(spec.conversation_id for spec in specs)
        prefix = shortest_unique_prefix(ids, target.conversation_id)

        summary = asyncio.run(repo.get_summary(target.conversation_id))
        conversation = asyncio.run(repo.get(target.conversation_id))
        eager = asyncio.run(repo.get_eager(target.conversation_id))
        viewed = asyncio.run(repo.view(prefix))
        projection = asyncio.run(repo.get_render_projection(target.conversation_id))

        assert summary is not None
        assert conversation is not None
        assert eager is not None
        assert viewed is not None
        assert projection is not None

        assert summary.id == target.conversation_id
        assert str(conversation.id) == target.conversation_id
        assert str(eager.id) == target.conversation_id
        assert str(viewed.id) == target.conversation_id
        assert projection.conversation.conversation_id == target.conversation_id
        assert len(conversation.messages) == len(target.messages)
        assert len(eager.messages) == len(target.messages)
        assert len(viewed.messages) == len(target.messages)
        assert len(projection.messages) == len(target.messages)
        assert projection.attachments == []
    finally:
        asyncio.run(repo.close())
        tempdir.cleanup()


@settings(
    deadline=None,
    max_examples=25,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    conversation_graph_strategy(),
    st.integers(min_value=0, max_value=5),
    st.integers(min_value=1, max_value=4),
)
def test_repository_iter_messages_and_stats_match_generated_conversation(
    specs,
    candidate_index: int,
    limit: int,
) -> None:
    """iter_messages() windows and conversation stats must agree with the generated source graph."""
    tempdir, repo = _seed_repo(specs)
    try:
        index = candidate_index % len(specs)
        target = specs[index]

        all_messages = asyncio.run(
            _collect_messages(repo.iter_messages(target.conversation_id))
        )
        limited_messages = asyncio.run(
            _collect_messages(repo.iter_messages(target.conversation_id, limit=limit))
        )
        stats = asyncio.run(repo.get_conversation_stats(target.conversation_id))

        assert len(all_messages) == len(target.messages)
        assert len(limited_messages) == min(limit, len(target.messages))
        assert stats is not None
        assert stats["total_messages"] == len(target.messages)
    finally:
        asyncio.run(repo.close())
        tempdir.cleanup()


@settings(
    deadline=None,
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(conversation_graph_strategy())
def test_repository_batch_message_counts_and_aggregate_stats_match_generated_graph(specs) -> None:
    """Batch counts and aggregate stats must agree with the generated graph."""
    tempdir, repo = _seed_repo(specs)
    try:
        ids = [spec.conversation_id for spec in specs]
        expected_counts = {spec.conversation_id: len(spec.messages) for spec in specs}
        expected_providers: dict[str, int] = {}
        for spec in specs:
            expected_providers[spec.provider] = expected_providers.get(spec.provider, 0) + 1

        assert asyncio.run(repo.get_message_counts_batch(ids)) == expected_counts

        stats = asyncio.run(repo.aggregate_message_stats(ids))
        assert stats["total"] == sum(expected_counts.values())
        assert stats["attachments"] == 0
        assert stats["providers"] == expected_providers
        assert stats["min_sort_key"] is not None
        assert stats["max_sort_key"] is not None
    finally:
        asyncio.run(repo.close())
        tempdir.cleanup()


async def _collect_messages(iterator) -> list:
    return [message async for message in iterator]


@settings(
    deadline=None,
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(conversation_graph_strategy())
def test_repository_archive_stats_match_generated_graph(specs) -> None:
    """Archive statistics must match the generated graph exactly."""
    tempdir, repo = _seed_repo(specs)
    try:
        stats = asyncio.run(repo.get_archive_stats())
        expected_conversations, expected_messages, expected_providers = _expected_archive_stats(specs)

        assert stats.total_conversations == expected_conversations
        assert stats.total_messages == expected_messages
        assert stats.provider_count == len(expected_providers)
        assert stats.providers == expected_providers
        assert stats.embedded_conversations == 0
        assert stats.embedded_messages == 0
        assert stats.embedding_coverage == 0.0
        assert stats.avg_messages_per_conversation == pytest.approx(
            expected_messages / expected_conversations
        )
        assert stats.db_size_bytes >= 0
    finally:
        asyncio.run(repo.close())
        tempdir.cleanup()


@settings(
    deadline=None,
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(conversation_graph_strategy())
def test_repository_get_parent_matches_generated_graph(specs) -> None:
    """get_parent() must return the direct generated parent or None."""
    tempdir, repo = _seed_repo(specs)
    try:
        for _index, spec in enumerate(specs):
            parent = asyncio.run(repo.get_parent(spec.conversation_id))
            expected_parent = None if spec.parent_index is None else specs[spec.parent_index].conversation_id
            if expected_parent is None:
                assert parent is None
            else:
                assert parent is not None
                assert str(parent.id) == expected_parent
    finally:
        asyncio.run(repo.close())
        tempdir.cleanup()


@settings(
    deadline=None,
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(conversation_graph_strategy())
def test_repository_stats_by_provider_match_generated_graph(specs) -> None:
    """get_stats_by(provider) must equal the provider histogram of the graph."""
    tempdir, repo = _seed_repo(specs)
    try:
        expected: dict[str, int] = {}
        for spec in specs:
            expected[spec.provider] = expected.get(spec.provider, 0) + 1

        assert asyncio.run(repo.get_stats_by("provider")) == expected
    finally:
        asyncio.run(repo.close())
        tempdir.cleanup()


def test_repository_search_views_agree_on_ranked_ids() -> None:
    """search() and search_summaries() must return the same ranked conversations."""
    specs = (
        ConversationSpec(
            conversation_id="conv-alpha",
            provider="claude",
            title="Alpha",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            parent_index=None,
            messages=(MessageSpec(role="user", text="needle alpha", has_tool_use=False, has_thinking=False),),
        ),
        ConversationSpec(
            conversation_id="conv-beta",
            provider="chatgpt",
            title="Beta",
            created_at="2024-01-02T00:00:00+00:00",
            updated_at="2024-01-02T00:00:00+00:00",
            parent_index=None,
            messages=(MessageSpec(role="assistant", text="needle beta", has_tool_use=False, has_thinking=False),),
        ),
        ConversationSpec(
            conversation_id="conv-gamma",
            provider="codex",
            title="Gamma",
            created_at="2024-01-03T00:00:00+00:00",
            updated_at="2024-01-03T00:00:00+00:00",
            parent_index=None,
            messages=(MessageSpec(role="assistant", text="unrelated", has_tool_use=False, has_thinking=False),),
        ),
    )
    tempdir, repo = _seed_repo(specs)
    try:
        summaries = asyncio.run(repo.search_summaries("needle", limit=5))
        conversations = asyncio.run(repo.search("needle", limit=5))

        assert [summary.id for summary in summaries] == ["conv-beta", "conv-alpha"]
        assert [str(conversation.id) for conversation in conversations] == ["conv-beta", "conv-alpha"]
    finally:
        asyncio.run(repo.close())
        tempdir.cleanup()


def test_repository_stats_by_calendar_buckets_match_generated_graph() -> None:
    """get_stats_by(month/year) must respect stored updated_at buckets."""
    specs = (
        ConversationSpec(
            conversation_id="conv-jan-1",
            provider="claude",
            title="Jan One",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            parent_index=None,
            messages=(MessageSpec(role="user", text="a", has_tool_use=False, has_thinking=False),),
        ),
        ConversationSpec(
            conversation_id="conv-jan-2",
            provider="chatgpt",
            title="Jan Two",
            created_at="2024-01-15T00:00:00+00:00",
            updated_at="2024-01-15T00:00:00+00:00",
            parent_index=None,
            messages=(MessageSpec(role="assistant", text="b", has_tool_use=False, has_thinking=False),),
        ),
        ConversationSpec(
            conversation_id="conv-feb",
            provider="codex",
            title="Feb",
            created_at="2024-02-01T00:00:00+00:00",
            updated_at="2024-02-01T00:00:00+00:00",
            parent_index=None,
            messages=(MessageSpec(role="assistant", text="c", has_tool_use=False, has_thinking=False),),
        ),
    )
    tempdir, repo = _seed_repo(specs)
    try:
        assert asyncio.run(repo.get_stats_by("year")) == {"2024": 3}
        assert asyncio.run(repo.get_stats_by("month")) == {"2024-01": 2, "2024-02": 1}
        assert asyncio.run(repo.get_stats_by("provider")) == {"chatgpt": 1, "claude": 1, "codex": 1}
    finally:
        asyncio.run(repo.close())
        tempdir.cleanup()


def test_repository_iter_summary_pages_partition_is_stable_across_page_sizes() -> None:
    """Concatenated pages must stay equal to the full summary order for different page sizes."""
    specs = (
        ConversationSpec(
            conversation_id="conv-1",
            provider="claude",
            title="One",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            parent_index=None,
            messages=(MessageSpec(role="user", text="1", has_tool_use=False, has_thinking=False),),
        ),
        ConversationSpec(
            conversation_id="conv-2",
            provider="chatgpt",
            title="Two",
            created_at="2024-01-02T00:00:00+00:00",
            updated_at="2024-01-02T00:00:00+00:00",
            parent_index=None,
            messages=(MessageSpec(role="assistant", text="2", has_tool_use=False, has_thinking=False),),
        ),
        ConversationSpec(
            conversation_id="conv-3",
            provider="codex",
            title="Three",
            created_at="2024-01-03T00:00:00+00:00",
            updated_at="2024-01-03T00:00:00+00:00",
            parent_index=None,
            messages=(MessageSpec(role="assistant", text="3", has_tool_use=False, has_thinking=False),),
        ),
    )
    tempdir, repo = _seed_repo(specs)
    try:
        full_ids = [summary.id for summary in asyncio.run(repo.list_summaries(limit=None))]
        for page_size in (1, 2, 5):
            page_ids = asyncio.run(_collect_summary_pages(repo, page_size))
            assert page_ids == full_ids
    finally:
        asyncio.run(repo.close())
        tempdir.cleanup()


def test_repository_save_conversation_model_contract() -> None:
    """Saving a Conversation model strips provider prefixes and is idempotent on hashes."""
    tempdir, repo = _empty_repo()
    try:
        conversation = Conversation(
            id="claude:thread-1",
            provider="claude",
            title="Saved Conversation",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            metadata={
                "content_hash": "conv-hash",
                "provider_meta": {"source": "law"},
                "custom": "value",
            },
            messages=[],
        )
        message = make_message(
            "msg-thread-1",
            "claude:thread-1",
            role="user",
            text="hello",
            content_hash="msg-hash",
        )

        first = asyncio.run(repo.save_conversation(conversation, [message], []))
        second = asyncio.run(repo.save_conversation(conversation, [message], []))
        stored = asyncio.run(repo.backend.get_conversation("claude:thread-1"))

        assert first["conversations"] == 1
        assert first["messages"] == 1
        assert second["skipped_conversations"] == 1
        assert second["skipped_messages"] == 1
        assert stored is not None
        assert stored.provider_conversation_id == "thread-1"
        assert stored.provider_meta == {"source": "law"}
        assert stored.metadata["custom"] == "value"
    finally:
        asyncio.run(repo.close())
        tempdir.cleanup()


def test_repository_list_count_and_summary_filters_forward_canonically() -> None:
    """List/count entry points must forward the exact same filter semantics."""
    records = [
        ConversationRecord(
            conversation_id="claude:conv-1",
            provider_name="claude",
            provider_conversation_id="conv-1",
            title="One",
            updated_at="2024-01-02T00:00:00+00:00",
            created_at="2024-01-01T00:00:00+00:00",
            content_hash="hash-1",
            provider_meta={},
            metadata={},
        )
    ]
    backend = MagicMock()
    backend.list_conversations = AsyncMock(return_value=records)
    backend.count_conversations = AsyncMock(return_value=7)
    repo = ConversationRepository(backend=backend)
    expected_conversations = [Conversation(id="claude:conv-1", provider="claude", title="One", messages=[])]
    repo._hydrate_conversations = AsyncMock(return_value=expected_conversations)

    conversations = asyncio.run(
        repo.list(
            limit=3,
            offset=2,
            provider="claude",
            providers=["claude", "codex"],
            since="2024-01-01",
            until="2024-01-31",
            title_contains="One",
            has_tool_use=True,
            has_thinking=True,
            min_messages=2,
            max_messages=5,
            min_words=10,
            has_file_ops=True,
            has_git_ops=True,
            has_subagent=True,
        )
    )
    summaries = asyncio.run(
        repo.list_summaries(
            limit=3,
            offset=2,
            provider="claude",
            providers=["claude", "codex"],
            source="archive",
            since="2024-01-01",
            until="2024-01-31",
            title_contains="One",
            has_tool_use=True,
            has_thinking=True,
            min_messages=2,
            max_messages=5,
            min_words=10,
            has_file_ops=True,
            has_git_ops=True,
            has_subagent=True,
        )
    )
    count = asyncio.run(
        repo.count(
            provider="claude",
            providers=["claude", "codex"],
            since="2024-01-01",
            until="2024-01-31",
            title_contains="One",
            has_tool_use=True,
            has_thinking=True,
            min_messages=2,
            max_messages=5,
            min_words=10,
            has_file_ops=True,
            has_git_ops=True,
            has_subagent=True,
        )
    )

    assert conversations == expected_conversations
    assert [summary.id for summary in summaries] == ["claude:conv-1"]
    assert count == 7
    assert backend.list_conversations.await_args_list[0].kwargs == {
        "limit": 3,
        "offset": 2,
        "provider": "claude",
        "providers": ["claude", "codex"],
        "source": None,
        "since": "2024-01-01",
        "until": "2024-01-31",
        "title_contains": "One",
        "has_tool_use": True,
        "has_thinking": True,
        "min_messages": 2,
        "max_messages": 5,
        "min_words": 10,
        "has_file_ops": True,
        "has_git_ops": True,
        "has_subagent": True,
    }
    assert backend.list_conversations.await_args_list[1].kwargs == {
        "limit": 3,
        "offset": 2,
        "provider": "claude",
        "providers": ["claude", "codex"],
        "source": "archive",
        "since": "2024-01-01",
        "until": "2024-01-31",
        "title_contains": "One",
        "has_tool_use": True,
        "has_thinking": True,
        "min_messages": 2,
        "max_messages": 5,
        "min_words": 10,
        "has_file_ops": True,
        "has_git_ops": True,
        "has_subagent": True,
    }
    assert backend.count_conversations.await_args.kwargs == {
        "provider": "claude",
        "providers": ["claude", "codex"],
        "since": "2024-01-01",
        "until": "2024-01-31",
        "title_contains": "One",
        "has_tool_use": True,
        "has_thinking": True,
        "min_messages": 2,
        "max_messages": 5,
        "min_words": 10,
        "has_file_ops": True,
        "has_git_ops": True,
        "has_subagent": True,
    }
    repo._hydrate_conversations.assert_awaited_once_with(records)


def test_repository_iter_summary_pages_stops_after_short_page() -> None:
    """Summary pagination must advance offsets and stop after the first short page."""
    backend = MagicMock()
    repo = ConversationRepository(backend=backend)
    repo.list_summaries = AsyncMock(
        side_effect=[
            [MagicMock(id="conv-1"), MagicMock(id="conv-2")],
            [MagicMock(id="conv-3")],
        ]
    )

    pages = asyncio.run(_collect_summary_pages(repo, 2))

    assert pages == ["conv-1", "conv-2", "conv-3"]
    assert repo.list_summaries.await_args_list[0].kwargs["offset"] == 0
    assert repo.list_summaries.await_args_list[1].kwargs["offset"] == 2


def test_repository_iter_summary_pages_forwards_filters_and_advances_by_page_length() -> None:
    repo = ConversationRepository(backend=MagicMock())
    repo.list_summaries = AsyncMock(
        side_effect=[
            [MagicMock(id="conv-1"), MagicMock(id="conv-2")],
            [MagicMock(id="conv-3"), MagicMock(id="conv-4")],
            [],
        ]
    )

    async def _collect() -> list[str]:
        pages = [
            page
            async for page in repo.iter_summary_pages(
                page_size=2,
                provider="claude",
                providers=["claude", "codex"],
                source="archive",
                since="2024-01-01",
                until="2024-01-31",
                title_contains="needle",
                has_tool_use=True,
                has_thinking=True,
                min_messages=2,
                max_messages=5,
                min_words=10,
                has_file_ops=True,
                has_git_ops=True,
                has_subagent=True,
            )
        ]
        return [summary.id for page in pages for summary in page]

    assert asyncio.run(_collect()) == ["conv-1", "conv-2", "conv-3", "conv-4"]
    kwargs0 = repo.list_summaries.await_args_list[0].kwargs
    kwargs1 = repo.list_summaries.await_args_list[1].kwargs
    kwargs2 = repo.list_summaries.await_args_list[2].kwargs
    assert kwargs0 == {"limit": 2, "offset": 0, "provider": "claude", "providers": ["claude", "codex"], "source": "archive", "since": "2024-01-01", "until": "2024-01-31", "title_contains": "needle", "has_tool_use": True, "has_thinking": True, "min_messages": 2, "max_messages": 5, "min_words": 10, "has_file_ops": True, "has_git_ops": True, "has_subagent": True}
    assert kwargs1["offset"] == 2
    assert kwargs2["offset"] == 4


def test_repository_save_via_backend_skips_message_writes_when_hash_matches() -> None:
    """Unchanged content still upserts the conversation row but skips child writes."""
    connection = _Connection([_Cursor(one={"content_hash": "same-hash"})])
    backend = MagicMock()
    backend.connection = MagicMock(return_value=_context(connection))
    backend.transaction = MagicMock(return_value=_context(None))
    backend.save_conversation_record = AsyncMock()
    backend.save_messages = AsyncMock()
    backend.upsert_conversation_stats = AsyncMock()
    backend.save_content_blocks = AsyncMock()
    backend.prune_attachments = AsyncMock()
    backend.save_attachments = AsyncMock()
    repo = ConversationRepository(backend=backend)

    conversation = ConversationRecord(
        conversation_id="claude:conv-1",
        provider_name="claude",
        provider_conversation_id="conv-1",
        title="Title",
        content_hash="same-hash",
        provider_meta={},
        metadata={},
    )
    messages = [
        MessageRecord(
            message_id="msg-1",
            conversation_id="claude:conv-1",
            role="user",
            text="hello",
            content_hash="msg-hash",
        )
    ]
    attachments = [
        AttachmentRecord(
            attachment_id="att-1",
            conversation_id="claude:conv-1",
        )
    ]

    result = asyncio.run(repo._save_via_backend(conversation, messages, attachments))

    assert result == {
        "conversations": 0,
        "messages": 0,
        "attachments": 0,
        "skipped_conversations": 1,
        "skipped_messages": 1,
        "skipped_attachments": 1,
    }
    backend.save_conversation_record.assert_awaited_once_with(conversation)
    backend.save_messages.assert_not_awaited()
    backend.upsert_conversation_stats.assert_not_awaited()
    backend.save_content_blocks.assert_not_awaited()
    backend.prune_attachments.assert_not_awaited()
    backend.save_attachments.assert_not_awaited()


def test_repository_save_via_backend_propagates_provider_name_and_collects_blocks() -> None:
    """Fresh saves must backfill message provider names and persist explicit and message-owned blocks."""
    connection = _Connection([_Cursor(one=None)])
    backend = MagicMock()
    backend.connection = MagicMock(return_value=_context(connection))
    backend.transaction = MagicMock(return_value=_context(None))
    backend.save_conversation_record = AsyncMock()
    backend.save_messages = AsyncMock()
    backend.upsert_conversation_stats = AsyncMock()
    backend.save_content_blocks = AsyncMock()
    backend.prune_attachments = AsyncMock()
    backend.save_attachments = AsyncMock()
    repo = ConversationRepository(backend=backend)

    message_block = ContentBlockRecord(
        block_id="blk-msg",
        message_id="msg-1",
        conversation_id="claude:conv-1",
        block_index=0,
        type="code",
        text="print('ok')",
    )
    explicit_block = ContentBlockRecord(
        block_id="blk-explicit",
        message_id="msg-1",
        conversation_id="claude:conv-1",
        block_index=1,
        type="thinking",
        text="reason",
    )
    message = MessageRecord(
        message_id="msg-1",
        conversation_id="claude:conv-1",
        role="assistant",
        text="answer",
        content_hash="msg-hash",
        content_blocks=[message_block],
    )
    attachment = AttachmentRecord(
        attachment_id="att-1",
        conversation_id="claude:conv-1",
    )
    conversation = ConversationRecord(
        conversation_id="claude:conv-1",
        provider_name="claude",
        provider_conversation_id="conv-1",
        title="Title",
        content_hash="new-hash",
        provider_meta={},
        metadata={},
    )

    result = asyncio.run(repo._save_via_backend(conversation, [message], [attachment], [explicit_block]))

    assert result == {
        "conversations": 1,
        "messages": 1,
        "attachments": 1,
        "skipped_conversations": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
    }
    saved_messages = backend.save_messages.await_args.args[0]
    assert [saved.provider_name for saved in saved_messages] == ["claude"]
    backend.upsert_conversation_stats.assert_awaited_once()
    saved_blocks = backend.save_content_blocks.await_args.args[0]
    assert {block.block_id for block in saved_blocks} == {"blk-explicit", "blk-msg"}
    backend.prune_attachments.assert_awaited_once_with("claude:conv-1", {"att-1"})
    backend.save_attachments.assert_awaited_once_with([attachment])


def test_repository_search_and_similarity_contracts() -> None:
    """Search variants must preserve ranked IDs and similarity grouping semantics."""
    backend = MagicMock()
    backend.search_conversations = AsyncMock(return_value=["conv-b", "conv-a"])
    backend.get_conversations_batch = AsyncMock(
        return_value=[
            ConversationRecord(
                conversation_id="conv-a",
                provider_name="claude",
                provider_conversation_id="a",
                title="A",
                content_hash="hash-a",
                provider_meta={},
                metadata={},
            ),
            ConversationRecord(
                conversation_id="conv-b",
                provider_name="codex",
                provider_conversation_id="b",
                title="B",
                content_hash="hash-b",
                provider_meta={},
                metadata={},
            ),
        ]
    )
    repo = ConversationRepository(backend=backend)
    repo._hydrate_conversations = AsyncMock(return_value=[Conversation(id="conv-b", provider="codex", messages=[])])
    repo.get_many = AsyncMock(return_value=[Conversation(id="conv-a", provider="claude", messages=[]), Conversation(id="conv-b", provider="codex", messages=[])])
    repo._get_message_conversation_mapping = AsyncMock(
        return_value={"m1": "conv-a", "m2": "conv-a", "m3": "conv-b"}
    )
    vector_provider = MagicMock()
    vector_provider.query.return_value = [("m2", 0.8), ("m1", 0.2), ("m3", 0.5)]

    summaries = asyncio.run(repo.search_summaries("needle", limit=2, providers=["codex", "claude"]))
    conversations = asyncio.run(repo.search("needle", limit=2, providers=["codex", "claude"]))
    similar = asyncio.run(repo.search_similar("needle", limit=2, vector_provider=vector_provider))
    similar_rows = asyncio.run(repo.similarity_search("needle", limit=2, vector_provider=vector_provider))

    assert [summary.id for summary in summaries] == ["conv-b", "conv-a"]
    assert conversations == [Conversation(id="conv-b", provider="codex", messages=[])]
    backend.search_conversations.assert_any_await("needle", limit=2, providers=["codex", "claude"])
    assert backend.get_conversations_batch.await_count == 2
    assert all(call.args == (["conv-b", "conv-a"],) for call in backend.get_conversations_batch.await_args_list)
    repo._hydrate_conversations.assert_awaited_once()
    repo.get_many.assert_awaited_once_with(["conv-a", "conv-b"])
    assert [str(conv.id) for conv in similar] == ["conv-a", "conv-b"]
    assert similar_rows == [
        ("conv-a", "m2", 0.8),
        ("conv-a", "m1", 0.2),
        ("conv-b", "m3", 0.5),
    ]


def test_repository_get_archive_stats_aggregates_sql_results_and_db_size(tmp_path: Path) -> None:
    """Archive stats must reflect SQL counts, provider rows, embedding rows, and db size."""
    db_path = tmp_path / "repo.db"
    db_path.write_bytes(b"x" * 128)
    connection = _Connection(
        [
            _Cursor(one=(3,)),
            _Cursor(one=(8,)),
            _Cursor(one=(2,)),
            _Cursor(rows=[{"provider_name": "claude", "count": 2}, {"provider_name": "codex", "count": 1}]),
            _Cursor(one=(2,)),
            _Cursor(one=(5,)),
        ]
    )
    backend = MagicMock()
    backend.db_path = db_path
    backend.connection = MagicMock(return_value=_context(connection))
    repo = ConversationRepository(backend=backend)

    stats = asyncio.run(repo.get_archive_stats())

    assert stats.total_conversations == 3
    assert stats.total_messages == 8
    assert stats.total_attachments == 2
    assert stats.providers == {"claude": 2, "codex": 1}
    assert stats.provider_count == 2
    assert stats.embedded_conversations == 2
    assert stats.embedded_messages == 5
    assert stats.embedding_coverage == pytest.approx((2 / 3) * 100)
    assert stats.db_size_bytes == 128


def test_repository_get_archive_stats_tolerates_embedding_query_failure(tmp_path: Path) -> None:
    db_path = tmp_path / "repo.db"
    db_path.write_bytes(b"x" * 64)

    class _FailingConnection(_Connection):
        async def execute(self, query: str, params=()):
            if "embedding_status" in query:
                raise RuntimeError("embedding table missing")
            return await super().execute(query, params)

    connection = _FailingConnection(
        [
            _Cursor(one=(2,)),
            _Cursor(one=(5,)),
            _Cursor(one=(1,)),
            _Cursor(rows=[{"provider_name": "claude", "count": 2}]),
        ]
    )
    backend = MagicMock()
    backend.db_path = db_path
    backend.connection = MagicMock(return_value=_context(connection))
    repo = ConversationRepository(backend=backend)

    stats = asyncio.run(repo.get_archive_stats())

    assert stats.total_conversations == 2
    assert stats.total_messages == 5
    assert stats.total_attachments == 1
    assert stats.providers == {"claude": 2}
    assert stats.embedded_conversations == 0
    assert stats.embedded_messages == 0


def test_repository_save_conversation_unprefixed_provider_id_contract() -> None:
    """Saving a non-prefixed conversation ID preserves provider_conversation_id verbatim."""
    tempdir, repo = _empty_repo()
    try:
        conversation = Conversation(
            id="thread-plain",
            provider="claude",
            title="Plain Conversation",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            metadata={"content_hash": "plain-hash"},
            messages=[],
        )

        result = asyncio.run(repo.save_conversation(conversation, [], []))
        stored = asyncio.run(repo.backend.get_conversation("thread-plain"))

        assert result["conversations"] == 1
        assert stored is not None
        assert stored.provider_conversation_id == "thread-plain"
    finally:
        asyncio.run(repo.close())
        tempdir.cleanup()


def test_repository_conversation_to_record_only_strips_matching_provider_prefix() -> None:
    repo = ConversationRepository(backend=MagicMock())
    prefixed = Conversation(
        id="claude:thread-1",
        provider="claude",
        title="Prefixed",
        metadata={"content_hash": "hash-1"},
        messages=[],
    )
    foreign = Conversation(
        id="chatgpt:thread-2",
        provider="claude",
        title="Foreign Prefix",
        metadata={"content_hash": "hash-2"},
        messages=[],
    )

    prefixed_record = repo._conversation_to_record(prefixed)
    foreign_record = repo._conversation_to_record(foreign)

    assert prefixed_record.provider_conversation_id == "thread-1"
    assert foreign_record.provider_conversation_id == "chatgpt:thread-2"
    assert prefixed_record.updated_at is None
    assert foreign_record.updated_at is None
    assert prefixed_record.provider_meta == {}
    assert foreign_record.provider_meta == {}


def test_records_to_conversation_preserves_order_and_parent_contract() -> None:
    """Record-to-model conversion must preserve IDs, ordering, and parent linkage."""
    conversation = ConversationRecord(
        conversation_id="conv-1",
        provider_name="claude",
        provider_conversation_id="conv-1",
        parent_conversation_id="parent-conv",
        title="Conversation",
        content_hash="hash-1",
        provider_meta={},
        metadata={},
    )
    messages = [
        MessageRecord(message_id="m1", conversation_id="conv-1", role="user", text="one", content_hash="hash-m1"),
        MessageRecord(message_id="m2", conversation_id="conv-1", role="assistant", text="two", content_hash="hash-m2"),
        MessageRecord(message_id="m3", conversation_id="conv-1", role="user", text="three", content_hash="hash-m3"),
    ]

    converted = _records_to_conversation(conversation, messages, [])

    assert str(converted.id) == "conv-1"
    assert str(converted.parent_id) == "parent-conv"
    assert [message.id for message in converted.messages] == ["m1", "m2", "m3"]


def test_repository_get_many_preserves_requested_order_and_skips_missing_records() -> None:
    backend = MagicMock()
    backend.get_conversations_batch = AsyncMock(
        return_value=[
            ConversationRecord(
                conversation_id="conv-b",
                provider_name="claude",
                provider_conversation_id="b",
                title="B",
                content_hash="hash-b",
                provider_meta={},
                metadata={},
            ),
            ConversationRecord(
                conversation_id="conv-a",
                provider_name="codex",
                provider_conversation_id="a",
                title="A",
                content_hash="hash-a",
                provider_meta={},
                metadata={},
            ),
        ]
    )
    repo = ConversationRepository(backend=backend)
    repo._hydrate_conversations = AsyncMock(
        return_value=[
            Conversation(id="conv-a", provider="codex", messages=[]),
            Conversation(id="conv-b", provider="claude", messages=[]),
        ]
    )

    conversations = asyncio.run(repo.get_many(["conv-a", "conv-missing", "conv-b"]))

    assert [str(conversation.id) for conversation in conversations] == ["conv-a", "conv-b"]
    backend.get_conversations_batch.assert_awaited_once_with(["conv-a", "conv-missing", "conv-b"])
    repo._hydrate_conversations.assert_awaited_once()
    call = repo._hydrate_conversations.await_args
    assert [record.conversation_id for record in call.args[0]] == ["conv-b", "conv-a"]
    assert call.kwargs["ordered_ids"] == ["conv-a", "conv-missing", "conv-b"]


def test_repository_get_stats_by_forwards_grouping_contract() -> None:
    backend = MagicMock()
    backend.get_stats_by = AsyncMock(return_value={"claude": 2, "codex": 1})
    repo = ConversationRepository(backend=backend)

    result = asyncio.run(repo.get_stats_by("provider"))

    assert result == {"claude": 2, "codex": 1}
    backend.get_stats_by.assert_awaited_once_with("provider")
