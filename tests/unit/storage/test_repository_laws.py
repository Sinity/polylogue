"""Law-based contracts for ConversationRepository."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.lib.models import Conversation
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from tests.infra.helpers import make_message
from tests.infra.strategies import (
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
