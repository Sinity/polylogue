"""Law-based contracts for ConversationRepository."""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from tests.infra.strategies import (
    conversation_graph_strategy,
    expected_sorted_ids,
    expected_tree_ids,
    root_index,
    seed_conversation_graph,
)


def _seed_repo(specs) -> tuple[TemporaryDirectory[str], ConversationRepository]:
    tempdir = TemporaryDirectory()
    db_path = Path(tempdir.name) / "repo.db"
    seed_conversation_graph(db_path, specs)
    backend = SQLiteBackend(db_path=db_path)
    return tempdir, ConversationRepository(backend=backend)


async def _collect_summary_pages(repo: ConversationRepository, page_size: int) -> list[str]:
    pages = [page async for page in repo.iter_summary_pages(page_size=page_size)]
    return [summary.id for page in pages for summary in page]


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
