"""Law-based contracts for ConversationRepository."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.lib.filters import ConversationFilter
from polylogue.lib.models import Conversation
from polylogue.sources import RecordBundle, save_bundle
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.repository import ConversationRepository, _records_to_conversation
from polylogue.storage.store import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
    RunRecord,
)
from tests.infra.storage_records import make_attachment, make_conversation, make_message
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
    return tempdir, ConversationRepository(backend=SQLiteBackend(db_path=db_path))


def _empty_repo() -> tuple[TemporaryDirectory[str], ConversationRepository]:
    tempdir = TemporaryDirectory()
    db_path = Path(tempdir.name) / "repo.db"
    return tempdir, ConversationRepository(backend=SQLiteBackend(db_path=db_path))


async def _collect_summary_pages(repo: ConversationRepository, page_size: int, **kwargs) -> list[str]:
    pages = [page async for page in repo.iter_summary_pages(page_size=page_size, **kwargs)]
    return [summary.id for page in pages for summary in page]


async def _collect_messages(iterator) -> list:
    return [message async for message in iterator]


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
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    conversation_graph_strategy(),
    st.integers(min_value=0, max_value=5),
    st.integers(min_value=1, max_value=4),
    st.integers(min_value=1, max_value=4),
)
def test_repository_graph_views_agree_on_generated_graph(
    specs,
    candidate_index: int,
    page_size: int,
    limit: int,
) -> None:
    """Repository read models must agree on the same generated archive graph."""
    tempdir, repo = _seed_repo(specs)
    try:
        expected_ids = expected_sorted_ids(specs)
        index = candidate_index % len(specs)
        target = specs[index]
        prefix = shortest_unique_prefix(tuple(expected_ids), target.conversation_id)
        provider_counts = _expected_archive_stats(specs)[2]
        expected_counts = {spec.conversation_id: len(spec.messages) for spec in specs}

        assert asyncio.run(repo.count()) == len(specs)
        assert [summary.id for summary in asyncio.run(repo.list_summaries(limit=None))] == expected_ids
        assert [str(conv.id) for conv in asyncio.run(repo.list(limit=None))] == expected_ids
        assert asyncio.run(_collect_summary_pages(repo, page_size)) == expected_ids
        assert [summary.id for summary in asyncio.run(repo.list_summaries(limit=limit))] == expected_ids[:limit]

        for provider in sorted(provider_counts):
            provider_specs = tuple(spec for spec in specs if spec.provider == provider)
            provider_ids = expected_sorted_ids(provider_specs)
            assert asyncio.run(repo.count(provider=provider)) == len(provider_ids)
            assert [summary.id for summary in asyncio.run(repo.list_summaries(limit=None, provider=provider))] == provider_ids
            assert [str(conv.id) for conv in asyncio.run(repo.list(limit=None, provider=provider))] == provider_ids
            assert asyncio.run(repo.get_provider_conversation_ids(provider)) == provider_ids

        summary = asyncio.run(repo.get_summary(target.conversation_id))
        conversation = asyncio.run(repo.get(target.conversation_id))
        eager = asyncio.run(repo.get_eager(target.conversation_id))
        viewed = asyncio.run(repo.view(prefix))
        projection = asyncio.run(repo.get_render_projection(target.conversation_id))
        root = asyncio.run(repo.get_root(target.conversation_id))
        tree = asyncio.run(repo.get_session_tree(target.conversation_id))
        parent = asyncio.run(repo.get_parent(target.conversation_id))
        all_messages = asyncio.run(_collect_messages(repo.iter_messages(target.conversation_id)))
        limited_messages = asyncio.run(
            _collect_messages(repo.iter_messages(target.conversation_id, limit=limit))
        )
        stats = asyncio.run(repo.get_conversation_stats(target.conversation_id))

        assert summary is not None
        assert conversation is not None
        assert eager is not None
        assert viewed is not None
        assert projection is not None
        assert str(summary.id) == target.conversation_id
        assert str(conversation.id) == target.conversation_id
        assert str(eager.id) == target.conversation_id
        assert str(viewed.id) == target.conversation_id
        assert projection.conversation.conversation_id == target.conversation_id
        assert len(conversation.messages) == len(target.messages)
        assert len(eager.messages) == len(target.messages)
        assert len(viewed.messages) == len(target.messages)
        assert len(projection.messages) == len(target.messages)
        assert projection.attachments == []
        assert str(root.id) == specs[root_index(specs, index)].conversation_id
        assert {str(conv.id) for conv in tree} == expected_tree_ids(specs, index)
        expected_parent = None if target.parent_index is None else specs[target.parent_index].conversation_id
        if expected_parent is None:
            assert parent is None
        else:
            assert parent is not None
            assert str(parent.id) == expected_parent
        assert len(all_messages) == len(target.messages)
        assert len(limited_messages) == min(limit, len(target.messages))
        assert stats is not None
        assert stats["total_messages"] == len(target.messages)

        assert asyncio.run(repo.get_message_counts_batch(list(expected_counts))) == expected_counts
        aggregate = asyncio.run(repo.aggregate_message_stats(list(expected_counts)))
        assert aggregate["total"] == sum(expected_counts.values())
        assert aggregate["attachments"] == 0
        assert aggregate["providers"] == provider_counts
        assert aggregate["min_sort_key"] is not None
        assert aggregate["max_sort_key"] is not None

        archive_stats = asyncio.run(repo.get_archive_stats())
        expected_conversations, expected_messages, _ = _expected_archive_stats(specs)
        assert archive_stats.total_conversations == expected_conversations
        assert archive_stats.total_messages == expected_messages
        assert archive_stats.provider_count == len(provider_counts)
        assert archive_stats.providers == provider_counts
        assert archive_stats.embedded_conversations == 0
        assert archive_stats.embedded_messages == 0
        assert archive_stats.embedding_coverage == 0.0
        assert archive_stats.avg_messages_per_conversation == pytest.approx(
            expected_messages / expected_conversations
        )
        assert archive_stats.db_size_bytes >= 0
        assert asyncio.run(repo.get_stats_by("provider")) == provider_counts
    finally:
        asyncio.run(repo.close())
        tempdir.cleanup()


def test_repository_missing_and_child_lookup_contracts() -> None:
    """Missing roots should fail explicitly while child hydration uses the listed records."""
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

    tempdir, empty_repo = _empty_repo()
    try:
        with pytest.raises(ValueError, match="not found"):
            asyncio.run(empty_repo.get_root("missing"))
    finally:
        asyncio.run(empty_repo.close())
        tempdir.cleanup()


def test_repository_search_and_calendar_contracts() -> None:
    """Search and calendar aggregations must preserve ranked IDs and bucket counts."""
    search_specs = (
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
            created_at="2024-02-03T00:00:00+00:00",
            updated_at="2024-02-03T00:00:00+00:00",
            parent_index=None,
            messages=(MessageSpec(role="assistant", text="unrelated", has_tool_use=False, has_thinking=False),),
        ),
    )
    tempdir, repo = _seed_repo(search_specs)
    try:
        summaries = asyncio.run(repo.search_summaries("needle", limit=5))
        conversations = asyncio.run(repo.search("needle", limit=5))
        assert [summary.id for summary in summaries] == ["conv-beta", "conv-alpha"]
        assert [str(conv.id) for conv in conversations] == ["conv-beta", "conv-alpha"]
        assert asyncio.run(repo.get_stats_by("year")) == {"2024": 3}
        assert asyncio.run(repo.get_stats_by("month")) == {"2024-01": 2, "2024-02": 1}
    finally:
        asyncio.run(repo.close())
        tempdir.cleanup()


def test_repository_summary_paging_forwarding_contracts() -> None:
    """Summary paging must preserve filters, advance offsets by page length, and stop on short pages."""
    repo = ConversationRepository(backend=MagicMock())
    repo.list_summaries = AsyncMock(
        side_effect=[
            [MagicMock(id="conv-1"), MagicMock(id="conv-2")],
            [MagicMock(id="conv-3")],
        ]
    )

    page_ids = asyncio.run(
        _collect_summary_pages(
            repo,
            2,
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
    )

    assert page_ids == ["conv-1", "conv-2", "conv-3"]
    kwargs0 = repo.list_summaries.await_args_list[0].kwargs
    kwargs1 = repo.list_summaries.await_args_list[1].kwargs
    assert kwargs0 == {
        "limit": 2,
        "offset": 0,
        "provider": "claude",
        "providers": ["claude", "codex"],
        "source": "archive",
        "since": "2024-01-01",
        "until": "2024-01-31",
        "title_contains": "needle",
        "has_tool_use": True,
        "has_thinking": True,
        "min_messages": 2,
        "max_messages": 5,
        "min_words": 10,
        "has_file_ops": True,
        "has_git_ops": True,
        "has_subagent": True,
    }
    assert kwargs1["offset"] == 2


def test_repository_public_save_projection_contracts() -> None:
    """Public save and projection helpers must preserve provider IDs, ordering, and metadata."""
    tempdir, repo = _empty_repo()
    try:
        prefixed = Conversation(
            id="claude:thread-1",
            provider="claude",
            title="Prefixed",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            metadata={"content_hash": "hash-1", "provider_meta": {"source": "law"}, "custom": "value"},
            messages=[],
        )
        plain = Conversation(
            id="thread-plain",
            provider="claude",
            title="Plain",
            created_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
            metadata={"content_hash": "hash-2"},
            messages=[],
        )
        foreign = Conversation(
            id="chatgpt:thread-2",
            provider="claude",
            title="Foreign Prefix",
            metadata={"content_hash": "hash-3"},
            messages=[],
        )
        message = make_message(
            "msg-thread-1",
            "claude:thread-1",
            role="user",
            text="hello",
            content_hash="msg-hash",
        )

        first = asyncio.run(repo.save_conversation(prefixed, [message], []))
        second = asyncio.run(repo.save_conversation(prefixed, [message], []))
        third = asyncio.run(repo.save_conversation(plain, [], []))
        stored_prefixed = asyncio.run(repo.backend.get_conversation("claude:thread-1"))
        stored_plain = asyncio.run(repo.backend.get_conversation("thread-plain"))
        prefixed_record = repo._conversation_to_record(prefixed)
        foreign_record = repo._conversation_to_record(foreign)

        assert first == {
            "conversations": 1,
            "messages": 1,
            "attachments": 0,
            "skipped_conversations": 0,
            "skipped_messages": 0,
            "skipped_attachments": 0,
        }
        assert second["skipped_conversations"] == 1
        assert second["skipped_messages"] == 1
        assert third["conversations"] == 1
        assert stored_prefixed is not None
        assert stored_plain is not None
        assert stored_prefixed.provider_conversation_id == "thread-1"
        assert stored_prefixed.provider_meta == {"source": "law"}
        assert stored_prefixed.metadata["custom"] == "value"
        assert stored_plain.provider_conversation_id == "thread-plain"
        assert prefixed_record.provider_conversation_id == "thread-1"
        assert foreign_record.provider_conversation_id == "chatgpt:thread-2"
        assert prefixed_record.updated_at is not None
        assert foreign_record.updated_at is None
    finally:
        asyncio.run(repo.close())
        tempdir.cleanup()


@pytest.mark.parametrize("content_unchanged", [True, False], ids=["unchanged", "changed"])
def test_repository_save_via_backend_contract(content_unchanged: bool) -> None:
    """Internal backend saves must either skip unchanged children or persist the full payload."""
    connection = _Connection([_Cursor(one={"content_hash": "same-hash"} if content_unchanged else None)])
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
    attachment = AttachmentRecord(attachment_id="att-1", conversation_id="claude:conv-1")
    conversation = ConversationRecord(
        conversation_id="claude:conv-1",
        provider_name="claude",
        provider_conversation_id="conv-1",
        title="Title",
        content_hash="same-hash" if content_unchanged else "new-hash",
        provider_meta={},
        metadata={},
    )

    result = asyncio.run(repo._save_via_backend(conversation, [message], [attachment], [explicit_block]))

    backend.save_conversation_record.assert_awaited_once_with(conversation)
    if content_unchanged:
        assert result == {
            "conversations": 0,
            "messages": 0,
            "attachments": 0,
            "skipped_conversations": 1,
            "skipped_messages": 1,
            "skipped_attachments": 1,
        }
        backend.save_messages.assert_not_awaited()
        backend.upsert_conversation_stats.assert_not_awaited()
        backend.save_content_blocks.assert_not_awaited()
        backend.prune_attachments.assert_not_awaited()
        backend.save_attachments.assert_not_awaited()
    else:
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


def test_repository_forwarding_and_lifecycle_contracts(tmp_path: Path) -> None:
    """Forwarders, lifecycle, and filter/list/count plumbing must stay canonical."""
    backend = MagicMock()
    backend.get_metadata = AsyncMock(return_value={"status": "reviewed"})
    backend.update_metadata = AsyncMock()
    backend.delete_metadata = AsyncMock()
    backend.add_tag = AsyncMock()
    backend.remove_tag = AsyncMock()
    backend.list_tags = AsyncMock(return_value={"important": 2})
    backend.set_metadata = AsyncMock()
    backend.delete_conversation = AsyncMock(return_value=True)
    backend.get_stats_by = AsyncMock(return_value={"claude": 2, "codex": 1})
    backend.count_conversations = AsyncMock(return_value=7)
    backend.list_conversations = AsyncMock(
        return_value=[
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
    )
    backend.record_run = AsyncMock()
    repo = ConversationRepository(backend=backend)
    repo._hydrate_conversations = AsyncMock(
        return_value=[Conversation(id="claude:conv-1", provider="claude", title="One", messages=[])]
    )
    run_record = RunRecord(
        run_id="run-1",
        timestamp="2024-01-01T00:00:00+00:00",
        counts={"conversations": 1, "messages": 2},
    )

    assert asyncio.run(repo.get_metadata("conv-1")) == {"status": "reviewed"}
    asyncio.run(repo.update_metadata("conv-1", "status", "done"))
    asyncio.run(repo.delete_metadata("conv-1", "status"))
    asyncio.run(repo.add_tag("conv-1", "important"))
    asyncio.run(repo.remove_tag("conv-1", "important"))
    assert asyncio.run(repo.list_tags(provider="claude")) == {"important": 2}
    asyncio.run(repo.set_metadata("conv-1", {"status": "done"}))
    assert asyncio.run(repo.delete_conversation("conv-1")) is True
    assert asyncio.run(repo.get_stats_by("provider")) == {"claude": 2, "codex": 1}
    asyncio.run(repo.record_run(run_record))
    assert isinstance(repo.filter(), ConversationFilter)

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

    assert [str(conv.id) for conv in conversations] == ["claude:conv-1"]
    assert [summary.id for summary in summaries] == ["claude:conv-1"]
    assert count == 7
    backend.list_conversations.assert_any_await(
        limit=3,
        offset=2,
        provider="claude",
        providers=["claude", "codex"],
        source=None,
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
    backend.list_conversations.assert_any_await(
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
    backend.count_conversations.assert_awaited_once_with(
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
    repo._hydrate_conversations.assert_awaited_once()

    async def lifecycle() -> None:
        db_path = tmp_path / "repository-context.db"
        async with ConversationRepository(db_path=db_path) as lifecycle_repo:
            assert await lifecycle_repo.get_conversation("missing") is None
        await lifecycle_repo.close()
        await lifecycle_repo.close()
        assert await lifecycle_repo.get_conversation("missing") is None

    asyncio.run(lifecycle())


def test_repository_embedding_and_similarity_contracts() -> None:
    """Embedding and similarity helpers must resolve providers, preserve ordering, and surface errors."""
    backend = MagicMock()
    messages = [
        MessageRecord(
            message_id="msg-1",
            conversation_id="conv-1",
            role="user",
            text="hello",
            content_hash="hash-1",
        ),
        MessageRecord(
            message_id="msg-2",
            conversation_id="conv-1",
            role="assistant",
            text="world",
            content_hash="hash-2",
        ),
    ]
    backend.get_messages = AsyncMock(return_value=messages)
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
    repo._hydrate_conversations = AsyncMock(
        return_value=[Conversation(id="conv-b", provider="codex", messages=[])]
    )
    repo.get_many = AsyncMock(
        return_value=[
            Conversation(id="conv-a", provider="claude", messages=[]),
            Conversation(id="conv-b", provider="codex", messages=[]),
        ]
    )
    repo._get_message_conversation_mapping = AsyncMock(
        return_value={"m1": "conv-a", "m2": "conv-a", "m3": "conv-b"}
    )
    supplied_provider = MagicMock()
    supplied_provider.query.return_value = [("m2", 0.8), ("m1", 0.2), ("m3", 0.5)]

    assert asyncio.run(repo.embed_conversation("conv-1", vector_provider=supplied_provider)) == 2
    supplied_provider.upsert.assert_called_once_with("conv-1", messages)

    resolved_provider = MagicMock()
    with patch(
        "polylogue.storage.search_providers.create_vector_provider",
        return_value=resolved_provider,
    ):
        assert asyncio.run(repo.embed_conversation("conv-1", vector_provider=None)) == 2

    with patch(
        "polylogue.storage.search_providers.create_vector_provider",
        return_value=None,
    ):
        with pytest.raises(ValueError, match="No vector provider available"):
            asyncio.run(repo.embed_conversation("conv-1", vector_provider=None))
        with pytest.raises(ValueError, match="No vector provider configured"):
            asyncio.run(repo.similarity_search("needle", vector_provider=None))

    summaries = asyncio.run(repo.search_summaries("needle", limit=2, providers=["codex", "claude"]))
    conversations = asyncio.run(repo.search("needle", limit=2, providers=["codex", "claude"]))
    similar = asyncio.run(repo.search_similar("needle", limit=2, vector_provider=supplied_provider))
    similar_rows = asyncio.run(repo.similarity_search("needle", limit=2, vector_provider=supplied_provider))

    assert [summary.id for summary in summaries] == ["conv-b", "conv-a"]
    assert conversations == [Conversation(id="conv-b", provider="codex", messages=[])]
    backend.search_conversations.assert_any_await("needle", limit=2, providers=["codex", "claude"])
    assert backend.get_conversations_batch.await_count == 2
    repo._hydrate_conversations.assert_awaited_once()
    repo.get_many.assert_awaited_once_with(["conv-a", "conv-b"])
    assert [str(conv.id) for conv in similar] == ["conv-a", "conv-b"]
    assert similar_rows == [
        ("conv-a", "m2", 0.8),
        ("conv-a", "m1", 0.2),
        ("conv-b", "m3", 0.5),
    ]


def test_repository_bundle_and_archive_stats_contracts(tmp_path: Path) -> None:
    """Bundle pruning and archive stats SQL aggregation must stay consistent."""

    async def bundle_roundtrip() -> None:
        db_path = tmp_path / "repository-prune.db"
        repository = ConversationRepository(db_path=db_path)
        try:
            initial_bundle = RecordBundle(
                conversation=make_conversation("conv:prune", provider_name="codex", content_hash="hash-v1"),
                messages=[
                    make_message(
                        "msg:prune",
                        "conv:prune",
                        text="hello",
                        timestamp="2026-03-12T10:00:00Z",
                        content_hash="msg-hash-v1",
                    )
                ],
                attachments=[
                    make_attachment("att-0", "conv:prune", "msg:prune"),
                    make_attachment("att-1", "conv:prune", "msg:prune"),
                    make_attachment("att-2", "conv:prune", "msg:prune"),
                ],
            )
            await save_bundle(initial_bundle, repository=repository)
            updated_bundle = RecordBundle(
                conversation=make_conversation("conv:prune", provider_name="codex", content_hash="hash-v2"),
                messages=[
                    make_message(
                        "msg:prune",
                        "conv:prune",
                        text="hello",
                        timestamp="2026-03-12T10:00:00Z",
                        content_hash="msg-hash-v1",
                    )
                ],
                attachments=[make_attachment("att-0", "conv:prune", "msg:prune")],
            )
            await save_bundle(updated_bundle, repository=repository)

            with open_connection(db_path) as conn:
                rows = conn.execute(
                    "SELECT attachment_id FROM attachments WHERE attachment_id LIKE 'att-%' ORDER BY attachment_id"
                ).fetchall()
            assert [row["attachment_id"] for row in rows] == ["att-0"]
        finally:
            await repository.close()

    asyncio.run(bundle_roundtrip())

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

    class _FailingConnection(_Connection):
        async def execute(self, query: str, params=()):
            if "embedding_status" in query:
                raise RuntimeError("embedding table missing")
            return await super().execute(query, params)

    failing_connection = _FailingConnection(
        [
            _Cursor(one=(2,)),
            _Cursor(one=(5,)),
            _Cursor(one=(1,)),
            _Cursor(rows=[{"provider_name": "claude", "count": 2}]),
        ]
    )
    failing_backend = MagicMock()
    failing_backend.db_path = tmp_path / "repo-missing.db"
    failing_backend.db_path.write_bytes(b"x" * 64)
    failing_backend.connection = MagicMock(return_value=_context(failing_connection))
    failing_repo = ConversationRepository(backend=failing_backend)

    missing_stats = asyncio.run(failing_repo.get_archive_stats())
    assert missing_stats.total_conversations == 2
    assert missing_stats.total_messages == 5
    assert missing_stats.total_attachments == 1
    assert missing_stats.providers == {"claude": 2}
    assert missing_stats.embedded_conversations == 0
    assert missing_stats.embedded_messages == 0


def test_repository_mapping_and_order_contracts() -> None:
    """Private mapping helpers must preserve SQL lookups, message order, and requested ID order."""
    mapping_connection = _Connection(
        [
            _Cursor(
                rows=[
                    {"message_id": "msg-1", "conversation_id": "conv-1"},
                    {"message_id": "msg-2", "conversation_id": "conv-2"},
                ]
            )
        ]
    )
    backend = MagicMock()
    backend.connection = MagicMock(return_value=_context(mapping_connection))
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

    mapping = asyncio.run(repo._get_message_conversation_mapping(["msg-1", "msg-2"]))
    conversations = asyncio.run(repo.get_many(["conv-a", "conv-missing", "conv-b"]))

    record = ConversationRecord(
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
    converted = _records_to_conversation(record, messages, [])

    assert mapping == {"msg-1": "conv-1", "msg-2": "conv-2"}
    assert mapping_connection.calls == [
        (
            "SELECT message_id, conversation_id FROM messages WHERE message_id IN (?,?)",
            ["msg-1", "msg-2"],
        )
    ]
    assert [str(conversation.id) for conversation in conversations] == ["conv-a", "conv-b"]
    backend.get_conversations_batch.assert_awaited_once_with(["conv-a", "conv-missing", "conv-b"])
    repo._hydrate_conversations.assert_awaited_once()
    assert [message.id for message in converted.messages] == ["m1", "m2", "m3"]
    assert str(converted.parent_id) == "parent-conv"
