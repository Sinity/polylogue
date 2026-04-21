from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pytest

from polylogue.lib.conversation_models import Conversation
from polylogue.lib.messages import MessageCollection
from polylogue.storage.backends.query_store import SQLiteQueryStore
from polylogue.storage.repository_archive_search import RepositoryArchiveSearchMixin
from polylogue.storage.search_models import ConversationSearchResult
from polylogue.storage.search_providers.hybrid_conversations import (
    _resolve_ranked_conversation_hits,
)
from polylogue.storage.store import ConversationRecord
from polylogue.types import ContentHash, ConversationId, Provider


def _conversation_record(conversation_id: str, *, title: str, provider_name: str = "chatgpt") -> ConversationRecord:
    return ConversationRecord(
        conversation_id=ConversationId(conversation_id),
        provider_name=provider_name,
        provider_conversation_id=f"provider-{conversation_id}",
        title=title,
        content_hash=ContentHash(f"hash-{conversation_id}"),
    )


@dataclass
class _FakeQueries(SQLiteQueryStore):
    hits: ConversationSearchResult
    records_by_id: dict[str, ConversationRecord]
    last_batch_ids: list[str] | None = None

    async def search_conversation_hits(
        self,
        query: str,
        limit: int = 20,
        providers: list[str] | None = None,
    ) -> ConversationSearchResult:
        del query, limit, providers
        return self.hits

    async def search_action_conversation_hits(
        self,
        query: str,
        limit: int = 20,
        providers: list[str] | None = None,
    ) -> ConversationSearchResult:
        del query, limit, providers
        return self.hits

    async def get_conversations_batch(self, ids: list[str]) -> list[ConversationRecord]:
        self.last_batch_ids = ids
        return [self.records_by_id[conversation_id] for conversation_id in ids]


class _FakeRepo(RepositoryArchiveSearchMixin):
    queries: _FakeQueries

    def __init__(self, queries: _FakeQueries) -> None:
        self.queries = queries
        self.ordered_ids_seen: list[str] | None = None

    async def _hydrate_conversations(
        self,
        conversation_records: list[ConversationRecord],
        *,
        ordered_ids: list[str] | None = None,
    ) -> list[Conversation]:
        self.ordered_ids_seen = ordered_ids
        return [
            Conversation(
                id=ConversationId(str(record.conversation_id)),
                provider=Provider.from_string(record.provider_name),
                messages=MessageCollection.empty(),
                title=record.title,
            )
            for record in conversation_records
        ]


def _memory_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE conversations (conversation_id TEXT PRIMARY KEY, provider_name TEXT NOT NULL)")
    conn.execute("CREATE TABLE messages (message_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL)")
    return conn


def test_resolve_ranked_conversation_hits_preserves_order_and_provider_scope() -> None:
    conn = _memory_conn()
    conn.executemany(
        "INSERT INTO conversations(conversation_id, provider_name) VALUES (?, ?)",
        [
            ("conv-a", "chatgpt"),
            ("conv-b", "claude"),
        ],
    )
    conn.executemany(
        "INSERT INTO messages(message_id, conversation_id) VALUES (?, ?)",
        [
            ("msg-a1", "conv-a"),
            ("msg-a2", "conv-a"),
            ("msg-b1", "conv-b"),
        ],
    )

    hits = _resolve_ranked_conversation_hits(
        conn,
        message_results=[("msg-b1", 0.9), ("msg-a1", 0.8), ("msg-a2", 0.7)],
        limit=10,
        scope_names=None,
    )
    assert hits.conversation_ids() == ["conv-b", "conv-a"]
    assert [hit.rank for hit in hits.hits] == [1, 2]

    scoped_hits = _resolve_ranked_conversation_hits(
        conn,
        message_results=[("msg-b1", 0.9), ("msg-a1", 0.8), ("msg-a2", 0.7)],
        limit=10,
        scope_names=["chatgpt"],
    )
    assert scoped_hits.conversation_ids() == ["conv-a"]
    assert [hit.rank for hit in scoped_hits.hits] == [1]


@pytest.mark.asyncio
async def test_repository_search_summaries_follow_conversation_hit_order() -> None:
    hits = ConversationSearchResult.from_ids(["conv-b", "conv-a"])
    queries = _FakeQueries(
        hits=hits,
        records_by_id={
            "conv-a": _conversation_record("conv-a", title="First"),
            "conv-b": _conversation_record("conv-b", title="Second"),
        },
    )
    repo = _FakeRepo(queries)

    summaries = await repo.search_summaries("storage", limit=5)

    assert queries.last_batch_ids == ["conv-b", "conv-a"]
    assert [summary.id for summary in summaries] == [ConversationId("conv-b"), ConversationId("conv-a")]
    assert [summary.title for summary in summaries] == ["Second", "First"]


@pytest.mark.asyncio
async def test_repository_search_and_action_search_pass_ordered_ids_to_hydration() -> None:
    hits = ConversationSearchResult.from_ids(["conv-b", "conv-a"])
    queries = _FakeQueries(
        hits=hits,
        records_by_id={
            "conv-a": _conversation_record("conv-a", title="First"),
            "conv-b": _conversation_record("conv-b", title="Second"),
        },
    )
    repo = _FakeRepo(queries)

    search_result = await repo.search("storage", limit=5)
    assert [str(conversation.id) for conversation in search_result] == ["conv-b", "conv-a"]
    assert repo.ordered_ids_seen == ["conv-b", "conv-a"]

    action_result = await repo.search_actions("storage", limit=5)
    assert [str(conversation.id) for conversation in action_result] == ["conv-b", "conv-a"]
    assert repo.ordered_ids_seen == ["conv-b", "conv-a"]
