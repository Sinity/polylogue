from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from polylogue.lib.conversation.models import Conversation
from polylogue.lib.json import JSONDocument
from polylogue.lib.message.messages import MessageCollection
from polylogue.pipeline.prepare import prepare_records
from polylogue.sources.parsers.drive import parse_chunked_prompt
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.query_store import SQLiteQueryStore
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.repository.archive.search import RepositoryArchiveSearchMixin
from polylogue.storage.runtime import ConversationRecord
from polylogue.storage.search.models import ConversationSearchEvidenceHit, ConversationSearchResult
from polylogue.storage.search_providers.hybrid_conversations import (
    _resolve_ranked_conversation_hits,
)
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
    attachment_hits: list[ConversationSearchEvidenceHit] | None = None
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

    async def search_conversation_evidence_hits(
        self,
        query: str,
        limit: int = 20,
        providers: list[str] | None = None,
        since: str | None = None,
    ) -> list[ConversationSearchEvidenceHit]:
        del query, limit, providers, since
        return [
            ConversationSearchEvidenceHit(
                conversation_id=hit.conversation_id,
                rank=hit.rank,
                score=hit.score,
                message_id=f"msg-{hit.conversation_id}",
                snippet=f"snippet for {hit.conversation_id}",
            )
            for hit in self.hits.hits
        ]

    async def search_attachment_identity_evidence_hits(
        self,
        query: str,
        limit: int = 20,
        providers: list[str] | None = None,
        since: str | None = None,
    ) -> list[ConversationSearchEvidenceHit]:
        del query, limit, providers, since
        return self.attachment_hits or []

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
async def test_repository_search_summary_hits_keep_evidence_and_conversation_order() -> None:
    hits = ConversationSearchResult.from_ids(["conv-b", "conv-a"])
    queries = _FakeQueries(
        hits=hits,
        records_by_id={
            "conv-a": _conversation_record("conv-a", title="First"),
            "conv-b": _conversation_record("conv-b", title="Second"),
        },
    )
    repo = _FakeRepo(queries)

    summary_hits = await repo.search_summary_hits("storage", limit=5, providers=["chatgpt"], since="2025-01-01")

    assert queries.last_batch_ids == ["conv-b", "conv-a"]
    assert [hit.conversation_id for hit in summary_hits] == ["conv-b", "conv-a"]
    assert [hit.rank for hit in summary_hits] == [1, 2]
    assert [hit.message_id for hit in summary_hits] == ["msg-conv-b", "msg-conv-a"]
    assert [hit.snippet for hit in summary_hits] == ["snippet for conv-b", "snippet for conv-a"]


@pytest.mark.asyncio
async def test_repository_search_summary_hits_prioritize_attachment_identity_evidence() -> None:
    hits = ConversationSearchResult.from_ids(["conv-b", "conv-a"])
    attachment_hit = ConversationSearchEvidenceHit(
        conversation_id="conv-a",
        rank=1,
        message_id="msg-attachment",
        snippet="attachment identity provider_meta.fileId=drive-file-1",
        match_surface="attachment",
        retrieval_lane="attachment",
    )
    queries = _FakeQueries(
        hits=hits,
        attachment_hits=[attachment_hit],
        records_by_id={
            "conv-a": _conversation_record("conv-a", title="Attachment Match"),
            "conv-b": _conversation_record("conv-b", title="Message Match"),
        },
    )
    repo = _FakeRepo(queries)

    summary_hits = await repo.search_summary_hits("drive-file-1", limit=5, providers=["gemini"])

    assert queries.last_batch_ids == ["conv-a", "conv-b"]
    assert [hit.conversation_id for hit in summary_hits] == ["conv-a", "conv-b"]
    assert [hit.rank for hit in summary_hits] == [1, 2]
    assert summary_hits[0].match_surface == "attachment"
    assert summary_hits[0].retrieval_lane == "attachment"
    assert summary_hits[0].message_id == "msg-attachment"
    assert summary_hits[1].match_surface == "message"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected_snippet"),
    [
        ("provider-attachment-218", "provider_meta.provider_id=provider-attachment-218"),
        ("drive-file-218", "provider_meta.fileId=drive-file-218"),
        ("drive-root-218", "provider_meta.driveId=drive-root-218"),
    ],
    ids=["provider-attachment-id", "drive-file-id", "drive-id"],
)
async def test_gemini_drive_attachment_id_is_searchable_after_parse_and_prepare(
    tmp_path: Path,
    query: str,
    expected_snippet: str,
) -> None:
    payload: JSONDocument = {
        "id": "gemini-attachment-identity",
        "displayName": "Gemini Attachment Identity",
        "chunkedPrompt": {
            "chunks": [
                {
                    "id": "msg-doc",
                    "role": "user",
                    "text": "Please review the attached project plan.",
                    "driveDocument": {
                        "id": "provider-attachment-218",
                        "fileId": "drive-file-218",
                        "driveId": "drive-root-218",
                        "name": "Project Plan",
                        "mimeType": "application/vnd.google-apps.document",
                    },
                }
            ]
        },
    }
    backend = SQLiteBackend(db_path=tmp_path / "attachment-identity.db")
    repo = ConversationRepository(backend=backend)
    try:
        parsed = parse_chunked_prompt("gemini", payload, "fallback-id")
        await prepare_records(
            parsed,
            "gemini-export.json",
            archive_root=tmp_path / "archive",
            backend=backend,
            repository=repo,
        )

        hits = await repo.search_summary_hits(query, limit=5, providers=["gemini"])
    finally:
        await repo.close()

    assert len(hits) == 1
    hit = hits[0]
    assert hit.conversation_id == "gemini:gemini-attachment-identity"
    assert hit.match_surface == "attachment"
    assert hit.retrieval_lane == "attachment"
    assert hit.message_id == "gemini:gemini-attachment-identity:msg-doc"
    assert hit.snippet is not None
    assert expected_snippet in hit.snippet
    assert 'name="Project Plan"' in hit.snippet


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
