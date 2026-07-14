from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.session.domain_models import Session
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument
from polylogue.core.sources import origin_from_provider
from polylogue.core.types import ContentHash, SessionId
from polylogue.sources.parsers.drive import parse_chunked_prompt
from polylogue.storage.repository import SessionRepository
from polylogue.storage.repository.archive.search import RepositoryArchiveSearchMixin
from polylogue.storage.runtime import SessionRecord
from polylogue.storage.search.models import (
    SessionSearchEvidenceRow,
    SessionSearchResult,
)
from polylogue.storage.search.models import (
    SessionSearchIdHit as StorageSessionSearchIdHit,
)
from polylogue.storage.search_providers.hybrid_sessions import (
    _resolve_ranked_session_hits,
)
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.query_store import SQLiteQueryStore
from tests.infra.live_ingest import ingest_session


def _session_record(session_id: str, *, title: str, source_name: str = "chatgpt") -> SessionRecord:
    return SessionRecord(
        session_id=SessionId(session_id),
        origin=origin_from_provider(Provider.from_string(source_name)),
        native_id=f"provider-{session_id}",
        title=title,
        content_hash=ContentHash(hashlib.sha256(f"hash-{session_id}".encode()).hexdigest()),
    )


@dataclass
class _FakeQueries(SQLiteQueryStore):
    hits: SessionSearchResult
    records_by_id: dict[str, SessionRecord]
    attachment_hits: list[SessionSearchEvidenceRow] | None = None
    last_batch_ids: list[str] | None = None

    async def search_session_hits(
        self,
        query: str,
        limit: int = 20,
        providers: list[str] | None = None,
    ) -> SessionSearchResult:
        del query, limit, providers
        return self.hits

    async def search_action_session_hits(
        self,
        query: str,
        limit: int = 20,
        providers: list[str] | None = None,
    ) -> SessionSearchResult:
        del query, limit, providers
        return self.hits

    async def search_session_evidence_hits(
        self,
        query: str,
        limit: int = 20,
        providers: list[str] | None = None,
        since: str | None = None,
    ) -> list[SessionSearchEvidenceRow]:
        del query, limit, providers, since
        return [
            SessionSearchEvidenceRow(
                session_id=hit.session_id,
                rank=hit.rank,
                score=hit.score,
                message_id=f"msg-{hit.session_id}",
                snippet=f"snippet for {hit.session_id}",
                score_components={"bm25_raw": hit.score} if hit.score is not None else {},
                score_kind="bm25" if hit.score is not None else None,
                lane_rank=hit.rank,
                raw_score=hit.score,
            )
            for hit in self.hits.hits
        ]

    async def search_attachment_identity_evidence_hits(
        self,
        query: str,
        limit: int = 20,
        providers: list[str] | None = None,
        since: str | None = None,
    ) -> list[SessionSearchEvidenceRow]:
        del query, limit, providers, since
        return self.attachment_hits or []

    async def get_sessions_batch(self, ids: list[str]) -> list[SessionRecord]:
        self.last_batch_ids = ids
        return [self.records_by_id[session_id] for session_id in ids]

    async def get_message_counts_batch(self, ids: list[str]) -> dict[str, int]:
        """Hydrator (#1630): return a count per session id keyed in records."""
        return {session_id: 42 + i for i, session_id in enumerate(ids)}


class _FakeRepo(RepositoryArchiveSearchMixin):
    queries: _FakeQueries

    def __init__(self, queries: _FakeQueries) -> None:
        self.queries = queries
        self.ordered_ids_seen: list[str] | None = None

    async def _hydrate_sessions(
        self,
        session_records: list[SessionRecord],
        *,
        ordered_ids: list[str] | None = None,
    ) -> list[Session]:
        self.ordered_ids_seen = ordered_ids
        return [
            Session(
                id=SessionId(str(record.session_id)),
                origin=record.origin,
                messages=MessageCollection.empty(),
                title=record.title,
            )
            for record in session_records
        ]


def _memory_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT NOT NULL)")
    conn.execute("CREATE TABLE messages (message_id TEXT PRIMARY KEY, session_id TEXT NOT NULL)")
    return conn


def test_resolve_ranked_session_hits_preserves_order_and_provider_scope() -> None:
    conn = _memory_conn()
    conn.executemany(
        "INSERT INTO sessions(session_id, origin) VALUES (?, ?)",
        [
            ("conv-a", "chatgpt"),
            ("conv-b", "claude"),
        ],
    )
    conn.executemany(
        "INSERT INTO messages(message_id, session_id) VALUES (?, ?)",
        [
            ("msg-a1", "conv-a"),
            ("msg-a2", "conv-a"),
            ("msg-b1", "conv-b"),
        ],
    )

    hits = _resolve_ranked_session_hits(
        conn,
        message_results=[("msg-b1", 0.9), ("msg-a1", 0.8), ("msg-a2", 0.7)],
        limit=10,
        scope_names=None,
    )
    assert hits.session_ids() == ["conv-b", "conv-a"]
    assert [hit.rank for hit in hits.hits] == [1, 2]

    scoped_hits = _resolve_ranked_session_hits(
        conn,
        message_results=[("msg-b1", 0.9), ("msg-a1", 0.8), ("msg-a2", 0.7)],
        limit=10,
        scope_names=["chatgpt"],
    )
    assert scoped_hits.session_ids() == ["conv-a"]
    assert [hit.rank for hit in scoped_hits.hits] == [1]


@pytest.mark.asyncio
async def test_repository_search_summaries_follow_session_hit_order() -> None:
    hits = SessionSearchResult(
        hits=[
            StorageSessionSearchIdHit(session_id="conv-b", rank=1, score=1.0),
            StorageSessionSearchIdHit(session_id="conv-a", rank=2, score=2.0),
        ]
    )
    queries = _FakeQueries(
        hits=hits,
        records_by_id={
            "conv-a": _session_record("conv-a", title="First"),
            "conv-b": _session_record("conv-b", title="Second"),
        },
    )
    repo = _FakeRepo(queries)

    summaries = await repo.search_summaries("storage", limit=5)

    assert queries.last_batch_ids == ["conv-b", "conv-a"]
    assert [summary.id for summary in summaries] == [SessionId("conv-b"), SessionId("conv-a")]
    assert [summary.title for summary in summaries] == ["Second", "First"]


@pytest.mark.asyncio
async def test_repository_search_summaries_hydrates_message_count_from_stats() -> None:
    """#1630: ``search_summaries`` must populate ``message_count`` from
    the sessions aggregate columns so the daemon HTTP search path and any other
    caller see a real total instead of None.

    Before the fix the message_count surfaced as 0 (None coerced) for every
    session in /api/sessions and /api/sessions?q=... results.
    """
    hits = SessionSearchResult(
        hits=[
            StorageSessionSearchIdHit(session_id="conv-b", rank=1, score=1.0),
            StorageSessionSearchIdHit(session_id="conv-a", rank=2, score=2.0),
        ]
    )
    queries = _FakeQueries(
        hits=hits,
        records_by_id={
            "conv-a": _session_record("conv-a", title="First"),
            "conv-b": _session_record("conv-b", title="Second"),
        },
    )
    repo = _FakeRepo(queries)

    summaries = await repo.search_summaries("storage", limit=5)

    counts = {str(summary.id): summary.message_count for summary in summaries}
    assert counts == {"conv-b": 42, "conv-a": 43}, (
        f"#1630: search_summaries must hydrate message_count from stats; got {counts}"
    )


@pytest.mark.asyncio
async def test_repository_search_summary_hits_keep_evidence_and_session_order() -> None:
    hits = SessionSearchResult(
        hits=[
            StorageSessionSearchIdHit(session_id="conv-b", rank=1, score=1.0),
            StorageSessionSearchIdHit(session_id="conv-a", rank=2, score=2.0),
        ]
    )
    queries = _FakeQueries(
        hits=hits,
        records_by_id={
            "conv-a": _session_record("conv-a", title="First"),
            "conv-b": _session_record("conv-b", title="Second"),
        },
    )
    repo = _FakeRepo(queries)

    summary_hits = await repo.search_summary_hits("storage", limit=5, origins=["chatgpt"], since="2025-01-01")

    assert queries.last_batch_ids == ["conv-b", "conv-a"]
    assert [hit.session_id for hit in summary_hits] == ["conv-b", "conv-a"]
    assert [hit.rank for hit in summary_hits] == [1, 2]
    assert [hit.message_id for hit in summary_hits] == ["msg-conv-b", "msg-conv-a"]
    assert [hit.snippet for hit in summary_hits] == ["snippet for conv-b", "snippet for conv-a"]
    assert [hit.lane_rank for hit in summary_hits] == [1, 2]
    assert [hit.raw_score for hit in summary_hits] == [1.0, 2.0]
    assert [hit.score_components for hit in summary_hits] == [{"bm25_raw": 1.0}, {"bm25_raw": 2.0}]


@pytest.mark.asyncio
async def test_repository_search_summary_hits_prioritize_attachment_identity_evidence() -> None:
    hits = SessionSearchResult.from_ids(["conv-b", "conv-a"])
    attachment_hit = SessionSearchEvidenceRow(
        session_id="conv-a",
        rank=1,
        message_id="msg-attachment",
        snippet="attachment identity attachment.provider_file_id=drive-file-1",
        match_surface="attachment",
        retrieval_lane="attachment",
    )
    queries = _FakeQueries(
        hits=hits,
        attachment_hits=[attachment_hit],
        records_by_id={
            "conv-a": _session_record("conv-a", title="Attachment Match"),
            "conv-b": _session_record("conv-b", title="Message Match"),
        },
    )
    repo = _FakeRepo(queries)

    summary_hits = await repo.search_summary_hits("drive-file-1", limit=5, origins=["gemini"])

    assert queries.last_batch_ids == ["conv-a", "conv-b"]
    assert [hit.session_id for hit in summary_hits] == ["conv-a", "conv-b"]
    assert [hit.rank for hit in summary_hits] == [1, 2]
    assert summary_hits[0].match_surface == "attachment"
    assert summary_hits[0].retrieval_lane == "attachment"
    assert summary_hits[0].message_id == "msg-attachment"
    assert summary_hits[1].match_surface == "message"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected_snippet"),
    [
        ("provider-attachment-218", "native.attachment=provider-attachment-218"),
        ("drive-file-218", "native.file=drive-file-218"),
        ("drive-root-218", "native.drive=drive-root-218"),
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
    repo = SessionRepository(backend=backend)
    try:
        parsed = parse_chunked_prompt("gemini", payload, "fallback-id")
        await ingest_session(
            parsed,
            backend=backend,
        )

        hits = await repo.search_summary_hits(query, limit=5, origins=["gemini"])
    finally:
        await repo.close()

    assert len(hits) == 1
    hit = hits[0]
    assert hit.session_id == "aistudio-drive:gemini-attachment-identity"
    assert hit.match_surface == "attachment"
    assert hit.retrieval_lane == "attachment"
    assert hit.message_id == "aistudio-drive:gemini-attachment-identity:msg-doc"
    assert hit.snippet is not None
    assert expected_snippet in hit.snippet
    assert 'name="Project Plan"' in hit.snippet


@pytest.mark.asyncio
async def test_repository_search_and_action_search_pass_ordered_ids_to_hydration() -> None:
    hits = SessionSearchResult.from_ids(["conv-b", "conv-a"])
    queries = _FakeQueries(
        hits=hits,
        records_by_id={
            "conv-a": _session_record("conv-a", title="First"),
            "conv-b": _session_record("conv-b", title="Second"),
        },
    )
    repo = _FakeRepo(queries)

    search_result = await repo.search("storage", limit=5)
    assert [str(session.id) for session in search_result] == ["conv-b", "conv-a"]
    assert repo.ordered_ids_seen == ["conv-b", "conv-a"]

    action_result = await repo.search_actions("storage", limit=5)
    assert [str(session.id) for session in action_result] == ["conv-b", "conv-a"]
    assert repo.ordered_ids_seen == ["conv-b", "conv-a"]


@pytest.mark.asyncio
async def test_get_archive_stats_groups_origins_from_origin_column(tmp_path: Path) -> None:
    """Regression for the #1743 stale-vocabulary bug: get_archive_stats grouped
    providers via ``SELECT source_name FROM sessions`` — the column is ``origin``
    post-split, so the query raises ``no such column: source_name`` whenever the
    method is reached. The only test of the calling path mocked the method, so
    the broken query shipped. Seed real sessions and assert the async stats path
    runs and groups by origin without raising.
    """
    backend = SQLiteBackend(db_path=tmp_path / "origin-stats.db")
    repo = SessionRepository(backend=backend)
    try:
        for idx in range(2):
            payload: JSONDocument = {
                "id": f"gemini-stats-{idx}",
                "displayName": f"Gemini stats {idx}",
                "chunkedPrompt": {"chunks": [{"id": "m", "role": "user", "text": "hi"}]},
            }
            await ingest_session(parse_chunked_prompt("gemini", payload, f"fallback-{idx}"), backend=backend)

        stats = await repo.get_archive_stats()
    finally:
        await repo.close()

    assert stats.total_sessions == 2
    assert stats.total_messages == 2
    assert stats.origins == {"aistudio-drive": 2}
