"""Contracts for the bounded staged topic-pack workflow."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from polylogue.product.workflows import TopicPackRequest, TopicPackResult, build_topic_pack


class FakeStore:
    def __init__(self) -> None:
        self.session = SimpleNamespace(
            id="claude-code-session:s1",
            title="Topic",
            messages=[
                SimpleNamespace(
                    id="m1",
                    text="bounded evidence",
                    blocks=[{"content_hash": bytes.fromhex("ab" * 32)}],
                ),
                SimpleNamespace(id="m2", text="second message", blocks=[]),
                SimpleNamespace(id="m3", text="must be bounded", blocks=[]),
            ],
        )

    async def search_summary_hits(
        self, query: str, limit: int = 20, origins: list[str] | None = None, since: str | None = None
    ) -> list[Any]:
        return [SimpleNamespace(summary=SimpleNamespace(id="claude-code-session:s1", title="Topic"), rank=1)][:limit]

    async def search_similar(self, text: str, limit: int = 10, vector_provider: Any = None) -> list[Any]:
        return [SimpleNamespace(id="claude-code-session:s2", title="Semantic topic")][:limit]

    async def get(self, session_id: str) -> Any:
        return self.session if session_id == str(self.session.id) else None

    async def resolve_id(self, id_prefix: str, *, strict: bool = False) -> str:
        return id_prefix

    async def list_summaries_by_query(self, query: Any) -> list[Any]:
        return []


@pytest.mark.asyncio
async def test_topic_pack_runs_without_vectors_and_reports_reason_and_hash_citation() -> None:
    result = await build_topic_pack(cast(Any, FakeStore()), TopicPackRequest("bounded evidence", max_messages=1))

    assert result.status == "ok"
    assert result.metadata["vector_status"] == "disabled"
    assert "vector expansion disabled" in result.gaps[0]
    assert result.metadata["content_hash_citations"] == 1
    assert result.context_pack[0]["citation"] == ("claude-code-session:s1::m1::block@sha256:" + "ab" * 32)
    assert len(result.context_pack) == 1


@pytest.mark.asyncio
async def test_topic_pack_vector_lane_is_provider_general_and_bounded() -> None:
    provider = object()
    result = await build_topic_pack(
        cast(Any, FakeStore()), TopicPackRequest("topic", vector_provider=cast(Any, provider), max_sessions=1)
    )

    assert result.metadata["vector_status"] == "ready"
    assert cast(dict[str, int], result.metadata["bounds"])["max_sessions"] == 1
    assert {item.reason for item in result.evidence} == {"fts"}
    assert "embedding" not in result.metadata["retrieval_channels_attempted"]


@pytest.mark.asyncio
async def test_topic_pack_keeps_both_lanes_and_fetches_past_overlapping_vector_hit() -> None:
    class OverlapStore(FakeStore):
        async def search_similar(self, text: str, limit: int = 10, vector_provider: Any = None) -> list[Any]:
            self.vector_limit = limit
            return [
                SimpleNamespace(id="claude-code-session:s1", title="Vector duplicate"),
                SimpleNamespace(id="claude-code-session:s2", title="New result"),
            ]

    store = OverlapStore()
    result = await build_topic_pack(
        cast(Any, store), TopicPackRequest("topic", vector_provider=cast(Any, object()), max_sessions=2)
    )
    assert store.vector_limit == 2
    assert len(result.sessions) == 2
    assert result.evidence[0].reason == "embedding/fts"


@pytest.mark.asyncio
async def test_seed_free_topic_uses_the_vector_lane_once_as_independent_retrieval() -> None:
    class SeedFreeStore(FakeStore):
        async def search_summary_hits(
            self, query: str, limit: int = 20, origins: list[str] | None = None, since: str | None = None
        ) -> list[Any]:
            return []

        async def search_similar(self, text: str, limit: int = 10, vector_provider: Any = None) -> list[Any]:
            self.vector_calls = getattr(self, "vector_calls", 0) + 1
            return [SimpleNamespace(id="claude-code-session:s2", title="Semantic topic")]

    store = SeedFreeStore()
    result = await build_topic_pack(cast(Any, store), TopicPackRequest("topic", vector_provider=cast(Any, object())))
    assert store.vector_calls == 1
    assert result.evidence[0].reason == "embedding"
    assert result.metadata["retrieval_channels_attempted"] == ["fts", "embedding", "time", "content"]


@pytest.mark.asyncio
async def test_topic_pack_uses_limited_message_iterator() -> None:
    class IterStore(FakeStore):
        async def iter_messages(self, session_id: str, *, limit: int) -> Any:
            self.requested_limit = limit
            for message in self.session.messages[:limit]:
                yield message

    store = IterStore()
    result = await build_topic_pack(cast(Any, store), TopicPackRequest("topic", max_messages=1))
    assert store.requested_limit == 1
    assert len(result.context_pack) == 1


@pytest.mark.asyncio
async def test_topic_pack_prefers_bounded_paged_read_with_hydrated_block_hash() -> None:
    class PagedStore(FakeStore):
        async def get_messages_paginated(self, session_id: str, *, limit: int, offset: int) -> Any:
            self.requested_limit = limit
            return self.session.messages[:limit], 3, None

        async def get(self, session_id: str) -> Any:
            raise AssertionError("the bounded page route must avoid eager session hydration")

    store = PagedStore()
    result = await build_topic_pack(cast(Any, store), TopicPackRequest("topic", max_messages=1))
    assert store.requested_limit == 1
    assert result.metadata["content_hash_citations"] == 1
    assert result.context_pack[0]["citation"].endswith("sha256:" + "ab" * 32)


def test_signals_require_issue_identifiers_and_stop_after_sixteen_candidates() -> None:
    from polylogue.product.workflows import _signals

    result = _signals(
        [{"text": "the issue is intermittent; issue with this; bead design; issue #42; bead polylogue-1tyq5"}]
    )
    assert result["issues"] == ["#42", "bead polylogue-1tyq5"]
    many = _signals([{"text": " ".join(f"path{i}.py" for i in range(1000))}])
    assert len(many["files"]) == 16


@pytest.mark.asyncio
async def test_topic_pack_citation_tracks_content_hash_drift() -> None:
    store = FakeStore()
    first = await build_topic_pack(cast(Any, store), TopicPackRequest("bounded evidence", max_messages=1))
    store.session.messages[0].blocks[0]["content_hash"] = bytes.fromhex("cd" * 32)
    second = await build_topic_pack(cast(Any, store), TopicPackRequest("bounded evidence", max_messages=1))

    assert first.context_pack[0]["citation"] != second.context_pack[0]["citation"]
    assert str(second.context_pack[0]["citation"]).endswith("sha256:" + "cd" * 32)
    assert second.metadata["quality_baseline"] == {
        "kind": "no-vector-fts",
        "session_count": 1,
        "session_ids": ["claude-code-session:s1"],
        "product_claims": False,
    }


def test_topic_pack_rejects_unbounded_or_empty_requests() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        TopicPackRequest(" ")
    with pytest.raises(ValueError, match="max_messages"):
        TopicPackRequest("topic", max_messages=0)


def test_topic_pack_to_dict_is_json_serializable_for_model_timestamps() -> None:
    class Model(BaseModel):
        created_at: Any

    result = TopicPackResult(
        status="ok",
        query="topic",
        sessions=(Model(created_at=datetime(2026, 9, 28, tzinfo=UTC)),),
        evidence=(),
        timeline=(),
        context_pack=(),
        gaps=(),
    )
    assert json.loads(json.dumps(result.to_dict()))["sessions"][0]["created_at"] == "2026-09-28T00:00:00Z"
