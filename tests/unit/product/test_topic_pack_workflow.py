"""Contracts for the bounded staged topic-pack workflow."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.product.workflows import TopicPackRequest, build_topic_pack


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


def test_topic_pack_rejects_unbounded_or_empty_requests() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        TopicPackRequest(" ")
    with pytest.raises(ValueError, match="max_messages"):
        TopicPackRequest("topic", max_messages=0)
