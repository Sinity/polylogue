from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from polylogue.archive.conversation.models import ConversationSummary
from polylogue.archive.models import Conversation
from polylogue.archive.query.plan import ConversationQueryPlan
from polylogue.archive.query.retrieval_search import (
    conversation_action_search_score,
    fetch_batched_filtered_conversations,
    score_action_search_text,
    search_action_results,
    search_hybrid_results,
    search_query_terms,
    search_query_text,
)
from polylogue.archive.query.search_hits import (
    ConversationSearchHit,
    plan_has_search_hit_evidence,
    search_hits_for_plan,
)
from polylogue.protocols import VectorProvider
from polylogue.types import ConversationId, Provider
from tests.infra.builders import make_conv, make_msg


@dataclass(frozen=True)
class _Request:
    limit: int | None = None
    offset: int = 0

    def with_limit(self, limit: int) -> _Request:
        return replace(self, limit=limit)

    def with_offset(self, offset: int) -> _Request:
        return replace(self, offset=offset)


def _conversation(conversation_id: str, *, text: str = "needle here", updated_hour: int = 0) -> Conversation:
    return make_conv(
        id=conversation_id,
        provider=Provider.CLAUDE_CODE,
        updated_at=datetime(2026, 4, 23, updated_hour, tzinfo=timezone.utc),
        messages=[make_msg(id=f"{conversation_id}-m1", role="assistant", text=text)],
    )


def test_search_query_text_terms_and_scoring_are_normalized() -> None:
    plan = ConversationQueryPlan(query_terms=("Alpha",), contains_terms=("Beta", ""))

    assert search_query_text(plan) == "Alpha Beta"
    assert search_query_terms(plan) == ("alpha", "beta")
    assert score_action_search_text("Alpha beta gamma", query_text="alpha beta", terms=("alpha", "beta")) == 30.0
    assert score_action_search_text("no match", query_text="alpha", terms=("beta",)) == 0.0


def test_conversation_action_search_score_uses_best_match_and_bonus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "polylogue.archive.semantic.facts.build_conversation_semantic_facts",
        lambda _conversation: SimpleNamespace(
            action_events=[
                SimpleNamespace(search_text="alpha beta"),
                SimpleNamespace(search_text="beta only"),
                SimpleNamespace(search_text=None),
            ]
        ),
    )

    score = conversation_action_search_score(
        _conversation("conv-score"),
        query_text="alpha beta",
        terms=("alpha", "beta"),
    )

    assert score == 31.0


@pytest.mark.asyncio
async def test_search_action_results_fallback_batches_scores_and_keeps_best(monkeypatch: pytest.MonkeyPatch) -> None:
    conv_a = _conversation("conv-a", updated_hour=9)
    conv_b = _conversation("conv-b", updated_hour=11)
    conv_c = _conversation("conv-c", updated_hour=10)
    repository = SimpleNamespace()
    repository.list_by_query = AsyncMock(
        side_effect=lambda request: {
            0: [conv_a, conv_b],
            2: [conv_c],
        }.get(request.offset, [])
    )

    async def _candidate_record_query_for(_plan: ConversationQueryPlan, _repository: object) -> tuple[_Request, bool]:
        return (_Request(), False)

    monkeypatch.setattr(
        "polylogue.archive.query.retrieval_candidates.candidate_record_query_for", _candidate_record_query_for
    )
    monkeypatch.setattr("polylogue.archive.query.retrieval_candidates.candidate_batch_limit", lambda _plan: 2)
    monkeypatch.setattr(
        "polylogue.archive.query.runtime.apply_common_filters", lambda _plan, batch, *, sql_pushed: batch
    )
    monkeypatch.setattr(
        "polylogue.archive.query.retrieval_search.conversation_action_search_score",
        lambda conversation, *, query_text, terms: {
            "conv-a": 1.0,
            "conv-b": 6.0,
            "conv-c": 4.0,
        }[str(conversation.id)],
    )

    results = await search_action_results(
        ConversationQueryPlan(query_terms=("needle",)),
        repository,
        limit=2,
    )

    assert [str(conversation.id) for conversation in results] == ["conv-b", "conv-c"]


@pytest.mark.asyncio
async def test_search_action_results_uses_ready_path_and_falls_back_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = SimpleNamespace(search_actions=AsyncMock())
    direct = [_conversation("conv-direct")]
    fallback = AsyncMock(return_value=[_conversation("conv-fallback")])

    monkeypatch.setattr(
        "polylogue.archive.query.retrieval_candidates.action_search_ready", AsyncMock(return_value=True)
    )
    monkeypatch.setattr("polylogue.archive.query.retrieval_search.search_action_results_fallback", fallback)

    repository.search_actions.return_value = direct
    ready_results = await search_action_results(
        ConversationQueryPlan(query_terms=("needle",), providers=(Provider.CHATGPT,)),
        repository,
        limit=3,
    )

    assert [str(conversation.id) for conversation in ready_results] == ["conv-direct"]
    repository.search_actions.assert_awaited_once_with("needle", limit=3, providers=["chatgpt"])
    fallback.assert_not_awaited()

    repository.search_actions.reset_mock(side_effect=True, return_value=True)
    repository.search_actions.side_effect = RuntimeError("boom")
    errored_results = await search_action_results(ConversationQueryPlan(query_terms=("needle",)), repository, limit=2)

    assert [str(conversation.id) for conversation in errored_results] == ["conv-fallback"]
    fallback.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_hybrid_results_orders_fused_ids_and_skips_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    conv_text = _conversation("conv-text")
    conv_action = _conversation("conv-action")
    conv_vector = _conversation("conv-vector")
    repository = SimpleNamespace(
        search=AsyncMock(return_value=[conv_text]),
        search_similar=AsyncMock(return_value=[conv_vector]),
    )

    monkeypatch.setattr(
        "polylogue.archive.query.retrieval_search.search_action_results",
        AsyncMock(return_value=[conv_action]),
    )
    monkeypatch.setattr(
        "polylogue.archive.query.retrieval_search.reciprocal_rank_fusion",
        lambda *_ranked: [
            ("conv-action", 0.9),
            ("conv-text", 0.8),
            ("missing", 0.7),
            ("conv-vector", 0.6),
        ],
    )

    results = await search_hybrid_results(
        ConversationQueryPlan(
            query_terms=("needle",),
            vector_provider=cast(
                VectorProvider,
                SimpleNamespace(
                    model="stub",
                    upsert=lambda conversation_id, messages: None,
                    query=lambda text, limit=10: [],
                ),
            ),
        ),
        repository,
        limit=4,
    )

    assert [str(conversation.id) for conversation in results] == [
        "conv-action",
        "conv-text",
        "conv-vector",
    ]


@pytest.mark.asyncio
async def test_fetch_batched_filtered_conversations_deduplicates_and_respects_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conv_a = _conversation("conv-a")
    conv_b = _conversation("conv-b")
    conv_c = _conversation("conv-c")
    repository = SimpleNamespace()
    repository.list_by_query = AsyncMock(
        side_effect=lambda request: {
            0: [conv_a, conv_b],
            2: [conv_a, conv_c],
        }.get(request.offset, [])
    )

    async def _candidate_record_query_for(_plan: ConversationQueryPlan, _repository: object) -> tuple[_Request, bool]:
        return (_Request(), False)

    monkeypatch.setattr(
        "polylogue.archive.query.retrieval_candidates.candidate_record_query_for", _candidate_record_query_for
    )
    monkeypatch.setattr("polylogue.archive.query.retrieval_candidates.candidate_batch_limit", lambda _plan: 2)
    monkeypatch.setattr("polylogue.archive.query.runtime.apply_full_filters", lambda _plan, batch, *, sql_pushed: batch)

    results = await fetch_batched_filtered_conversations(
        ConversationQueryPlan(limit=3),
        repository,
    )

    assert [str(conversation.id) for conversation in results] == ["conv-a", "conv-b", "conv-c"]


@pytest.mark.asyncio
async def test_search_hits_for_plan_handles_empty_simple_and_fallback_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = SimpleNamespace(search_summary_hits=AsyncMock())
    repository.search_summary_hits.return_value = [
        ConversationSearchHit(
            summary=ConversationSummary(id=ConversationId("conv-summary"), provider=Provider.CHATGPT),
            message_id=None,
            rank=1,
            retrieval_lane="dialogue",
            match_surface="message",
            snippet="needle",
        )
    ]

    assert plan_has_search_hit_evidence(ConversationQueryPlan()) is False
    assert await search_hits_for_plan(ConversationQueryPlan(query_terms=("   ",)), repository) == []

    monkeypatch.setattr("polylogue.archive.query.search_hits.plan_has_fields_matching", lambda _plan, _predicate: False)
    simple_hits = await search_hits_for_plan(
        ConversationQueryPlan(
            query_terms=("needle",),
            providers=(Provider.CHATGPT,),
            limit=5,
            since=datetime(2026, 4, 20, tzinfo=timezone.utc),
        ),
        repository,
    )

    assert [hit.conversation_id for hit in simple_hits] == ["conv-summary"]
    repository.search_summary_hits.assert_awaited_once_with(
        "needle",
        limit=5,
        providers=["chatgpt"],
        since="2026-04-20T00:00:00+00:00",
    )

    async def _list(_self: ConversationQueryPlan, _repository: object) -> list[object]:
        return [_conversation("conv-message", text="needle in message")]

    monkeypatch.setattr(ConversationQueryPlan, "list", _list)
    fallback_hits = await search_hits_for_plan(
        ConversationQueryPlan(similar_text="semantic needle", retrieval_lane="actions"),
        repository,
    )

    assert [hit.conversation_id for hit in fallback_hits] == ["conv-message"]
    assert fallback_hits[0].retrieval_lane == "semantic"
