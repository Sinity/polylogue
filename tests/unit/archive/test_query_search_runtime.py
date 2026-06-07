from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from polylogue.archive.models import Session
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.query.plan_execution import list_for_plan
from polylogue.archive.query.retrieval_search import (
    fetch_batched_filtered_sessions,
    score_action_search_text,
    search_action_results,
    search_hybrid_results,
    search_query_terms,
    search_query_text,
    session_action_search_score,
)
from polylogue.archive.query.search_hits import (
    plan_has_search_hit_evidence,
    search_hits_for_plan,
)
from polylogue.config import Config, Source
from polylogue.protocols import VectorProvider
from polylogue.types import Provider
from tests.infra.archive_scenarios import native_session_id_for
from tests.infra.builders import make_conv, make_msg
from tests.infra.storage_records import SessionBuilder


@dataclass(frozen=True)
class _Request:
    limit: int | None = None
    offset: int = 0

    def with_limit(self, limit: int) -> _Request:
        return replace(self, limit=limit)

    def with_offset(self, offset: int) -> _Request:
        return replace(self, offset=offset)


def _session(session_id: str, *, text: str = "needle here", updated_hour: int = 0) -> Session:
    return make_conv(
        id=session_id,
        provider=Provider.CLAUDE_CODE,
        updated_at=datetime(2026, 4, 23, updated_hour, tzinfo=timezone.utc),
        messages=[make_msg(id=f"{session_id}-m1", role="assistant", text=text)],
    )


def test_search_query_text_terms_and_scoring_are_normalized() -> None:
    plan = SessionQueryPlan(query_terms=("Alpha",), contains_terms=("Beta", ""))

    assert search_query_text(plan) == "Alpha Beta"
    assert search_query_terms(plan) == ("alpha", "beta")
    assert score_action_search_text("Alpha beta gamma", query_text="alpha beta", terms=("alpha", "beta")) == 30.0
    assert score_action_search_text("no match", query_text="alpha", terms=("beta",)) == 0.0


def test_session_action_search_score_uses_best_match_and_bonus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "polylogue.archive.semantic.facts.build_session_semantic_facts",
        lambda _session: SimpleNamespace(
            actions=[
                SimpleNamespace(search_text="alpha beta"),
                SimpleNamespace(search_text="beta only"),
                SimpleNamespace(search_text=None),
            ]
        ),
    )

    score = session_action_search_score(
        _session("conv-score"),
        query_text="alpha beta",
        terms=("alpha", "beta"),
    )

    assert score == 31.0


@pytest.mark.asyncio
async def test_search_action_results_fallback_batches_scores_and_keeps_best(monkeypatch: pytest.MonkeyPatch) -> None:
    conv_a = _session("conv-a", updated_hour=9)
    conv_b = _session("conv-b", updated_hour=11)
    conv_c = _session("conv-c", updated_hour=10)
    repository = SimpleNamespace()
    repository.list_by_query = AsyncMock(
        side_effect=lambda request: {
            0: [conv_a, conv_b],
            2: [conv_c],
        }.get(request.offset, [])
    )

    async def _candidate_record_query_for(_plan: SessionQueryPlan, _repository: object) -> tuple[_Request, bool]:
        return (_Request(), False)

    monkeypatch.setattr(
        "polylogue.archive.query.retrieval_candidates.candidate_record_query_for", _candidate_record_query_for
    )
    monkeypatch.setattr("polylogue.archive.query.retrieval_candidates.candidate_batch_limit", lambda _plan: 2)
    monkeypatch.setattr(
        "polylogue.archive.query.runtime.apply_common_filters", lambda _plan, batch, *, sql_pushed: batch
    )
    monkeypatch.setattr(
        "polylogue.archive.query.retrieval_search.session_action_search_score",
        lambda session, *, query_text, terms: {
            "conv-a": 1.0,
            "conv-b": 6.0,
            "conv-c": 4.0,
        }[str(session.id)],
    )

    from polylogue.archive.query.retrieval_search import search_action_results_fallback

    results = await search_action_results_fallback(
        SessionQueryPlan(query_terms=("needle",)),
        repository,
        limit=2,
    )

    assert [str(session.id) for session in results] == ["conv-b", "conv-c"]


@pytest.mark.asyncio
async def test_search_action_results_uses_ready_path_and_raises_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.errors import DatabaseError

    repository = SimpleNamespace(search_actions=AsyncMock())
    direct = [_session("conv-direct")]

    monkeypatch.setattr(
        "polylogue.archive.query.retrieval_candidates.action_search_ready", AsyncMock(return_value=True)
    )

    repository.search_actions.return_value = direct
    ready_results = await search_action_results(
        SessionQueryPlan(query_terms=("needle",), origins=("chatgpt-export",)),
        repository,
        limit=3,
    )

    assert [str(session.id) for session in ready_results] == ["conv-direct"]
    repository.search_actions.assert_awaited_once_with("needle", limit=3, providers=["chatgpt"])

    repository.search_actions.reset_mock(side_effect=True, return_value=True)
    repository.search_actions.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        await search_action_results(SessionQueryPlan(query_terms=("needle",)), repository, limit=2)

    monkeypatch.setattr(
        "polylogue.archive.query.retrieval_candidates.action_search_ready", AsyncMock(return_value=False)
    )
    with pytest.raises(DatabaseError, match="Action search index is not fresh"):
        await search_action_results(SessionQueryPlan(query_terms=("needle",)), repository, limit=2)


@pytest.mark.asyncio
async def test_search_hybrid_results_orders_fused_ids_and_skips_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    conv_text = _session("conv-text")
    conv_action = _session("conv-action")
    conv_vector = _session("conv-vector")
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

    results, lane_ranks = await search_hybrid_results(
        SessionQueryPlan(
            query_terms=("needle",),
            vector_provider=cast(
                VectorProvider,
                SimpleNamespace(
                    model="stub",
                    upsert=lambda session_id, messages: None,
                    query=lambda text, limit=10: [],
                ),
            ),
        ),
        repository,
        limit=4,
    )

    assert [str(session.id) for session in results] == [
        "conv-action",
        "conv-text",
        "conv-vector",
    ]
    # Verify lane rank contributions
    assert lane_ranks["conv-action"]["action"] == 1
    assert lane_ranks["conv-text"]["text"] == 1
    assert lane_ranks["conv-vector"]["vector"] == 1


@pytest.mark.asyncio
async def test_fetch_batched_filtered_sessions_deduplicates_and_respects_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conv_a = _session("conv-a")
    conv_b = _session("conv-b")
    conv_c = _session("conv-c")
    repository = SimpleNamespace()
    repository.list_by_query = AsyncMock(
        side_effect=lambda request: {
            0: [conv_a, conv_b],
            2: [conv_a, conv_c],
        }.get(request.offset, [])
    )

    async def _candidate_record_query_for(_plan: SessionQueryPlan, _repository: object) -> tuple[_Request, bool]:
        return (_Request(), False)

    monkeypatch.setattr(
        "polylogue.archive.query.retrieval_candidates.candidate_record_query_for", _candidate_record_query_for
    )
    monkeypatch.setattr("polylogue.archive.query.retrieval_candidates.candidate_batch_limit", lambda _plan: 2)
    monkeypatch.setattr("polylogue.archive.query.runtime.apply_full_filters", lambda _plan, batch, *, sql_pushed: batch)

    results = await fetch_batched_filtered_sessions(
        SessionQueryPlan(limit=3),
        repository,
    )

    assert [str(session.id) for session in results] == ["conv-a", "conv-b", "conv-c"]


@pytest.mark.asyncio
async def test_list_for_plan_preserves_rank_order_for_search_without_explicit_sort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    older_but_better_ranked = _session("conv-rank-1", updated_hour=9)
    newer_but_worse_ranked = _session("conv-rank-2", updated_hour=12)

    async def _fetch_candidates(
        _plan: SessionQueryPlan,
        _repository: object,
        *,
        summaries: bool,
    ) -> tuple[list[Session], bool]:
        assert summaries is False
        return [older_but_better_ranked, newer_but_worse_ranked], True

    monkeypatch.setattr("polylogue.archive.query.plan_execution.fetch_candidates", _fetch_candidates)

    results = await list_for_plan(
        SessionQueryPlan(query_terms=("needle",), sort=None),
        SimpleNamespace(),
    )

    assert [str(session.id) for session in results] == ["conv-rank-1", "conv-rank-2"]


@pytest.mark.asyncio
async def test_list_for_plan_explicit_date_sort_overrides_rank_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    older_but_better_ranked = _session("conv-rank-1", updated_hour=9)
    newer_but_worse_ranked = _session("conv-rank-2", updated_hour=12)

    async def _fetch_candidates(
        _plan: SessionQueryPlan,
        _repository: object,
        *,
        summaries: bool,
    ) -> tuple[list[Session], bool]:
        assert summaries is False
        return [older_but_better_ranked, newer_but_worse_ranked], True

    monkeypatch.setattr("polylogue.archive.query.plan_execution.fetch_candidates", _fetch_candidates)

    results = await list_for_plan(
        SessionQueryPlan(query_terms=("needle",), sort="date"),
        SimpleNamespace(),
    )

    assert [str(session.id) for session in results] == ["conv-rank-2", "conv-rank-1"]


@pytest.mark.asyncio
async def test_search_hits_for_plan_handles_empty_and_lexical_paths(tmp_path: Path) -> None:
    """``search_hits_for_plan`` executes over the archive.

    The function now takes ``(plan, config)`` (no repository) and resolves
    hits through ``archive_search_hits``. This pins the empty-evidence,
    whitespace-only, and lexical (``dialogue``) contracts against a real
    seeded ``index.db``.
    """
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    render_root = tmp_path / "render"
    render_root.mkdir(parents=True, exist_ok=True)
    db_path = archive_root / "index.db"

    (
        SessionBuilder(db_path, "conv-summary")
        .provider(Provider.CHATGPT.value)
        .title("Needle Doc")
        .updated_at("2026-04-22T12:00:00+00:00")
        .add_message("m1", role="user", text="needle in the haystack here")
        .save()
    )

    config = Config(
        archive_root=archive_root,
        render_root=render_root,
        sources=[Source(name="test", path=tmp_path / "inbox")],
        db_path=db_path,
    )

    # A plan with no search-bearing fields carries no evidence.
    assert plan_has_search_hit_evidence(SessionQueryPlan()) is False

    # Whitespace-only query terms degrade to no hits.
    assert await search_hits_for_plan(SessionQueryPlan(query_terms=("   ",)), config) == []

    # Lexical (dialogue) lane returns evidence-bearing hits with the native
    # session id, the dialogue lane label, and an FTS snippet.
    lexical_hits = await search_hits_for_plan(
        SessionQueryPlan(
            query_terms=("needle",),
            origins=("chatgpt-export",),
            limit=5,
            since=datetime(2026, 4, 20, tzinfo=timezone.utc),
        ),
        config,
    )

    native_id = native_session_id_for("chatgpt", "conv-summary")
    assert [hit.session_id for hit in lexical_hits] == [native_id]
    assert lexical_hits[0].retrieval_lane == "dialogue"
    assert "needle" in (lexical_hits[0].snippet or "")
