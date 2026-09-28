from __future__ import annotations

from dataclasses import fields
from unittest.mock import AsyncMock

import pytest

from polylogue.api import search_envelope_builder
from polylogue.api.search_envelope_builder import build_archive_search_envelope, build_search_envelope_for_spec
from polylogue.archive.query.expression import compile_expression_into
from polylogue.archive.query.search_hits import session_search_hit_from_summary
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.archive.session.domain_models import SessionSummary
from polylogue.core.enums import Origin
from polylogue.core.types import SessionId
from polylogue.surfaces.authority import build_authority_envelope
from polylogue.surfaces.cursor_identity import search_cursor_request_identity
from polylogue.surfaces.payloads import (
    InvalidSearchCursorError,
    SessionSearchHitPayload,
    build_search_cursor,
)


def _dialogue_cursor() -> str:
    token = build_search_cursor([_dialogue_cursor_payload()])
    assert token is not None
    return token


async def _fake_count(self: SessionQuerySpec, config: object, *, vector_provider: object = None) -> int:
    del self, config, vector_provider
    return 0


@pytest.mark.asyncio
async def test_search_envelope_builder_accepts_auto_followup_for_resolved_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SessionQuerySpec, "count", _fake_count)
    operations = AsyncMock()
    operations.search_session_hits = AsyncMock(return_value=[])

    envelope = await build_archive_search_envelope(
        operations,
        query="needle",
        retrieval_lane="auto",
        cursor=_dialogue_cursor(),
    )

    assert envelope.hits == ()
    operations.search_session_hits.assert_awaited_once()
    spec = operations.search_session_hits.await_args.args[0]
    assert isinstance(spec, SessionQuerySpec)
    assert spec.retrieval_lane == "auto"


@pytest.mark.asyncio
async def test_spec_builder_preserves_filters_when_advancing_cursor_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SessionQuerySpec, "count", _fake_count)
    operations = AsyncMock()
    operations.search_session_hits = AsyncMock(return_value=[])
    operations.diagnose_query_miss = AsyncMock(side_effect=RuntimeError("diagnostics unavailable"))
    cursor = _dialogue_cursor()
    spec = SessionQuerySpec.from_params(
        {
            "query": "needle",
            "origin": "chatgpt-export",
            "tag": "important",
            "since": "2026-05-01",
            "limit": 10,
            "offset": 20,
            "cursor": cursor,
        }
    )

    envelope = await build_search_envelope_for_spec(
        operations,
        spec,
        limit=10,
        offset=20,
    )

    operations.search_session_hits.assert_awaited_once()
    fetch_spec = operations.search_session_hits.await_args.args[0]
    assert isinstance(fetch_spec, SessionQuerySpec)
    assert fetch_spec.origins == ("chatgpt-export",)
    assert fetch_spec.tags == ("important",)
    assert fetch_spec.since == "2026-05-01"
    assert fetch_spec.offset == 1
    assert fetch_spec.limit == 20
    assert fetch_spec.cursor == cursor
    assert envelope.limit == 10
    # The page was fetched after the cursor anchor (rank 1), not at the
    # request's offset, so the envelope reports the offset it actually used.
    assert envelope.offset == 1


@pytest.mark.asyncio
async def test_spec_builder_authority_reports_full_matches_and_processed_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anti-vacuity: a capped page must distinguish the 100-match total from 10 fetched hits."""

    async def count_matches(self: SessionQuerySpec, config: object, *, vector_provider: object = None) -> int:
        del self, config, vector_provider
        return 100

    monkeypatch.setattr(SessionQuerySpec, "count", count_matches)
    operations = AsyncMock()
    summary = SessionSummary(id=SessionId("chatgpt:page-hit"), origin=Origin.CHATGPT_EXPORT, title="Page hit")
    hit = session_search_hit_from_summary(
        summary,
        rank=1,
        retrieval_lane="dialogue",
        match_surface="message",
        message_id="m1",
        snippet="needle",
        score=-1.0,
        score_kind="bm25",
    )
    operations.search_session_hits = AsyncMock(return_value=[hit] * 10)
    monkeypatch.setattr(
        search_envelope_builder,
        "authority_for_config",
        lambda *args, **kwargs: build_authority_envelope(
            archive_epoch="epoch",
            generation_id="generation",
            tier_schema_versions={},
            server_identity="direct",
            started_at=kwargs.get("started_at"),
        ),
    )

    envelope = await build_search_envelope_for_spec(
        operations,
        SessionQuerySpec.from_params({"query": "needle", "limit": 10}),
        limit=10,
    )

    assert envelope.total == 100
    assert len(envelope.hits) == 10
    assert envelope.authority is not None
    assert envelope.authority.matched == 100
    assert envelope.authority.analyzed == 10


@pytest.mark.asyncio
async def test_spec_builder_rejects_cursor_from_a_different_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SessionQuerySpec, "count", _fake_count)
    operations = AsyncMock()
    operations.search_session_hits = AsyncMock(return_value=[])

    first = compile_expression_into("needle", SessionQuerySpec.from_params({"limit": 1}))
    cursor = build_search_cursor(
        [_dialogue_cursor_payload()],
        request_identity=search_cursor_request_identity(
            {
                field.name: getattr(first, field.name)
                for field in fields(first)
                if field.name not in {"cursor", "offset", "limit", "vector_provider", "predicates"}
            }
        ),
    )
    assert cursor is not None

    second = compile_expression_into("other", SessionQuerySpec.from_params({"limit": 1, "cursor": cursor}))
    with pytest.raises(InvalidSearchCursorError, match="different ranked-search request"):
        await build_search_envelope_for_spec(operations, second, limit=1)
    operations.search_session_hits.assert_not_awaited()


def _dialogue_cursor_payload() -> SessionSearchHitPayload:
    summary = SessionSummary(
        id=SessionId("chatgpt:cursor-anchor"),
        origin=Origin.CHATGPT_EXPORT,
        title="Cursor anchor",
    )
    hit = session_search_hit_from_summary(
        summary,
        rank=1,
        retrieval_lane="dialogue",
        match_surface="message",
        message_id="m1",
        snippet="anchor",
        score=-5.0,
        score_kind="bm25",
    )
    return SessionSearchHitPayload.from_search_hit(hit)


@pytest.mark.asyncio
async def test_filter_only_structured_spec_lists_sessions_with_absolute_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A structured filter with no search text still returns its sessions.

    Anti-vacuity: gate the fallback on ``boolean_predicate`` alone again and a
    compact ``origin:`` filter reports no hits; rank from the display offset
    again and the cursor page's ranks restart at 1 instead of continuing at 6.
    """
    import polylogue.archive.query.search_hits as search_hits

    monkeypatch.setattr(SessionQuerySpec, "count", _fake_count)

    def hit_from_session(
        session: SessionSummary, *, query_terms: tuple[str, ...], rank: int, **kwargs: object
    ) -> object:
        del query_terms
        return session_search_hit_from_summary(
            session, rank=rank, retrieval_lane="auto", match_surface="session", message_id=None, snippet=""
        )

    monkeypatch.setattr(search_hits, "session_search_hit_from_session", hit_from_session)
    operations = AsyncMock()
    operations.search_session_hits = AsyncMock(return_value=[])
    operations.list_sessions_for_spec = AsyncMock(
        return_value=[
            SessionSummary(id=SessionId("chatgpt:a"), origin=Origin.CHATGPT_EXPORT, title="A"),
            SessionSummary(id=SessionId("chatgpt:b"), origin=Origin.CHATGPT_EXPORT, title="B"),
        ]
    )
    spec = SessionQuerySpec.from_params({"origin": "chatgpt-export", "limit": 2, "offset": 5})

    envelope = await build_search_envelope_for_spec(operations, spec, limit=2, offset=5)

    operations.list_sessions_for_spec.assert_awaited_once()
    assert [hit.match.rank for hit in envelope.hits] == [6, 7]


@pytest.mark.asyncio
async def test_filter_only_cursor_page_reports_its_own_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cursor page's continuation fields describe the page it fetched.

    Anti-vacuity: pass the request's display offset (0) to the envelope again
    and page two reports ``next_offset == 2`` while returning ranks 3-4.
    """
    import polylogue.archive.query.search_hits as search_hits

    async def count_ten(self: SessionQuerySpec, config: object, *, vector_provider: object = None) -> int:
        del self, config, vector_provider
        return 10

    monkeypatch.setattr(SessionQuerySpec, "count", count_ten)

    def hit_from_session(
        session: SessionSummary, *, query_terms: tuple[str, ...], rank: int, **kwargs: object
    ) -> object:
        del query_terms
        return session_search_hit_from_summary(
            session, rank=rank, retrieval_lane="auto", match_surface="session", message_id=None, snippet=""
        )

    monkeypatch.setattr(search_hits, "session_search_hit_from_session", hit_from_session)
    pages = [
        [SessionSummary(id=SessionId(f"chatgpt:{name}"), origin=Origin.CHATGPT_EXPORT, title=name) for name in names]
        for names in (("a", "b"), ("c", "d"))
    ]
    operations = AsyncMock()
    operations.search_session_hits = AsyncMock(return_value=[])
    operations.list_sessions_for_spec = AsyncMock(side_effect=pages)

    first = await build_search_envelope_for_spec(
        operations, SessionQuerySpec.from_params({"origin": "chatgpt-export", "limit": 2}), limit=2
    )
    assert first.next_cursor is not None
    second = await build_search_envelope_for_spec(
        operations,
        SessionQuerySpec.from_params({"origin": "chatgpt-export", "limit": 2, "cursor": first.next_cursor}),
        limit=2,
    )

    assert [hit.match.rank for hit in second.hits] == [3, 4]
    assert second.next_offset == 4
