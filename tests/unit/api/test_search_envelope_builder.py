from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from polylogue.api.search_envelope_builder import build_archive_search_envelope, build_search_envelope_for_spec
from polylogue.archive.query.search_hits import session_search_hit_from_summary
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.archive.session.domain_models import SessionSummary
from polylogue.core.enums import Origin
from polylogue.core.types import SessionId
from polylogue.surfaces.payloads import SessionSearchHitPayload, build_search_cursor


def _dialogue_cursor() -> str:
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
    token = build_search_cursor([SessionSearchHitPayload.from_search_hit(hit)])
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
    assert envelope.offset == 20
