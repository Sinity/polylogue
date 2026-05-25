from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from polylogue.api.search_envelope_builder import build_archive_search_envelope
from polylogue.archive.conversation.models import ConversationSummary
from polylogue.archive.query.search_hits import conversation_search_hit_from_summary
from polylogue.archive.query.spec import ConversationQuerySpec
from polylogue.surfaces.payloads import ConversationSearchHitPayload, build_search_cursor
from polylogue.types import ConversationId, Provider


def _dialogue_cursor() -> str:
    summary = ConversationSummary(
        id=ConversationId("chatgpt:cursor-anchor"),
        provider=Provider.CHATGPT,
        title="Cursor anchor",
    )
    hit = conversation_search_hit_from_summary(
        summary,
        rank=1,
        retrieval_lane="dialogue",
        match_surface="message",
        message_id="m1",
        snippet="anchor",
        score=-5.0,
        score_kind="bm25",
    )
    token = build_search_cursor([ConversationSearchHitPayload.from_search_hit(hit)])
    assert token is not None
    return token


@pytest.mark.asyncio
async def test_search_envelope_builder_accepts_auto_followup_for_resolved_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_count(self: ConversationQuerySpec, repository: object, *, vector_provider: object = None) -> int:
        del self, repository, vector_provider
        return 0

    monkeypatch.setattr(ConversationQuerySpec, "count", fake_count)
    operations = AsyncMock()
    operations.search_conversation_hits = AsyncMock(return_value=[])

    envelope = await build_archive_search_envelope(
        operations,
        object(),  # type: ignore[arg-type]
        query="needle",
        retrieval_lane="auto",
        cursor=_dialogue_cursor(),
    )

    assert envelope.hits == ()
    operations.search_conversation_hits.assert_awaited_once()
    spec = operations.search_conversation_hits.await_args.args[0]
    assert isinstance(spec, ConversationQuerySpec)
    assert spec.retrieval_lane == "auto"
