from __future__ import annotations

from datetime import datetime, timezone

from polylogue.archive.conversation.models import Conversation, ConversationSummary
from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.lib.roles import Role
from polylogue.lib.search_hits import (
    build_search_snippet,
    conversation_search_hit_from_conversation,
    conversation_search_hit_from_summary,
    search_hit_surface,
    search_query_text,
    search_terms,
)
from polylogue.types import ConversationId, Provider


def _conversation() -> Conversation:
    return Conversation(
        id=ConversationId("chatgpt:search"),
        provider=Provider.CHATGPT,
        title="Search evidence",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        messages=MessageCollection(
            messages=[
                Message(id="m1", role=Role.USER, text="intro text", provider=Provider.CHATGPT),
                Message(id="m2", role=Role.ASSISTANT, text="the exact Needle appears here", provider=Provider.CHATGPT),
            ]
        ),
    )


def test_search_query_text_and_terms_normalize_empty_and_split_terms() -> None:
    assert search_query_text(("  alpha beta  ", "", " gamma ")) == "alpha beta gamma"
    assert search_terms(("Alpha beta", "  GAMMA  ", "")) == ("alpha", "beta", "gamma")


def test_build_search_snippet_anchors_earliest_match_and_adds_ellipses() -> None:
    text = "prefix " * 20 + "needle " + "suffix " * 30

    snippet = build_search_snippet(text, ("needle",))

    assert snippet.startswith("...")
    assert "needle" in snippet
    assert snippet.endswith("...")


def test_search_hit_from_conversation_preserves_match_evidence() -> None:
    hit = conversation_search_hit_from_conversation(
        _conversation(),
        query_terms=("needle",),
        rank=3,
        retrieval_lane="dialogue",
        score=0.42,
    )

    assert hit.conversation_id == "chatgpt:search"
    assert hit.message_id == "m2"
    assert hit.match_surface == "message"
    assert hit.snippet == "the exact Needle appears here"
    assert hit.score == 0.42
    assert hit.with_message_count(7).summary.message_count == 7


def test_search_hit_from_summary_and_lane_surface_are_explicit() -> None:
    summary = ConversationSummary(
        id=ConversationId("claude-ai:summary"),
        provider=Provider.CLAUDE_AI,
        title="Summary",
    )

    hit = conversation_search_hit_from_summary(
        summary,
        rank=1,
        retrieval_lane="attachment",
        match_surface="attachment",
        message_id=None,
        snippet="provider_meta.fileId=abc",
    )

    assert hit.conversation_id == "claude-ai:summary"
    assert hit.match_surface == "attachment"
    assert search_hit_surface("actions") == "action"
    assert search_hit_surface("hybrid") == "hybrid"
    assert search_hit_surface("semantic") == "semantic"
    assert search_hit_surface("dialogue") == "message"
