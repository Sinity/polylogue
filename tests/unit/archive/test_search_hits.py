from __future__ import annotations

from datetime import datetime, timezone

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.archive.query.search_hits import (
    DEFAULT_SEARCH_SNIPPET_MAX_CHARS,
    bound_search_snippet,
    build_search_snippet,
    search_hit_surface,
    search_query_text,
    search_terms,
    session_search_hit_from_session,
    session_search_hit_from_summary,
)
from polylogue.archive.session.domain_models import Session, SessionSummary
from polylogue.core.enums import Origin
from polylogue.core.types import SessionId


def _session() -> Session:
    return Session(
        id=SessionId("chatgpt:search"),
        origin=Origin.CHATGPT_EXPORT,
        title="Search evidence",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        messages=MessageCollection(
            messages=[
                Message(id="m1", role=Role.USER, text="intro text", origin=Origin.CHATGPT_EXPORT),
                Message(
                    id="m2",
                    role=Role.ASSISTANT,
                    text="the exact Needle appears here",
                    origin=Origin.CHATGPT_EXPORT,
                ),
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


def test_bound_search_snippet_never_returns_full_payload() -> None:
    snippet = "alpha\n" + ("payload " * 200) + "omega"

    bounded = bound_search_snippet(snippet)

    assert bounded is not None
    assert len(bounded) <= DEFAULT_SEARCH_SNIPPET_MAX_CHARS
    assert bounded.endswith("...")
    assert "\n" not in bounded


def test_search_hit_from_session_preserves_match_evidence() -> None:
    hit = session_search_hit_from_session(
        _session(),
        query_terms=("needle",),
        rank=3,
        retrieval_lane="dialogue",
        score=0.42,
    )

    assert hit.session_id == "chatgpt:search"
    assert hit.message_id == "m2"
    assert hit.match_surface == "message"
    assert hit.snippet == "the exact Needle appears here"
    assert hit.score == 0.42
    assert hit.with_message_count(7).summary.message_count == 7


def test_search_hit_from_summary_and_lane_surface_are_explicit() -> None:
    summary = SessionSummary(
        id=SessionId("claude-ai:summary"),
        origin=Origin.CLAUDE_AI_EXPORT,
        title="Summary",
    )

    hit = session_search_hit_from_summary(
        summary,
        rank=1,
        retrieval_lane="attachment",
        match_surface="attachment",
        message_id=None,
        snippet="provider_meta.fileId=abc",
    )

    assert hit.session_id == "claude-ai:summary"
    assert hit.match_surface == "attachment"
    assert search_hit_surface("actions") == "action"
    assert search_hit_surface("hybrid") == "hybrid"
    assert search_hit_surface("semantic") == "semantic"
    assert search_hit_surface("dialogue") == "message"


def test_search_hit_from_summary_bounds_provider_snippet() -> None:
    summary = SessionSummary(
        id=SessionId("claude-ai:summary"),
        origin=Origin.CLAUDE_AI_EXPORT,
        title="Summary",
    )

    hit = session_search_hit_from_summary(
        summary,
        rank=1,
        retrieval_lane="dialogue",
        match_surface="message",
        message_id="msg",
        snippet="needle " + ("full transcript payload " * 100),
    )

    assert hit.snippet is not None
    assert len(hit.snippet) <= DEFAULT_SEARCH_SNIPPET_MAX_CHARS
    assert hit.snippet.endswith("...")
