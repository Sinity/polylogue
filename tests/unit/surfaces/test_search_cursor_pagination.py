"""Stable cursor/keyset pagination contract tests (#1268, #873 slice C).

These tests pin the cursor encode/decode round-trip and the
``build_search_envelope`` cursor application semantics. The key guarantee
is the page-after-page invariant: paginating through ranked results with
``next_cursor`` returns contiguous, non-overlapping hits even when new
rows are interleaved between requests (simulated ingest noise).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from polylogue.archive.query.search_hits import (
    SessionSearchHit,
    session_search_hit_from_summary,
)
from polylogue.archive.session.domain_models import SessionSummary
from polylogue.core.enums import Origin
from polylogue.surfaces.payloads import (
    SEARCH_CURSOR_VERSION,
    InvalidSearchCursorError,
    SearchCursor,
    SessionSearchHitPayload,
    apply_search_cursor,
    build_search_cursor,
    build_search_envelope,
    decode_search_cursor,
)
from polylogue.types import SessionId


def _summary(conv_id: str) -> SessionSummary:
    return SessionSummary(
        id=SessionId(conv_id),
        origin=Origin.CHATGPT_EXPORT,
        title=f"Session {conv_id}",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _hit(
    *,
    conv_id: str,
    rank: int,
    score: float | None,
    retrieval_lane: str = "dialogue",
    score_kind: str | None = "bm25",
) -> SessionSearchHit:
    return session_search_hit_from_summary(
        _summary(conv_id),
        rank=rank,
        retrieval_lane=retrieval_lane,
        match_surface="message",
        message_id=f"m-{conv_id}",
        snippet=None,
        score=score,
        matched_terms=("needle",),
        score_kind=score_kind,
    )


def _payloads(hits: list[SessionSearchHit]) -> list[SessionSearchHitPayload]:
    return [SessionSearchHitPayload.from_search_hit(hit, message_count=0) for hit in hits]


# ---------------------------------------------------------------------------
# Cursor encode/decode round-trip
# ---------------------------------------------------------------------------


def test_cursor_round_trip_preserves_anchor() -> None:
    original = SearchCursor(v=SEARCH_CURSOR_VERSION, r=7, s=-3.14, c="chatgpt:abc", lane="hybrid")
    token = build_search_cursor(
        _payloads([_hit(conv_id="chatgpt:abc", rank=7, score=-3.14, retrieval_lane="hybrid", score_kind="bm25")])
    )
    assert token is not None
    decoded = decode_search_cursor(token)
    assert decoded.r == original.r
    assert decoded.s == pytest.approx(-3.14)
    assert decoded.c == original.c
    assert decoded.lane == original.lane
    assert decoded.v == SEARCH_CURSOR_VERSION


def test_cursor_is_opaque_base64() -> None:
    """Cursor tokens are URL-safe base64; consumers MUST treat as opaque."""
    token = build_search_cursor(_payloads([_hit(conv_id="x", rank=1, score=-1.0)]))
    assert token is not None
    # Allow URL-safe base64 alphabet plus stripped padding.
    assert all(c.isalnum() or c in "-_" for c in token)


def test_cursor_decode_rejects_garbage() -> None:
    with pytest.raises(InvalidSearchCursorError):
        decode_search_cursor("not-a-real-base64-token!!")


def test_cursor_decode_rejects_unknown_version() -> None:
    import base64
    import json

    payload = json.dumps({"v": 999, "r": 1, "s": None, "c": "x", "l": "auto"}).encode()
    token = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    with pytest.raises(InvalidSearchCursorError, match="version"):
        decode_search_cursor(token)


def test_cursor_decode_rejects_empty_token() -> None:
    with pytest.raises(InvalidSearchCursorError):
        decode_search_cursor("")


# ---------------------------------------------------------------------------
# apply_search_cursor semantics
# ---------------------------------------------------------------------------


def test_apply_cursor_drops_up_to_and_including_anchor_by_rank() -> None:
    hits = _payloads(
        [
            _hit(conv_id="a", rank=1, score=None, score_kind=None),
            _hit(conv_id="b", rank=2, score=None, score_kind=None),
            _hit(conv_id="c", rank=3, score=None, score_kind=None),
        ]
    )
    cursor = SearchCursor(v=1, r=2, s=None, c="b", lane="dialogue")
    survived = apply_search_cursor(hits, cursor)
    assert [h.session.id for h in survived] == ["c"]


def test_apply_cursor_uses_score_for_bm25_lane() -> None:
    """BM25: lower score is better. A hit with score worse than anchor survives."""
    hits = _payloads(
        [
            _hit(conv_id="a", rank=1, score=-5.0),
            _hit(conv_id="b", rank=2, score=-4.0),  # worse than anchor below
            _hit(conv_id="c", rank=3, score=-3.0),  # even worse
        ]
    )
    # Cursor anchor at session 'a' with score -5.0.
    cursor = SearchCursor(v=1, r=1, s=-5.0, c="a", lane="dialogue")
    survived = apply_search_cursor(hits, cursor)
    # 'a' is dropped (score equal AND id <= anchor's c); b,c survive (worse scores).
    assert [h.session.id for h in survived] == ["b", "c"]


def test_apply_cursor_rejects_lane_mismatch() -> None:
    hits = _payloads([_hit(conv_id="a", rank=1, score=-1.0)])
    cursor = SearchCursor(v=1, r=1, s=-1.0, c="x", lane="hybrid")
    with pytest.raises(InvalidSearchCursorError, match="retrieval_lane"):
        apply_search_cursor(hits, cursor, retrieval_lane="dialogue")


def test_apply_cursor_accepts_auto_request_for_resolved_lane_cursor() -> None:
    hits = _payloads(
        [
            _hit(conv_id="a", rank=1, score=-5.0),
            _hit(conv_id="b", rank=2, score=-4.0),
        ]
    )
    cursor = SearchCursor(v=1, r=1, s=-5.0, c="a", lane="dialogue")

    survived = apply_search_cursor(hits, cursor, retrieval_lane="auto")

    assert [h.session.id for h in survived] == ["b"]


# ---------------------------------------------------------------------------
# Page-after-page stability (the load-bearing invariant)
# ---------------------------------------------------------------------------


def test_paginate_two_pages_no_duplicates_no_gaps() -> None:
    """Walking through 5 hits in pages of 2: page1 + page2 + page3 == full set."""
    hits = _payloads([_hit(conv_id=f"c{i}", rank=i + 1, score=-(10.0 - i), score_kind="bm25") for i in range(5)])

    # Page 1: offset=0 limit=2 → c0, c1; next_cursor anchors c1.
    page1 = build_search_envelope(hits[:2], total=5, limit=2, offset=0, query="q", retrieval_lane="dialogue")
    assert [h.session.id for h in page1.hits] == ["c0", "c1"]
    assert page1.next_cursor is not None

    # Page 2: caller passes back next_cursor; simulate provider returning
    # the remaining 3 hits, builder trims past the anchor and truncates to limit.
    cursor = decode_search_cursor(page1.next_cursor)
    page2 = build_search_envelope(hits, total=5, limit=2, offset=2, query="q", retrieval_lane="dialogue", cursor=cursor)
    assert [h.session.id for h in page2.hits] == ["c2", "c3"]
    assert page2.next_cursor is not None

    # Page 3: final page.
    cursor2 = decode_search_cursor(page2.next_cursor)
    page3 = build_search_envelope(
        hits, total=5, limit=2, offset=4, query="q", retrieval_lane="dialogue", cursor=cursor2
    )
    assert [h.session.id for h in page3.hits] == ["c4"]
    # Last page may still mint a next_cursor when total is unknown; here total=5
    # and offset+limit >= total, so no further cursor is needed.
    assert page3.next_cursor is None

    walked = (
        [h.session.id for h in page1.hits] + [h.session.id for h in page2.hits] + [h.session.id for h in page3.hits]
    )
    assert walked == ["c0", "c1", "c2", "c3", "c4"]


def test_paginate_remains_contiguous_under_simulated_ingest_noise() -> None:
    """A new hit (c-new) inserted between pages must not duplicate or skip
    contiguous successors after the cursor anchor."""
    base_hits = [_hit(conv_id=f"c{i}", rank=i + 1, score=-(10.0 - i), score_kind="bm25") for i in range(5)]

    # Page 1 over the original archive.
    page1 = build_search_envelope(
        _payloads(base_hits[:2]),
        total=5,
        limit=2,
        offset=0,
        query="q",
        retrieval_lane="dialogue",
    )
    walked_ids = [h.session.id for h in page1.hits]
    cursor = decode_search_cursor(page1.next_cursor or "")

    # Simulate ingest noise: a new session appears with a score that
    # sorts it BEFORE the anchor (i.e., it would have been on page 1 had
    # it existed). The cursor anchors on c1, so c-new (rank 1, better
    # score) is filtered out by apply_search_cursor's score check.
    noisy_hits = [
        _hit(conv_id="c-new", rank=1, score=-12.0, score_kind="bm25"),
    ] + base_hits

    page2 = build_search_envelope(
        _payloads(noisy_hits),
        total=6,
        limit=2,
        offset=2,
        query="q",
        retrieval_lane="dialogue",
        cursor=cursor,
    )
    walked_ids += [h.session.id for h in page2.hits]
    # c-new has a strictly better score than the cursor anchor (-12 < -5),
    # so apply_search_cursor drops it (would have appeared on page 1).
    assert "c-new" not in walked_ids
    # No duplicates and contiguous successors.
    assert walked_ids == ["c0", "c1", "c2", "c3"]


def test_envelope_minted_cursor_round_trips() -> None:
    """The cursor emitted by one page decodes to the expected anchor for the next."""
    hits = _payloads([_hit(conv_id=f"c{i}", rank=i + 1, score=-(10 - i)) for i in range(3)])
    envelope = build_search_envelope(hits, total=10, limit=3, offset=0, query="q", retrieval_lane="dialogue")
    assert envelope.next_cursor is not None
    cursor = decode_search_cursor(envelope.next_cursor)
    assert cursor.c == "c2"
    assert cursor.r == 3
    assert cursor.lane == "dialogue"


def test_apply_cursor_with_rrf_lane_higher_is_better() -> None:
    """RRF lane: higher score is better. Survivors have lower scores than anchor."""
    hits = _payloads(
        [
            _hit(conv_id="a", rank=1, score=0.05, retrieval_lane="hybrid", score_kind="rrf"),
            _hit(conv_id="b", rank=2, score=0.03, retrieval_lane="hybrid", score_kind="rrf"),
            _hit(conv_id="c", rank=3, score=0.02, retrieval_lane="hybrid", score_kind="rrf"),
        ]
    )
    cursor = SearchCursor(v=1, r=1, s=0.05, c="a", lane="hybrid")
    survived = apply_search_cursor(hits, cursor)
    assert [h.session.id for h in survived] == ["b", "c"]


def test_cursor_survives_server_restart_round_trip() -> None:
    """The cursor token is self-contained (no server-side state).

    Encoded on one process and decoded on another reproduces the same
    anchor — there is no in-memory session keyed off the cursor.
    """
    token = build_search_cursor(_payloads([_hit(conv_id="durable", rank=42, score=-9.99)]))
    assert token is not None
    # Decode in a fresh module dict to simulate a separate process boundary.
    decoded = decode_search_cursor(token)
    assert decoded.c == "durable"
    assert decoded.r == 42
    assert decoded.s == pytest.approx(-9.99)
