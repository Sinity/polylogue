"""Property tests for cursor/keyset pagination stability (#1268).

These properties guarantee the page-after-page invariant under
hypothesis-generated rank orderings: walking a result set in pages with
``next_cursor`` produces a contiguous, duplicate-free prefix of the
ground truth ordering.
"""

from __future__ import annotations

from datetime import datetime, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.archive.query.search_hits import session_search_hit_from_summary
from polylogue.archive.session.domain_models import SessionSummary
from polylogue.core.enums import Origin
from polylogue.surfaces.payloads import (
    SessionSearchHitPayload,
    build_search_envelope,
    decode_search_cursor,
)
from polylogue.types import SessionId


def _payload(conv_id: str, rank: int, score: float) -> SessionSearchHitPayload:
    summary = SessionSummary(
        id=SessionId(conv_id),
        origin=Origin.CHATGPT_EXPORT,
        title=f"Session {conv_id}",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    hit = session_search_hit_from_summary(
        summary,
        rank=rank,
        retrieval_lane="dialogue",
        match_surface="message",
        message_id=f"m-{conv_id}",
        snippet=None,
        score=score,
        matched_terms=("needle",),
        score_kind="bm25",
    )
    return SessionSearchHitPayload.from_search_hit(hit, message_count=0)


# Strategy: distinct session ids (sorted) with strictly-decreasing
# bm25 scores so the natural lane ordering is unambiguous.
_n_hits = st.integers(min_value=3, max_value=15)
_page_size = st.integers(min_value=1, max_value=6)


@given(n=_n_hits, page=_page_size)
@settings(max_examples=40, deadline=None)
def test_full_walk_covers_all_hits_in_order(n: int, page: int) -> None:
    """Walking through N hits in pages of `page` returns the full ordered set."""
    hits = [_payload(conv_id=f"c{i:03d}", rank=i + 1, score=-float(100 - i)) for i in range(n)]

    walked: list[str] = []
    cursor_token: str | None = None
    offset = 0
    iterations = 0
    while True:
        iterations += 1
        if iterations > n + 5:  # safety net
            raise AssertionError(f"pagination did not terminate after {iterations} iterations")
        cursor = decode_search_cursor(cursor_token) if cursor_token else None
        # Surface fetches `offset..` rows; simulate by passing the full
        # remaining tail (offset onwards). build_search_envelope's cursor
        # trim plus limit truncation give the page.
        remaining = hits[offset:]
        if cursor is not None:
            # Caller advances fetch to cursor.r; the builder still trims.
            offset = cursor.r
            remaining = hits[offset:]
        envelope = build_search_envelope(
            remaining,
            total=n,
            limit=page,
            offset=offset,
            query="needle",
            retrieval_lane="dialogue",
            cursor=cursor,
        )
        page_ids = [h.session.id for h in envelope.hits]
        if not page_ids:
            break
        # No duplicates between pages.
        assert not (set(page_ids) & set(walked)), f"duplicate ids across pages: {page_ids} vs walked={walked}"
        walked.extend(page_ids)
        if envelope.next_cursor is None:
            break
        cursor_token = envelope.next_cursor

    assert walked == [f"c{i:03d}" for i in range(n)], f"walked={walked} expected={[f'c{i:03d}' for i in range(n)]}"


@given(n=_n_hits, page=_page_size)
@settings(max_examples=20, deadline=None)
def test_cursor_round_trip_is_deterministic(n: int, page: int) -> None:
    """The cursor emitted from the same page anchors the same successor."""
    hits = [_payload(conv_id=f"c{i:03d}", rank=i + 1, score=-float(100 - i)) for i in range(n)]
    page1 = build_search_envelope(hits[:page], total=n, limit=page, offset=0, query="q", retrieval_lane="dialogue")
    if page1.next_cursor is None:
        return  # single-page result; trivially stable
    page1_again = build_search_envelope(
        hits[:page], total=n, limit=page, offset=0, query="q", retrieval_lane="dialogue"
    )
    assert page1.next_cursor == page1_again.next_cursor
