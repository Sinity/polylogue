"""Contract pin for ranked-search per-hit explanation payload (#873).

The ranked-search envelope is the shared evidence carrier across CLI JSON,
MCP, API, and daemon. These tests pin the field set of
``SessionSearchMatchPayload`` and the ``score_kind`` interpretation,
and record bounded contract evidence so downstream consumers can see what
the surface promises.
"""

from __future__ import annotations

from datetime import datetime, timezone

from polylogue.archive.query.search_hits import (
    SessionSearchHit,
    default_score_kind,
    session_search_hit_from_summary,
)
from polylogue.archive.session.domain_models import SessionSummary
from polylogue.core.enums import Origin
from polylogue.surfaces.payloads import (
    SessionSearchHitPayload,
    SessionSearchMatchPayload,
)
from polylogue.types import SessionId


def _summary() -> SessionSummary:
    return SessionSummary(
        id=SessionId("chatgpt:explain"),
        origin=Origin.CHATGPT_EXPORT,
        title="Explain me",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _dialogue_hit() -> SessionSearchHit:
    return session_search_hit_from_summary(
        _summary(),
        rank=1,
        retrieval_lane="dialogue",
        match_surface="message",
        message_id="m1",
        snippet="…matched [needle]…",
        score=-7.42,
        matched_terms=("needle",),
        score_components={},
    )


def _hybrid_hit() -> SessionSearchHit:
    components = {
        "text_rank": 1.0,
        "text_rrf": 1.0 / 61,
        "vector_rank": 2.0,
        "vector_rrf": 1.0 / 62,
    }
    return session_search_hit_from_summary(
        _summary(),
        rank=1,
        retrieval_lane="hybrid",
        match_surface="hybrid",
        message_id="m1",
        snippet="…matched [needle]…",
        score=components["text_rrf"] + components["vector_rrf"],
        matched_terms=("needle",),
        score_components=components,
        lane_rank=1,
        lane_contribution=components["text_rrf"],
        raw_score=components["text_rrf"] + components["vector_rrf"],
    )


def test_default_score_kind_per_lane_is_documented() -> None:
    assert default_score_kind("dialogue") == "bm25"
    assert default_score_kind("auto") == "bm25"
    assert default_score_kind("hybrid") == "rrf"
    assert default_score_kind("semantic") == "vector_distance"
    # Lanes without a numeric score (attachment, actions) return None so
    # consumers do not render a score column for them.
    assert default_score_kind("attachment") is None
    assert default_score_kind("actions") is None


def test_dialogue_hit_carries_bm25_score_kind() -> None:
    hit = _dialogue_hit()
    assert hit.score_kind == "bm25"
    payload = SessionSearchHitPayload.from_search_hit(hit)
    assert payload.match.score_kind == "bm25"
    assert payload.match.score == -7.42
    assert payload.match.matched_terms == ("needle",)
    assert payload.match.retrieval_lane == "dialogue"
    assert payload.match.match_surface == "message"


def test_hybrid_hit_carries_rrf_score_and_per_lane_components() -> None:
    hit = _hybrid_hit()
    payload = SessionSearchHitPayload.from_search_hit(hit)
    assert payload.match.score_kind == "rrf"
    assert payload.match.score is not None and payload.match.score > 0
    # Per-lane RRF explanation is preserved end-to-end.
    assert "text_rank" in payload.match.score_components
    assert "text_rrf" in payload.match.score_components
    assert "vector_rank" in payload.match.score_components
    assert "vector_rrf" in payload.match.score_components
    assert payload.match.lane_rank == 1
    assert payload.match.lane_contribution == payload.match.score_components["text_rrf"]
    assert payload.match.raw_score == payload.match.score
    # Fused score equals the sum of lane RRF contributions.
    expected = payload.match.score_components["text_rrf"] + payload.match.score_components["vector_rrf"]
    assert abs(payload.match.score - expected) < 1e-12


def test_match_payload_required_field_set_is_stable() -> None:
    """The ``match`` payload's declared fields must include the why-this-matched
    evidence required by #873 — drop a field and this test fails loudly."""
    declared = set(SessionSearchMatchPayload.model_fields)
    required = {
        "rank",
        "retrieval_lane",
        "match_surface",
        "message_id",
        "snippet",
        "score",
        "score_kind",
        "matched_terms",
        "score_components",
        "lane_rank",
        "lane_contribution",
        "raw_score",
    }
    missing = required - declared
    assert not missing, f"SessionSearchMatchPayload lost required fields: {missing}"


def test_explanation_shape_contract_evidence() -> None:
    """Record bounded evidence of the ranked-hit explanation shape so the
    cross-surface contract is auditable from the verification dashboard."""
