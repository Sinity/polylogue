"""Per-hit why-this-matched evidence wiring (#1267, slice B of #873).

Pins that ``matched_terms`` and ``score_components`` actually flow through
the substrate-side search paths into ``SessionSearchHit`` / the
``SearchEnvelope``:

- FTS-only (dialogue lane): tokenized query terms + ``bm25_raw`` component
- Hybrid (RRF fusion): per-lane ``<lane>_rank`` / ``<lane>_rrf`` components
- Vector-only (semantic lane): ``vector_distance`` score_kind, single term
- Attachment-identity lane: identifier as term, ``score_kind=None``
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from polylogue.archive.query.search_hits import (
    _hybrid_score_components,
    session_search_hit_from_summary,
)
from polylogue.archive.session.domain_models import SessionSummary
from polylogue.core.enums import Origin
from polylogue.core.types import SessionId
from polylogue.storage.search.query_support import extract_match_terms
from polylogue.surfaces.payloads import SessionSearchHitPayload


def _summary(session_id: str = "chatgpt:hit") -> SessionSummary:
    return SessionSummary(
        id=SessionId(session_id),
        origin=Origin.CHATGPT_EXPORT,
        title="Sample",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )


class TestExtractMatchTerms:
    """``extract_match_terms`` is the single source of truth for the
    ``matched_terms`` field on FTS-driven hits."""

    def test_plain_terms_are_lowercased_and_deduped(self) -> None:
        assert extract_match_terms("Refactor refactor SCHEMA") == ("refactor", "schema")

    def test_fts_operators_are_stripped(self) -> None:
        assert extract_match_terms("refactor AND schema NOT test") == (
            "refactor",
            "schema",
            "test",
        )

    def test_prefix_asterisk_is_stripped(self) -> None:
        assert extract_match_terms("refactor*") == ("refactor",)

    def test_quoted_phrase_yields_constituent_tokens(self) -> None:
        # Phrase-search punctuation is stripped; the reader still expects
        # the literal tokens to highlight.
        assert extract_match_terms('"null pointer exception"') == (
            "null",
            "pointer",
            "exception",
        )

    def test_empty_query_returns_empty_tuple(self) -> None:
        assert extract_match_terms("") == ()
        assert extract_match_terms("   ") == ()


class TestDialogueExplanation:
    """Dialogue (FTS5) hits carry tokenized terms + bm25_raw component."""

    def test_dialogue_hit_surfaces_matched_terms_and_bm25_component(self) -> None:
        terms = extract_match_terms("refactor schema")
        score = -7.42
        hit = session_search_hit_from_summary(
            _summary(),
            rank=1,
            retrieval_lane="dialogue",
            match_surface="message",
            message_id="msg-1",
            snippet="…schema refactor lands…",
            score=score,
            matched_terms=terms,
            score_components={"bm25_raw": score},
            score_kind="bm25",
        )
        payload = SessionSearchHitPayload.from_search_hit(hit)
        match = payload.match
        assert match.retrieval_lane == "dialogue"
        assert match.score_kind == "bm25"
        assert match.matched_terms == ("refactor", "schema")
        assert match.score_components == {"bm25_raw": score}
        assert match.score == score


class TestHybridExplanation:
    """Hybrid hits expose per-lane rank + RRF contribution for every lane
    that contributed to the fused score (#1267 acceptance criterion 2)."""

    def test_hybrid_components_for_two_lane_overlap(self) -> None:
        lane_info: dict[str, int | None] = {"text": 1, "action": None, "vector": 2}
        components, fused = _hybrid_score_components(lane_info)
        assert fused is not None and fused > 0
        # Exactly the contributing lanes appear in the components.
        assert components["text_rank"] == 1.0
        assert components["text_rrf"] == pytest.approx(1 / 61)
        assert components["vector_rank"] == 2.0
        assert components["vector_rrf"] == pytest.approx(1 / 62)
        assert "action_rank" not in components
        assert "action_rrf" not in components
        # Fused score is the sum of all *_rrf contributions.
        assert fused == pytest.approx(components["text_rrf"] + components["vector_rrf"])

    def test_hybrid_components_for_all_three_lanes(self) -> None:
        # AC: "hybrid hit exposes both dialogue lane rank AND actions lane
        # rank when both contribute" — extended to the full lane set.
        lane_info: dict[str, int | None] = {"text": 1, "action": 3, "vector": 2}
        components, fused = _hybrid_score_components(lane_info)
        assert fused is not None
        for lane in ("text", "action", "vector"):
            assert f"{lane}_rank" in components
            assert f"{lane}_rrf" in components
        assert fused == pytest.approx(components["text_rrf"] + components["action_rrf"] + components["vector_rrf"])

    def test_hybrid_components_empty_when_no_lane_contributes(self) -> None:
        components, fused = _hybrid_score_components({"text": None, "vector": None})
        assert components == {}
        assert fused is None

    def test_hybrid_hit_round_trips_through_envelope_payload(self) -> None:
        lane_info: dict[str, int | None] = {"text": 1, "vector": 2}
        components, fused = _hybrid_score_components(lane_info)
        hit = session_search_hit_from_summary(
            _summary("chatgpt:hybrid"),
            rank=1,
            retrieval_lane="hybrid",
            match_surface="hybrid",
            message_id="msg-1",
            snippet="…hybrid match…",
            score=fused,
            matched_terms=("schema",),
            score_components=components,
            score_kind="rrf",
        )
        payload = SessionSearchHitPayload.from_search_hit(hit)
        match = payload.match
        assert match.score_kind == "rrf"
        # Both lane decompositions survived envelope serialization.
        assert match.score_components["text_rank"] == 1.0
        assert match.score_components["vector_rank"] == 2.0
        assert match.score is not None and match.score == pytest.approx(
            match.score_components["text_rrf"] + match.score_components["vector_rrf"]
        )


class TestSemanticExplanation:
    """Vector-only hits carry ``vector_distance`` score_kind. The raw
    distance lives in ``score``; ``matched_terms`` reflects the natural
    language probe rather than tokenized FTS terms."""

    def test_semantic_hit_carries_vector_distance_score_kind(self) -> None:
        hit = session_search_hit_from_summary(
            _summary("chatgpt:semantic"),
            rank=1,
            retrieval_lane="semantic",
            match_surface="semantic",
            message_id="msg-9",
            snippet=None,
            score=0.0421,
            matched_terms=("how does the daemon converge",),
            score_components={},
            score_kind="vector_distance",
        )
        payload = SessionSearchHitPayload.from_search_hit(hit)
        assert payload.match.score_kind == "vector_distance"
        assert payload.match.score == 0.0421
        assert payload.match.matched_terms == ("how does the daemon converge",)


class TestAttachmentExplanation:
    """Identity-only attachment hits carry the matched identifier as the
    single term and no numeric score."""

    def test_attachment_hit_carries_identifier_and_null_score_kind(self) -> None:
        hit = session_search_hit_from_summary(
            _summary("chatgpt:attach"),
            rank=1,
            retrieval_lane="attachment",
            match_surface="attachment",
            message_id="msg-3",
            snippet="attachment: file-abc123",
            score=None,
            matched_terms=("file-abc123",),
            score_components={},
            score_kind=None,
        )
        payload = SessionSearchHitPayload.from_search_hit(hit)
        assert payload.match.score_kind is None
        assert payload.match.score is None
        assert payload.match.matched_terms == ("file-abc123",)
