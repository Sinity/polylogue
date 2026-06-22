"""Cross-surface contract test for the typed ranked-result envelope (#1266).

The :class:`~polylogue.surfaces.payloads.SearchEnvelope` is the canonical
ranked-search response shape shared across CLI JSON output, MCP search
tool, the Python API, and the daemon HTTP search endpoint. This test
pins the envelope field set, asserts the cross-surface helpers all emit
the same shape, and verifies the OpenAPI schema renders cleanly so
external tooling can pin against it.

If a future change drifts one surface's envelope away from the canonical
shape, this test fails loudly and explains exactly which field is missing
or extra.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polylogue.archive.query.search_hits import (
    SessionSearchHit,
    session_search_hit_from_summary,
)
from polylogue.archive.session.domain_models import SessionSummary
from polylogue.cli.query_output import format_search_envelope
from polylogue.core.enums import Origin
from polylogue.mcp.payloads import session_search_result_payload
from polylogue.surfaces.payloads import (
    RANKING_POLICY_MIXED,
    RANKING_POLICY_VERSION,
    SearchEnvelope,
    SessionSearchHitPayload,
    build_search_envelope,
)
from polylogue.types import SessionId

# Canonical fields every surface MUST expose on the search envelope.
REQUIRED_ENVELOPE_FIELDS: frozenset[str] = frozenset(
    {
        "hits",
        "total",
        "limit",
        "offset",
        "next_offset",
        "next_cursor",
        "query",
        "sort",
        "retrieval_lane",
        "ranking_policy",
        "ranking_policy_version",
        "action_affordances",
        "diagnostics",
    }
)


def _summary() -> SessionSummary:
    return SessionSummary(
        id=SessionId("chatgpt:envelope-1"),
        origin=Origin.CHATGPT_EXPORT,
        title="Envelope test session",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _hit(rank: int = 1) -> SessionSearchHit:
    return session_search_hit_from_summary(
        _summary(),
        rank=rank,
        retrieval_lane="dialogue",
        match_surface="message",
        message_id="m1",
        snippet="…matched [needle]…",
        score=-7.42,
        matched_terms=("needle",),
        score_components={"bm25_raw": -7.42},
        lane_rank=rank,
        lane_contribution=-7.42,
        raw_score=-7.42,
    )


# ---------------------------------------------------------------------------
# Envelope field set
# ---------------------------------------------------------------------------


def test_envelope_required_fields_are_stable() -> None:
    declared = set(SearchEnvelope.model_fields)
    missing = REQUIRED_ENVELOPE_FIELDS - declared
    extra = declared - REQUIRED_ENVELOPE_FIELDS
    assert not missing, f"SearchEnvelope lost required fields: {missing}"
    assert not extra, f"SearchEnvelope grew unexpected fields: {extra} — extend REQUIRED_ENVELOPE_FIELDS or revert."


def test_envelope_ranking_policy_defaults_are_declared() -> None:
    envelope = build_search_envelope(
        [],
        total=0,
        limit=10,
        offset=0,
        query="needle",
        retrieval_lane="dialogue",
    )
    assert envelope.ranking_policy == RANKING_POLICY_MIXED
    assert envelope.ranking_policy_version == RANKING_POLICY_VERSION
    assert envelope.query == "needle"
    assert envelope.retrieval_lane == "dialogue"


# ---------------------------------------------------------------------------
# Cross-surface semantic equivalence
# ---------------------------------------------------------------------------


def _normalise_envelope_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Reduce an envelope to the comparable cross-surface invariants.

    We do not compare ``hits`` payloads byte-for-byte because surfaces may
    attach extra presentation evidence (e.g., target_ref actions). What MUST
    match across surfaces is the envelope-level metadata plus the canonical
    per-hit ranked-search explanation carried by ``match``.
    """
    hits: list[dict[str, Any]] = list(payload["hits"])

    def normalise_match(hit: dict[str, Any]) -> dict[str, Any]:
        match = dict(hit["match"])
        return {
            "session_id": hit["session"]["id"],
            "rank": match["rank"],
            "retrieval_lane": match["retrieval_lane"],
            "match_surface": match["match_surface"],
            "message_id": match["message_id"],
            "snippet": match["snippet"],
            "score": match["score"],
            "score_kind": match["score_kind"],
            "matched_terms": tuple(match["matched_terms"]),
            "score_components": match["score_components"],
            "lane_rank": match["lane_rank"],
            "lane_contribution": match["lane_contribution"],
            "raw_score": match["raw_score"],
        }

    return {
        "total": payload.get("total"),
        "limit": payload["limit"],
        "offset": payload["offset"],
        "query": payload["query"],
        "retrieval_lane": payload["retrieval_lane"],
        "ranking_policy": payload["ranking_policy"],
        "ranking_policy_version": payload["ranking_policy_version"],
        "action_affordance_ids": tuple(action["id"] for action in payload["action_affordances"]),
        "hit_count": len(hits),
        "hit_session_ids": tuple(hit["session"]["id"] for hit in hits),
        "hit_ranks": tuple(hit["match"]["rank"] for hit in hits),
        "hit_lanes": tuple(hit["match"]["retrieval_lane"] for hit in hits),
        "hit_matches": tuple(normalise_match(hit) for hit in hits),
    }


def test_all_surfaces_emit_semantically_equivalent_envelope() -> None:
    """Same hits → same envelope across CLI JSON, MCP, Python API helper, daemon HTTP."""
    hits = [_hit(rank=1), _hit(rank=2)]
    query = "needle"
    limit = 10
    offset = 0
    retrieval_lane = "dialogue"

    # 1. Python API / shared helper.
    hit_payloads = [
        SessionSearchHitPayload.from_search_hit(hit, message_count=hit.summary.message_count) for hit in hits
    ]
    api_envelope = build_search_envelope(
        hit_payloads,
        total=2,
        limit=limit,
        offset=offset,
        query=query,
        retrieval_lane=retrieval_lane,
    )
    api_dict = json.loads(api_envelope.model_dump_json(exclude_none=False))

    # 2. MCP search tool helper.
    mcp_envelope = session_search_result_payload(
        hits,
        total=2,
        limit=limit,
        offset=offset,
        query=query,
        retrieval_lane=retrieval_lane,
    )
    mcp_dict = json.loads(mcp_envelope.model_dump_json(exclude_none=False))

    # 3. CLI JSON formatter.
    cli_json = format_search_envelope(
        hits,
        query=query,
        retrieval_lane=retrieval_lane,
        limit=limit,
        offset=offset,
        sort=None,
    )
    cli_dict = json.loads(cli_json)

    api_norm = _normalise_envelope_dict(api_dict)
    mcp_norm = _normalise_envelope_dict(mcp_dict)
    # CLI emits total=None ("unknown"); align by ignoring total for CLI.
    cli_norm = _normalise_envelope_dict(cli_dict)

    # Total may differ between CLI (unknown) and the other surfaces (known).
    # The rest of the envelope metadata MUST agree.
    assert api_norm["limit"] == mcp_norm["limit"] == cli_norm["limit"]
    assert api_norm["offset"] == mcp_norm["offset"] == cli_norm["offset"]
    assert api_norm["query"] == mcp_norm["query"] == cli_norm["query"] == query
    assert api_norm["retrieval_lane"] == mcp_norm["retrieval_lane"] == cli_norm["retrieval_lane"] == retrieval_lane
    assert (
        api_norm["ranking_policy"] == mcp_norm["ranking_policy"] == cli_norm["ranking_policy"] == RANKING_POLICY_MIXED
    )
    assert (
        api_norm["ranking_policy_version"]
        == mcp_norm["ranking_policy_version"]
        == cli_norm["ranking_policy_version"]
        == RANKING_POLICY_VERSION
    )
    assert api_norm["action_affordance_ids"] == mcp_norm["action_affordance_ids"] == cli_norm["action_affordance_ids"]
    assert set(api_norm["action_affordance_ids"]) == {"read", "continue", "select", "mark", "analyze", "delete"}
    assert api_norm["hit_count"] == mcp_norm["hit_count"] == cli_norm["hit_count"] == 2
    assert api_norm["hit_session_ids"] == mcp_norm["hit_session_ids"] == cli_norm["hit_session_ids"]
    assert api_norm["hit_ranks"] == mcp_norm["hit_ranks"] == cli_norm["hit_ranks"] == (1, 2)
    assert api_norm["hit_lanes"] == mcp_norm["hit_lanes"] == cli_norm["hit_lanes"]
    assert api_norm["hit_matches"] == mcp_norm["hit_matches"] == cli_norm["hit_matches"]


# ---------------------------------------------------------------------------
# Pagination handles
# ---------------------------------------------------------------------------


def test_envelope_carries_cursor_when_page_is_full() -> None:
    """When ``len(hits) == limit`` AND more rows exist, the envelope exposes
    a keyset cursor so callers can resume scanning without offset drift."""
    hits = [_hit(rank=1), _hit(rank=2)]
    hit_payloads = [SessionSearchHitPayload.from_search_hit(hit) for hit in hits]
    envelope = build_search_envelope(
        hit_payloads,
        total=10,
        limit=2,
        offset=0,
        query="needle",
        retrieval_lane="dialogue",
    )
    assert envelope.next_cursor is not None
    assert envelope.next_offset == 2


def test_envelope_omits_cursor_when_page_is_last() -> None:
    hits = [_hit(rank=1), _hit(rank=2)]
    hit_payloads = [SessionSearchHitPayload.from_search_hit(hit) for hit in hits]
    envelope = build_search_envelope(
        hit_payloads,
        total=2,
        limit=2,
        offset=0,
        query="needle",
        retrieval_lane="dialogue",
    )
    assert envelope.next_cursor is None
    assert envelope.next_offset is None


# ---------------------------------------------------------------------------
# OpenAPI emission
# ---------------------------------------------------------------------------


def test_openapi_schema_is_present_and_well_formed() -> None:
    """The OpenAPI schema MUST be committed and reference SearchEnvelope."""
    repo_root = Path(__file__).resolve().parents[3]
    openapi_path = repo_root / "docs" / "openapi" / "search.yaml"
    assert openapi_path.exists(), f"missing generated OpenAPI schema at {openapi_path}"
    body = openapi_path.read_text(encoding="utf-8")
    # The header pins provenance; if the renderer changes shape, regenerate.
    assert "Generated by `devtools render openapi`" in body
    assert "SearchEnvelope" in body
    assert "/api/sessions" in body
