"""Shared builder for the typed ranked-result search envelope (#1266).

The Python API exposes :meth:`polylogue.api.Polylogue.search_envelope` as
the typed entry point for ranked search. Daemon HTTP and MCP build the
same envelope from the same primitives. This module factors out the
build sequence — spec construction, hit fetch, total count, miss
diagnostics, hit payload conversion, envelope assembly — so the surface
adapters stay thin and the per-file LOC budget on
``polylogue/api/archive.py`` is respected.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import replace
from typing import TYPE_CHECKING

from polylogue.surfaces.payloads import (
    InvalidSearchCursorError,
    QueryMissDiagnosticsPayload,
    SearchEnvelope,
    SessionSearchHitPayload,
    build_search_envelope,
    decode_search_cursor,
    search_cursor_lane_matches_request,
)

if TYPE_CHECKING:
    from polylogue.api import Polylogue
    from polylogue.archive.query.spec import SessionQuerySpec


def _search_query_text(spec: SessionQuerySpec) -> str:
    plan = spec.to_plan()
    if plan.fts_terms:
        return " ".join(plan.fts_terms)
    if plan.similar_text:
        return plan.similar_text
    return ""


async def build_search_envelope_for_spec(
    facade: Polylogue,
    spec: SessionQuerySpec,
    *,
    limit: int | None = None,
    offset: int | None = None,
    query: str | None = None,
) -> SearchEnvelope:
    """Build a :class:`SearchEnvelope` from an already-normalized query spec.

    Daemon HTTP accepts the full ``SessionQuerySpec`` filter surface,
    while the public Python API exposes a smaller keyword facade. Keeping the
    cursor, diagnostics, and hit-payload assembly here prevents those surfaces
    from drifting while still letting each caller own parameter parsing.
    """
    display_limit = limit if limit is not None else (spec.limit or 50)
    display_offset = offset if offset is not None else spec.offset
    decoded_cursor = decode_search_cursor(spec.cursor) if spec.cursor else None
    if decoded_cursor is not None and not search_cursor_lane_matches_request(decoded_cursor.lane, spec.retrieval_lane):
        raise InvalidSearchCursorError(
            f"cursor was minted for retrieval_lane={decoded_cursor.lane!r} but this request is {spec.retrieval_lane!r}"
        )

    fetch_spec = spec
    if decoded_cursor is not None:
        # Advance the SQL fetch past the cursor anchor; the builder will drop
        # any straggler rows whose (score, session_id) sort at or before
        # the anchor under the lane's natural ordering.
        fetch_spec = replace(
            spec,
            offset=decoded_cursor.r,
            limit=(spec.limit or display_limit) + display_limit,
            cursor=spec.cursor,
        )

    hits = await facade.search_session_hits(fetch_spec)
    total = await spec.count(facade.config)
    diagnostics_payload: QueryMissDiagnosticsPayload | None = None
    if not hits and spec.has_filters():
        with suppress(Exception):
            raw_diag = await facade.diagnose_query_miss(spec)
            diagnostics_payload = QueryMissDiagnosticsPayload.from_diagnostics(raw_diag)
    hit_payloads = [
        SessionSearchHitPayload.from_search_hit(hit, message_count=hit.summary.message_count) for hit in hits
    ]
    resolved_lane = hits[0].retrieval_lane if hits else spec.retrieval_lane
    return build_search_envelope(
        hit_payloads,
        total=total,
        limit=display_limit,
        offset=display_offset,
        query=query if query is not None else _search_query_text(spec),
        retrieval_lane=resolved_lane,
        sort=spec.sort,
        diagnostics=diagnostics_payload,
        cursor=decoded_cursor,
    )


async def build_archive_search_envelope(
    facade: Polylogue,
    *,
    query: str,
    limit: int = 50,
    offset: int = 0,
    origin: str | None = None,
    since: str | None = None,
    until: str | None = None,
    retrieval_lane: str = "auto",
    sort: str | None = None,
    cursor: str | None = None,
) -> SearchEnvelope:
    """Build a :class:`SearchEnvelope` from an archive operations + repo pair.

    Centralised so CLI, MCP, daemon HTTP, and the Python API all assemble
    the envelope from the same primitives (#1266).

    ``cursor`` is an opaque keyset token previously returned as
    :attr:`SearchEnvelope.next_cursor` (#1268). When supplied, the
    underlying fetch is advanced past the cursor anchor and the response
    page begins strictly after it. The cursor's encoded rank is used as
    the effective offset, so a follow-up call with the same query and
    lane returns the contiguous successor page even if the archive has
    grown between requests.
    """
    from polylogue.archive.query.spec import SessionQuerySpec

    spec = SessionQuerySpec.from_params(
        {
            "query": query,
            "origin": origin,
            "since": since,
            "until": until,
            "retrieval_lane": retrieval_lane,
            "sort": sort,
            "limit": limit,
            "offset": offset,
            "cursor": cursor,
        },
        strict=True,
    )
    return await build_search_envelope_for_spec(
        facade,
        spec,
        limit=limit,
        offset=offset,
        query=query,
    )


__all__ = ["build_archive_search_envelope", "build_search_envelope_for_spec"]
