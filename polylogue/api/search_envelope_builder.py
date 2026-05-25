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
from typing import TYPE_CHECKING

from polylogue.surfaces.payloads import (
    ConversationSearchHitPayload,
    InvalidSearchCursorError,
    QueryMissDiagnosticsPayload,
    SearchEnvelope,
    build_search_envelope,
    decode_search_cursor,
    search_cursor_lane_matches_request,
)

if TYPE_CHECKING:
    from polylogue.operations import ArchiveOperations
    from polylogue.storage.repository import ConversationRepository


async def build_archive_search_envelope(
    operations: ArchiveOperations,
    repository: ConversationRepository,
    *,
    query: str,
    limit: int = 50,
    offset: int = 0,
    provider: str | None = None,
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
    from polylogue.archive.query.spec import ConversationQuerySpec

    decoded_cursor = decode_search_cursor(cursor) if cursor else None
    if decoded_cursor is not None and not search_cursor_lane_matches_request(decoded_cursor.lane, retrieval_lane):
        raise InvalidSearchCursorError(
            f"cursor was minted for retrieval_lane={decoded_cursor.lane!r} but this request is {retrieval_lane!r}"
        )

    # Advance the SQL fetch past the cursor anchor; the builder will drop
    # any straggler rows whose (score, conversation_id) sort at or before
    # the anchor under the lane's natural ordering.
    effective_offset = decoded_cursor.r if decoded_cursor is not None else offset
    spec = ConversationQuerySpec.from_params(
        {
            "query": query,
            "provider": provider,
            "since": since,
            "until": until,
            "retrieval_lane": retrieval_lane,
            "sort": sort,
            # Fetch one extra page worth so the post-fetch cursor trim
            # cannot starve the response when the anchor row drifts.
            "limit": limit + (limit if decoded_cursor is not None else 0),
            "offset": effective_offset,
            "cursor": cursor,
        },
        strict=True,
    )
    hits = await operations.search_conversation_hits(spec)
    total = await spec.count(repository)
    diagnostics_payload: QueryMissDiagnosticsPayload | None = None
    if not hits and spec.has_filters():
        with suppress(Exception):
            raw_diag = await operations.diagnose_query_miss(spec)
            diagnostics_payload = QueryMissDiagnosticsPayload.from_diagnostics(raw_diag)
    hit_payloads = [
        ConversationSearchHitPayload.from_search_hit(hit, message_count=hit.summary.message_count) for hit in hits
    ]
    resolved_lane = hits[0].retrieval_lane if hits else retrieval_lane
    return build_search_envelope(
        hit_payloads,
        total=total,
        limit=limit,
        offset=offset,
        query=query,
        retrieval_lane=resolved_lane,
        sort=sort,
        diagnostics=diagnostics_payload,
        cursor=decoded_cursor,
    )


__all__ = ["build_archive_search_envelope"]
