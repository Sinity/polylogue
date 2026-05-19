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
    QueryMissDiagnosticsPayload,
    SearchEnvelope,
    build_search_envelope,
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
) -> SearchEnvelope:
    """Build a :class:`SearchEnvelope` from an archive operations + repo pair.

    Centralised so CLI, MCP, daemon HTTP, and the Python API all assemble
    the envelope from the same primitives (#1266).
    """
    from polylogue.archive.query.spec import ConversationQuerySpec

    spec = ConversationQuerySpec.from_params(
        {
            "query": query,
            "provider": provider,
            "since": since,
            "until": until,
            "retrieval_lane": retrieval_lane,
            "sort": sort,
            "limit": limit,
            "offset": offset,
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
    )


__all__ = ["build_archive_search_envelope"]
