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
from time import monotonic
from typing import TYPE_CHECKING

from polylogue.operations.authority import authority_for_config
from polylogue.surfaces.authority import AuthorityEnvelope
from polylogue.surfaces.payloads import (
    QueryMissDiagnosticsPayload,
    SearchEnvelope,
    SessionSearchHitPayload,
    build_search_envelope,
)

if TYPE_CHECKING:
    from polylogue.api import Polylogue
    from polylogue.archive.query.search_hits import SessionSearchHit
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
    serving_identity: str = "direct",
    authority: AuthorityEnvelope | None = None,
    request_scope_fingerprint: str | None = None,
    result_scope_fingerprint: str | None = None,
) -> SearchEnvelope:
    """Build a :class:`SearchEnvelope` from an already-normalized query spec.

    Daemon HTTP accepts the full ``SessionQuerySpec`` filter surface,
    while the public Python API exposes a smaller keyword facade. Keeping the
    cursor, diagnostics, and hit-payload assembly here prevents those surfaces
    from drifting while still letting each caller own parameter parsing.
    """
    started_at = monotonic()
    display_limit = limit if limit is not None else (spec.limit or 50)
    display_offset = offset if offset is not None else spec.offset
    fetch_spec = replace(spec, limit=display_limit, offset=display_offset)
    hits: list[SessionSearchHit] = await facade.search_session_hits(fetch_spec)
    execution = getattr(hits, "execution", None)
    # A vector candidate page is not the archive-wide hybrid union. Counting
    # through the semantic-only list route can report zero beside lexical hits
    # and suppress continuation. As on the daemon route, qualify that total as
    # unknown; an action-only page likewise cannot use the dialogue count.
    ranked_only = bool(spec.similar_text or spec.similar_session_id or spec.retrieval_lane in {"hybrid", "actions"})
    total = None if ranked_only else await spec.count(facade.config)
    diagnostics_payload: QueryMissDiagnosticsPayload | None = None
    if not hits and spec.has_filters():
        with suppress(Exception):
            raw_diag = await facade.diagnose_query_miss(spec)
            diagnostics_payload = QueryMissDiagnosticsPayload.from_diagnostics(raw_diag)
    hit_payloads = [
        SessionSearchHitPayload.from_search_hit(hit, message_count=hit.summary.message_count) for hit in hits
    ]
    resolved_lane = hits[0].retrieval_lane if hits else spec.retrieval_lane
    envelope = build_search_envelope(
        hit_payloads,
        total=total,
        limit=display_limit,
        offset=display_offset,
        query=query if query is not None else _search_query_text(spec),
        retrieval_lane=resolved_lane,
        sort=spec.sort,
        diagnostics=diagnostics_payload,
        execution=execution,
        authority=authority
        or authority_for_config(
            facade.config,
            server_identity="daemon" if serving_identity == "daemon" else "direct",
            started_at=started_at,
        ).model_copy(
            update={
                "matched": total,
                "analyzed": len(hits),
                "request_scope_fingerprint": request_scope_fingerprint,
                "result_scope_fingerprint": result_scope_fingerprint,
            }
        ),
    )
    return envelope


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
    serving_identity: str = "direct",
    authority: AuthorityEnvelope | None = None,
    request_scope_fingerprint: str | None = None,
    result_scope_fingerprint: str | None = None,
) -> SearchEnvelope:
    """Build a :class:`SearchEnvelope` from an archive operations + repo pair.

    Centralised so CLI, MCP, daemon HTTP, and the Python API all assemble
    the envelope from the same primitives (#1266).

    ``cursor`` is an opaque keyset token previously returned as
    :attr:`SearchEnvelope.next_cursor` (#1268). When supplied, the
    producer relocates a present anchor in the current pinned relation and
    continues after it. A removed anchor uses its saved complete order key.
    Corpus-dependent reranking can revisit earlier rows; a cursor does not
    freeze the archive or the whole walk.
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
        serving_identity=serving_identity,
        authority=authority,
        request_scope_fingerprint=request_scope_fingerprint,
        result_scope_fingerprint=result_scope_fingerprint,
    )


__all__ = ["build_archive_search_envelope", "build_search_envelope_for_spec"]
