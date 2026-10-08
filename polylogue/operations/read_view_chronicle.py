"""Pinned-archive execution for the CLI chronicle read view."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import TYPE_CHECKING

from polylogue.archive.hydration import (
    archive_message_query_row_to_domain,
    archive_message_to_domain,
    archive_summary_to_domain,
)
from polylogue.archive.message.models import Message
from polylogue.archive.query.sorting import OffsetSampledPage
from polylogue.core.enums import MaterialOrigin, Origin
from polylogue.operations.query_lowering import cli_read_request
from polylogue.surfaces.chronicle import (
    build_chronicle_projection_payload,
    build_chronicle_session_payload,
)

if TYPE_CHECKING:
    from polylogue.archive.query.plan import SessionQueryPlan
    from polylogue.archive.session.domain_models import Session, SessionSummary
    from polylogue.core.protocols import VectorProvider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSummary, ArchiveStore


DEFAULT_CHRONICLE_EDGE_LIMIT = 8
_PAGE_SIZE = 256
_AUTHORED_ORIGINS = frozenset({MaterialOrigin.HUMAN_AUTHORED.value, MaterialOrigin.ASSISTANT_AUTHORED.value})
_DIALOGUE_ROLES = frozenset({"user", "assistant"})
_POST_FILTER_CHUNK = 200


def _edge_limit(payload: Mapping[str, object]) -> int:
    projection = payload.get("projection")
    raw = projection.get("edge_limit") if isinstance(projection, Mapping) else None
    if raw is None:
        raw = payload.get("edge_limit", DEFAULT_CHRONICLE_EDGE_LIMIT)
    if isinstance(raw, bool):
        return int(raw) or 1
    if isinstance(raw, int):
        return max(raw, 1)
    if isinstance(raw, str):
        try:
            return max(int(raw), 1)
        except ValueError:
            pass
    return DEFAULT_CHRONICLE_EDGE_LIMIT


def _value(value: object) -> str:
    return str(getattr(value, "value", value))


def chronicle_edges(
    archive: ArchiveStore,
    session_id: str,
    edge_limit: int,
    *,
    origin: Origin,
) -> tuple[list[Message], list[Message], int]:
    """Collect bounded edge rows while counting the full composed transcript."""

    edge_rows = edge_limit * 5
    if not archive.has_prefix_lineage(session_id):
        roles = tuple(sorted(_DIALOGUE_ROLES))
        material_origins = tuple(sorted(_AUTHORED_ORIGINS))
        total = archive.count_session_messages(
            [session_id],
            roles=roles,
            message_type="message",
            material_origins=material_origins,
        )
        first_rows = archive.query_session_messages(
            [session_id],
            limit=edge_rows,
            sort_direction="asc",
            roles=roles,
            message_type="message",
            material_origins=material_origins,
        )
        last_rows = (
            archive.query_session_messages(
                [session_id],
                limit=edge_rows,
                offset=max(total - edge_rows, 0),
                sort_direction="asc",
                roles=roles,
                message_type="message",
                material_origins=material_origins,
            )
            if total > edge_rows
            else []
        )
        first_domain = [archive_message_query_row_to_domain(message) for message in first_rows]
        last_domain = [archive_message_query_row_to_domain(message) for message in last_rows]
        first_ids = {message.id for message in first_domain}
        return first_domain, [message for message in last_domain if message.id not in first_ids], total

    first: list[Message] = []
    last: deque[Message] = deque(maxlen=edge_rows)
    offset = 0
    total_matching = 0
    total_messages: int | None = None
    while total_messages is None or offset < total_messages:
        page = archive.read_session_page(session_id, limit=_PAGE_SIZE, offset=offset)
        messages = page.messages
        if total_messages is None:
            total_messages = page.total_message_count
            if total_messages is None:
                total_messages = len(messages)
        if not messages:
            break
        for message in messages:
            if (
                _value(message.role) not in _DIALOGUE_ROLES
                or _value(message.message_type) != "message"
                or _value(message.material_origin) not in _AUTHORED_ORIGINS
            ):
                continue
            domain_message = archive_message_to_domain(message, origin=origin)
            total_matching += 1
            if len(first) < edge_rows:
                first.append(domain_message)
            last.append(domain_message)
        offset += len(messages)
    first_ids = {message.id for message in first}
    last_messages = [message for message in last if message.id not in first_ids]
    return first, last_messages, total_matching


def _chronicle_plan(payload: Mapping[str, object], *, vector_provider: VectorProvider | None) -> SessionQueryPlan:
    params_raw = payload.get("params", {})
    if not isinstance(params_raw, Mapping):
        raise ValueError("chronicle params must be an object")
    params = {str(key): value for key, value in params_raw.items()}
    # A null limit is the declared five-session default, as an absent one is.
    if params.get("limit") is None:
        params["limit"] = 5
    query_terms = params.get("query", ())
    if not isinstance(query_terms, (list, tuple)):
        raise ValueError("chronicle query terms must be a list")
    params["query"] = list(query_terms)
    session_id = payload.get("session_id")
    if session_id is not None:
        params["conv_id"] = str(session_id)
    return cli_read_request(params).selection.to_plan(vector_provider=vector_provider)


def chronicle_needs_complete_scan(plan: SessionQueryPlan) -> bool:
    """Whether this chronicle selection must read and hydrate every candidate.

    A composed-count sort needs recomposed totals. Ranked retrieval settles
    the complete eligible relation before selecting its result window.
    """
    from polylogue.archive.query.archive_execution import _COMPOSED_COUNT_SORTS, _ranked_window

    return _ranked_window(plan) or plan.sort in _COMPOSED_COUNT_SORTS


def chronicle_payload_is_scan(payload: Mapping[str, object]) -> bool:
    """Whether a ``read.chronicle`` request is archive-scan work, decided before it runs."""
    return chronicle_needs_complete_scan(_chronicle_plan(payload, vector_provider=None))


def _select_summaries(
    payload: Mapping[str, object],
    *,
    archive: ArchiveStore,
    vector_provider: VectorProvider | None,
) -> list[SessionSummary]:
    """Run query candidates through the plan's filters, order and page cut."""

    from dataclasses import replace

    from polylogue.archive.hydration import archive_envelope_to_session
    from polylogue.archive.query.archive_execution import (
        _COMPOSED_COUNT_SORTS,
        _archive_summaries,
        _ranked_window,
    )

    plan = _chronicle_plan(payload, vector_provider=vector_provider)
    if _ranked_window(plan):
        # Membership and comparison keys are settled on the held frame before
        # a page is hydrated. A sample retains only its requested summary rows.
        from polylogue.archive.query.sorting import SessionReservoir

        ranked_reservoir = SessionReservoir[ArchiveSessionSummary](plan.sample) if plan.sample is not None else None
        remaining_offset = plan.offset

        def sample_batch(rows: list[ArchiveSessionSummary]) -> None:
            nonlocal remaining_offset
            skipped = min(remaining_offset, len(rows))
            remaining_offset -= skipped
            assert ranked_reservoir is not None
            ranked_reservoir.offer(rows[skipped:])

        rows = _archive_summaries(
            plan,
            archive,
            config=None,
            archive_root=archive.archive_root,
            default_limit=5,
            complete=ranked_reservoir is not None,
            on_batch=sample_batch if ranked_reservoir is not None else None,
            full_sort=plan.needs_content_loading() or plan.sort in _COMPOSED_COUNT_SORTS,
        )
        rows = ranked_reservoir.items() if ranked_reservoir is not None else rows[plan.offset :]
        return plan._finalize([archive_summary_to_domain(row) for row in rows])

    # A composed-count sort (messages/words/longest/tokens) ranks a lineage
    # child by its full inherited-prefix-plus-tail total, but the index only
    # stores that child's own tail count; windowing the SQL fetch by the
    # stored count first can exclude the child a composed rank would have
    # kept. Fetch every candidate uncut and rank the hydrated, composed
    # sessions instead, exactly as the generic session-list route does
    # (``archive_execution.read``'s ``composed_order``/``complete`` path).
    composed_order = plan.sort in _COMPOSED_COUNT_SORTS
    complete = chronicle_needs_complete_scan(plan)
    fetch_plan = replace(plan, limit=None, offset=0) if complete else plan
    if plan.needs_content_loading() or composed_order:
        # One hydration per candidate, chunk by chunk. A composed-count order
        # keeps only the best ``offset + limit`` sessions seen so far, so a
        # one-row page over a large archive never holds every transcript.
        # A sampled request draws uniformly from every qualified candidate
        # through a reservoir of the sample's size. A complete scan streams
        # its candidate summaries batch by batch, keeping only the summaries
        # of sessions still in that bound.
        bound = None if plan.limit is None else (plan.offset or 0) + plan.limit
        best: list[Session] = []
        reservoir: OffsetSampledPage[Session] | None = (
            OffsetSampledPage(offset=plan.offset or 0, sample=plan.sample, sort=plan._sort_sessions)
            if plan.sample
            else None
        )
        matched_ids: set[str] = set()
        summary_by_id: dict[str, SessionSummary] = {}

        def consume(rows: list[ArchiveSessionSummary]) -> None:
            nonlocal best
            for start in range(0, len(rows), _POST_FILTER_CHUNK):
                chunk = rows[start : start + _POST_FILTER_CHUNK]
                for row in chunk:
                    summary_by_id[row.session_id] = archive_summary_to_domain(row)
                sessions = [
                    archive_envelope_to_session(
                        archive.read_session(row.session_id),
                        display_label=row.display_label,
                        display_label_source=row.display_label_source,
                    )
                    for row in chunk
                ]
                kept = plan._apply_full_filters(sessions, sql_pushed=True)
                if composed_order and reservoir is not None:
                    reservoir.offer(kept)
                    best = reservoir.items()
                elif composed_order:
                    best = plan._sort_sessions([*best, *kept])
                    if bound is not None:
                        best = best[:bound]
                else:
                    matched_ids.update(str(session.id) for session in kept)
                if composed_order:
                    retained = {str(session.id) for session in best}
                    for session_id in [key for key in summary_by_id if key not in retained]:
                        del summary_by_id[session_id]

        rows = _archive_summaries(
            fetch_plan,
            archive,
            config=None,
            archive_root=archive.archive_root,
            default_limit=5,
            complete=complete,
            on_batch=consume if complete else None,
        )
        if not complete:
            consume(rows)
        if composed_order:
            ordered = [summary_by_id[str(session.id)] for session in best]
        else:
            ordered = [
                summary
                for summary in (archive_summary_to_domain(row) for row in rows)
                if str(summary.id) in matched_ids
            ]
    else:
        rows = _archive_summaries(
            fetch_plan,
            archive,
            config=None,
            archive_root=archive.archive_root,
            default_limit=5,
            complete=complete,
        )
        summaries = [archive_summary_to_domain(row) for row in rows]
        ordered = plan._apply_common_filters(summaries, sql_pushed=True)
    ranked = bool(plan.similar_text or plan.similar_session_id or plan.retrieval_lane in {"semantic", "hybrid"})
    if (plan.has_post_filters() or ranked or composed_order) and plan.offset:
        ordered = ordered[plan.offset :]
    return plan._finalize(ordered)


def execute_chronicle_read(
    payload: Mapping[str, object],
    *,
    archive: ArchiveStore,
    vector_provider: VectorProvider | None = None,
) -> dict[str, object]:
    """Execute one chronicle projection against the caller's pinned archive."""
    summaries = _select_summaries(payload, archive=archive, vector_provider=vector_provider)
    edge_limit = _edge_limit(payload)
    sessions = []
    for summary in summaries:
        first, last, total = chronicle_edges(
            archive,
            str(summary.id),
            edge_limit,
            origin=Origin.from_string(summary.origin),
        )
        sessions.append(
            build_chronicle_session_payload(
                summary,
                first_messages=first,
                last_messages=last,
                total_matching_messages=total,
                edge_limit=edge_limit,
            )
        )
    result = build_chronicle_projection_payload(sessions, edge_limit=edge_limit)
    wire: dict[str, object] = {"view": "chronicle", "payload": result.model_dump(mode="json")}
    return wire


__all__ = ["DEFAULT_CHRONICLE_EDGE_LIMIT", "execute_chronicle_read"]
