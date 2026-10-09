"""Pinned-archive execution for the CLI chronicle read view."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polylogue.archive.hydration import (
    archive_message_to_domain,
)
from polylogue.archive.message.models import Message
from polylogue.core.enums import MaterialOrigin, Origin
from polylogue.operations.query_lowering import cli_read_request
from polylogue.surfaces.chronicle import (
    build_chronicle_projection_payload,
    build_chronicle_session_payload,
)

if TYPE_CHECKING:
    from polylogue.archive.query.plan import SessionQueryPlan
    from polylogue.core.protocols import VectorProvider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


DEFAULT_CHRONICLE_EDGE_LIMIT = 8
_PAGE_SIZE = 256
_AUTHORED_ORIGINS = frozenset({MaterialOrigin.HUMAN_AUTHORED.value, MaterialOrigin.ASSISTANT_AUTHORED.value})
_DIALOGUE_ROLES = frozenset({"user", "assistant"})


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


@dataclass(frozen=True)
class ChronicleEdges:
    first: list[Message]
    last: list[Message]
    total: int
    lineage_complete: bool
    lineage_truncation_reason: str | None


def chronicle_edges(
    archive: ArchiveStore,
    session_id: str,
    edge_limit: int,
    *,
    origin: Origin,
) -> ChronicleEdges:
    """Stream the composed transcript and retain its exact nonempty prose edges."""
    first: list[Message] = []
    last: deque[Message] = deque(maxlen=edge_limit)
    offset = 0
    total_matching = 0
    complete = True
    reason = None
    while True:
        archive.check_operation_read()
        page = archive.read_session_page(session_id, limit=_PAGE_SIZE, offset=offset)
        complete = complete and page.lineage_complete
        reason = reason or page.lineage_truncation_reason
        for message in page.messages:
            if (
                _value(message.role) not in _DIALOGUE_ROLES
                or _value(message.message_type) != "message"
                or _value(message.material_origin) not in _AUTHORED_ORIGINS
            ):
                continue
            total_matching += 1
            domain = archive_message_to_domain(message, origin=origin)
            if not (domain.text or "").strip():
                continue
            if len(first) < edge_limit:
                first.append(domain)
            last.append(domain)
        offset += len(page.messages)
        if not page.messages or (page.total_message_count is not None and offset >= page.total_message_count):
            break
    first_ids = {message.id for message in first}
    return ChronicleEdges(
        first, [message for message in last if message.id not in first_ids], total_matching, complete, reason
    )


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


def execute_chronicle_read(
    payload: Mapping[str, object],
    *,
    archive: ArchiveStore,
    vector_provider: VectorProvider | None = None,
) -> dict[str, object]:
    """Execute one chronicle projection against the caller's pinned archive."""
    from polylogue.operations.read_view_selection import select_read_view_summaries

    summaries = select_read_view_summaries(
        _chronicle_plan(payload, vector_provider=vector_provider), archive=archive, default_limit=5
    )
    edge_limit = _edge_limit(payload)
    sessions = []
    for summary in summaries:
        edges = chronicle_edges(
            archive,
            str(summary.id),
            edge_limit,
            origin=Origin.from_string(summary.origin),
        )
        sessions.append(
            build_chronicle_session_payload(
                summary,
                first_messages=edges.first,
                last_messages=edges.last,
                total_matching_messages=edges.total,
                lineage_complete=edges.lineage_complete,
                lineage_truncation_reason=edges.lineage_truncation_reason,
                edge_limit=edge_limit,
            )
        )
    result = build_chronicle_projection_payload(sessions, edge_limit=edge_limit)
    wire: dict[str, object] = {"view": "chronicle", "payload": result.model_dump(mode="json")}
    return wire


__all__ = ["DEFAULT_CHRONICLE_EDGE_LIMIT", "execute_chronicle_read"]
