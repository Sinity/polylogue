"""Attach related query units to selected sessions (the ``with <units>`` clause).

This is the post-selection projection helper shared by every read surface. It
takes a page of selected session ids plus the requested unit names and returns,
per unit, the JSON-ready row payloads bucketed by session id.

The fetch path is unit-agnostic by construction: the SQL query method and row
payload model are both resolved from the query-unit descriptor registry
(``metadata.py``), so enabling a new unit for projection is a one-line change in
``WITH_PROJECTION_SUPPORTED_UNITS`` once its session-scoping fetch is confirmed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from polylogue.archive.query.metadata import QueryUnitDescriptor, query_unit_descriptor
from polylogue.archive.query.predicate import (
    QueryBoolPredicate,
    QueryFieldPredicate,
    QueryFieldRef,
    QueryPredicate,
)
from polylogue.core.json import JSONDocument
from polylogue.surfaces import payloads as surface_payloads
from polylogue.surfaces.payloads import model_json_document

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

#: Defensive cap on rows fetched per page across all selected sessions, so a
#: pathological session with thousands of assertions cannot blow up one page.
_MAX_ROWS_PER_SESSION = 200
_MAX_ROWS_PER_PAGE = 5000


def _session_id_field_predicate(session_id: str) -> QueryFieldPredicate:
    """Build a bound ``session.id = <id>`` predicate.

    Bound directly (rather than parsed from a string) because session ids can
    contain ``:`` separators that the DSL would otherwise tokenize.
    """

    return QueryFieldPredicate(
        field="session.id",
        values=(session_id,),
        op="=",
    ).with_field_ref(QueryFieldRef(scope="session", name="id", source_name="session.id"))


def _session_scope_predicate(session_ids: Sequence[str]) -> QueryPredicate | None:
    predicates = [_session_id_field_predicate(session_id) for session_id in session_ids]
    if not predicates:
        return None
    if len(predicates) == 1:
        return predicates[0]
    return QueryBoolPredicate("or", tuple(predicates))


def _row_session_id(row: Any) -> str | None:
    """Resolve the owning session id for a fetched unit row.

    Most unit rows carry a ``session_id`` column directly. Assertion rows scope
    by ``target_ref`` (``session:<id>``) instead, so fall back to parsing it.
    """

    session_id = getattr(row, "session_id", None)
    if isinstance(session_id, str) and session_id:
        return session_id
    target_ref = getattr(row, "target_ref", None)
    if isinstance(target_ref, str) and target_ref.startswith("session:"):
        return target_ref[len("session:") :]
    return None


def _fetch_unit_rows(
    archive: ArchiveStore,
    descriptor: QueryUnitDescriptor,
    predicate: QueryPredicate,
    *,
    limit: int,
) -> Sequence[Any]:
    method_name = descriptor.sql_query_method
    if method_name is None:
        raise ValueError(f"query unit {descriptor.unit!r} is not wired to a SQL executor")
    query_method = cast(Any, getattr(archive, method_name))
    return cast(
        Sequence[Any],
        query_method(
            predicate,
            limit=limit,
            offset=0,
            session_filters=None,
            sort="time",
            sort_direction="asc",
        ),
    )


def fetch_attached_units(
    archive: ArchiveStore,
    session_ids: Sequence[str],
    units: Sequence[str],
) -> dict[str, dict[str, tuple[JSONDocument, ...]]]:
    """Return attached-unit rows per unit, bucketed by session id.

    Shape: ``{unit_name: {session_id: (row_payload, ...)}}`` where each
    ``row_payload`` is a JSON-ready dict produced by the descriptor-owned row
    payload model. Sessions with no rows for a unit are omitted from that unit's
    bucket.
    """

    result: dict[str, dict[str, tuple[JSONDocument, ...]]] = {}
    if not session_ids or not units:
        return result
    predicate = _session_scope_predicate(session_ids)
    if predicate is None:
        return result
    fetch_limit = min(len(session_ids) * _MAX_ROWS_PER_SESSION, _MAX_ROWS_PER_PAGE)
    selected = set(session_ids)
    for unit in units:
        descriptor = query_unit_descriptor(unit)
        if descriptor is None:
            raise ValueError(f"unknown query unit for projection: {unit!r}")
        payload_model = getattr(surface_payloads, descriptor.payload_model, None)
        if payload_model is None or not hasattr(payload_model, "from_row"):
            raise ValueError(f"query unit {descriptor.unit!r} has no row payload model")
        rows = _fetch_unit_rows(archive, descriptor, predicate, limit=fetch_limit)
        buckets: dict[str, list[JSONDocument]] = {}
        for row in rows:
            session_id = _row_session_id(row)
            if session_id is None or session_id not in selected:
                continue
            payload = cast(Any, payload_model).from_row(row)
            buckets.setdefault(session_id, []).append(model_json_document(payload, exclude_none=True))
        result[descriptor.unit] = {session_id: tuple(rows) for session_id, rows in buckets.items()}
    return result


__all__ = ["fetch_attached_units"]
