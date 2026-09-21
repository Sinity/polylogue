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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

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
    from polylogue.archive.query.expression import WithUnitWindow
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

#: Declared ceiling on rows fetched per page across all selected sessions, so a
#: pathological session with thousands of assertions cannot blow up one page.
#: The ceiling is real work protection and stays; what changed is that hitting
#: it is now *reported*. It used to cut the projection in silence: a session
#: with 250 messages answered ``with messages`` with exactly 200 rows, no flag
#: anywhere in the result and a clean ``ok`` outcome -- a complete-looking
#: projection over a truncated row set.
_MAX_ROWS_PER_SESSION = 200
_MAX_ROWS_PER_PAGE = 5000
_MAX_ATTACHED_TEXT_CHARS = 2000

#: Named gap a bounded projection carries so the envelope degrades instead of
#: reporting a cut row set as the session's complete one.
ATTACHED_UNIT_TRUNCATED_GAP = "attached_unit_truncated"


@dataclass(frozen=True, slots=True)
class AttachedUnitRows:
    """Attached-unit rows plus every named gap the fetch had to accept.

    The gaps are a return value rather than an optional out-parameter on
    purpose: an optional collector is a gap channel a caller can forget, and
    a caller forgetting one is exactly how a truncated facet scope reached
    the daemon route as ``outcome: ok`` (see ``_facets_payload``).
    """

    rows: dict[str, dict[str, tuple[JSONDocument, ...]]]
    gaps: tuple[str, ...] = ()


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


def _compact_attached_payload(payload: JSONDocument) -> JSONDocument:
    compacted = dict(payload)
    for field in ("text", "output_text"):
        value = compacted.get(field)
        if isinstance(value, str) and len(value) > _MAX_ATTACHED_TEXT_CHARS:
            compacted[field] = value[:_MAX_ATTACHED_TEXT_CHARS]
            compacted[f"{field}_truncated_chars"] = len(value) - _MAX_ATTACHED_TEXT_CHARS
    return compacted


def _select_payload_fields(
    payload: JSONDocument,
    fields: Sequence[str],
) -> JSONDocument:
    if not fields:
        return payload
    selected = set(fields)
    return cast(JSONDocument, {key: value for key, value in payload.items() if key in selected})


def _validate_payload_fields(
    unit: str,
    payload_model: Any,
    fields: Sequence[str],
) -> None:
    if not fields:
        return
    model_fields = getattr(payload_model, "model_fields", None)
    if not isinstance(model_fields, dict):
        return
    supported = set(model_fields)
    unknown = sorted(set(fields) - supported)
    if unknown:
        supported_text = ", ".join(sorted(supported))
        raise ValueError(
            f"with {unit} field selection contains unsupported field(s): {', '.join(unknown)}; "
            f"supported fields: {supported_text}"
        )


def _fetch_unit_rows(
    archive: ArchiveStore,
    descriptor: QueryUnitDescriptor,
    predicate: QueryPredicate,
    *,
    limit: int,
    sort_direction: Literal["asc", "desc"] = "asc",
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
            sort_direction=sort_direction,
        ),
    )


def _fetch_session_unit_rows(
    archive: ArchiveStore,
    descriptor: QueryUnitDescriptor,
    session_ids: Sequence[str],
    *,
    limit: int,
    sort_direction: Literal["asc", "desc"] = "asc",
) -> Sequence[Any] | None:
    method_name = {
        "message": "query_session_messages",
        "action": "query_session_actions",
        "file": "query_session_files",
    }.get(descriptor.unit)
    if method_name is None or not hasattr(archive, method_name):
        return None
    query_method = cast(Any, getattr(archive, method_name))
    return cast(
        Sequence[Any],
        query_method(
            session_ids,
            limit=limit,
            offset=0,
            sort_direction=sort_direction,
        ),
    )


def _bracket_field_value(payload_document: JSONDocument, descriptor: QueryUnitDescriptor, key: str) -> Any:
    attr = descriptor.row_field_attributes.get(key, key)
    return payload_document.get(attr)


def _matches_window_predicates(
    payload_document: JSONDocument,
    descriptor: QueryUnitDescriptor,
    window: WithUnitWindow,
) -> bool:
    for key, value in window.predicates:
        actual = _bracket_field_value(payload_document, descriptor, key)
        if actual is None or str(actual).lower() != value.lower():
            return False
    return True


def _apply_window_trim(rows: list[JSONDocument], window: WithUnitWindow | None) -> list[JSONDocument]:
    """Trim a session's already time-ascending row list to ``first:N``/``last:N``."""

    if window is None or window.window is None:
        return rows
    kind, count = window.window
    return rows[:count] if kind == "first" else rows[-count:]


def fetch_attached_units(
    archive: ArchiveStore,
    session_ids: Sequence[str],
    units: Sequence[str],
    unit_fields: dict[str, tuple[str, ...]] | None = None,
    unit_windows: Mapping[str, WithUnitWindow] | None = None,
) -> AttachedUnitRows:
    """Return attached-unit rows per unit, bucketed by session id.

    ``rows`` has shape ``{unit_name: {session_id: (row_payload, ...)}}`` where
    each ``row_payload`` is a JSON-ready dict produced by the descriptor-owned
    row payload model. Sessions with no rows for a unit are omitted from that
    unit's bucket.

    ``gaps`` names every unit whose fetch reached ``fetch_limit``. One row past
    the ceiling is requested and discarded, so reaching it is observed rather
    than inferred from a row count that could legitimately equal the bound. The
    gap is suppressed only when every selected session is *provably* complete
    for that unit -- a ``first:N``/``last:N`` window whose ``N`` every session's
    bucket already satisfies, which the matching fetch direction guarantees is
    the correct N. Otherwise the ceiling may have cut a session's rows, and the
    caller must degrade rather than present the bucket as that session's whole
    row set.

    ``unit_windows`` (polylogue-fnm.2) carries an optional per-unit
    :class:`~polylogue.archive.query.expression.WithUnitWindow`: bracket
    equality predicates are applied to each fetched row (against the unit's
    declared ``row_field_attributes``) before the field selection/
    compaction below, then an optional ``first:N``/``last:N`` trim is applied
    per session. Both operate on the already-fetched, capped row set -- they
    narrow what is attached, they do not push a predicate down to the SQL
    fetch itself. A ``last:N`` window fetches that unit in descending time
    order instead of the default ascending order (then restores ascending
    order before trimming), so the per-session/per-page row cap
    (``_MAX_ROWS_PER_SESSION``) captures the session's tail instead of
    silently capturing only its head -- without this, `last:N` on a session
    with more than the cap's worth of rows would trim the wrong end.
    Predicate + ``last:N`` together on a session whose *matching* rows are
    still sparser than the fetch cap can still under-fetch; that now lands in
    ``gaps`` like every other bounded fetch here rather than passing as a
    complete answer.
    """

    result: dict[str, dict[str, tuple[JSONDocument, ...]]] = {}
    gaps: list[str] = []
    if not session_ids or not units:
        return AttachedUnitRows(result)
    predicate = _session_scope_predicate(session_ids)
    if predicate is None:
        return AttachedUnitRows(result)
    fetch_limit = min(len(session_ids) * _MAX_ROWS_PER_SESSION, _MAX_ROWS_PER_PAGE)
    selected = set(session_ids)
    for unit in units:
        descriptor = query_unit_descriptor(unit)
        if descriptor is None:
            raise ValueError(f"unknown query unit for projection: {unit!r}")
        payload_model = getattr(surface_payloads, descriptor.payload_model, None)
        if payload_model is None or not hasattr(payload_model, "from_row"):
            raise ValueError(f"query unit {descriptor.unit!r} has no row payload model")
        selected_fields = () if unit_fields is None else unit_fields.get(descriptor.unit, ())
        _validate_payload_fields(descriptor.unit, payload_model, selected_fields)
        window = None if unit_windows is None else unit_windows.get(descriptor.unit)
        wants_tail = window is not None and window.window is not None and window.window[0] == "last"
        fetch_direction: Literal["asc", "desc"] = "desc" if wants_tail else "asc"
        # One row beyond the ceiling makes "the ceiling bound this fetch"
        # observable. Without the probe row a fetch returning exactly
        # ``fetch_limit`` rows is indistinguishable from a population that
        # happens to hold exactly that many.
        probe_limit = fetch_limit + 1
        rows = _fetch_session_unit_rows(
            archive, descriptor, session_ids, limit=probe_limit, sort_direction=fetch_direction
        )
        if rows is None:
            rows = _fetch_unit_rows(archive, descriptor, predicate, limit=probe_limit, sort_direction=fetch_direction)
        ceiling_reached = len(rows) > fetch_limit
        # Trim in fetch order, then restore ascending order: the probe row is
        # the last row the fetch produced in whichever direction it ran.
        rows = list(rows[:fetch_limit])
        if fetch_direction == "desc":
            rows = list(reversed(rows))
        buckets: dict[str, list[JSONDocument]] = {}
        for row in rows:
            session_id = _row_session_id(row)
            if session_id is None or session_id not in selected:
                continue
            payload = cast(Any, payload_model).from_row(row)
            payload_document = model_json_document(payload, exclude_none=not bool(selected_fields))
            if (
                window is not None
                and window.predicates
                and not _matches_window_predicates(payload_document, descriptor, window)
            ):
                continue
            buckets.setdefault(session_id, []).append(
                _select_payload_fields(
                    _compact_attached_payload(payload_document),
                    selected_fields,
                )
            )
        result[descriptor.unit] = {
            session_id: tuple(_apply_window_trim(rows, window)) for session_id, rows in buckets.items()
        }
        if ceiling_reached:
            window_count = window.window[1] if window is not None and window.window is not None else None
            provably_complete = window_count is not None and all(
                len(buckets.get(session_id, ())) >= window_count for session_id in selected
            )
            gap = f"{ATTACHED_UNIT_TRUNCATED_GAP}:{descriptor.unit}:{fetch_limit}"
            if not provably_complete and gap not in gaps:
                gaps.append(gap)
    return AttachedUnitRows(result, tuple(gaps))


__all__ = ["ATTACHED_UNIT_TRUNCATED_GAP", "AttachedUnitRows", "fetch_attached_units"]
