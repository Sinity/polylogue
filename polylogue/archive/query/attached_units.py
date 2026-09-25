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

#: Declared ceiling on the rows one session contributes to a projection, so a
#: pathological session with thousands of assertions cannot blow up one page.
#: The ceiling is real work protection and stays; what changed is that hitting
#: it is now *reported* (it used to cut the projection in silence) and that it
#: is the bound of each session rather than of the page.
_MAX_ROWS_PER_SESSION = 200

#: Declared ceiling on rows fetched per page across all selected sessions.
#: It is shared out EQUALLY: every selected session gets the same allowance,
#: so what a session receives no longer depends on how many sessions share
#: the page or where its rows sort within it (polylogue-fvsjn).
_MAX_ROWS_PER_PAGE = 5000

_MAX_ATTACHED_TEXT_CHARS = 2000

#: Named gap a bounded projection carries so the envelope degrades instead of
#: reporting a cut row set as the session's complete one.
ATTACHED_UNIT_TRUNCATED_GAP = "attached_unit_truncated"


class AttachedUnitPageTooWideError(ValueError):
    """The page holds more sessions than the ceiling can probe and serve.

    A refusal rather than a smaller number, under the 2026-09-21 ruling: an
    allowance of zero is not a bound on the answer, it is the absence of one,
    and quietly serving nothing for every session would be the silent
    truncation this module exists to stop reporting as a complete answer.
    """

    code = "attached_unit_page_exceeds_row_budget"

    def __init__(self, session_count: int) -> None:
        super().__init__(
            f"{session_count} sessions on one page leave no rows per session "
            f"after reserving one truncation probe within the declared page "
            f"ceiling of {_MAX_ROWS_PER_PAGE}"
        )
        self.session_count = session_count


def _per_session_allowance(session_count: int) -> int:
    """Return the row allowance EVERY selected session gets on this page.

    Equal by construction. The page ceiling used to be spent in the page's
    own row order, so a long session could take the allowance a short one
    never received -- and under a tail-first (``last:N``) read a short
    session received zero rows while every other session was truncated. Both
    declared ceilings still hold; only their distribution changed.
    """

    # One additional row per session is fetched to prove whether the bound
    # truncated that session. Reserve those probes inside the declared page
    # ceiling instead of allowing ``count * (allowance + 1)`` to exceed it.
    allowance = min(_MAX_ROWS_PER_SESSION, _MAX_ROWS_PER_PAGE // session_count - 1)
    if allowance < 1:
        raise AttachedUnitPageTooWideError(session_count)
    return allowance


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
    per_session_limit: int,
    sort_direction: Literal["asc", "desc"] = "asc",
) -> Sequence[Any]:
    """Fetch a predicate-scoped unit, bounded per owning target.

    The assertion unit scopes by ``target_ref`` rather than a session column,
    so its partition is named differently; it is the same bound.
    """

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
            per_target_limit=per_session_limit,
        ),
    )


def _fetch_session_unit_rows(
    archive: ArchiveStore,
    descriptor: QueryUnitDescriptor,
    session_ids: Sequence[str],
    *,
    limit: int,
    per_session_limit: int,
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
            per_session_limit=per_session_limit,
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

    Every selected session is bounded separately and equally, at
    ``_per_session_allowance`` rows (polylogue-fvsjn). The bound therefore
    belongs to the session rather than to the page: what a session receives
    no longer depends on how many sessions share the page or where its rows
    sort within it, and no session can be starved to zero rows while another
    is truncated. A page holding more sessions than the page ceiling can give
    one row each is refused by name rather than served an allowance of zero.

    ``gaps`` names every unit for which at least one session actually reached
    its bound. One row past each session's allowance is requested and
    discarded, so the cut is observed per session rather than inferred from a
    page-level row count that could legitimately equal the bound. The gap
    carries the per-session bound, so the number it names is the number that
    applied to each session. It is suppressed only when every *truncated*
    session is provably complete for the requested ``first:N``/``last:N``
    window -- the matching fetch direction guarantees that N is the correct
    end. Otherwise the bound may have cut a session's rows, and the caller
    must degrade rather than present the bucket as that session's whole row
    set.

    ``unit_windows`` (polylogue-fnm.2) carries an optional per-unit
    :class:`~polylogue.archive.query.expression.WithUnitWindow`: bracket
    equality predicates are applied to each fetched row (against the unit's
    declared ``row_field_attributes``) before the field selection/
    compaction below, then an optional ``first:N``/``last:N`` trim is applied
    per session. Both operate on the already-fetched, capped row set -- they
    narrow what is attached, they do not push a predicate down to the SQL
    fetch itself. A ``last:N`` window fetches that unit in descending time
    order instead of the default ascending order (then restores ascending
    order before trimming), so each session's allowance captures its tail
    instead of silently capturing only its head -- without this, `last:N` on
    a session with more than its allowance would trim the wrong end.
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
    selected = list(dict.fromkeys(session_ids))
    per_session_limit = _per_session_allowance(len(selected))
    # One row beyond each session's allowance makes "the bound cut THIS
    # session" observable. Without the probe row a session returning exactly
    # its allowance is indistinguishable from one holding exactly that many.
    probe_limit = per_session_limit + 1
    fetch_limit = len(selected) * probe_limit
    selected_ids = set(selected)
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
        fetched = _fetch_session_unit_rows(
            archive,
            descriptor,
            selected,
            limit=fetch_limit,
            per_session_limit=probe_limit,
            sort_direction=fetch_direction,
        )
        if fetched is None:
            fetched = _fetch_unit_rows(
                archive,
                descriptor,
                predicate,
                limit=fetch_limit,
                per_session_limit=probe_limit,
                sort_direction=fetch_direction,
            )
        # Bucket in fetch order so the probe row is the LAST row of the
        # session it belongs to, in whichever direction the fetch ran.
        fetched_rows: dict[str, list[Any]] = {}
        for row in fetched:
            session_id = _row_session_id(row)
            if session_id is None or session_id not in selected_ids:
                continue
            fetched_rows.setdefault(session_id, []).append(row)
        truncated_sessions = {
            session_id for session_id, session_rows in fetched_rows.items() if len(session_rows) > per_session_limit
        }
        rows: list[Any] = []
        for session_id in selected:
            session_rows = fetched_rows.get(session_id, ())[:per_session_limit]
            rows.extend(reversed(session_rows) if fetch_direction == "desc" else session_rows)
        buckets: dict[str, list[JSONDocument]] = {}
        for row in rows:
            session_id = _row_session_id(row)
            if session_id is None:
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
        if truncated_sessions:
            window_count = window.window[1] if window is not None and window.window is not None else None
            provably_complete = window_count is not None and all(
                len(buckets.get(session_id, ())) >= window_count for session_id in truncated_sessions
            )
            gap = f"{ATTACHED_UNIT_TRUNCATED_GAP}:{descriptor.unit}:{per_session_limit}"
            if not provably_complete and gap not in gaps:
                gaps.append(gap)
    return AttachedUnitRows(result, tuple(gaps))


__all__ = ["ATTACHED_UNIT_TRUNCATED_GAP", "AttachedUnitRows", "fetch_attached_units"]
