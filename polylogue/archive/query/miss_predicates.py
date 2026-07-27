"""Bounded predicate-drop diagnosis for zero-hit session queries (polylogue-jnj.12).

When a compound query returns zero hits, this module explains *which*
top-level ANDed field predicate on :class:`SessionQuerySpec` is responsible by
re-running a cheap ``COUNT``-only probe with that one field cleared back to
its declared dataclass default. Predicates whose removal alone unblocks a
non-empty result are surfaced as :class:`QueryMissReason` entries so an
operator (or agent) can see which filter(s) are responsible and by how much,
instead of a dead-end "no sessions matched".

Everything here reuses the same execution path the real query would have
used (:meth:`SessionQuerySpec.count`, itself backed by
:class:`~polylogue.archive.filter.filters.SessionFilter` /
:mod:`~polylogue.archive.query.archive_execution`) -- there is no separate FTS
or SQL path built for diagnosis.

Scope limits (kept explicit rather than silently approximate):

* Only top-level ANDed field predicates declared in
  :data:`~polylogue.archive.query.fields.QUERY_FIELD_DESCRIPTORS` are
  attributed. A query compiled from the DSL with a custom
  ``boolean_predicate`` tree (arbitrary AND/OR/NOT nesting, action
  sequences-as-predicates, lineage predicates) is *not* decomposed --
  attribution is skipped and a single informational reason notes the scope
  limit rather than guessing at a subtree.
* At most :data:`MAX_PREDICATE_PROBES` clause-drop ``COUNT`` probes run per
  diagnosis. When more predicates are active than the bound, only the first
  ones in descriptor-declaration order are probed -- the excess is not
  attributed. This is a real generality limit (not a combinatorial subset
  search), chosen so an unusually wide ad-hoc query cannot turn "explain the
  miss" into a second expensive scan.
* A query with fewer than two active top-level predicates is not attributed:
  with only one predicate active, dropping it just reproduces the
  archive-wide count, which is already covered by the existing
  archive-empty/fallback reasons in
  :mod:`polylogue.archive.query.miss_diagnostics`.
* The "nearest non-empty relaxation" suggestion is implemented only for
  ``since``/``until`` (date-window) predicates, where the boundary value is
  cheaply derivable with a single ``LIMIT 1`` sorted probe under the
  remaining filters. For other predicate kinds (origin, tag, tool, action,
  text, ...) no relaxation value is fabricated -- the clause-drop count is
  reported and nothing more.
"""

from __future__ import annotations

import logging
from dataclasses import MISSING, dataclass, fields, replace
from typing import TYPE_CHECKING, Any, cast

from polylogue.archive.query.fields import QUERY_FIELD_DESCRIPTORS, QueryFieldDescriptor
from polylogue.archive.query.miss_types import QueryMissReason

if TYPE_CHECKING:
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.config import Config

logger = logging.getLogger(__name__)

#: Hard cap on clause-drop COUNT probes per miss diagnosis. Bounded so a
#: pathological many-filter query cannot turn "explain the miss" into a
#: second expensive scan.
MAX_PREDICATE_PROBES = 6

_DATE_RELAXATION_FIELDS = ("since", "until")


@dataclass(frozen=True, slots=True)
class PredicateProbeResult:
    """Outcome of the bounded clause-drop scan."""

    reasons: tuple[QueryMissReason, ...]
    #: Descriptor names (e.g. ``"since"``) whose removal alone produced a
    #: non-empty count -- the confirmed culprits, in probe order.
    culprit_fields: tuple[str, ...]


def _spec_field_default(spec_cls: type[SessionQuerySpec], attr: str) -> object:
    for declared in fields(spec_cls):
        if declared.name != attr:
            continue
        if declared.default is not MISSING:
            return declared.default
        if declared.default_factory is not MISSING:
            return declared.default_factory()
        raise AttributeError(f"{spec_cls.__name__}.{attr} has no static default")
    raise AttributeError(attr)


def _replace_spec_fields(selection: SessionQuerySpec, overrides: dict[str, object]) -> SessionQuerySpec:
    """Type-erased ``dataclasses.replace`` for a dynamic field-name set.

    The probes here compute which fields to clear from
    :data:`QUERY_FIELD_DESCRIPTORS` at runtime, so the override mapping is
    necessarily ``dict[str, object]`` -- mypy's ``dataclasses.replace``
    plugin only accepts literal per-field keyword types, hence the single
    explicit erasure point instead of scattering ``# type: ignore`` at every
    call site.
    """
    return cast("SessionQuerySpec", replace(cast(Any, selection), **overrides))


def _droppable_descriptors(selection: SessionQuerySpec) -> list[QueryFieldDescriptor]:
    """Active top-level field predicates eligible for clause-drop probing."""
    return [
        descriptor
        for descriptor in QUERY_FIELD_DESCRIPTORS
        if descriptor.selection_filter and descriptor.spec_attr is not None and descriptor.is_active_for_spec(selection)
    ]


async def probe_predicate_zeroing(
    selection: SessionQuerySpec,
    config: Config,
    *,
    bound: int = MAX_PREDICATE_PROBES,
) -> PredicateProbeResult:
    """Bounded clause-drop scan: which top-level predicate(s) zeroed the set.

    Returns one :class:`QueryMissReason` (code ``predicate_zeroed_set``) per
    probed predicate whose removal alone would have produced a non-empty
    result, in :data:`QUERY_FIELD_DESCRIPTORS` declaration order, capped at
    *bound* probes total.
    """
    if selection.boolean_predicate is not None:
        return PredicateProbeResult(
            reasons=(
                QueryMissReason(
                    code="predicate_attribution_skipped_boolean_tree",
                    severity="info",
                    summary=(
                        "This query compiles to a custom boolean predicate tree; clause-drop "
                        "attribution only covers top-level filters, not nested AND/OR/NOT."
                    ),
                ),
            ),
            culprit_fields=(),
        )
    active = _droppable_descriptors(selection)
    if len(active) < 2:
        return PredicateProbeResult(reasons=(), culprit_fields=())

    spec_cls = type(selection)
    reasons: list[QueryMissReason] = []
    culprits: list[str] = []
    for descriptor in active[:bound]:
        assert descriptor.spec_attr is not None
        try:
            default_value = _spec_field_default(spec_cls, descriptor.spec_attr)
        except AttributeError:
            logger.warning("probe_predicate_zeroing: no static default for field=%s", descriptor.name)
            continue
        candidate = _replace_spec_fields(selection, {descriptor.spec_attr: default_value})
        try:
            count = await candidate.count(config)
        except Exception:
            logger.exception("probe_predicate_zeroing: count probe failed for field=%s", descriptor.name)
            continue
        if count <= 0:
            continue
        description = descriptor.describe_spec(selection) or descriptor.name
        reasons.append(
            QueryMissReason(
                code="predicate_zeroed_set",
                severity="info",
                summary=f"Without {description}, {count} session(s) would match.",
                detail=f"field={descriptor.name}",
                count=count,
            )
        )
        culprits.append(descriptor.name)
    return PredicateProbeResult(reasons=tuple(reasons), culprit_fields=tuple(culprits))


async def _date_relaxation_reason(
    selection: SessionQuerySpec,
    config: Config,
    *,
    field: str,
) -> QueryMissReason:
    """Nearest non-empty since/until boundary via a single ``LIMIT 1`` probe.

    *field* must already be a confirmed culprit (dropping it alone produced a
    non-empty count). This drops just that one bound, sorts toward it (the
    most recent match for a too-strict ``since``, the oldest match for a
    too-strict ``until``), and reads the single closest row -- one extra
    indexed query, no row hydration beyond one summary.

    Always returns a reason: an explicit "no cheap relaxation found" reason
    when the boundary can't be determined, never a fabricated value.
    """
    newest_first = field == "since"
    overrides: dict[str, object] = {
        "sort": "date",
        "reverse": not newest_first,
        "limit": 1,
        "offset": 0,
        "sample": None,
        "with_units": (),
        "with_unit_fields": {},
    }
    if field == "since":
        overrides["since"] = None
    else:
        overrides["until"] = None
    relaxed = _replace_spec_fields(selection, overrides)
    try:
        summaries = await relaxed.list_summaries(config)
    except Exception:
        logger.exception("_date_relaxation_reason: probe failed for field=%s", field)
        summaries = []
    boundary = summaries[0].updated_at if summaries else None
    if boundary is None:
        return QueryMissReason(
            code="predicate_relaxation_unavailable",
            severity="info",
            summary=f"No cheap relaxation value could be determined for `{field}`.",
            detail=f"field={field}",
        )
    return QueryMissReason(
        code="predicate_relaxation_suggested",
        severity="info",
        summary=f"Relaxing `{field}` to {boundary.isoformat()} would include the nearest matching session.",
        detail=f"field={field}; nearest_boundary={boundary.isoformat()}",
    )


async def probe_date_relaxation_reasons(
    selection: SessionQuerySpec,
    config: Config,
    *,
    culprit_fields: tuple[str, ...],
) -> tuple[QueryMissReason, ...]:
    """Relaxation suggestions for any confirmed since/until culprits."""
    fields_to_probe = [field for field in _DATE_RELAXATION_FIELDS if field in culprit_fields]
    if not fields_to_probe:
        return ()
    return tuple([await _date_relaxation_reason(selection, config, field=field) for field in fields_to_probe])


async def probe_fts_structured_disagreement(
    selection: SessionQuerySpec,
    config: Config,
) -> QueryMissReason | None:
    """Cheap two-probe check: do FTS terms and structured filters disagree?

    Computes the count with only the FTS/contains terms retained (all
    structured filters dropped) and the count with only the structured
    filters retained (FTS/contains terms dropped) via the same
    :meth:`SessionQuerySpec.count` path the real query uses. When both are
    independently non-empty but the combined query is not, the FTS hits and
    the structured-filtered set are disjoint -- that disagreement, not a
    broken query, is why nothing matched both.
    """
    if selection.boolean_predicate is not None or not selection.query_terms:
        return None
    structured = [
        descriptor
        for descriptor in _droppable_descriptors(selection)
        if descriptor.spec_attr not in {"query_terms", "contains_terms"}
    ]
    if not structured:
        return None

    spec_cls = type(selection)
    fts_only_overrides: dict[str, object] = {}
    for descriptor in structured:
        assert descriptor.spec_attr is not None
        try:
            fts_only_overrides[descriptor.spec_attr] = _spec_field_default(spec_cls, descriptor.spec_attr)
        except AttributeError:
            continue
    fts_only = _replace_spec_fields(selection, fts_only_overrides)
    structured_only = replace(selection, query_terms=(), contains_terms=())
    try:
        fts_only_count = await fts_only.count(config)
        structured_only_count = await structured_only.count(config)
    except Exception:
        logger.exception("probe_fts_structured_disagreement: probe failed")
        return None
    if fts_only_count > 0 and structured_only_count > 0:
        return QueryMissReason(
            code="fts_structured_disagreement",
            severity="info",
            summary=(
                f"The search text alone matches {fts_only_count} session(s) and the structured "
                f"filters alone match {structured_only_count} session(s), but none satisfy both."
            ),
            detail=f"fts_only_count={fts_only_count}; structured_only_count={structured_only_count}",
        )
    return None


__all__ = [
    "MAX_PREDICATE_PROBES",
    "PredicateProbeResult",
    "probe_date_relaxation_reasons",
    "probe_fts_structured_disagreement",
    "probe_predicate_zeroing",
]
