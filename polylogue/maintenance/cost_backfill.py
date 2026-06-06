"""Cost-row backfill helper (#1140).

Identifies stale ``session_profiles`` rows that pre-date the cost basis
split (#1136). Those rows have a populated ``total_cost_usd`` column but
were materialized before per-basis ``provider_reported_usd``,
``api_equivalent_usd``, ``catalog_priced_usd``, etc. were introduced as
typed payload fields. The current substrate keeps those basis fields in
the ``evidence_payload`` / ``inference_payload`` JSON envelopes via
``CostBasisPayload``; stale rows therefore have ``total_cost_usd`` set
but their evidence payload lacks the typed basis split.

The backfill is one-shot, planner-driven, and idempotent:

1. **Detect** single-basis rows via ``find_single_basis_cost_rows()`` — pure-SQL
   read against ``session_profiles`` selecting rows whose
   ``cost_provenance`` is the untyped ``"unknown"`` marker (or other
   pre-basis marker) and ``total_cost_usd > 0``.
2. **Tag** them with the typed source label ``single-basis-cost``
   so downstream surfaces can render "why" instead of an opaque zero.
   The tag flows through ``ArchiveInsightProvenance`` on the rebuilt
   insight.
3. **Schedule** a rebuild via :func:`plan_cost_backfill` — returns a
   :class:`BackfillOperation` targeting the canonical session-profile
   rebuild target. The maintenance planner already owns the actual
   rebuild path; this helper is the bridge that classifies the work and
   reasons about its scope.

The helper is intentionally pure-function-shaped (the actual DB writes
go through :func:`polylogue.maintenance.planner.execute_backfill`), so
it stays exercisable in unit tests without spinning up the full
maintenance executor.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol

from polylogue.core.json import json_document
from polylogue.maintenance.invalidation import InvalidationReason
from polylogue.maintenance.planner import (
    BackfillKind,
    BackfillOperation,
    BackfillStatus,
    MaintenanceScope,
)
from polylogue.maintenance.scope import MaintenanceScopeFilter

__all__ = [
    "SINGLE_BASIS_COST_SOURCE",
    "SINGLE_BASIS_COST_PROVENANCE_MARKERS",
    "SESSION_PROFILES_REBUILD_TARGET",
    "SingleBasisCostRow",
    "SingleBasisCostRowReader",
    "find_single_basis_cost_rows",
    "plan_cost_backfill",
]


#: Source tag attached to stale single-basis cost rows after classification.
#: Surfaces should render this verbatim so users see "why" a single-basis
#: total is in play instead of a typed split.
SINGLE_BASIS_COST_SOURCE = "single-basis-cost"


#: Cost-provenance values treated as stale single-basis rows. These are
#: the values that pre-date the #1136 basis split. ``"unknown"`` is the
#: default on session_profile rows whose materializer never set a basis
#: provenance. ``"mixed"`` and ``"provider_reported"`` are post-#1136 and
#: are explicitly excluded.
SINGLE_BASIS_COST_PROVENANCE_MARKERS: frozenset[str] = frozenset({"unknown", ""})


#: Canonical session-profile rebuild target name. The actual target spec
#: lives in :mod:`polylogue.maintenance.targets`; the backfill delegates
#: rebuild execution there rather than duplicating the rebuild logic.
SESSION_PROFILES_REBUILD_TARGET = "session_profiles"


@dataclass(frozen=True, slots=True)
class SingleBasisCostRow:
    """One stale single-basis session-profile row identified for backfill."""

    session_id: str
    source_name: str
    total_cost_usd: float
    cost_provenance: str


class SingleBasisCostRowReader(Protocol):
    """Read seam for :func:`find_single_basis_cost_rows`.

    A reader returns the rows in ``session_profiles`` whose cost columns
    look like the pre-#1136 single-basis shape. Implementations are
    expected to issue a single SELECT against the archive SQLite
    database; the indirection lets unit tests substitute an in-memory
    fixture without spinning up a full archive.
    """

    def __call__(
        self,
        *,
        provenance_markers: frozenset[str] = SINGLE_BASIS_COST_PROVENANCE_MARKERS,
        min_total_usd: float = 0.0,
    ) -> tuple[SingleBasisCostRow, ...]: ...


def find_single_basis_cost_rows(
    reader: SingleBasisCostRowReader,
    *,
    provenance_markers: frozenset[str] = SINGLE_BASIS_COST_PROVENANCE_MARKERS,
    min_total_usd: float = 0.0,
) -> tuple[SingleBasisCostRow, ...]:
    """Return stale single-basis session-profile rows from ``reader``.

    A row is stale when its ``cost_provenance`` matches one of
    ``provenance_markers`` AND its ``total_cost_usd`` is strictly greater
    than ``min_total_usd``. Both conditions are necessary: rows whose
    ``total_cost_usd`` is zero have nothing to backfill (the basis split
    would be all zeros), and rows whose provenance is already typed
    (``provider_reported``, ``mixed``, etc.) carry the split via the
    evidence payload and are intentionally excluded.

    The function is a thin filter over ``reader`` so callers can supply
    either a live SQLite-backed reader or an in-memory fixture without
    duplicating the classification logic.
    """
    rows = reader(provenance_markers=provenance_markers, min_total_usd=min_total_usd)
    return tuple(
        row for row in rows if row.cost_provenance in provenance_markers and row.total_cost_usd > min_total_usd
    )


def plan_cost_backfill(
    rows: tuple[SingleBasisCostRow, ...],
    *,
    dry_run: bool = True,
) -> BackfillOperation:
    """Plan a backfill operation that rebuilds stale single-basis session profiles.

    The returned :class:`BackfillOperation` targets the canonical
    ``session_profiles`` rebuild path. Executing it through
    :func:`polylogue.maintenance.planner.execute_backfill` causes those
    rows to be re-materialized with the #1136 basis split populated and
    ``ArchiveInsightProvenance`` tagged with :data:`SINGLE_BASIS_COST_SOURCE`.

    Returns a ``PENDING`` operation when ``dry_run`` is true (the default
    — surfaces should preview before executing). Pass ``dry_run=False``
    in non-test callers to let the operation be handed straight to the
    executor.

    The operation carries a typed :class:`InvalidationReason` of
    ``STALE_MATERIALIZER_VERSION`` because the materializer that wrote
    these rows pre-dates the basis-split addition.
    """
    operation_id = str(uuid.uuid4())
    scope = MaintenanceScope(
        targets=(SESSION_PROFILES_REBUILD_TARGET,),
        filter=MaintenanceScopeFilter(
            session_ids=tuple(row.session_id for row in rows) or None,
        ),
    )
    affected = len(rows)
    estimated_time_s = affected / 50.0 if affected > 0 else 0.0
    results = [
        json_document(
            {
                "session_id": row.session_id,
                "source_name": row.source_name,
                "total_cost_usd": row.total_cost_usd,
                "cost_provenance": row.cost_provenance,
                "source": SINGLE_BASIS_COST_SOURCE,
            }
        )
        for row in rows
    ]
    return BackfillOperation(
        operation_id=operation_id,
        kind=BackfillKind.DERIVED_REBUILD,
        targets=(SESSION_PROFILES_REBUILD_TARGET,),
        status=BackfillStatus.PENDING,
        affected_rows=affected,
        estimated_time_s=estimated_time_s,
        results=results,
        scope=scope,
        reason=InvalidationReason.STALE_MATERIALIZER_VERSION,
    )
