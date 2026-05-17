"""Cost-rows migration helper (#1140).

Identifies legacy ``session_profiles`` rows that pre-date the cost basis
split (#1136). Those rows have a populated ``total_cost_usd`` column but
were materialized before per-basis ``provider_reported_usd``,
``api_equivalent_usd``, ``catalog_priced_usd``, etc. were introduced as
typed payload fields. The current substrate keeps those basis fields in
the ``evidence_payload`` / ``inference_payload`` JSON envelopes via
``CostBasisPayload``; legacy rows therefore have ``total_cost_usd`` set
but their evidence payload lacks the typed basis split.

The migration is one-shot, planner-driven, and idempotent:

1. **Detect** legacy rows via ``find_legacy_cost_rows()`` — pure-SQL
   read against ``session_profiles`` selecting rows whose
   ``cost_provenance`` is the legacy ``"unknown"`` marker (or other
   pre-#1136 marker) and ``total_cost_usd > 0``.
2. **Tag** them with the typed source label ``legacy-single-basis``
   so downstream surfaces can render "why" instead of an opaque zero.
   The tag flows through ``ArchiveInsightProvenance`` on the rebuilt
   insight.
3. **Schedule** a rebuild via :func:`plan_cost_migration` — returns a
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

__all__ = [
    "LEGACY_COST_SOURCE",
    "LEGACY_COST_PROVENANCE_MARKERS",
    "SESSION_PROFILES_REBUILD_TARGET",
    "LegacyCostRow",
    "LegacyCostRowReader",
    "find_legacy_cost_rows",
    "plan_cost_migration",
]


#: Source tag attached to legacy cost rows after the migration tags them.
#: Surfaces should render this verbatim so users see "why" a single-basis
#: total is in play instead of a typed split.
LEGACY_COST_SOURCE = "legacy-single-basis"


#: Cost-provenance values that the migration treats as legacy. These are
#: the values that pre-date the #1136 basis split. ``"unknown"`` is the
#: default on session_profile rows whose materializer never set a basis
#: provenance. ``"mixed"`` and ``"provider_reported"`` are post-#1136 and
#: are explicitly *not* migrated.
LEGACY_COST_PROVENANCE_MARKERS: frozenset[str] = frozenset({"unknown", ""})


#: Canonical session-profile rebuild target name. The actual target spec
#: lives in :mod:`polylogue.maintenance.targets`; the migration delegates
#: rebuild execution there rather than duplicating the rebuild logic.
SESSION_PROFILES_REBUILD_TARGET = "session_profiles"


@dataclass(frozen=True, slots=True)
class LegacyCostRow:
    """One legacy session-profile row identified for cost migration."""

    conversation_id: str
    provider_name: str
    total_cost_usd: float
    cost_provenance: str


class LegacyCostRowReader(Protocol):
    """Read seam for :func:`find_legacy_cost_rows`.

    A reader returns the rows in ``session_profiles`` whose cost columns
    look like the pre-#1136 single-basis shape. Implementations are
    expected to issue a single SELECT against the archive SQLite
    database; the indirection lets unit tests substitute an in-memory
    fixture without spinning up a full archive.
    """

    def __call__(
        self,
        *,
        provenance_markers: frozenset[str] = LEGACY_COST_PROVENANCE_MARKERS,
        min_total_usd: float = 0.0,
    ) -> tuple[LegacyCostRow, ...]: ...


def find_legacy_cost_rows(
    reader: LegacyCostRowReader,
    *,
    provenance_markers: frozenset[str] = LEGACY_COST_PROVENANCE_MARKERS,
    min_total_usd: float = 0.0,
) -> tuple[LegacyCostRow, ...]:
    """Return legacy single-basis session-profile rows from ``reader``.

    A row is "legacy" when its ``cost_provenance`` matches one of
    ``provenance_markers`` AND its ``total_cost_usd`` is strictly greater
    than ``min_total_usd``. Both conditions are necessary: rows whose
    ``total_cost_usd`` is zero have nothing to migrate (the basis split
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


def plan_cost_migration(
    legacy_rows: tuple[LegacyCostRow, ...],
    *,
    dry_run: bool = True,
) -> BackfillOperation:
    """Plan a backfill operation that rebuilds session profiles for legacy rows.

    The returned :class:`BackfillOperation` targets the canonical
    ``session_profiles`` rebuild path. Executing it through
    :func:`polylogue.maintenance.planner.execute_backfill` causes those
    rows to be re-materialized with the #1136 basis split populated and
    ``ArchiveInsightProvenance`` tagged with :data:`LEGACY_COST_SOURCE`.

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
        filter=json_document(
            {
                "cost_basis": LEGACY_COST_SOURCE,
                "conversation_ids": [row.conversation_id for row in legacy_rows],
                "dry_run": dry_run,
            }
        ),
    )
    affected = len(legacy_rows)
    estimated_time_s = affected / 50.0 if affected > 0 else 0.0
    results = [
        json_document(
            {
                "conversation_id": row.conversation_id,
                "provider_name": row.provider_name,
                "total_cost_usd": row.total_cost_usd,
                "cost_provenance": row.cost_provenance,
                "source": LEGACY_COST_SOURCE,
            }
        )
        for row in legacy_rows
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
