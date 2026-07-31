"""The process-wide default metric registry (rxdo.9.1 identity/schema layer).

:class:`~polylogue.insights.measurement.metric.MetricDefinition` and
:class:`~polylogue.insights.measurement.metric.MetricRegistry` (PR #2888,
merged) had zero production callers -- the corrective AC's second consumer
path ("one hash resolves through both query/analysis and statistical-
registry paths") depends on ``polylogue-9l5.7``'s statistics registry, which
remains unstarted. Building that whole registry/composition epic is out of
scope here (see ``polylogue-rxdo.9`` epic-expansion guard).

This module is the bounded, honest slice available without it: a real,
process-wide registry populated with one concrete metric definition for an
existing, already-computed construct (session cost, ``cost/pricing.py``/
``cost/outlook.py``), reachable through the MCP ``get`` tool
(``get(ref="metric:session_cost_usd")``, see ``polylogue/mcp/
server_cutover.py``). This proves the identity/registry machinery resolves
through a real production surface -- it does NOT execute the metric (no
composition/aggregation engine exists yet; that is 9l5.7's job) or attach a
``metric_ref`` to computed values anywhere. Both remain open scope.
"""

from __future__ import annotations

from polylogue.insights.measurement.metric import MetricDefinition, MetricRegistry

#: Session-level USD cost: provider-reported totals where available,
#: catalog-priced (LiteLLM) estimates otherwise -- mirrors the mixed-basis
#: reality already documented in ``docs/cost-model.md`` and computed by
#: ``polylogue/archive/semantic/pricing.py:estimate_session_cost`` /
#: ``polylogue/cost/outlook.py``. Declared ``mixed-declared`` rather than
#: ``single-authority`` because the two lanes (provider-reported vs.
#: catalog-estimated) are intentionally blended, per the 9l5.7 bead's own
#: denominator-hazards checklist ("outcome_conditioned_cost must never
#: silently mix provider-reported with catalog estimates" -- this metric
#: names the mixing explicitly instead).
SESSION_COST_USD_METRIC = MetricDefinition(
    construct="total estimated USD cost for one session",
    unit="usd",
    unit_source="session_costs",
    aggregation="sum",
    grain="logical",
    required_enumeration="exact",
    measurement_authority=("provider-reported", "catalog-estimated"),
    provenance_mixing="mixed-declared",
    output_schema="usd:float",
)

#: Process-wide default registry. A module-level singleton is the correct
#: shape for an in-process content-addressed identity registry (mirrors
#: ``polylogue.insights.registry.INSIGHT_REGISTRY``) -- registration is
#: idempotent by content hash, so re-importing this module never double
#: -registers or drifts.
DEFAULT_METRIC_REGISTRY = MetricRegistry()
DEFAULT_METRIC_REGISTRY.register(SESSION_COST_USD_METRIC, name="session_cost_usd")


__all__ = ["DEFAULT_METRIC_REGISTRY", "SESSION_COST_USD_METRIC"]
