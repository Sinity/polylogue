"""The process-wide default metric registry (rxdo.9.1 identity/schema layer).

:class:`~polylogue.insights.measurement.metric.MetricDefinition` and
:class:`~polylogue.insights.measurement.metric.MetricRegistry` (PR #2888,
merged) had zero production callers -- the corrective AC's second consumer
path ("one hash resolves through both query/analysis and statistical-
registry paths") depends on ``polylogue-9l5.7``'s statistics registry, which
remains unstarted. Building that whole registry/composition epic is out of
scope here (see ``polylogue-rxdo.9`` epic-expansion guard).

This module is the bounded, honest slice available without it: a real,
process-wide registry populated with concrete metric definitions for
existing, already-computed constructs -- session cost (``cost/pricing.py``/
``cost/outlook.py``) and, since polylogue-t0p, plan-completion rate
(``insights/claude_todo_projection.py``) -- reachable through the MCP ``get``
tool (``get(ref="metric:session_cost_usd")`` /
``get(ref="metric:plan_completion_rate")``, see ``polylogue/mcp/
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

#: Plan-vs-outcome measure (polylogue-t0p): fraction of a session's Claude
#: Code plan (``~/.claude/todos/*.json``, ``insights/claude_todo_projection.py``)
#: marked ``completed`` at its LATEST observed snapshot. ``structural``, not
#: ``heuristic``: the status string is a provider-reported field Claude Code
#: itself writes, never inferred from transcript prose. ``census`` enumeration
#: (not a sample) -- every admitted snapshot for a session is used, not a
#: subset -- so no sampling interval attaches per the registry's own
#: census-vs-sample doctrine (9l5.7 bead notes, ``MeasureSpec`` design).
#: Denominator hazard: a session with an empty plan (``item_count == 0``) has
#: ``completion_rate is None`` (``ClaudeTodoSnapshot.completion_rate``), not
#: zero -- "nothing planned" must never render as "nothing done".
PLAN_COMPLETION_RATE_METRIC = MetricDefinition(
    construct="fraction of a session's latest Claude Code TODO plan marked completed",
    unit="ratio",
    unit_source="claude_todo_plan_states",
    aggregation="mean",
    grain="logical",
    required_enumeration="census",
    measurement_authority=("structural",),
    provenance_mixing="single-authority",
    null_policy="exclude",
    output_schema="ratio:float|null",
)

#: Process-wide default registry. A module-level singleton is the correct
#: shape for an in-process content-addressed identity registry (mirrors
#: ``polylogue.insights.registry.INSIGHT_REGISTRY``) -- registration is
#: idempotent by content hash, so re-importing this module never double
#: -registers or drifts.
DEFAULT_METRIC_REGISTRY = MetricRegistry()
DEFAULT_METRIC_REGISTRY.register(SESSION_COST_USD_METRIC, name="session_cost_usd")
DEFAULT_METRIC_REGISTRY.register(PLAN_COMPLETION_RATE_METRIC, name="plan_completion_rate")


__all__ = ["DEFAULT_METRIC_REGISTRY", "PLAN_COMPLETION_RATE_METRIC", "SESSION_COST_USD_METRIC"]
