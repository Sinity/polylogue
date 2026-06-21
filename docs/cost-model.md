# Cost Model

Polylogue tracks per-session and per-cycle AI cost as a typed, multi-basis
estimate. This page explains the basis taxonomy, subscription plan
configuration, cycle outlook semantics, the single-basis backfill path, and the
non-authoritative caveat that governs every number displayed.

> **Non-authoritative.** Polylogue is not a billing system. Subscription
> quotas, prices, and overage rules ship as a dated curated seed and may
> drift from vendor reality at any time. Use these numbers for trend and
> coverage analysis, not for invoicing or hard rate-limit enforcement.

## Overview

Cost flows through four substrate-owned stages:

1. **Estimate** — per-session token usage is priced against the
   curated price map, an exact provider-reported total, or both
   (`polylogue/archive/semantic/pricing.py`).
2. **Materialize** — the estimate is folded into the typed
   `SessionCostInsight` and `CostRollupInsight` payloads
   (`polylogue/insights/archive.py`).
3. **Aggregate** — session-cost insights are folded to per-day USD usage
   for the current billing cycle
   (`polylogue/cost/aggregation.py`).
4. **Project** — daily usage is projected to a cycle outlook with quota
   pressure, overage, coverage, and confidence
   (`polylogue/cost/outlook.py`).

Each stage is pure-function-shaped on its inputs. The CLI (`polylogue
cost outlook`), MCP tool (`cost_outlook`), and Python API
(`Polylogue.cost_outlook`) are leaf adapters over the same typed
`CycleOutlook` envelope.

## Basis Taxonomy

A single estimate carries cost on five independent axes
(`CostBasisPayload` in `polylogue/archive/semantic/pricing.py`):

| Basis | Meaning |
| --- | --- |
| `provider_reported_usd` | Cost reported verbatim by the provider. Populated only when the source supplies an exact total. Preserved without rounding or scaling. |
| `api_equivalent_usd` | What the same usage would cost against API pricing. Mirrors `provider_reported_usd` when an exact total is present; otherwise filled from catalog. |
| `subscription_equivalent_usd` | What the same usage would cost against the user's subscription plan. Always zero unless the cost cluster is configured with a quota basis. |
| `catalog_priced_usd` | Catalog-priced estimate from the curated LiteLLM-shaped seed (`polylogue/archive/semantic/pricing.py:PRICING`). |
| `tool_surcharge_usd` | Tool/sidecar usage surcharge. Tracked separately so tool-heavy sessions don't silently inflate the headline cost. |

**Bases are independent — they do not sum to `total_usd`.** The same
underlying token usage can be expressed on multiple axes (e.g. an exact
provider total *and* a parallel catalog reconciliation against the same
tokens). `total_usd` is a single-number summary kept for backwards
compatibility: it draws from `basis.provider_reported_usd` when an exact
total is present, otherwise from `basis.catalog_priced_usd`. Consumers
that need a specific basis should read `basis.<field>` directly.

### Status, confidence, and the estimated flag

Every estimate carries a discrete `status`:

| Status | Meaning | Confidence |
| --- | --- | --- |
| `exact` | Derived from a provider-reported total | 0.95 |
| `priced` | Every message had a known model and token usage | 0.85 |
| `partial` | At least one priced row was missing a model or tokens | 0.55 |
| `unavailable` | No priced row at all; `unavailable_reason` is set | 0.0 |

The session-profile materialization layer propagates `status != 'exact'`
through `SessionEvidencePayload.cost_is_estimated`, so any consumer that
reads the session-profile row sees an explicit "this is estimated" flag
without needing to re-derive it.

### Unavailable reasons

When `status == 'unavailable'`, the estimate carries a discrete
`unavailable_reason` (`CostUnavailableReason`):

| Reason | Triggered when |
| --- | --- |
| `no_messages` | The session has no messages |
| `no_tokens` | Messages exist but none reported token usage |
| `no_model` | Token usage is present but the model is unknown |
| `no_price` | A model is set but no catalog entry exists |
| `provider_zero` | The provider reported a zero cost (preserved, not inferred missing) |
| `subscription_unconfigured` | A subscription plan was expected but none is loaded |

Surfaces render `unavailable_reason` verbatim so users see *why* a cost
is missing instead of an opaque `$0.00`.

## Per-Model Breakdown

Sessions that touch more than one model surface a `per_model_breakdown`
tuple. Each `CostModelBreakdown` row carries its own `usage`, `basis`,
and `total_usd`. Mixed-model rows are never collapsed into one opaque
total.

The contract suite (`tests/unit/cost/test_contract_suite.py`) pins:

* sum of `per_model_breakdown[i].total_usd` reconciles to
  `total_usd` within float-rounding tolerance;
* for each basis axis, sum across breakdown rows reconciles to the same
  axis on the aggregate.

## Subscription Plans

Plans are typed (`SubscriptionPlan` in `polylogue/cost/plans.py`) and
loaded from two sources:

* **`polylogue-curated-seed`** — the well-known plans baked into the
  cost cluster (`claude-pro`, `claude-max-5x`, `claude-max-20x`,
  `chatgpt-plus`, `chatgpt-pro`, `github-copilot-pro`,
  `gemini-advanced`). Each carries an explicit `effective_date` and a
  `notice` describing it as non-authoritative.
* **`user-config`** — `[[cost.subscription.plans]]` rows in
  `polylogue.toml`. User rows always override seed rows by `name` and
  are tagged `source = "user-config"` with `confidence = 1.0`.

A plan declares an optional `quota` plus `quota_basis`; without both,
the cycle outlook reports `QuotaPressureMissing(reason='no_quota_configured')`
rather than fabricating a synthetic zero.

The `cycle_anchor_day` is restricted to `[1, 28]` to avoid month-end
length edge cases (29/30/31). All cycle math runs in UTC, so DST
transitions are a no-op.

## Cycle Outlook

`build_cycle_outlook(plan, daily_usage, *, now, method)` projects the
current billing cycle:

* `window` carries half-open `[start, end)` instants plus an
  `elapsed_days` / `remaining_days` / `total_days` split.
* `cycle_to_date`, `burn_rate_per_day`, and `projected_total` are
  per-basis dicts (e.g. `{"credits": 30_000.0, "usd": 4.5}`).
  Subscription-equivalent and API-equivalent figures are never merged
  into one unlabelled "cost" number.
* `projection_method` is one of `linear`, `trailing-7d-mean`, or
  `eom-naive`. Callers pick the method; the engine never hides it.
* `quota_pressure` is `QuotaPressure` when the plan declares a quota,
  otherwise `QuotaPressureMissing(reason='no_quota_configured')`.
* `coverage_ratio` falls below `1.0` when the daily-usage sequence is
  missing a day inside the cycle-to-date window;
  `incomplete_days` lists the missing ISO dates.
* `confidence` combines plan confidence with `coverage_ratio` to
  discourage projecting from sparse data.

The projection is monotone non-decreasing in `used`: a heavier cycle
never produces a lower projection. The contract suite asserts this for
both the standalone `project_linear` primitive and the full
`build_cycle_outlook` path.

## Surfaces

| Surface | Entrypoint | Returns |
| --- | --- | --- |
| Python API | `Polylogue.cost_outlook(plan, *, now=None, method=...)` | `CycleOutlook | None` |
| CLI | `polylogue analyze --cost-outlook --plan <name> [--method ...] [-f json]` | Plain or JSON `CycleOutlook` payload |
| MCP | `cost_outlook(plan, method='linear')` tool | JSON `CycleOutlook` envelope |

All three surfaces share the same typed `CycleOutlook` envelope. The CLI
plain renderer visibly labels subscription-quota math as
non-authoritative and tags estimated USD figures.

## Single-Basis Backfill

Session-profile rows materialized before the basis split (#1136) carry
a populated `total_cost_usd` column but lack the per-basis values in
their evidence payload. The backfill helper in
`polylogue/maintenance/cost_backfill.py` identifies those rows and
schedules a typed rebuild:

1. `find_single_basis_cost_rows(reader)` selects rows whose `cost_provenance`
   is in `SINGLE_BASIS_COST_PROVENANCE_MARKERS` (currently `{"unknown", ""}`)
   and whose `total_cost_usd` is strictly positive. Already-typed
   provenance values (`provider_reported`, `mixed`) are intentionally
   excluded.
2. `plan_cost_backfill(rows)` returns a typed
   `BackfillOperation` with `kind = DERIVED_REBUILD`,
   `targets = ('session_profiles',)`,
   `reason = STALE_MATERIALIZER_VERSION`, and a scope filter carrying
   the source tag `single-basis-cost` plus the session-id list.
3. The maintenance planner
   (`polylogue/maintenance/planner.py:execute_backfill`) consumes the
   operation and re-materializes the affected session profiles. The
   rebuilt rows carry the basis split via the standard #1136 path; the
   source tag flows through the rebuild's
   `ArchiveInsightProvenance` so downstream surfaces can render *why*
   the row was rebuilt.

The backfill is one-shot and idempotent: a rebuilt row no longer
matches the stale provenance markers, so a second pass detects zero
candidates.

## Caveats

* **Curated seed drift.** Vendor prices and quotas change without
  notice. The curated seed carries `effective_date` and `notice`
  fields; user overrides in `polylogue.toml` always win.
* **Subscription-equivalent is not vendor-authoritative.** Polylogue's
  subscription quota math is an estimate of what the user's plan would
  charge for the observed usage. It does not query vendor billing
  endpoints and does not enforce rate limits.
* **Coverage is observable.** When the daily-usage sequence has gaps
  inside the cycle-to-date window, `coverage_ratio` drops and the
  missing days are surfaced in `incomplete_days`. Treat these as a
  signal that the projection is extrapolating from sparse data.
* **Provider-reported zero is preserved.** A `total_cost_usd == 0`
  reported by the provider routes through the usage estimator
  (the `provider_zero` reason in `CostUnavailableReason`) so the
  distinction between "no cost reported" and "cost reported as zero"
  is never silently elided.

## See Also

* [Insights overview](insights.md) — cost cluster context within the
  broader insight pipeline.
* [Data model](data-model.md) — typed payloads and storage shape.
* [CLI reference](cli-reference.md) — `polylogue analyze --cost-outlook` flags
  and JSON schema.
* [MCP reference](mcp-reference.md) — `cost_outlook` tool contract.
* [Configuration](configuration.md) — `[[cost.subscription.plans]]` in
  `polylogue.toml`.
