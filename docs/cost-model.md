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
analyze --cost-outlook`), MCP tool (`cost_outlook`), and Python API
(`Polylogue.cost_outlook`) are leaf adapters over the same typed
`CycleOutlook` envelope.


## Provider Usage Accounting Coverage

Provider usage accounting is distinct from cost estimation. The authoritative
provider evidence, when present, lives in `session_provider_usage_events`;
`session_model_usage` is a rebuildable rollup/read model; transcript words and
text length are estimate-only signals. `polylogue analyze usage` reports all
three streams separately so a missing provider counter does not appear as a
provider-reported zero.

| Origin | Provider | Declared coverage | Event stream | Token semantics |
| --- | --- | --- | --- | --- |
| `claude-code-session` | `claude-code` | exact where `message.usage` exists | `message_usage` | per-message/request usage; cache read and cache creation are separate lanes |
| `codex-session` | `codex` | exact where `token_count` exists | `token_count` | `last_token_usage` is current/request-window telemetry; `total_token_usage` is cumulative and session-global, so rollups take the latest total per session |
| `chatgpt-export` | `chatgpt` | estimate-only | transcript text | exports do not carry reliable per-request provider token counters |
| `claude-ai-export` | `claude-ai` | estimate-only | transcript text | exports preserve conversation text, not exact provider usage counters |
| `aistudio-drive` | `gemini` | partial | message token fields | some records may carry output `tokenCount`; input/cache/cumulative semantics are unavailable |
| `gemini-cli-session` | `gemini-cli` | partial | message token fields | generic usage dictionaries may be materialized but provider-specific cumulative semantics are not declared |
| `hermes-session` | `hermes` | exact where `state.db` counters exist | `token_count` | cumulative session counters; cache read/write and reasoning lanes stay separate |
| `antigravity-session` | `antigravity` | unsupported | transcript text | no exact provider usage parser is declared |
| `unknown-export` | `unknown` | unsupported | transcript text | no exact provider usage parser is declared |

The diagnostics payload exposes both the declared coverage and the observed
state for each origin:

* `exact_provider_telemetry` means every materialized session for an exact
  origin has provider usage event rows.
* `partial_provider_telemetry` means only some sessions carry exact event rows,
  or a partial origin has token fields without exact request/cumulative/cache
  semantics.
* `missing_provider_telemetry` means an exact origin has sessions but no
  provider usage events are materialized.
* `estimate_only` and `unsupported` are deliberate non-authoritative states;
  they are not zero-usage states.
* `acquired_not_materialized` means `source.db` has parseable raw rows that have
  not been represented in `index.db`.
* `stale_rollup` means provider usage events exist but `session_model_usage` no
  longer matches the event-derived rollup.

Cache read/write tokens are never folded into generic input/output tokens. The
usage counter labels are `input_tokens`, `output_tokens`,
`cached_input_tokens`, `cache_write_tokens`, and `reasoning_output_tokens`. For
Codex cumulative events, reasoning output is folded into model rollup output only
after the provider-event lane has preserved it separately.

Rebuild policy follows the fresh-first archive model. If usage events, source
raw rows, or rollups disagree, move the stale index tier aside and rebuild
`index.db` from `source.db`/raw archives. Do not patch a live archive manually to
make usage totals line up.

`session_model_usage` self-heals like every other session insight
(`session_profiles`, `session_latency_profiles`, ...): it carries an
`insight_materialization('provider_usage')` stamp keyed on
`SESSION_INSIGHT_MATERIALIZER_VERSION`, and the same session-insight rebuild
path that repairs a stale/missing `session_profile` also re-derives
`session_model_usage` from `session_provider_usage_events`/`messages`
whenever that stamp is stale or missing. This runs automatically on the
daemon's periodic session-insight drain, hot-source convergence, and
convergence-debt retry — a manual `polylogue ops reset --index` is no longer
required to pick up a provider-usage materializer fix or a zero-token bug fix
for existing sessions; it remains available as a fallback for a full rebuild.

### Dual cost view on the usage ledger

`polylogue analyze usage` / the `provider_usage` MCP tool / `ProviderUsageReport`
(`polylogue/storage/usage.py`) report **two independent cost bases** per
pricing lane, matching the [Basis Taxonomy](#basis-taxonomy) split used
elsewhere:

* `catalog_api_equivalent_usd` — API list-price equivalent, computed through
  the single LiteLLM-backed catalog (`PRICING` in
  `polylogue/archive/semantic/pricing.py`).
* `subscription_credit_usd` — what the same usage would cost against the
  curated Claude Code Pro-tier subscription credit model
  (`polylogue/archive/semantic/subscription_pricing.py`): cache reads are
  free, cache writes bill at the input rate, and output bills at 5x input
  (matching Anthropic's API rate ratio — a `MODEL_CREDIT_RATES` entry with
  `output_credits == input_credits` was a bug, fixed and regression-tested in
  `tests/unit/storage/test_cost_queries.py`). It is `0.0` for models without
  a declared credit rate (non-Claude models) — never a fabricated figure.

Catalog coverage is explicit. When any model row lacks a catalog price, the
complete `catalog_api_equivalent_usd` is `null` and its evidence value is
`unknown`; exact token evidence remains `known`.
`catalog_priced_subtotal_usd` retains the numeric subtotal for matched rows,
with the unmatched row count and coverage exclusion explaining why it is not
the complete total. Empty usage frames are unknown rather than a measured
zero. The same contract is carried by each lane and the physical/logical
archive totals through `exact_total_tokens_evidence` and
`catalog_api_equivalent_evidence`.

The two bases are always reported separately and never summed; a
cache-heavy Claude session's `subscription_credit_usd` is strictly below its
`catalog_api_equivalent_usd` for the same lane. Both draw from the shared
`credits_to_usd()` conversion (`credit_cost / tier.credit_pool *
tier.monthly_fee_usd`, default the Pro tier) so every surface reporting a
subscription-equivalent dollar figure uses the same tier assumption; see the
[Caveats](#caveats) below — this is not vendor-authoritative billing.

### Codex disjoint billing lanes

Codex (OpenAI) `token_count` events report **overlapping** token counts:
`input_tokens` is *inclusive* of `cached_input_tokens`, and `output_tokens` is
*inclusive* of `reasoning_output_tokens`. Verified across the operator's full
real corpus (1.84M token_count events): `cached <= input` on 100% of rows, and
`total == input + output` on 98.9% (reasoning is a subset of output, not an
additional term). Because a coding agent re-sends its whole context every turn,
cached input is the dominant term — **~96% of Codex input** on the real archive.

The cost model bills `input` and `cache_read` as **separate additive lanes**
(`pricing.py:_cost_components`, the Anthropic convention where `input` means
fresh/uncached input). So the rollup subtracts cached out of input —
`fresh_input = input - cached` — before storing the
`session_model_usage` lanes. After this mapping `fresh_input + cache_read`
reconstructs the provider's input exactly, so each token is billed once.
Storing input inclusive of cached *and* a separate `cache_read` lane would bill
the cached tokens twice (once at the full input rate, once at the cache-read
rate); on the real corpus that inflated Codex API-list-equivalent cost **7.69x**
($76,856 → $591,103). The mapping lives in
`_provider_usage_disjoint_lanes` (`storage/sqlite/archive_tiers/write.py`).

**Reproduce it.** `scripts/cost_accounting_demo.py` ingests a crafted Codex
session through the real writer, reads the materialized rollup back, prices it
with the real catalog, and prints the corrected lanes next to the pre-fix
double-billed cost — all from hand-checkable token numbers, no mocks:

```bash
uv run python scripts/cost_accounting_demo.py
```

**The same bug class also existed at per-message granularity** (polylogue-f2qv.2):
`sources/parsers/codex.py:_token_usage` (per-message `input_tokens`/
`output_tokens`/`cache_read_tokens`/`cache_write_tokens` on `ParsedMessage`,
independent of the `token_count` event rollup above) stored Codex's raw
inclusive-of-cache `input_tokens` unmodified, so `estimate_message_cost`/
`compute_session_cost` (which read the `messages.input_tokens` column
directly and bill it as a separate additive lane alongside `cache_read_tokens`)
double-billed the same way. Fixed by subtracting cache out of input at parse
time, so `messages.input_tokens` carries the same disjoint-lane contract
cross-provider (Claude's native `input_tokens` already excludes cache and is
untouched). Guarded by
`tests/unit/core/test_pricing.py::test_disjoint_input_cache_lanes_survive_parse_write_and_pricing`
(parser-to-writer-to-pricing consequence guard) and
`tests/unit/sources/test_parsers_codex.py` (parser-level disjointness) plus
the Claude control case in
`tests/unit/sources/test_parsers_claude_code_artifacts.py::test_message_input_tokens_stays_raw_unlike_codex_disjoint_fix`.

Current Codex JSONL places exact usage in nested `token_count` events
(`last_token_usage` and session-global `total_token_usage`); current sampled
message records do not carry usage. The direct-message usage path above remains
a supported compatibility path. At the event tier,
`reasoning_output_tokens` stays separately queryable. At the priced model tier,
`output_tokens` remains the provider's inclusive output total because reasoning
uses the same output rate and must not be added again. Thus the logical
completion/reasoning partition is preserved as evidence without inventing a
second additive reasoning cost lane or a duplicate aggregate schema.

A captured run is committed at
[docs/examples/cost-accounting-demo.txt](examples/cost-accounting-demo.txt)
(synthetic, no private data).

Operators can cross-verify the real archive against Codex's own authoritative
token store (`~/.codex/state_5.sqlite`, joined on the thread UUID in the
`codex-session:<uuid>` session id). On the operator's archive the per-thread
`MAX(total_tokens)` vs authoritative `tokens_used` ratio has **median 1.000**;
the aggregate is higher only because the archive retains sessions the live tool
has pruned (36 B tokens across 96 threads):

```bash
uv run python scripts/cost_accounting_demo.py \
  --archive ~/.local/share/polylogue --codex-state ~/.codex/state_5.sqlite
```

> Existing archives carry rollups computed before this fix; the corrected lanes
> apply to new ingests. Re-materialize with the fresh-first rebuild path above
> to update stored cost.

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
