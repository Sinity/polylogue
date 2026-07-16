---
created: "2026-06-28"
purpose: "Fold scripts/agent_forensics.py into Polylogue's materialized insight substrate, then delete it."
status: "active"
project: "polylogue"
---

# Fold agent_forensics.py into the insight substrate

## Why this is "doubly wrong"
`scripts/agent_forensics.py` (719 lines) is both (1) a **standalone** script that
opens `index.db` read-only and runs ad-hoc SQL, and (2) a **dynamic** analysis —
it recomputes every metric at query time instead of reading materialized derived
read-model tables. The operator principle: analyses must be **materialized into
the schema as derived read-model tables** (like `session_model_usage`,
`session_profiles`, the run projection), computed at ingest/read.

Crucially, the script's own queries already read the materialized base tables
(`session_model_usage`, `session_provider_usage_events`, `session_work_events`,
`sessions`) — so almost every analytic is a **re-derivation of something already
materialized** or already exposed as an insight type. The fold-in is mostly
**delete + point CLI/MCP at existing rollups**, plus a small number of genuinely
new monthly/distribution rollups.

## 1. Analytics the script computes (enumerated)

From `analyze()` (lines 244-380) and `build_report()`:

| # | Analytic | Source query in script |
|---|----------|------------------------|
| A1 | Scale: session count, message count, block count | `COUNT(*)` over sessions/messages/blocks |
| A2 | Span (min/max date) | `MIN/MAX(sort_key_ms)` over sessions |
| A3 | Sessions by origin | `GROUP BY origin` |
| A4 | Token economy by cost provenance (input/output/cache_read/cache_write/cost/sessions) | `session_model_usage GROUP BY cost_provenance` |
| A5 | Total cost, total tokens per class (rollup of A4) | python sum over A4 |
| A6 | Cost by model (top 10, cost>0) | `session_model_usage GROUP BY model_name` |
| A7 | Cost provenance breakdown (count + cost) | `session_model_usage GROUP BY cost_provenance` |
| A8 | Per-session cost distribution: median, p90, max, log-band histogram | `SUM(cost_usd) GROUP BY session_id` then python percentiles/histogram |
| A9 | Subscription credit estimate (per-family + total, plan-months) | hand-rolled `_CREDIT_RATES` over `session_model_usage WHERE priced` |
| A10 | Reasoning tokens total | `SUM(last_reasoning_output_tokens)` over `session_provider_usage_events` |
| A11 | Sessions per month | `strftime('%Y-%m') GROUP BY ym` over sessions |
| A12 | Tokens per month | `SUM(last_total_tokens) GROUP BY ym` over provider usage events |
| A13 | Model evolution: tokens/month × top-5 model | `GROUP BY ym, model_name` over provider usage events |
| A14 | Work-event type counts (top 12) | `session_work_events GROUP BY work_event_type` |
| A15 | Session-length distribution: median/p90/max + log-band histogram | `message_count` over sessions, python percentiles |
| A16 | Cache amplification ratio (cache_read/input, blended $/M) | derived from A4 priced row |
| — | SVG chart rendering (`bar_chart`/`hbar_chart`/`line_chart`) | presentation only — discard, see §4 |

## 2. Mapping to existing materialized infrastructure

Confirmed by reading `polylogue/insights/` and `polylogue/storage/insights/`.

| # | Already materialized? | Where | Insight type / surface |
|---|----------------------|-------|------------------------|
| A1 | YES | base tables `sessions`/`messages`/`blocks` | `ArchiveStore.stats_by` → CLI `--stats-by`, MCP `get_stats_by`/`aggregate_sessions`/`archive_coverage` |
| A2 | YES | `sessions.sort_key_ms` | same; also `archive_coverage` buckets |
| A3 | YES | `archive_coverage` group_by=origin | insight type `archive_coverage` (registry.py:517) |
| A4 | YES | `session_model_usage` (per-session, per-model, cost_provenance) materialized by `storage/usage.py`; aggregated into `CostRollupInsight.usage`/`.basis` | insight type `cost_rollups` (registry.py:614); MCP `cost_rollups` |
| A5 | YES | sum of `cost_rollups` rows | `cost_rollups` aggregate |
| A6 | YES | `cost_rollups` keyed by (origin, normalized_model), `total_usd` | `cost_rollups` |
| A7 | YES (status_counts) | `CostRollupInsight.status_counts` / `priced_session_count` / `unavailable_session_count` | `cost_rollups` |
| A8 | PARTIAL | per-session cost exists (`session_costs` insight, `SessionEvidencePayload.total_cost_usd`) but **the distribution (median/p90/max/histogram) is NOT materialized** | NEW — see §3 |
| A9 | YES | `CostBasisPayload.subscription_equivalent_usd` is a materialized basis axis (pricing.py:112), exposed as `cost_rollups` field `sub_usd` (registry.py:635). Script's hand-rolled `_CREDIT_RATES` is **redundant and inferior** — duplicates pricing the substrate already does. | `cost_rollups` (sub_usd/api_usd/catalog_usd/provider_usd) |
| A10 | YES | `SessionEvidencePayload` token fields + provider usage events table; reasoning total is summable | provider usage table; could surface via cost_rollups usage. Currently only raw table. |
| A11 | PARTIAL | day & week summaries exist (`DaySessionSummaryRecord`, `WeekSessionSummaryInsight`) and `archive_coverage` group_by=day/week — **but no MONTH bucket** | NEW month axis — see §3 |
| A12 | PARTIAL | tokens are in per-session profiles + day summaries, but **no token-per-month rollup** | NEW — see §3 |
| A13 | NO | no month×model token rollup exists; `cost_rollups` groups by (origin, model) with **no temporal axis** | NEW — see §3 |
| A14 | YES | `session_work_events` materialized; aggregate counts | insight type `session_work_events` (registry.py:411); MCP `session_work_events` / `workflow_shape_distribution` |
| A15 | PARTIAL | `message_count` per session materialized on `sessions`; distribution not stored | NEW (or accept on-read percentile) — see §3 |
| A16 | YES (derivable) | cache_read & input are materialized in `CostUsagePayload`/`session_model_usage`; ratio is arithmetic over already-stored sums | compute in read surface from `cost_rollups.usage` |

**Headline finding:** A4-A7, A9, A14 (the token economy, cost, subscription, and
workflow core) are **fully materialized already**. The subscription-credit view
(A9) in particular is the substrate's `subscription_equivalent_usd` basis axis —
the script reinventing it with `she-llac.com` rates is exactly the duplication
the operator wants removed. The script should consume `cost_rollups`, not
re-price.

## 3. What is NOT yet materialized → proposed derived read-models

Four gaps, all temporal/distribution rollups. Two design options per gap:
**(a)** extend an existing aggregate rollup, **(b)** add a new rollup table.

### Gap 1 — Monthly corpus rollup (A11, A12) — EXTEND existing summaries
The day/week summary aggregate already exists (`archive_summaries.py`,
`summarize_day`/`summarize_week`, `DaySessionSummaryRecord`). Add a **month**
peer:
- New record `MonthSessionSummaryRecord` (mirror `DaySessionSummaryRecord`:
  `month` (`YYYY-MM`), `source_name`, `session_count`, `logical_session_count`,
  `total_messages`, `total_words`, `total_cost_usd`,
  `total_input_tokens`/`output`/`cache_read`/`cache_write` (add token sums —
  day record currently lacks token columns; carry them from
  `SessionEvidencePayload.total_*_tokens`), `work_event_breakdown`).
- Builder `aggregate_month_session_summary_insights()` next to the week reducer
  in `archive_summaries.py` (reduces day summaries by `month` the same way the
  week reducer reduces by ISO week, line 165-174).
- Cheapest path: add `group_by="month"` to the existing `archive_coverage`
  insight (registry.py:517, currently origin/day/week). That alone satisfies
  A11 (sessions/month) and A12 if coverage carries token sums.

### Gap 2 — Model evolution: month × model token/cost rollup (A13) — NEW table
No existing rollup has both a model and a temporal axis. Add:
- New record `ModelMonthUsageRecord` keyed by `(source_name, month,
  normalized_model)` with `session_count`, `total_tokens`,
  token-class sums, `total_usd`, plus the basis axes
  (`api_equivalent_usd`/`subscription_equivalent_usd`).
- New insight type `model_month_usage` in `registry.py` (mirrors
  `cost_rollups` but with a `month` bucket).
- Builder `aggregate_model_month_usage_insights()` in `archive_rollups.py`
  alongside `aggregate_cost_rollup_insights` (line 203) — same per-model merge
  machinery, additionally bucketed by `profile_bucket_day(profile)[:7]`.

### Gap 3 — Distributions: per-session cost & session-length histograms (A8, A15)
Two sub-options:
- **(a) Materialize** a small `distribution_rollup` table: `(metric, band, count)`
  for `cost_usd` and `message_count` log bands, plus stored `median/p90/max`
  scalars. Rebuilt by the aggregate materializer.
- **(b) Compute on read** in the report surface from already-materialized
  per-session rows (`session_costs`, `sessions.message_count`). Percentiles over
  N≈16k rows are cheap.
- **Recommendation: (a)** to honor the "materialized, not dynamic" principle —
  the whole reason the script is being deleted is that it computes these
  dynamically. A `session_metric_distribution` derived table (metric, log-band,
  count, p50/p90/max) keeps the report a pure reader.

### Gap 4 — Reasoning tokens & cache amplification (A10, A16)
- A16 (cache amplification ratio + blended $/M) is **pure arithmetic** over
  `cost_rollups.usage` (cache_read/input) and `total_usd`/total_tokens — no new
  table; compute in the read surface from the materialized rollup.
- A10 (reasoning total) — fold `reasoning_output_tokens` into the monthly/model
  rollups' token sums (carry it as a token class), sourced from
  `session_provider_usage_events.last_reasoning_output_tokens`.

### Materializer hook
All aggregate rollups (tag, day, week, cost) are built by the **aggregate
refresh** path, not the per-session bundle in
`storage/insights/session/rebuild.py:_build_record_bundle` (575-644, which is
per-session: profile, latency, work-events, phases, run projection). The new
month/model-month/distribution reducers attach to the **same aggregate-rebuild
pass** that already calls `build_session_tag_rollup_records` /
`aggregate_cost_rollup_insights` over the full profile set. Register the new
insight types in `polylogue/insights/registry.py` and add their builders in
`polylogue/insights/archive_rollups.py` + `archive_summaries.py`; storage records
go in `polylogue/storage/insights/aggregate/records.py`. New tables = **index
schema version bump** (deletes-then-defines, rebuild from source — see
CONTRIBUTING.md "Schema-Touching Changes"); they are derived/rebuildable, no
user-data impact.

## 4. Thin read surface that replaces the script

The script's value is a **single bundled longitudinal report**. Replace it with
a thin reader that reads only materialized rollups:

### CLI: `polylogue insights forensics` (a.k.a. report)
- New subcommand under the existing `ops_insights_command` / `analyze insights`
  group (`polylogue/cli/commands/insights.py:198-204`).
- Pure reader: pulls `archive_coverage` (origin + month), `cost_rollups`
  (token economy, cost-by-model, subscription axes), the new
  `model_month_usage`, `session_metric_distribution`, and `session_work_events`
  aggregate. Computes only trivial arithmetic (cache ratio A16, plan-months).
- `--format json|md`; JSON via the standard `emit_success` envelope. No SQL in
  the command — all data via `fetch_insights(...)` registry dispatch.
- SVG charting (`bar_chart`/`hbar_chart`/`line_chart`, ~140 lines): this is
  **presentation**, belongs in `polylogue/rendering/` if charts are still
  wanted, NOT in the analytic path. Recommend dropping inline SVG; if charts are
  desired, add a `rendering/forensics_charts.py` leaf that consumes the same
  insight payloads. The report markdown itself moves to a rendering template.

### MCP
The MCP surface is already sufficient for agents: `cost_rollups`,
`session_costs`, `get_stats_by`, `aggregate_sessions`, `archive_coverage`,
`session_work_events`, `workflow_shape_distribution`, `provider_usage`. Add one
new tool only if the bundled report is wanted as a single agent pull —
`get_forensics_report` mirroring `get_postmortem_bundle` (a distilled
agent-preferred composite over the rollups above). Otherwise no MCP change
needed; the new `model_month_usage` insight gets an MCP list tool for free via
the registry.

### Docs
- Replace `docs/agent-forensics.md` content with a short section in the existing
  insights/analytics docs describing `polylogue insights forensics` and the
  rollup tables it reads. Cross-link from `docs/internals.md` schema-version
  notes (new tables) and `docs/architecture.md` Ring 2 derived read-models.

## 5. Deletion plan (safe once §3-4 land)

`scripts/agent_forensics.py` + `docs/agent-forensics.md` are safe to delete when:
1. Monthly summary (Gap 1) + `model_month_usage` (Gap 2) + distribution rollup
   (Gap 3) are materialized and registered, with the index schema version
   bumped and rebuild-from-source documented.
2. `polylogue insights forensics` (read-only over rollups) reproduces every A1-A16
   analytic, sourcing A9 from `subscription_equivalent_usd` (not the script's
   hand-rolled rates) and A16/A10 as arithmetic over the rollups.
3. The docs section replaces `docs/agent-forensics.md`.

Then remove both files in the same PR that adds the command (surgical renewal —
no deprecation shim). Check for references first:
`grep -rn "agent_forensics\|agent-forensics" .` (also CI workflow command
linters `devtools verify doc-commands` / `ci-workflows`, and any `scripts/`
reference in docs). `scripts/cost_accounting_demo.py` is unrelated and stays.

### Residual notes
- The script's two accounting traps are already honored by the substrate:
  cost-provenance separation = `CostRollupInsight.status_counts` /
  `priced_session_count`; per-event-delta-vs-cumulative = the substrate stores
  `last_*` deltas in `session_provider_usage_events`. No new logic needed — just
  do not re-sum cumulative columns in the reader (read rollups, not raw events).
- A9's `she-llac.com` credit rates and `_MAX20_CREDITS_PER_MONTH` plan cap are
  the one piece of *information* not in the substrate today: the substrate's
  `subscription_equivalent_usd` is a USD basis, not a credit/plan-month view. If
  the plan-months framing is wanted, the credit rates + plan cap belong in the
  pricing catalog (`storage/sqlite/archive_tiers/pricing_seed.py` /
  `archive/semantic/pricing.py`), not in a script — materialize as a basis axis
  or a config constant consumed by the reader.
