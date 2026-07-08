# Insights Rigor Matrix

This is the durable per-product rigor contract for materialized insights
in the Polylogue archive (#1275). Each row describes how to read one
insight product — which fields carry direct evidence from the source
archive, which fields carry probabilistic inference, what fallback
markers callers should consult, what readiness semantics apply, and
which consumer-facing fields are stable surface contract.

The source of truth for this matrix is
`polylogue/insights/rigor.py:list_rigor_contracts()`. The audit runner
that uses the matrix to roll up the live archive is
`polylogue/insights/audit.py:build_insight_rigor_audit_report`, exposed
as `polylogue ops insights audit` on the CLI, `insight_rigor_audit` on the
MCP server, and `Polylogue.insight_rigor_audit()` on the async library
API.

## How to read a row

- **Evidence payload** — the dotted path to a non-optional payload on
  the insight item that is fully grounded in archive counts and
  timestamps. Empty means the product is aggregate-only or
  inference-only.
- **Inference payload** — the dotted path to the probabilistic payload.
  Empty means the product is evidence-only.
- **Fallback markers** — dotted paths whose truthy value flags a row
  as having taken a fallback rather than a fully-grounded inference.
  Consumers that need high-rigor rows only must filter these out.
- **Confidence field** — dotted path to a `[0, 1]` confidence score
  (when the product carries one). Used by the audit runner to bucket
  rows into `low` (<0.34), `mid` (<0.67), `high` (>=0.67), and
  `unknown` (missing or non-numeric).
- **Readiness semantics** — short prose describing how consumers should
  decide whether a row is consumable.
- **Versions** — materialization version fields the row carries. The
  audit runner counts rows whose version is below the current target
  as `stale_version_count`.

## Matrix

### `session_profiles` — Session Profiles

- Evidence payload: `evidence`
- Inference payload: `inference`
- Fallback markers: `enrichment.fallback_reasons`
- Confidence field: `enrichment.confidence`
- Versions: `materializer_version`, `inference_version`,
  `enrichment_version`
- Readiness: Evidence payload is fully grounded in archive counts and
  timestamps. Inference payload is probabilistic — consult
  `inference.support_level` and `inference.engaged_duration_source`
  for grounding. Enrichment payload is also probabilistic and carries
  intent/outcome summaries plus `enrichment.support_level` /
  `enrichment.confidence`. Profiles missing an `inference` payload
  should be treated as evidence-only.
- Consumer-facing fields: `session_id`, `provider_name`, `title`,
  `semantic_tier`, `evidence`, `inference`, `enrichment`,
  `provenance`, `inference_provenance`, `enrichment_provenance`.

### `session_work_events` — Work Events

- Evidence payload: `evidence`
- Inference payload: `inference`
- Fallback markers: `inference.fallback_inference`
- Confidence field: `inference.confidence`
- Versions: `materializer_version`, `inference_version`
- Readiness: Evidence payload describes the message-range and timing
  footprint of the event. Inference payload classifies kind/summary;
  rows with `inference.fallback_inference == True` were emitted by the
  heuristic fallback and should be treated as low-rigor.
- Boundary: `inference.heuristic_label` is a coarse event label inferred from
  local file/tool/text signals. It is not the session-level workflow taxonomy;
  consumers that need whole-session semantics should use
  `session_profiles.workflow_shape` and `session_profiles.terminal_state`.
- Consumer-facing fields: `event_id`, `session_id`,
  `provider_name`, `event_index`, `evidence`, `inference`.

### `session_phases` — Session Phases

- Evidence payload: `evidence`
- Inference payload: _(none)_
- Fallback markers: _(none)_
- Confidence field: _(none)_
- Versions: `materializer_version`
- Readiness: Evidence payload describes the phase's message-range
  timing and tool counts. Phases are deterministic time-gap intervals,
  not intent labels or probabilistic workflow classifications; consumers
  that need intent should use work-event heuristics or session-level
  workflow fields.
- Consumer-facing fields: `phase_id`, `session_id`,
  `provider_name`, `phase_index`, `evidence`.

### `work_threads` — Work Threads

- Evidence payload: `thread`
- Inference payload: _(none)_
- Fallback markers: _(none)_
- Confidence field: _(none)_
- Versions: `materializer_version`
- Readiness: Thread payload is a deterministic rollup over session
  parent/child links; there is no probabilistic inference layer.
  Rigor is governed by the underlying parent-link evidence.
- Consumer-facing fields: `thread_id`, `root_id`, `dominant_repo`,
  `thread`.

### `session_tag_rollups` — Session Tag Rollups

- Evidence payload: _(none — aggregate)_
- Inference payload: _(none — aggregate)_
- Fallback markers: _(none)_
- Confidence field: _(none)_
- Versions: `materializer_version`
- Readiness: Tag rollups aggregate explicit and auto tag counts.
  Auto-tag rows derive from probabilistic enrichment; explicit-tag
  rows are direct evidence. Inspect `explicit_count` vs `auto_count`
  for rigor.
- Consumer-facing fields: `tag`, `session_count`,
  `explicit_count`, `auto_count`, `provider_breakdown`,
  `repo_breakdown`.

### `archive_coverage` — Archive Coverage

- Evidence payload: _(none — aggregate)_
- Inference payload: `work_event_breakdown`
- Fallback markers: _(none)_
- Confidence field: _(none)_
- Versions: _(none — see notes)_
- Readiness: Session/message/word/cost/duration counts and origin/repo
  breakdowns are deterministic SQL aggregates. `work_event_breakdown` is
  the one probabilistic field: its keys are the heuristic
  `work_event_type` labels the session_work_events materializer assigns,
  so it is an aggregation over inferred labels, not raw evidence.
  `provenance` is only populated for day/week grouping (never for the
  default provider grouping).
- Notes: `provenance.materializer_version` is a hardcoded literal `1` for
  day/week grouping with no dedicated version constant, and absent
  entirely for provider grouping — not declared as a version field.
- Field contracts (9e5.29): `avg_messages_per_session`, `avg_user_words`,
  `avg_authored_user_words`, `avg_assistant_words`, `tool_use_percentage`,
  and `thinking_percentage` are `derived` fields whose declared
  denominator (`session_count`, `user_message_count`,
  `authored_user_message_count`, `assistant_message_count`,
  `session_count`, `session_count` respectively) being zero renders the
  field `None` — never a fabricated `0.0`. Day/week grouping does not
  compute the per-role-message-count averages or the percentage fields
  at all (no per-type message counts fetched there), so those fields
  render `None` on every day/week row today, not only zero-denominator
  ones — a documented coverage gap, not a bug.
- Consumer-facing fields: `bucket`, `group_by`, `source_name`,
  `session_count`, `message_count`, `total_cost_usd`,
  `tool_use_percentage`, `thinking_percentage`, `work_event_breakdown`,
  `origin_breakdown`, `repos_active`.

### `tool_usage` — Tool Usage

- Evidence payload: _(none — aggregate)_
- Inference payload: _(none)_
- Fallback markers: `has_coverage_gaps`
- Confidence field: _(none)_
- Versions: `materializer_version`
- Readiness: Every field is a deterministic count, distinct-value count,
  or presence flag read from the canonical actions view; there is no
  heuristic/estimate layer. Check `has_coverage_gaps` (or the per-entry
  `provider_coverage[].data_available`) to distinguish a genuine zero
  tool-use count from an origin with no ingested action data at all.
- Consumer-facing fields: `entries`, `provider_coverage`,
  `total_call_count`, `total_distinct_tools`, `providers_with_data`,
  `providers_without_data`, `has_coverage_gaps`.

### `session_costs` — Session Costs

- Evidence payload: _(none)_
- Inference payload: `estimate`
- Fallback markers: `estimate.missing_reasons`, `estimate.unavailable_reason`
- Confidence field: `estimate.confidence`
- Versions: `materializer_version`
- Readiness: `session_id`/`source_name`/`title`/timestamps are direct
  archive facts. The nested `estimate` payload carries the pricing
  outcome: `estimate.status` is one of exact/priced/partial/unavailable,
  `estimate.confidence` quantifies trust in a non-exact price, and a
  non-empty `estimate.missing_reasons` or a set
  `estimate.unavailable_reason` flags a fallback/unpriced row.
- Consumer-facing fields: `session_id`, `source_name`, `title`,
  `created_at`, `updated_at`, `estimate`, `provenance`.

### `cost_rollups` — Cost Rollups

- Evidence payload: _(none — aggregate)_
- Inference payload: _(none)_
- Fallback markers: `unavailable_session_count`
- Confidence field: `confidence`
- Versions: _(none — see notes)_
- Readiness: session/priced/unavailable counts, `status_counts`,
  `total_usd`, `basis`, and `usage` are grounded SQL sums/counts.
  `confidence` is the one probabilistic signal — a session-count-weighted
  average of the same per-row confidence heuristic `session_costs` uses.
- Notes: `provenance.materializer_version` is a hardcoded literal `0`, a
  sentinel for "computed live at query time", not a stored materialized
  artifact — not declared as a version field.
- Consumer-facing fields: `source_name`, `model_name`,
  `normalized_model`, `session_count`, `priced_session_count`,
  `unavailable_session_count`, `status_counts`, `total_usd`, `basis`,
  `usage`, `confidence`, `per_model_breakdown`.

### `usage_timeline` — Usage Timeline

- Evidence payload: _(none — aggregate)_
- Inference payload: _(none)_
- Fallback markers: _(none)_
- Confidence field: _(none)_
- Versions: _(none — see notes)_
- Readiness: session/event counts, token usage, and `stored_cost_usd` are
  grounded SQL sums. `subscription_credits` is a catalog-rate estimate
  whenever stored credits are absent, indistinguishable in the payload
  from a genuinely stored credit figure — inspect `cost_provenance_counts`
  (exact/priced/estimated/unknown label counts) to judge how much of the
  bucket's cost basis is exact vs. estimated.
- Notes: `provenance.materializer_version` is the same hardcoded literal
  `0` live-aggregation sentinel as `cost_rollups` — not declared as a
  version field.
- Consumer-facing fields: `bucket`, `group_by`, `source_name`,
  `model_name`, `normalized_model`, `session_count`, `event_count`,
  `usage`, `reasoning_output_tokens`, `stored_cost_usd`,
  `subscription_credits`, `cost_provenance_counts`.

### `archive_debt` — Archive Debt

- Evidence payload: _(none)_
- Inference payload: _(none)_
- Fallback markers: _(none)_
- Confidence field: _(none)_
- Versions: _(none — see notes)_
- Readiness: Every row is a live, deterministic health-check result over
  current archive tables (FTS sync, orphaned profile rows, materialization
  staleness, etc.) with no inference or fallback layer. `healthy` is
  literally `issue_count == 0`, not an estimate.
- Notes: no materializer/inference/enrichment version field exists on
  this model at all — every row is computed live, not read from a
  materialized artifact with a version to track.
- Consumer-facing fields: `debt_name`, `category`, `maintenance_target`,
  `issue_count`, `healthy`, `destructive`, `detail`.

## Coverage policy

Every insight product registered in
`polylogue/insights/registry.py:INSIGHT_REGISTRY` must appear above or in
`polylogue/insights/rigor.py:RIGOR_EXEMPT` with an inline justification
(genuinely non-number-bearing products only). `devtools lab policy
insight-honesty` enforces this statically; the audit runner reports an
uncovered product's `coverage_status` as `"uncovered"` rather than
silently omitting it (9e5.28).

## Audit CLI

```bash
polylogue ops insights audit
polylogue ops insights audit --format json
polylogue ops insights audit --insight session_work_events
polylogue ops insights audit --sample-limit 2000
```

Every registered product appears in the report. A product with a contract
reports:

- `sample_size` — rows inspected (bounded by `--sample-limit`)
- `evidence_count`, `inference_count`, `fallback_count` — number of
  rows where the corresponding payload or marker is present
- `stale_version_count` — rows whose materializer/inference/enrichment
  version is below the current target
- `confidence_distribution` — `low`/`mid`/`high`/`unknown` buckets

The runner is read-only and uses the same operations façade as the
CLI/MCP/API surfaces, so the rigor numbers reflect what consumers see
through ordinary list calls.
