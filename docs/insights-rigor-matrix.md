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
- Inference payload: `inference`
- Fallback markers: `inference.fallback_inference`
- Confidence field: `inference.confidence`
- Versions: `materializer_version`, `inference_version`
- Readiness: Evidence payload describes the phase's message-range
  timing and tool counts. Inference payload carries the phase-kind
  classification with a confidence score; `inference.fallback_inference`
  flags heuristic fallback rows.
- Consumer-facing fields: `phase_id`, `session_id`,
  `provider_name`, `phase_index`, `evidence`, `inference`.

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

## Audit CLI

```bash
polylogue ops insights audit
polylogue ops insights audit --format json
polylogue ops insights audit --insight session_work_events
polylogue ops insights audit --sample-limit 2000
```

For each contracted product, the audit reports:

- `sample_size` — rows inspected (bounded by `--sample-limit`)
- `evidence_count`, `inference_count`, `fallback_count` — number of
  rows where the corresponding payload or marker is present
- `stale_version_count` — rows whose materializer/inference/enrichment
  version is below the current target
- `confidence_distribution` — `low`/`mid`/`high`/`unknown` buckets

The runner is read-only and uses the same operations façade as the
CLI/MCP/API surfaces, so the rigor numbers reflect what consumers see
through ordinary list calls.
