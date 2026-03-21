# Artifact Cohort Control Plane Program

Date: 2026-03-21
Status: executed

## Goal

Make artifact coverage, cohorting, and support proof durable, queryable, and
operator-visible without inventing a second parsing truth.

The control plane should materialize facts Polylogue already knows at runtime:

- source artifact identity
- raw blob linkage
- effective provider identity
- artifact taxonomy/classification reason
- decode status
- schema eligibility
- schema resolution support state
- bundle/cohort identity
- Claude sidecar linkage

## Implemented Shape

### 1. Durable artifact ledger in SQLite

Added `artifact_observations` as a derived but durable control-plane table.

Each row represents one observed source artifact, not just one unique raw blob.
That distinction matters:

- `raw_conversations` deduplicates by `raw_id`
- `artifact_observations` preserves source-level identity by
  `source_name + source_path + source_index`

This means two files with identical bytes can still produce:

- one raw row
- two artifact observation rows

That is the correct shape for proof, coverage, and sidecar linkage.

### 2. Observation rows are persisted during acquisition

Acquisition now persists:

- the deduplicated raw blob
- the durable artifact observation

in the same runtime path.

This removed the old bug where duplicate `raw_id`s within one acquisition pass
were skipped before the artifact ledger saw them.

### 3. Historical archives are hydrated on demand

`check --proof`, `check --artifacts`, and `check --cohorts` hydrate missing
historical observation rows from existing `raw_conversations` before reading the
ledger.

This is intentionally a durable metadata update, not a second ephemeral scan.

### 4. `check` now exposes three sibling projections

- `polylogue check --proof`
  - provider-level support proof, unknowns, decode failures, and Claude sidecar linkage
- `polylogue check --artifacts`
  - row-level durable artifact observations
- `polylogue check --cohorts`
  - grouped cohort summaries over the same ledger

All three surfaces read from one underlying truth.

### 5. Path-only Claude sidecars are now acquired

Source walking no longer skips:

- `sessions-index.json`
- `bridge-pointer.json`

These are acquired as artifacts and classified as recognized non-parseable
sidecars instead of being invisible to coverage accounting.

Historical archives only gain those rows after a fresh acquisition pass sees
them. That is the honest boundary.

### 6. Claude subagent provider identity is path-aware

Claude subagent JSONL streams can look Codex-shaped at the payload level.

Provider inference now preserves `claude-code` when the path is inside
`/subagents/`, so:

- Claude subagent streams stay in the Claude cohort/proof surface
- Claude sidecar linkage does not get split across providers

## Query Semantics

### Support statuses

- `supported_parseable`
- `recognized_unparsed`
- `unsupported_parseable`
- `decode_failed`
- `unknown`

### Cohorts

Current cohort identity is the durable exact-structure cohort
(`schema_cluster_id`) plus surrounding artifact/support/resolution fields.

That is enough to make cohorts inspectable and stable without inventing a
parallel cohort registry beside schema package manifests.

## Why this is the right shape

This program does not add a second parser, a second classifier, or a second
schema resolver.

It materializes existing runtime decisions into durable rows and then projects
proof/artifact/cohort views from those rows.

That makes the system:

- more observable
- less ad hoc
- cheaper to verify repeatedly
- more honest about coverage gaps

## Remaining frontier after this

The next best step is not more artifact machinery. It is Step 5 from the main
program:

- publication/repo-shape cleanup
- site/reporting read-model cleanup
- managed artifact placement and repo-overwhelm reduction
