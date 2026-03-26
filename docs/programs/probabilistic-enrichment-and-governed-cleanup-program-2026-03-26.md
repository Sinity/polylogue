[← Back to README](./README.md)

# Probabilistic Enrichment And Governed Cleanup Program

Date: 2026-03-26
Status: executed broad execution program
Role: executed record for probabilistic enrichment, retrieval-band rollout, and governed live cleanup

Replaced as the live queue by:

- `product-and-runtime-topology-cleanup-program-2026-03-26.md`

## Execution Record

This program is executed.

The main codebase changes were:

- heuristic inference contract hardening across durable session products,
  adding explicit `support_level`, `support_signals`,
  `engaged_duration_source`, fallback markers, and inference-strength fields
  for profiles, work events, and phases
- a separate probabilistic enrichment product family added to durable
  `session_profiles`, including contract/storage/query support for intent,
  outcome, blockers, refined work kind, confidence, support, and provenance
- public/operator convergence for enrichment products across CLI, archive
  library, sync, repository, and MCP via `products enrichments`,
  `list_session_enrichment_products(...)`, and `session_enrichments`
- enrichment retrieval rollout through `session_profile_enrichment_fts`,
  derived-model status, archive health exposure, retrieval-band reporting, and
  repair/debt accounting
- governed live cleanup closure for `orphaned_content_blocks` and
  `orphaned_attachments`, including preview/apply/validation lineage and live
  archive cleanup application
- new validation lanes for heuristic inference, probabilistic enrichment,
  enrichment live dogfooding, and governed cleanup live validation

Live archive closure after execution:

- `products status --json` reports `5618` profiles, `17833` work events,
  `12634` phases, `4592` tag rollups, and `1295` day summaries, all ready
- `products enrichments --json` returns durable enrichment products with live
  confidence/support/provenance payloads
- `products debt --json` reports zero actionable debt, with
  `orphaned_content_blocks` and `orphaned_attachments` both validated after
  governed apply plus validation preview
- `check --json` reports zero orphaned content blocks and zero orphaned
  attachments, while all retrieval/status bands except transcript embeddings
  are ready
- `embed --stats --json` now reports `retrieval_enrichment` ready alongside the
  existing evidence/inference bands

Validation and live proofs that closed this program:

- `ruff check $(git diff --name-only --diff-filter=d -- '*.py')`
- `pytest -q -n 0 tests/unit/cli/test_products.py tests/unit/core/test_facade_api.py tests/unit/mcp/test_tool_contracts.py tests/unit/storage/test_embedding_stats.py tests/unit/core/test_health_core.py`
  → `134 passed`
- `pytest -q -n 0 tests/unit/cli/test_check.py tests/unit/cli/test_embed.py tests/unit/cli/test_click_app.py tests/unit/devtools/test_validation_lanes.py tests/unit/storage/test_backend.py tests/integration/test_health.py`
  → `284 passed`
- `python -m devtools.run_validation_lanes --lane probabilistic-enrichment-hardening`
- `python -m devtools.run_validation_lanes --lane governed-cleanup-live`
- `journalctl -u earlyoom --since '60 minutes ago'` showed no new kills during
  closure work

Prerequisite executed programs:

- `evidence-and-stewardship-platform-convergence-program-2026-03-24.md`
- `cleanup-and-architectural-debt-retirement-program-2026-03-24.md`
- `semantic-product-normalization-and-toolchain-convergence-program-2026-03-24.md`
- `domain-read-model-and-live-archive-stewardship-program-2026-03-24.md`

Primary design inputs:

- the now-executed evidence/inference tier split
- live archive dogfooding of `products`, `check`, `embed --stats`, and the
  named evidence/stewardship validation lanes
- the explicit observation that heuristic semantics already influence
  downstream timelines and summaries, but their quality and provenance are
  still only moderately strong
- the still-open live archive debt around orphaned content blocks,
  orphaned attachments, and missing transcript embeddings

## One-Line Goal

Improve inference quality without polluting the evidence tier, make optional
probabilistic enrichment a first-class governed product family, and close the
remaining destructive live cleanup debt with durable lineage and validation.

## Why This Is The Right Next Campaign

The last wave made Polylogue honest about the difference between evidence and
inference. The next drag is that the inference tier is still mostly heuristic
and yet increasingly important:

- work events, phases, engaged duration, project inference, and auto-tags still
  derive from deterministic-but-approximate rules
- downstream systems already consume these outputs for timeline and activity
  views
- the archive has the raw material for better enrichment:
  - user turns
  - action events
  - touched files and repos
  - timestamp and chronology evidence
- transcript embeddings are still absent, so semantic retrieval and future
  enrichment have no production control plane yet
- the archive still has explicit cleanup debt that is previewable but not yet
  fully governed through apply-and-validate lineage

This means the next broad queue should not be another structural breakup wave.
It should turn Polylogue into a stronger evidence-first intelligence platform:
better probabilistic inference, clearer enrichment provenance, and governed live
cleanup.

## Program Thesis

Polylogue should converge around five truths:

1. evidence remains canonical and deterministic
2. heuristic inference remains separate, versioned, and confidence-scored
3. richer probabilistic enrichment can exist, but only as its own governed
   product family above the evidence tier
4. retrieval and embeddings should operate on deliberate bands, not “whatever
   transcript text exists”
5. destructive live cleanup should have the same lineage, preview/apply, and
   validation discipline as derived-model rebuilds

## Architectural Rules

### 1. No Hidden Promotion Of Guesses

LLM- or embedding-assisted outputs must not silently overwrite evidence-tier
facts or merge invisibly into deterministic heuristic products.

### 2. User-Intent And Action Signals Come First

When building new retrieval and enrichment bands, prioritize user messages,
action summaries, touched paths, repo signals, and explicit outcomes over bulk
assistant text.

### 3. Confidence And Provenance Are Mandatory

Every new probabilistic enrichment family must expose:

- model or algorithm family
- version
- prompt/config lineage where relevant
- confidence/support
- input band summary

### 4. Cleanup Must Be Governed, Not Hidden

Orphaned content blocks and orphaned attachments should move from “known debt”
to preview/apply/validate lineage with durable records and explicit operator
surfaces.

### 5. Live Archive Proof Remains Mandatory

New enrichment and cleanup surfaces are not complete until they run on the real
archive under explicit validation and memory budgets.

## Phase 1: Heuristic Inference Contract Hardening

### Goal

Make the current heuristic inference tier narrower, better calibrated, and more
honest before adding richer probabilistic layers.

### Main Work

- separate strongly supported heuristic inference from weak fallback inference
- add explicit support/evidence summaries for work events, phases, engaged
  duration, project inference, decisions, and auto-tags
- calibrate confidence outputs against actual available evidence, not only
  derived dominant-category counts
- narrow merged product defaults where the current mixed view encourages
  consumers to treat inference as fact

### Acceptance

- heuristic products expose materially better support metadata
- weak inference is mechanically distinguishable from stronger inference

## Phase 2: Probabilistic Enrichment Product Family

### Goal

Add a separate governed enrichment tier above the heuristic layer.

### Main Work

- define durable enrichment product contracts for session intent, outcome,
  blockers, and higher-level work-kind refinement
- add storage/readiness/provenance for optional embedding- or model-backed
  enrichment outputs
- make these products queryable from CLI, library, sync, repository, and MCP
  without changing evidence-tier truth

### Acceptance

- enrichment outputs are queryable and clearly separate from heuristic session
  products
- no consumer needs to guess whether a field came from evidence, heuristics, or
  probabilistic enrichment

## Phase 3: Retrieval And Embedding Rollout

### Goal

Turn retrieval bands into a real production substrate rather than a status-only
 surface.

### Main Work

- add durable band definitions for user-intent, action-summary, evidence
  summary, inference summary, and optional enrichment summary retrieval text
- wire transcript and product embedding materialization onto those bands
- expose freshness/provenance and coverage for each band
- ensure semantic query tooling can pivot between transcript-only,
  evidence-rich, inference-rich, and enrichment-rich retrieval

### Acceptance

- `embed --stats` reports real band coverage, not only empty readiness slots
- archive retrieval can target the right semantic bands explicitly

## Phase 4: Consumer And Operator Contract Convergence

### Goal

Make downstream consumers and operator surfaces use the upgraded semantic model
deliberately.

### Main Work

- tighten session/product APIs so evidence, heuristic inference, and
  probabilistic enrichment can be requested separately or composed intentionally
- align grouped stats, maintenance status, publication summaries, and MCP
  product tools with the new enrichment tier
- document downstream migration expectations for Lynchpin-class consumers

### Acceptance

- public surfaces use one consistent tier vocabulary
- operator outputs no longer blur heuristic inference and richer enrichment

## Phase 5: Governed Live Cleanup

### Goal

Close the explicit destructive archive debt with durable lineage and validation.

### Main Work

- add preview/apply/validate lineage for orphaned content-block cleanup
- add preview/apply/validate lineage for orphaned attachment cleanup
- expose cleanup governance and latest validation state as durable products
- ensure cleanup operations preserve derived-model readiness and product
  consistency across rebuilds

### Acceptance

- destructive cleanup targets are no longer only “preview debt”
- live archive governance surfaces can show preview, apply, and validation
  history per cleanup family

## Phase 6: Validation, Memory, And Refactoring Closure

### Goal

Close the campaign with explicit proof that enrichment and cleanup are
operationally sound.

### Main Work

- add named validation lanes for:
  - heuristic inference contracts
  - probabilistic enrichment contracts
  - retrieval-band materialization
  - governed cleanup preview/apply/validate
  - live archive enrichment and cleanup memory budgets
- continue retiring broad mixed-role modules exposed during this wave, but only
  where they block the semantic and governance goals above

### Acceptance

- live validation lanes pass against the real archive
- earlyoom/memory regressions are caught by named lanes rather than rediscovered
  manually

## Validation And Closure

This program is complete only when all of the following are true:

- heuristic inference exposes stronger support/confidence metadata
- probabilistic enrichment exists as a separate, governed product family
- retrieval and embeddings have real band materialization and status
- destructive cleanup lineage covers preview, apply, and validate
- live archive validation and memory-budget lanes pass for the new surfaces

## Recommended Execution Order

1. heuristic inference contract hardening
2. probabilistic enrichment product family
3. retrieval and embedding rollout
4. consumer and operator contract convergence
5. governed live cleanup
6. validation, memory, and refactoring closure
