[← Back to README](./README.md)

# Evidence And Stewardship Platform Convergence Program

Date: 2026-03-24
Status: executed
Role: executed broad queue after the cleanup-only debt-retirement wave

Absorbs and replaces as the live queue:

- `consumer-contracts-and-governed-live-cleanup-program-2026-03-24.md`
- `cleanup-and-architectural-debt-retirement-program-2026-03-24.md`
- the semantic reliability concerns raised during archive dogfooding around
  heuristic work events, phases, engaged time, decisions, and project
  attribution

Prerequisite executed programs:

- `cleanup-and-architectural-debt-retirement-program-2026-03-24.md`
- `semantic-product-normalization-and-toolchain-convergence-program-2026-03-24.md`
- `domain-read-model-and-live-archive-stewardship-program-2026-03-24.md`
- `runtime-substrate-decomposition-and-contract-hardening-program-2026-03-24.md`
- `archive-data-products-and-live-governance-program-2026-03-24.md`

Primary design inputs:

- `../planning-and-analysis-map-2026-03-21.md`
- the just-executed cleanup-only retirement wave
- live `products`, `check`, `embed --stats`, and semantic-query dogfooding
- the explicit observation that some current semantic products are evidence-rich
  but heuristic, and downstream consumers can currently treat them too much like
  canonical facts

## One-Line Goal

Separate explicit archive evidence from inferred semantics, expose that split as
the durable ecosystem contract, and make stewardship, retrieval, and downstream
consumption operate on that clearer two-tier model.

## Why This Is The Right Next Campaign

The cleanup wave reduced structural debt. The next drag is semantic and
contractual:

- Polylogue already extracts strong evidence: timestamps, roles, actions, tool
  usage, touched paths, repo paths, attachments, counts
- Polylogue also computes heuristic products: work events, phase kinds,
  engaged minutes, decisions, some project attribution, auto-tags
- those tiers currently coexist inside the same session-product family and can
  be consumed as if they had the same stability
- downstream systems such as Lynchpin increasingly depend on these products
- retrieval, embeddings, and future LLM enrichment should build on the evidence
  tier, not blur it

This means the next broad queue should not just add more heuristics. It should
make the semantic contract honest, typed, versioned, and durable.

## Execution Record

Executed on 2026-03-26.

The main implementation outcomes were:

- explicit evidence-tier and inference-tier payload contracts were added to the
  durable session-profile, work-event, and phase product families
- evidence/inference provenance, inference versioning, and tier-specific search
  bands were persisted in the session-product read models
- profile products now support explicit `merged`, `evidence`, and `inference`
  retrieval across CLI, library, sync, repository, and MCP surfaces
- retrieval health and embedding health now distinguish transcript,
  evidence-retrieval, and inference-retrieval bands
- session-product status, repair, debt, and health surfaces now report the
  evidence/inference split explicitly instead of one mixed “semantic” state
- live migration compatibility was closed for upgraded databases, including
  blank tier-search text, legacy `payload_json` columns, and migrated rows with
  empty tier payload bodies
- named contract and live validation lanes now cover evidence-tier contracts,
  inference-tier contracts, mixed consumer surfaces, retrieval-band readiness,
  and live archive migration under memory budget

Live-archive closure evidence:

- `python -m devtools.run_validation_lanes --lane evidence-stewardship-live`
  passed against `/home/sinity/.local/share/polylogue/polylogue.db`
- `products status --json` now reports the durable session-product bands ready:
  `5618` profiles, `17833` work events, `12634` phases, `4592` tag rollups,
  `1295` day summaries, and `1295` week summaries
- `products profiles --tier evidence --json`,
  `products profiles --tier inference --json`, `products work-events --json`,
  and `products phases --json` all returned live tiered products successfully
- `embed --stats --json` now reports retrieval-band readiness separately
- the maintenance memory-budget lane stayed well within budget (`77.9 MB` peak
  RSS for the live maintenance preview)

Remaining live archive debt is now explicit rather than hidden:

- `15781` orphaned content blocks
- `2378` orphaned attachment rows
- transcript embeddings still intentionally remain unmaterialized

## Program Thesis

Polylogue should converge around five explicit truths:

1. evidence-tier facts are canonical and deterministic
2. inference-tier products are separate, versioned, confidence-scored, and
   provenance-carrying
3. downstream consumers should be able to query either tier explicitly instead
   of guessing what is inferred
4. retrieval/embedding/intelligence surfaces should compose user intent,
   action summaries, and evidence-rich session products deliberately
5. live archive stewardship should stay first-class and attach to both evidence
   and inference product readiness

## Architectural Rules

### 1. Do Not Present Heuristics As Facts

Work kind, phase kind, engaged minutes, decision extraction, weak project
inference, and auto-tags must not be exposed as if they were the same kind of
truth as timestamps, action events, explicit paths, or explicit provider data.

### 2. One Canonical Evidence Band

Explicit evidence should have one obvious home and contract reused across
session products, retrieval, MCP, CLI, sync, and downstream consumption.

### 3. Inference Must Carry Provenance

Every inferred semantic family should expose at least:

- inference version
- inference source family
- confidence / support
- evidence references or summary inputs

### 4. Retrieval Must Use The Right Text

Embeddings and semantic retrieval should prioritize the highest-signal content:

- user turns
- action-event summaries
- touched files/projects/branches
- explicit outcomes when present

not indiscriminately the entire transcript body.

### 5. Live Governance Remains Part Of Closure

Any migration from the current mixed semantic products to the new two-tier model
must be closed against the real archive with readiness, rebuild, and memory
budget checks.

## Phase 1: Evidence-Tier Contract Extraction

### Goal

Make the explicit evidence layer a first-class durable contract rather than a
set of internals embedded inside semantic products.

### Main Work

- define canonical durable evidence contracts for:
  - first/last message timestamps
  - action events and tool summaries
  - explicit repo/project/path evidence
  - attachments and media evidence
  - provider/source identifiers and chronology evidence
- expose those contracts through archive products / repository / CLI / MCP /
  sync surfaces without requiring full conversation hydration
- identify current session-product fields that are actually evidence and move
  them to the evidence tier cleanly

### Acceptance

- consumers can page explicit evidence without consuming heuristic work-kind or
  phase labeling payloads
- evidence products are queryable by session date, first-message time, provider,
  project, repo, and action family

## Phase 2: Inference-Tier Product Governance

### Goal

Move heuristic session semantics onto a clearly inferred tier with explicit
confidence and provenance.

### Main Work

- split current heuristic products into typed inference contracts for:
  - work events / work kind
  - session phases / phase kind
  - engaged duration
  - decision extraction
  - weak project attribution fallback
  - auto-tags derived from inferred semantics
- add inference versioning, confidence/support fields, and readiness tracking
- ensure durable session-product rebuilds can refresh evidence and inference
  tiers independently when only one layer changes

### Acceptance

- no heuristic field is exposed without an inference/provenance story
- operators and downstream consumers can distinguish evidence-backed from
  inferred values mechanically

## Phase 3: Consumer Contract Convergence

### Goal

Make the two-tier semantic model the actual ecosystem API.

### Main Work

- widen product APIs so downstream consumers can explicitly request:
  - evidence-tier only
  - inference-tier only
  - merged views when appropriate
- align CLI, MCP, library, and sync results so they use the same tier language
- update the downstream integration expectations for Lynchpin-class consumers so
  they stop treating inferred semantics as canonical facts by default

### Acceptance

- consumer-facing outputs name their tier honestly
- no major consumer-facing surface forces mixed evidence/inference payloads when
  the caller only needs one tier

## Phase 4: Retrieval, Embeddings, And Intelligence Alignment

### Goal

Make semantic retrieval build on the right archive signals.

### Main Work

- define retrieval text bands for:
  - user-intent text
  - action-event summaries
  - evidence summaries
  - inferred summaries
- expose embedding freshness/provenance per retrieval band
- add archive-health/readiness surfaces for retrieval-band coverage
- ensure grouped stats and semantic search can pivot on evidence vs inference
  rather than only transcript/message text

### Acceptance

- retrieval health explains what is actually embedded
- evidence-rich and inference-rich retrieval lanes are intentionally separate

## Phase 5: Stewardship And Live Migration Governance

### Goal

Migrate the live archive to the clearer two-tier semantic model safely.

### Main Work

- add migration/rebuild paths for evidence-tier and inference-tier products
- persist maintenance lineage for those migrations
- validate that the live archive remains queryable throughout the transition
- keep destructive cleanup lineage and semantic-product migration lineage
  together in the broader stewardship plane

### Acceptance

- live rebuild status shows evidence-tier and inference-tier readiness
- maintenance history records migrations explicitly

## Phase 6: Validation And Refactoring Closure

### Goal

Close the campaign with explicit proof that the new semantic model is narrower,
clearer, and operationally sound.

### Main Work

- add named validation lanes for:
  - evidence-tier contracts
  - inference-tier contracts
  - mixed consumer contract compatibility
  - retrieval-band readiness
  - live archive migration and memory budget
- continue narrowing any still-broad modules exposed during this campaign,
  especially remaining domain-local dense internals that still mix evidence,
  inference, and retrieval shaping

### Acceptance

- the evidence/inference split is reflected in code topology and public
  contracts, not just docs
- live validation lanes pass against the real archive

## Validation And Closure

This program is complete only when all of the following are true:

- durable evidence-tier products and inference-tier products both exist and are
  queryable
- consumer-facing contracts expose tier/provenance explicitly
- retrieval/embedding health distinguishes evidence vs inference bands
- named validation lanes cover evidence contracts, inference contracts, live
  migration, and memory budget
- live archive checks and dogfooding confirm the migration story

## Recommended Execution Order

1. evidence-tier contract extraction
2. inference-tier product governance
3. consumer contract convergence
4. retrieval, embeddings, and intelligence alignment
5. stewardship and live migration governance
6. validation and refactoring closure
