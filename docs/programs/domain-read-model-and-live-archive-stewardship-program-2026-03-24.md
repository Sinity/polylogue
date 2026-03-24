# Domain Read-Model And Live Archive Stewardship Program

Date: 2026-03-24
Status: active execution program
Role: canonical next broad queue after the runtime-substrate decomposition and contract-hardening campaign

Absorbs and extends as the live queue:

- `runtime-substrate-decomposition-and-contract-hardening-program-2026-03-24.md`
- `archive-data-products-and-live-governance-program-2026-03-24.md`
- `source-boundary-and-runtime-governance-program-2026-03-23.md`
- `archive-intelligence-platform-convergence-program-2026-03-23.md`
- the still-relevant live-governance and refactoring reservoir from:
  - `platform-wide-architecture-and-refactoring-program-2026-03-23.md`
  - `canonical-archive-platform-program-2026-03-19.md`
  - `refactoring-first-streamlining-program-2026-03-19.md`
  - `testing-reliability-expansion-program-2026-03-14.md`

Prerequisite executed programs:

- `runtime-substrate-decomposition-and-contract-hardening-program-2026-03-24.md`
- `archive-data-products-and-live-governance-program-2026-03-24.md`
- `source-boundary-and-runtime-governance-program-2026-03-23.md`
- `archive-intelligence-platform-convergence-program-2026-03-23.md`
- `semantic-stack-convergence-program-2026-03-23.md`
- `schema-and-evidence-pipeline-convergence-program-2026-03-23.md`
- `core-architecture-convergence-program-2026-03-23.md`

Primary design inputs:

- `../planning-and-analysis-map-2026-03-21.md`
- `../../.claude/scratch/027-architecture-review-2026-03-23.md`
- post-runtime-substrate hotspot scan from 2026-03-24
- live archive governance output after the runtime-substrate closure pass

## One-Line Goal

Turn Polylogue’s remaining broad high-level data/consumer surfaces into explicit
domain-banded read-model and stewardship subsystems, while making live archive
cleanup and external-consumer behavior governed enough to support real daily
use without code archaeology.

## Why This Is The Right Next Campaign

The runtime substrate is now narrower, but the next broad architectural drag has
shifted upward in the stack rather than disappearing.

Current hotspot scan after the runtime-substrate wave:

- `polylogue/storage/backends/queries/session_products.py` — 765 LOC
- `polylogue/rendering/semantic_surface_declarations.py` — 749 LOC
- `polylogue/lib/models.py` — 714 LOC
- `polylogue/storage/repository_reads.py` — 712 LOC
- `polylogue/lib/query_execution.py` — 552 LOC
- `polylogue/storage/action_event_lifecycle.py` — 536 LOC
- `polylogue/operations/archive.py` — 509 LOC
- `polylogue/cli/commands/products.py` — 468 LOC
- `polylogue/storage/backends/async_sqlite.py` — 432 LOC
- `polylogue/cli/check_workflow.py` — 405 LOC

The important point is that these are no longer low-level substrate monoliths.
They are the remaining places where Polylogue still mixes:

1. domain models with too many unrelated concepts in one root
2. repository reads and product queries across several domain bands
3. external-consumer/analytics surfaces with public-contract drift
4. live archive cleanup debt that is visible but not yet first-class governed
5. productized read models and retrieval/analytics surfaces that still meet too
   late in the stack

## Program Thesis

Polylogue should now converge the high-level archive stack around three things:

1. domain-banded canonical models
2. explicit durable read-model query surfaces
3. governed live archive stewardship

The result should be a platform where:

- conversation/message/block/action/session/work/product semantics are easier to
  trace in code
- external consumers query stable read products rather than reconstructing them
- live cleanup debt is explicit, scoped, and lineage-backed
- analytics, products, retrieval, and maintenance operate on the same durable
  truth surfaces

## Architectural Rules

### 1. Domain Roots Must Stop Mixing Unrelated Shapes

`models.py`-style roots should not keep accumulating every archive concept.
Conversation/message/block/action/session/product models need clearer ownership.

### 2. Repository Reads Must Be Banded By Read Model

Archive reads, durable product reads, maintenance/status reads, and retrieval
queries should not share one large repository-read surface by accident.

### 3. Consumer Surfaces Must Query Durable Truth

CLI/MCP/library/archive-product consumers should stop rebuilding or reshaping
the same data differently in each layer.

### 4. Live Cleanup Debt Must Be Governed, Not Just Reported

If the archive says there are `15,781` orphaned content blocks, the codebase
should expose that debt through scoped, explainable cleanup control planes and
validation lanes instead of leaving it as a standing warning.

### 5. Declarative Catalogs Are Fine, But They Must Be Clearly Isolated

Large declaration-only modules such as semantic surface catalogs are acceptable
only when the declaration role is explicit and separated from execution logic.

## Phase 1: Core Domain Model Decomposition

### Goal

Break the remaining broad domain roots into explicit bands.

### Main Work

- decompose `lib/models.py` into domain-focused modules for at least:
  - conversation/message graph models
  - content blocks / attachment-related models
  - render/view/projection models
  - action/event-related models still living there
- keep imports and downstream semantics canonical rather than leaving multiple
  parallel definitions
- revisit adjacent roots (`types.py`, semantic helpers) so the moved models land
  in coherent ownership rather than a new sprawl

### Acceptance Criteria

- `lib/models.py` stops being a broad catch-all root
- the moved model bands are easier to navigate and own
- downstream imports trace back to one deliberate domain module per concept

## Phase 2: Repository Read And Product Query Convergence

### Goal

Split archive reads, product reads, and maintenance/status reads into explicit
bands instead of broad shared surfaces.

### Main Work

- decompose `storage/repository_reads.py` into at least:
  - archive conversation/message/tree reads
  - durable product reads
  - maintenance/readiness/status reads
  - retrieval-oriented aggregate reads
- narrow `storage/backends/queries/session_products.py` into:
  - profile/work/phase queries
  - thread queries
  - tag/day/week summary queries
  - readiness/support aggregations
- keep public repository contracts stable while reducing mixed ownership

### Acceptance Criteria

- repository read ownership is explainable by read-model band
- session-product SQL/query code stops living in one large mixed file
- product/status/retrieval surfaces consume the same underlying query bands

## Phase 3: External Consumer And Analytics Contract Convergence

### Goal

Stop letting consumer-facing analytics/products drift across CLI, operations,
library, and MCP layers.

### Main Work

- revisit `operations/archive.py`, `cli/commands/products.py`, `cli/helpers.py`,
  and related consumer surfaces
- centralize provider/product analytics contracts so tests and external
  consumers stop depending on incidental module exports
- make versioned archive-product payloads and analytics payloads trace through
  one canonical contract path
- confirm Lynchpin-style consumption is aligned with the public surface instead
  of relying on private reconstruction

### Acceptance Criteria

- analytics/product consumers depend on stable public contracts
- library/CLI/MCP payloads align more directly
- import-path drift like the recent `ProviderMetrics` break becomes less likely

## Phase 4: Live Archive Stewardship And Cleanup Convergence

### Goal

Make the remaining live archive cleanup debt a first-class governed subsystem.

### Main Work

- turn orphaned-content-block cleanup into an explicit scoped target with:
  - clear evidence
  - preview/apply lineage
  - validation lanes
  - live-safe operator workflows
- align orphaned attachment/content-block/read-model cleanup under one
  stewardship vocabulary
- make health, products, and maintenance preview surfaces agree on the meaning
  of archive debt

### Acceptance Criteria

- live cleanup debt is queryable, scoped, and governable
- health, repair, and product status stop describing the same debt differently
- at least one named live stewardship lane covers the real archive debt path

## Phase 5: Retrieval, Products, And Analytics Surface Convergence

### Goal

Make productized read models participate more directly in retrieval and grouped
analytics instead of staying parallel to the message/action lanes.

### Main Work

- revisit grouped stats and retrieval surfaces so session/work/product read
  models can be targeted more directly where it makes semantic sense
- evaluate whether higher-level product retrieval needs its own query/result
  contract instead of ad hoc archive-operation assembly
- align embedding/readiness/product status surfaces around the same public
  archive-product vocabulary

### Acceptance Criteria

- retrieval/analytics/product surfaces have fewer parallel result grammars
- durable read products participate more directly in operator queries
- external consumers need less custom stitching to answer archive-level
  questions

## Phase 6: Validation, Memory, And Live Governance Hardening

### Goal

Keep the next high-level refactor wave tied to real archive use, not just local
module tidiness.

### Main Work

- add named lanes for:
  - domain-model and repository-read closure
  - live archive stewardship / cleanup preview / apply
  - external consumer and analytics contracts
  - productized retrieval/analytics memory budgets where relevant
- dogfood the real archive and keep `earlyoom` sensitivity explicit in the lane
  story

### Acceptance Criteria

- the new high-level read/stewardship surfaces have named validation lanes
- live governance stays part of the closure loop
- memory and consumer-surface regressions become reproducible rather than
  anecdotal

## Execution Order

1. core domain model decomposition
2. repository read and product query convergence
3. external consumer and analytics contract convergence
4. live archive stewardship and cleanup convergence
5. retrieval, products, and analytics surface convergence
6. validation, memory, and live governance hardening

## First Concrete Slice

Start where the remaining structural drag is strongest and most central:

- split `lib/models.py` into explicit domain-banded modules
- split `storage/repository_reads.py` and `storage/backends/queries/session_products.py`
  into archive/product/status query bands
- then use that narrower read-model substrate to drive the next consumer and
  live-stewardship convergence cuts
