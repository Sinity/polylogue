# Archive Data Products And Live Governance Program

Date: 2026-03-24
Status: active execution program
Role: canonical next broad queue after the source-boundary and runtime-governance campaign

Absorbs and extends as the live queue:

- `source-boundary-and-runtime-governance-program-2026-03-23.md`
- `archive-intelligence-platform-convergence-program-2026-03-23.md`
- `semantic-stack-convergence-program-2026-03-23.md`
- the still-relevant live-governance and consumer-contract reservoir from:
  - `testing-reliability-expansion-program-2026-03-14.md`
  - `canonical-archive-platform-program-2026-03-19.md`
  - `refactoring-first-streamlining-program-2026-03-19.md`

Prerequisite executed programs:

- `source-boundary-and-runtime-governance-program-2026-03-23.md`
- `platform-wide-architecture-and-refactoring-program-2026-03-23.md`
- `archive-intelligence-platform-convergence-program-2026-03-23.md`
- `semantic-stack-convergence-program-2026-03-23.md`
- `schema-and-evidence-pipeline-convergence-program-2026-03-23.md`
- `core-architecture-convergence-program-2026-03-23.md`

Primary design inputs:

- `../planning-and-analysis-map-2026-03-21.md`
- `../../.claude/scratch/027-architecture-review-2026-03-23.md`
- live-archive dogfooding from the 2026-03-23/24 governance wave
- downstream consumer friction observed through Lynchpin warehouse use

## One-Line Goal

Turn Polylogue’s runtime truth into durable, externally consumable archive data
products while making live-archive cleanup, retrieval, and maintenance
governance explicit enough to support real downstream consumers without local
archaeology.

## Why This Is The Right Next Campaign

The just-executed campaign finished source boundaries and runtime maintenance
control planes, but it also made the next broad gap unambiguous:

1. semantic/session products are still mostly runtime-built Python products
   rather than durable read models with freshness/provenance
2. downstream consumers such as Lynchpin still depend on internal Polylogue
   semantics that are not yet presented as one stable product/export contract
3. the live archive now exposes cleanup debt clearly (`orphaned_content_blocks`,
   orphaned attachment refs, older derived drift), but cleanup lineage and
   scoped execution are still too shallow
4. retrieval, grouped stats, and embeddings now have richer action/event truth,
   but they still do not exploit higher-level session products as first-class
   retrieval targets
5. maintenance and validation are stronger, but they are still primarily
   operator-driven rather than governed as durable archive products with
   machine-readable lineage

## Program Thesis

Polylogue should not stop at being an archive runtime. It should expose durable,
versioned archive data products:

1. session- and work-level semantic products
2. stable external-consumer payloads
3. cleanup/repair plans with lineage and scoped execution
4. retrieval surfaces that can target messages, actions, and higher-level
   archive products coherently
5. live governance loops that validate those products against the real archive

## Architectural Rules

### 1. Semantic Products Must Become Durable Read Models

If a semantic/session product is important to Polylogue or to downstream
consumers, it should stop living only as transient Python assembly.

### 2. Consumer Contracts Must Be Versioned And Public

Downstream consumers should not need private helpers, internal imports, or
ad hoc inference of Polylogue internals. Versioned payloads and explicit query
surfaces should exist instead.

### 3. Cleanup Needs Lineage, Not Just Deletion

Live archive cleanup must explain what would change, why it is safe or
destructive, what descendants it affects, and how it maps to current archive
issues.

### 4. Retrieval Should Span Archive Product Layers

Messages, action events, session/work products, and embeddings should participate
in one retrieval/ranking story instead of remaining isolated strata.

### 5. Live Governance Must Be Reproducible

If the real archive matters day to day, the validation, cleanup preview, and
consumer-product checks must be committed as named lanes, not one-off terminal
rituals.

## Phase 1: Durable Semantic And Session Product Read Models

### Goal

Materialize the current semantic/session products as durable archive read
models with provenance, freshness, and rebuild semantics.

### Main Work

- define canonical durable products for at least:
  - session profiles
  - work events
  - work threads
  - session tags
  - high-value markdown/export summaries where they are currently recomputed
- persist materializer version, source lineage, freshness, and pending/stale
  counts for those products
- make semantic/session products load through one repository/control-plane path
  instead of parallel in-memory builders

### Acceptance Criteria

- key semantic/session products are queryable without recomputing them from
  scratch every time
- product freshness and materializer versions are inspectable
- archive health can explain product readiness uniformly

## Phase 2: External Consumer Contract Convergence

### Goal

Give downstream consumers one stable, public Polylogue contract for archive
products.

### Main Work

- define versioned machine-readable payloads for the durable session/work
  products
- expose those products consistently through library, CLI, and MCP surfaces
- remove remaining private-helper style downstream dependencies
- make Polylogue’s own exported surfaces sufficient for Lynchpin-style
  warehouse ingestion

### Acceptance Criteria

- external consumers no longer need private imports or runtime archaeology
- one product/query contract exists across library/CLI/MCP
- versioned payloads are documented and testable

## Phase 3: Live Archive Cleanup Planning And Lineage

### Goal

Make cleanup a first-class governed archive operation rather than a bag of
destructive SQL routines.

### Main Work

- add cleanup-plan/report surfaces that explain:
  - candidate rows
  - scope
  - descendant impact
  - reason/evidence
  - whether a cleanup is archive-destructive or derived-only
- support scoped cleanup targets instead of all-or-nothing cleanup
- make preview/apply/report lineage durable and queryable
- close currently visible live debt like orphaned content blocks through an
  intentional operator path rather than open-ended warning status

### Acceptance Criteria

- cleanup preview is explainable and scoped
- cleanup apply leaves durable lineage
- live archive debt can be reduced intentionally rather than merely observed

## Phase 4: Retrieval And Archive Product Convergence

### Goal

Let retrieval, grouped stats, and embeddings operate across archive product
layers instead of only messages/actions.

### Main Work

- make retrieval aware of session/work products as first-class searchable
  units or facets
- improve grouped stats/query output to span conversations, action events, and
  durable semantic products coherently
- decide where embeddings should target messages, actions, or session/work
  products distinctly
- reduce duplicated ranking and summary logic across query, MCP, and consumer
  export surfaces

### Acceptance Criteria

- retrieval can target more than raw message text meaningfully
- grouped stats and downstream exports use one archive-intelligence grammar
- embedding freshness/provenance extends to the product layers it actually feeds

## Phase 5: Maintenance Automation And Background Governance

### Goal

Move from manual maintenance-only workflows to explicit upkeep automation and
rebuild governance.

### Main Work

- define named upkeep operations for derived models and product read models
- make freshness drift actionable through rebuild/recompute entrypoints
- decide what should run after ingest, what should stay operator-triggered, and
  what should remain preview-only
- keep these operations typed and auditable

### Acceptance Criteria

- major derived/product surfaces have deliberate upkeep rules
- health reports point to concrete upkeep actions
- maintenance stops being a loose collection of commands

## Phase 6: Module Topology And Refactoring Follow-Through

### Goal

Use the product/governance convergence to further narrow the remaining broad
runtime clusters.

### Main Work

- split any remaining broad semantic/session-product modules once the durable
  product seams are real
- narrow cleanup/export/warehouse-oriented modules around the new product
  boundaries
- keep package exports aligned to the true public contract rather than internal
  implementation convenience

### Acceptance Criteria

- remaining broad modules shrink around the new durable product seams
- module topology reflects archive product boundaries more honestly

## Phase 7: Live Governance And Consumer Validation Lanes

### Goal

Make the real archive and real downstream-consumer scenarios part of committed
validation.

### Main Work

- add named validation lanes for:
  - durable semantic/session product rebuilds
  - consumer payload/export contracts
  - cleanup preview/apply/report flows
  - live archive product and retrieval memory budgets
- dogfood those lanes against the actual archive and downstream integration
  scenarios

### Acceptance Criteria

- real-archive and consumer-contract regressions are reproducible
- memory/performance budgets are explicit at the product layer
- archive governance and consumer confidence share one validation story

## Execution Order

1. durable semantic and session product read models
2. external consumer contract convergence
3. live archive cleanup planning and lineage
4. retrieval and archive product convergence
5. maintenance automation and background governance
6. module topology and refactoring follow-through
7. live governance and consumer validation lanes

## First Concrete Slice

Start with the highest-leverage bridge between current work and real usage:

- define durable read models for session profiles and work events
- expose one stable machine payload for them
- wire live freshness/provenance into health and maintenance
- dogfood the resulting surface against the current Polylogue archive and the
  Lynchpin-style consumer path
