[← Back to README](./README.md)

# Semantic Product Normalization And Toolchain Convergence Program

Date: 2026-03-24
Status: executed
Role: executed convergence program for semantic/session product normalization, operator/toolchain narrowing, schema convergence, and live cleanup governance

Absorbs and extends as the live queue:

- `domain-read-model-and-live-archive-stewardship-program-2026-03-24.md`
- `runtime-substrate-decomposition-and-contract-hardening-program-2026-03-24.md`
- `archive-data-products-and-live-governance-program-2026-03-24.md`
- `semantic-stack-convergence-program-2026-03-23.md`
- `schema-and-evidence-pipeline-convergence-program-2026-03-23.md`
- the still-relevant normalization/refactoring reservoir from:
  - `platform-wide-architecture-and-refactoring-program-2026-03-23.md`
  - `canonical-archive-platform-program-2026-03-19.md`
  - `refactoring-first-streamlining-program-2026-03-19.md`
  - `testing-reliability-expansion-program-2026-03-14.md`

Prerequisite executed programs:

- `domain-read-model-and-live-archive-stewardship-program-2026-03-24.md`
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
- the 2026-03-24 hotspot scan after the domain read-model campaign
- live `products tags`, `products day-summaries`, and project/work-product dogfooding
- live archive debt output showing explicit orphaned content-block and
  attachment debt

## Execution Record

This program is now executed.

Implemented outcomes:

- canonical project/repo normalization now routes through one shared layer in
  `polylogue/lib/project_normalization.py`, and the durable profile/tag/day/week
  products consume it instead of late-stage ad hoc cleanup
- session-product lifecycle ownership is split across rebuild, refresh, status,
  batch, aggregate, storage, and thread modules instead of the old
  `session_product_support.py` umbrella
- archive-product operator surfaces now route through narrower workflow and
  rendering modules, and archive-debt products now expose explicit governance
  stage plus preview/apply lineage
- semantic surface declarations, schema generation/tooling/proof, and the
  remaining Claude/Drive parser seams are decomposed by role rather than
  historical accretion
- validation now has named lanes:
  - `semantic-product-normalization`
  - `semantic-product-live`
  - `semantic-product-hardening`

Live proof on `/home/sinity/.local/share/polylogue/polylogue.db`:

- `check --json --repair --target session_products` rebuilt `47,593` durable
  session-product rows and left `session_profiles`, `session_work_events`,
  `session_phases`, `day_session_summaries`, and `week_session_summaries` ready
- `products tags --json` no longer leaks regex fragments, markdown leftovers,
  or malformed repo identifiers into project breakdowns; the remaining
  underscore-prefixed names are real workspace projects
- `products debt --json` now exposes governance stages directly:
  `session_products=validated`,
  `orphaned_content_blocks=previewed`,
  `orphaned_attachments=previewed`
- the named lane `python -m devtools.run_validation_lanes --lane semantic-product-hardening`
  passed end to end

## One-Line Goal

Normalize Polylogue’s semantic/session product layer, narrow the remaining broad
toolchain/operator modules, and turn live archive cleanup and consumer-facing
semantics into explicit, governable, and stable platform contracts.

## Why This Is The Right Next Campaign

The previous campaign closed the banding and public-contract gap around durable
read models. The next structural drag is no longer “missing products”; it is
that the semantic/session product layer still contains noisy normalization,
broad orchestration modules, and several late-stage consumer reshaping paths.

Current hotspots and live signals:

- `polylogue/storage/session_product_support.py` — 821 LOC
- `polylogue/rendering/semantic_surface_declarations.py` — 749 LOC
- `polylogue/storage/session_product_status.py` — 397 LOC
- `polylogue/operations/archive.py` — 524 LOC
- `polylogue/cli/commands/products.py` — 569 LOC
- `polylogue/schemas/roundtrip_proof.py` — 547 LOC
- `polylogue/schemas/generation_analysis.py` — 556 LOC
- `polylogue/schemas/generation_workflow.py` — 480 LOC
- `polylogue/schemas/tooling_registry.py` — 485 LOC
- `polylogue/sources/parsers/claude.py` — 552 LOC
- `polylogue/sources/parsers/drive.py` — 441 LOC

The live archive also surfaced a semantic-quality problem that is more
important than file size alone: project and repo canonicalization is still
noisy enough that product outputs can emit regex fragments, markdown leftovers,
and malformed repo identifiers. That means the next program must combine
refactoring with normalization correctness rather than treating them as
independent concerns.

## Program Thesis

Polylogue should now converge around four explicit platform truths:

1. semantic/session products are canonical domain outputs, not late-stage
   formatting accidents
2. product normalization rules are shared and testable, not spread across
   attribution/profile/tag/day-summary surfaces
3. schema/proof/toolchain modules should be narrow and clearly separated
   between declaration, analysis, generation, and operator execution
4. live archive cleanup and external-consumer semantics should be governed by
   explicit lineage and stable payloads

## Architectural Rules

### 1. Semantic Products Must Have One Normalization Path

Project names, repo paths, session dates, work kinds, and action/project
evidence must not be normalized differently by profile builders, tag/day/week
rollups, analytics, and CLI formatting.

### 2. Product/Operator Modules Must Stop Being Mixed Control Planes

Broad modules such as `products.py`, `operations/archive.py`, and
`session_product_support.py` should be decomposed so domain logic, query logic,
status/rebuild logic, and rendering/output logic are owned separately.

### 3. Schema Tooling Must Be Banded By Role

Runtime-safe registry logic, declaration catalogs, generation analysis,
roundtrip proof, and synthetic generation should not keep meeting in broad
mixed files.

### 4. Live Cleanup Must Stay Governed

The live archive’s explicit debt should remain visible, lineage-backed, and
validation-covered. Cleanup automation must not silently mutate archive data.

### 5. Consumer Contracts Must Be Stable And Reusable

CLI, library, MCP, and downstream consumers such as Lynchpin should consume the
same stable product payloads and normalization semantics rather than
reconstructing them differently.

## Phase 1: Semantic Product Normalization

### Goal

Create a canonical normalization layer for project/repo/session semantic
products and route all relevant product builders through it.

### Main Work

- identify every path where project names, repo paths, session dates, and work
  labels are normalized today
- extract a canonical normalization layer for:
  - project/repo canonicalization
  - path-to-project attribution
  - session-date choice for product surfaces
  - work-event/project summary naming
- route profile products, tag rollups, day/week summaries, provider analytics,
  and archive-product outputs through that shared layer
- add targeted regression coverage for malformed/project-noise cases observed in
  the live archive

### Acceptance Criteria

- malformed project names no longer leak into durable product outputs
- repo/project naming is shared across profile, tag, summary, and analytics
  products
- live dogfooding on `products tags` / project stats produces visibly cleaner
  canonical outputs

## Phase 2: Session Product Lifecycle And Query Decomposition

### Goal

Break the remaining broad session-product support/status/lifecycle modules into
 focused, role-owned parts.

### Main Work

- decompose `session_product_support.py` into:
  - product row builders
  - rebuild orchestration
  - refresh/selection helpers
  - status/readiness support
- narrow `session_product_status.py` into:
  - readiness queries
  - stale/provenance checks
  - operator-facing status shaping
- keep lifecycle/state changes canonical and traceable through one explicit
  control plane

### Acceptance Criteria

- session-product support/status ownership is explainable by role
- rebuild/readiness logic no longer shares one broad mixed module
- maintenance and health surfaces consume the same lifecycle truth

## Phase 3: Archive Product Operator Surface Narrowing

### Goal

Reduce late-stage reshaping in archive-product operator surfaces and make public
archive-product APIs easier to consume directly.

### Main Work

- decompose `cli/commands/products.py` into request parsing, workflow, and
  rendering bands
- narrow `operations/archive.py` so product/library calls do less ad hoc
  shaping
- revisit `archive_products.py`, `facade.py`, `sync.py`, and MCP product tools
  so payload contracts are versioned once and routed consistently
- confirm external-consumer use cases can stay on durable products without
  private reconstruction

### Acceptance Criteria

- products CLI becomes a thin consumer of canonical product workflows
- archive operations stop mixing product construction, analytics shaping, and
  output glue
- library/sync/MCP surfaces align directly with stable archive-product
  contracts

## Phase 4: Semantic Proof And Surface Catalog Narrowing

### Goal

Reduce the remaining semantic proof/catalog broad files and clarify declaration
versus execution ownership.

### Main Work

- decompose `semantic_surface_declarations.py` into clearer surface families or
  declaration bands
- narrow the proof/runtime contract edge so declarations, evaluators, and proof
  orchestration stay separated
- ensure semantic surface declarations are shared by proof, QA, and any
  operator-facing contract inspection surfaces

### Acceptance Criteria

- semantic surface catalogs are declaration-only and easier to navigate
- proof execution no longer depends on broad declaration modules doing mixed
  work
- QA and proof surfaces compose from the same explicit contracts

## Phase 5: Schema Toolchain Narrowing

### Goal

Split the remaining schema tooling/generation/proof modules by role and tighten
their contracts.

### Main Work

- decompose `generation_analysis.py`, `generation_workflow.py`,
  `tooling_registry.py`, and `roundtrip_proof.py` into smaller role-owned
  modules
- keep runtime-safe registry logic separate from tooling-only analysis and
  generation
- align schema roundtrip proof, generation analysis, and operator workflows on
  shared typed contracts

### Acceptance Criteria

- schema tooling is navigable by role instead of by historical accretion
- runtime-vs-tooling boundaries stay explicit
- proof/generation/operator surfaces share stable typed contracts

## Phase 6: Source Parser Normalization And Fidelity Cleanup

### Goal

Use the recent archive dogfooding to tighten the broad remaining provider
parser seams, especially where semantic/session products depend on them.

### Main Work

- revisit `sources/parsers/claude.py` and `sources/parsers/drive.py`
- extract clearer helper bands for message normalization, attachment handling,
  timeline inference, and provider-specific metadata
- ensure parser output is shaped to feed the canonical semantic/product
  normalization layer rather than leaving cleanup to late-stage products

### Acceptance Criteria

- parser modules narrow materially
- attachment/timeline/provider-meta handling is easier to trace
- downstream semantic/product normalization needs fewer provider-specific hacks

## Phase 7: Live Cleanup Lineage And Governance

### Goal

Turn the remaining explicit archive debt into a governable, lineage-backed live
cleanup program without mixing it into ordinary repair paths.

### Main Work

- add explicit lineage and status for archive-debt cleanup categories
- separate “preview/apply/validated” governance for destructive debt targets
  from safe derived-model repair paths
- add named live lanes for:
  - cleanup preview integrity
  - post-apply validation on non-destructive targets
  - debt-status reporting stability

### Acceptance Criteria

- archive debt remains explicit and lineage-backed
- destructive cleanup is separated from safe repair conceptually and in code
- live governance lanes cover the cleanup control plane

## Phase 8: Validation, Memory, And Live Governance Hardening

### Goal

Keep this campaign tied to measurable validation, memory budgets, and live
archive proof instead of purely local refactors.

### Main Work

- add named lanes for semantic-product normalization, archive-product contract
  stability, schema-toolchain decomposition, and live cleanup governance
- keep memory-budget checks on the broad operator and maintenance workflows
- dogfood product/query/analytics surfaces against the live archive while the
  refactors land

### Acceptance Criteria

- the campaign has named local and live validation lanes
- memory governance stays explicit for retrieval and maintenance surfaces
- the live archive remains a first-class proof environment, not an afterthought

## Execution Order

1. semantic product normalization
2. session product lifecycle and query decomposition
3. archive product operator surface narrowing
4. semantic proof and surface catalog narrowing
5. schema toolchain narrowing
6. source parser normalization and fidelity cleanup
7. live cleanup lineage and governance
8. validation, memory, and live governance hardening

## Definition Of Done

This campaign is done when:

- semantic/session product outputs use one canonical normalization path
- the remaining broad session-product/operator/toolchain modules are decomposed
  by role
- live archive debt is governed with clearer cleanup lineage
- external consumer product contracts are stable across CLI, library, sync, and
  MCP
- named validation and live-governance lanes prove the refactor on the real
  archive
