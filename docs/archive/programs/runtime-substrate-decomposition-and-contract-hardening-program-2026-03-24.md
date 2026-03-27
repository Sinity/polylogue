# Runtime Substrate Decomposition And Contract Hardening Program

Date: 2026-03-24
Status: executed implementation program
Role: canonical execution record for the runtime-substrate decomposition and contract-hardening campaign

Absorbs and extends as the live queue:

- `archive-data-products-and-live-governance-program-2026-03-24.md`
- `platform-wide-architecture-and-refactoring-program-2026-03-23.md`
- `archive-intelligence-platform-convergence-program-2026-03-23.md`
- `source-boundary-and-runtime-governance-program-2026-03-23.md`
- the still-relevant refactoring/governance reservoir from:
  - `refactoring-first-streamlining-program-2026-03-19.md`
  - `canonical-archive-platform-program-2026-03-19.md`
  - `testing-reliability-expansion-program-2026-03-14.md`

Prerequisite executed programs:

- `archive-data-products-and-live-governance-program-2026-03-24.md`
- `source-boundary-and-runtime-governance-program-2026-03-23.md`
- `platform-wide-architecture-and-refactoring-program-2026-03-23.md`
- `archive-intelligence-platform-convergence-program-2026-03-23.md`
- `semantic-stack-convergence-program-2026-03-23.md`
- `schema-and-evidence-pipeline-convergence-program-2026-03-23.md`
- `core-architecture-convergence-program-2026-03-23.md`

Primary design inputs:

- `../planning-and-analysis-map-2026-03-21.md`
- `../../.claude/scratch/027-architecture-review-2026-03-23.md`
- current code-outward hotspot scan from the 2026-03-24 closure pass
- live archive and validation-lane dogfooding from the archive-data-products wave

## One-Line Goal

Turn Polylogue’s remaining broad runtime clusters into smaller, explicit
subsystems with harder public contracts, narrower module ownership, and
validation that measures the real operational surfaces rather than only their
internals.

## Execution Outcome

This campaign is now executed.

Implemented outcomes:

- query planning, retrieval, semantic reconciliation, grouping, and output
  policy are now split across explicit runtime modules:
  - `lib/query_support.py`
  - `lib/query_runtime.py`
  - `lib/query_retrieval.py`
  - `lib/query_sorting.py`
  - `cli/query_grouped_stats.py`
  - `cli/query_list_output.py`
  - `cli/query_profile_stats.py`
  - `cli/query_semantic_slice.py`
  - `cli/query_semantic_stats.py`
  - `cli/query_sql_stats.py`
  - `cli/query_stats_structured.py`
- session-product lifecycle is no longer one mixed engine; status, rebuild,
  refresh/delete upkeep, and support helpers are isolated in:
  - `storage/session_product_status.py`
  - `storage/session_product_rebuild.py`
  - `storage/session_product_refresh.py`
  - `storage/session_product_support.py`
- maintenance is now a smaller governed control plane instead of one omnibus
  module:
  - `storage/repair_support.py`
  - `storage/repair_cleanup.py`
  - `storage/repair_derived.py`
  - `storage/repair_control.py`
- row-model ownership is narrower through `storage/store_core.py` and
  `storage/store_products.py`, while the public store root stays canonical
- semantic-proof registry topology is now split into explicit declaration,
  model, and evaluation layers:
  - `rendering/semantic_surface_declarations.py`
  - `rendering/semantic_surface_models.py`
  - `rendering/semantic_surface_evaluation.py`
- backend/schema ownership is materially narrower:
  - `backends/async_sqlite_archive.py`
  - `backends/async_sqlite_derived.py`
  - `backends/async_sqlite_raw.py`
  - `backends/schema_ddl.py`
  - `backends/schema_upgrade.py`
- MCP tool wiring no longer lives in one broad registration module:
  - `mcp/server_query_tools.py`
  - `mcp/server_mutation_tools.py`
  - `mcp/server_read_tools.py`
  - `mcp/server_maintenance_tools.py`
  - `mcp/server_product_tools.py`
- named validation lanes now cover the campaign directly:
  - `runtime-substrate-contracts`
  - `runtime-substrate-live`
  - `runtime-substrate-hardening`
- the live archive was exercised through those new runtime-substrate lanes:
  - bounded retrieval RSS stayed within budget
  - bounded maintenance preview RSS stayed within budget
  - no new `earlyoom` kills occurred during the closure pass
  - the real archive still reports the intentional live debt of `15,781`
    orphaned content blocks, now as explicit governed cleanup debt rather than
    silent drift

## Why This Is The Right Next Campaign

The archive-data-products wave closed a real product/governance gap, but it also
made the next structural drag obvious. Polylogue is now functionally stronger
than its remaining module topology.

Current hotspot scan:

- `polylogue/storage/session_product_lifecycle.py` — 1366 LOC
- `polylogue/cli/query_summary_output.py` — 1048 LOC
- `polylogue/lib/query_execution.py` — 982 LOC
- `polylogue/rendering/semantic_surface_registry.py` — 888 LOC
- `polylogue/storage/repair.py` — 885 LOC
- `polylogue/storage/backends/async_sqlite.py` — 843 LOC
- `polylogue/storage/backends/schema.py` — 722 LOC
- `polylogue/lib/models.py` — 714 LOC
- `polylogue/mcp/server_tools.py` — 667 LOC
- `polylogue/storage/repository_reads.py` — 642 LOC
- `polylogue/storage/store.py` — 616 LOC
- `polylogue/storage/backends/queries/session_products.py` — 597 LOC

Those are no longer random “big files.” They are the remaining places where
Polylogue still mixes too many responsibilities in one runtime boundary:

1. query execution, grouping, and output policy still cross too many layers
2. session-product lifecycle, status, rebuild, and refresh all still live in
   one huge module
3. maintenance/repair is now more truthful, but it is still one large control
   plane rather than discrete governed subsystems
4. backend/schema/store/query modules still carry multiple domain bands at once
5. operator and external-consumer contracts are better, but still spread across
   broad CLI/MCP/facade surfaces

## Program Thesis

The next architectural win is not another feature wave. It is substrate
decomposition with contract hardening:

1. smaller runtime modules with clearer ownership
2. fewer mixed read/write/status/rebuild files
3. public surfaces that compose explicit workflows instead of exposing their
   internal wiring
4. validation lanes that track memory, rebuild cost, and live correctness for
   those new boundaries

## Architectural Rules

### 1. Split By Responsibility, Not By File Size Alone

A large module should only be decomposed when the new boundaries correspond to
real semantic seams: planning vs execution, read models vs maintenance, schema
definition vs migration, rendering vs aggregation, and so on.

### 2. One Canonical Contract Per Surface

If a concept is public, it gets one canonical request/result contract consumed
across CLI, MCP, facade, and sync surfaces. No parallel shape drift.

### 3. Status And Repair Must Share The Same Truth

Readiness, drift detection, rebuild logic, and operator rendering must all use
the same lifecycle state rather than recomputing “almost the same” counts in
different places.

### 4. Validation Must Track Steady State, Not Just Happy Path

Every decomposed subsystem needs at least one validation lane that exercises the
steady-state operator surface it claims to improve.

### 5. Refactoring Must Remove Old Mixed Ownership

The goal is not to add helper modules and keep the old omnibus as a facade
forever. Once a new boundary is real, the old mixed-role ownership should be
reduced aggressively.

## Phase 1: Query Engine And Output Substrate Decomposition

### Goal

Reduce the remaining query monoliths into explicit plan, retrieval, grouping,
and rendering layers.

### Main Work

- split `lib/query_execution.py` into:
  - candidate selection / pushdown planning
  - runtime semantic reconciliation
  - grouped-stats assembly
  - retrieval-lane orchestration
- split `cli/query_summary_output.py` into:
  - grouped stats aggregators
  - machine-output serializers
  - terminal renderers
- narrow `operations/archive.py`, `facade.py`, and sync wrappers around those
  canonical query contracts
- keep `--action`, `--tool`, `--path`, product-aware stats, and retrieval-lane
  behavior unified while removing cross-layer duplication

### Acceptance Criteria

- query execution logic no longer lives in one 900+ LOC module
- grouped stats and retrieval surfaces stop re-deriving semantics across CLI
  layers
- live and test queries produce the same results before and after decomposition

## Phase 2: Session Product Lifecycle And Governance Decomposition

### Goal

Break `session_product_lifecycle.py` into durable status/rebuild/refresh/
aggregate ownership instead of one all-purpose engine.

### Main Work

- split lifecycle logic into at least:
  - status and drift calculation
  - full rebuild orchestration
  - per-conversation refresh/delete upkeep
  - aggregate tag/day/week rebuild support
- narrow `session_product_rows.py` and `session_products.py` around clearer row
  ownership
- keep product status, repair, delete-time cleanup, and live rebuild semantics
  on one canonical path

### Acceptance Criteria

- session-product lifecycle stops being one 1300+ LOC catch-all
- product status and repair logic reuse shared lifecycle helpers instead of
  reassembling counts
- targeted refresh/delete flows remain correct under tests and live archive use

## Phase 3: Maintenance And Cleanup Control-Plane Decomposition

### Goal

Turn repair/cleanup into a smaller, typed control plane instead of a single
large `repair.py` omnibus.

### Main Work

- split safe repair, destructive cleanup, preview/build-manifest logic, and
  maintenance lineage recording into focused modules
- make maintenance target descriptors explicit rather than stringly glued
- keep `check --repair/--cleanup --preview` and maintenance-run persistence on
  one request/result workflow
- make the live orphaned-content-block debt and similar archive cleanup targets
  easier to reason about in code and validation

### Acceptance Criteria

- `repair.py` stops being the only place where maintenance semantics live
- preview/apply/report flows remain machine-readable and consistent
- destructive and derived-only paths are easier to validate separately

## Phase 4: Storage Backend, Schema, And Row-Model Narrowing

### Goal

Reduce mixed ownership in the storage substrate.

### Main Work

- split `backends/async_sqlite.py` by actual domain bands:
  - conversation/message writes
  - derived/product writes
  - raw/provenance writes
  - maintenance/meta helpers
- split `backends/schema.py` into clearer schema bands:
  - archive core tables
  - derived/action/product tables
  - indexes/FTS/triggers
  - maintenance/provenance tables
- narrow `storage/store.py` so row models are grouped more honestly by domain
- reduce the remaining breadth of `repository_reads.py` and product query modules

### Acceptance Criteria

- backend and schema ownership are more inspectable by subsystem
- row-model declarations no longer pile unrelated archive domains into one file
- validation still covers migrations/bootstrapping/read paths cleanly

## Phase 5: Operator, MCP, And External Contract Convergence

### Goal

Reduce the remaining breadth in operator-facing command surfaces and MCP tool
wiring while hardening public contracts.

### Main Work

- narrow `mcp/server_tools.py` around shared query/product/support helpers
- narrow `cli/check_workflow.py`, `cli/commands/check.py`, and `cli/click_app.py`
  around typed workflows instead of option/plumbing sprawl
- keep archive product, maintenance, and query contracts aligned across CLI,
  MCP, facade, and sync surfaces
- reduce public export breadth where it still exceeds the true stable contract

### Acceptance Criteria

- CLI and MCP surfaces use fewer hand-assembled request/result shapes
- public archive/product contracts are easier to trace end to end
- external consumers need less internal code archaeology

## Phase 6: Semantic And Proof Surface Topology Narrowing

### Goal

Reduce the remaining large semantic and proof modules now that products and
query surfaces are stronger.

### Main Work

- split `rendering/semantic_surface_registry.py` around:
  - surface declarations
  - contract evaluation
  - alias/group expansion
- revisit `lib/semantic_facts.py` and adjacent semantic product helpers for
  remaining mixed responsibilities
- keep proof, products, and query surfaces consuming the same semantic facts
  contracts

### Acceptance Criteria

- semantic proof topology becomes easier to navigate and evolve
- surface declarations and execution logic no longer live in one mixed file

## Phase 7: Validation, Memory, And Live Governance Hardening

### Goal

Extend the validation story from “functions pass” to “subsystems stay healthy
under real operator use.”

### Main Work

- add named lanes for:
  - query/grouped-output memory budgets
  - session-product rebuild budgets
  - maintenance preview/apply/report workflows
  - MCP/archive-product contract surfaces
- dogfood those lanes on the live archive where safe
- keep `earlyoom` sensitivity and live RSS ceilings part of the regression story

### Acceptance Criteria

- major decomposed subsystems have named validation lanes
- memory regressions become reproducible instead of anecdotal
- live archive operator paths stay part of the architectural closure loop

## Execution Order

1. query engine and output substrate decomposition
2. session product lifecycle and governance decomposition
3. maintenance and cleanup control-plane decomposition
4. storage backend, schema, and row-model narrowing
5. operator, MCP, and external contract convergence
6. semantic and proof surface topology narrowing
7. validation, memory, and live governance hardening

## First Concrete Slice

Start with the highest-leverage structural pair:

- split `query_execution.py` and `query_summary_output.py` into smaller plan /
  retrieval / grouping / rendering ownership
- split `session_product_lifecycle.py` so status, rebuild, refresh, and
  aggregate upkeep stop sharing one monolithic module

That opens the path for the rest of the storage and operator narrowing work
without adding another temporary facade wave.
