# Platform-Wide Architecture And Refactoring Program

Date: 2026-03-23
Status: active execution program
Role: canonical repository-wide architecture and refactoring queue after archive-intelligence convergence

Absorbs and supersedes as the live queue:

- `archive-intelligence-platform-convergence-program-2026-03-23.md`
- the still-open simplification reservoir portions of:
  - `canonical-archive-platform-program-2026-03-19.md`
  - `refactoring-first-streamlining-program-2026-03-19.md`
  - `testing-reliability-expansion-program-2026-03-14.md`

Prerequisite executed programs:

- `archive-intelligence-platform-convergence-program-2026-03-23.md`
- `semantic-stack-convergence-program-2026-03-23.md`
- `schema-and-evidence-pipeline-convergence-program-2026-03-23.md`
- `core-architecture-convergence-program-2026-03-23.md`
- `runtime-contract-and-validation-lanes-program-2026-03-22.md`

Primary design/audit inputs:

- `../planning-and-analysis-map-2026-03-21.md`
- `../../.claude/scratch/027-architecture-review-2026-03-23.md`
- `canonical-archive-platform-program-2026-03-19.md`
- `refactoring-first-streamlining-program-2026-03-19.md`

## One-Line Goal

Reduce Polylogue’s remaining architectural drag by collapsing wrapper strata,
quarantining tooling from runtime, narrowing public surfaces, and turning the
remaining over-wide subsystems into smaller one-way modules with stronger
verification.

## Why This Is The Right Next Campaign

The archive-intelligence wave closed the sharpest semantic and retrieval gaps:

- canonical action events now exist
- query truth is action-aware
- retrieval health is explicit
- named validation and memory-budget lanes exist

That means the remaining drag is less about missing capabilities and more about
codebase shape:

1. fluent query/filter shells still wrap the canonical execution engine more
   than they should
2. storage state still spans backend, repository, indexing, lifecycle, and
   prepare/write surfaces with more intermediate seams than necessary
3. CLI, MCP, showcase, publication, and report layers now share semantics, but
   not enough shared workflow/result infrastructure
4. schema tooling and synthetic/toolchain mass are improved but still wider
   than runtime-facing code should be
5. the public package/module topology is still broader than the actual stable
   API story
6. performance and memory budgets now exist, but they should actively govern
   refactors rather than trail them

## Program Thesis

Polylogue should now optimize for:

1. fewer public roots
2. smaller internal modules with one-way dependencies
3. typed workflows instead of wrapper stacks
4. runtime/tooling separation that is visible in imports and file layout
5. validation lanes that prove both correctness and resource discipline

## Architectural Rules

### 1. Keep Runtime First-Class And Tooling Quarantined

Runtime archive ingestion/query/render paths must not depend “upward” on schema
generation, synthetic corpora, diffing, mutation utilities, or report assembly.

### 2. Prefer One Canonical Workflow Per Operator Outcome

If CLI, MCP, QA, showcase, and publication all expose the same underlying
operation, they should share one typed workflow/result surface rather than
parallel local orchestration.

### 3. Thin Adapters May Exist, But Only At Real Boundaries

Boundary adapters are acceptable when they bind:

- Click/TUI/MCP/publication to workflows
- storage infra to typed repository operations
- runtime semantics to higher-level products

They are not acceptable as inert forwarding layers left behind after a split.

### 4. Refactors Must Delete Old Parallel Paths

No compatibility shells, no “legacy” aliases, no leaving both the old and new
shape active after the same change.

### 5. Every Boundary Reduction Must Carry Verification

Each phase should leave behind:

- direct unit coverage of the new canonical seam
- one operator-level or integration-level regression proving the behavior still
  works end to end
- where relevant, a memory or runtime-budget check

## Phase 1: Query And Filter Shell Convergence

### Goal

Finish collapsing the remaining wrapper mass around canonical query execution.

### Current Problems

- `ConversationFilter` still exists as a fluent shell around the real immutable
  execution plan
- summary/list/count behavior still has multiple convenience surfaces
- query/output routing still carries some planning knowledge outside the plan
  engine

### Main Work

- reduce `polylogue/lib/filters.py` to a strict adapter or remove it entirely
  where practical
- move remaining query-route branching onto typed query request/plan objects
- shrink special-case query-output code that reinterprets plan semantics
- make SQL-pushdown/readiness choices wholly plan-driven and testable

### Acceptance Criteria

- one canonical query execution language drives list, count, stats, and stream
- `ConversationFilter` no longer carries meaningful independent behavior
- query-output modules format results; they do not decide semantic truth

## Phase 2: Storage State-Band And Lifecycle Convergence

### Goal

Turn storage into a smaller, more deliberate stack:

- backend infra
- repository workflow
- typed state/read models
- explicit lifecycle jobs

### Current Problems

- raw/validate/parse/prepare/write/index state still spans many modules
- lifecycle jobs like FTS/action-event repair are present but not yet expressed
  as one coherent state-band story
- backend/query-store/repository responsibilities are improved but still broad

### Main Work

- continue shrinking `SQLiteBackend` toward connection/transaction/write
  primitives only
- move remaining status/lifecycle/business semantics outward from backend code
- unify state-view models for raw/validation/parse/prepare/index/publication
- simplify repository read/write mixins where they still mirror lower layers

### Acceptance Criteria

- backend code is infrastructural, not semantic
- repository owns archive workflow semantics
- lifecycle repair/rebuild jobs are explicit, typed, and reused

## Phase 3: Operator Workflow And Rendering Convergence

### Goal

Make operator-facing surfaces share typed workflow/result contracts instead of
parallel orchestration.

### Current Problems

- CLI, MCP, showcase, publication, and some report surfaces still project the
  same operations through partly separate code paths
- result serialization and human rendering are closer than before, but not yet
  uniformly modeled

### Main Work

- define shared workflow/result models for major operator outcomes:
  - query
  - check/proof
  - publication
  - QA/showcase runs
- reduce per-surface projection logic
- continue narrowing output/render modules to formatting only

### Acceptance Criteria

- the same operator outcome has one canonical typed result
- CLI/MCP/showcase/publication are adapters, not independent controllers
- machine output and human output derive from the same workflow result

## Phase 4: Schema Toolchain And Synthetic Stack Isolation

### Goal

Further split runtime schema authority from tooling/generation/synthetic mass.

### Current Problems

- runtime registry separation landed, but tooling/operator/generation stacks are
  still large
- synthetic corpus and generation flows remain a significant cluster

### Main Work

- further split schema tooling into:
  - runtime authority
  - operator workflow
  - generation/build
  - synthetic corpus
  - diff/audit/report helpers
- reduce direct cross-imports between runtime and tooling-only code
- tighten the operator surface so `operator_workflow` stays the single front
  door for schema operations

### Acceptance Criteria

- runtime schema code no longer reaches into tooling internals
- synthetic generation is quarantined from runtime
- schema operators share one typed workflow stack

## Phase 5: Semantic Product Surface Consolidation

### Goal

Turn the higher-level semantic products into smaller consumers of shared facts
instead of overlapping product-local inference.

### Current Problems

- session profiles, work events, summaries, threads, tagging, and attribution
  are much better aligned, but still form a broad cluster
- the action-event and semantic-facts layers now exist, but some higher-level
  products still re-compose them locally

### Main Work

- identify the minimal canonical semantic product surfaces
- move remaining product-specific inference out of shared runtime modules
- define explicit derived-model builders for session/work analytics

### Acceptance Criteria

- shared semantic runtime stays small
- product-specific analytics are explicit consumers, not hidden extensions
- the archive’s semantic API becomes easier to query and reason about

## Phase 6: Package Root And Module Topology Narrowing

### Goal

Finish reducing Polylogue’s public/import surface to what is actually stable and
intended.

### Current Problems

- package roots and some module groupings still expose more than they should
- historical convenience imports can still blur the true architecture

### Main Work

- narrow `polylogue/__init__.py`, `polylogue/lib/__init__.py`, and similar
  package roots further where appropriate
- remove leftover forwarding modules that no longer earn their keep
- document the intended import graph and enforce it in tests where useful

### Acceptance Criteria

- public roots reflect the real supported API
- internal-only modules stop masquerading as stable entry points
- import graph direction is easier to audit

## Phase 7: Validation, Memory, And Performance Governance

### Goal

Make the new architecture self-policing.

### Main Work

- extend validation lanes so each major refactor phase has a named proving lane
- add more explicit memory-budget checks for the known heavy query/report paths
- use live-archive dogfooding to validate real operator workloads after each
  major architectural cut
- keep pruning the specific patterns that previously triggered `earlyoom`

### Acceptance Criteria

- every major refactor phase ends with a named validation slice
- memory regressions are caught by committed checks, not operator surprise
- live-archive dogfooding remains part of the design loop

## Execution Order

1. query and filter shell convergence
2. storage state-band and lifecycle convergence
3. operator workflow and rendering convergence
4. schema toolchain and synthetic stack isolation
5. semantic product surface consolidation
6. package root and module topology narrowing
7. validation, memory, and performance governance

## First Concrete Slice

Start with the query/filter shell and storage state-band boundary together:

- finish reducing `ConversationFilter`
- tighten plan-driven summary/list/count routing
- further narrow backend/repository/lifecycle seams around state views and
  maintenance jobs

That first slice should produce the next concrete code change wave.
