[← Back to README](./README.md)

# Cleanup And Architectural Debt Retirement Program

Date: 2026-03-24
Status: planned focused cleanup program
Role: cleanup-only architectural debt retirement plan adjacent to the live execution queue

Relationship to the live queue:

- complements rather than replaces `consumer-contracts-and-governed-live-cleanup-program-2026-03-24.md`
- narrows scope to internal cleanup, ownership reduction, and debt deletion
- intentionally excludes new semantic enrichment, new consumer feature families,
  and product-surface widening except where required to remove debt

Absorbs cleanup residue and extrapolates from:

- `consumer-contracts-and-governed-live-cleanup-program-2026-03-24.md`
- `semantic-product-normalization-and-toolchain-convergence-program-2026-03-24.md`
- `domain-read-model-and-live-archive-stewardship-program-2026-03-24.md`
- `runtime-substrate-decomposition-and-contract-hardening-program-2026-03-24.md`
- `platform-wide-architecture-and-refactoring-program-2026-03-23.md`
- `core-architecture-convergence-program-2026-03-23.md`
- `refactoring-first-streamlining-program-2026-03-19.md`
- `.claude/scratch/027-architecture-review-2026-03-23.md`
- `.claude/scratch/019-polylogue-architecture-audit-2026-03-20.md`

Primary current evidence:

- remaining broad mixed-role files observed on 2026-03-24, including:
  - `polylogue/storage/backends/schema_ddl.py` (`725`)
  - `polylogue/site/templates.py` (`671`)
  - `polylogue/rendering/semantic_surface_declarations.py` (`657`)
  - `polylogue/lib/semantic_facts.py` (`563`)
  - `polylogue/storage/backends/queries/conversations.py` (`558`)
  - `polylogue/lib/query_execution.py` (`552`)
  - `polylogue/storage/action_event_lifecycle.py` (`536`)
  - `polylogue/storage/backends/queries/raw.py` (`521`)
  - `polylogue/storage/backends/query_store.py` (`513`)
  - `polylogue/storage/search_providers/sqlite_vec.py` (`507`)
  - `polylogue/showcase/qa_report.py` (`491`)
  - `polylogue/schemas/code_detection.py` (`473`)
  - `polylogue/facade.py` (`470`)
  - `polylogue/storage/search.py` (`469`)
  - `polylogue/schemas/semantic_inference.py` (`458`)
  - `polylogue/cli/commands/schema.py` (`456`)
  - `polylogue/showcase/qa_runner.py` (`448`)
  - `polylogue/rendering/semantic_proof_facts.py` (`445`)
  - `polylogue/lib/raw_payload.py` (`442`)
  - `polylogue/schemas/runtime_registry.py` (`439`)
  - `polylogue/pipeline/services/validation.py` (`437`)
  - `polylogue/storage/backends/async_sqlite.py` (`432`)
  - `polylogue/storage/repository_archive_reads.py` (`426`)

## One-Line Goal

Delete remaining structural debt and broad mixed-role ownership so Polylogue is
composed of explicit, narrow subsystems rather than large files that still
coordinate several concerns at once.

## Non-Goals

This program is not for:

- new semantic inference features
- embeddings or LLM enrichment work
- widening external consumer capabilities beyond what cleanup requires
- adding more facades that only move code without deleting overlap
- compatibility shells left behind to preserve pre-cleanup shapes

## Cleanup Thesis

Polylogue has already completed several convergence waves. The remaining debt
is no longer primarily about missing capabilities. It is about residual broad
ownership:

1. files that still mix declaration, orchestration, storage, and rendering
2. modules that remain authoritative in more than one conceptual layer
3. durable read models whose builders, storage, and operator wiring still
   expose too much of each other
4. high-level surfaces that still know too much about lower-level shapes
5. cleanup/refactor waves that have narrowed bands but left broad roots behind

This program should therefore measure success by removal and narrowing, not by
feature count.

## Architectural Rules

### 1. Delete Overlap, Do Not Relabel It

If two modules still perform the same coordination or translation role, choose
one and remove the other. Do not keep twin paths as “thin” compatibility
ownership.

### 2. One Layer, One Authority

Each of these concerns should have exactly one obvious owner:

- query planning
- retrieval execution
- semantic evidence extraction
- inferred semantic products
- archive product storage
- operator rendering
- schema tooling
- maintenance lifecycle

### 3. Declarations Must Not Also Execute

Large declaration tables should not own execution logic, and execution modules
should not also be the long-term declaration registry.

### 4. Public Roots Must Be Small

`facade`, CLI command modules, and package `__init__` roots should compose
smaller subsystems and not carry substantial business logic.

### 5. Every Cleanup Slice Must Remove Something Real

Each slice should end with at least one of:

- deleted module
- deleted helper family
- deleted duplicated translation path
- deleted broad public root responsibility
- deleted repeated query/render/storage logic

## Debt Inventory

### A. Query And Retrieval Substrate Debt

Current drag:

- `lib/query_execution.py` still carries planning-adjacent execution,
  readiness-aware fallback, grouped behavior, and semantic reconciliation
- `storage/backends/query_store.py`, `storage/backends/queries/conversations.py`,
  `storage/search.py`, and `storage/search_providers/sqlite_vec.py` still span
  overlapping retrieval concerns
- grouped stats and query output still retain some route-local knowledge of
  profile/action/product shapes

Desired end state:

- one canonical query-plan executor
- one retrieval band for summary/list/FTS/vector/hybrid access
- grouped stats consuming the same retrieval contract rather than route-local
  special cases

### B. Semantic Runtime Debt

Current drag:

- `lib/semantic_facts.py` is still both evidence extraction and a broad
  semantic runtime utility root
- `session_profile`, `work_events`, `phases`, `decisions`, and action events
  still mix evidence-tier and inference-tier meaning inside one product family
- some downstream surfaces still treat inferred semantics as if they were
  equivalent to explicit evidence

Desired end state:

- one evidence-tier semantic band
- one inference-tier semantic band
- explicit ownership boundaries between raw evidence, inferred products, and
  aggregate rollups

### C. Storage And Lifecycle Debt

Current drag:

- `action_event_lifecycle.py`, `async_sqlite.py`, `schema_ddl.py`,
  `queries/raw.py`, and `repository_archive_reads.py` remain broad ownership
  nodes
- DDL, lifecycle rules, read queries, and repair behavior are not yet narrow
  enough across archive products, action events, and raw payload bands

Desired end state:

- DDL separated by concern family rather than one giant root
- lifecycle modules split into action events, session products, archive debt,
  and maintenance lineage with no cross-owned internals
- raw payload queries narrowed away from archive-product and retrieval logic

### D. Schema Toolchain Debt

Current drag:

- `rendering/semantic_surface_declarations.py`,
  `schemas/code_detection.py`, `schemas/semantic_inference.py`,
  `schemas/runtime_registry.py`, and `cli/commands/schema.py` still remain
  broad and partly cross-layer

Desired end state:

- schema/operator CLI as a thin command surface over typed workflows
- runtime schema authority separated from detection, generation, and semantic
  inference support
- proof/catalog declarations narrowed into reusable sub-bands

### E. Operator And Public-Surface Debt

Current drag:

- `facade.py` is still too broad
- some CLI command roots remain larger than the workflows they should wrap
- MCP and CLI still duplicate some query/product contract shaping
- `qa_report.py` and `qa_runner.py` still carry too much mixed reporting logic

Desired end state:

- facade banded by archive/query/products/maintenance or reduced in role
- command modules mostly declarative and workflow-owned
- MCP/CLI contract shaping reused from shared request/response helpers
- QA/report composition clearly split into execution, summarization, and file
  persistence bands

### F. Site, Showcase, And Rendering Debt

Current drag:

- `site/templates.py` is still a very broad rendering/declaration root
- semantic proof declarations/facts remain too large
- HTML/rendering support still contains broad mixed rendering utilities

Desired end state:

- template environment, page families, and large template data builders split
  by page concern
- semantic proof declaration/fact support narrowed further
- rendering support modules aligned by output family rather than broad helper
  accumulation

## Execution Phases

### Phase 1: Query And Retrieval Debt Retirement

Targets:

- `polylogue/lib/query_execution.py`
- `polylogue/storage/backends/query_store.py`
- `polylogue/storage/backends/queries/conversations.py`
- `polylogue/storage/search.py`
- `polylogue/storage/search_providers/sqlite_vec.py`
- related grouped stats/output helpers

Required outcomes:

- delete duplicated retrieval orchestration
- isolate semantic post-filtering from base retrieval planning
- split vector/hybrid/search-provider coordination away from archive summary
  querying
- ensure action/profile/product stats consume the same narrowed retrieval band

Acceptance:

- no broad mixed query-execution root remains above about one orchestration
  responsibility
- route-specific retrieval branches shrink materially or disappear

### Phase 2: Semantic Runtime Boundary Cleanup

Targets:

- `polylogue/lib/semantic_facts.py`
- `polylogue/lib/session_profile.py`
- `polylogue/lib/work_events.py`
- `polylogue/lib/phases.py`
- `polylogue/lib/decisions.py`
- `polylogue/lib/action_events.py`

Required outcomes:

- separate evidence extraction from inferred semantic product construction
- stop forcing one module family to own both factual and heuristic semantics
- narrow auto-tag, engaged-time, and work-kind derivation into explicit
  inference bands

Acceptance:

- evidence-tier and inference-tier semantics are visibly separate in module
  ownership
- no downstream consumer needs to import a broad semantic omnibus to use one
  narrow semantic contract

### Phase 3: Storage, DDL, And Lifecycle Narrowing

Targets:

- `polylogue/storage/backends/schema_ddl.py`
- `polylogue/storage/backends/async_sqlite.py`
- `polylogue/storage/action_event_lifecycle.py`
- `polylogue/storage/backends/queries/raw.py`
- `polylogue/storage/repository_archive_reads.py`

Required outcomes:

- band DDL by concern family
- remove mixed archive-read, raw-read, and lifecycle ownership
- narrow async backend to true backend responsibilities
- delete duplicated repair/readiness computations that still live in several
  layers

Acceptance:

- DDL, lifecycle, and query bands are split by domain concern
- backend modules no longer expose broad read orchestration that belongs in
  repository/query layers

### Phase 4: Schema Toolchain And Operator Debt Retirement

Targets:

- `polylogue/rendering/semantic_surface_declarations.py`
- `polylogue/schemas/code_detection.py`
- `polylogue/schemas/semantic_inference.py`
- `polylogue/schemas/runtime_registry.py`
- `polylogue/cli/commands/schema.py`

Required outcomes:

- separate declarations from evaluators
- separate runtime authority from tooling-side support
- narrow schema command surfaces onto typed workflows and renderers

Acceptance:

- no broad schema/tooling file remains that simultaneously defines registry
  state, inference support, and operator workflows

### Phase 5: Operator Root And Reporting Cleanup

Targets:

- `polylogue/facade.py`
- broad CLI command roots
- MCP request/response contract helpers
- `polylogue/showcase/qa_report.py`
- `polylogue/showcase/qa_runner.py`

Required outcomes:

- reduce facade to a small public composition root or split it by domain
- delete duplicated shaping code across CLI, sync, MCP, and QA/report surfaces
- narrow QA execution vs reporting vs persistence

Acceptance:

- public roots become visibly smaller and thinner
- contract shaping is shared, not repeated at each boundary

### Phase 6: Site, Rendering, And Showcase Topology Cleanup

Targets:

- `polylogue/site/templates.py`
- `polylogue/rendering/semantic_proof_facts.py`
- `polylogue/rendering/renderers/html.py`
- remaining broad rendering/showcase helpers

Required outcomes:

- split template-declaration and template-data building concerns
- narrow proof fact extraction helpers further
- align output-family renderers around clearly scoped modules

Acceptance:

- no single site/rendering/support file remains the obvious catch-all for an
  entire output family

### Phase 7: Root Narrowing, Deletion Pass, And Final Debt Audit

Targets:

- package `__init__` roots
- residual helper/compatibility modules
- dead translation helpers
- leftover cleanup-only shims created by earlier waves

Required outcomes:

- delete dead exports and dead wrappers
- document the new module topology once
- rerun a code-outward audit to verify the cleanup wave actually removed debt

Acceptance:

- the final delta includes real file deletion and public-root narrowing
- the cleanup audit shows fewer broad ownership nodes than before the program

## Validation And Closure

This program is complete only when all of the following are true:

- `ruff check` passes for all touched Python files
- targeted pytest slices cover each cleanup phase
- named validation lanes are added for the heaviest runtime/control-plane
  cleanup slices
- a fresh live memory-budget run confirms no cleanup regression on query/check
  heavy paths
- `git diff --check` passes
- the final execution record can point to deleted overlap, not only moved code

## Recommended Execution Order

1. query and retrieval debt retirement
2. semantic runtime boundary cleanup
3. storage, DDL, and lifecycle narrowing
4. schema toolchain and operator debt retirement
5. operator root and reporting cleanup
6. site, rendering, and showcase topology cleanup
7. root narrowing, deletion pass, and final debt audit

## Done-State

The program should leave Polylogue in a state where:

- the remaining big files are mostly declarative tables or inherently dense
  schema/HTML assets, not mixed-role orchestration roots
- inferred semantics are clearly separable from explicit evidence
- public roots are small enough that the real owners are obvious
- later feature work does not have to fight cleanup debt before it can land
