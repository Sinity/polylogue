# Polylogue Core Architecture Convergence Program

Date: 2026-03-23
Status: planned execution program
Role: canonical next implementation campaign for architectural streamlining after the schema/proof/publication/runtime closure wave

See also:

- `refactoring-first-streamlining-program-2026-03-19.md`
- `canonical-archive-platform-program-2026-03-19.md`
- `runtime-contract-and-validation-lanes-program-2026-03-22.md`
- `schema-package-authority-program-2026-03-22.md`
- `../planning-and-analysis-map-2026-03-21.md`
- `.claude/scratch/027-architecture-review-2026-03-23.md`

## One-Line Goal

Make Polylogue feel like one system again by collapsing the remaining parallel
truth surfaces for query execution, storage access, operator routing, showcase
verification, schema runtime authority, and public API shape.

## Why This Is Now The Main Frontier

The previous wave closed the major product-level correctness lanes:

- artifact proof
- package-aware schema authority
- semantic proof
- publication control plane
- site/repo-shape cleanup
- machine/runtime validation lanes

What remains is no longer missing capability. It is architectural drag inside
the main code:

1. immutable query intent still compiles back into mutable filter execution
2. backend, query-store, and repository still express too much of the same read
   surface
3. query-first CLI semantics still live inside Click argument rewriting
4. showcase verification is still too embedded in Python code and still forms
   import cycles with CLI modules
5. schema runtime resolution and schema-generation/promotion still live in the
   same mutual-dependency cluster
6. package-root exports still oversell what belongs in the archive-core API

This campaign is the next one because the repo is now functionally rich enough
that the best win is not another feature lane. It is convergence.

## Program Thesis

Polylogue should have:

1. one canonical query request model
2. one canonical query execution engine
3. one canonical low-level storage read surface
4. one canonical operator request/front-door model
5. one verification runner that treats CLI/MCP/site as external surfaces rather
   than import-time neighbors
6. one schema runtime authority layer beneath one schema tooling layer
7. one public API that clearly distinguishes archive core from higher-order
   semantic analysis

## Non-Goals

This campaign is not:

- a provider-parser rewrite
- a schema inference redesign
- a feature-cutting exercise
- a new proof/publication lane
- a mass test-suite reshuffle for its own sake

Those may follow. This program is specifically about collapsing remaining
structural duplication and over-wide boundaries in main code.

## Architectural Rules

### 1. Prefer Deletion Through Convergence

Do not add a new layer unless it immediately deletes or empties an old one.

### 2. Specs And Plans Must Execute Directly

Do not keep `typed spec -> mutable builder -> executor` as the main path.
Typed intent should feed canonical execution directly.

### 3. Low-Level Read Surfaces Must Be Singular

If `SQLiteQueryStore` is the query authority, wrapper layers should narrow
around it, not restate its whole interface.

### 4. Verification Must Cross Real Boundaries

Showcase and QA should continue black-boxing operator surfaces, but they should
not depend on broad import cycles with those same surfaces.

### 5. Schema Runtime And Schema Tooling Must Be One-Way

Runtime consumers may depend on runtime schema authority.
Generation/promotion tooling may depend on runtime schema authority.
Runtime resolution must not depend back on promotion/generation machinery.

### 6. Public Exports Must Match Product Shape

The package root should primarily expose archive-core access. Higher-order
semantic-analysis helpers should be available, but not presented as the same
kind of API.

## Execution Order

The order matters. Earlier steps are chosen to shrink the dependency fan-out
for later ones.

1. query execution convergence
2. storage surface convergence
3. operator front-door normalization
4. showcase/QA data-model and runner convergence
5. schema runtime/tooling split
6. public API narrowing

Cross-cutting sync/async cleanup runs alongside Steps 1 through 4.

## Step 1: Query Execution Convergence

### Goal

Replace the current split between:

- `ConversationQuerySpec`
- mutable `ConversationFilter`
- `filter_executor`
- `filter_runtime`

with one canonical query-execution path.

### Current Problems

- `ConversationQuerySpec.build_filter()` recompiles immutable request intent
  into mutable builder state.
- `filter_executor.py` inspects `ConversationFilter` internals directly.
- `ConversationFilter` still combines user-facing fluent ergonomics with
  canonical execution truth.
- CLI, MCP, and library surfaces therefore converge only halfway.

### Target Shape

Introduce a canonical internal query plan or request object that directly owns:

- selection constraints
- pushdown eligibility
- content-loading requirements
- fetch sizing
- route preferences

Then make execution operate on that plan directly.

`ConversationFilter` should become one of two things:

1. a thin fluent builder that only produces the canonical plan/request
2. or a compatibility-free replacement removed in favor of plan/spec-driven
   fluent helpers on the archive facade

The preferred direction is still to keep a fluent archive-facing API, but it
must stop being the executor’s storage of record.

### Main Modules

- `polylogue/lib/query_spec.py`
- `polylogue/lib/filter_executor.py`
- `polylogue/lib/filter_runtime.py`
- `polylogue/lib/filters.py`
- `polylogue/storage/repository_reads.py`
- `polylogue/cli/query_plan.py`
- `polylogue/operations/archive.py`
- `polylogue/mcp/server_tools.py`

### Required Refactors

1. define one internal execution-plan model for conversation queries
2. move SQL-pushdown computation onto that model
3. make repository read methods accept canonical query-plan objects or one
   lower-level read-query request shape
4. keep `ConversationQuerySpec` as the canonical external typed request surface
5. remove direct executor dependence on `ConversationFilter` private fields
6. make `ArchiveOperations.query_conversations()` execute through the new path
7. make MCP list/search tools use the same path with no separate semantic route

### Acceptance Criteria

- `ConversationQuerySpec` no longer compiles to mutable builder state as the
  canonical execution path
- `filter_executor.py` is either deleted or reduced to a thin adapter layer
- repository list/count/summary execution can run from a typed plan/request
- CLI and MCP query semantics share one execution engine
- summary-vs-full-content selection remains explicit and testable

### Verification

- all query exec/unit/property tests
- MCP query tool tests
- query-route integration tests
- semantic proof read-surface suite

## Step 2: Storage Surface Convergence

### Goal

Collapse the broad overlap between:

- `SQLiteBackend`
- `SQLiteQueryStore`
- `ConversationRepository`

without reintroducing hidden ambient state.

### Current Problems

- `SQLiteQueryStore` is already the low-level read/query authority
- `SQLiteBackend` re-exposes much of that surface with similar signatures
- `ConversationRepository` then wraps again
- backend still mixes transaction/connection concerns with business operations
- there are still multiple “where should I read this from?” answers

### Target Shape

Three clear layers:

1. connection/transaction runtime
2. low-level read/query store
3. repository/service write and hydration layer

The backend should own:

- connection lifecycle
- transaction lifecycle
- pool/bulk helpers
- low-level write primitives only where necessary

The query store should own:

- read/query methods
- query batching
- aggregate/stat reads

The repository should own:

- hydration into domain models
- write orchestration
- archive-level persistence semantics
- optional vector/search helpers if still justified

### Main Modules

- `polylogue/storage/backends/async_sqlite.py`
- `polylogue/storage/backends/query_store.py`
- `polylogue/storage/repository.py`
- `polylogue/storage/repository_reads.py`
- `polylogue/storage/repository_writes.py`
- `polylogue/storage/repository_vectors.py`
- `polylogue/storage/backends/queries/*.py`

### Required Refactors

1. delete broad read/query wrappers from `SQLiteBackend` where `SQLiteQueryStore`
   is already canonical
2. make repository read mixins depend on `queries` directly, not parallel backend
   read methods
3. narrow backend to connection, transaction, schema-init, pool/bulk, and write
   concerns
4. decide whether repository vector helpers belong in repository or in a
   dedicated search service
5. replace archive summary paths that hydrate full conversations unnecessarily
   with aggregate/stat queries where possible
6. remove duplicated count/list/search signatures from wrapper layers

### Acceptance Criteria

- no broad duplicate read surface on backend and query-store simultaneously
- repository is the canonical hydrated/domain surface
- query-store is the canonical record/query surface
- backend reads as infrastructure rather than as a second repository
- `ArchiveOperations.summary_stats()` stops loading full conversations just to
  compute archive summary numbers

### Verification

- storage unit/integration suites
- scale and FTS tests
- archive stats and CLI summary tests
- benchmark spot-check for heavy list/stats paths

## Step 3: Operator Front-Door Normalization

### Goal

Move query-first/root CLI semantics out of Click argument rewriting and into a
typed front-door request model.

### Current Problems

- `click_app.py` still carries a huge root option surface
- `QueryFirstGroupBase.parse_args()` rewrites positional args into hidden
  `--query-term`
- query semantics are split across Click decorators, frontdoor rewrite rules,
  query-spec parsing, and query-route planning
- sync `asyncio.run()` entrypoints are still scattered across commands and
  helpers

### Target Shape

Introduce one typed root command request model covering:

- root query-mode invocation
- explicit subcommand invocation
- root stats/no-args invocation
- machine-output mode

Click should become a binder onto that model, not the place where query-mode
semantics are invented.

At the same time, collapse ad hoc `asyncio.run()` usage behind a deliberate
small sync-entry utility so operator surfaces share one sync/async handoff
policy.

### Main Modules

- `polylogue/cli/click_app.py`
- `polylogue/cli/query_frontdoor.py`
- `polylogue/cli/query.py`
- `polylogue/cli/helpers.py`
- `polylogue/cli/commands/run.py`
- `polylogue/cli/commands/generate.py`
- `polylogue/cli/commands/embed.py`
- `polylogue/cli/commands/tags.py`
- `polylogue/site/builder.py`
- `polylogue/sync.py`

### Required Refactors

1. define a typed root CLI request/front-door model
2. make Click bind to that model instead of hiding arg-rewrite semantics
3. narrow `click_app.py` to app construction and binding
4. keep machine-error behavior unchanged, but route it through the same root
   model
5. introduce one shared sync-entry helper and replace scattered `asyncio.run()`
   use in CLI/operator modules
6. keep query-route planning, but make it downstream of the typed request model

### Acceptance Criteria

- no hidden `--query-term` rewrite as the main query-mode trick
- root CLI behavior is explainable as typed request parsing plus route planning
- sync entry behavior is consistent across root CLI, commands, site build, and
  sync facade
- machine-contract tests remain green

### Verification

- click app tests
- machine contract tests
- query-mode integration tests
- CLI snapshot/help tests
- site CLI tests

## Step 4: Showcase And QA Convergence

### Goal

Keep showcase/QA as first-class verification, but make it less code-shaped and
less cyclic with the operator surface.

### Current Problems

- the exercise catalog is a large inline Python tuple
- the runner imports the CLI root directly
- showcase/QA/report modules still participate in real import cycles
- verification data definitions and verification execution are too tightly bound

### Target Shape

Split showcase into:

1. exercise catalog data
2. exercise runner
3. invariant registry
4. QA composition/projection

The exercise catalog should move out of inline Python code into a typed data
format that the runner loads and validates.

The runner should target the real entrypoint boundary:

- either the installed/module CLI entrypoint,
- or a very thin invocation adapter that mirrors that boundary exactly.

The point is not to stop black-box testing. The point is to stop the verifier
from living in the same import cycle as the thing it verifies.

### Main Modules

- `polylogue/showcase/exercises.py`
- `polylogue/showcase/runner.py`
- `polylogue/showcase/invariants.py`
- `polylogue/showcase/qa_runner.py`
- `polylogue/showcase/report.py`
- `polylogue/cli/commands/qa.py`
- `polylogue/cli/click_app.py`

### Required Refactors

1. define a typed serialized exercise-catalog format
2. move the static catalog out of inline Python constants
3. load and validate the catalog explicitly at runner startup
4. narrow showcase runner to exercise execution and validation
5. make runner target the real CLI entrypoint boundary without importing broad
   CLI internals unnecessarily
6. break the existing showcase/CLI import cycle
7. keep QA composition typed and separate from exercise catalog data

### Acceptance Criteria

- the exercise catalog is not an inline Python mega-tuple anymore
- the runner no longer depends on broad CLI/report import cycles
- showcase remains black-box and deterministic
- QA/report projections still produce the same durable artifacts

### Verification

- showcase unit tests
- QA/report tests
- CLI QA tests
- validation-lane runner tests

## Step 5: Schema Runtime/Tooling Split

### Goal

Separate schema runtime authority from schema tooling without weakening package
authority or proof/report surfaces.

### Current Problems

- `SchemaRegistry` still owns runtime resolution, package I/O, diffs, manifests,
  and promotion concepts
- `schema_generation.py` imports registry types and package types directly
- the schema dependency graph still contains a real cycle
- runtime consumers still live too close to cluster/promotion concerns

### Target Shape

Two explicit schema layers:

1. runtime schema authority
2. schema tooling and generation

Runtime layer owns:

- provider normalization for schema lookup
- package catalogs and package loading
- element schema loading
- payload-to-package resolution
- runtime-facing schema explanation types

Tooling layer owns:

- sampling
- clustering/profile-family evidence
- package assembly
- cluster manifests
- promotion workflows
- schema diff/report tooling

Tooling may read runtime package models. Runtime must not depend back on
tooling concerns.

### Main Modules

- `polylogue/schemas/registry.py`
- `polylogue/schemas/schema_generation.py`
- `polylogue/schemas/sampling.py`
- `polylogue/schemas/packages.py`
- `polylogue/schemas/verification.py`
- `polylogue/schemas/validator.py`
- `polylogue/cli/commands/schema.py`
- `polylogue/sources/emitter.py`
- `polylogue/pipeline/services/parsing.py`

### Required Refactors

1. extract runtime registry/package-resolution logic into a narrower runtime
   module or modules
2. move cluster/promotion/diff/manifests into tooling modules
3. keep package models shared, but one-way
4. remove the current schema import cycle
5. keep `resolve_payload()` semantics unchanged from the perspective of parser
   dispatch and validation
6. ensure proof/verification/reporting uses the runtime authority, not tooling
   internals

### Acceptance Criteria

- runtime schema resolution no longer depends on generation/promotion modules
- tooling depends on runtime package authority, not the other way around
- schema import cycle disappears
- parser dispatch, validation, proof, and schema CLI remain package-aware

### Verification

- schema registry/unit/integration tests
- schema operator workflow tests
- artifact proof/cohort tests
- parsing/validator tests

## Step 6: Public API Narrowing

### Goal

Make package exports match actual product shape.

### Current Problems

- package root exports archive core plus higher-order semantic-analysis helpers
- `polylogue.lib` also exports a broad mixed surface
- this makes the library story look broader and flatter than the runtime
  architecture actually is

### Target Shape

Two clearer public surfaces:

1. archive-core API
2. semantic-analysis/reporting API

That does not require deleting useful helpers. It requires exporting them from
the right place and stopping the root package from implying they are the same
kind of contract.

### Main Modules

- `polylogue/__init__.py`
- `polylogue/lib/__init__.py`
- `polylogue/facade.py`
- `polylogue/sync.py`
- docs and examples that import package-root semantic helpers

### Required Refactors

1. define the intended root-package export contract explicitly
2. move higher-order semantic-analysis exports behind more precise modules
3. keep archive-core usage simple:
   - facade
   - sync facade
   - core domain models
   - query access
4. update docs/examples/tests to import semantic-analysis helpers from specific
   modules instead of package root

### Acceptance Criteria

- `polylogue.__init__` stops acting as a giant mixed lazy export surface
- semantic-analysis helpers remain available but no longer blur the archive-core
  story
- library examples align with the narrowed shape

### Verification

- package import tests
- sync/facade tests
- docs/examples sanity checks

## Cross-Cutting Track: Sync/Async Boundary Cleanup

This track should run with Steps 1 through 4.

### Goal

Make sync entry behavior deliberate and singular.

### Scope

- `polylogue/sync.py`
- CLI command modules using `asyncio.run()`
- `site/builder.py`
- helper modules that still call async code directly

### Rule

There should be one small sanctioned sync-entry utility or family of utilities,
not many local `asyncio.run()` calls scattered across operator modules.

## Suggested Commit Decomposition

1. `refactor: unify query execution on typed plans`
2. `refactor: collapse duplicate storage read surfaces`
3. `refactor: normalize cli front-door and sync entry`
4. `refactor: externalize showcase catalog and narrow runner`
5. `refactor: split schema runtime authority from tooling`
6. `refactor: narrow package-root api surface`

Each of those may still need 2 to 4 atomic commits internally, but they should
land as coherent phase bundles rather than as scattered local cleanups.

## Exit Criteria

- query execution has one canonical typed path
- storage read access has one canonical low-level surface
- root CLI semantics no longer depend on hidden Click arg rewriting
- showcase verification no longer participates in broad CLI import cycles
- schema runtime authority no longer depends on schema tooling
- package-root exports clearly distinguish archive core from semantic-analysis
  helpers
- the remaining large modules are central because of real product shape, not
  because parallel responsibilities survived there

## Deferred Only After This Program

These are valid later programs, but not part of this one:

- further proof-surface expansion
- provider-parser local simplification passes
- deeper performance work on rendering or indexing
- feature cuts based on the now-clearer architecture
