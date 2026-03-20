# Polylogue Refactoring-First Streamlining Program

Date: 2026-03-19

## Purpose

This document extracts a maximally ambitious, refactoring-first program from
the broader proof, cohesion, and observability work.

It is intentionally biased toward changes that are unambiguously simplifying:

- fewer parallel architecture shells
- fewer local result dialects
- fewer duplicated orchestration paths
- clearer package and repository topology
- stronger reuse of schemas and verification infrastructure

It does **not** start by deciding which features to cut.
The first move is to make the codebase more integrated and readable so later
feature cuts, if any, are obvious rather than speculative.

## One-Line Goal

Make Polylogue feel like one coherent system with one canonical set of
operations, one raw-artifact boundary, one schema-driven contract layer, one
verification subsystem, and one readable repository topology.

## Why This Program Exists

Polylogue's main implementation is not a mess. It has a real center:

- source discovery and raw ingestion
- validation against schemas
- provider parsing into canonical models
- storage/query/render/export

But the repo is harder to understand than the core product warrants because
multiple strong subsystems still express similar truths in different ways.

The result is not primarily wasted functionality. It is duplicated shape:

- several application shells over the same archive operations
- several result/report grammars over the same idea of status plus evidence
- several orchestration paths around verification and seeded workspaces
- schema authority split across schema code, parsers, source walking, and
  synthetic generation
- root and docs topology that make the project look broader and noisier than it
  really is

## Restated Constraint

The streamlining must preserve the project's actual ambitions:

- verification is core, not decoration
- schemas are supposed to reduce hardcoding, not just annotate samples
- provider-specific handling is allowed, but it should live at explicit
  boundaries
- major simplifications should come from convergence and deletion of
  duplicate paths, not from removing difficult but valuable capabilities

## Refactoring Rules

### 1. Refactor Before Cutting

Do not start by deleting features whose value is still obscured by duplicate
infrastructure.

First:

- converge the architecture
- remove parallel shells
- centralize shared models and operations

Then evaluate what remains unjustified.

### 2. Verification Stays Core

`polylogue/showcase/` is not just demo code. It currently contains:

- exercise definitions
- invariant checks
- seeded workspace orchestration
- QA runner logic
- report generation
- VHS capture helpers

That should be simplified and renamed more clearly over time, but it should
remain a first-class product subsystem.

### 3. Schema Authority Should Increase, Not Decrease

The schema system is not a side project.

Refactoring should push schemas underneath runtime behavior:

- stronger cohort and wire-format contracts
- stronger parser routing authority
- stronger structural walking authority
- less hardcoded truth stranded in ad hoc parser and source logic

### 4. Only Add New Infrastructure If It Deletes More Than It Adds

Some enabling additions are worthwhile, but only when they immediately replace
duplicate implementation paths.

Examples:

- a shared operation handler layer
- a canonical result envelope
- a canonical artifact-record interface

Bad additions are abstractions with no immediate deletions behind them.

## Current Structural Problems

### 1. Too Many Application Shells

The same archive behavior is exposed through too many partially-overlapping
surfaces:

- `polylogue/services.py`
- `polylogue/facade.py`
- `polylogue/sync.py`
- `polylogue/mcp/server.py`
- CLI command implementations

The codebase needs one canonical operational core and thinner adapters.

### 2. Storage Is Wider Than Necessary

`polylogue/storage/repository.py` is a large multipurpose surface sitting above
an already substantial backend/query stack.

Too much storage behavior is expressed as:

- repository wrapper
- backend wrapper
- query helper
- store/result object

The simplification target is fewer layers, not just smaller methods.

### 3. Raw Artifact Boundaries Are Spread Out

`polylogue/sources/source.py` currently carries too many roles:

- source walking
- raw byte decoding
- provider detection
- parse dispatch
- ZIP filtering
- cursor bookkeeping
- source-specific enrichment

This should be split into explicit boundaries and pushed toward
`polylogue/lib/raw_payload.py` plus smaller source/runtime components.

### 4. Result And Report Grammars Are Fragmented

The codebase has too many local result families for essentially similar
concepts:

- pipeline stage outcomes
- health checks
- schema audits
- showcase/QA reports
- storage/search summaries

This makes both implementation and observability harder to follow.

### 5. Verification Orchestration Is Duplicated

Verification is rightfully central, but its implementation is split across:

- `polylogue/showcase/runner.py`
- `polylogue/showcase/qa_runner.py`
- `polylogue/showcase/report.py`
- `polylogue/showcase/vhs.py`
- root `demos/`
- CLI `qa`
- stale `demo` references in docs and CI

The right move is not "remove verification". It is:

- build one verification engine
- give it multiple projections
- stop maintaining several overlapping execution stories

### 6. Schema Authority Is Split

Schemas already matter for validation and synthetic generation, but too much
runtime truth still lives elsewhere:

- parser routing
- provider detection
- wire-format assumptions
- provider-specific structural walking

This makes the schema system simultaneously ambitious and under-authoritative.

### 7. Repository Topology Is Overwhelming

The root currently makes the project look more chaotic than it is:

- many top-level artifact-looking directories
- `docs/` mixing canonical docs with plans, dialogue notes, and artifact
  warehouses
- stale historical references to `polylogue demo` remain in dated planning notes,
  while active docs and help surfaces now use `generate` and `qa`

This is a real maintainability problem, not just aesthetics.

## Target Shape

When this refactoring program is done, Polylogue should read as:

1. one operational core
2. one raw-artifact and schema contract layer
3. one verification subsystem
4. one storage/query surface
5. one readable repo and docs topology

The system can still have multiple public surfaces:

- CLI
- MCP
- Python API
- static site

But those should be thin views over the same internal operations.

## Program Pillars

## Pillar 1: Canonical Operations Core

### Goal

Replace overlapping application shells with one shared operation layer.

### Main Changes

- Define a small set of canonical operations for:
  - ingest
  - query
  - render/export
  - verify
  - generate seed/demo data
  - schema maintenance
- Make CLI, MCP, and library-facing surfaces invoke those operations rather
  than reconstructing their own logic.
- Slim `facade.py` into a genuine adapter or remove it.
- Collapse `sync.py` into the same operational core instead of keeping a second
  public wrapper stack.

### Why This Is Refactoring, Not New Product Work

This does not add major capability. It removes duplicated entry paths and makes
all surfaces easier to reason about.

### Exit Criteria

- there is one obvious place to read archive operation logic
- CLI and MCP are visibly thinner
- `facade.py` no longer re-implements repository/search/stat orchestration

## Pillar 2: Runtime Scope And Dependency Convergence

### Goal

Use one runtime-scope pattern everywhere.

### Main Changes

- Promote `RuntimeServices` or its successor into the canonical dependency
  bundle.
- Route CLI commands, MCP handlers, verification runners, and library entry
  points through the same scope construction.
- Delete ambient setup logic that currently appears in several places.

### Exit Criteria

- one canonical way to get config, repository, storage backend, search, and
  rendering services
- fewer hand-constructed service bundles across surfaces

## Pillar 3: Raw Artifact Boundary Cleanup

### Goal

Make the pre-parse boundary explicit and small.

### Main Changes

- Split `sources/source.py` into narrower responsibilities:
  - source walking
  - raw byte acquisition
  - raw payload decoding
  - artifact/provider classification
  - parse dispatch
- Make `lib/raw_payload.py` the canonical place for raw payload identity and
  decode outcomes.
- Reduce source-specific enrichment code inside generic dispatch layers.

### Why This Matters

This is the cleanest way to reduce hardcoded drift without pretending provider
differences do not exist.

### Exit Criteria

- `sources/source.py` stops being a mini-platform
- raw payload behavior is easier to inspect independently of provider parsing

## Pillar 4: Schema-Authority Refactor

### Goal

Keep schemas central, but make them more operational and less sideways.

### Main Changes

- Separate schema concerns more clearly:
  - structural inference
  - semantic hints
  - privacy/redaction
  - cohort/version management
  - runtime contract use
- Strengthen schema use in runtime:
  - parser routing hints
  - cohort/wire-format matching
  - generic tree/stream walkers where possible
  - stronger synthetic generation contracts
- Reduce side machinery that does not materially strengthen runtime authority.

### Important Constraint

This is **not** a "make schemas smaller by making them less important" move.
It is:

- less architectural sprawl around schemas
- more runtime authority flowing from schemas

### Exit Criteria

- schemas play a larger role in runtime contracts
- hand-maintained parsers shrink toward policy hooks where feasible
- schema code is easier to understand by concern, not just by file size

## Pillar 5: One Verification Subsystem

### Goal

Turn verification, seeded execution, QA, and capture into one coherent
subsystem with multiple projections.

### Main Changes

- Converge `showcase/runner.py` and `showcase/qa_runner.py` into one seeded/live
  verification engine.
- Treat:
  - exercises
  - invariants
  - reports
  - captures
  - seed/demo generation hooks
  as different views over one execution model.
- Keep VHS/capture support only as a projection of the same verification run,
  not as a separate orchestration story.
- Decide later whether the package should stay named `showcase` or become
  `verification`; do not force the naming decision before the code converges.

### Why This Matters

Verification is supposed to be part of what makes Polylogue trustworthy.
That argues for **deeper integration**, not for ejecting it from the codebase.

### Exit Criteria

- one verification workspace lifecycle
- one verification execution path
- one machine-readable run record
- optional human-readable and capture projections layered on top

## Pillar 6: One Result And Observability Grammar

### Goal

Make execution readable without every subsystem inventing its own report shape.

### Main Changes

- Define a small shared result envelope for major operations:
  - subject
  - status
  - counts
  - evidence/details
  - warnings/errors
  - timings
- Refactor health, schema audit, pipeline stage summaries, and verification
  reporting toward that grammar.
- Reuse existing explicit-evidence patterns from schema explainability where
  they are genuinely informative.

### Important Constraint

Do not turn every heuristic into theater.
Only record decisions or evidence that answer a real operator question or drive
runtime behavior.

### Exit Criteria

- fewer local report/result dialects
- easier cross-surface observability
- execution narratives and machine-readable outputs share the same core facts

## Pillar 7: Storage Surface Reduction

### Goal

Collapse storage to the fewest layers that still preserve clean boundaries.

### Main Changes

- Split storage into:
  - lower-level query/write backend
  - higher-level archive operations/hydration layer
- Remove pass-through repository methods that add little value.
- Consolidate tag/metadata/stats/search mutations and fetches around canonical
  query objects rather than repeated ad hoc wrappers.
- Prefer DB-resident operational records over ad hoc side files when the data is
  structured, queryable, and part of ongoing runtime truth.

### Exit Criteria

- smaller public repository surface
- less indirection for common storage flows
- clearer ownership of query composition versus hydration/business logic

## Pillar 8: Public API Surface Slimming

### Goal

Reduce the number of mixed, partially-overlapping public import surfaces.

### Main Changes

- Make top-level exports more intentional.
- Avoid maintaining parallel "friendly" wrappers over the same archive
  functions.
- Route public Python API access through the same operation core used by CLI and
  MCP.

### Exit Criteria

- easier answer to "what is the supported Python API?"
- less duplication between top-level package exports, facade types, and storage
  objects

## Pillar 9: Repository Topology Simplification

### Goal

Make the repo look like the product it actually is.

### Main Changes

- Reduce visible top-level noise where possible:
  - generated/output directories stay ignored and clearly non-canonical
  - root should visually center on source, tests, docs, devtools
- Split `docs/` into clearer strata such as:
  - canonical guides/reference
  - verification architecture
  - historical plans and archives
- Repair stale public narratives:
  - README
  - CI workflows
  - docs
  so they reflect current commands and architecture
- Decide deliberately what remains under root `demos/` after the verification
  subsystem converges, instead of leaving an older parallel story in place.

### Exit Criteria

- root is easier to scan
- docs are less flat and less overwhelming
- the public story of the project matches the actual code paths

## Decisions Already Made

These choices should be treated as settled unless implementation evidence
forces a revision.

### 1. Verification Remains In The Main Package

The current `polylogue/showcase/` area is part of Polylogue's trust and QA
surface. It should be converged and renamed more clearly if needed, but not
treated as optional demo cruft by default.

### 2. Schemas Become More Authoritative

The schema system should move closer to runtime control, not further away from
it. The refactor should reduce hardcoded structural truth outside schemas where
practical.

### 3. `RuntimeServices` Is The Seed Of The Runtime Core

The existing runtime-scope model is the correct direction. New work should
converge on it rather than bypassing it.

### 4. `ConversationQuerySpec` Remains The Seed Of Query Intent

Do not invent a parallel query mini-language for other surfaces. Broaden the
existing query intent model or compile into it.

### 5. `showcase` Versus `verification` Is Deferred

First converge the code. Then rename the package if the new shape makes that
clearly beneficial.

### 6. DB-Resident Structured Truth Is Preferred

Whenever runtime records are structured, queryable, and used across runs, they
should be DB-backed first and optionally exported to files.

### 7. No Early Feature-Cut Program

The first refactoring waves should not depend on deciding whether weaker
features survive. Architecture should be streamlined first.

## Concrete Target Architecture

This section defines the intended code shape closely enough that
implementation should not have to rediscover the obvious boundaries.

## 1. Operations Layer

Add one shared internal operations package, likely `polylogue/operations/`,
with modules close to:

- `operations/ingest.py`
- `operations/query.py`
- `operations/verification.py`
- `operations/generate.py`
- `operations/schema.py`
- `operations/export.py`

Responsibilities:

- accept canonical runtime scope plus typed request objects
- execute product behavior
- return shared outcome envelopes

Non-responsibilities:

- CLI formatting
- MCP transport details
- top-level package convenience wrappers

The CLI, MCP server, and public Python API should all call into this layer.

## 2. Runtime Scope

Keep `polylogue/services.py` as the runtime-scope home unless a rename becomes
necessary.

Target shape:

- `RuntimeServices` owns config/backend/repository/search/render dependencies
- CLI `AppEnv` becomes a UI wrapper around runtime scope, not a second runtime
  model
- MCP global service state should be a transport-level holder for the same
  runtime scope, not a parallel lifecycle design

The plan should treat `AppEnv` as:

- `ui`
- `services`

and nothing more ambitious.

## 3. Sources And Raw Payload Boundary

The current `sources/source.py` responsibilities should be redistributed into a
small source pipeline, for example:

- `sources/walker.py`
- `sources/raw_io.py`
- `sources/zip_io.py`
- `sources/dispatch.py`
- `sources/parsers/*`

And the raw payload layer should become the canonical home for:

- decode result
- wire format
- provider hint or provider classification
- artifact classification
- malformed JSONL counts

Important design correction:

`lib/raw_payload.py` should not depend on `sources/source.py` for provider
detection. That dependency points the wrong direction and should be inverted by
extracting provider detection into a smaller shared helper or moving it into the
raw payload layer directly.

## 4. Verification Subsystem

Keep the subsystem in the main package, but refactor it toward this shape:

- `showcase/catalog.py`
- `showcase/workspace.py`
- `showcase/engine.py`
- `showcase/projections/report.py`
- `showcase/projections/capture.py`
- `showcase/invariants.py`
- `showcase/generators.py`

Where:

- `catalog` defines exercises and dependency graphs
- `workspace` owns isolated environment creation and seeding
- `engine` executes exercises and stages
- `projections/*` turn one verification run into JSON/Markdown/VHS/cookbook
  outputs
- `invariants` evaluates run outputs, not orchestration
- `generators` produces derived exercises

The key design choice is:

- one run model
- many projections

not:

- several runners that happen to produce similar outputs

## 5. Storage Shape

Keep two main storage layers:

- low-level backend/query execution
- high-level archive operations and hydration

The repository should become smaller and more opinionated, not remain a wide
pass-through surface.

Good repository responsibilities:

- hydrate canonical models
- batch retrieval
- conversation-scoped writes
- archive-level mutations that compose several backend operations

Bad repository responsibilities:

- exposing every backend helper one-for-one
- acting as a second place to define filter logic that already exists in query
  objects

## 6. Result Grammar

Define one shared internal outcome shape, likely around:

- `status`
- `subject`
- `counts`
- `details`
- `warnings`
- `errors`
- `timings`
- `artifacts`

This should become the spine for:

- pipeline stage outcomes
- verification reports
- health/audit checks
- schema audit summaries
- operation-level summaries

Not every existing result class must disappear, but each should become either:

- a thin typed specialization of the same grammar
- or removable

## 7. Public Python Surface

The public Python surface should end up as:

- an async client over the operations layer
- an optional sync bridge over the same operations layer

It should **not** remain:

- repository API
- facade API
- sync facade API
- top-level mixed exports

all at once.

## Concrete Ownership By Existing Files

This section maps the current code to the intended future ownership.

### `polylogue/services.py`

Keep and strengthen.
It should remain the canonical runtime dependency scope.

### `polylogue/cli/types.py`

Keep, but narrow.
`AppEnv` should stay a CLI concern and stop accreting application logic.

### `polylogue/facade.py`

Shrink drastically or replace.
Its value should be as a thin async client over canonical operations, not as a
parallel application core.

### `polylogue/sync.py`

Keep only as a transport bridge over the async surface.
It should not define a second product API.

### `polylogue/sources/source.py`

Split aggressively.
This is one of the highest-leverage refactor targets in the repo.

### `polylogue/lib/raw_payload.py`

Strengthen and make more central.
It should own the canonical pre-parse payload identity model.

### `polylogue/storage/repository.py`

Narrow and sharpen.
Keep it if it becomes a real hydration/archive-operations layer; otherwise split
or rename to match its smaller role.

### `polylogue/showcase/runner.py` and `polylogue/showcase/qa_runner.py`

Converge into one engine plus one workspace lifecycle.
Do not preserve both orchestration paths.

### `polylogue/showcase/report.py` and `polylogue/showcase/vhs.py`

Keep only as projections of the unified verification run model.

### `polylogue/schemas/schema_generation.py`

Refactor by concern rather than shaving logic blindly.
The main split should separate:

- structural inference
- semantic annotation
- redaction/privacy
- cohort/version management
- registry writes

## Implementation Invariants

These rules should be enforced during refactoring to prevent architecture drift.

### 1. No New Public Surface May Skip The Operations Layer

If a new CLI/MCP/library action is added, it should call a canonical operation
instead of embedding product logic locally.

### 2. No New Raw Decode Logic Outside The Raw Payload Layer

Raw JSON/JSONL decode, malformed-line accounting, and provider/artifact
classification should not be reimplemented in random callers.

### 3. No New Workspace-Bootstrap Logic Outside Verification Workspace Code

Fresh HOME/XDG/archive-root setup should live in one place.

### 4. No New Report Dataclasses Without A Shared-Grammar Justification

If a new result type appears, it should either instantiate the shared outcome
model or explain why it cannot.

### 5. Parser-Specific Code Must Explain Why Generic Structural Walking Fails

Provider hooks are allowed, but the default pressure should be toward shared
schema-driven structure handling where that is realistic.

### 6. File Outputs Must Be Projections Of Structured Runtime Truth

Avoid new ad hoc JSON/Markdown side files that become shadow truth.

## Detailed Migration Waves

These waves are narrower than the high-level pillar sequence and should be
treated as the implementation roadmap.

### Wave 1: Extract The Operations Core

Touch:

- CLI command handlers
- MCP entrypoints
- `facade.py`
- `sync.py`

Deliverables:

- shared operations package
- first operation request/response types
- CLI/MCP/facade routed through them for a small set of commands

Primary deletion target:

- duplicated query/list/stats/search orchestration

### Wave 2: Split `sources/source.py`

Touch:

- `sources/source.py`
- `lib/raw_payload.py`
- parser dispatch call sites

Deliverables:

- clear source walking module
- clear raw payload decode/classify module
- clear dispatch module

Primary deletion target:

- mixed walking/decode/dispatch logic in one file

### Wave 3: Unify Verification Execution

Touch:

- `showcase/runner.py`
- `showcase/qa_runner.py`
- `showcase/report.py`
- `showcase/vhs.py`
- CLI `qa`

Deliverables:

- one verification run model
- one workspace lifecycle
- one engine
- projections layered on top

Primary deletion target:

- duplicate workspace creation and pipeline execution logic

### Wave 4: Introduce Shared Outcome Grammar

Touch:

- verification subsystem
- health
- schema audit
- pipeline results where low risk

Deliverables:

- core outcome/evidence model
- first conversions of existing result/report types

Primary deletion target:

- local report dialect proliferation

### Wave 5: Narrow Repository And Clarify Backend Boundaries

Touch:

- `storage/repository.py`
- backend/query modules
- high-churn storage callers

Deliverables:

- slimmer repository interface
- clearer split between query execution and hydration/archive operations

Primary deletion target:

- pass-through repository wrappers

### Wave 6: Refactor Schema Authority

Touch:

- `schemas/schema_generation.py`
- schema registry/inference modules
- parser routing where schemas can take over
- synthetic generation contracts

Deliverables:

- clearer schema-by-concern structure
- stronger schema role in runtime contracts

Primary deletion target:

- runtime structural truth stranded outside schema/cohort contracts

### Wave 7: Public API Slimming And Repo Topology Cleanup

Touch:

- top-level exports
- README/docs/CI references
- docs tree structure
- final `showcase`/`demos` placement decisions

Deliverables:

- consistent public API story
- consistent verification story
- cleaner docs/root topology

Primary deletion target:

- stale `demo` narrative
- mixed canonical and historical docs at one flat level

## Non-Goals

These should explicitly **not** be attempted as part of the first refactoring
waves.

### 1. Do Not Replace All Provider Parsers In One Shot

The goal is to shrink and discipline parser-specific code, not to pretend all
providers already fit one generic parser.

### 2. Do Not Rename Major Packages Before Their Internals Converge

Do not spend early effort on `showcase` versus `verification` or similar naming
work before the underlying architecture is simplified.

### 3. Do Not Move Every File-Backed Output Into SQLite

Only move structured operational truth into the DB. Large human-facing
materializations can stay file-backed.

### 4. Do Not Invent Broad New Taxonomies Just To Describe The Refactor

New internal types are acceptable only when they immediately replace local
dialects or unblock deletions.

### 5. Do Not Force Feature Pruning Before The Codebase Is Streamlined

Some features may still end up removed. That decision should come after the
duplicate architecture around them has been collapsed.

## Radical Simplifiers Worth Serious Consideration

The earlier sections describe the strong refactoring path that preserves most of
the current package topology.

This section goes further. These are the genuinely large architectural moves
that could make Polylogue much simpler and more coherent if the project is
willing to absorb substantial disruption.

They should not be treated as mandatory, but they are serious options rather
than daydreams.

## Option A: Rebuild Polylogue Around A Compiler Pipeline

### Thesis

Treat Polylogue as a compiler from raw provider artifacts to canonical archive
facts and derived projections.

The core stages become:

1. source scan
2. raw artifact graph
3. structural IR
4. semantic IR
5. derived projections

### What This Changes

The key shift is to introduce an explicit intermediate representation between
raw payloads and canonical conversations.

That means:

- provider-specific parsers no longer have to do every step at once
- schemas can bind to structural IR instead of only validating raw payloads
- verification can assert properties at multiple explicit stages
- projections like render/search/site/export operate on clearer upstream truths

Important constraint:

- embeddings or heuristic semantic inference may assist classification, drift
  detection, and candidate remapping
- they should not silently become the sole source of runtime parsing truth

The explicit IR boundary is better than a magical "self-healing parser" that
quietly rewires extraction logic at runtime.

Important refinement:

- a mostly-universal structural lowerer is still a serious goal
- it should be built from inspectable heuristics, schema annotations, and
  corpus-backed wire-shape evidence
- it does not need to generalize in the abstract to arbitrary unknown providers
- it only needs to work well across the supported provider cohorts that
  Polylogue can actually inspect, infer schemas for, and regression-test

### Structural IR

The structural IR would model the few wire-shape families that actually matter:

- message tree
- linear message stream
- record stream
- bundled conversation list
- chunk stream
- sidecar metadata artifact

These are much fewer than the number of providers.

That is the opening for a heuristics-backed universal parser in the practical
sense:

- not one parser for any imaginable export on earth
- but one generic structural lowering engine for the small number of wire-shape
  families that recur across the supported providers

### Semantic IR

The semantic IR would then model:

- conversation identity
- chronology
- turns/messages
- tool calls/results
- reasoning traces
- attachments
- branch and subagent relations

Polylogue already has pieces of this in canonical conversation models and
viewport/content-block semantics. The radical move is to make that a formal IR
boundary rather than an effect of handwritten parser code.

This also creates a cleaner destination for the current schema semantic-role
heuristics. Today they mostly annotate schemas and help synthetic generation.
In the more ambitious version, they should help drive structural lowering into
semantic IR.

### Why This Could Be The Big Win

This is the most plausible path to:

- fewer handwritten parsers
- more schema authority
- better observability
- clearer verification
- less duplicated provider-specific logic

More precise claim:

- not necessarily zero handwritten parser code
- but much less handwritten full-parser code
- more provider-specific escape hatches over a shared lowering engine

### What It Would Cost

- a major migration of parser responsibilities
- explicit IR design work
- temporary duplication while old and new paths coexist

### Recommendation

If Polylogue takes exactly one radical swing, this is the best one.

The most promising version is:

- schema-guided, heuristics-backed structural lowering
- validated against the real historical corpora already on hand
- inspectable in inferred schemas and audit output
- with provider-specific hooks kept as bounded exceptions

## Option B: Turn The Pipeline Into A Build Graph

### Thesis

Replace the current stage-string pipeline with a build system over typed
targets.

Today the pipeline already behaves somewhat like this, but only informally.
Make it explicit.

### Example Targets

- `raw_artifacts`
- `validated_artifacts`
- `structural_ir`
- `semantic_archive`
- `fts_index`
- `render_markdown`
- `render_html`
- `site`
- `verification_seeded`
- `schema_registry`
- `local_reality_scan`

### Why This Matters

It would unify:

- pipeline execution
- site generation
- search/index refresh
- verification runs
- schema generation
- drift/provenance records

into one dependency model.

### What It Simplifies

- `run_sources()` becomes target execution instead of stage branching
- verification becomes "build target + assertions"
- DB and file materializations become named products with provenance
- run records become build records

### Recommendation

This pairs extremely well with Option A.
If Polylogue becomes compiler-shaped, it should probably also become
build-graph-shaped.

Important constraint:

- `verification_seeded` is the canonical reproducible proof target
- `local_reality_scan` is an operator workflow for inspecting real exports and
  drift, not a committed or CI-stable QA artifact

## Option C: Move To Provider Drivers Instead Of Spread Provider Knowledge

### Thesis

Each provider should own one explicit driver descriptor instead of scattering
provider truth across:

- source detection
- parser modules
- schema inference
- synthetic wire formats
- capability assumptions
- verification fixtures

### A Driver Would Own

- source discovery hints
- artifact classification hints
- wire-format descriptors
- structural IR lowering hooks
- semantic lowering hooks
- schema bundle references
- synthetic corpus generation config
- verification scenario contributions

The driver should ideally **not** own a giant handwritten parser in the final
state. The better end-state is:

- driver declares wire-shape and semantic hints
- generic lowering engine does most of the work
- driver supplies exceptions where the provider topology is genuinely unusual

### Why This Could Simplify The Whole Codebase

Adding or revising a provider becomes "edit the provider driver package" rather
than "grep the repo for provider-name branches".

It would also expose where provider-specific logic still exists, which is the
first step toward shrinking it.

### Recommendation

This is the cleanest way to keep provider specificity without letting it leak
everywhere.

It is also a better fit than either extreme:

- worse extreme: one giant handwritten parser per provider forever
- other worse extreme: a magical universal parser with no explicit provider
  boundary

The right target is:

- explicit provider driver boundary
- shared structural lowering engine
- schema-guided heuristics doing most of the repeatable work
- provider hooks only where needed

## Option D: Split The Codebase Into Kernel, Providers, Projections, Adapters

### Thesis

The current package structure is understandable, but it still mixes product
surfaces with archive core.

A more radical package split would be:

- `polylogue/kernel/`
- `polylogue/providers/`
- `polylogue/projections/`
- `polylogue/adapters/`

### Intended Meaning

`kernel/`

- runtime scope
- operations
- storage
- IRs
- result grammar

`providers/`

- provider drivers
- raw models
- provider-specific lowering hooks

`projections/`

- rendering
- search documents
- exports
- summaries
- capture/report outputs

`adapters/`

- CLI
- MCP
- Python API
- site command surface
- verification command surface

### Why This Might Be Worth It

It makes the repo legible at a glance:

- what is core
- what is provider-specific
- what is output/projection logic
- what is transport/interface logic

### Recommendation

This is high-disruption, but becomes much more attractive once the operations
core and verification convergence are in place.

## Option E: Make Projections First-Class And Delete Surface-Specific Formatting Logic

### Thesis

A lot of Polylogue's perceived complexity comes from each surface deciding for
itself how to shape output.

Instead, define a small projection layer that emits named projections such as:

- conversation summary row
- full semantic conversation document
- search hit document
- site page model
- export bundle model
- verification report model

Then make CLI/MCP/site/export consume those.

### Why This Is Bigger Than It Sounds

It would sharply reduce:

- output duplication
- formatter drift
- surface-specific metadata decisions
- repeated summary logic

### Recommendation

This is a strong complement to the compiler/build-graph direction, but weaker
as a standalone radical move.

## Recommended Combined Radical Direction

If Polylogue chooses to be truly ambitious, the best combined direction is:

1. compiler-shaped core
2. build-graph execution model
3. provider driver architecture
4. kernel/providers/projections/adapters package split

In short:

- raw artifacts are compiled into structural IR
- structural IR is lowered into semantic IR
- semantic IR is stored as canonical archive truth
- named projections are built from it
- CLI/MCP/site/verification are thin adapters over the same build and
  projection system

And the parser strategy inside that direction should be:

- inspectable heuristics first
- schema-annotated lowering second
- provider-driver exceptions third
- embeddings only as supporting evidence, never hidden runtime magic

This would be a much larger change than the base refactoring program, but it is
the clearest route to a codebase that is simultaneously:

- less hardcoded
- more verifiable
- more observable
- more comprehensible
- easier to extend without adding local mini-systems

## What Not To Choose As The Radical Direction

These big moves are unlikely to simplify the project enough to justify their
cost.

### 1. Do Not Split Into Separate Services

Polylogue does not need microservices or process boundaries for its core
problems. That would increase surface area and operational complexity.

### 2. Do Not Replace SQLite As The Main Simplification Strategy

The main complexity problems are architectural and semantic, not caused by
SQLite.

### 3. Do Not Bet On Fully Automatic Schema-Inferred Parsing Everywhere

The right target is fewer handwritten parsers plus more schema-driven shared
walking, not magical elimination of all provider-specific lowering hooks.

### 4. Do Not Build A Giant Plugin System First

A driver architecture is useful. A generic plugin marketplace abstraction is
not a simplifier at this stage.

## Allowed Non-Refactoring Work

This program is refactoring-first, but a few enabling additions are justified if
they prevent duplicate work later.

Allowed examples:

- introducing a shared operation-handler layer if it immediately replaces
  duplicated CLI/MCP/facade logic
- introducing a shared result envelope if it immediately replaces several local
  outcome types
- introducing a minimal artifact-record interface if it immediately simplifies
  source/parsing/schema boundaries
- introducing DB-backed run records for verification/reporting if it lets
  multiple ad hoc file outputs collapse into projections

Not allowed:

- new architectural registries with no runtime consumers
- new observability types with no concrete operator question
- new feature work disguised as "foundation"

## Sequencing

The order matters.

0. Schema model correction
1. Canonical operations core
2. Runtime scope convergence
3. Raw artifact boundary cleanup
4. Storage surface reduction
5. Verification subsystem convergence
6. Result/observability grammar unification
7. Schema-authority refactor
8. Public API slimming
9. Repository topology simplification

Notes on ordering:

- schema correction comes first because current clustering/version semantics are
  not yet strong enough to serve as the authority layer for later refactors
- operations and runtime scope come first because they enable deletions
- raw artifact cleanup should happen before deep schema/runtime authority work
- verification convergence should happen before repo-topology decisions about
  `showcase` versus `demos`
- schema-authority refactor should happen after the raw/runtime core is cleaner,
  otherwise schema work will keep binding to unstable plumbing

## What This Program Should Remove

Not specific features at first, but specific kinds of duplication:

- parallel application shells
- parallel storage wrappers
- parallel verification runners
- parallel report grammars
- stale command narratives
- mixed docs strata at the same level

If a feature still looks weak after these deletions, it becomes much easier to
judge honestly.

## Prerequisite Wave 0: Schema Model Correction

Before the larger refactoring program, fix the schema model so it represents the
right thing.

## Why This Must Come First

The current runtime schema system is materially better than the old
one-schema-per-provider state, but it still version-promotes raw artifact
clusters too directly.

That creates the wrong top-level meaning:

- versions are cluster order, not true provider format versions
- `default_version` currently means “largest/default cluster”, not “latest” or
  “actual provider version”
- Claude Code subagent streams are promoted as separate top-level versions even
  though they are part of the main session format family

If later parser/runtime refactors are built on that model, the architecture will
become cleaner around the wrong schema truth.

## Wave 0 Goal

Move from:

- cluster = version

to:

- cluster = evidence
- element schema = contract for one artifact/component kind
- version package = actual promoted provider version

## Wave 0 Target Model

### 1. Cluster

Keep unsupervised/raw grouping as an internal evidence object:

- stable cluster id
- artifact kind
- profile tokens
- representative paths
- sample counts

Clusters stay useful, but they should stop being the public top-level version
unit.

### 2. Element Schema

Introduce schemas for specific provider-format elements, for example:

- `main-stream`
- `subagent-stream`
- `agent-meta`
- `session-index`

Element schemas are the right place to preserve sidecars and secondary streams
without pretending they are independent provider versions.

### 3. Version Package

Promote actual provider versions as folders/bundles that contain multiple
element schemas when needed.

Conceptually:

```text
schemas/
  claude-code/
    versions/
      v1/
        package.json
        elements/
          main-stream.schema.json.gz
          subagent-stream.schema.json.gz
          agent-meta.schema.json.gz
      v2/
        ...
```

The package manifest should describe:

- provider
- version id
- default/latest/recommended status
- chronology metadata (`first_seen`, `last_seen`)
- included element schemas
- source clusters feeding each element
- matching rules for routing payloads into the package

## Wave 0 Design Constraints

### 1. Sidecars Stay Unglued From Core Parseable Conversation Schemas

They should remain excluded from conversation-bearing schema-unit extraction when
appropriate.

But they should still be representable as adjunct element schemas inside a
version package if they are part of the provider format.

### 2. Subagent Streams Are Part Of The Main Claude Code Version Family

They are not independent top-level provider versions.

They may still have their own element schema, but they belong inside the same
version package as the corresponding main session stream family.

### 3. Version Numbers Should Mean Provider Versions, Not Cluster Rank

Cluster ordering can still help choose promotion priority, but it should not
define public version identity.

### 4. `latest`, `default`, And `recommended` Must Be Distinct Concepts

The model should support:

- `latest_version`: newest promoted version by chronology
- `default_version`: safe default match/fallback
- `recommended_version`: optional operator-promoted preferred version

Those should not collapse into one field.

## Wave 0 Deliverables

### A. Replace Flat Provider Version Files With Version Packages

Move from:

- `provider/vN.schema.json.gz`

to a version folder manifest plus element schemas.

### B. Split Promotion Into Two Steps

1. cluster -> element schema
2. element schemas -> version package

This is the core correction.

### C. Rework Matching

Runtime matching should resolve:

1. provider
2. artifact kind / element kind
3. version package
4. element schema within that package

### D. Rework CLI Surfaces

`schema list`, `schema compare`, and `schema promote` should operate on version
packages and element schemas explicitly, not only on flat `vN` files.

### E. Preserve Backward Readability During Migration

During the migration, existing versioned schemas can be read and converted into
the new package model. But the target surface should be the package model only.

## Wave 0 Success Criteria

- Claude Code subagent schemas are no longer top-level versions by themselves
- sidecar/meta schemas can exist as adjunct elements without contaminating core
  conversation schemas
- version identifiers reflect actual provider format families
- `latest` and `default` stop meaning “largest cluster”
- later parser/runtime refactors can safely treat schemas as the authority
  layer

## Wave 0 Implementation Decisions

This section exists to remove the main remaining ambiguity from Wave 0:
how version packages are actually assembled and matched.

### 1. Anchor Versus Adjunct Element Kinds

Not every artifact kind may create a top-level version package.

Anchor element kinds:

- `conversation_document`
- `conversation_record_stream`

Adjunct element kinds:

- `subagent_conversation_stream`
- `agent_sidecar_meta`
- `session_index`
- `bridge_pointer`
- other non-primary element kinds that may later matter for full provider support

Rule:

- only anchor element kinds may create a new top-level version package
- adjunct element kinds may only attach to an anchor-backed package
- if a provider has no adjunct elements, a package may contain only one anchor
  element schema

This means Claude Code subagent streams can never become top-level versions by
themselves.

### 2. Element Clustering Stays Mostly As-Is

The current schema-unit extraction and cluster formation is already useful for
element schemas:

- schema units remain classified by artifact kind
- sidecars remain excluded from core conversation-bearing schema-unit extraction
  where appropriate
- clustering by profile tokens and dominant keys remains the evidence source

The first implementation cut should reuse the current cluster machinery and only
change what gets promoted.

### 3. Package Assembly Is Anchor-Led

Package assembly should proceed in this order:

1. generate element clusters exactly as now
2. identify promoted anchor clusters
3. create one provisional version-package candidate per promoted anchor cluster
4. attach adjunct clusters to anchor packages using bundle-scope evidence
5. merge anchor packages only if they are clearly the same format family

That keeps the main semantic boundary simple:

- anchor cluster decides package existence
- adjunct cluster decides package completeness

### 4. Introduce Bundle Scope As An Explicit Provider Capability

Wave 0 needs one provider-aware concept that the current code does not have:
`bundle_scope`.

`bundle_scope` is the provider-specific key that says:
"these raw artifacts belong to the same export/session family and may therefore
contribute to the same version package."

Initial bundle-scope rules:

- Claude Code:
  session id derived from either the main `*.jsonl` filename stem or the parent
  directory that owns `subagents/`, `agent-*.meta.json`, and sibling sidecars
- Codex:
  file path or stable session-path grouping, which likely yields a one-element
  package in current observed data
- document providers like ChatGPT/Gemini/Claude AI:
  one raw document family per export file unless observed data proves a richer
  bundle relationship

Bundle scope is allowed to use path/layout rules.
That is acceptable because package identity is partly a layout fact, not purely
payload shape.

### 4.1. Record Bundle Scope And Observed Time On Schema Evidence

The current schema-unit/cluster path is missing two things that package assembly
needs:

- `bundle_scope`
- real observed chronology

Wave 0 should extend schema evidence objects so each unit can carry:

- `raw_id`
- `bundle_scope`
- `observed_at`
- `source_path`
- `artifact_kind`
- `profile_tokens`

Observed chronology should prefer:

1. `file_mtime` from `raw_conversations`
2. `acquired_at` when file mtime is unavailable
3. path/session fallback only if no stored time exists

Using generation time for cluster chronology is incorrect and must stop.

### 4.2. Bundle Membership, Not Just Cluster Membership

Package assembly should not operate on clusters alone.
It should operate on observed bundle memberships:

- one bundle scope may contain several raw artifacts
- those artifacts map to element clusters
- package assembly uses those co-occurrence facts

This matters because the same cluster may appear in multiple package versions,
and some bundle scopes contain duplicate main-stream raws from mirrored archive
roots.

### 4.3. Dedupe Within Bundle Scope Before Package Assembly

Bundle scopes should deduplicate repeated evidence before package assembly,
especially for:

- mirrored copies of the same main session file
- repeated sidecars
- repeated adjunct streams from duplicated roots

The first implementation should dedupe at least by:

- `bundle_scope`
- `artifact_kind`
- stable structural/profile identity

and keep the full raw-path set only as supporting evidence.

### 4.4. Do Not Overload One Cluster ID With Two Different Meanings

Wave 0 should carry two different evidence identifiers explicitly:

- `exact_structure_id`: stable fingerprint of a concrete structural shape
- `profile_family_id`: broader family identity based on profile tokens

The current code mixes these concepts too loosely.
Package assembly and runtime resolution need both:

- exact structure ids for precise matching and dedupe
- profile family ids for clustering and conservative merge decisions

One identifier should not pretend to serve both jobs.

### 5. Attachment Rules Use Co-Occurrence, Not Guesswork

Adjunct clusters should attach to anchor-backed packages only when at least one
of these is true:

- shared bundle scope
- explicit path/layout relationship
- stable repeated co-occurrence with the same anchor cluster family

In practice, the first implementation should bias heavily toward shared bundle
scope and path relationship, because those are much easier to verify.

### 5.1. Orphan Adjunct Evidence Must Be Preserved

The observed corpus already contains adjunct-only bundle scopes where the main
session raw is missing from the archive snapshot.

Those cases should not create a top-level version package.
Instead they should be recorded as:

- orphan adjunct evidence
- unresolved package attachment

That preserves completeness without letting missing anchor raws distort the
version model.

### 6. Package Merge Rules Must Stay Conservative

Two anchor-backed package candidates should only merge when all of the following
are true:

- same provider
- same anchor artifact kind
- high structural/profile similarity between the anchor clusters
- overlapping or adjacent chronology window
- no strong evidence of divergent adjunct structure

If that threshold is not met, keep separate packages.
It is better to over-split than to reintroduce giant amalgamated schemas.

### 7. Version Numbering Is Assigned After Package Assembly

Version numbers should not exist during clustering.

They are assigned only after package candidates are finalized:

- sort by chronology and promotion lineage first
- use operator override only when chronology is ambiguous
- persist `latest_version`, `default_version`, and optional
  `recommended_version` separately

This avoids the current "cluster rank becomes public version id" mistake.

### 8. One Package May Produce Several Element Schemas

Each finalized version package should generate:

- one schema per included element kind
- one package manifest that names those schemas and the source clusters

If multiple clusters feed the same package element, their reservoirs should be
merged before generating that element schema.

More precisely:

- package elements are generated from the subset of raw memberships assigned to
  that package
- clusters are evidence inputs, not ownership containers

This allows one recurring adjunct cluster family to contribute evidence to more
than one version package when the corpus shows that relationship.

This is how Claude Code can have:

- one package version
- one main-stream schema
- one subagent-stream schema
- optional adjunct sidecar schemas

without pretending the subagent stream is a separate provider version.

### 9. Runtime Matching Must Return More Than A Flat Version String

The current `match_payload_version()` API returns only one version token.
Wave 0 should replace that with a richer resolution result:

- provider
- bundle scope, when derivable
- artifact/element kind
- version package id
- element schema id
- match reason / evidence

Consumers may then choose what level they need:

- validator may validate against one element schema
- parser routing may use package + element kind
- CLI explain/compare may inspect the full package

Resolution order should be:

1. explicit bundle-scope package link, when available
2. exact element-structure match
3. profile-family similarity match within the same element kind
4. package default/fallback

That order keeps the matching behavior legible and debuggable.

### 10. Synthetic And Validation Consumers Must Rewire Immediately

Wave 0 is not done if only the registry changes.
The first consumers that must move to package-aware reads are:

- schema validator
- synthetic corpus generation
- schema CLI surfaces

Otherwise the old flat-version mental model will remain the real architecture.

### 11. Committed Baselines And Runtime Packages Must Stop Telling Different Stories

Today the committed provider baselines and the runtime clustered schemas are
different truths.

Wave 0 should end with one canonical story:

- either committed schemas become package manifests plus element schemas
- or committed baselines become explicitly historical fixtures rather than the
  default truth

There should not be a long-lived state where the repo advertises one schema
model while runtime uses another.

### 12. Claude Code Is The Design Anchor

Claude Code should be treated as the proving case for Wave 0, because it has:

- main session streams
- subagent streams
- sidecar metadata
- layout-based relationships

If the package model works cleanly for Claude Code, the simpler providers will
follow much more easily.

## Wave 0 Open Questions That Still Need Implementation-Time Validation

These are no longer architectural unknowns, but they still require empirical
validation against the corpus during implementation.

### A. Do Any Observed Providers Need More Than One Anchor Element Per Package?

The expected answer is "probably no" for the current corpus.
If observed data disproves that, the package manifest must allow more than one
anchor element family inside a single package.

### B. How Often Should Similar Anchor Packages Merge?

The conservative default is:

- do not merge unless evidence is strong

Implementation should measure how many candidate packages would merge under
several thresholds before finalizing that rule.

### C. Should Adjunct Schemas Be Generated For All Sidecars Or Only Supported Ones?

The initial answer should likely be:

- generate adjunct schemas for observed sidecars that are structurally stable
- keep clearly noisy or tiny one-off sidecars as observed artifacts without
  promoted schemas

### D. How Much Of The Old Flat-Version Registry API Should Survive Internally?

The target surface should be package-aware only.
If a temporary adapter is needed during migration, it should be explicitly
temporary and deleted within the same migration wave once consumers are rewired.

## Expected Payoff

If executed well, this program should produce:

- a materially clearer codebase without major functionality loss
- fewer files that feel "special" or one-off
- a smaller conceptual gap between schemas, parsers, verification, and runtime
- more leverage for later proof, observability, and semantic-preservation work
- much clearer signals about which features are truly essential and which are
  only surviving because the current architecture obscures their cost

## Success Criteria

The program succeeds if a new contributor can answer these questions quickly:

- where do core archive operations live?
- what is the raw artifact boundary?
- how do schemas influence runtime behavior?
- how does verification run?
- how do CLI, MCP, and Python API relate?
- where does structured runtime truth live?
- what directories and docs are canonical versus historical?

If those answers are easy, the codebase will be both leaner and more legible
before any harder feature-pruning decisions are even necessary.
