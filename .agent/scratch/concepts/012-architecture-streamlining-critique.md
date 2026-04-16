---
created: "2026-04-12T23:40:00+02:00"
purpose: "Architecture critique and target-form streamlining plan for Polylogue"
status: "active"
project: "polylogue"
---

# Polylogue Architecture Streamlining Critique

## Reading

This note is grounded in direct inspection of:

- `polylogue/facade.py`
- `polylogue/sync.py`
- `polylogue/sync_bridge.py`
- `polylogue/services.py`
- `polylogue/storage/backends/connection.py`
- `polylogue/storage/backends/async_sqlite.py`
- `polylogue/storage/backends/schema_upgrade.py`
- `polylogue/storage/backends/async_sqlite_schema.py`
- `polylogue/storage/repository.py`
- `polylogue/operations/archive.py`
- `polylogue/cli/click_app.py`
- `polylogue/cli/click_command_registration.py`
- `polylogue/cli/query.py`
- `polylogue/cli/commands/run.py`
- `polylogue/showcase/exercise_models.py`
- `polylogue/showcase/exercises.py`
- `polylogue/showcase/dimensions.py`
- `polylogue/showcase/workspace.py`
- `polylogue/showcase/qa_runner_stages.py`
- `polylogue/schemas/synthetic/core.py`
- `devtools/benchmark_campaign.py`
- `devtools/quality_registry.py`
- `devtools/validation_lane_base.py`

## Current strengths

- There is a real async-first runtime story:
  - `Polylogue` -> `RuntimeServices` -> `ConversationRepository` -> `SQLiteBackend`.
- The archive/product separation is directionally correct:
  - raw substrate and derived products are distinct module families.
- CLI command registration is centralized and clean.
- `devtools` already functions as a repo control plane, not just a script bucket.
- Synthetic generation exists and is schema-driven, which is the right long-term substrate for verification.

## What is wrong

### 1. The repo still has duplicate semantic roots

There are too many competing top-level organizing models:

- product/runtime path
- showcase/exercise path
- devtools validation lane path
- benchmark campaign path
- QA/reporting path

They are adjacent, but they are not compiled from one semantic source.

Symptoms:

- `Exercise` is its own root model in `polylogue/showcase/exercise_models.py`.
- validation lanes have a separate registry shape in `devtools/validation_lane_base.py`.
- benchmark campaigns have yet another registry shape in `devtools/benchmark_campaign.py`.
- `QualityRegistry` just aggregates these separate registries instead of describing one deeper model.

This creates drift pressure by construction.

### 2. Async-first architecture is real, but sync SQLite still exists as a parallel backend surface

The intended canonical path is:

- `Polylogue` in `polylogue/facade.py`
- `RuntimeServices` in `polylogue/services.py`
- `ConversationRepository` in `polylogue/storage/repository.py`
- `SQLiteBackend` in `polylogue/storage/backends/async_sqlite.py`

But `polylogue/storage/backends/connection.py` is not just test plumbing.
It owns:

- sync connection caching
- default DB path logic
- read-connection policy
- schema ensure
- sqlite-vec loading
- scope filter helpers

That means the system still has two backend stories:

- canonical async backend
- broad sync/raw sqlite helper surface

This is a major source of drift, especially because schema/bootstrap logic is duplicated too.

### 3. Schema/version/bootstrap logic is duplicated across sync and async paths

`schema_upgrade.py` and `async_sqlite_schema.py` both encode:

- schema version handling
- legacy-layout rejection
- extension/index/table application
- session-product evolution
- action-event evolution

This is not just annoying duplication. It means correctness changes can land in one path and silently lag in the other.

This is one of the clearest architectural debts in the repo.

### 4. The CLI is mostly well-shaped, but some verification/orchestration concepts are in the wrong plane

`polylogue` should be the product CLI.

That includes:

- query-first archive access
- `run`
- `doctor`
- `products`
- `schema`
- `reset`
- `mcp`

But verification orchestration feels split awkwardly:

- showcase/QA/exercises live under product package space
- validation/benchmark control plane lives under `devtools`

The result is that `polylogue audit` / QA / showcase style concerns are too close to the product CLI ontology.

The more rigorous shape is:

- `polylogue`: user/product/archive operations
- `devtools`: verification, campaigns, scenarios, generated coverage, canaries

### 5. `Exercise` is too low-level to be the root verification abstraction

`Exercise` currently means roughly:

- one CLI invocation
- one validation object
- some presentation metadata

That makes it a decent execution artifact, but a poor semantic source of truth.

It cannot naturally model:

- corpus shape
- workload semantics beyond CLI args
- archive-state assertions
- benchmark expectations
- memory budgets
- convergence/differential checks

This is why showcase, QA, benchmark campaigns, and validation lanes keep needing parallel abstractions.

### 6. Synthetic generation is underpowered relative to the verification ambition

`SyntheticCorpus` is schema-native, which is excellent.
But the public control surface is still shallow:

- `count`
- `messages_per_conversation`
- `seed`
- `style`

That is enough for demos and smoke tests.
It is not enough for:

- provider-mix control
- grouped-export topology
- giant-record shape
- malformed/tolerated pathology injection
- repo/path-noise distributions
- action-event density
- retrieval/index stress shape

So the repo has the right substrate but not enough controllability.

### 7. The system lacks a first-class artifact/dependency/invalidation model

A lot of runtime/maintenance complexity revolves around:

- raw records
- parsed state
- derived products
- indexes
- rendered/site artifacts
- maintenance debt

But this dependency graph is not declared as a first-class model.

So health, repair, and verification logic has to rediscover dependencies indirectly.

This is one reason drift shows up in places like action-event read-model vs FTS health/debt accounting.

## Target final form

### A. One semantic verification root: ScenarioSpec

Use a richer schema-native root:

- `ScenarioSpec`
  - `CorpusSpec`
  - `WorkloadSpec`
  - `AssertionSpec`
  - optional `PresentationSpec`

Everything else becomes a compiled view:

- synthetic corpora
- acceptance tests
- benchmark scenarios
- validation lanes
- QA sessions
- VHS/demo captures
- maybe completions/help contract exercises where sensible

### B. Product CLI vs verification control plane split

Keep:

- `polylogue` = product/archive CLI
- `devtools` = verification/control plane

Then move toward:

- showcase/exercise orchestration compiled and launched through `devtools`
- `polylogue` exposing only genuine product/archive behavior

### C. Artifact graph as first-class metadata

Introduce a declarative model describing:

- artifact classes
- producers
- consumers
- freshness rules
- invalidation rules
- repair targets

This can drive:

- health/debt/repair consistency checks
- convergence tests
- scenario assertions
- coverage maps

### D. One canonical backend semantics surface

Make async backend semantics canonical in code structure, not just by intention.

That likely means:

- one schema/bootstrap module
- one connection-policy model
- sync access reduced to adapters over canonical backend semantics
- fewer production responsibilities in raw `sqlite3` helpers

### E. Scenario-compiled verification, not hand-curated parallel registries

Replace separate roots with projections:

- `ScenarioSpec -> exercise`
- `ScenarioSpec -> benchmark`
- `ScenarioSpec -> validation lane item`
- `ScenarioSpec -> QA session`
- `ScenarioSpec -> VHS capture`

## Recommended architectural moves

### 1. Introduce scenario models under devtools first

Do not refactor the whole repo immediately.
Start with a new verification-semantic root in `devtools`, not under showcase.

Reason:

- avoids destabilizing the product package too early
- lets existing showcase/benchmark/lane registries compile from the new model incrementally

### 2. Extract shared schema/bootstrap semantics

Highest-value substrate cleanup:

- unify `schema_upgrade.py` and `async_sqlite_schema.py`
- expose one canonical schema/bootstrap/incompatibility policy

This is a runtime correctness improvement, not just cleanup.

### 3. Treat `Exercise` as a compiled execution artifact

Do not keep growing `Exercise`.
Instead:

- preserve it as a transport/presentation layer for CLI-driven checks
- compile it from `ScenarioSpec`

### 4. Make synthetic generation far more controllable

Expand from style-based generation to structured corpus control:

- provider composition
- record/container topology
- conversation size distributions
- feature density
- pathology injection
- scale knobs

Long-term, add real-data -> synthetic-spec distillation.

### 5. Add maps as durable repo artifacts

Useful maps to maintain:

- artifact map
- workload map
- invariant map
- dependency/invalidation map
- cost map
- surface map

These should become how coverage and architecture reasoning are discussed.

## Execution order

1. Unify schema/bootstrap semantics.
2. Add artifact/dependency metadata for one narrow area.
3. Introduce `ScenarioSpec` in `devtools`.
4. Compile one existing verification slice from it:
   - ideally action-event FTS consistency or stale validation-state recovery.
5. Recast showcase exercises as a projection, not the semantic root.
6. Expand synthetic corpus controllability.

## Short version

Polylogue is not structurally chaotic. The core problem is that it has several mostly-good subsystems that still describe verification, maintenance, and backend semantics in parallel instead of from one canonical model.

The final form should be:

- one canonical substrate/backend semantics path
- one artifact/dependency graph
- one scenario compiler
- `polylogue` as product CLI
- `devtools` as verification/control plane
- showcase/QA/benchmarks/lanes as projections rather than peer ontologies
