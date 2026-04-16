---
created: "2026-04-12T21:58:00+02:00"
purpose: "Deeper streamlining ideas: scenarios, command split, automagic tooling, and system maps"
status: "active"
project: "polylogue"
---

# Verification Streamlining and Maps

## Main Observation

The current system already has many of the right pieces, but they are arranged
as adjacent tools rather than as one coherent verification platform.

Existing components:

- schema-driven synthetic generation
- showcase exercises
- QA runner
- VHS generation
- validation lanes
- benchmark campaigns
- pipeline probes
- query-memory budget checks
- integration and unit tests

The current weakness is not lack of features. It is that these components are
not compiled from one expressive source model.

## Stronger Unifying Concept

The "scenario registry" idea should likely grow into a richer abstraction:

- **ScenarioSpec** is the semantic root
- everything else is a projection or compiler target

Suggested form:

```text
ScenarioSpec
  corpus
  workload
  assertions
  presentation
  tags
```

That means:

- synthetic generation is no longer a helper for showcase only
- exercises are no longer a separate ontology
- benchmark campaigns are no longer just test-file buckets
- validation lanes are no longer hardcoded command lists

## Exercises Should Probably Be Compiled, Not Authored As The Root

Current `Exercise` is a good execution artifact, but not a strong semantic
source model.

Why it is too low-level:

- it is mostly CLI-invocation-shaped
- validation is output-oriented rather than system-law-oriented
- it mixes human/demo concerns with execution concerns
- it does not naturally encode corpus/workload/assertion separation

Better role:

- `Exercise` should become a **compiled presentation/execution view** of a
  scenario

Roughly:

```text
ScenarioSpec -> ExerciseSpec -> CLI/VHS/QA rendering
```

Where `ExerciseSpec` keeps only:

- command args
- expected visible output constraints
- capture metadata
- tier/grouping/presentation information

This would make showcase/VHS materially simpler and less special-case-driven.

## We Probably Are Cramming Too Much Into Some Commands

### 1. `polylogue audit`

Current reality:

- synthetic vs live
- source scoping
- fresh workspace behavior
- ingest orchestration
- stage gating
- schema audit
- exercises
- invariants
- snapshot archival
- VHS capture
- a `generate` subcommand

This is too much for one production-facing command family.

The concept itself is valid, but the current command is overloaded.

Better shape:

- keep archive/operator commands under `polylogue`
- move verification-orchestration to `devtools`

Likely direction:

- `polylogue schema audit` remains schema-specific
- `polylogue doctor` remains health/repair-specific
- scenario/showcase/seeded verification orchestration moves under `devtools`

Example future split:

- `devtools scenario run ...`
- `devtools scenario qa ...`
- `devtools scenario bench ...`
- `devtools scenario capture ...`

### 2. `polylogue doctor`

`doctor` is broad, but it is conceptually coherent:

- inspect health
- show debt
- optionally repair / cleanup

It should probably stay user-facing.

But it should expose a more explicit internal model:

- health probes
- debt graph
- repair plan
- execution plan

This can power both CLI output and test harnesses.

### 3. `polylogue run`

`run` is also broad, but the broadness is legitimate because it models the
pipeline stages.

The better path here is not command split, but stronger stage/workload models
that verification tooling can reuse.

## Devtools Should Likely Become The Verification Control Plane

This seems like the cleanest long-term model.

Production CLI:

- archive operations
- queries
- repairs
- schema operations
- normal user actions

Devtools:

- scenario orchestration
- scenario compilation
- benchmark campaigns
- validation lanes
- QA and showcase verification
- fixture generation/distillation
- scenario maps / coverage reports

That separation would reduce conceptual clutter in the production CLI while
making verification tooling much more explicit.

## More Powerful Harnesses From The Other Side

Instead of only improving scenario metadata, also improve the harness model.

### Harness should understand phases

A generic verification harness should be able to run:

- setup
- corpus generation / staging
- workload execution
- state inspection
- assertion evaluation
- artifact emission

This is richer than:

- run one command
- check exit code
- grep output

### Harness should understand state

For many scenarios, the most useful assertions are about archive state:

- table counts
- freshness versions
- health/debt consistency
- index readiness
- raw validation state
- progress emission

So the harness should be able to compose:

- CLI assertions
- DB state assertions
- provenance assertions
- performance budgets

### Harness should understand differential checks

This repo repeatedly breaks where two code paths are supposed to converge.

Harness-level first-class support for:

- path A vs path B
- full rebuild vs incremental
- sample decode vs stream decode
- repair preview vs repair execution
- health view vs repair plan

would be extremely high leverage.

## Automagic / Introspection Opportunities

There are several places where introspection and metadata could reduce manual
verification code.

### 1. Command-surface introspection

Already partly present:

- recursive Click command traversal
- generated help exercises
- generated JSON-contract exercises

Can go further:

- infer machine-contract scenarios from commands exposing JSON mode
- infer destructive-preview scenarios from commands exposing `--preview`
- infer output-mode matrix scenarios from command option metadata

### 2. Schema annotation exploitation

Polylogue already has schema annotations and semantic roles.

Use them more aggressively to auto-generate:

- corpus distributions
- relation invariants
- semantic feature coverage maps
- synthetic deep-structure constraints

### 3. Production-code verification metadata

Some verification-critical semantics could be surfaced explicitly in code, not
rediscovered externally.

Examples:

- stage metadata:
  - reads raw rows
  - writes normalized rows
  - invalidates specific read models
- repair target metadata:
  - categories
  - destructive/safe
  - state predicates
  - dependent surfaces
- derived-model metadata:
  - source tables
  - freshness keys
  - readiness predicates

This could enable more automagic:

- coverage maps
- repair/debt consistency checks
- scenario compilation
- dependency-aware invalidation tests

### 4. External-library leverage

Potentially useful:

- property/state-machine testing patterns
- richer JSON-schema-driven data generation
- graph modeling for dependencies and coverage
- CLI introspection utilities

The goal is not cleverness for its own sake; the goal is to let the system
describe itself better.

## Useful Maps To Construct

This is probably one of the highest-leverage reasoning improvements.

We need maps that partition the system exhaustively and non-overlappingly.

### Map 1: Dataflow / artifact map

What artifacts exist, and how they transform:

```text
source artifact
-> raw conversation
-> validated raw state
-> parsed conversation/messages/content blocks
-> derived read models
-> indexes / FTS / embeddings
-> rendered/site artifacts
-> query/user surfaces
```

Use:

- reason about state transitions
- locate stale vs pending vs orphaned drift
- assign ownership for tests and repairs

### Map 2: Workload map

Which commands/workloads touch which artifact classes:

- `run acquire`
- `run parse`
- `run materialize`
- `run render`
- `run site`
- `run index`
- `doctor`
- `doctor --repair --target ...`
- `products ...`
- `search/query`

Use:

- generate non-overlapping workload scenarios
- reason about convergence and overlap

### Map 3: Invariant map

For each artifact class, define:

- freshness invariant
- completeness invariant
- consistency invariant
- repair path
- visibility surface

Use:

- law-driven test generation
- health/debt/repair alignment

### Map 4: Cost map

For each stage/workload:

- wall
- RSS
- major I/O
- dominant record shapes
- scaling variable

Use:

- benchmark prioritization
- memory optimization

### Map 5: Surface map

Partition external surfaces:

- human CLI
- machine JSON CLI
- Python facade
- MCP
- QA/showcase/VHS
- docs/generated references
- devtools verification control plane

Use:

- decide what belongs in production CLI vs devtools
- prevent duplicate coverage machinery

### Map 6: Dependency map for read models

Explicit graph:

- source tables
- derived tables
- indexes
- repair targets
- invalidation triggers

Use:

- automated stale-state tests
- rebuild/repair planning
- differential testing

## Recommended Structural Direction

The most elegant target shape looks like:

1. **ScenarioSpec** as semantic root
2. **CorpusSpec** schema-native and synthetic-first
3. **Harness** that runs workloads and evaluates assertions over CLI + state +
   budgets
4. **Compilers/projections**:
   - scenario -> validation lane
   - scenario -> benchmark campaign
   - scenario -> showcase exercise
   - scenario -> VHS capture recipe
   - scenario -> QA report template

This is better than keeping:

- a showcase catalog
- a benchmark registry
- lane catalogs
- ad hoc probes

as mostly separate hand-maintained systems.

## Immediate Next Meta Step

Do not try to migrate everything at once.

First prove the model on one vertical slice:

- one schema-native `CorpusSpec`
- one `ScenarioSpec`
- one correctness harness path
- one benchmark projection
- one exercise/VHS projection if it stays natural

If that works cleanly, expand from there.
