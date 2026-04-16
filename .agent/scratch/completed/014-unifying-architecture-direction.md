# Unifying Architecture Direction

## Thesis

Polylogue does not primarily need simplification by deletion or flattening. It
needs **semantic unification**: fewer independent roots that all try to model
verification, orchestration, and system shape.

Today the codebase already contains most of the right pieces, but they are
spread across several near-root abstractions:

- pipeline stage models
- schema/operator inference and verification requests
- showcase exercises and QA sessions
- validation lanes
- benchmark campaigns
- the `devtools` command catalog and quality registry

The problem is not lack of capability. The problem is that these capabilities
do not compile from one stronger model.

## Current Near-Roots

### 1. Pipeline state

`polylogue/pipeline/stage_models.py` provides typed models for acquire/validate
pipeline state. This is already close to a canonical runtime/workload substrate.

Strength:

- explicit typed stage outcomes
- archive-facing semantics

Limitation:

- models pipeline transitions only
- does not describe read models, verification claims, or operator workloads

### 2. Schema/operator requests

`polylogue/schemas/operator_models.py` and
`polylogue/schemas/verification_requests.py` already model inference and proof
workflows over real artifacts.

Strength:

- typed request/response discipline
- strong fit for corpus understanding and evidence surfaces

Limitation:

- scoped to schema/evidence workflows
- not connected to benchmark, QA, or operator workload orchestration

### 3. Showcase exercises

`polylogue/showcase/exercise_models.py` models one CLI invocation plus
validation.

Strength:

- concrete execution unit
- already useful for QA and capture

Limitation:

- too CLI-shaped to be the semantic root
- mixes workload, validation, and presentation
- cannot naturally express archive-state assertions, performance budgets,
  or non-CLI projections

### 4. Quality registry

`devtools/quality_registry.py` already aggregates lanes and campaigns.

Strength:

- already behaving as a proto control-plane inventory

Limitation:

- aggregation only
- indexes existing registries instead of owning their semantics

### 5. QA command

`polylogue/cli/commands/qa.py` (`polylogue audit`) is useful, but it is
carrying too many responsibilities:

- data mode selection
- workspace orchestration
- ingestion
- schema audit
- exercise execution
- invariant execution
- capture
- snapshotting

This is a useful operator workflow, but it is not a good root abstraction.

## Better Root Model

The cleanest unifying shape is a small family of semantic roots.

### ArtifactSpec

Describes the durable things Polylogue produces and maintains.

Examples:

- raw conversations
- parsed conversations
- validated raws
- action-event tables
- FTS indexes
- session products
- rendered/site artifacts

Responsibilities:

- dependency edges
- invalidation rules
- freshness rules
- health/debt/repair semantics
- cost ownership where relevant

This should be close to the archive substrate, not a docs-only concept.

### CapabilitySpec

Describes what the system can do and through which surfaces.

Examples:

- acquire
- validate
- parse
- materialize
- repair action-event FTS
- run cold doctor
- verify showcase
- render docs

Responsibilities:

- entry surface (`polylogue`, `devtools`, Python API, MCP)
- arguments/options
- mutability / side effects
- required artifacts
- resulting artifacts
- output contracts

This is the right place to unify command introspection, not just for help text
and completions but for machine-usable control-plane metadata.

### ScenarioSpec

Describes a proof-worthy or benchmark-worthy situation.

Suggested shape:

- `CorpusSpec`
- `WorkloadSpec`
- `AssertionSpec`
- optional `PresentationSpec`

This should be authored once and compiled into:

- an exercise
- a QA session or slice
- a validation-lane item
- a benchmark campaign item
- optionally a VHS/demo capture

This makes `Exercise` an execution artifact, not a semantic root.

## Stronger View Of Existing Components

### Schema inference is already proto-distillation

`schema_inference` and `operator_inference` already do a large part of
"real-data distillation". The missing extension is not understanding real
artifacts, but exporting that understanding as:

- richer `CorpusSpec`
- pathology profiles
- workload-relevant distributions
- equivalence targets for synthetic generation

So the correct move is:

- extend schema/operator inference
- do **not** invent a separate parallel "distillation" subsystem

### Exercises should be compiled, not authored as roots

An authored exercise is too low-level. What should be authored is a scenario.
Then the system can compile:

- CLI exercise args
- expected output constraints
- whether VHS capture is meaningful
- whether live/synthetic modes are allowed

### QA should move toward devtools control-plane ownership

`polylogue audit` is valuable, but architecturally it is mixing product-space
and control-plane orchestration. The better end state is:

- `polylogue`: product/archive operations
- `devtools`: verification/orchestration control plane

The user-facing convenience command can still exist, but its semantics should
come from the control-plane model, not from ad hoc orchestration logic.

## The Most Elegant Unifying Shape

The deeper idea is not "scenario registry". It is a **system graph** that
contains:

- artifacts
- capabilities
- scenarios

From that graph, the repo can derive:

- docs and inventories
- completions and machine discovery
- coverage maps
- verification plans
- benchmark campaign inventories
- showcase/QA/demo manifests

This is better than a registry pile because each node type has a clear
semantic role:

- artifacts answer "what exists / what depends on what"
- capabilities answer "what can the system do"
- scenarios answer "what should be proven"

## Useful Maps To Generate

If the graph exists, `devtools` should be able to render maps that are actually
useful for reasoning.

### Artifact map

- raw -> validated -> parsed -> read models -> indexes -> render/site

### Dependency / invalidation map

- for each derived artifact, what invalidates it
- for each repair target, what it rebuilds

### Capability map

- which surfaces expose which capabilities
- which capabilities are product-facing vs control-plane-only

### Coverage map

- which scenarios cover which artifacts and capabilities
- which assertions are enforced by tests, benchmarks, QA, or canaries

### Cost map

- stage/capability -> wall / RSS / scaling variable

This is the real path to exhaustive, non-overlapping reasoning.

## Migration Direction

The likely best migration is incremental, but each slice should move semantic
ownership upward rather than adding more adapters.

### Slice 1

Define a small internal graph for one area:

- raw records
- action-event table
- action-event FTS
- health/debt/repair relationships

Then derive:

- one health proof
- one repair plan
- one validation scenario
- one benchmark scenario

### Slice 2

Extend schema/operator inference to emit a richer `CorpusSpec` from real data.

### Slice 3

Compile one `ScenarioSpec` into:

- showcase exercise
- benchmark campaign
- validation-lane entry

### Slice 4

Move the semantics behind `polylogue audit` into `devtools`, leaving the CLI
as a thinner product/operator entrypoint if still desired.

## Final Form

The most compelling final form is:

- `polylogue` is the archive platform
- `devtools` is the control plane
- the system is self-describing through artifacts, capabilities, and scenarios
- schema inference distills real data into synthetic corpus specs
- showcase, QA, benchmarks, and validation lanes are compiled projections
- coverage and cost are mapped explicitly rather than inferred from tribal
  knowledge

That is more unifying than "simpler" in the flattening sense, and it preserves
the codebase's real strengths instead of discarding them.
