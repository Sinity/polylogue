---
created: "2026-04-13T00:00:00+02:00"
purpose: "Reality-checked module layout and remaining migration work for unification"
status: "active"
project: "polylogue"
---

# Module Layout And Migration For Unification

## Objective

This note used to propose a future layout for the unification work. A large part
of that layout has now landed, but not always in the exact files originally
predicted.

The right question now is not "did we create the exact package tree from this
note?" The right question is "where does semantic ownership actually live now,
and what migration work is still justified?"

## Landed Ownership

### 1. Artifact graph

Landed in:

- `polylogue/artifact_graph.py`

This is close enough to the originally intended runtime ownership. It is a
runtime-semantic root, and `devtools` consumes it rather than owning it.

The earlier proposed move under `polylogue/storage/` is not yet justified by a
clear semantic gain. Do not move it just to match an old scratch note.

### 2. Operation metadata

Landed in:

- `polylogue/operations/specs.py`
- `polylogue/operations/__init__.py`

This part is real and should be treated as done in spirit.

### 3. Scenario and execution semantics

Landed in:

- `polylogue/scenarios/assertions.py`
- `polylogue/scenarios/execution.py`
- `polylogue/scenarios/executable.py`
- `polylogue/scenarios/runtime.py`
- `polylogue/scenarios/sources.py`
- `polylogue/scenarios/corpus.py`
- `polylogue/scenarios/cli_surfaces.py`
- `polylogue/scenarios/product_surfaces.py`
- `polylogue/scenarios/operational_surfaces.py`

This is the main reason the old proposal of a separate `devtools/scenarios/`
compiler package is no longer clearly correct. The semantic roots migrated into
`polylogue/scenarios/`, while `devtools` now mostly aggregates and projects them.

### 4. Control-plane aggregation

Landed in:

- `devtools/authored_scenario_catalog.py`
- `devtools/quality_registry.py`
- benchmark, mutation, validation, and projection catalogs that consume the
  shared roots

This means projection logic is still concentrated in `devtools`, which was the
important part of the original plan.

### 5. Corpus distillation

First-form distillation landed in:

- `polylogue/scenarios/corpus.py`
- `polylogue/schemas/operator_inference.py`

This is not the exact `polylogue/schemas/corpus_distillation.py` split proposed
earlier, but the semantics are present:

- inferred corpus specs
- inferred corpus scenarios
- richer corpus profiles
- package-aware profile compilation

## What Did Not Land Exactly As Proposed

### `devtools/scenarios/`

This package does not exist.

That is not automatically a problem. The current split is:

- semantic roots in `polylogue/scenarios/`
- aggregation/projection in `devtools/`

That is arguably better than the original proposal, because it keeps shared
meaning closer to the product/runtime side instead of making verification code
the owner of every scenario concept.

### `polylogue/schemas/corpus_distillation.py`

This exact module does not exist.

Again, that is not automatically a problem. The question is whether current
corpus distillation work has become too large or too entangled in
`polylogue/scenarios/corpus.py` and `polylogue/schemas/operator_inference.py`.

At the moment, the answer is "not yet clearly enough to justify another split."

## Still Open Module/Layout Work

### 1. Broader runtime drift outside the scenario system

These remain real architectural/runtime issues:

- sync/async storage duplication
- schema bootstrap duplication
- fragmented DB-path and config policy
- likely-dead `polylogue/cli/run_execution_workflow.py`
- legacy compatibility paths around raw/runtime handling

These are still open and should not be hidden by the scenario/control-plane
progress.

### 2. Invariant-family placement

The next architecture phase needs a home for stronger:

- state-transition suites
- differential suites
- convergence suites
- retrieval/readiness law suites

The important constraint is:

- keep the law harnesses near tests or verification code
- keep semantic targets bound to runtime roots

This should be decided based on the first serious invariant-family wave, not in
the abstract.

### 3. Deeper corpus-profile distillation

The current `CorpusProfile` is richer than before, but it is still an early
profile, not a full synthetic-first archive pathology model.

If this area grows significantly, then a more explicit split under
`polylogue/schemas/` may become justified. Not before.

## Migration Policy Going Forward

### Keep

- runtime semantic ownership near runtime code
- `polylogue/scenarios/` as the shared semantic root for scenario execution,
  assertions, corpus authorship, and CLI surface families
- `devtools` as aggregation/projection/control-plane orchestration

### Avoid

- inventing a new package tree only to satisfy an old note
- moving modules without a concrete semantic simplification
- promoting `devtools` into the owner of runtime meaning

## Revised Next Moves

### Step 1: invariant-family wave

Use the existing roots to build stronger:

- action-event law suites
- validation/quarantine law suites
- differential and convergence tests

This is now higher leverage than further package reshuffling.

### Step 2: benchmark coverage expansion

Attach broader large-archive and pathology benchmarks to the current authored
scenario roots.

### Step 3: only then reconsider file splits

If one of these areas grows too large:

- corpus/profile distillation
- authored scenario aggregation
- invariant-family compilation

then split by pressure, not by the historical proposal in this note.

## Best Near-Term End State

The realistic near-term end state is:

- semantic roots in `polylogue/scenarios/`, `polylogue/operations/`, and
  `polylogue/artifact_graph.py`
- projection/control-plane aggregation in `devtools`
- broader correctness and benchmark coverage built on those roots
- runtime architectural cleanup handled as its own serious workstream

That is a better target than forcing the repo into the exact package tree this
note once predicted.
