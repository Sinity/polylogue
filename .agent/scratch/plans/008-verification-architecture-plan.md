---
created: "2026-04-12T21:32:00+02:00"
purpose: "Current verification architecture plan after the unification wave"
status: "active"
project: "polylogue"
---

# Verification Architecture Plan

## Context

This note used to describe a mostly-unbuilt verification architecture. That is
no longer accurate.

The branch now has a real shared verification substrate:

- `polylogue/scenarios/assertions.py`
- `polylogue/scenarios/execution.py`
- `polylogue/scenarios/executable.py`
- `polylogue/scenarios/runtime.py`
- `polylogue/scenarios/corpus.py`
- `polylogue/scenarios/cli_surfaces.py`
- `polylogue/scenarios/product_surfaces.py`
- `polylogue/scenarios/operational_surfaces.py`
- `devtools/authored_scenario_catalog.py`

Those modules already unify:

- showcase exercises and QA inputs
- validation lanes
- mutation campaigns
- durable benchmark campaigns
- synthetic benchmark campaigns
- inferred corpus scenarios

So the remaining plan is no longer "invent a scenario registry". The remaining
plan is to push that substrate into the places where correctness and performance
are still too ad hoc.

## Landed Architecture

### Shared scenario substrate

Current semantic roots:

- `AssertionSpec`
  - one outcome vocabulary across showcase, lanes, and benchmarks
- `ExecutionSpec`
  - one execution model for subprocess-backed scenarios
- `ExecutableScenario`
  - shared authored executable scenario type
- `CorpusSpec` / `CorpusScenario` / `CorpusProfile`
  - shared corpus authorship, inferred corpus compilation, and profile metadata
- authored scenario catalog
  - one cached control-plane aggregation root

This part is done enough that the old "generic scenario registry" wording is
stale.

### Unified projection layer

Current projections already compile from shared authored objects into:

- benchmark campaign entries
- validation lane entries and families
- mutation campaign entries
- showcase exercise generation
- quality reference rendering

This means the main duplication problem in the verification/control-plane area
has already been addressed.

### Inferred corpus scenario compilation

Schema/operator inference now emits:

- inferred `CorpusSpec`s
- inferred `CorpusScenario`s
- richer `CorpusProfile` metadata

This means real-data distillation exists in first form already, even though it
is not yet the final shape.

## Still Open

### 1. Invariant families are still too weak

What is still missing is not the scenario substrate. It is stronger correctness
families built on top of it.

Needed:

- archive substrate laws
- state-transition tests
- differential tests across duplicate paths
- convergence tests for stale -> healthy transitions
- retrieval/index readiness laws
- preview/destructive safety laws

The branch repeatedly found drift between:

- sample decoder vs streaming decoder
- health vs repair preview
- repair vs read behavior
- full rebuild vs targeted refresh
- index readiness vs actual query behavior

Those are still the highest-value law targets.

### 2. Scenario benchmark coverage is still too shallow

The benchmark substrate exists, but scenario coverage is still narrow relative
to the real pressure points discovered in live runs.

Still-needed benchmark classes:

- cold `doctor`
- cold `stats`
- large-archive search and product reads
- parse rerun on pathological raws
- materialize from empty archive
- targeted repair to green
- giant grouped JSONL ingest
- action-event repair/index convergence

The infrastructure is no longer the blocker. Coverage is.

### 3. Live canaries still carry too much epistemic load

Live archive runs should now serve only three purposes:

- periodic confirmation against reality
- fixture discovery
- final canary before trusting a larger architectural change

They should not continue to be the primary place where invariant gaps are
discovered.

### 4. Heavy runtime paths still need verification-oriented hardening

The active operator brief remains correct about open runtime/product debt:

- stale validation state for parseable raws
- malformed raw quarantine policy
- action-event FTS / debt / repair drift
- heavy memory pressure during live runs

That is still real unfinished work and should not be erased by architectural
optimism.

## Next Architecture Wave

### A. Build invariant families on top of the existing substrate

This is now the highest-leverage move.

The current shared scenario/assertion/execution model is good enough to support:

- reusable law harnesses
- scenario-driven differential tests
- state-transition suites
- convergence proofs

This should be implemented against the currently proven paths first:

- action-event rows / FTS / debt / repair
- raw validation / parse backlog / quarantine

### B. Expand benchmark coverage using existing authored scenario roots

Do not invent another benchmark architecture.

Instead:

- attach more benchmark entries to existing authored scenario families
- keep durable artifacts and comparison/reporting on the current campaign path
- make large-archive and pathology scenarios first-class benchmark citizens

### C. Tighten live canary policy

The live archive should graduate from "main discovery tool" to "periodic proof
surface".

Desired end state:

- scenario-driven synthetic/local verification catches most drift first
- live canaries confirm rather than discover
- live findings feed back into richer corpus profiles and fixtures

## Explicitly Superseded Ideas

These were useful during discovery but should no longer be treated as the main
plan:

- adding a new generic `devtools/scenario_registry.py`
- treating benchmark split as primarily an inventory-model problem
- talking about "unifying around scenarios" as if the substrate does not exist

The substrate exists. The remaining work is to use it more aggressively.

## Current Deliverables

The active verification agenda is:

1. add stronger invariant families
2. expand scenario-shaped benchmark coverage
3. reduce live canary epistemic load
4. keep feeding real archive findings back into corpus/profile fixtures

That is the current plan. Anything broader should justify itself against those
four items.
