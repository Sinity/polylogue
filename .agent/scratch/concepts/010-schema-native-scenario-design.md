---
created: "2026-04-12T21:47:00+02:00"
purpose: "Schema-native scenario system for synthetic verification, benchmarks, QA, and showcase"
status: "active"
project: "polylogue"
---

# Schema-Native Scenario Design

## Context

Polylogue already has unusually strong ingredients:

- annotated provider schemas
- schema-driven synthetic generation
- showcase exercises
- QA runner
- VHS generation
- validation lanes
- benchmark campaigns
- pipeline probes

The current weak point is not missing capability. It is that these surfaces do
not share one expressive, schema-native scenario abstraction.

The current synthetic fixture interface is too thin:

- `count`
- `messages_per_conversation`
- `seed`
- `style`

That is enough for demos. It is not enough for systematic verification.

## Core Thesis

The next-generation verification system should be centered on a single
**ScenarioSpec** abstraction that is:

- schema-native
- provider-aware
- archive-centric
- expressive enough to drive:
  - synthetic corpus generation
  - acceptance tests
  - benchmark campaigns
  - validation lanes
  - QA/showcase exercises
  - VHS captures where useful

This should replace the current loose federation of:

- benchmark names
- showcase exercise metadata
- ad hoc fixture generation arguments
- manually curated validation lane scopes

## Better Conceptual Split

The right generic structure is not just "scenario registry". It is a small
family of related specs.

### 1. CorpusSpec

Defines **what archive shape to synthesize**.

Should include:

- provider mix
- conversation count
- message count distributions
- attachment/tool-use/thinking/action-event densities
- timestamp/topology structure
- grouped-export shape
- JSON vs JSONL vs HTML wire format variants
- malformed/truncated/tolerated pathology injection
- giant-record shape
- repo/path/working-directory signal distributions
- semantic-role distributions derived from schema annotations

This should be primarily schema-driven.

### 2. WorkloadSpec

Defines **what the system does to the corpus**.

Examples:

- `run all`
- `run parse --reparse`
- `doctor --json`
- `doctor --repair --target session_products`
- `stats --format json`
- `search`
- `products profiles`
- `verify-showcase`

This lets the same corpus be used for:

- correctness checks
- benchmark scenarios
- operator UX checks

### 3. AssertionSpec

Defines **what must be true** after a workload.

Examples:

- no stale validation rows remain
- health/debt/repair agree
- stale products are rejected before repair
- FTS readiness matches indexable rows
- malformed raws are quarantined, not silently accepted
- progress is emitted during long operations
- result JSON shape matches contract
- max RSS stays under budget

This should unify:

- test assertions
- benchmark budgets
- QA checks

### 4. PresentationSpec

Defines optional human/demo overlays:

- cookbook description
- showcase grouping
- VHS capture eligibility
- operator narrative

This means showcase/VHS become views over the same scenarios, not a parallel
hand-maintained ontology.

## What Should Improve In Synthetic Generation

### Current limit

Synthetic generation is currently provider-schema-driven, but the interface is
too low-dimensional for deep verification. It is great at "generate plausible
provider-shaped data", but weaker at "generate a corpus with this exact class of
system behavior".

### Needed upgrade: scenario-directed generation

Synthetic generation should accept not just "style", but a structured
**CorpusSpec** with constraints like:

- provider mix:
  - `claude-code: 70%`
  - `codex: 25%`
  - `gemini: 5%`
- conversation topology:
  - short / medium / long distribution
  - tool-heavy vs prose-heavy ratios
  - grouped-export chunking behavior
- semantic traits:
  - file ops density
  - git activity density
  - repo attribution noise ratio
  - action-event richness
- failure traits:
  - lone-surrogate tolerated lines
  - truly malformed lines
  - incomplete grouped bundles
  - stale-product archive states
- scale traits:
  - giant single record
  - many medium records
  - cold-read-heavy archive

## Real-Data -> Synthetic-Spec Distillation

This is likely the highest-leverage long-term capability.

Instead of copying real data, Polylogue should be able to **distill** it into a
synthetic scenario spec that preserves deep structural equivalence without
preserving literal content.

### Distillation pipeline

1. Inspect a real archive or raw source.
2. Extract structural distributions:
   - providers
   - conversation length histograms
   - block-type ratios
   - action-event density
   - attachment density
   - timestamp spacing
   - grouped-record patterns
   - large-record percentiles
   - repo/path-signal shapes
   - malformed/tolerated pathology classes
3. Project those into a `CorpusSpec`.
4. Generate a synthetic archive from that spec.
5. Compare structural metrics between real and synthetic outputs.

### Deep equivalence target

The synthetic archive should match the real one on:

- schema conformance
- provider wire shape
- distributional properties
- operator-relevant workload cost shape
- semantic feature density
- known pathology classes

Not on literal values or private content.

## What Should Become More Generic

### Showcase exercises

Current `Exercise` is useful, but too CLI-demo-shaped.

It should be derivable from a more general scenario model:

- scenario id
- workload step
- presentation metadata
- expected visible outputs

Then VHS and QA can compile from that.

### Benchmark campaigns

Current campaigns are test-file-oriented. They should become scenario-oriented:

- benchmark campaign = selection of scenario/workload/assertion tuples

### Validation lanes

Validation lanes should be able to target:

- all scenarios with correctness assertions
- all scenarios with memory budgets
- all scenarios marked as operator-smoke

rather than being manually assembled lists forever.

## Proposed Generic Object Model

```text
ScenarioSpec
  corpus: CorpusSpec
  workload: WorkloadSpec
  assertions: AssertionSpec
  presentation: PresentationSpec?
  tags: [correctness, performance, operator, pathology, showcase, live-canary]
```

Possible sub-objects:

```text
CorpusSpec
  source_kind: synthetic | distilled | mixed
  provider_mix
  archive_shape
  semantic_profile
  pathology_profile
  scale_profile

WorkloadSpec
  command / pipeline stage / repair target / query workload
  cold/warm mode
  setup/teardown

AssertionSpec
  invariants
  state transitions
  output contracts
  performance budgets
  memory budgets

PresentationSpec
  description
  showcase group
  VHS enabled
  cookbook text
```

## Guiding Preference

Generated data should be the default and strongly preferred.

Use real data only for:

- distillation
- canary validation
- discovering new pathology classes

Do not rely on committed real-data fixtures unless synthetic generation cannot
yet express the needed shape.

## Immediate Design Implications

### 1. Upgrade SyntheticCorpus API

Instead of:

- `count`
- `messages_per_conversation`
- `seed`
- `style`

it should also accept something like:

- `corpus_spec`
- or `scenario_spec`

### 2. Add distillation tooling

Potential devtools surface:

- `devtools distill-scenario --from-archive ...`
- `devtools distill-scenario --from-raw ...`

Outputs:

- a scenario/corpus manifest
- optional comparison report

### 3. Compile existing surfaces from ScenarioSpec

Potential compilers:

- scenario -> synthetic fixtures
- scenario -> benchmark campaign
- scenario -> validation lane
- scenario -> showcase exercise
- scenario -> QA expectation set
- scenario -> VHS capture recipe

## Recommended Next Step

Do not immediately refactor the whole system.

First, implement one minimal vertical slice:

1. define `ScenarioSpec` / `CorpusSpec`
2. teach synthetic generation to accept one richer corpus shape
3. drive one correctness scenario and one benchmark scenario from it
4. prove the abstraction works before migrating all existing surfaces

## Why This Matters

This is the path to a verification system that is:

- more expressive
- more synthetic-data-first
- less dependent on ad hoc live reruns
- better at preserving privacy
- better at reproducing deep workload structure
- better at scaling into many test/benchmark/showcase surfaces without drift
