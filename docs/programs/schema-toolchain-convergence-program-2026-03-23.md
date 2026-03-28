# Schema Toolchain Convergence Program

Date: 2026-03-23
Status: planned execution program
Role: canonical next implementation campaign after the core architecture convergence wave

See also:

- `core-architecture-convergence-program-2026-03-23.md`
- `schema-package-authority-program-2026-03-22.md`
- `canonical-archive-platform-program-2026-03-19.md`
- `refactoring-first-streamlining-program-2026-03-19.md`
- `../planning-and-analysis-map-2026-03-21.md`
- `.claude/scratch/018-wave0-schema-package-design.md`
- `.claude/scratch/026-schema-taxonomy-and-versioning.md`

## One-Line Goal

Make Polylogue's schema tooling, synthetic corpus generation, and schema
operator surfaces behave like one evidence-driven toolchain rather than a set
of overlapping supermodules.

## Why This Is Now The Main Frontier

The convergence campaign removed the broad cross-cutting architecture drag in:

- query execution
- storage reads
- CLI front-door routing
- showcase runner boundaries
- schema runtime authority vs tooling
- package-root API shape

What remains is no longer the main runtime architecture. It is the schema
toolchain cluster itself.

That cluster is now the largest and least-factored remaining center of gravity
in main code:

- `schemas/schema_generation.py`
- `schemas/synthetic/core.py`
- `schemas/sampling.py`
- `schemas/unified.py`
- `schemas/semantic_inference.py`
- `cli/commands/schema.py`

Those modules are not peripheral. They are where Polylogue's strongest claims
about being:

- non-hardcode-y,
- package-aware,
- synthetic-data-capable,
- schema-explainable,
- and self-verifying

actually cash out.

The main problems are now internal to that toolchain:

1. `schema_generation.py` still mixes observation loading, clustering, field
   stats, annotation, package assembly, persistence, and CLI-facing result
   shaping.
2. `sampling.py` still mixes corpus reading, schema-unit extraction, structure
   profiling, and helper logic used by both runtime and tooling.
3. runtime schema authority still imports parts of the sampling layer, which
   means the runtime/tooling split is conceptually improved but not fully clean.
4. `synthetic/core.py` still mixes package lookup, semantic value generation,
   relational enforcement, provider tree/stream assembly, and wire-format
   serialization in one monolith.
5. schema operator commands still orchestrate several distinct tooling stages
   directly instead of running one typed schema-toolchain workflow.

## Program Thesis

Polylogue should have:

1. one runtime-safe schema observation and payload-structure layer
2. one tooling-only schema generation and package-assembly pipeline
3. one synthetic generation stack that consumes package authority cleanly
4. one schema operator workflow surface that reports typed results
5. one roundtrip proof lane tying package authority, synthetic generation,
   parser lowering, validation, and proof reporting together

## Non-Goals

This program is not:

- a provider-parser rewrite
- a new package-authority redesign
- a storage-backend refactor
- a proof/publication/site campaign
- a feature-cutting pass

Those are valid later programs. This one is specifically about converging the
schema toolchain around the already-canonical package model.

## Architectural Rules

### 1. Runtime Structure Helpers Must Not Live In Tooling Buckets

If runtime registry resolution needs payload shape extraction, bundle-scope
derivation, or structure matching, those helpers must live in a runtime-safe
module rather than inside `sampling.py`.

### 2. Observation, Assembly, And Emission Are Different Stages

Do not keep:

- "read corpus",
- "decide package/profile evidence",
- "annotate schemas",
- "write packages/manifests"

inside one orchestrator file as if they were one concern.

### 3. Synthetic Generation Must Consume Package Truth, Not Side Knowledge

Synthetic generation should consume package manifests, element schemas, and
wire-format configuration through explicit typed inputs. It should not need to
reconstruct hidden authority from scattered heuristics.

### 4. Heuristics Must Produce Inspectable Evidence

Semantic-role inference, profile-family grouping, and package assignment can
remain heuristic, but their outputs must be:

- typed,
- explainable,
- and reusable by operator/report surfaces.

### 5. Roundtrip Proof Is Part Of The Toolchain, Not A Separate Story

The schema toolchain is not complete unless it can prove:

package authority -> synthetic emission -> parser lowering -> validation ->
proof/report surfaces

using one coherent set of typed results.

## Execution Order

The order matters. Earlier steps finish the split between runtime-safe helpers
and tooling logic; later steps build on that cleaner seam.

1. runtime-safe observation helper extraction
2. schema generation pipeline decomposition
3. synthetic corpus stack decomposition
4. schema operator workflow convergence
5. schema roundtrip and synthetic proof lane

## Step 1: Runtime-Safe Observation Helper Extraction

### Goal

Finish the runtime/tooling split by extracting runtime-safe structure helpers
out of `sampling.py`.

### Current Problems

- `runtime_registry.py` imports `_resolve_provider_config`,
  `derive_bundle_scope`, `extract_schema_units_from_payload`, and
  `profile_similarity` from `sampling.py`
- `sampling.py` is nominally tooling/corpus-facing, but part of it is now
  effectively runtime infrastructure
- this keeps the runtime/tooling boundary conceptually fuzzier than the last
  convergence wave intended

### Target Shape

Split the current sampling layer into two clearer strata:

1. runtime-safe payload structure helpers
2. tooling-only corpus iteration and observation loading

The runtime-safe layer should own:

- provider config resolution for schema-unit extraction
- payload-to-schema-unit extraction
- bundle-scope derivation
- exact-structure/profile token helpers needed at resolution time

The tooling layer should own:

- DB/session corpus reading
- sample iteration
- max-sample policies
- observation gathering workflows

### Main Modules

- `polylogue/schemas/sampling.py`
- `polylogue/schemas/runtime_registry.py`
- new runtime-safe helper modules under `polylogue/schemas/`

### Required Refactors

1. extract payload-structure and profile helpers from `sampling.py`
2. make `runtime_registry.py` depend only on runtime-safe helper modules
3. leave corpus-reading and sample-iteration logic in tooling-only modules
4. keep package-resolution semantics unchanged

### Acceptance Criteria

- runtime registry no longer imports `sampling.py`
- payload resolution still returns the same package/version/element/evidence
  shape
- bundle-scope and structure matching remain inspectable and test-covered

### Verification

- runtime registry tests
- validation tests
- parsing/emitter tests
- artifact proof/cohort tests

## Step 2: Schema Generation Pipeline Decomposition

### Goal

Break `schema_generation.py` into explicit stages with typed handoffs.

### Current Problems

- one module still owns the end-to-end pipeline from raw units to written
  packages
- field stats, semantic annotation, relation inference, clustering, and package
  assembly are not represented as distinct pipeline stages
- operator/CLI-facing result shaping is mixed into the same module

### Target Shape

Decompose schema generation into:

1. observation collection
2. field-stat and structure analysis
3. schema annotation
4. package assembly
5. output emission/result reporting

The top-level orchestrator should become a thin workflow module over typed stage
results.

### Main Modules

- `polylogue/schemas/schema_generation.py`
- `polylogue/schemas/field_stats.py`
- `polylogue/schemas/semantic_inference.py`
- `polylogue/schemas/relational_inference.py`
- `polylogue/schemas/tooling_registry.py`
- new generation-stage modules under `polylogue/schemas/`

### Required Refactors

1. introduce typed stage-result models for observation, annotation, and package
   assembly
2. move package assembly out of the same module that performs raw field
   analysis
3. keep redaction/privacy reporting explicit as part of stage outputs
4. keep bundle/package evidence visible and typed all the way through
5. reduce `schema_generation.py` to orchestration plus top-level workflow entry

### Acceptance Criteria

- `schema_generation.py` is no longer the monolithic implementation home for
  every stage
- package assembly has a dedicated typed boundary
- stage outputs are reusable by CLI/report tooling without re-parsing side data
- schema-generation results still write identical package authority

### Verification

- schema generation unit tests
- schema operator workflow tests
- bundled package asset checks
- redaction/annotation tests

## Step 3: Synthetic Corpus Stack Decomposition

### Goal

Make synthetic generation a clear package-driven subsystem instead of a single
god-object.

### Current Problems

- `synthetic/core.py` mixes package lookup, schema traversal, semantic value
  generation, relation solving, provider-specific conversation shaping, and
  wire-format serialization
- the package-resolution and wire-format layers are not explicit enough
- showcase/demo/QA fixture generation therefore depends on a monolith

### Target Shape

Split synthetic generation into:

1. package/schema selection
2. schema-driven value emission
3. provider-specific conversation/stream/tree builders
4. wire-format serialization
5. style/theme overlays for showcase use

### Main Modules

- `polylogue/schemas/synthetic/core.py`
- `polylogue/schemas/synthetic/semantic_values.py`
- `polylogue/schemas/synthetic/relations.py`
- `polylogue/schemas/synthetic/wire_formats.py`
- `polylogue/showcase/workspace.py`
- `polylogue/cli/commands/generate.py`

### Required Refactors

1. turn `SyntheticCorpus` into a thin façade over explicit generation stages
2. move package lookup/version/element selection into a dedicated layer
3. separate provider conversation builders from raw schema traversal
4. keep showcase-style theming as an overlay, not the core generation path
5. make typed generation reports reusable by showcase and CLI operators

### Acceptance Criteria

- `synthetic/core.py` is no longer the only place where generation truth lives
- package version/element selection is explicit and testable
- showcase and CLI generation use the same typed generation workflow
- synthetic roundtrips still work across supported providers

### Verification

- synthetic corpus tests
- synthetic semantic wiring tests
- showcase fixture generation tests
- generate-command tests

## Step 4: Schema Operator Workflow Convergence

### Goal

Make schema CLI surfaces run through one typed tooling workflow rather than
directly orchestrating generation/promotion steps ad hoc.

### Current Problems

- `cli/commands/schema.py` still knows too much about underlying generation and
  registry-tooling mechanics
- result shaping is spread across command handlers instead of a typed workflow
  result
- operator surfaces have to reconstruct what stage did what

### Target Shape

Introduce one schema-toolchain workflow layer that can power:

- `polylogue schema generate`
- `polylogue schema list`
- `polylogue schema explain`
- `polylogue schema compare`
- cluster/manifest export surfaces

with typed result objects.

### Main Modules

- `polylogue/cli/commands/schema.py`
- `polylogue/schemas/schema_inference.py`
- tooling workflow modules under `polylogue/schemas/`

### Required Refactors

1. define typed operator/workflow result models for generation and explanation
2. reduce `cli/commands/schema.py` to binding and rendering
3. keep package evidence, chronology, and promotion information exposed through
   the same workflow results
4. avoid new compatibility wrappers; move commands onto the new workflow
   directly

### Acceptance Criteria

- schema CLI binds onto typed workflow results instead of driving internal
  stages directly
- package/cluster/evidence terminology remains consistent across commands
- operator output becomes easier to extend without reaching into internals

### Verification

- schema CLI tests
- schema operator workflow integration tests
- deterministic output tests where relevant

## Step 5: Schema Roundtrip And Synthetic Proof Lane

### Goal

Turn the schema toolchain into a provable loop rather than a set of plausible
independent components.

### Current Problems

- synthetic generation, parser lowering, validation, and artifact/proof
  reporting are all strong individually, but they are not yet framed as one
  first-class toolchain proof lane
- the existing test corpus exercises many pieces, but not as one named
  end-to-end convergence contract

### Target Shape

One explicit proof lane that demonstrates:

1. package authority selection
2. synthetic emission for a package element
3. parser/lowering roundtrip
4. validation against the resolved package element schema
5. artifact proof / QA / showcase visibility of the same package truth

### Main Modules

- `polylogue/schemas/synthetic/*`
- `polylogue/schemas/validator.py`
- `polylogue/storage/artifact_observations.py`
- `polylogue/showcase/workspace.py`
- `polylogue/showcase/qa_runner.py`
- relevant test and proof suites

### Required Refactors

1. define a named roundtrip proof workflow or fixture family
2. reuse the same package-selection truth through every step
3. make failures preserve stage-specific evidence, not generic mismatch output
4. expose the proof lane through tests and, where useful, operator/report
   surfaces

### Acceptance Criteria

- Polylogue has a first-class schema-toolchain roundtrip proof
- package version/element truth survives synthetic emission, parsing, and
  validation without side channels
- failures are inspectable by stage and provider

### Verification

- schema generation tests
- synthetic roundtrip tests
- parser/emitter/validator tests
- QA/showcase fixture tests

## Suggested Commit Decomposition

1. `refactor: extract runtime-safe schema observation helpers`
2. `refactor: decompose schema generation pipeline`
3. `refactor: split synthetic corpus generation stack`
4. `refactor: converge schema operator workflow`
5. `test: add schema roundtrip proof lane`

## Exit Criteria

- runtime schema authority no longer depends on tooling-only sampling modules
- schema generation stages are explicit and typed
- synthetic generation has a clear package-selection and emission pipeline
- schema CLI binds onto typed workflow results
- one named roundtrip proof lane demonstrates the whole toolchain coherently

## Deferred Only After This Program

- pipeline prepare/source simplification
- deeper async SQLite backend narrowing
- provider-parser local cleanup passes
- further site/report/publication work
