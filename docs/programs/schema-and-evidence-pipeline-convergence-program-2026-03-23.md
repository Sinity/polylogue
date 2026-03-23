# Schema And Evidence Pipeline Convergence Program

Date: 2026-03-23
Status: planned execution program
Role: canonical next implementation campaign after the core architecture convergence wave

See also:

- `core-architecture-convergence-program-2026-03-23.md`
- `schema-package-authority-program-2026-03-22.md`
- `canonical-archive-platform-program-2026-03-19.md`
- `refactoring-first-streamlining-program-2026-03-19.md`
- `testing-reliability-expansion-program-2026-03-14.md`
- `../planning-and-analysis-map-2026-03-21.md`
- `.claude/scratch/018-wave0-schema-package-design.md`
- `.claude/scratch/026-schema-taxonomy-and-versioning.md`

## One-Line Goal

Make Polylogue's full evidence loop behave like one coherent system:

raw artifact -> schema evidence -> synthetic emission -> prepare/persist ->
verification/proof -> operator surfaces.

## Why This Is Now The Main Frontier

The convergence campaign removed the broad architectural drag in:

- query execution
- storage reads
- CLI front-door routing
- showcase runner boundaries
- schema runtime authority vs tooling
- package-root API shape

What remains is now concentrated in one adjacent, high-value band of the
system:

- schema tooling and package assembly
- synthetic corpus generation
- raw-artifact observation and proof
- prepare/persist transformation logic
- source-walking/raw-acquisition seams
- the still-overbroad async SQLite write backend used by those paths

These are not random leftovers. Together they form Polylogue's core
"evidence pipeline" story: the part that determines whether the system is
actually explainable, self-verifying, and not hardcode-y in practice.

The biggest remaining cluster now spans:

- `schemas/schema_generation.py`
- `schemas/sampling.py`
- `schemas/synthetic/core.py`
- `schemas/verification.py`
- `storage/artifact_observations.py`
- `pipeline/prepare.py`
- `sources/source.py`
- `storage/backends/async_sqlite.py`
- `cli/commands/schema.py`

This is now the right next campaign because it is both:

1. the largest remaining center of architectural mass
2. the place where Polylogue's strongest product thesis still depends on
   somewhat overgrown implementation seams

## Program Thesis

Polylogue should have:

1. one runtime-safe structure observation layer
2. one tooling-only schema generation and package-assembly pipeline
3. one synthetic generation stack that consumes package truth directly
4. one raw-to-record preparation pipeline with explicit typed stages
5. one durable artifact-proof/verification workflow over the same evidence
6. one narrower async write backend that acts like infrastructure, not a
   second business-logic home
7. one operator layer for schema/proof workflows that reports typed results

## Non-Goals

This program is not:

- a provider-parser rewrite
- a new package-authority redesign
- a site/publication campaign
- a query/CLI campaign
- a feature-cutting pass

Those are valid later programs. This one is specifically about converging the
schema-and-evidence pipeline around the already-canonical package model.

## Architectural Rules

### 1. Runtime-Safe Helpers Must Not Hide In Tooling Modules

If runtime resolution or artifact inspection needs payload structure helpers,
those helpers belong in runtime-safe modules, not in tooling/corpus-iteration
modules.

### 2. Observation, Assembly, Preparation, And Proof Are Different Stages

Do not keep:

- "observe raw artifacts"
- "derive structure/profile evidence"
- "annotate and assemble packages"
- "prepare records and materialize attachments"
- "persist proof/read models"

inside one implementation bucket simply because they are all schema-adjacent.

### 3. Synthetic Generation Must Consume Package Truth, Not Side Knowledge

Synthetic generation should consume package manifests, element schemas, and
wire-format configuration through explicit typed inputs. It should not
reconstruct hidden truth from scattered heuristics.

### 4. Proof Must Reuse The Same Evidence Surfaces

Artifact proof, schema verification, QA, and roundtrip tests should build from
the same package/evidence/read-model surfaces rather than rescanning or
re-deriving parallel truth.

### 5. Backends Should Look Like Infrastructure

The async SQLite backend should own connection, transaction, schema-init, pool,
and low-level write primitives. Evidence/business workflows should live above
it.

## Execution Order

The order matters. Earlier steps sharpen the runtime/tooling boundary and the
raw-to-record pipeline before later proof and backend narrowing.

1. runtime-safe observation helper extraction
2. schema generation pipeline decomposition
3. synthetic corpus stack decomposition
4. prepare/source transformation convergence
5. artifact proof and verification workflow convergence
6. async SQLite write-path narrowing
7. schema/evidence operator workflow convergence
8. named roundtrip proof lane

## Step 1: Runtime-Safe Observation Helper Extraction

### Goal

Finish the runtime/tooling split by extracting runtime-safe structure helpers
out of `sampling.py`.

### Current Problems

- `runtime_registry.py` imports `_resolve_provider_config`,
  `derive_bundle_scope`, `extract_schema_units_from_payload`, and
  `profile_similarity` from `sampling.py`
- `artifact_observations.py` also depends on structure helpers from
  `sampling.py`
- `sampling.py` is nominally tooling/corpus-facing, but part of it is now
  effectively runtime infrastructure

### Target Shape

Split the current sampling layer into two clearer strata:

1. runtime-safe payload structure and evidence helpers
2. tooling-only corpus iteration and sample loading

The runtime-safe layer should own:

- provider config resolution for schema-unit extraction
- payload-to-schema-unit extraction
- bundle-scope derivation
- exact-structure/profile token helpers needed at runtime
- cohort/profile identity helpers needed by durable artifact observations

The tooling layer should own:

- DB/session corpus reading
- sample iteration
- max-sample policies
- observation gathering workflows

### Main Modules

- `polylogue/schemas/sampling.py`
- `polylogue/schemas/runtime_registry.py`
- `polylogue/storage/artifact_observations.py`
- new runtime-safe helper modules under `polylogue/schemas/`

### Required Refactors

1. extract payload-structure and profile helpers from `sampling.py`
2. make runtime registry depend only on runtime-safe helper modules
3. make artifact observation inspection depend on the same runtime-safe helpers
4. leave corpus-reading and sample-iteration logic in tooling-only modules
5. keep package-resolution and cohort semantics unchanged

### Acceptance Criteria

- runtime registry no longer imports `sampling.py`
- artifact observation inspection no longer imports `sampling.py`
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
- field stats, semantic annotation, relation inference, clustering, package
  assembly, and persistence are not represented as explicit stages
- operator-facing result shaping is mixed into the same module

### Target Shape

Decompose schema generation into:

1. observation collection
2. field-stat and structure analysis
3. schema annotation
4. package assembly
5. manifest/package emission
6. top-level result reporting

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
5. reduce `schema_generation.py` to orchestration plus workflow entry

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
- showcase/demo/QA fixture generation therefore depends on a monolith
- generation reports are implicit rather than typed

### Target Shape

Split synthetic generation into:

1. package/schema selection
2. schema-driven value emission
3. provider conversation/stream/tree builders
4. wire-format serialization
5. style/theme overlays for showcase use
6. typed generation reports

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

## Step 4: Prepare/Source Transformation Convergence

### Goal

Make the raw-to-record path explicit and typed instead of split across broad
source-walking and preparation modules.

### Current Problems

- `pipeline/prepare.py` still mixes content canonicalization, ID assignment,
  DB-aware enrichment, attachment materialization, and persistence orchestration
- `sources/source.py` still mixes traversal, raw acquisition iteration, and
  parse-oriented walking
- the overall transformation from raw artifact to save-ready record bundle is
  not represented as a clean staged pipeline

### Target Shape

Decompose the ingest transformation into:

1. source walking and raw artifact discovery
2. parse/harmonize/content canonicalization
3. record derivation
4. DB-aware enrichment / stable-ID reconciliation
5. filesystem materialization plan
6. persistence handoff

### Main Modules

- `polylogue/pipeline/prepare.py`
- `polylogue/sources/source.py`
- `polylogue/pipeline/services/parsing.py`
- `polylogue/sources/emitter.py`

### Required Refactors

1. split `prepare.py` into explicit transform/enrichment/materialization stages
2. narrow `source.py` toward traversal/raw iteration only
3. keep the harmonized-content step explicit and reusable
4. make the save-ready bundle boundary typed and obvious
5. preserve current parse/save semantics while reducing responsibility overlap

### Acceptance Criteria

- `prepare.py` no longer acts as the single mixed implementation home for the
  whole raw-to-record path
- `source.py` is clearer about being traversal/acquisition infrastructure
- record derivation and DB-aware enrichment are distinct stages
- attachment materialization is explicit rather than hidden inside a broad
  preparation module

### Verification

- prepare pipeline tests
- parsing service tests
- source-law tests
- run/ingest workflow tests

## Step 5: Artifact Proof And Verification Workflow Convergence

### Goal

Turn durable artifact observations and schema verification into one clearer
evidence workflow.

### Current Problems

- `verification.py` still contains multiple proof/report modes in one large
  module
- `artifact_observations.py` both inspects artifacts and serves as a durable
  read-model workflow
- artifact proof, schema verification, and cohort listing are closely related
  but not yet staged as one typed workflow family

### Target Shape

Split the evidence workflow into:

1. raw artifact inspection
2. durable observation persistence
3. cohort/proof aggregation
4. schema verification over observed corpus
5. operator/report projections

### Main Modules

- `polylogue/schemas/verification.py`
- `polylogue/storage/artifact_observations.py`
- `polylogue/showcase/qa_runner.py`
- `polylogue/cli/commands/check.py`

### Required Refactors

1. separate inspection/persistence/aggregation concerns in artifact
   observations
2. separate schema verification from artifact-proof aggregation logic
3. define typed workflow results for proof and verification passes
4. keep QA/check/report surfaces consuming those typed results
5. preserve durable package/evidence truth across all projections

### Acceptance Criteria

- verification and artifact-proof flows are staged explicitly
- durable artifact observations remain the canonical read model
- proof/verification output becomes easier to extend without reaching across
  unrelated workflow layers
- QA/check/report continue to agree on package/cohort/proof truth

### Verification

- verification tests
- artifact proof/cohort tests
- check-command tests
- QA/report tests

## Step 6: Async SQLite Write-Path Narrowing

### Goal

Make the async backend look more like infrastructure and less like a second
application layer for the evidence pipeline.

### Current Problems

- `async_sqlite.py` still owns schema init, connection management, pool logic,
  bulk modes, transactions, and a broad set of write/business operations
- evidence-oriented flows such as raw storage, artifact observation upserts, and
  conversation persistence still rely on a fairly broad backend surface
- this keeps business semantics too close to the backend

### Target Shape

Narrow the backend toward:

1. connection lifecycle
2. transaction lifecycle
3. schema initialization
4. bulk/pool helpers
5. low-level write primitives only

Evidence/business workflows should move above that layer.

### Main Modules

- `polylogue/storage/backends/async_sqlite.py`
- `polylogue/storage/backends/queries/*.py`
- `polylogue/storage/artifact_observations.py`
- `polylogue/storage/repository_writes.py`
- `polylogue/pipeline/prepare.py`

### Required Refactors

1. identify and extract backend methods that are really workflow/business steps
2. move those steps upward into repository/evidence workflow layers
3. keep low-level write/query helpers on query modules where appropriate
4. preserve transaction and bulk-ingest semantics

### Acceptance Criteria

- backend reads as infrastructure plus low-level write support
- evidence and persistence workflows stop being encoded directly in backend
  methods
- bulk-ingest and transaction behavior remain explicit and tested

### Verification

- backend tests
- storage write tests
- ingest/prepare workflow tests
- artifact observation persistence tests

## Step 7: Schema/Evidence Operator Workflow Convergence

### Goal

Make schema and proof CLI surfaces bind onto typed workflows instead of driving
internal stages directly.

### Current Problems

- `cli/commands/schema.py` still knows too much about underlying generation and
  registry-tooling mechanics
- schema and proof surfaces are related but still not presented as one operator
  family over the evidence pipeline

### Target Shape

Introduce one schema/evidence workflow layer that can power:

- `polylogue schema generate`
- `polylogue schema list`
- `polylogue schema explain`
- `polylogue schema compare`
- `polylogue check --proof`
- related cohort/verification surfaces

with typed result objects.

### Main Modules

- `polylogue/cli/commands/schema.py`
- `polylogue/cli/commands/check.py`
- schema/evidence workflow modules under `polylogue/schemas/`

### Required Refactors

1. define typed operator/workflow result models for generation, explanation,
   verification, and proof
2. reduce CLI command modules to binding and rendering
3. keep package evidence, chronology, cohort, and promotion information exposed
   through the same workflow results
4. avoid new compatibility wrappers; move commands onto the new workflow
   directly

### Acceptance Criteria

- schema and proof CLI commands bind onto typed workflow results instead of
  driving internals directly
- package/cluster/evidence terminology remains consistent across commands
- operator output becomes easier to extend without reaching into deep internals

### Verification

- schema CLI tests
- check CLI tests
- schema operator workflow integration tests
- deterministic output tests where relevant

## Step 8: Named Roundtrip Proof Lane

### Goal

Turn the entire schema-and-evidence pipeline into a first-class provable loop.

### Current Problems

- synthetic generation, parser lowering, preparation, persistence, validation,
  and proof reporting are all individually strong
- but they are not yet framed as one named end-to-end convergence contract

### Target Shape

One explicit proof lane that demonstrates:

1. package authority selection
2. synthetic emission for a package element
3. parser/lowering roundtrip
4. prepare/enrichment/persistence roundtrip
5. validation against the resolved package element schema
6. durable artifact observation and proof/report visibility of the same truth

### Main Modules

- `polylogue/schemas/synthetic/*`
- `polylogue/pipeline/prepare.py`
- `polylogue/schemas/validator.py`
- `polylogue/storage/artifact_observations.py`
- `polylogue/schemas/verification.py`
- `polylogue/showcase/workspace.py`

### Required Refactors

1. define a named roundtrip proof workflow or fixture family
2. reuse the same package-selection truth through every step
3. make failures preserve stage-specific evidence, not generic mismatch output
4. expose the proof lane through tests and, where useful, QA/check/report
   surfaces

### Acceptance Criteria

- Polylogue has a first-class schema-and-evidence roundtrip proof
- package version/element truth survives synthetic emission, parsing,
  preparation, persistence, validation, and proof
- failures are inspectable by stage and provider

### Verification

- schema generation tests
- synthetic roundtrip tests
- parser/emitter/validator tests
- prepare/persist tests
- QA/showcase/proof tests

## Suggested Commit Decomposition

1. `refactor: extract runtime-safe schema observation helpers`
2. `refactor: decompose schema generation pipeline`
3. `refactor: split synthetic corpus generation stack`
4. `refactor: converge prepare and source transformation boundaries`
5. `refactor: converge artifact proof and verification workflows`
6. `refactor: narrow async sqlite evidence write path`
7. `refactor: converge schema and proof operator workflows`
8. `test: add schema and evidence roundtrip proof lane`

## Exit Criteria

- runtime schema authority and artifact inspection no longer depend on
  tooling-only sampling modules
- schema generation stages are explicit and typed
- synthetic generation has a clear package-selection and emission pipeline
- raw-to-record preparation is staged explicitly
- artifact proof and schema verification form one coherent workflow family
- async SQLite reads as infrastructure plus low-level write support
- schema/proof CLI surfaces bind onto typed workflows
- one named roundtrip proof lane demonstrates the whole evidence pipeline

## Deferred Only After This Program

- provider-parser local cleanup passes
- further site/report/publication work
- deeper feature cuts based on the clearer post-program architecture
