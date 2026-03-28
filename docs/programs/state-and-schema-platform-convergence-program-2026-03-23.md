# State And Schema Platform Convergence Program

Date: 2026-03-23
Status: planned execution program
Role: canonical next implementation campaign after semantic-stack convergence

See also:

- `semantic-stack-convergence-program-2026-03-23.md`
- `schema-and-evidence-pipeline-convergence-program-2026-03-23.md`
- `core-architecture-convergence-program-2026-03-23.md`
- `canonical-archive-platform-program-2026-03-19.md`
- `refactoring-first-streamlining-program-2026-03-19.md`
- `../planning-and-analysis-map-2026-03-21.md`
- `.claude/scratch/027-architecture-review-2026-03-23.md`

## One-Line Goal

Make Polylogue's remaining runtime state and schema-tooling mass behave like
one intentional platform:

raw payload -> canonical state models -> validation/prepare/write pipeline ->
schema tooling/operator workflows -> roundtrip/audit proof.

## Why This Is Now The Main Frontier

The recent campaigns already closed the biggest cross-cutting truth surfaces:

- query execution and CLI front door
- package-root API shape
- artifact/cohort/proof control plane
- publication control plane and site decomposition
- schema package authority
- semantic stack convergence

What remains is no longer the broad semantic/product story. It is a remaining
internal platform cluster spread across two tightly related areas:

1. runtime state and persistence seams
2. schema tooling and schema-operator seams

The strongest live hotspots are now:

- `storage/backends/async_sqlite.py`
- `storage/store.py`
- `lib/raw_payload.py`
- `pipeline/services/validation.py`
- `pipeline/prepare.py`
- `schemas/runtime_registry.py`
- `schemas/tooling_registry.py`
- `schemas/generation_workflow.py`
- `schemas/generation_analysis.py`
- `schemas/roundtrip_proof.py`
- `cli/commands/schema.py`

This is the right next campaign because these modules now form the biggest
remaining mini-platform inside the repo: they carry a lot of business meaning,
operator behavior, and proof/report logic while still being partly split across
runtime models, backend primitives, tooling helpers, and command glue.

## Program Thesis

Polylogue should have:

1. one canonical set of runtime state models for raw envelopes, prepared write
   sets, validation outcomes, and durable operator artifacts
2. one async backend that acts like infrastructure rather than a second home
   for business workflows
3. one typed raw -> validate -> parse -> prepare -> persist pipeline contract
4. one schema tooling stack where analysis, assembly, emission, diff, and
   promotion are explicit layers
5. one operator workflow surface for schema generation, audit, compare,
   explain, promotion, and roundtrip proof
6. one named verification lane proving the full state/schema platform

## Non-Goals

This program is not:

- another semantic-proof expansion wave
- a renderer/site/template redesign
- a new provider parser campaign
- a TUI or MCP campaign
- a test-suite reshuffle for its own sake

Those may follow. This one is specifically about the remaining state and schema
platform mass.

## Architectural Rules

### 1. State Models Must Be Singular

If a piece of runtime truth exists in both:

- raw envelope helpers
- storage record models
- validation/prepare intermediate results

then the next iteration should collapse that to one canonical typed model or
one clearly staged translation boundary.

### 2. Backend Code Must Be Infrastructure

`SQLiteBackend` should own:

- connections
- transactions
- pooling/reuse
- low-level query primitives

It should not keep growing as a second workflow/orchestration surface.

### 3. Runtime Registry And Tooling Registry Must Stay One-Way

Runtime schema authority may be read by tooling.
Tooling must not re-expand into a second runtime-resolution path.

### 4. Operator Commands Must Be Thin

`polylogue schema ...` should become a shell over typed operator requests and
typed operator results, not a second place where generation/proof policy lives.

### 5. Roundtrip Proof Must Share The Same Typed Stage Shapes

Roundtrip proof should not keep building bespoke state snapshots if the runtime
pipeline already has typed stage outputs available.

## Execution Order

1. runtime state-model normalization
2. async backend write-surface narrowing
3. raw/validation/prepare pipeline convergence
4. schema tooling decomposition
5. schema operator and roundtrip-proof convergence
6. named state-and-schema verification lane

## Step 1: Runtime State-Model Normalization

### Goal

Define a smaller, clearer set of canonical typed models for runtime state.

### Current Problems

- `lib/raw_payload.py` defines decoded raw-envelope truth
- `storage/store.py` defines raw records, message/content records, run/publication
  artifacts, and other durable models
- `pipeline/services/validation.py` defines its own stage outcome shapes
- `pipeline/prepare.py` defines preparation/write-set shapes

These are all reasonable locally, but together they still encode overlapping
concepts in too many places.

### Target Shape

Separate the platform into explicit state bands:

1. raw payload envelopes
2. validation outcomes
3. prepared write bundles
4. durable storage records
5. durable operator artifacts (runs, publications, proof reports where relevant)

### Main Modules

- `polylogue/lib/raw_payload.py`
- `polylogue/storage/store.py`
- `polylogue/pipeline/services/validation.py`
- `polylogue/pipeline/prepare.py`

### Acceptance Criteria

- stage-local models are clearly named by stage and not reused ambiguously
- obvious overlap between raw envelope / validation / prepare state disappears
- operator artifacts are clearly separated from conversation/message/raw records

## Step 2: Async Backend Write-Surface Narrowing

### Goal

Reduce `SQLiteBackend` to infrastructure and move remaining business-write logic
to clearer repository/query/write surfaces.

### Current Problems

- `async_sqlite.py` is still one of the largest files in the repo
- it mixes connection lifecycle, schema initialization, pooling, low-level SQL,
  and some business-shaped operations
- the backend/query-store/repository split is better than before, but the write
  side is still broader than it should be

### Target Shape

Split the write platform into:

1. backend infrastructure
2. low-level query/write helpers
3. repository-owned business workflows

### Main Modules

- `polylogue/storage/backends/async_sqlite.py`
- `polylogue/storage/backends/queries/*`
- `polylogue/storage/repository_*`

### Acceptance Criteria

- backend responsibilities are connection/transaction/pool/schema-only
- business writes no longer hide in backend convenience methods
- repository write paths become more obviously canonical

## Step 3: Raw / Validation / Prepare Pipeline Convergence

### Goal

Make the raw-to-persist pipeline read like one staged typed contract instead of
loosely neighboring helpers.

### Current Problems

- validation still creates stage-local results with their own count/state shape
- prepare still mixes transformation, identity enrichment, and persistence prep
- roundtrip/operator workflows must know too much about the internal staging

### Target Shape

Define one explicit staged pipeline contract:

1. acquired raw envelope
2. validated raw record
3. parsed conversation batch
4. prepared write bundle
5. persisted result

### Main Modules

- `polylogue/pipeline/services/acquisition.py`
- `polylogue/pipeline/services/validation.py`
- `polylogue/pipeline/services/parsing.py`
- `polylogue/pipeline/prepare.py`

### Acceptance Criteria

- stage outputs are typed and composable
- roundtrip proof can consume them without bespoke reconstruction
- operator/report code stops depending on stage internals

## Step 4: Schema Tooling Decomposition

### Goal

Break the remaining schema tooling mini-platform into explicit layers.

### Current Problems

- `generation_analysis.py`, `generation_workflow.py`, `tooling_registry.py`,
  `semantic_inference.py`, and `audit.py` still form a dense tooling cluster
- `cli/commands/schema.py` exposes a wide operator surface over that cluster
- the generation/tooling side is still the strongest remaining “platform inside
  the platform”

### Target Shape

Split tooling into:

1. observation/selection inputs
2. cluster/profile analysis
3. package assembly and emission
4. schema diff/audit/explain/promotion tools
5. operator request/result workflows

### Main Modules

- `polylogue/schemas/generation_analysis.py`
- `polylogue/schemas/generation_workflow.py`
- `polylogue/schemas/tooling_registry.py`
- `polylogue/schemas/semantic_inference.py`
- `polylogue/schemas/audit.py`

### Acceptance Criteria

- generation analysis no longer doubles as operator workflow glue
- promotion/audit/diff concerns are clearly separate from package emission
- runtime registry remains the sole runtime authority

## Step 5: Schema Operator And Roundtrip-Proof Convergence

### Goal

Make schema CLI and roundtrip proof consume the same typed operator-stage
surfaces.

### Current Problems

- `cli/commands/schema.py` still contains a broad manual projection surface
- `roundtrip_proof.py` is a large bespoke workflow even though the runtime
  stages already exist
- proof, generation, compare, explain, and promote are closer than they should
  be to command/printing code

### Target Shape

Converge around:

1. typed operator request/result models
2. shared state/schema workflow helpers
3. thin schema CLI projection
4. roundtrip proof built from canonical stage outputs

### Main Modules

- `polylogue/cli/commands/schema.py`
- `polylogue/schemas/operator_workflow.py`
- `polylogue/schemas/roundtrip_proof.py`

### Acceptance Criteria

- schema CLI becomes thinner and more declarative
- roundtrip proof reuses canonical stage outputs where possible
- operator/report surfaces share typed result models instead of manual payload shaping

## Step 6: Named State-And-Schema Verification Lane

### Goal

Add one explicit verification lane for the state/schema platform itself.

### The Lane Should Prove

- raw envelope and provider/artifact identity helpers
- validation and prepare stage contracts
- schema generation/operator workflows
- roundtrip proof and related report projections
- backend/write-path invariants for the narrowed storage surface

### Completion Gate

This program is only complete when it has:

- focused unit coverage for state-model and backend narrowing
- focused workflow coverage for schema operator surfaces
- roundtrip-proof regression coverage over the converged stage contracts
- one named validation lane for the full state/schema platform

## Expected Outcome

After this program, Polylogue should have a much smaller remaining internal
platform story:

- runtime state has a clear staged model
- the async backend looks like infrastructure
- schema tooling is clearly a tooling stack rather than a diffuse subsystem
- schema CLI and roundtrip proof read from the same typed workflows
- the repo has a cleaner answer to “where does archive truth live, and where
  does schema/tooling truth live?”
