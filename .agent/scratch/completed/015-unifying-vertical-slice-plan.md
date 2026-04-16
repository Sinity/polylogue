# Unifying Vertical Slice Plan

## Status

- `2026-04-13`: first slice implemented for the action-event path.
- `2026-04-13`: second falsifier slice implemented for the raw-state planning path.
- `2026-04-13`: explicit artifact/dependency map introduced for the two proven paths.
- `2026-04-13`: sync/async schema bootstrap semantics unified behind one shared snapshot + extension plan.
- `2026-04-13`: control-plane semantic roots partially unified:
  - validation lanes, mutation campaigns, durable benchmark campaigns, and
    synthetic benchmark campaigns each now have dedicated authored catalog
    modules
  - showcase exercises and generated showcase families now root in
    `ExerciseScenario` instead of compiling directly to `Exercise`
  - runtime scenario coverage tracks declared operation targets, not just
    runtime artifact-path coverage
  - QA extra generated exercise families now come from one shared scenario
    selector instead of being named in parallel
  - the declared-operation coverage map is currently complete
  - operation lookup/resolution now has one shared catalog root instead of
    passive tuple exports plus repeated local name maps
- Concrete implementation landed in:
  - `polylogue/storage/action_event_artifacts.py`
  - `polylogue/storage/action_event_status.py`
  - `polylogue/storage/derived_status_products.py`
  - `polylogue/storage/derived_status.py`
  - `polylogue/storage/embedding_stats_support.py`
  - `polylogue/storage/repair.py`
  - `polylogue/storage/raw_ingest_artifacts.py`
  - `polylogue/pipeline/services/planning_backlog.py`
  - `polylogue/pipeline/services/planning_runtime.py`
  - `polylogue/artifact_graph.py`
  - `devtools/artifact_graph.py`
  - `polylogue/storage/backends/schema_bootstrap.py`
  - `polylogue/storage/backends/schema_upgrade.py`
  - `polylogue/storage/backends/async_sqlite_schema.py`
  - `polylogue/scenarios/metadata.py`
  - `polylogue/operations/specs.py`
  - `devtools/validation_catalog.py`
  - `devtools/mutation_catalog.py`
  - `devtools/mutation_scenario_catalog.py`
  - `devtools/benchmark_catalog.py`
  - `devtools/benchmark_scenario_catalog.py`
  - `devtools/synthetic_benchmark_catalog.py`
  - `devtools/scenario_projection_catalog.py`
  - `devtools/scenario_coverage.py`
  - `polylogue/showcase/catalog_loader.py`
  - `polylogue/showcase/exercises.py`
  - `polylogue/showcase/generators.py`
  - `polylogue/showcase/qa_runner_stages.py`
- The action-event path now has one shared semantic model for status, debt, repair,
  and retrieval-evidence stale accounting.
- The raw-state planning path now has one shared semantic model for backlog and
  force-reparse eligibility across SQL selection and scan-time planning.
- The two proven paths are now also named explicitly in one shared artifact map
  rather than remaining implicit in code and scratch notes only.
- The sync and async backend bootstrap paths now ask the same schema questions and
  apply the same current-version extension sequence instead of drifting as near-copies.
- The control-plane now has fewer competing semantic roots; authored catalogs are
  increasingly distinct from compiled/reporting views.
- Operation metadata is now closer to a real substrate: the graph and scenario
  layers resolve through shared catalogs rather than rebuilding their own
  transient lookup structures.
- This is enough evidence to treat the unifying approach as real rather than
  overfit to one path.

## Goal

Prove a unifying architecture on one path first, then validate that the model
generalizes by applying it to a second path with different failure modes.

This avoids both extremes:

- premature generic multipath infrastructure
- endless local fixes with no semantic center

## Stronger Interpretation Of "Unifying"

The elegant target is not a large family of peer registries.

The cleanest likely shape is:

- **Artifact graph** as the root of durable system state
- **Annotated operations** as the root of what the system can do
- **Scenarios** as authored proofs/benchmarks over artifacts + operations

This is slightly tighter than treating `CapabilitySpec` as a fully separate
authored root. In many cases, capabilities should be *derived* from annotated
operations rather than maintained in parallel.

So the likely final semantic layers are:

1. `ArtifactNode`
2. `OperationSpec`
3. `ScenarioSpec`

Where:

- artifact nodes describe state, dependency, freshness, and repairability
- operations describe transformations/probes over artifacts
- scenarios select corpus + operation + assertions + optional presentation

Everything else becomes a projection.

## Why This Is Better Than The Current Shape

Today Polylogue has several near-roots:

- pipeline stage models
- schema/operator verification requests
- showcase exercises
- validation lanes
- benchmark campaigns
- QA orchestration

Those are all useful, but they are peers. That is what creates drift.

The unifying move is:

- move semantic ownership into artifact graph + annotated operations
- compile exercises, QA slices, benchmark campaigns, docs, completions, and
  maps from those roots

## Root Types

### ArtifactNode

Represents one durable or derived state-bearing thing.

Suggested fields:

- `name`
- `kind`
- `layer` (`raw`, `parsed`, `derived`, `index`, `surface`)
- `depends_on`
- `freshness_rule`
- `probe`
- `repair_operation`
- `health_projection`
- `cost_tags`

Important point:

This must live close to archive/runtime semantics, not as docs metadata only.

### OperationSpec

Represents a real action, query, or probe.

Suggested fields:

- `name`
- `surface` (`polylogue`, `devtools`, `api`, `mcp`, internal)
- `mode` (`read`, `write`, `repair`, `benchmark`, `proof`)
- `consumes`
- `produces`
- `side_effects`
- `cli_entrypoint` or callable reference
- `progress_contract`
- `telemetry_contract`

Best implementation path:

- operations should be declared by annotation/registration on existing command
  or service functions where possible
- do not author a second full command ontology by hand

### ScenarioSpec

Represents something worth proving or measuring.

Suggested fields:

- `name`
- `corpus`
- `operation`
- `assertions`
- `presentation` (optional)
- `tags`
- `coverage_targets`

Compilation targets:

- showcase exercise
- QA slice
- validation-lane item
- benchmark scenario
- VHS/demo capture when useful

## First Vertical Slice: Action-Event Path

This is the best proof path because it already crosses:

- substrate semantics
- derived-model freshness
- FTS indexing
- doctor/health
- debt accounting
- repair planning

### Current Files

- `polylogue/storage/action_event_status.py`
- `polylogue/storage/derived_status_products.py`
- `polylogue/storage/repair.py`
- `polylogue/storage/derived_status.py`
- relevant tests under `tests/unit/storage/` and `tests/integration/`

### Artifact Graph For Path 1

#### Source artifact

- `tool_use_source_blocks`
  - source of truth: `content_blocks.type = 'tool_use'` joined to valid conversations
  - owns expected conversation/document counts

#### Derived read model

- `action_event_rows`
  - materialized rows derived from tool-use source blocks
  - freshness depends on:
    - source alignment
    - materializer version
    - orphan status

#### Derived index

- `action_event_fts`
  - depends only on `action_event_rows`
  - freshness depends on exact or probe-mode row/index alignment

#### Surface projections

- `doctor_action_event_health`
- `archive_debt_action_event`
- `repair_action_event_target`

These should become projections of the same artifact graph rather than separate
logic islands.

### Operations For Path 1

- `probe_action_event_rows`
- `probe_action_event_fts`
- `repair_action_event_rows`
- `repair_action_event_fts`
- `project_action_event_health`
- `project_action_event_debt`

The current code already implements these semantics piecemeal; the slice should
factor them into shared operations instead of adding another wrapper.

### Scenario Classes For Path 1

1. `action_event_converged`
- source, rows, and FTS all aligned

2. `action_event_missing_rows`
- source count exceeds materialized conversations/rows

3. `action_event_stale_rows`
- materializer version mismatch or stale rows

4. `action_event_fts_pending`
- rows exist, FTS behind

5. `action_event_fts_extra_rows`
- FTS contains stale extra rows beyond source read model

This last one is crucial: it is exactly the kind of drift that simple pending
count helpers miss.

### Assertions For Path 1

- health, debt, and repair preview agree on the same underlying state
- repair count equals graph-derived delta
- repair operation converges to healthy state
- probe-only and deep modes degrade explicitly but remain semantically aligned

### Projections To Build For Path 1

#### Proof/test projection

- one law-oriented test family over the scenario classes above
- not five bespoke tests with duplicated setup

#### Benchmark projection

- one repair benchmark scenario for action-event convergence
- records wall/RSS and changed-row counts

#### Exercise/QA projection

- optional, if there is still presentation value
- should be compiled, not hand-authored first

## Second Vertical Slice: Validation / Reparse Path

This is the falsifier for the model. It is intentionally different:

- raw-payload decoding
- schema validation
- parseability policy
- quarantine vs refresh
- backlog planning / reparse

### Current Files

- `polylogue/lib/raw_payload_decode.py`
- `polylogue/pipeline/services/planning_backlog.py`
- parse/validation service files
- `polylogue/cli/commands/run.py`

### Artifact Graph For Path 2

- `raw_payload`
- `validation_state`
- `parse_backlog`
- `parsed_conversations`
- `quarantine_reason`

### Operations For Path 2

- `decode_raw_payload`
- `sample_raw_payload`
- `validate_raw_payload`
- `collect_validation_backlog`
- `collect_parse_backlog`
- `reset_parse_status`
- `reparse_selected_raws`

### Scenario Classes For Path 2

1. `raw_parseable_clean`
2. `raw_parseable_with_tolerated_jsonl_irregularity`
3. `raw_truly_malformed`
4. `raw_stale_failed_validation_but_now_parseable`
5. `raw_force_reparse_scope_limited`

### Assertions For Path 2

- sample decode and stream decode agree on malformed-line classification where policy says they should
- parseable raws re-enter parse backlog after the right refresh operation
- truly malformed raws remain quarantined with preserved detail
- force-reparse scope stays selected, not global

## What Must Be Compiled, Not Re-authored

If the architecture is real, the following should be derived from artifacts +
operations + scenarios:

- health/debt detail strings
- repair preview counts
- benchmark scenario inventory
- validation-lane inventory
- showcase exercise inventory where relevant
- docs/control-plane inventories

If any of those still require parallel handwritten truth, the unification has
not gone far enough.

## Where This Should Live

### Runtime truth

Artifact and operation semantics should live near:

- `polylogue/storage/`
- `polylogue/pipeline/`
- `polylogue/operations/`
- `polylogue/schemas/`

### Control-plane compilation

Scenario compilation and mapping should live under:

- `devtools/`

### Presentation layers

These should consume compiled projections:

- `polylogue/showcase/`
- `polylogue/cli/commands/qa.py`
- docs renderers

## Maps To Generate From The Model

### 1. Artifact map

For each node:

- producer
- dependencies
- freshness rule
- repair operation

### 2. Operation map

For each operation:

- surface
- consumes
- produces
- side effects
- telemetry/progress contracts

### 3. Coverage map

For each scenario:

- covered artifacts
- covered operations
- proof projection
- benchmark projection
- canary/demo projection

### 4. Cost map

For each operation/scenario:

- wall
- RSS
- dominant scaling variable

## Implementation Order

### Phase 1

Implement the artifact graph + operation metadata for the action-event path only.

### Phase 2

Compile one scenario family from it into:

- tests
- one benchmark
- optionally one QA/exercise slice

Progress:

- done for generated showcase help/JSON-contract surfaces
- done for durable benchmark campaigns
- done for shared scenario metadata across showcase + benchmark + registry artifacts

### Phase 3

Apply the same architecture to the validation/reparse path.

Refined next seam:

- preserve the same model through QA/session/report/control-plane maps
- then test the model on a structurally different path:
  - `raw_payload -> validation_state -> parse_backlog -> parsed/quarantine`

Progress:

- control-plane map now includes explicit runtime operations for:
  - raw validation / parse backlog planning
  - action-event materialization / indexing / health projection
- the same artifact graph now describes both proven paths with:
  - nodes
  - curated paths
  - operations
- operations are now a real authored substrate in `polylogue/operations/specs.py`
- artifact graph consumes runtime operation specs instead of defining operations inline

### Phase 4

Only after both paths work, start multipath generalization.

Refined next choices:

- consume `OperationSpec` more directly from another control-plane surface
- or bind one authored scenario family to declared operation ids instead of freeform strings

Progress:

- the second benchmark root (`devtools/benchmark_campaigns.py`) now also uses authored scenarios instead of:
  - free registry dict
  - duplicated dispatcher match blocks
- remaining high-value follow-up on this seam:
  - synthetic benchmark scenarios now join the shared quality registry as a distinct category
  - generated quality reference now reflects both benchmark families

Refined next choices:

- decide whether operation ids should become first-class in authored scenario families instead of freeform `operation_targets`
- or consume the new operation substrate from another control-plane map, not just the artifact graph

## Success Criteria

The architecture is working if:

- the action-event path no longer has separately-authored truths for status,
  debt, and repair
- the validation/reparse path can reuse the same scenario machinery without
  awkward special-casing
- exercises/benchmarks/lanes start looking compiled rather than curated
- `devtools` can render useful maps from the same model

If path 1 works but path 2 requires a fundamentally different shape, the model
is overfit and should be corrected before generalization.
