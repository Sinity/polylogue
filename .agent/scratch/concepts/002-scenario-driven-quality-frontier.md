# Scenario-Driven Quality Frontier

> Supersedes `foamy-doodling-cerf.md`.
>
> The old plan was directionally useful but based on a false premise: polylogue is not "light on property tests". It already has a large law/property/fuzz/snapshot/mutation/benchmark/live-probe surface. The next wave should not be "add a `tests/property/` silo". It should be a coherent testing architecture that expresses more of the system with less handwritten test code.

## Context

Polylogue already has:

- a large Hypothesis footprint spread across `tests/unit`, `tests/property`, and `tests/integration`,
- rich strategy infrastructure in `tests/infra/strategies/`,
- synthetic corpus generation via `SyntheticCorpus`,
- real-pipeline probes via `devtools/pipeline_probe.py`,
- validation lanes through `devtools run-validation-lanes`,
- mutation campaigns through `devtools mutmut-campaign`,
- benchmark campaigns through `devtools benchmark-campaign`,
- golden/snapshot testing for CLI, rendering, and UI.

The next testing program should respect that existing control plane and extend it.

## Status

Still active as a forward plan.

The control-plane projection work that originally lived here has already landed.
The remaining frontier is the next testing wave driven by the mutation ledger:

- `providers-semantics`
- `repository`
- `filters`
- `source-detection`
- `drive-client`
- `ui-core`

Those fronts should drive the next law/testing wave.

---

## Strategic Shift

### Old framing

"Property-first testing" with more generated tests in `tests/property/`.

### New framing

**Scenario-driven quality frontier**:

1. declarative archive scenarios,
2. semantic oracles shared across surfaces,
3. stateful lifecycle testing,
4. mutation-guided law densification,
5. live-failure reduction into durable regressions,
6. performance/memory growth contracts.

This is broader, more correct, and more leverageful than just adding more Hypothesis files.

---

## Core Principles

1. **No new `tests/property` silo as the center of gravity**
   Keep tests near their subsystem. Use `tests/property/` only for genuinely cross-cutting suites.

2. **One scenario, many surfaces**
   The same generated or curated archive case should be usable against repository, facade, CLI, MCP, site, and product layers.

3. **Mutation-guided prioritization**
   Prioritize fronts with surviving mutants instead of chasing raw test-count inflation.

4. **Semantic oracles over string assertions**
   Compare normalized facts, not surface formatting, unless the formatting itself is the contract.

5. **Real-pipeline coverage where it matters**
   The major hole is not parser-only roundtrip. It is payload -> parse -> transform -> persist -> hydrate -> query/product/render.

6. **Failure capture must become reusable input**
   Live-archive and probe failures should become deterministic local regressions automatically.

---

## New Testing Aspects We Should Add

### 1. Stateful model-based testing

We currently have many local laws, but very little explicit state-machine coverage.

Add Hypothesis state machines for:

- archive lifecycle operations,
- maintenance preview/apply flows,
- Drive auth/retry/folder-resolution behavior,
- grouped source iteration / cursor update / replay behavior.

This covers whole-operation sequences that unit laws miss.

### 2. Cross-surface differential testing

The same semantic request should agree across:

- repository,
- facade,
- CLI JSON output,
- MCP tool output,
- site/search render surfaces,
- product/materialized read models.

These are not separate test domains. They are alternate views over the same archive semantics.

### 3. End-to-end persistence/hydration laws

Add laws that start from real provider payloads or synthetic corpus bytes and prove:

`payload -> parse -> transform_to_records -> save_bundle -> conversation_from_records -> query/product/render`

preserves the facts we care about.

### 4. Growth and complexity contracts

We already benchmark fixed points. We should also test **growth envelopes**:

- time across archive size tiers,
- memory across archive size tiers,
- query latency shape under increasing result-set size,
- pipeline stage shape under increasing raw batch sizes.

This is stronger than a single budget number.

### 5. Live-failure reduction

We already have archive-subset probes. We should add automatic reduction from:

- live archive subset,
- probe manifest,
- or failed scenario

into a minimal deterministic regression case under `tests/data/regressions/`.

### 6. Mutation-guided law synthesis

Mutation survivors already tell us where semantics are underspecified. Build the next wave around those fronts instead of inventing new "coverage categories".

## High-Leverage Abstractions

### A. `ArchiveScenario`

New file:

- `tests/infra/archive_scenarios.py`

Purpose:

- declarative archive case spec,
- supports seeding through builders or the real pipeline,
- can stage provider exports, grouped ZIPs, JSONL streams, archive subsets, or direct record graphs.

Shape:

```python
ArchiveScenario(
    name="chatgpt-minimal-roundtrip",
    input_mode="synthetic" | "builders" | "archive-subset",
    providers=("chatgpt",),
    source_shape="single-file" | "grouped-json" | "zip-mixed",
    count=3,
    seed=42,
    mutations=[...],
    expected=ScenarioExpectations(...),
)
```

This becomes the single source for many tests.

### B. `SemanticFacts`

New file:

- `tests/infra/semantic_facts.py`

Purpose:

Normalize conversations, query results, products, and rendered outputs into stable semantic facts:

- IDs
- message count
- role multiset
- title
- created/updated bounds
- tag set
- attachment count
- content-block kinds
- thread/tree closure
- product counts and chronology fields

This lets us test meaning instead of formatting.

### C. `SurfaceAdapter` + `SurfaceOracle`

New files:

- `tests/infra/surfaces.py`
- `tests/infra/oracles.py`

Purpose:

Expose the same scenario through:

- repository
- facade
- CLI
- MCP
- site
- product readers

and compare semantic facts.

This removes large amounts of repeated "invoke CLI / parse JSON / compare fields" boilerplate.

### D. `ArchiveStateMachine`

New file:

- `tests/infra/state_machines.py`

Purpose:

Model stateful sequences like:

- save bundle
- re-save modified bundle
- add/remove tag
- set/delete metadata
- delete conversation
- rebuild index
- run maintenance preview/apply
- verify search/list/tree/product visibility after each step

This is the right abstraction for repository/maintenance fronts.

### E. `RegressionCase` / `ProbeCapture`

New files:

- `tests/infra/regression_cases.py`
- `devtools/regression_capture.py`

Purpose:

Take failures from `pipeline_probe`, archive-subset runs, or scenario runs and persist them as:

- minimized manifest,
- durable fixture bundle,
- replayable test case metadata.

This closes the loop from live failure to local regression.

### F. `GrowthBudget`

New file:

- `tests/infra/growth_budgets.py`

Purpose:

Run the same scenario at tiered sizes and assert envelope properties:

- near-linear time bounds where expected,
- bounded RSS growth,
- no cliff regressions between tiers.

This complements benchmark campaigns instead of replacing them.

## Workstreams

## Branch 1: `feature/testing/archive-scenarios`

### Goal

Create the reusable scenario/oracle substrate that future testing work depends on.

### Commits

**1. `test: add archive scenario and semantic fact harness`**

New files:

- `tests/infra/archive_scenarios.py`
- `tests/infra/semantic_facts.py`
- `tests/infra/oracles.py`

Core capabilities:

- scenario seeding through builders or real pipeline,
- semantic fact extraction from conversations, query results, products, renders.

**2. `test: add surface adapters for repository, facade, cli, mcp, and site`**

New files:

- `tests/infra/surfaces.py`
- `tests/infra/query_cases.py`

**3. `refactor: port selected existing contract suites to scenario/oracle harness`**

Refactor a small number of representative suites first:

- query tool contracts,
- rendering preservation,
- one repository graph/view suite,
- one source/provider fidelity suite.

### Why first

Everything else gets cheaper once the repo has a common scenario/oracle language.

### Verification

```bash
nix develop -c pytest -q tests/unit/mcp tests/property/test_rendering_preservation.py
devtools run-validation-lanes --lane query-routing
devtools run-validation-lanes --lane archive-data-products
```

---

## Branch 2: `feature/testing/persistence-hydration-laws`

### Goal

Close the major end-to-end gap:

`payload -> parse -> transform -> save -> hydrate -> query -> product -> render`

### Commits

**1. `test: add provider payload to hydrated conversation roundtrip laws`**

New tests:

- `tests/unit/pipeline/test_roundtrip_hydration_laws.py`

Properties:

- message count preserved,
- role multiset preserved,
- title stability,
- parent/branch relations preserved,
- content block type / semantic_type preservation,
- attachment counts and ref-count invariants,
- timestamps remain parseable and ordered.

**2. `test: add grouped-source and emit-path equivalence laws`**

New tests:

- `tests/unit/sources/test_emit_path_laws.py`

Properties:

- grouped vs individual source shapes produce equivalent hydrated facts,
- ZIP-mixed inputs do not change provider ownership semantics,
- source splitting and save-bundle behavior agree.

**3. `test: add hydrated-conversation to product invariants`**

New tests:

- `tests/unit/storage/test_product_hydration_laws.py`

Properties:

- products reference real hydrated facts,
- chronology/product counts agree with source,
- product tiers consume the same canonical hydrated conversation facts.

### Why this matters

This is the highest-value addition relative to the old plan because it tests the actual durable archive path, not just parser-local behavior.

### Verification

```bash
nix develop -c pytest -q tests/unit/pipeline/test_roundtrip_hydration_laws.py tests/unit/storage/test_product_hydration_laws.py
devtools run-validation-lanes --lane source-provider-fidelity
devtools run-validation-lanes --lane domain-read-model-contracts
```

---

## Branch 3: `feature/testing/archive-state-machines`

### Goal

Add stateful model-based testing for archive lifecycle and maintenance behavior.

### Commits

**1. `test: add repository lifecycle state machine`**

New files:

- `tests/infra/state_machines.py`
- `tests/unit/storage/test_repository_state_machine.py`

Operations:

- save conversation/bundle
- re-save modified version
- add/remove tag
- set/delete metadata
- delete conversation
- rebuild index

Invariants:

- no dangling message/attachment refs
- list/search/view/tree remain internally consistent
- count agrees with list and summaries
- repeated idempotent operations converge
- independent operations commute where expected

**2. `test: add maintenance preview/apply state machine`**

New tests:

- `tests/unit/cli/test_maintenance_state_machine.py`

Operations:

- preview debt
- apply cleanup/repair
- rerun preview
- query affected products

Invariants:

- preview/apply agreement
- applied changes remove exactly the previewed debt
- no false-positive cleanup deletes
- maintenance surfaces stay machine-readable

**3. `test: add drive auth and retry state machine`**

New tests:

- `tests/unit/sources/test_drive_state_machine.py`

Operations:

- load credentials
- expired token
- refresh
- retry
- permission failure
- folder resolution

### Why this matters

This is the cleanest way to attack the repository and drive-client mutation fronts.

### Verification

```bash
nix develop -c pytest -q tests/unit/storage/test_repository_state_machine.py tests/unit/sources/test_drive_state_machine.py
devtools mutmut-campaign run repository
devtools mutmut-campaign run drive-client
```

---

## Branch 4: `feature/testing/search-and-retrieval-frontier`

### Goal

Densify the `filters`, `source-detection`, and retrieval-lane fronts using metamorphic and differential testing.

### Commits

**1. `test: add retrieval metamorphic laws`**

New tests:

- `tests/unit/cli/test_retrieval_metamorphic.py`

Properties:

- provider-filter results are subsets of unfiltered results,
- action lane results are subsets of hybrid results when action data is present,
- hybrid results contain every exact FTS hit above the same limit when vector data is absent,
- list/count/summary/search agree on the same query semantics,
- sort descriptions and pick output remain stable under semantically equivalent filter orderings.

**2. `test: add ranking differential laws`**

New tests:

- `tests/unit/storage/test_ranking_differential.py`

Differentials:

- hybrid search vs explicit fusion of backend fixtures,
- summary sort helpers vs repository ordering,
- filter picker output vs direct query-plan construction.

**3. `test: add source detection adversary matrix`**

New tests:

- `tests/unit/sources/test_source_adversary_matrix.py`

Cases:

- misleading names vs payload shape,
- mixed ZIP sources,
- foreign-provider blobs,
- grouped/individual ambiguity,
- malformed but provider-shaped envelopes.

### Why this matters

This work directly attacks the surviving `filters` and `source-detection` mutation fronts.

### Verification

```bash
nix develop -c pytest -q tests/unit/cli/test_retrieval_metamorphic.py tests/unit/sources/test_source_adversary_matrix.py
devtools mutmut-campaign run filters
devtools mutmut-campaign run source-detection
devtools benchmark-campaign run search-filters
```

---

## Branch 5: `feature/testing/providers-semantics-frontier`

### Goal

Make the semantic extraction front much harder to mutate and much easier to express.

### Commits

**1. `test: add semantic oracle coverage for unified adapter fronts`**

New tests:

- `tests/unit/sources/test_semantic_oracle_frontier.py`

Focus:

- `extract_content_blocks`
- `to_meta`
- fallback Claude Code extraction
- shared reasoning/tool/token/cost extraction
- provider viewport shaping

**2. `test: add foreign-provider rejection and ambiguity laws`**

New tests:

- `tests/unit/sources/test_foreign_provider_laws.py`

Properties:

- wrong-provider payloads never crash,
- detection ambiguity is explicit and bounded,
- parsers reject foreign shapes cleanly,
- fallback dispatch never "adopts" a foreign payload silently when shape evidence is absent.

**3. `fuzz: extend fuzz targets to harmonization and hydrators`**

Extend:

- `tests/fuzz/fuzz_json_parsers.py`
- add `tests/fuzz/fuzz_unified_semantics.py`
- add `tests/fuzz/fuzz_hydrators.py`

### Why this matters

The mutation baseline is explicit that this front is still under-specified despite good reach.

### Verification

```bash
nix develop -c pytest -q tests/unit/sources/test_semantic_oracle_frontier.py tests/unit/sources/test_foreign_provider_laws.py
devtools mutmut-campaign run providers-semantics
```

---

## Branch 6: `feature/testing/live-reduction`

### Goal

Unify the operator workflows and make live failures feed directly into durable local regressions.

### Commits

**1. `devtools: add regression capture for probe and live subset failures`**

New files:

- `devtools/regression_capture.py`
- `tests/infra/regression_cases.py`

Capabilities:

- archive-subset manifest capture,
- fixture bundle persistence,
- replay helper for local tests,
- shrink metadata and provenance.

### Why this matters

It is the "express more with less" move for the operator side of the testing system.

### Verification

```bash
nix develop -c pytest -q tests/unit/devtools
devtools run-validation-lanes --list
devtools mutmut-campaign list
devtools benchmark-campaign list
```

---

## Recommended Execution Order

1. `archive-scenarios`
2. `persistence-hydration-laws`
3. `archive-state-machines`
4. `search-and-retrieval-frontier`
5. `providers-semantics-frontier`
6. `live-reduction`

This order matters. Branch 1 creates the vocabulary; Branches 2-5 spend it; Branch 6 unifies the control plane.

---

## What This Plan Intentionally Does Not Do

- It does **not** pretend the repo lacks property tests.
- It does **not** center new work in `tests/property/`.
- It does **not** optimize for raw test-count growth.
- It does **not** duplicate existing test control planes with another registry.

---

## End State

When this program lands, polylogue should have:

- a scenario/oracle substrate shared across subsystem tests,
- state-machine coverage for archive and maintenance lifecycles,
- true end-to-end persistence/hydration laws,
- stronger semantic and source-detection fronts guided by mutation evidence,
- growth-budget checks in addition to fixed benchmarks,
- automatic reduction of live failures into local regressions,
- a unified quality registry projecting into validation lanes, mutation campaigns, and benchmarks.

That is a materially larger and more correct testing architecture than the original plan, while still subsuming its useful aims.
