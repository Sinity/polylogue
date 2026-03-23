# Archive Intelligence Platform Convergence Program

Date: 2026-03-23
Status: active execution program
Role: canonical integrated implementation campaign after the latest semantic-query dogfooding wave

Absorbs and supersedes as the live execution queue:

- `state-and-schema-platform-convergence-program-2026-03-23.md`
- the current-open-frontier note in `../planning-and-analysis-map-2026-03-21.md`
- the still-live reservoir portions of:
  - `canonical-archive-platform-program-2026-03-19.md`
  - `refactoring-first-streamlining-program-2026-03-19.md`
  - `testing-reliability-expansion-program-2026-03-14.md`

Prerequisite executed programs:

- `semantic-stack-convergence-program-2026-03-23.md`
- `schema-and-evidence-pipeline-convergence-program-2026-03-23.md`
- `core-architecture-convergence-program-2026-03-23.md`
- `runtime-contract-and-validation-lanes-program-2026-03-22.md`
- `read-surface-proof-and-showcase-hardening-program-2026-03-22.md`
- `publication-control-plane-program-2026-03-22.md`
- `schema-package-authority-program-2026-03-22.md`

Primary local design/audit inputs:

- `../planning-and-analysis-map-2026-03-21.md`
- `.claude/scratch/027-architecture-review-2026-03-23.md`
- `.claude/scratch/019-polylogue-architecture-audit-2026-03-20.md`
- `.claude/scratch/018-wave0-schema-package-design.md`
- `.claude/scratch/026-schema-taxonomy-and-versioning.md`

## One-Line Goal

Turn Polylogue into an archive-intelligence platform whose runtime state,
action semantics, query surfaces, FTS/embedding retrieval, schema tooling,
and validation lanes all consume one deliberate canonical model stack.

## Why This Is The Right Integrated Next Phase

The latest code and live-archive dogfooding changed the shape of the problem.

What is already substantially converged:

- canonical query/spec/front-door routing
- schema package authority
- semantic facts and semantic proof contracts
- publication/report control plane
- runtime contract and validation-lane basics

What is still materially incomplete:

1. action semantics are still derived from low-level `ToolCall` viewports rather
   than a first-class action/event layer
2. query truth is now more honest, but still does too much on hydrated
   conversations instead of durable action-aware retrieval structures
3. FTS and embedding retrieval remain mostly message-text-centric even though
   modern archive value increasingly depends on tools, file paths, shell
   commands, plan-control events, and action sequences
4. the old state-and-schema platform mass is still real:
   raw/validate/prepare/write state bands, async backend width, schema tooling
   layering, and operator workflow convergence all remain unfinished
5. validation/performance lanes exist, but they do not yet explicitly prove the
   dogfooded retrieval/semantic stack on the real archive under memory pressure

So the right next campaign is no longer “state and schema only”, and it is not
“query or embeddings only”. It is one bigger convergence move across:

- action intelligence
- retrieval and ranking
- runtime state bands
- schema/operator tooling
- validation/performance dogfood lanes

## Program Thesis

Polylogue should have:

1. one canonical action/event layer above raw tool blocks
2. one query/retrieval contract that consumes those actions directly
3. one retrieval control plane spanning FTS, semantic stats, and embeddings
4. one typed raw -> validate -> parse -> prepare -> persist state ladder
5. one schema-tooling/operator stack that reads those same typed states
6. one named validation program proving the whole archive-intelligence loop on
   both fixtures and the live archive

## Program Inheritance Matrix

### 1. From `state-and-schema-platform-convergence-program-2026-03-23.md`

Carry forward as active execution work:

- runtime state-model normalization
- async backend write-surface narrowing
- raw/validation/prepare pipeline convergence
- schema tooling decomposition
- schema operator and roundtrip-proof convergence

### 2. From `canonical-archive-platform-program-2026-03-19.md`

Carry forward as architectural north-star constraints:

- database-first operational truth
- canonical internal languages
- explicit capability/evidence/execution surfaces
- fewer local dialects for semantics and retrieval

### 3. From `refactoring-first-streamlining-program-2026-03-19.md`

Carry forward as simplification rules:

- no new infrastructure without immediate convergence
- thin operators over typed workflows
- delete parallel shells rather than layering one more abstraction
- keep verification central

### 4. From `testing-reliability-expansion-program-2026-03-14.md`

Carry forward as the open validation reservoir:

- stronger machine/CLI contracts
- explicit slow/large/live validation lanes
- long-haul and archive-scale verification
- scenario-driven showcase/QA proving

### 5. From the latest semantic/query dogfooding work

Carry forward as new concrete requirements:

- action/tool/path query truth must be based on runtime canonical semantics
- “other” must keep shrinking through intentional taxonomy improvements
- query/FTS/embedding surfaces must become action-aware, not just message-aware
- later retrieval must stop re-interpreting raw tool blocks in parallel

## Non-Goals

This program is not:

- a new provider-parser expansion wave
- a renderer/site redesign
- a TUI redesign
- a docs-only cleanup pass
- a pure benchmark campaign detached from product semantics

Those may happen later. This campaign is specifically about archive
intelligence and the remaining platform core.

## Architectural Rules

### 1. `ToolCall` Stops Being The Highest Semantic Unit

`ToolCall` remains the raw viewport.
Downstream consumers should progressively move to canonical action/event facts.

### 2. SQL Semantic Filters May Narrow Candidates, But Runtime Facts Own Truth

If persisted semantic labels and canonical runtime facts disagree, runtime
facts win. Candidate narrowing must remain conservative relative to that rule.

### 3. FTS, Stats, Similarity, And Attribution Must Speak The Same Semantic Language

No separate ad hoc interpretation of tool usage in query output, embeddings,
work-event extraction, and attribution.

### 4. Backend Code Must Stay Infrastructure

`SQLiteBackend` remains infra. Retrieval semantics, ranking, and archive
business workflows belong in typed query/repository/platform layers.

### 5. Validation Lanes Must Prove Dogfooded Reality

The live archive is not just a demo target. It is now part of the design loop.
Verification must include archive-scale semantic/query checks, not only unit
fixtures.

## Execution Order

1. first-class action/event layer
2. query and retrieval contract convergence
3. FTS and embedding control-plane convergence
4. runtime state-band convergence
5. async backend narrowing and retrieval performance hardening
6. schema-tooling and operator-workflow convergence
7. named archive-intelligence validation lanes

## Executed Progress

Completed so far in the live codebase:

- first-class action events now replace the older action-fact seam as the
  canonical message-scoped semantic action layer
- semantic consumers now read that event layer directly:
  - conversation semantic facts
  - attribution
  - work-event extraction
  - query semantic matching
  - grouped action/tool stats
- ordered action-sequence querying is now available across CLI and MCP
- action-text querying now consumes normalized event retrieval text across CLI
  and MCP
- post-filter conversation retrieval now batches candidate hydration for
  date-sorted limited semantic queries, which removed the live `earlyoom`
  failure on `--action file_edit --stats-by tool --format json --limit 50`
- explicit retrieval lanes now exist for dialogue FTS, action-text retrieval,
  and hybrid retrieval through the canonical query surface
- archive stats, top-level summary output, health checks, and MCP stats now
  expose embedding coverage and pending-embedding counts
- grouped semantic stats now batch summary-backed hydration instead of broad
  speculative conversation loads, which materially reduced live RSS during
  `--stats-by action --format json` dogfooding
- `embed --stats` now has a machine-readable JSON contract
- named archive-intelligence validation lanes now exist for:
  - retrieval-dogfood
  - embeddings-coverage
  - schema-roundtrip
  - archive-intelligence
  - live-archive-small
  - live-archive-slow
  - memory-budget

Still open from this program:

- action-aware retrieval/FTS and embeddings convergence
- broader runtime state-band and backend narrowing work
- remaining schema-tooling/operator convergence

## Step 1: First-Class Action/Event Layer

### Goal

Introduce the canonical semantic layer above `ToolCall`.

### Current Problems

- `ToolCall` and `ActionFact` exist, but they are still too low-level and too
  message-local to be the main archive-intelligence primitive
- query grouping, attribution, work-event extraction, and semantic proof still
  reconstruct meaning from slightly different local views
- no durable archive-wide action/event read model exists

### Target Shape

Define a canonical action/event model with explicit kinds such as:

- file read / write / edit
- shell command
- git operation
- search / code navigation
- web fetch / browser context
- agent control / plan control
- subagent spawn
- unknown-but-explicit action

Each action/event should carry:

- conversation/message identity
- provider and tool identity
- canonical action kind
- normalized target fields:
  - paths
  - command
  - cwd
  - query
  - URL
  - branch
- evidence/backreferences to raw content blocks
- confidence or derivation source where useful

### Main Modules

- `polylogue/lib/action_facts.py`
- `polylogue/lib/semantic_facts.py`
- `polylogue/lib/viewports.py`
- `polylogue/pipeline/semantic.py`
- new storage/read-model support as needed

### Acceptance Criteria

- one canonical action/event model exists and is documented in code
- action grouping/querying/attribution/work-event extraction consume it
- new semantic classification improvements land in one place, not many

## Step 2: Query And Retrieval Contract Convergence

### Goal

Make query semantics action-aware, sequence-aware, and less message-text-only.

### Current Problems

- action/tool/path filters are now more truthful, but still operate over
  hydrated conversation scans
- grouped stats are better, but the query language still has no canonical
  higher-order action contract
- retrieval remains mostly conversation/message-centric instead of
  action-aware

### Target Shape

Extend the query contract to reason over canonical actions:

- action kind
- tool name
- path
- shell command text
- git command/branch
- search query text
- web target
- action co-occurrence and later sequence constraints where feasible

Representative target queries:

- sessions that edited a file and then ran tests
- sessions that only searched/read and never wrote
- sessions that touched a repo path without dialogue drift overwhelming ranking

### Main Modules

- `polylogue/lib/query_spec.py`
- `polylogue/lib/query_execution.py`
- `polylogue/storage/query_models.py`
- `polylogue/storage/backends/queries/*`
- `polylogue/cli/query_*`
- `polylogue/mcp/server_tools.py`

### Acceptance Criteria

- query semantics are expressed in terms of canonical action/event fields
- action-aware grouping and filtering do not require one-off output hacks
- machine output stays coherent across CLI and MCP

## Step 3: FTS And Embedding Control-Plane Convergence

### Goal

Make retrieval deliberate across dialogue text, action summaries, and semantic
similarity.

### Current Problems

- FTS still mainly indexes/render-ranks message text
- embeddings are honest when absent, but still not a first-class operational
  retrieval plane
- no archive-level contract says what is embedded, what is searchable, and why

### Target Shape

Define explicit retrieval surfaces such as:

1. dialogue text
2. action/event summaries
3. combined hybrid retrieval
4. archive-health visibility for embedding coverage and freshness

Potential structural moves:

- action/event summary text in FTS
- separate retrieval fields or lanes for dialogue vs action content
- explicit embedding manifest/coverage stats
- hybrid ranking that can use both textual and action-aware signals

### Main Modules

- `polylogue/storage/index.py`
- `polylogue/storage/fts_lifecycle.py`
- `polylogue/storage/search.py`
- `polylogue/storage/search_providers/*`
- `polylogue/cli/commands/embed.py`
- retrieval/report surfaces under `cli/`, `mcp/`, and `health_*`

### Acceptance Criteria

- FTS and embeddings have explicit, inspectable archive coverage
- retrieval can intentionally use action-aware text, not just message bodies
- operator surfaces can explain what retrieval lane was used

## Step 4: Runtime State-Band Convergence

### Goal

Finish the unresolved runtime state work from the prior planned program.

### Current Problems

- raw envelopes, validation results, prepared bundles, and durable records are
  still better than before but not yet one deliberate typed ladder
- later proof/operator code still knows too much about stage-local shapes

### Target Shape

Make the raw-to-persist path explicit:

1. acquired raw envelope
2. validated raw state
3. parsed conversation/message state
4. prepared write bundle
5. persisted durable records

### Main Modules

- `polylogue/lib/raw_payload.py`
- `polylogue/pipeline/services/validation.py`
- `polylogue/pipeline/services/parsing.py`
- `polylogue/pipeline/prepare*.py`
- `polylogue/storage/store.py`

### Acceptance Criteria

- state-band models are stage-named and unambiguous
- operator/report/proof code can consume them without bespoke reconstruction
- accidental overlap between stage-local models is materially reduced

## Step 5: Async Backend Narrowing And Retrieval Performance Hardening

### Goal

Keep the truth fixes while avoiding broad archive scans that do not earn their
cost.

### Current Problems

- truthful runtime semantic reconciliation can broaden candidate fetches
- some archive queries are now correct but slower than they should be
- the `earlyoom` history shows that some validation/test paths still need more
  explicit memory discipline

### Target Shape

Split the work into:

1. conservative candidate narrowing rules
2. backend infra slimming
3. archive-scale retrieval budgets and memory-safe validation lanes

### Main Modules

- `polylogue/storage/backends/async_sqlite.py`
- `polylogue/storage/backends/queries/*`
- `polylogue/storage/repository_*`
- `devtools/run_validation_lanes.py`
- performance/scale test modules

### Acceptance Criteria

- candidate narrowing rules are explicit and testable
- backend/business overlap keeps shrinking
- archive-scale query and validation runs have named budgets
- known memory blow-up paths are either fixed or isolated into explicit slow lanes

## Step 6: Schema Tooling And Operator Workflow Convergence

### Goal

Finish the other half of the old state-and-schema frontier under the new
archive-intelligence model.

### Current Problems

- schema tooling still behaves like a mini-platform
- operator schema commands still know more about internal workflow shaping than
  they should
- roundtrip proof should eventually consume the canonical state/action ladder

### Target Shape

Keep:

- runtime schema authority
- tooling analysis/assembly/promotion
- operator workflows

but make their boundaries thinner and more one-way, with operator commands
acting as shells over typed workflow requests/results.

### Main Modules

- `polylogue/schemas/runtime_registry.py`
- `polylogue/schemas/tooling_registry.py`
- `polylogue/schemas/generation_*`
- `polylogue/schemas/operator_workflow.py`
- `polylogue/schemas/roundtrip_proof.py`
- `polylogue/cli/commands/schema.py`

### Acceptance Criteria

- tooling layers are explicit
- operator commands are thin
- roundtrip proof consumes canonical typed state where available

## Step 7: Named Archive-Intelligence Validation Lanes

### Goal

Prove the full converged system on fixtures and the live archive.

### Lanes To Add Or Expand

- `archive-intelligence`
- `retrieval-dogfood`
- `embeddings-coverage`
- `schema-roundtrip`
- `live-archive-small`
- `live-archive-slow`
- `memory-budget`

### Required Coverage

- stale semantic labels upgraded by runtime facts
- action-aware query truth
- action-aware grouped stats
- FTS vs action-aware query interplay
- embedding coverage/freshness and absent-embedding operator behavior
- archive-scale performance budgets
- memory-safe execution under representative live-archive workloads

### Acceptance Criteria

- named validation lanes exist and are documented
- at least one lane proves retrieval semantics against the live archive
- the next dogfooding pass can rely on existing validation instead of ad hoc checking

## Suggested Implementation Wave Order

Wave 1:

- Step 1 basic action/event layer tightening
- Step 2 query-contract migration onto it

Wave 2:

- Step 3 retrieval/FTS/embedding control plane
- Step 5 candidate narrowing and performance hardening

Wave 3:

- Step 4 runtime state-band convergence
- Step 6 schema-tooling/operator convergence

Wave 4:

- Step 7 named validation lanes and live-archive proving

## Completion Criteria

This program is complete when:

- action/event semantics are first-class and shared
- query, stats, FTS, embeddings, attribution, and work-event extraction stop
  re-interpreting raw tool blocks independently
- runtime state and schema-tooling mass are both substantially reduced and more
  one-way
- archive-scale validation lanes prove the system on both fixture and live data

## Immediate First Cut

If starting from current `master`, begin with:

1. canonical action/event model expansion over the current `ActionFact`
2. query contract migration to action/event truth
3. explicit retrieval-surface design for FTS and embeddings

That is the sharpest next move exposed by the code and by live-archive
dogfooding.
