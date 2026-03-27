# Source Boundary And Runtime Governance Program

Date: 2026-03-23
Status: executed
Role: executed convergence program for source/provider boundary cleanup, runtime maintenance governance, derived-model provenance/freshness control-plane work, and live-archive validation

Absorbs and extends as the live queue:

- `platform-wide-architecture-and-refactoring-program-2026-03-23.md`
- the still-relevant runtime-maintenance reservoir from:
  - `testing-reliability-expansion-program-2026-03-14.md`
  - `canonical-archive-platform-program-2026-03-19.md`
  - `refactoring-first-streamlining-program-2026-03-19.md`

Prerequisite executed programs:

- `platform-wide-architecture-and-refactoring-program-2026-03-23.md`
- `archive-intelligence-platform-convergence-program-2026-03-23.md`
- `semantic-stack-convergence-program-2026-03-23.md`
- `schema-and-evidence-pipeline-convergence-program-2026-03-23.md`
- `core-architecture-convergence-program-2026-03-23.md`

Primary design inputs:

- `../planning-and-analysis-map-2026-03-21.md`
- `../../.claude/scratch/027-architecture-review-2026-03-23.md`
- live-archive dogfooding findings from the 2026-03-23 refactor wave

## One-Line Goal

Turn Polylogue from a mostly-converged archive platform into a cleaner
long-lived live-archive system by finishing source/provider boundary cleanup,
making maintenance/repair/provenance first-class runtime surfaces, and
shrinking the remaining broad runtime/output clusters.

## Execution Summary

This program is now executed.

The main outcomes are:

- source traversal, parsed iteration, raw acquisition, and Drive runtime access
  now live in explicit modules instead of the old `sources/source.py` /
  `sources/drive_client.py` umbrellas
- runtime maintenance is now typed and explicit across health, cache
  provenance, derived-model readiness, maintenance selection, and publication
  summaries
- safe maintenance and destructive cleanup are now separated in operator
  workflows, including preview-only machine output
- live/archive and cached/archive semantics are explicit in operator output and
  MCP payloads
- named local and live governance validation lanes now cover source/provider
  fidelity, maintenance control-plane behavior, and live maintenance-preview
  memory budgets

## Why This Is The Right Next Campaign

The last broad campaign removed a lot of wrapper drag, but the real archive now
shows the next bottlenecks clearly:

1. source/provider/Drive boundaries still carry historical compatibility shape
   and mixed responsibilities
2. runtime maintenance is stronger than before, but still spread across health,
   repair, lifecycle, cached checks, and operator entrypoints
3. derived models such as action events, FTS, embeddings, and publication
   manifests are now explicit, but their freshness/provenance/repair story is
   not yet one coherent control plane
4. some result surfaces still mix live truth, cached truth, and per-surface
   projection policy
5. the live archive still contains older corruption artifacts
   (`orphaned_content_blocks`) that should be governed intentionally instead of
   being left as archaeology

## Program Thesis

The next broad improvement is not “more features”. It is operational
architecture:

1. source boundaries should be explicit and dogfoodable
2. runtime maintenance should be a first-class control plane, not scattered
   helper functions
3. derived read models should expose provenance, readiness, and repair through
   one consistent grammar
4. output surfaces should clearly distinguish live truth from cached truth
5. live-archive health should be something Polylogue can explain and repair,
   not just report

## Architectural Rules

### 1. Source Boundaries Must Be Real

Walking, fetching, decoding, parsing, replay, and provider adaptation must have
clear ownership. No compatibility umbrellas that still hide the real runtime
shape.

### 2. Maintenance Must Be Typed And Reused

Health, repair, rebuild, and cache-refresh operations should share typed
results and lifecycle helpers instead of each surface deciding repair semantics
locally.

### 3. Derived Models Need Provenance

If Polylogue persists or derives a read model, it should also know:

- what version/materializer produced it
- whether it is fresh
- what is missing or orphaned
- how it can be repaired

### 4. Cached Output Must Never Masquerade As Live Truth

Cached health/publication/proof data is fine, but operator surfaces must make
cache age and live-vs-cached semantics explicit and consistent.

### 5. Live Dogfooding Governs Design

Every major slice should be proven both in focused tests and against the actual
archive used day to day.

## Phase 1: Source Boundary Convergence

### Goal

Finish turning ingestion/source handling into clear one-way layers.

### Main Work

- further collapse or delete compatibility shells around source walking and
  Drive access
- separate source discovery, remote acquisition, decoding, parser dispatch, and
  replay fallbacks more explicitly
- revisit provider-specific rough edges that real archive use has exposed:
  Gemini chronology/attachments/title derivation, Codex chronology, Claude
  adjunct handling, and similar source-limited heuristics
- make source/provider boundaries easier to query and validate directly

### Acceptance Criteria

- source traversal no longer masquerades as parser/runtime authority
- provider-specific fallbacks are localized and testable
- source-layer ownership is clearer in imports and file layout

## Phase 2: Runtime Maintenance And Repair Control Plane

### Goal

Make runtime maintenance a deliberate control plane instead of a scattered set
of repairs.

### Main Work

- converge health, repair, lifecycle, and maintenance status around typed
  runtime-maintenance results
- add explicit maintenance coverage for currently exposed live issues such as
  orphaned content blocks and stale derived rows
- decide which repairs are safe derived-data repairs versus destructive archive
  cleanup, and encode that distinction in operator workflows
- reduce hidden cache behavior in maintenance/health entrypoints

### Acceptance Criteria

- maintenance outcomes have one coherent result grammar
- safe derived-data repairs are easy to run and verify
- destructive cleanup is explicit rather than accidental

## Phase 3: Derived-Model Provenance And Freshness Convergence

### Goal

Unify readiness/freshness/provenance semantics for action events, FTS,
embeddings, publications, and similar derived surfaces.

### Main Work

- define a shared provenance/freshness vocabulary for durable derived models
- keep materializer versions, freshness, pending work, and repair candidates
  visible through one control-plane story
- reduce per-subsystem status logic drift

### Acceptance Criteria

- derived models can explain readiness and staleness uniformly
- health and operator surfaces stop inventing one-off status semantics
- live archive state is easier to reason about and debug

## Phase 4: Operator Output And Cache Semantics Convergence

### Goal

Make operator surfaces honest and consistent about live results, cached
results, and machine-readable contracts.

### Main Work

- converge `check`, publication, QA/report, and similar output surfaces on
  clearer live-vs-cached result semantics
- reduce surface-local JSON shaping where one shared workflow result can exist
- ensure cached reads remain clearly marked and refreshable

### Acceptance Criteria

- machine output and human output derive from one canonical result per outcome
- cached surfaces do not hide stale state
- operator workflows remain scriptable and explainable

## Phase 5: Rendering, Export, And Publication Runtime Narrowing

### Goal

Further shrink the still-broad output/runtime surfaces that sit downstream of
core archive truth.

### Main Work

- keep narrowing rendering/export/publication modules toward formatting and
  materialization rather than local business logic
- reuse canonical semantic/action/query/maintenance surfaces instead of
  re-deriving archive facts inside output code
- continue deleting broad forwarding or convenience modules that no longer earn
  their keep

### Acceptance Criteria

- output modules read canonical runtime state instead of rebuilding it
- module topology around rendering/export/publication is easier to audit

## Phase 6: Live-Archive Governance Lanes

### Goal

Make live-archive maintenance and source/provider correctness part of committed
validation, not ad hoc operator work.

### Main Work

- add named validation lanes for live-archive maintenance, source/provider
  fidelity, and maintenance memory budgets
- capture the real archive corruption/freshness classes we have already seen:
  orphan descendants, cached-health staleness, read-model readiness drift, FTS
  lag, and similar issues
- dogfood source/query/maintenance flows against the real archive after each
  major slice

### Acceptance Criteria

- live governance scenarios are reproducible
- resource budgets remain explicit
- regressions reappear in committed lanes rather than one-off operator memory

## Execution Order

1. source boundary convergence
2. runtime maintenance and repair control plane
3. derived-model provenance and freshness convergence
4. operator output and cache semantics convergence
5. rendering/export/publication runtime narrowing
6. live-archive governance lanes

## First Concrete Slice

Start with source boundary and maintenance control-plane work together:

- tighten the source/Drive/runtime boundary
- add explicit maintenance results around health/repair/cache/read-model status
- make the live archive’s orphan/readiness/freshness issues explainable and
  governable without local archaeology
