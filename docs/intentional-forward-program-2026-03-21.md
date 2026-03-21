# Polylogue Intentional Forward Program

Date: 2026-03-21

Collates and updates:

- `refactoring-first-streamlining-program-2026-03-19.md`
- `canonical-archive-platform-program-2026-03-19.md`
- `artifact-and-semantic-proof-program-2026-03-19.md`
- `artifact-and-semantic-proof-commit-plan-2026-03-19.md`

These remain useful design references, but this document is the current
forward-moving program.

## Purpose

Turn the broad March 19 planning wave into one intentional execution sequence
that matches the current codebase rather than the hoped-for future shape.

The guiding constraint is unchanged:

> make Polylogue as self-evidently verifiable and non-hardcode-y as possible,
> while still explicitly handling as much real data and metadata as possible.

## Current Read

Polylogue already has strong parts:

- a real raw-artifact and schema boundary
- a real repository/query core
- a real black-box verification harness
- a real outcome grammar for health and audits

The next blockers are not missing subsystems. They are mismatched top-level
contracts:

1. execution semantics are not fully honest or typed enough
2. QA composition is still manually assembled from parallel result shapes
3. `sources/source.py` still makes the ingest boundary look muddier than it is
4. proof/report expansion should come after those seams are straighter, not
   before

## Program Order

### Step 1: Honest Execution Contract

Goal:

- make run/watch/operator semantics honest about new versus changed archive
  activity

Why first:

- the current CLI/watch contract overclaims "new conversations"
- more proof and QA reporting should not be layered on top of a muddy execution
  contract

Scope:

- pipeline run drift calculation
- run result/operator formatting
- watch/notify/webhook/exec semantics
- targeted docs/tests

Status:

- started and executed in this working pass

### Step 2: Typed QA Composition

Goal:

- keep the verification workspace and black-box exercise model
- replace ad hoc QA result flattening with a typed composition of:
  - audit outcomes
  - showcase exercise results
  - invariant outcomes
  - artifact manifests

Why second:

- `showcase` is valuable
- `QAResult` is currently the clearest top-level composition seam that still
  hand-assembles truth

Scope:

- `showcase/qa_runner.py`
- `showcase/report.py`
- CLI `qa`

Primary deletions:

- lossy `audit_report.to_json()` storage at the QA core
- repeated field projection logic across QA report outputs

Status:

- executed in this working pass

### Step 3: Source Boundary Collapse

Goal:

- make `dispatch.py` the explicit parse authority
- demote or split `sources/source.py` so it stops pretending to own decode,
  dispatch, walking, and storage-adjacent bundle concepts at once

Why third:

- runtime architecture is already cleaner than the package surface suggests
- this is now a packaging and contract problem more than a parser rewrite

Scope:

- `sources/source.py`
- `sources/dispatch.py`
- `sources/decoders.py`
- `sources/emitter.py`
- `pipeline/prepare.py`

Primary deletions:

- wrapper-only parse helpers
- runtime monkeypatching at the source boundary
- storage-owned concepts exported from `sources`

### Step 4: Proof And QA Surface Integration

Goal:

- expose artifact/cohort/proof surfaces only after execution and QA contracts
  are straighter

Why here:

- the proof program is still correct in direction
- but operator-visible proof should consume cleaned-up execution/QA truths, not
  parallel ad hoc ones

Scope:

- `check --artifacts`
- `check --cohorts`
- `check --proof`
- QA report inclusion of proof/support coverage
- Claude `subagents/` sidecar linkage proofing

### Step 5: Publication And Repo Shape Cleanup

Goal:

- keep `site` as a repository/read-model consumer
- slim its internal monolith where useful
- reduce repo-topology overwhelm

Scope:

- `site/builder.py`
- docs tree structure
- managed artifact placement
- stale demo-era references

## Architectural Rules

### 1. Do Not Force One Universal Report Type

Polylogue has at least three real report categories:

- execution reports
- check/audit reports
- measurement reports

Unify their interfaces and composition rules, not necessarily their concrete
classes.

### 2. Do Not Expand Proof Surfaces Ahead Of Core Truth

Execution truth first, then QA composition, then proof integration.

### 3. Keep Verification As A First-Class Product Surface

`showcase` is not optional fluff. It is part of how Polylogue proves itself.

### 4. Prefer Shared Projections Over Repeated Field Flattening

If one result needs to be rendered as text, JSON, Markdown, and manifest data,
the shared projection step should be typed and singular.

## Exit Criteria For This Program

- run/watch/operator semantics distinguish new and changed archive activity
- QA has one typed top-level session contract instead of manual flattening
- source parsing ownership is explicit and package boundaries match runtime
- proof surfaces consume shared underlying facts rather than parallel local
  summaries
- the repo reads more like one coherent archive platform and less like several
  strong subsystems loosely stacked together
