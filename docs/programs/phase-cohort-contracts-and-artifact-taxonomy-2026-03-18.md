# Phase Proposal: Cohort Contracts and Artifact Taxonomy

Date: 2026-03-18

## One-Line Goal

Replace the current one-schema-per-provider amalgam with heuristic, multi-version
schema inference that emits separate schema files for structurally distinct
provider variants, and back that with explicit artifact classification where
provider roots contain mixed file kinds.

## Why This Is The Next Broad Phase

Polylogue already has strong pieces:

- raw byte preservation
- schema inference
- schema clustering and promotion
- provider adapters
- synthetic generation
- generated showcase and QA artifacts

What it does not have is automatic version-aware schema generation inside the
inferencer itself.

Today, the project still splits truth across:

- inferred provider schemas
- baseline packaged schemas
- optional promoted registry versions
- provider-specific sampling config
- provider-specific typed adapters
- provider-specific synthetic wire-format fixups
- operator conventions about which raw files are real conversations vs metadata

That split is exactly why the "amalgamated schema" problem remains unsolved.
The project can infer one muddy provider-wide union, and it can separately
cluster shapes, but the generator does not naturally emit version-separated
schema artifacts from those findings.

## Restated Problem

The actual issue is narrower and more concrete than "we need a grand unified
contract layer".

Today, a provider like Claude Code or Codex is treated as though it has one
effective schema at generation time:

- the inferencer sees many samples
- it feeds them into one Genson merge
- structural fingerprints only limit duplicate contribution
- the output is still one amalgamated provider schema

What is wanted instead:

1. infer structurally distinct provider versions automatically
2. cluster them heuristically, not just by exact fingerprint identity
3. emit them as separate schema files
4. preserve a provider manifest that explains the cluster/version landscape
5. let validator, synthetic generation, and QA operate against those separate
   versions rather than a single provider-wide union

So the primary phase is really:

## Heuristic Provider Version Inference

The inferencer should stop being "provider -> one schema" and become:

- `provider -> many inferred version candidates`
- `version candidate -> one schema file`
- `provider -> manifest describing those version candidates`

## Core Thesis

The next phase should make the raw-to-proof pipeline look like this:

1. Acquire raw bytes.
2. Partition samples into heuristic version clusters.
3. Emit one schema per promoted cluster/version.
4. Parse only artifacts declared parseable by that contract.
5. Generate synthetic corpora from those same contracts.
6. Run showcase/QA against the same promoted cohort set.
7. Produce a coverage report proving what was supported, skipped, quarantined,
   or newly unknown.

If a structural variant is real, it should not be diluted into one provider-wide
union. It should become a version candidate with provenance.

## The Amalgamated Schema Problem, Solved Properly

The fix is not "infer a better union schema".

The fix is to make clustering/version inference part of schema generation
itself, and to split one provider schema space into three distinct artifacts:

### 1. Exact Version Schemas

Each structurally distinct shape becomes a version candidate with its own schema and
provenance:

- provider
- artifact kind
- cluster/version ID
- sample count
- first seen / last seen
- representative paths
- support status

These schemas are what validation and parser routing should use.

### 2. Provider Version Manifest

Each provider gets a manifest describing all known structural versions and how
they relate:

- promoted versions
- quarantined clusters
- ignored sidecars
- dominant versions
- compatibility notes
- parser entrypoint to use
- synthetic generation policy

This becomes the operator-facing truth for "what structural versions exist for
provider X".

### 3. Optional Amalgamated View

Keep a provider-level amalgamated schema only as an explanatory coverage view.
It may still be useful for:

- browsing
- quick drift summaries
- high-level documentation

But it must stop being the canonical validator/generator/runtime contract.

The amalgam is a report, not an executable authority.

## Heuristics First, Config Second

This phase should lean hard into heuristic inference.

Primary mechanisms should be:

- structural fingerprints
- shape similarity
- semantic annotation similarity
- dominant-key overlap
- field optionality/requiredness changes
- source-path and acquisition-window correlation
- record-type distributions for record-oriented providers

Configuration should exist, but as override/support:

- provider aliases
- path exclusions
- sidecar declarations
- clustering thresholds
- promotion policies

Config should not become a hidden second parser.

## New Foundational Layer: Artifact Taxonomy

Before parse, Polylogue should classify raw files and payloads into explicit
artifact kinds. Example families:

- `conversation_document`
- `conversation_record_stream`
- `subagent_conversation_stream`
- `session_index`
- `bridge_pointer`
- `agent_sidecar_meta`
- `tool_progress_only_stream`
- `unknown_runtime_artifact`

Each raw artifact should carry:

- `runtime_provider`
- `schema_provider`
- `artifact_kind`
- `cohort_id`
- `support_status`
- `parse_policy`
- `validation_policy`
- `classification_confidence`
- `classification_reason`

This is supporting machinery for the versioning problem, not a replacement for
it. It resolves the tension between "support everything we can obtain" and
"avoid hardcoded behavior":

- we support everything by classifying everything
- we avoid hardcoding by making classification and version manifests explicit,
  inspectable, and testable

## Why The Claude Subagent Structure Matters

The newer Claude Code `subagents/` layout is exactly the kind of shape that
breaks one-schema-per-provider inference.

Observed realities include:

- `agent-*.jsonl` conversation streams
- `agent-*.meta.json` sidecar metadata files
- extra top-level fields not present in older assumptions:
  - `slug`
  - `requestId`
  - `sourceToolAssistantUUID`
  - `toolUseID`
  - `parentToolUseID`

The crucial lesson is not just "add a few more fields".

The lesson is that provider roots now contain multiple artifact kinds, so the
inferencer must know which files contribute to versioned conversation schemas
and which are sidecars.

For this specific structure, phase-one expectations should be:

- `agent-*.jsonl` becomes a recognized `subagent_conversation_stream` cohort
- `agent-*.meta.json` becomes a recognized `agent_sidecar_meta` cohort
- sidecar metadata is archived and surfaced in coverage reports
- sidecar metadata does not emit empty conversations
- subagent streams retain linkage to parent session/tool invocation when known

## Desired Inferencer Behavior

For each provider, inference should become a pipeline like:

1. Load raw samples.
2. Classify artifact kind heuristically.
3. Discard or separately archive non-conversation sidecars from schema-building.
4. Build structural fingerprints for parseable artifacts.
5. Group fingerprints into heuristic clusters.
6. Generate one schema per cluster.
7. Compare those schemas for semantic closeness and possible version ancestry.
8. Emit:
   - `schemas/<provider>/clusters/<cluster-id>.schema.json.gz`
   - `schemas/<provider>/versions/vN.schema.json.gz`
   - `schemas/<provider>/manifest.json`
   - optional provider-level amalgam report

That is the concrete solution to the amalgamated-schema issue.

## Canonical Contract Shape

Each promoted contract should include more than raw JSON Schema.

Recommended contract contents:

- structural schema
- semantic annotations
- relational annotations
- parse strategy
- sample granularity
- artifact kind
- parseability flag
- preferred typed adapter
- synthetic generation hints
- drift classification rules
- known sidecar relationships

That means the contract becomes the one thing consumed by:

- acquisition/classification
- validation
- parser routing
- typed provider adapters
- synthetic generation
- showcase generation
- QA reporting

## Required Runtime Shifts

### A. Classification Before Parsing

Do not let parse heuristics decide artifact kind implicitly from filename
patterns and partial payload inspection alone.

Introduce a classifier stage that outputs a durable decision record.

### B. Version-Aware Validation

Validation should choose a promoted version schema, not a provider-wide latest
union.

If no cohort matches, the artifact becomes:

- `unknown`
- `quarantined`
- or `ignored sidecar`

with an explicit reason.

### C. Version-Aware Synthetic Generation

Synthetic generation should stop consuming only one provider `latest` union.

Instead it should support:

- generate from one specific promoted version
- generate a weighted mixed corpus across promoted versions
- generate unsupported/edge cohorts for negative tests

### D. Version-Aware QA

`polylogue qa` should gain a version matrix lane:

- one run per promoted version
- one mixed-provider run
- one unsupported-artifact expectation run
- one live drift scan against real raw corpus

The output should prove both capability and boundaries.

## Verification Model

This phase should make Polylogue self-evidently verifiable through four proof
surfaces.

### 1. Corpus Coverage Report

For a real corpus, emit a report showing:

- total raw artifacts
- classified artifacts
- promoted cohorts hit
- unsupported cohorts
- ignored sidecars
- unknown artifacts

The operator should be able to answer:
"What exactly do we know how to handle, and what is left?"

### 2. Version Roundtrip Matrix

For each promoted version:

- validate raw sample
- parse it
- normalize it
- render/query it
- regenerate synthetic examples
- re-parse synthetic examples

If any promoted version cannot complete that loop, it is not actually promoted.

### 3. Unsupported-Artifact Proofs

There should be explicit tests proving that unsupported or sidecar artifacts are
handled correctly:

- archived
- classified
- reported
- not misparsed as conversations

### 4. Drift Frontier Report

Live verification should show not just invalid records, but where the unknown
frontier is expanding:

- new artifact kinds
- new cohorts under known kinds
- old cohorts with structural drift

## Test Strategy Implications

This phase should retire a large amount of weak schema-adjacent testing and
replace it with stronger contract tests.

Good replacements:

- cohort fixtures derived from real classified artifacts
- parser crashlessness by actual provider/cohort contract
- classification laws
- unsupported-artifact tests
- QA matrix tests that assert report coverage

Tests that should become secondary:

- report-shape-only tests that never exercise real catalog entries
- CLI/schema workflow tests that patch registry objects but never run a real
  infer -> classify -> promote -> validate path

## Concrete Workstreams

### Workstream A: Raw Artifact Classification

- add artifact classifier
- store classification outcome durably
- distinguish parseable conversations from sidecars/index metadata

### Workstream B: Heuristic Versioned Registry

- persist inferred cluster schemas as first-class runtime artifacts
- persist provider composition manifests
- treat provider amalgams as reports only

### Workstream C: Runtime Contract Unification

- validator reads version contract
- synthetic generator reads version contract
- parser routing reads version contract
- audit reads version contract

### Workstream D: QA Matrix and Coverage Reporting

- add version matrix mode to `polylogue qa`
- emit coverage and drift-frontier artifacts
- prove sidecars are classified but not parsed

### Workstream E: Claude Runtime Shape Support

- add explicit handling for Claude subagent streams and sidecars
- preserve parent/child linkage
- surface new metadata fields through normalized/provider-meta layers where useful

## Exit Criteria

This phase is complete only when all of the following are true:

- no raw artifact reaches parse without explicit artifact classification
- no validator/synthetic/audit path quietly consumes a different schema source
- promoted contracts are version-native, not provider-wide unions
- unsupported artifacts appear in coverage reports instead of silently
  disappearing or becoming empty conversations
- Claude `agent-*.meta.json` style sidecars are recognized and do not materialize
  as empty subagent conversations
- every promoted version has an executable proof loop:
  validate -> parse -> normalize -> synthetic -> re-parse -> QA

## First Immediate Slice

If this phase starts now, the first slice should be:

1. Introduce raw artifact classification with an explicit `agent_sidecar_meta`
   kind for Claude subagent `.meta.json`.
2. Prevent those sidecars from producing conversations.
3. Add a coverage report that counts them.
4. Use that machinery to define the general version/contract model for all
   providers.

That first slice is small enough to land, but it forces the correct shape of
the whole phase.
