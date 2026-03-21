# Polylogue Canonical Archive Platform Program

Date: 2026-03-19
Status: strategic reference, not the live execution queue
Role: broad north-star architecture program

Current execution entrypoint:

- `intentional-forward-program-2026-03-21.md`
- `planning-and-analysis-map-2026-03-21.md`

Supersedes, at the program level:

- `artifact-and-semantic-proof-program-2026-03-19.md`
- `artifact-and-semantic-proof-commit-plan-2026-03-19.md`

Those documents remain useful as narrower design slices, but this document is
the broader, integrated architecture program.

## Purpose

This document defines the next major unifying program for Polylogue:
turning it from a collection of strong subsystems into a more coherent archive
platform with a small number of canonical internal languages, a database-first
control plane, explicit support boundaries, and measurable semantic
preservation.

The goal is not just "better proofs" or "less hardcoding". It is a better
integrated system.

## One-Line Goal

Make Polylogue a contract-driven archive platform whose core truths live in the
database, whose surfaces all consume the same internal models, and whose
support, semantics, decisions, outputs, and drift are all explicit,
inspectable, and readable.

## Restated Ideal

The ideal is:

> the project as self-evidently verifiable and non-hardcode-y as we can make
> it, while still completely supporting, understanding, and handling as much
> data and metadata as we can obtain.

This has three important consequences:

1. "less hardcoded" does **not** mean "no provider-specific knowledge"
2. "self-evidently verifiable" does **not** mean only adding reports
3. "support as much as possible" requires explicit handling of unknown,
   sidecar-only, metadata-only, and partially-supported artifacts

It also has a fourth consequence:

4. "execution should be understandable" means Polylogue must expose the
   important decisions it made, why it made them, what evidence it used, and
   what outcome followed

The way to optimize all three at once is:

- keep provider-specific knowledge at explicit wire-format boundaries
- make intermediate truths canonical and typed
- store operational truth in the database
- record important decisions and outcomes as structured, readable evidence
- use reports as one consumer of that truth, not the truth itself

## Core Thesis

Polylogue should center itself on a few canonical internal languages:

- `RuntimeScope`
- `CapabilityDescriptor`
- `ArtifactIdentity`
- `DecisionRecord`
- `CohortManifest`
- `SemanticFactSet`
- `QueryIntent`
- `ProjectionSpec`
- `ExecutionTrace`
- `VerificationResult`
- `RunBundle`

Everything else should either:

- produce one of those
- consume one of those
- or be deleted as a local dialect once those become canonical

This is the broadest cohesion move available.

## Architectural Principle: Database First, Filesystem Deliberate

Many "managed data products" should indeed live in the database.

The clean split is:

### 1. Database = Operational Truth

Use SQLite for structured, queryable, comparable facts:

- raw artifact ledger
- artifact classifications and support states
- cohort manifests and promotion state
- capability registry state
- semantic facts and loss summaries
- decision records and execution traces
- run records and drift deltas
- verification/audit/proof results
- product manifests and freshness state
- linkage graphs (sidecars, subagents, attachments, branches)

### 2. Filesystem = Heavyweight Or User-Facing Materialization

Keep on disk:

- rendered markdown/html trees
- static site output
- exported conversation files
- media/asset blobs
- capture artifacts (VHS/GIF)
- optionally large archived proof bundles for human inspection

But these should be described by database records, not treated as the sole
source of truth.

### 3. Repository Files = Reviewed Baselines

Keep in the git repo:

- committed provider schemas
- migration definitions
- templates
- exercise catalogs
- benchmark definitions
- pinned design docs

These are reviewed baselines, not runtime state.

## Implication For Existing Code

Several existing filesystem products should become DB-first or DB-backed:

- `health.json` should become a cached DB-resident verification report with
  optional export
- showcase manifests and QA summaries should be stored as run-scoped DB records
  with optional JSON/Markdown emission
- artifact/proof manifests should live in DB first, with export as a projection
- run metadata already has a natural home in the existing `runs` table and
  should grow rather than fragment into ad hoc files

## What Should Remain File-Backed

Not everything should move into SQLite.

Avoid storing large rendered site trees, captures, or long-lived user-facing
export bundles as giant DB blobs unless there is a specific reason. Instead:

- store the materialized files on disk
- store their manifest, lineage, hash, size, and freshness metadata in DB

That gives both operational clarity and practical storage behavior.

## Current Strengths To Build On

The codebase already contains the seeds of this architecture:

- `RawPayloadEnvelope` in `polylogue/lib/raw_payload.py`
- `ConversationQuerySpec` in `polylogue/lib/query_spec.py`
- `RuntimeServices` in `polylogue/services.py`
- `HealthCheck` / `HealthReport` in `polylogue/health.py`
- `CheckResult` / `AuditReport` in `polylogue/schemas/audit.py`
- `SchemaCluster` / `ClusterManifest` in `polylogue/schemas/registry.py`
- `RunRecord` and existing `runs` table in storage
- provider-agnostic semantic helpers in `polylogue/lib/provider_semantics.py`
- `PipelineMetrics` and run observers
- schema explain/redaction reporting with explicit evidence annotations

The integration program should reuse and widen those, not replace them.

## Program Outcomes

When this program is complete, Polylogue should have:

- a database-first operational control plane
- explicit capability descriptors for providers and sources
- artifact and semantic truth represented as canonical internal models
- readable decision and execution narratives for major operations
- a unified relationship graph across conversations, branches, sidecars,
  subagents, attachments, and tools
- CLI, MCP, facade, site, and export surfaces all driven by shared operation
  and projection layers
- verification/proof/drift systems that consume the same underlying facts
- fewer local special-case dialects and less duplicated interpretation logic

## Pillar 1: Canonical Internal Languages

## Goal

Define and enforce the small set of core typed models that all major subsystems
must share.

## Main Changes

Formalize a narrow set of canonical models:

- `RuntimeScope`
- `CapabilityDescriptor`
- `ArtifactIdentity`
- `DecisionRecord`
- `CohortManifest`
- `SemanticFactSet`
- `QueryIntent`
- `ProjectionSpec`
- `ExecutionTrace`
- `VerificationResult`
- `RunBundle`

## Why This Matters

Polylogue currently has several half-generalized abstractions. The system will
cohere more if these stop being local helpers and become the actual backbone.

## Existing Seeds

- `ConversationQuerySpec`
- `RuntimeServices`
- `RawPayloadEnvelope`
- `HealthReport`
- `AuditReport`
- `ClusterManifest`

## Exit Criteria

- new cross-cutting features are expected to produce/consume canonical models
- surface-specific mini-languages stop proliferating

## Pillar 2: Capability Registry

## Goal

Make provider/source-specific knowledge explicit, inspectable, and bounded.

## Main Changes

Introduce or formalize capability descriptors covering:

- acquisition mode
- provider aliases and canonical schema provider
- artifact hints
- parser binding
- schema/cohort support
- synthetic generation support
- semantic extraction policy
- preservation/export policy

## Why This Matters

Provider knowledge currently exists across:

- parser modules
- detection logic
- source discovery
- schema registry
- synthetic wire formats
- semantic helpers

That spread is manageable only if it is deliberate and structured.

## Exit Criteria

- provider-specific logic is mostly discoverable through explicit descriptors
- remaining wire-format hardcoding is easy to locate and justify

## Pillar 3: Database-Resident Control Plane

## Goal

Move operational truth into SQLite and treat files as materializations rather
than primary state where appropriate.

## Main Changes

Expand the DB-backed state model to include:

- artifact ledger
- support/policy status
- cohort manifests and promotion outcomes
- decision records and execution traces
- verification/proof/audit results
- run lineage and drift summaries
- product manifests and freshness records
- linkage graph facts

## Product Storage Policy

### Store primarily in DB

- small/structured facts
- manifests
- summaries
- status/state transitions
- provenance
- hashes
- drift deltas

### Store primarily on filesystem, indexed in DB

- render trees
- site output
- exports
- large captures
- large binary assets

## Existing Seeds

- `RawConversationRecord`
- `RunRecord`
- `runs` table
- health cache logic
- showcase manifest generation

## Exit Criteria

- operational tools query DB first for state
- filesystem products have DB-backed manifests/provenance
- ad hoc JSON cache/state files are minimized

## Pillar 4: Decision Observability And Execution Narrative

## Goal

Make major Polylogue operations understandable in human-readable and
machine-readable ways by recording important decisions, evidence, and outcomes
as first-class data.

## Main Changes

Introduce explicit observability records such as:

- `DecisionRecord`
- `ExecutionTrace`
- `StageNarrative`

Each important decision should be able to express:

- subject:
  - run
  - source
  - artifact
  - cohort
  - schema field
  - conversation
  - projection/export
- stage
- chosen action
- reason
- evidence
- confidence
- policy
- outcome
- correlation identifiers linking it back to the owning run/artifact/product

The system should expose these records at three readable levels:

- concise operator summary
- step-by-step narrative view
- full structured JSON/DB record

## Important Rule

Do not log everything. Record meaningful decisions and state transitions.

The schema subsystem already shows the right style:

- semantic inference scores candidates with confidence and evidence
- schema generation emits `x-polylogue-confidence`
- schema generation emits `x-polylogue-evidence`
- redaction reporting records action and reason
- `schema explain` turns those records into readable operator output

That pattern should become the platform standard.

## Existing Seeds

- `polylogue/schemas/semantic_inference.py`
- `polylogue/schemas/schema_generation.py`
- `polylogue/schemas/redaction_report.py`
- `polylogue/cli/commands/schema.py`
- `polylogue/lib/metrics.py`
- `polylogue/pipeline/observers.py`
- `polylogue/health.py`
- `polylogue/schemas/audit.py`

## Exit Criteria

- major pipeline and control-plane stages emit readable decision records
- an operator can explain why an artifact was classified, routed, skipped,
  quarantined, linked, or rendered a certain way
- run output can be inspected as a narrative, not just raw logs or final counts
- important failures preserve reason and evidence rather than collapsing into
  opaque error strings

## Pillar 5: Artifact Identity And Cohort Intelligence

## Goal

Turn raw artifact analysis into a reusable runtime intelligence layer.

## Main Changes

- durable artifact ledger
- explicit classification evidence and confidence
- policy layer:
  - parse
  - enrich-only
  - validate
  - ignore-for-schema
  - quarantine
- cohort manifests and cohort-bound promotion
- sidecar linkage
- drift detection for new/changed cohorts

## Important Upgrade Over The Earlier Proof Plan

Artifact analysis should not stop at proving what exists. It should actively
drive:

- parser routing
- schema matching
- validation mode
- triage and quarantine
- QA targeting
- synthetic generation inputs

## Exit Criteria

- classification affects runtime decisions, not just reports
- new cohorts are actionable operational events

## Pillar 6: Semantic Identity And Explainable Loss

## Goal

Make Polylogue measure what meaning survives across parsing, normalization,
rendering, querying, and export.

## Main Changes

- canonical semantic fact model
- explicit loss taxonomy
- boundary-level preservation records
- projection/export preservation policies
- semantic diffing between surfaces

## High-Value Semantic Categories

- role
- chronology
- message identity
- tool use/result structure
- subagent spawns
- reasoning traces
- attachments/media references
- branch/continuation relationships
- important provider metadata

## Exit Criteria

- critical semantic loss becomes mechanically detectable
- surfaces declare what they preserve and what they intentionally omit

## Pillar 7: Relationship Graph And Projection Convergence

## Goal

Represent core archive relationships once and reuse them everywhere.

## Main Changes

Normalize graph-like relations for:

- parent/child conversations
- sidechains and continuations
- subagent spawn relationships
- sidecar-to-primary linkage
- message parentage / branch position
- attachment references
- tool call/result relationships where possible

Then build named projections on top:

- summary projection
- detail projection
- dialogue projection
- tool-focused projection
- proof/provenance projection
- site index projection
- MCP payload projection

## Why This Matters

Right now multiple surfaces decide independently what matters. Projection
convergence reduces drift between:

- CLI
- MCP
- facade
- site
- exports
- proofs

## Exit Criteria

- major surfaces consume named projections instead of locally composing fields

## Pillar 8: Unified Operations Layer

## Goal

Make CLI, MCP, facade, and long-running workflows execute through shared typed
operations rather than surface-specific orchestration.

## Main Changes

Promote explicit operations such as:

- query
- mutate tags/metadata/delete
- ingest
- render
- build site
- generate/promote schema
- run proof/audit
- export

Each should take typed intents and return typed result objects.

## Existing Seeds

- `ConversationQuerySpec`
- `RuntimeServices`
- facade methods
- MCP handlers already building query specs

## Exit Criteria

- the same operation core can back CLI, MCP, and library APIs
- surface-specific orchestration shrinks

## Pillar 9: Managed Data Products

## Goal

Treat every derived product as a first-class object with provenance,
freshness, and rebuild policy.

## Managed Product Families

- schema manifests and promoted schemas
- search indexes
- render outputs
- static site builds
- export bundles
- health/proof/QA reports
- run bundles
- capture artifacts

## Storage Policy

For each product family define:

- canonical identifier
- authoritative store:
  - DB row
  - filesystem artifact
  - both
- provenance fields
- freshness policy
- invalidation rules
- rebuild command/operation

## Important Rule

Most manifests and summaries belong in DB.
Most large materializations belong on disk.

## Exit Criteria

- product lineage and freshness are queryable
- rebuilds and drift checks operate from structured metadata, not path scanning

## Pillar 10: Unified Verification, Drift, And Benchmarking

## Goal

Make all verification lanes consume the same underlying facts and report
grammar.

## Main Changes

Unify:

- health checks
- schema audits
- proof reports
- showcase QA
- mutation campaigns
- benchmarks
- drift detection

through a shared verification result grammar.

## Important Extension

Add proof/control-plane benchmarks:

- artifact classification throughput
- cohort assignment throughput
- proof report generation cost
- semantic preservation audit cost

## Exit Criteria

- verification results are comparable across lanes
- drift reports can compare runs coherently

## Pillar 11: Deletion And Architectural Tightening

## Goal

Remove local variants and stale layers once canonical ones exist.

## Main Changes

- delete duplicated report grammars where superseded
- collapse local helper patterns into shared capabilities
- remove stale docs that describe older architecture
- reduce surface-specific field composition
- shrink ad hoc file-based runtime state

## Important Rule

Do not let the new architecture coexist forever with the old one. Cohesion only
improves if redundant layers are removed.

## Sequence

The recommended order is:

1. canonical internal languages
2. capability registry
3. database-resident control plane
4. decision observability and execution narrative
5. artifact identity and cohort intelligence
6. semantic identity and explainable loss
7. relationship graph and projection convergence
8. unified operations layer
9. managed data products
10. unified verification, drift, and benchmarks
11. deletion and architectural tightening

This is not purely linear, but the order matters.

In particular:

- semantic-loss work should not race ahead of artifact/control-plane work
- projection convergence should follow graph/semantic stabilization
- product-management work should be DB-first from the start

## Concrete Storage Decisions

To keep the program well-specified, the default placement policy should be:

### DB-authoritative

- artifact records
- artifact classifications
- support/policy statuses
- cohort manifests
- linkage facts
- semantic facts
- loss summaries
- run records
- drift deltas
- verification results
- product manifests

### Filesystem-authoritative, DB-indexed

- render outputs
- site output
- export bundles
- assets/media
- VHS/GIF captures
- large archived bundles for human inspection

### Repository-authoritative

- committed schemas
- migrations
- templates
- exercise catalogs
- docs

## Reuse Strategy

The program should explicitly reuse and widen:

- `RawPayloadEnvelope` rather than inventing a parallel artifact identity object
- `RunRecord` / `runs` table rather than inventing a separate run store
- `HealthReport` and `AuditReport` patterns rather than inventing more local
  report dialects
- `ClusterManifest` rather than inventing a second cohort manifest format
- `ConversationQuerySpec` rather than letting each surface speak its own query
  language
- `RuntimeServices` rather than allowing new ambient service access patterns

## Success Criteria

This program is complete when all of the following are true:

- operational truth is mostly DB-resident and queryable
- provider/source-specific knowledge is explicit and bounded
- artifact support is cohort-driven and inspectable
- semantic preservation is measurable
- CLI, MCP, facade, site, and export surfaces consume shared operations and
  projections
- managed products have explicit provenance and freshness policies
- verification/drift systems compare runs coherently
- local duplicate dialects and ad hoc state files have been removed where
  superseded

## Closing Thesis

The main problem is not a lack of tests, not a lack of reports, and not even a
lack of abstractions. The deeper issue is that Polylogue still has too many
local truths.

The integrated answer is:

- fewer canonical internal languages
- more database-resident operational truth
- clearer wire-format boundaries
- explicit projection/operation layers
- proofs and reports generated from shared facts rather than parallel logic

That is the route to making the project:

- more cohesive
- less hardcoded in the dangerous sense
- more complete in what it can support
- and more self-evidently trustworthy.
