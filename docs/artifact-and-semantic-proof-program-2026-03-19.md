# Polylogue Artifact And Semantic Proof Program

Date: 2026-03-19
Status: reference program; artifact/cohort half partly executed, semantic-preservation half remains future work
Role: narrower proof-oriented architecture program

Current execution entrypoint:

- `intentional-forward-program-2026-03-21.md`
- `artifact-cohort-control-plane-program-2026-03-21.md`
- `planning-and-analysis-map-2026-03-21.md`

## Purpose

This document defines a unified next major program for Polylogue:

1. prove what every raw artifact is, how it is classified, and which contract
   owns it
2. prove what meaning survives after parsing, normalization, rendering,
   querying, and export

The first half prevents silent support gaps.
The second half prevents silent semantic loss.

Together, they move Polylogue toward a system that is self-evidently
verifiable, explicit about support boundaries, and much less dependent on
implicit provider folklore or hardcoded assumptions.

## One-Line Goal

Make Polylogue produce a proof-bearing chain from raw artifact acquisition to
semantic preservation, so the project can explain:

- what it saw
- how it classified it
- which cohort/schema/parser owns it
- what meaning it extracted
- what meaning it preserved
- what it intentionally dropped
- what it still does not understand

## Why This Is The Next Broad Program

Polylogue already has unusually strong ingredients:

- heuristic artifact taxonomy
- schema clustering and promoted versions
- provider-specific typed parsers
- schema-driven synthetic corpora
- composable QA/showcase execution
- typed semantic projection structures
- semantic extraction for tools, subagents, and reasoning traces

But its current proof story is still split.

It can often prove:

- "this command worked"
- "this fixture parsed"
- "this schema validated"
- "this provider root mostly looked right"

It still struggles to prove, end to end:

- "every raw file was accounted for"
- "recognized sidecars were preserved but not misparsed"
- "this cohort is supported by an explicit parser/schema contract"
- "this normalized conversation preserved the important meaning from raw input"
- "this renderer/export/query surface did not silently lose essential semantics"

That missing proof chain is the next ambitious frontier.

## Program Thesis

Polylogue should be able to emit one chain of evidence:

1. acquire raw bytes
2. classify every artifact
3. assign each artifact to a cohort
4. bind parseable cohorts to schema/version/parser contracts
5. normalize parse results into a canonical semantic representation
6. render/export/query that representation through multiple surfaces
7. report both support coverage and semantic preservation/loss

That chain has two distinct pillars.

## Pillar I: Artifact Cohort Proofs

This pillar answers:

> What is this raw thing, why do we think so, and is it supported?

Primary failures caught here:

- mixed roots treated as if all files were conversations
- sidecars accidentally parsed as empty or garbage conversations
- provider-wide schemas swallowing unrelated shapes
- unknown artifacts disappearing without evidence
- support relying on hidden path heuristics rather than inspectable outcomes

## Pillar II: Semantic Preservation And Explainable Loss

This pillar answers:

> After parsing and normalization, what meaning survived, what changed, and
> what was intentionally or unintentionally lost?

Primary failures caught here:

- structured tool semantics flattened into plain text
- subagent relationships disappearing after normalization
- reasoning traces present in raw/provider data but lost in exports
- branch structure rendered incorrectly
- attachments or metadata silently dropped across formats

The pillars should be implemented sequentially, but designed as one program.

## Design Principles

### 1. Support Starts With Classification

Polylogue should support all obtainable data by first classifying it, not by
assuming everything must immediately become a conversation.

Recognized sidecars, metadata documents, pointers, and indices are valid
supported outcomes even when they are not parseable as conversations.

### 2. Unknown Must Be First-Class

Unknown or quarantined artifacts are not inherently failures. They are durable
evidence that support boundaries were encountered honestly.

### 3. Cohorts Drive Contracts

Provider-wide union schemas may still exist as reports, but executable support
should bind to cohorts and promoted versions.

### 4. Semantics Must Be Measurable

Semantic preservation cannot remain a subjective reading of output. It needs a
canonical, typed contract and a loss taxonomy.

### 5. Loss Must Be Declared, Not Discovered By Accident

If a renderer/export/query surface intentionally omits certain metadata, that
must appear as an explicit policy, not as an undocumented side effect.

### 6. Heuristics Are Allowed Only When They Are Inspectable

Heuristics are acceptable if Polylogue records:

- reason
- confidence
- representative evidence
- competing possibilities when relevant
- promotion/quarantine outcome

### 7. Proof Beats Optimism

The program should prefer durable evidence artifacts over broad claims like
"full support" or "provider-agnostic normalization" unless those claims are
mechanically backed.

## Current Baseline

Polylogue already has enough foundations to begin this program without a
parallel architecture:

- artifact classification infrastructure
- sidecar recognition for Claude subagent `.meta.json`
- schema clustering and manifest support
- promoted version retrieval
- schema-driven Hypothesis strategies
- synthetic showcase fixture generation
- structured QA JSON/Markdown outputs
- typed content blocks, tool calls, reasoning traces, and semantic extraction

What remains missing is not a core parser rewrite. It is proof integration.

## Program Outcomes

When this program is complete, Polylogue should have:

- a durable raw artifact ledger
- explicit cohort manifests and support statuses
- sidecar linkage and provenance enrichment
- a support matrix for providers/cohorts/versions
- a canonical semantic contract
- an explainable-loss model across transform boundaries
- proof reports for both artifact coverage and semantic preservation
- QA outputs that explain not just command behavior but support and meaning
  preservation

## Part A: Artifact Cohort Proofs

## Goal

Make Polylogue emit a proof-bearing, cohort-level account of all raw artifacts,
so support boundaries, sidecars, parser coverage, unknowns, and promoted schema
versions are all visible, testable, and reproducible.

## Current Problem

Truth is still split across:

- raw payload envelopes and artifact classification
- schema registry manifests and promoted versions
- parser selection and provider adapters
- synthetic corpus generation
- showcase exercises and invariant checks

Polylogue can often answer:

- "this command worked"
- "this schema parses"
- "this fixture roundtripped"

It still cannot answer, in one report:

- how many raw artifacts were seen
- how many were parseable conversation-bearing artifacts
- how many were recognized sidecars
- how many were quarantined or unknown
- which cohorts are promoted and contract-backed
- which cohorts remain unsupported but visible

## New Canonical Objects For Part A

### 1. Raw Artifact Record

Each acquired file or payload sample should produce a raw artifact record with
fields such as:

- `artifact_id`
- `provider`
- `runtime_provider`
- `source_path`
- `wire_format`
- `artifact_kind`
- `classification_reason`
- `classification_confidence`
- `parse_as_conversation`
- `schema_eligible`
- `payload_fingerprint`
- `structural_profile_tokens`
- `cohort_id`
- `support_status`
- `parser_name`
- `matched_version`
- `linked_sidecar_targets`

### 2. Cohort Manifest

Each cohort manifest should describe:

- provider
- artifact kind
- cohort id
- sample count
- byte count
- representative paths
- first seen / last seen
- dominant profile tokens
- support status
- parser binding
- promoted schema version
- synthetic generation policy
- linked sidecar kinds
- open risks / quarantine reason

### 3. Archive Proof Report

Each proof run should emit a report answering:

- how many raw artifacts were seen
- how they were classified
- what share were parseable
- what share were schema-eligible
- what share were supported by promoted cohorts
- what sidecars were recognized
- what unknown/quarantined cohorts remain
- what regressions were detected relative to the previous proof run

### 4. Support Matrix

Polylogue should maintain a generated support matrix keyed by:

- provider
- artifact kind
- cohort/version
- parser support
- schema support
- synthetic support
- showcase support
- invariant support

## Support Status Model

Every artifact and cohort should carry one of a small set of explicit statuses:

- `supported_parseable`
- `supported_sidecar`
- `recognized_unparsed`
- `quarantined`
- `unknown`

This is important because "not parsed" must split into:

- intentionally sidecar-only
- recognized but not yet parser-supported
- genuinely unknown

## Claude Code `subagents/` As First Driver

The sampled Claude Code `subagents/` layout is an ideal forcing function.

Observed structure:

- many `agent-*.jsonl` event streams
- many `agent-*.meta.json` sidecars
- sampled/aggregated `meta.json` contents contained only `agentType`
- observed `agentType` values included:
  - `general-purpose`
  - `boilerplate-scribe`
  - `feature-dev:code-architect`

Required behavior:

1. `agent-*.jsonl` classifies as `subagent_conversation_stream`
2. `agent-*.meta.json` classifies as `agent_sidecar_meta`
3. sidecars never produce empty conversations
4. sidecars link to corresponding streams where possible
5. proof reports count linked/orphaned/ambiguous sidecars
6. schema inference excludes those sidecars from parseable conversation schema
   promotion

## Workstreams For Part A

### A1. Raw Artifact Ledger

- persist artifact-level classification records
- record reason/confidence/support status
- make the ledger queryable and exportable

### A2. Cohort Inference And Manifest Promotion

- preserve artifact-kind boundaries during clustering
- promote cohorts, not loose provider unions
- make quarantine and sidecar-only cohorts explicit

### A3. Sidecar Linkage And Provenance Enrichment

- associate sidecars with primary streams/documents
- surface linked/orphaned/ambiguous sidecars in reports

### A4. Cohort-Driven Synthetic And Parser Proofs

- generate synthetic fixtures from promoted cohorts
- add per-cohort parser proof lanes

### A5. QA And Operator Surfaces

- add `check --artifacts`, `check --cohorts`, `check --proof`
- extend `qa` outputs with support matrices and cohort summaries

## Exit Criteria For Part A

- every raw artifact is accounted for in a durable ledger
- promoted schemas are cohort-bound and traceable
- sidecars are recognized and surfaced without misparsing
- `check --proof` can emit human and machine-readable support reports
- `qa` includes support coverage, not just showcase command outcomes

## Part B: Semantic Preservation And Explainable Loss

## Goal

Make Polylogue prove what meaning survives after parsing, normalization,
rendering, querying, and export, and explicitly account for intentional and
unintentional loss.

## Current Problem

Polylogue already has rich semantic structures:

- typed `ContentBlock`
- typed `ToolCall`
- `ReasoningTrace`
- normalized roles
- tool categorization
- semantic extraction for git, file ops, and subagent spawns

But it still lacks a first-class proof surface for semantic preservation across
boundaries:

- raw payload -> provider parse
- provider parse -> normalized conversation/message/content blocks
- normalized IR -> markdown/html/plaintext/json/csv/org/obsidian exports
- normalized IR -> query/list/stats/search surfaces

The project can prove many local properties, but not yet:

- what semantics were preserved
- what were normalized
- what were dropped
- whether the drops were intentional
- whether two surfaces preserve the same core meaning

## New Canonical Objects For Part B

### 1. Canonical Semantic Contract

Define a stable set of meaning-bearing facts for conversations/messages, such
as:

- role
- chronology
- message identity
- dialogue structure
- tool use structure
- tool result linkage
- subagent spawn metadata
- reasoning traces
- attachments
- token/cost/model metadata
- branch/subagent relationships

This contract should be typed and machine-checkable.

### 2. Loss Taxonomy

Each transform boundary should be able to classify outcomes as:

- `preserved`
- `normalized_equivalent`
- `redacted`
- `intentionally_omitted`
- `unsupported`
- `ambiguous`
- `unexpectedly_lost`

### 3. Boundary Loss Record

For each transformation stage, record:

- input semantic facts
- output semantic facts
- preserved facts
- dropped facts
- transformed facts
- policy justification where applicable

### 4. Semantic Diff Report

Given two surfaces, Polylogue should be able to report:

- identical meaning
- formatting-only differences
- metadata loss only
- critical semantic loss

## Workstreams For Part B

### B1. Canonical Semantic Contract

- formalize the normalized meaning-bearing fact set
- make it explicit which surfaces are expected to preserve which fact classes

### B2. Boundary Instrumentation

- instrument parse, normalize, render, export, and query paths
- emit explainable-loss summaries

### B3. Format Preservation Policies

For each output surface, declare what must be preserved and what may be omitted:

- markdown
- html
- plaintext
- json
- csv
- org
- obsidian
- list/stats/search/query projections

### B4. Semantic Diff Engine

- compare canonical semantic facts across boundaries
- detect critical loss and classify acceptable normalization

### B5. QA And Proof Integration

- extend proof reports with semantic preservation summaries
- highlight critical semantic loss separately from artifact/support failures

## Example Failures Part B Should Catch

- structured tool-use blocks flattened into plain text with no declared policy
- subagent spawn metadata present in normalized content blocks but absent from
  relevant exports
- reasoning traces surviving in one output surface and disappearing in another
  without explanation
- branch or continuation relationships rendered incorrectly
- attachment references silently lost from surfaces that should at least mention
  them

## Exit Criteria For Part B

- Polylogue can emit semantic loss reports per run
- every major export/query surface has a declared preservation policy
- critical semantic categories have cross-surface invariants
- silent semantic loss becomes mechanically detectable

## Unified QA Vision

When both parts are in place, `polylogue qa` and `polylogue check --proof`
should answer two different but linked questions:

### 1. Support Proof

- what artifacts were seen
- how they were classified
- which cohorts were promoted
- which sidecars were preserved
- which cohorts remain unknown or quarantined

### 2. Meaning Proof

- what semantics were extracted
- what semantics were preserved across major surfaces
- what was intentionally omitted
- what was unexpectedly lost

That is the real end state:

not just "the tool ran"

but:

- "the archive was accounted for"
- "support boundaries were explicit"
- "meaning preservation was measured"

## Testing Strategy

The program should use a layered strategy.

### 1. Unit Layer

- artifact classification
- sidecar linking
- support status transitions
- semantic fact extraction
- loss taxonomy classification

### 2. Contract Layer

- cohort/schema promotion
- payload-to-cohort matching
- parser binding correctness
- canonical semantic contract verification
- format preservation policy checks

### 3. Integration Layer

- mixed provider roots with sidecars and unknowns
- ingestion plus ledger persistence
- normalization through rendering/export/query
- proof report generation

### 4. Regression Corpus Layer

Maintain ugly but representative fixture trees including:

- Claude main sessions
- Claude `subagents/` streams and sidecars
- session index sidecars
- provider metadata sidecars
- unknown artifacts
- conversations with reasoning traces, tool calls, attachments, and branching

The point is to test real mixed conditions, not only idealized fixtures.

## Risks

### 1. Too Many Truth Surfaces

If ledgers, manifests, schemas, semantic contracts, and proof reports drift
apart, the program fails.

Mitigation:

- derive downstream reports from ledger/manifests/contracts rather than
  duplicating logic

### 2. Heuristic Drift

Artifact classification and semantic equivalence can both become hidden
hardcoding.

Mitigation:

- require reasons/confidence/policies
- keep proof artifacts reviewable
- build regression corpora around ugly real inputs

### 3. Storage And Complexity Growth

Artifact ledgers and loss reports can become large.

Mitigation:

- keep records compact
- use hashes/fingerprints for large payload evidence
- separate aggregate reports from heavyweight sample payloads

### 4. Overfitting To Current Provider Layouts

Claude `subagents/` is the forcing example, not the final abstraction.

Mitigation:

- define artifacts, cohorts, and semantic facts generically
- keep provider-specific hints subordinate to explicit classification/policy

## Recommended Delivery Order

1. Raw artifact ledger and support status model
2. Cohort manifests and sidecar-aware promotion boundaries
3. Claude `subagents/` sidecar linkage and proof reporting
4. `check --artifacts`, `check --cohorts`, `check --proof`
5. Canonical semantic contract and loss taxonomy
6. Boundary instrumentation for parse/normalize/render/export/query
7. Semantic diff/proof engine
8. QA/report integration across both pillars

This order is important.

Part B depends on Part A giving a trustworthy artifact and contract base.

## Concrete Success Criteria

This unified program is complete when all of the following are true:

- Polylogue can classify all sampled provider roots into explicit artifact kinds
  with durable evidence
- sidecars such as Claude `agent-*.meta.json` are recognized, counted, and
  linked where possible, and never materialize as empty conversations
- promoted schemas are cohort-bound and traceable to evidence
- `polylogue check --proof` can emit both support and semantic preservation
  reports
- `polylogue qa` includes proof coverage, not just command outcomes
- unknown/quarantined cohorts are visible rather than silent
- every promoted parseable cohort has parser/synthetic proof lanes
- every major output surface has a declared semantic preservation policy
- critical semantic loss is mechanically detectable

## Closing Thesis

Polylogue does not primarily need more hidden cleverness. It needs a better
proof surface.

The ambition of this program is not just broader support and not just better
rendering. It is to make Polylogue auditable at two levels:

- artifact truth
- semantic truth

So the system can explain, with evidence:

- what we saw
- what we classified
- what we promoted
- what we parsed
- what we preserved as sidecars
- what meaning we kept
- what meaning we intentionally omitted
- what we still do not understand

That is the route to becoming both less hardcoded and more complete.
