# Artifact And Semantic Proof Program: 5-Commit Execution Plan

Date: 2026-03-19

## Purpose

This document turns the broader
`artifact-and-semantic-proof-program-2026-03-19.md` program into a concrete
execution sequence of five coherent commits.

The goal is not to slice the work into arbitrary micro-commits. Each commit
should establish a defensible, test-backed unit of progress that leaves the
tree in a meaningful state.

## Execution Rule

The sequence is deliberately ordered:

1. establish artifact truth
2. bind support to cohorts
3. prove the Claude `subagents/` mixed-root case
4. expose proof surfaces to operators
5. layer semantic preservation and explainable loss on top

Do not start the semantic-loss work before artifact/cohort truth is stable.

## Commit 1: Artifact Ledger Foundation

## Goal

Make raw artifact classification durable, queryable, and visible.

## Why First

Nothing else in the program is trustworthy if raw artifacts can still disappear
into transient in-memory classification with no persistent audit trail.

## Main Changes

- Introduce a persistent raw artifact record model.
- Record, at minimum:
  - artifact id
  - provider/runtime provider
  - source path
  - wire format
  - artifact kind
  - classification reason
  - classification confidence
  - parseability
  - schema eligibility
  - support status
  - cohort placeholder
  - parser/version placeholders
- Persist artifact rows during acquisition and raw payload envelope creation.
- Add the first operator surface:
  - `polylogue check --artifacts`
  - `polylogue check --artifacts --json`

## Likely Files

- `polylogue/lib/raw_payload.py`
- `polylogue/lib/artifact_taxonomy.py`
- `polylogue/storage/`
- `polylogue/cli/commands/check.py`
- relevant storage schemas/migrations

## Tests

- raw artifact records are written for ingested samples
- sidecars are recorded with non-parseable support states
- unknown artifacts are visible rather than dropped
- `check --artifacts --json` returns a stable machine-readable envelope

## Done Criteria

- every ingested raw file produces a durable artifact ledger record
- proof output can enumerate artifact classifications without parsing

## Suggested Commit Message

`feat: add raw artifact ledger and support status tracking`

## Commit 2: Cohort Manifests And Promotion Boundaries

## Goal

Make support bind to explicit artifact cohorts instead of implicit provider-wide
schema unions.

## Why Second

Once raw artifacts are durable, the next question is: which groups of artifacts
share a contract, and which do not?

## Main Changes

- Extend schema clustering/manifests with artifact-kind-aware cohort ids.
- Introduce explicit support states:
  - `supported_parseable`
  - `supported_sidecar`
  - `recognized_unparsed`
  - `quarantined`
  - `unknown`
- Prevent sidecar-only or metadata-only cohorts from entering parseable schema
  promotion.
- Make promoted schemas traceable back to cohort evidence.
- Add cohort-oriented reporting to `check`.

## Likely Files

- schema clustering/inference modules
- `polylogue/schemas/registry.py`
- manifest/report generation code
- `polylogue/cli/commands/check.py`

## Tests

- mixed artifact roots do not collapse into one union schema
- cohort manifests preserve artifact-kind boundaries
- promoted versions point back to cohort evidence
- sidecar-only cohorts remain schema-ineligible

## Done Criteria

- Polylogue can explain promoted, quarantined, and sidecar-only cohorts
- schema promotion is cohort-bound, not broad-provider folklore

## Suggested Commit Message

`feat: bind schema promotion to artifact cohorts`

## Commit 3: Claude `subagents/` Sidecar Linkage

## Goal

Turn the Claude Code `subagents/` layout into a fully accounted-for proof case.

## Why Third

This is the first concrete ugly mixed-root example the program should handle
end-to-end.

## Main Changes

- Link `agent-*.meta.json` sidecars to `agent-*.jsonl` streams by basename or
  stronger evidence when available.
- Enrich linked subagent streams with sidecar provenance such as `agentType`.
- Surface sidecar linkage quality in proof outputs:
  - linked
  - orphaned
  - ambiguous
- Ensure sidecars are preserved in support accounting without producing
  conversations.

## Likely Files

- Claude source/parser code
- artifact ledger/linkage layer
- proof report generation
- provider docs if needed

## Tests

- linked sidecar enriches corresponding stream metadata
- orphan sidecar remains visible in proof output
- sidecar never materializes as an empty conversation
- sampled `agentType` values survive into proof/provenance surfaces where
  appropriate

## Fixture Expectations

Add a small realistic regression corpus with:

- paired `agent-abc.jsonl` + `agent-abc.meta.json`
- stream without sidecar
- sidecar without stream
- non-subagent Claude session in adjacent paths

## Done Criteria

- Claude `subagents/` is no longer a speculative edge case
- proof output can distinguish linked vs orphaned sidecars

## Suggested Commit Message

`feat: link claude subagent sidecars into proof surface`

## Commit 4: Proof Surfaces In `check` And `qa`

## Goal

Expose artifact/cohort proof as stable operator-facing outputs.

## Why Fourth

By this point the underlying evidence exists. It should now be visible through
canonical CLI surfaces instead of remaining internal plumbing.

## Main Changes

- Add:
  - `polylogue check --cohorts`
  - `polylogue check --proof`
- Extend `qa` outputs with:
  - artifact summary
  - cohort summary
  - support matrix
  - unknown/quarantine summary
  - sidecar linkage summary
- Make proof outputs available in both JSON and stable Markdown forms.

## Likely Files

- `polylogue/cli/commands/check.py`
- `polylogue/cli/commands/qa.py`
- `polylogue/showcase/report.py`
- `polylogue/showcase/qa_runner.py`
- possibly `polylogue/showcase/invariants.py`

## Tests

- stable JSON contract for proof output
- deterministic Markdown support summary
- snapshot tests for human-readable proof output
- `qa` report includes support proof sections, not just exercise summaries

## Done Criteria

- an operator can answer "what do we support and why?" from CLI output alone
- proof artifacts are diffable and machine-readable

## Suggested Commit Message

`feat: add proof reports to check and qa`

## Commit 5: Semantic Contract And Explainable Loss

## Goal

Start Pillar II on top of the now-trustworthy artifact/cohort base.

## Why Last

Semantic preservation analysis is only credible once the raw artifact and
support model is explicit.

## Main Changes

- Define a canonical semantic fact set for normalized conversations/messages.
- Add a loss taxonomy for major boundaries:
  - parse -> normalize
  - normalize -> render
  - normalize -> export
  - normalize -> query projection
- Start with the highest-value semantic categories:
  - role
  - chronology
  - tool use/result structure
  - subagent spawns
  - reasoning traces
  - attachment presence
  - branch/continuation relations
- Add proof/diff reporting for preserved vs intentionally omitted vs
  unexpectedly lost semantics.

## Likely Files

- `polylogue/lib/viewports.py`
- `polylogue/pipeline/semantic.py`
- rendering/export modules
- query projection modules
- proof/report generation code

## Tests

- semantic fact extraction contract tests
- renderer/export preservation policy tests
- critical-loss detection tests for dropped tool/subagent/reasoning semantics
- regression tests confirming acceptable normalization does not get flagged as
  loss

## Done Criteria

- Polylogue can emit a semantic preservation/loss report
- major surfaces have declared preservation expectations
- silent loss of critical semantics becomes mechanically detectable

## Suggested Commit Message

`feat: add semantic preservation contract and loss taxonomy`

## Cross-Commit Guardrails

These rules apply throughout the sequence.

### 1. No Silent Parallel Truth Surfaces

Do not introduce handwritten support tables that drift from the ledger or
manifests. Derive reports from canonical records.

### 2. No Sidecar Regressions

Any new sidecar handling must keep the invariant:

- recognized sidecars are preserved and surfaced
- recognized sidecars do not become empty conversations

### 3. No Premature Semantic Theory

Do not over-design semantic equivalence before the artifact/cohort proof layer
is stable and exposed.

### 4. Every Commit Must Add Proof, Not Just Plumbing

Each commit should end with a new visible capability:

- commit 1: artifact ledger output
- commit 2: cohort-aware promotion/reporting
- commit 3: Claude subagent sidecar accounting
- commit 4: operator-visible proof surfaces
- commit 5: semantic loss reporting

## Recommended Verification Commands

These should evolve as the work lands, but the target shape is:

```bash
nix develop -c pytest -q --ignore=tests/integration
POLYLOGUE_FORCE_PLAIN=1 nix develop -c polylogue check --artifacts --json
POLYLOGUE_FORCE_PLAIN=1 nix develop -c polylogue check --proof --json
POLYLOGUE_FORCE_PLAIN=1 nix develop -c polylogue qa --only exercises --tier 0
```

Later in the sequence:

```bash
POLYLOGUE_FORCE_PLAIN=1 nix develop -c polylogue check --cohorts --json
POLYLOGUE_FORCE_PLAIN=1 nix develop -c polylogue qa --json
```

## Final Success Condition

After these five commits, Polylogue should be materially closer to a system
that can prove both:

- artifact truth
- semantic truth

without hiding support gaps or meaning loss behind implicit assumptions.
