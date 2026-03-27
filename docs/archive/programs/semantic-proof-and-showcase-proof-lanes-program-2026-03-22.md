# Polylogue Semantic Proof And Showcase Proof-Lanes Program

Date: 2026-03-22
Status: executed subprogram
Role: executed slice of the post-schema frontier covering semantic-preservation proofing and black-box proof-lane showcase coverage

See also:

- `artifact-and-semantic-proof-program-2026-03-19.md`
- `testing-reliability-expansion-program-2026-03-14.md`
- `planning-and-analysis-map-2026-03-21.md`

## Purpose

Execute the first concrete semantic-preservation proof slice after the artifact
control plane and schema package-authority work were completed.

The target was not a vague "more semantics" story. It was one specific,
canonical chain:

1. prove meaning preservation for the canonical markdown render surface
2. surface that proof through operator-facing `check`
3. compose it into QA/session/report outputs
4. embed its summary into publication/site manifests
5. expose the proof lanes to black-box showcase/QA catalog coverage

## Why This Slice

The broader frontier after schema authority was:

- semantic-preservation proofing
- testing/showcase expansion

The highest-leverage first cut was canonical markdown, because it is already a
stable operator surface and a major downstream publication/export boundary.

## Executed Scope

### 1. Canonical Markdown Semantic Proof Core

Implemented a typed semantic-preservation report for markdown rendering:

- per-conversation proof
- per-provider rollups
- declared-loss vs critical-loss taxonomy
- measurable checks for:
  - renderable message sections
  - role sections
  - timestamps
  - attachment lines
  - empty-message omission
  - typed thinking/tool semantic flattening

Primary module:

- `polylogue/rendering/semantic_proof.py`

### 2. `check --semantic-proof`

Added a first-class `check` lane for semantic-preservation proofing:

- `--semantic-proof`
- `--semantic-provider`
- `--semantic-limit`
- `--semantic-offset`

Output surfaces:

- JSON payload under `semantic_proof`
- plain-text summary with metric and provider rollups

Primary module:

- `polylogue/cli/commands/check.py`

### 3. QA And Report Integration

Added semantic proof as a first-class QA stage:

- `QAResult.semantic_proof_report`
- `QAResult.semantic_proof_status`
- durable `semantic-proof.json`
- `qa-session.json` stage payload
- QA summary and Markdown sections

Primary modules:

- `polylogue/showcase/qa_runner.py`
- `polylogue/showcase/report.py`

### 4. Publication And Site Embedding

Added compact semantic-proof summaries to typed publication manifests and site
build outputs.

Primary modules:

- `polylogue/publication.py`
- `polylogue/site/publication_support.py`
- `polylogue/site/builder.py`

### 5. Showcase / Black-Box Proof-Lane Coverage

Added first-class showcase exercises for the proof surfaces:

- `check-proof-json`
- `check-cohorts-json`
- `check-semantic-proof`
- `check-semantic-proof-json`

Primary module:

- `polylogue/showcase/exercises.py`

## Verification

Executed focused verification covering:

- semantic-proof core
- `check` CLI proof surfaces
- QA/session/report composition
- publication/site manifest embedding
- showcase catalog wiring
- `check --help` terminal snapshot contract

Representative slices:

- `tests/unit/core/test_semantic_proof.py`
- `tests/unit/cli/test_check.py`
- `tests/unit/showcase/test_qa_runner.py`
- `tests/unit/showcase/test_report.py`
- `tests/unit/site/test_builder.py`
- `tests/unit/cli/test_site.py`
- `tests/unit/showcase/test_exercise_catalog.py`
- `tests/unit/cli/test_terminal_snapshots.py::TestCommandOutputs::test_check_output_snapshot`

## Result

Polylogue now has an end-to-end semantic-proof chain for the canonical markdown
surface, and that chain is visible from:

- `check`
- QA/session artifacts
- publication/site manifests
- showcase exercise catalog

## Remaining Frontier After This Subprogram

This does not finish all future semantic-proof work.

Still open:

- semantic-preservation proofing for additional transform/query/export surfaces
- stronger black-box QA around the new proof lanes on seeded/live corpora
- broader testing/showcase hardening from the 2026-03-14 backlog

But the frontier is no longer "semantic proof is still only planned". The
canonical markdown surface is now implemented and wired through the main
operator and verification stack.
