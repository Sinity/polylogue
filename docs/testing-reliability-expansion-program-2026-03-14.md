# Polylogue Testing, Reliability, and Showcase Expansion Program

Date: 2026-03-14

## Purpose

This document defines the next major implementation program for Polylogue's
testing, runtime reliability, schema intelligence, synthetic generation,
terminal regression coverage, and showcase/QA automation.

It is written as a standalone execution plan. It assumes no prior familiarity
with design discussions, chat transcripts, or earlier planning notes.

The plan is intentionally ambitious. It is not a grab-bag of small fixes. It
is a coordinated program to move Polylogue from "well-tested core archive tool"
to "self-verifying, schema-aware, deterministic archive platform with unified
demo and QA generation."

## Current Baseline

Polylogue already has strong foundations:

- Schema inference already emits several `x-polylogue-*` annotations, including
  value sets, formats, numeric ranges, frequencies, array lengths, multiline
  hints, and dynamic-key/reference hints.
- Schema validation already exists and can be exercised from `check --schemas`.
- Synthetic corpus generation already exists and can produce provider-shaped
  inputs for all current providers.
- Showcase execution already exists and produces summary, cookbook, JSON report,
  and audit JSON artifacts.
- VHS-based terminal recordings already exist for demos.
- Textual pilot tests exist for the TUI.
- Renderer snapshot tests already exist for HTML output.
- Migration safety, storage correctness, concurrency guards, and moderate-scale
  performance budgets already exist.

The purpose of this program is to unify and deepen those pieces into one
coherent platform.

## Program Outcomes

When this program is complete, Polylogue should have all of the following:

- A strict machine-consumable CLI contract when JSON output is requested.
- A single runtime verification surface centered on `polylogue check`.
- A semantic schema layer that understands message roles, message bodies,
  timestamps, conversation titles, and structural relationships.
- A largely provider-agnostic synthetic generator driven by schema semantics.
- Local, operator-driven schema clustering, versioning, comparison, and
  promotion workflows.
- End-to-end ingestion hostility coverage, including partial corruption,
  interruption, and extreme chronology handling.
- Deterministic PTY/ANSI regression testing for terminal output.
- A single canonical showcase system that can generate text artifacts, VHS
  captures, QA manifests, and archived bundles from the same scenario catalog.
- A data-gravity validation harness for very large archives and long-running
  storage/index operations.

## Program Principles

### 1. One Health Surface

Runtime diagnostics are centered on `polylogue check`. Polylogue should have
one health entrypoint with multiple lanes and one clear operator surface.

### 2. One Showcase System

Terminal demos, showcase exercises, and QA bundle generation should be driven
from one scenario catalog. Hand-maintained divergence between `demo`, `showcase`,
`qa`, and `demos/tapes` should be reduced over time.

### 3. Schema Semantics Are the Source of Truth

Provider-specific synthetic hacks should be replaced by schema-level semantics
and relationships wherever possible. Provider-specific logic should remain only
where it describes wire format or transport-level shape.

### 4. Determinism Is a First-Class Requirement

Generated QA artifacts must be diffable. Terminal regression tests must not
depend on CPU timing. Showcase runs must have stable output modes for both
machine verification and human inspection.

### 5. Fast CI and Slow Validation Must Be Separate Lanes

Default CI should remain fast and stable. Large-scale, long-running, or
resource-intensive validation belongs in explicit slow lanes, benchmark lanes,
or operator-invoked campaigns.

### 6. Operator-Driven Schema Promotion

Schema promotion should be explicit and reviewable. Polylogue should support
local schema inference, comparison, and promotion workflows with durable
artifacts and clear diffs.

## Workstream A: Machine-Consumable CLI Contract and Runtime Health

## Goal

Make `--json` a hard CLI contract and expand `check` into the canonical runtime
verification surface.

## Why This Matters

Polylogue increasingly serves as both a human CLI and a machine surface for
automation, MCP integration, scripted archive operations, and QA harnesses.
That means the CLI cannot reserve structured output only for success paths.

The target invariant is:

- If a command accepts JSON output and the caller asks for JSON, the command
  must emit valid JSON to stdout for both success and failure paths.

This includes:

- invalid flags
- missing arguments
- invalid enum values
- invalid file paths
- runtime exceptions
- failed health checks

## Implementation Plan

### A1. Add a Root Machine-Error Adapter

Introduce a small CLI error adapter layer that sits above Click command
invocation and standardizes structured failures.

Primary changes:

- Add a module such as `polylogue/cli/machine_errors.py`.
- Define a stable error envelope, for example:

```json
{
  "status": "error",
  "code": "invalid_arguments",
  "message": "No such option: --bad-flag",
  "command": ["check"],
  "details": {
    "option": "--bad-flag"
  }
}
```

- Add helpers for:
  - `invalid_arguments`
  - `invalid_path`
  - `runtime_error`
  - `dependency_missing`
  - `unsupported_environment`

### A2. Detect Machine-Output Intent Before Click Fails

Standard Click failure occurs before command logic runs, so the root command
must detect JSON intent from raw argv before normal parsing completes.

Primary changes:

- Extend the CLI entrypoint in `polylogue/cli/click_app.py`.
- Add a raw argv pre-scan for:
  - `--json`
  - output flags implying JSON if present in future command surfaces
- Wrap root invocation in exception handlers for:
  - `click.UsageError`
  - `click.BadParameter`
  - `click.ClickException`
  - unexpected exceptions

Behavioral rules:

- JSON errors go to stdout as structured JSON.
- Human-readable errors continue to go to stderr for non-JSON invocations.
- Exit codes remain non-zero.

### A3. Expand `check` Into Runtime Health and Environment Verification

Keep the existing archive and schema health checks, but grow `check` into the
single runtime verification surface.

Target sub-lanes:

- `check --runtime`
- `check --runtime --json`
- `check --runtime --verbose`
- `check --runtime --fix` only where safe and already aligned with existing
  repair semantics

Runtime checks should verify:

- database path exists or can be created
- database is writable
- schema is current
- FTS tables exist and are healthy
- optional sqlite-vec availability is visible and correctly reported
- archive root and render root are writable
- config paths resolve correctly
- Google credential paths are present when Drive features are configured
- terminal mode and Textual/Rich capabilities are intelligibly reported
- VHS availability is reported when showcase capture is requested

### A4. Normalize JSON Success and Failure Shapes Across Commands

Commands that already emit JSON should converge on a common top-level contract.

Recommended success envelope shape:

```json
{
  "status": "ok",
  "result": { ... }
}
```

Recommended failure envelope shape:

```json
{
  "status": "error",
  "code": "invalid_arguments",
  "message": "...",
  "details": { ... }
}
```

This can be introduced incrementally if a full migration would be too noisy,
but the new machine-error adapter should already emit the standardized form.

## Files to Change

- `polylogue/cli/click_app.py`
- `polylogue/cli/commands/check.py`
- `polylogue/cli/helpers.py`
- `polylogue/ui/__init__.py`
- `polylogue/ui/facade.py`
- `polylogue/paths.py`
- `tests/unit/cli/test_check.py`
- `tests/unit/cli/test_command_surfaces.py`
- new subprocess-focused tests such as:
  - `tests/integration/test_cli_machine_contract.py`

## Verification

- Add subprocess tests for:
  - invalid flag + `--json`
  - missing argument + `--json`
  - invalid path + `--json`
  - unexpected exception path + `--json`
- Add runtime-health tests for writable and non-writable paths.
- Confirm success and failure both return parseable JSON for machine mode.

## Exit Criteria

- `polylogue check --json --bad-flag` emits JSON, not Click text.
- `polylogue check --runtime --json` exercises runtime checks and emits JSON.
- Every JSON-capable command has at least one success-path test and one
  failure-path test.

## Workstream B: Semantic Schema Inference

## Goal

Teach schema inference to identify what important fields mean, not just what
their raw JSON types look like.

## Target Semantic Annotations

Introduce schema-level annotations for:

- `x-polylogue-semantic-role`
- `x-polylogue-time-delta`
- `x-polylogue-foreign-key`
- `x-polylogue-mutually-exclusive`
- `x-polylogue-string-length`

The first release of semantic roles should cover:

- `message_container`
- `message_role`
- `message_body`
- `message_timestamp`
- `conversation_title`

## Why This Matters

This is the foundation for:

- provider-agnostic synthetic generation
- better schema comparison
- better drift interpretation
- stronger parser diagnostics
- more realistic showcase data

## Implementation Plan

### B1. Extend Field Statistics Collection

Expand `polylogue/schemas/field_stats.py` to gather:

- string length statistics:
  - min
  - max
  - average
  - standard deviation
- newline incidence
- approximate entropy indicators
- distinct value counts
- distinct values per conversation
- monotonicity scores for ordered samples inside arrays
- candidate parent/child reference matches
- mutually exclusive field co-occurrence matrices
- container density metrics:
  - average array length
  - average object fanout
  - nested object depth

### B2. Add a Semantic Candidate Scoring Layer

Add a semantic scoring module, for example:

- `polylogue/schemas/semantic_inference.py`

Responsibilities:

- score container candidates
- score message-role candidates
- score message-body candidates
- score timestamp candidates
- score title candidates
- attach confidence and evidence

Candidate heuristics:

- `message_container`
  - deep repeated arrays or maps of objects
  - high element count
  - internal object structural similarity
- `message_role`
  - low-cardinality strings
  - strong reuse across records
  - values like `user`, `assistant`, `system`, `model`, `tool`, `human`
- `message_body`
  - high average length
  - multiline incidence
  - medium or high entropy
  - visible markdown/code fence patterns
- `message_timestamp`
  - numeric epoch or RFC3339 shape
  - monotonic increase within message order
  - repeated use in container entries
- `conversation_title`
  - short strings outside the container
  - high cardinality across conversations
  - low multiline rate

### B3. Add Relational Inference

Add a relation inference module, for example:

- `polylogue/schemas/relational_inference.py`

Responsibilities:

- detect foreign-key-like references
- detect temporal offsets
- detect mutually exclusive field groups
- detect field-length distributions worth preserving

Target outputs:

- `x-polylogue-foreign-key`
  - path to the referenced ID field or dynamic-key container
- `x-polylogue-time-delta`
  - reference field
  - observed min/max delta
  - optional percentile summary
- `x-polylogue-mutually-exclusive`
  - lists of field names that never co-occur
- `x-polylogue-string-length`
  - min/max/avg/stddev

### B4. Teach Schema Generation to Emit Semantic and Relational Annotations

Extend `polylogue/schemas/schema_generation.py` so semantic and relational
annotations are attached during the same pass that already emits the current
`x-polylogue-*` metadata.

The emitted schema should be rich enough for generation and comparison, but
still safe to store and review.

Privacy requirements:

- preserve current content-field suppression rules
- do not leak raw high-cardinality user strings into committed annotations
- keep value lists bounded and filtered

### B5. Add Confidence and Evidence Fields

Avoid opaque magic. Each semantic inference should carry enough explanation for
debugging and review.

Example:

```json
{
  "x-polylogue-semantic-role": "message_body",
  "x-polylogue-confidence": 0.91,
  "x-polylogue-evidence": {
    "avg_length": 381.2,
    "newline_rate": 0.74,
    "container_path": "$.mapping.*.message"
  }
}
```

## Files to Change

- `polylogue/schemas/field_stats.py`
- `polylogue/schemas/schema_generation.py`
- `polylogue/schemas/schema_inference.py`
- new modules such as:
  - `polylogue/schemas/semantic_inference.py`
  - `polylogue/schemas/relational_inference.py`
- existing tests and new focused tests such as:
  - `tests/unit/core/test_schema_semantic_inference.py`
  - `tests/unit/core/test_schema_relational_inference.py`
  - `tests/unit/core/test_schema_annotation_contracts.py`

## Verification

- Unit tests with hand-shaped fixtures for each semantic role.
- Property tests for monotonic timestamp detection and exclusivity detection.
- Golden-schema tests proving annotations land on known provider paths.

## Exit Criteria

- Generated schemas contain semantic-role annotations for current providers.
- Generated schemas contain at least first-release relational annotations.
- Semantic inference can be inspected and explained in tests.

## Workstream C: Provider-Agnostic Synthetic Generation

## Goal

Make synthetic generation primarily schema-driven and semantics-driven, with
provider-specific logic limited to wire-format boundaries.

## Why This Matters

This is the bridge between schema intelligence and:

- realistic demos
- realistic QA artifacts
- scalable synthetic benchmark corpora
- reduced maintenance burden for new provider variants

## Implementation Plan

### C1. Split Generation Into Clear Layers

Refactor `polylogue/schemas/synthetic.py` into explicit layers:

- schema traversal
- semantic value synthesis
- relational satisfaction
- wire-format assembly
- showcase narrative styling

If the file becomes too large to manage safely, split it into a package such as:

- `polylogue/schemas/synthetic/core.py`
- `polylogue/schemas/synthetic/semantic_values.py`
- `polylogue/schemas/synthetic/relations.py`
- `polylogue/schemas/synthetic/wire_formats.py`
- `polylogue/schemas/synthetic/showcase.py`

### C2. Drive Content From Semantic Roles

Generation rules should become:

- `message_role`
  - choose from observed roles or normalized canonical role sets
- `message_body`
  - choose from showcase themes in showcase mode
  - choose plausible medium-length content in default mode
- `message_timestamp`
  - generate sequential timestamps using inferred base and deltas
- `conversation_title`
  - synthesize short titles aligned with the selected conversation theme

### C3. Drive Structure From Relational Annotations

Relational generation rules should become:

- `x-polylogue-foreign-key`
  - build actual reference graphs rather than random IDs
- `x-polylogue-time-delta`
  - produce coherent time relationships
- `x-polylogue-mutually-exclusive`
  - never populate conflicting fields together
- `x-polylogue-string-length`
  - generate realistic output density for status fields, titles, code blocks,
    and long text bodies

### C4. Minimize Provider-Specific Fixups

The target state is:

- provider config describes only encoding and envelope layout
- semantic correctness comes from the schema
- graph correctness comes from relational annotations

Manual provider logic should remain only for:

- top-level JSON vs JSONL
- known container placement
- record framing or file layout

### C5. Preserve Showcase Quality

Showcase mode should remain visually coherent. It should not regress to random
gibberish simply because generation became more generic.

Recommended approach:

- keep `_SHOWCASE_THEMES`
- bind themes to `conversation_title`, `message_body`, and optional instructions
- allow deterministic seed-driven output so demos and QA are reproducible

## Files to Change

- `polylogue/schemas/synthetic.py`
- `polylogue/schemas/registry.py`
- `polylogue/showcase/fixtures/`
- tests such as:
  - `tests/unit/core/test_synthetic_semantics.py`
  - `tests/unit/core/test_synthetic_relations.py`
  - `tests/unit/core/test_synthetic_zero_knowledge.py`

## Verification

- Parse-roundtrip tests for every provider.
- Snapshot-like tests for showcase-mode outputs.
- New-provider-shaped fixture tests proving semantic generation works even when
  field names differ but inferred semantics are present.

## Exit Criteria

- Most existing `_fix_*` semantic hacks are removed.
- Graph and time coherence come from annotations, not provider branches.
- Showcase outputs remain high quality and deterministic.

## Workstream D: Local Schema Clustering, Versioning, and Promotion

## Goal

Move from one merged schema per provider to a local operator workflow that can
cluster distinct structures, register versions, compare them, and promote them.

## Why This Matters

Without clustering, long-lived providers accumulate muddy "union schemas" that
become less useful over time.

## Implementation Plan

### D1. Promote Structural Fingerprints Into Real Clusters

Current structural fingerprinting should become the basis of version groups.

Each sample should be assignable to:

- provider
- cluster fingerprint
- optional cohort metadata:
  - source path
  - file cohort
  - acquisition date window
  - export date

### D2. Introduce a Cluster Manifest

For each generated cluster, store metadata such as:

- provider
- cluster ID
- sample count
- first seen / last seen
- representative paths
- dominant keys
- confidence
- promoted version if any

Suggested storage shape under runtime schema storage:

- `schemas/<provider>/clusters/<cluster-id>.schema.json.gz`
- `schemas/<provider>/manifest.json`
- `schemas/<provider>/versions/v2.schema.json.gz`

Use a pointer file or manifest entry for "latest" rather than relying on
filesystem symlinks alone.

### D3. Add Explicit CLI Surfaces for Schema Operations

Add a new `schema` command group for operator workflows.

Target commands:

- `polylogue schema infer --provider chatgpt`
- `polylogue schema infer --provider chatgpt --cluster`
- `polylogue schema list --provider chatgpt`
- `polylogue schema compare --provider chatgpt --from v2 --to v3`
- `polylogue schema promote --provider chatgpt --cluster <id>`
- `polylogue schema explain --provider chatgpt --version latest`

### D4. Teach Compare to Classify Changes

Comparison output should explicitly classify:

- additive changes
- subtractive changes
- type mutations
- changed requiredness
- changed semantic annotations
- changed relational annotations

This should be available in:

- JSON
- plain text
- Markdown report form

### D5. Integrate Promotion With Validation

Once promoted, new versions should automatically become visible to:

- `SchemaRegistry`
- `SchemaValidator`
- synthetic generation
- `check --schemas`

## Files to Change

- `polylogue/schemas/registry.py`
- `polylogue/schemas/schema_generation.py`
- `polylogue/schemas/validator.py`
- `polylogue/cli/click_app.py`
- new CLI command module such as:
  - `polylogue/cli/commands/schema.py`
- tests such as:
  - `tests/unit/core/test_schema_registry_versions.py`
  - `tests/unit/core/test_schema_cluster_assignment.py`
  - `tests/integration/test_schema_operator_workflow.py`

## Verification

- Generate clustered schemas from mixed-shape corpora in tests.
- Compare versions and assert correct classification.
- Promote a version and confirm validator + synthetic generator pick it up.

## Exit Criteria

- Providers can have multiple promoted versions.
- Clustered inference is reviewable and operator-driven.
- Schema compare produces durable artifacts.

## Workstream E: Ingestion Hostility, Partial Corruption, and Chronological Fuzzing

## Goal

Prove the ingestion pipeline behaves correctly under malformed records,
mixed-valid batches, interrupted runs, and extreme timestamp conditions.

## Why This Matters

Polylogue ingests long-lived export formats and large archives. It must isolate
bad records without poisoning the rest of a run.

## Implementation Plan

### E1. Add Large-Batch Partial-Corruption Tests

Add end-to-end ingestion tests that:

- generate a large JSONL file
- corrupt exactly one line in the middle
- run `polylogue run`
- assert:
  - valid records still ingest
  - malformed records are isolated
  - counts/logs reflect the failure
  - rerunning remains idempotent

Variants:

- one malformed line
- one truncated final line
- one line with bad UTF-8
- one line with wrong provider envelope

### E2. Improve Parse Error Context for Verification

Where needed, enrich parsing and acquisition error reporting so tests can assert:

- source path
- line number
- provider
- failure class
- quarantine or skip outcome

This likely requires small changes in:

- acquisition path
- raw payload decoding
- parsing service summaries
- observer output

### E3. Add Mid-Run Interruption Tests

Add subprocess-based or signal-driven tests that:

- begin a real ingest run
- interrupt mid-run
- verify:
  - database remains consistent
  - no partial attachment corruption
  - rerun picks up cleanly
  - idempotency and content-hash checks still behave

This is distinct from watch-loop `KeyboardInterrupt` behavior. The target is
mid-ingest interruption, not only loop cancellation.

### E4. Add Chronological Fuzzing

Create end-to-end chronology tests using records with:

- 1970-adjacent timestamps
- 2038-adjacent timestamps
- far-future-but-valid timestamps
- tomorrow
- mixed numeric/string/ISO formats
- missing timestamps alongside present ones

Verify:

- sorting is stable
- indexing works
- rendering does not crash
- timestamp display stays sensible
- out-of-range values degrade safely

### E5. Add Archive-Sized Batch Helpers

Create test helpers for:

- large JSONL generation
- controlled line corruption
- deterministic timestamp pattern generation
- rerun assertions

Suggested helper area:

- `tests/infra/large_batches.py`
- `tests/infra/chaos_sources.py`

## Files to Change

- `polylogue/pipeline/services/parsing.py`
- `polylogue/lib/raw_payload.py`
- `polylogue/lib/timestamps.py`
- `polylogue/pipeline/runner.py`
- `polylogue/cli/run_observers.py`
- tests such as:
  - `tests/integration/test_ingestion_chaos.py`
  - `tests/integration/test_ingestion_interrupts.py`
  - `tests/integration/test_chronology_extremes.py`

## Verification

- End-to-end runs against generated large JSONL fixtures.
- Subprocess interruption tests.
- Chronology-focused search/render assertions.

## Exit Criteria

- One malformed record no longer implies one failed batch.
- Interruption tests prove restart cleanliness.
- Extreme chronology cases are handled explicitly and reproducibly.

## Workstream F: Deterministic PTY/ANSI Regression Testing

## Goal

Add mathematical regression testing for terminal output, distinct from visual
GIF recording and distinct from behavioral TUI pilot tests.

## Why This Matters

Polylogue already has:

- Rich progress output
- CLI presentation logic
- Textual UI behavior tests
- VHS recordings

What it lacks is deterministic verification of actual terminal rendering.

## Implementation Plan

### F1. Add a PTY Harness

Add a subprocess PTY runner for tests, separate from plain subprocess helpers.

Suggested helper module:

- `tests/infra/pty_cli.py`

Capabilities:

- run a command under a pseudo-terminal
- fix terminal width and height
- capture raw ANSI output
- capture final virtual terminal grid

### F2. Add a Virtual Terminal Grid Renderer

Use a terminal emulator library such as `pyte` to convert ANSI output into a
stable final-grid representation.

This allows snapshotting:

- final terminal state
- visible lines
- cursor position if needed

Preferred snapshot target:

- final grid, not raw frame-by-frame escape noise

### F3. Freeze Time and Progress Where Needed

To make terminal output deterministic:

- inject a frozen clock for progress and elapsed time where feasible
- provide deterministic progress observers for tests
- suppress or stabilize transient timers

This may require:

- a clock abstraction
- test-only progress observer variants
- stable timing mode for CLI rendering

### F4. Add Snapshot Tests for Representative CLI Scenarios

Snapshot candidates:

- `polylogue run --preview`
- `polylogue run --source inbox`
- `polylogue check`
- `polylogue --latest`
- selected error surfaces

Snapshot stores can continue using syrupy if convenient, but the captured
object should be the stable terminal grid string.

## Files to Change

- `polylogue/ui/__init__.py`
- `polylogue/cli/run_observers.py`
- `tests/infra/cli_subprocess.py`
- new helper modules such as:
  - `tests/infra/pty_cli.py`
  - `tests/infra/frozen_clock.py`
- tests such as:
  - `tests/unit/cli/test_terminal_snapshots.py`
  - `tests/integration/test_cli_pty_contracts.py`

## Verification

- Snapshot tests are stable across repeated local runs.
- Snapshot tests are stable in CI.
- Progress/spinner output no longer flakes due to timing.

## Exit Criteria

- Terminal rendering has deterministic snapshot coverage.
- Snapshot failures are meaningful diffs, not timing noise.

## Workstream G: Showcase, VHS, and QA Convergence

## Goal

Unify showcase execution, VHS capture, demo generation, and QA artifact
bundling into one coherent system.

## Why This Matters

Polylogue currently has:

- showcase exercises
- cookbook and summary generation
- audit JSON session output
- handcrafted VHS tapes and a demo workflow
- QA directory snapshotting

These should become one pipeline with one scenario catalog and several output
forms.

## Target Artifact Set

A single showcase run should be able to emit:

- summary text
- machine-readable JSON report
- stable Markdown QA session
- stable Markdown cookbook
- stable artifact manifest with hashes
- selected VHS GIF or video captures
- archived QA bundle

## Implementation Plan

### G1. Make Showcase Scenarios the Source of Truth

Extend `polylogue/showcase/exercises.py` so each scenario can describe:

- command or interaction type
- seeded/live mode constraints
- whether VHS capture applies
- expected artifact class
- optional capture script or interaction steps

Two scenario classes may be useful:

- command scenarios
- interactive scenarios

Both should still live in one catalog.

### G2. Generate VHS From Showcase Metadata

Instead of maintaining VHS tapes entirely by hand, generate at least part of
the capture layer from showcase metadata.

Recommended split:

- command-only showcase scenarios can generate tapes automatically
- complex TUI flows can keep explicit interaction scripts, but still be
  registered through the showcase catalog

This yields:

- less divergence between docs and tests
- fewer duplicate definitions
- easier scenario expansion

### G3. Add Stable Markdown QA Session Output

Keep JSON audit sessions for archival and machine use, but also add a stable,
diffable Markdown QA session report.

The committed Markdown artifact should avoid run-specific timestamps in the
body. Time-bearing metadata can remain in archive copies or manifest files.

Recommended outputs:

- `showcase-session.md`
- `showcase-cookbook.md`
- `showcase-manifest.json`

### G4. Teach `qa --only exercises` to Emit Visual + Text Bundles

`polylogue qa --only exercises` should be able to:

- seed data
- run showcase scenarios
- generate report artifacts
- generate VHS captures for selected scenarios
- write a stable bundle to an output directory

Suggested flags:

- `--capture none|vhs|all`
- `--capture-filter`
- `--qa-markdown`
- `--manifest`
- `--archive-qa`

### G5. Teach `qa` to Archive Showcase Bundles

Extend `polylogue qa` so it can archive:

- showcase text artifacts
- showcase visual artifacts
- manifests
- schema compare outputs
- benchmark summaries

Rather than only snapshotting arbitrary directories, it should understand a
showcase bundle as a first-class source.

### G6. Split Verification and Refresh Workflows

The repository should have two different automation lanes:

- verification lane
  - regenerate stable showcase text artifacts
  - compare against committed outputs
  - fail on drift
- refresh lane
  - regenerate visual assets
  - optionally update committed demo/showcase assets

The verification lane should focus on stable textual and manifest artifacts.
Visual assets can remain in the refresh lane unless stable visual hashing
becomes good enough for strict gating.

## Files to Change

- `polylogue/showcase/exercises.py`
- `polylogue/showcase/runner.py`
- `polylogue/showcase/report.py`
- `polylogue/cli/commands/demo.py`
- `polylogue/cli/commands/qa.py`
- `demos/generate.sh`
- `demos/tapes/`
- `.github/workflows/demos.yml`
- add or update workflows for showcase verification
- tests such as:
  - `tests/unit/showcase/test_exercise_catalog.py`
  - `tests/unit/showcase/test_report.py`
  - `tests/integration/test_showcase_bundle.py`

## Verification

- Deterministic showcase bundle generation with fixed seed.
- Stable Markdown artifacts under repeated generation.
- VHS capture works from showcase-defined scenarios.
- `qa` can archive a showcase bundle with one command.

## Exit Criteria

- Showcase is the canonical scenario system.
- VHS capture is wired into that system.
- QA archiving understands showcase bundles directly.

## Workstream H: Data-Gravity and Long-Haul Validation

## Goal

Add a validation harness for very large archives, long rebuilds, and heavy
storage/index workloads without making default CI unusably slow.

## Why This Matters

Polylogue is an archive tool. Its failure modes at scale are different from
its failure modes on small fixtures.

## Implementation Plan

### H1. Preserve Fast CI and Moderate-Scale Budgets

Keep the current moderate-scale budgets as the default fast regression lane.

Those budgets already exercise:

- list performance
- batch retrieval
- FTS lookup
- stats filter pushdown
- semantic filter pushdown

### H2. Add a Synthetic Archive Generator for Large Campaigns

Create a synthetic large-archive generator that can produce:

- millions of messages
- multiple providers
- coherent timestamps
- realistic message length distributions
- optional content blocks

This should be built on top of the improved semantic synthetic engine from
Workstreams B and C.

### H3. Add Long-Haul Campaign Scripts

Add explicit benchmark/stress scripts for:

- full FTS rebuild
- incremental index updates
- schema application on large DBs
- common filter scans
- archive startup and health checks

Recommended storage for results:

- `docs/benchmark-campaigns/`

### H4. Capture Operational Metrics

Each long-haul campaign should record:

- wall time
- DB size before/after
- indexed row counts
- peak RSS if measurable
- lock durations if measurable
- success/failure summary

### H5. Define Multiple Scale Lanes

Recommended scale lanes:

- fast CI lane
  - existing moderate budgets
- slow local lane
  - hundreds of thousands to low millions
- stretch lane
  - maximal archive sizes for operator-run campaigns

Only the fast lane should run by default in CI.

## Files to Change

- `polylogue/schemas/synthetic.py`
- `tests/unit/storage/test_scale.py`
- new scripts or devtools such as:
  - `devtools/large_archive_campaign.py`
  - `devtools/rebuild_index_campaign.py`
- `docs/benchmark-campaigns/README.md`

## Verification

- Campaign scripts produce durable Markdown and JSON reports.
- Moderate-scale budgets continue to protect default CI.
- Large-lane runs are reproducible and reviewable.

## Exit Criteria

- Polylogue has a documented path for validating large-archive behavior.
- Benchmark artifacts are reproducible and stored durably.

## Recommended Execution Order

The recommended order is designed to maximize leverage and keep each later
workstream easier to verify than it would be otherwise.

### Phase 1: Machine Contract and Runtime Health

Do first:

- Workstream A

Reason:

- It improves every later automation surface.
- It creates the machine-safe CLI contract needed by showcase and QA tooling.

### Phase 2: Semantic Inference Foundation

Do second:

- Workstream B

Reason:

- It provides the schema vocabulary needed by synthetic generation and schema
  promotion workflows.

### Phase 3: Synthetic Rewrite and Schema Promotion

Do third:

- Workstream C
- Workstream D

Reason:

- Semantic annotations become useful immediately.
- Operator-driven clustered versioning becomes reviewable once schemas have
  richer semantics.

### Phase 4: Hostility and Chronology Coverage

Do fourth:

- Workstream E

Reason:

- It validates the pipeline more deeply once generator and schema surfaces are
  richer and more expressive.

### Phase 5: Deterministic Terminal Regression Infrastructure

Do fifth:

- Workstream F

Reason:

- It provides the deterministic assertion layer needed before fully unifying
  showcase and visual generation.

### Phase 6: Showcase, VHS, and QA Convergence

Do sixth:

- Workstream G

Reason:

- It benefits from stable CLI contracts, semantic synthetic data, deterministic
  capture primitives, and richer validation surfaces.

### Phase 7: Data-Gravity Campaigns

Do seventh:

- Workstream H

Reason:

- It is easiest to validate at the end, once semantic generation, pipeline
  hardening, and showcase/archive tooling are already stronger.

## Suggested Milestone Deliverables

### Milestone 1

- JSON machine-error contract
- `check --runtime`
- subprocess machine-contract tests

### Milestone 2

- first semantic-role annotations
- first relational annotations
- golden annotation tests

### Milestone 3

- provider-agnostic semantic generation path
- reduced provider fixup surface
- clustered schema manifests and compare reports

### Milestone 4

- ingestion chaos suite
- interruption suite
- chronology suite

### Milestone 5

- PTY harness
- terminal grid snapshots
- deterministic progress/time handling

### Milestone 6

- showcase-driven VHS generation
- stable Markdown QA session
- QA bundle archiving for showcase outputs

### Milestone 7

- large-archive campaign scripts
- benchmark reports under `docs/benchmark-campaigns/`

## Final Success Criteria

This program is complete when all of the following are true:

- JSON-capable CLI commands preserve structured output on failure.
- `check` is the canonical runtime verification surface.
- Schemas encode semantic and relational meaning, not only type shape.
- Synthetic generation is mostly driven by schema annotations rather than
  provider-specific semantic fixups.
- Schema versioning and promotion are explicit, clustered, and reviewable.
- Ingestion is proven resilient under corruption, interruption, and chronology
  extremes.
- Terminal rendering has deterministic regression coverage.
- Showcase, VHS capture, and QA archiving are one coherent workflow.
- Large-archive validation is a documented, reproducible practice rather than an
  ad hoc manual exercise.
