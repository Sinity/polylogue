# Source, Beads, and history evidence

## Snapshot and repository evidence

The snapshot identity comes from the supplied `polylogue-manifest.json`, cross-checked with the bundled Git repository:

- generation timestamp `2026-07-18T013442Z`;
- branch `master`;
- commit `bf8191b3f56aa40da8f271df7f3385c712825497`;
- dirty working tree;
- remote default branch at the same commit.

The supplied dirty source delta consists of three files: `polylogue/archive/query/unit_results.py`, `polylogue/daemon/http.py`, and `polylogue/hooks/__init__.py`. Its reconstructed patch SHA-256 is `f30102453e3576c9049eb65dcea7265fa99ac694c45531b06b0a34dd85279e9f` and its size is 2,272 bytes.

The Chisel worktree archive excludes many tracked hidden/generated files. Their apparent deletions were excluded from all diff generation and are not implementation intent.

## Architectural source anchors

`AGENTS.md` establishes the central constraints used by this implementation:

- domain semantics belong in the substrate rather than a parallel feature framework;
- raw/source data is authoritative and normalized/index data is reproducible;
- structured tool/action fields own outcome semantics;
- missing information remains `NULL` rather than being guessed;
- focused tests should exercise the real production route;
- new production modules require topology projection regeneration.

Artifact and raw-path anchors:

- `polylogue/archive/artifact_taxonomy/models.py`: canonical artifact-kind vocabulary.
- `polylogue/archive/artifact_taxonomy/runtime.py`: path and payload classification, including the prior Hermes sidecar exclusion.
- `polylogue/sources/source_acquisition.py` and `source_acquisition_components.py`: canonical whole-file raw acquisition into BlobStore.
- `polylogue/archive/raw_payload/decode.py`: canonical raw decoding, provider inference, and artifact attachment before parse.
- `polylogue/pipeline/services/acquisition.py`: raw-record persistence without semantic parse.
- `polylogue/pipeline/services/ingest_worker.py`: canonical retained-blob decode → artifact plan → shared dispatch → normalized payload route.
- `polylogue/pipeline/services/parsing.py` and parsing workflow/batch code: materialization of retained raw records.
- `polylogue/sources/source_parsing.py` and `emitter.py`: direct/re-ingest route and raw-capture conventions.
- `polylogue/sources/sqlite_snapshot.py` plus Hermes state tests: precedent for exact retained SQLite snapshots without reading live mutable bytes directly.

Parser and fidelity anchors:

- `polylogue/sources/dispatch.py`: shared provider detection, lowering, and parser invocation.
- `polylogue/sources/parsers/base.py`: typed parsed sessions, messages, content blocks, and session events.
- `polylogue/sources/parsers/hermes_state.py`: Hermes profile-qualified identity, raw reproducibility, and machine-readable fidelity conventions.
- `polylogue/sources/parsers/hermes_spans.py`: Hermes observer-event and delivery-correlation conventions.
- `polylogue/sources/import_explain.py`: non-mutating explain route and fidelity projection.
- `polylogue/sources/origin_specs.py`: origin admission modes, parser paths, and fixture ownership.

Normalized storage/read anchors:

- existing message/block materialization writes paired tool-use/result structures;
- the existing `actions` relation projects command, output, `is_error`, and exit code structurally;
- `session_events` retains typed lossless producer/session evidence;
- `polylogue/storage/sqlite/queries/session_events.py`, query-store mixins, backend mixins, and repository mixins are the established read path.

These anchors led to the chosen design: verification rows become normal tool structures plus typed events, rather than a bespoke verification table or a query that reinterprets output prose.

## Test anchors

The following existing tests constrained behavior:

- `tests/unit/sources/test_parsers_local_agent.py`: Hermes sidecar files are non-session terminal artifacts.
- `tests/unit/core/test_sampling.py`: binary verification evidence is intentionally excluded before JSON decoding and represented in the terminal ledger.
- `tests/unit/sources/test_source_laws.py`: raw capture, grouped whole-file retention, emitter reuse, provider detection, source iteration, and BlobStore streaming contracts.
- `tests/unit/sources/test_hermes_import_explain.py`: Hermes fidelity payload shape and caveat behavior.
- `tests/unit/sources/test_origin_specs.py`: OriginSpec parity and completeness.
- `tests/unit/cli/test_import_explain.py`: public explain-import counts and bounded decode behavior.
- pipeline acquisition/parsing tests: retained raw records are decoded later through the canonical worker and replayed idempotently.

The new test file uses those same production seams. It does not instantiate a private verification-only persistence layer.

## Beads findings

### `polylogue-fs1`

This epic requires an evidence-honest Hermes bridge: retained artifact identity, stable correlation, explicit missingness, and no fidelity claims beyond observed bytes. The implementation follows that direction by separating the binary artifact boundary from the provisional JSON mapping.

### `polylogue-fs1.1`

The reproducible state-database lane established the precedent of snapshotting mutable SQLite bytes before parse, profile-qualifying identity, and retaining lossless events. Verification evidence reuses the profile-aware correlation convention but does not pretend its unknown SQLite schema matches `state.db`.

### `polylogue-fs1.2` and `polylogue-fs1.2.1`

The observer lane requires producer-positive session identity, append/replay discipline, and visible unmatched debt. The follow-up explicitly says synthetic repository markers cannot prove real wire fidelity. That directly governs the verification fixture: it is useful for integration behavior but cannot make the producer mapping exact.

### `polylogue-fs1.3`

This closed lane introduced machine-readable fidelity bands: exact, absent, redacted, degraded, and inferred. The verification declaration uses those existing types and keeps field mappings below exact until real bytes exist.

### `polylogue-fs1.14`

This lane requires stable profile-qualified joins, independently retained raw artifacts, non-destructive enrichment, and explicit unmatched/conflict debt. The query first honors exact profile-qualified identity, permits unqualified resolution only when unique, and exposes unmatched/ambiguous states.

### `polylogue-fs1.15`

This lane requires typed degraded read/health states without adding another database. The implementation adds typed row/correlation models and reuses existing archive tables.

### `polylogue-segf`

This record formalized that a self-authored marker fixture is not real producer evidence. It was later absorbed into the real ATIF/ATOF proof work. The same evidence standard is applied here.

### `polylogue-1xc.14.1.1`

The terminal-artifact ledger notes record that two real 32,768-byte Hermes `verification_evidence.db` blobs began with the SQLite header. They were not malformed JSON; they were typed non-session sidecars that sampling must intentionally exclude before decode. This proves artifact identity at the filename/header level, but it does not reveal table schema or row semantics.

## Relevant history

The following commits were inspected as current architectural precedent:

- `9e92b6b6d7656f315dd491b8f5f59049d104e868` — 2026-07-10 — `fix(hermes): make state database imports reproducible (#2639) (#2639)`.
- `352051b57921f69fbc9e1d5a7ea150fb4616589d` — 2026-07-13 — `feat(import): declare Hermes source fidelity (#2789)`.
- `17e0137095603e95bcaede067ccf5a9432bf39e8` — 2026-07-14 — `feat(hermes): lifecycle-event spool, ATIF importer, delivery correlation (#2876)`.
- `bec46ea3e862f4aafdd36c9a82b39da8807ea6e7` — 2026-07-14 — `fix(hermes): verify live ATIF export fidelity (#2903)`.
- `c20286459cf2c3d1e4c968a8584f13e7cd382ff2` — 2026-07-17 — `feat(schemas): derive archive workload profiles (#2934)`.
- `df37b5bc44d900d8886154a335ebf5d07fde16b0` — 2026-07-17 — `fix(schemas): classify evidence sidecars before decoding (#2973)`.

The real ATIF proof in `bec46ea...` supersedes older notes that no real ATIF bytes existed. It does not establish anything about the private verification-ledger schema. The latest sidecar classification in `df37b5...` remains authoritative for binary `verification_evidence.db`: identify it before generic JSON decode and record intentional exclusion.

## Contradictions resolved

Older planning language could be read as suggesting a marker-only Hermes observer design or a direct reader for every SQLite sidecar. Current source and later Beads/history supersede that interpretation:

- real ATIF/ATOF and verification-ledger evidence are distinct artifacts;
- real ATIF proof does not prove verification-ledger columns;
- the terminal sidecar classification is correct and must remain non-session until an acquisition/export contract exists;
- this mission requests structured verification evidence, so the implementation adds a versioned replaceable export boundary while preserving the binary terminal classification;
- a new standalone SQLite reader would violate both the mission boundary and the repository raw/parser/normalized route.

No evidence was found that a stable public Hermes verification-ledger schema, exporter command, table layout, or status vocabulary already exists in the supplied source. The package therefore does not invent one as fact.

## Fidelity boundary and falsification evidence

The package can be falsified or refined by the paired local lane. The following observations would require code or fixture changes:

- the real file uses different table/column names or nested encoding;
- command/canonical command are not independently stored;
- status values have different terminal meanings;
- exit code uses a sentinel other than SQL `NULL` for unknown;
- changed paths are JSON text, a child table, a delimiter string, or unordered;
- ledger identity points to a different Hermes session key than `sessions.native_id`;
- profile qualification is absent or encoded differently;
- row order is not stable, requires an explicit sequence column, or permits updates/deletes;
- producer schema/version/hash metadata is stored elsewhere.

Those are not implementation defects to hide with permissive parsing. They are the exact observations required to revise the versioned acquisition envelope and promote only supported fidelity capabilities.
