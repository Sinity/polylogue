# Hermes verification-ledger test design and execution record

## Test intent

The test package is built around production dependencies rather than parser-only snapshots. It exercises artifact admission, exact raw retention, shared dispatch, canonical acquisition and parse services, ordinary normalized persistence, the existing action relation, lossless session events, deterministic replay, target correlation, repository filtering, and import-fidelity reporting.

The synthetic corpus contains seven rows:

1. matched success with exit code `0`;
2. matched failure with exit code `2`;
3. partial running row with null canonical command, exit code, output, and changed paths;
4. explicit `passed` status conflicting with exit code `1`;
5. profile-qualified unmatched session with `skipped`, null exit code, explicit empty paths, and output text `not run`;
6. missing identity with explicit success status and null exit code;
7. malformed non-object sentinel.

The fixture itself labels its provenance as synthetic and leaves real source hash/schema metadata null.

## New vertical-slice tests

### `test_artifact_boundary_distinguishes_sqlite_sidecar_from_versioned_export`

Production dependencies:

- `classify_artifact_path`
- `ArtifactKind`
- `classify_artifact`
- Hermes payload detection

What it proves:

- binary `verification_evidence.db` is typed verification evidence but not parsed as a session;
- the versioned JSON envelope is typed verification evidence and is parseable;
- provider detection routes the envelope to Hermes.

Anti-vacuity mutation: reverting the binary sidecar to generic metadata, deleting the versioned classifier branch, making the sidecar parseable, or removing Hermes provider detection makes this test fail.

### `test_raw_file_reaches_shared_dispatch_and_preserves_synthetic_provenance`

Production dependencies:

- `parse_one_source_path`
- bounded acquisition-side envelope probe
- BlobStore whole-file retention/publication
- `_SessionEmitter` precomputed raw handling
- ordinary dispatch/parser path
- artifact session events

What it proves:

- one source artifact produces the expected three deterministic observer sessions;
- every session references the same exact source SHA-256 and byte size;
- the retained blob bytes equal the input bytes, including formatting;
- synthetic provenance and null real-source hash remain visible.

Anti-vacuity mutation: removing the whole-artifact probe, parsing the original after a different blob is retained, ignoring `precomputed_raw`, or reverting to per-payload JSON reserialization breaks the hash/size/byte assertions.

### `test_import_explain_declares_synthetic_fidelity_without_real_byte_claims`

Production dependencies:

- import-explain detector/parser route
- Hermes verification fidelity declaration
- typed import payloads

What it proves:

- explain-import reports three sessions and six actions;
- retained JSON replay is the only exact capability;
- producer field mappings are never exact;
- artifact identity remains inferred;
- session correlation and row quality are degraded where fixture rows demand it;
- the caveat explicitly says no real `~/.hermes` bytes were supplied.

Anti-vacuity mutation: promoting command/status/correlation to exact, removing row-quality accounting, routing to generic Hermes JSON fidelity, or deleting the real-byte caveat makes the test fail.

### `test_parser_is_deterministic_and_retains_partial_malformed_and_missing_identity_rows`

Production dependencies:

- Hermes verification normalizer
- deterministic hashing helpers
- parsed session-event model
- row-state logic

What it proves:

- parsing identical logical payloads produces byte-equivalent model dumps;
- all seven rows survive;
- partial nulls remain null;
- missing identity remains explicit;
- malformed input remains a typed event without a fabricated message;
- no producer field capability is declared exact.

Anti-vacuity mutation: random IDs, dropping malformed rows, coercing null paths to empty, deriving a canonical command, or replacing typed row states with generic errors makes the test fail.

### `test_end_to_end_replay_outcomes_correlation_and_typed_query`

Production dependencies:

- `AcquisitionService.acquire_sources`
- streamed BlobStore acquisition and raw record persistence
- `ParsingService.parse_from_raw`
- `ingest_worker` raw-envelope classification/dispatch
- session/message/block/event/action materialization
- `actions` structural view
- typed SQLite query/backend/repository route
- target native-ID correlation

What it proves:

- canonical acquisition retains exact bytes and one raw record;
- canonical parsing produces three observer sessions and no parse failures;
- a replay of the same raw record produces no changed session IDs;
- counts for sessions, messages, blocks, session events, and actions remain unchanged;
- typed query output remains identical across replay;
- command/canonical/kind/scope/status/exit/output/path fields are exposed;
- exit code is authoritative during status conflict;
- running/skipped rows with null exit remain outcome-unknown;
- `not run` output prose is not classified as failure;
- explicit success status with null exit can establish success;
- matched, unmatched, and missing identity states are visible;
- malformed rows have no source message/action;
- repository filtering by resolved target session returns only matched rows.

Representative mutations that must fail:

- Remove paired tool blocks or the `actions` join: command/output/outcome assertions fail.
- Infer error from output prose: the skipped `not run` row becomes an error and fails.
- Coerce null exit/outcome to zero/false: the running and skipped rows fail.
- Let status override nonzero exit: the conflict row fails.
- Drop missing/unmatched rows: correlation cardinality and identity assertions fail.
- Match the unqualified raw ID across multiple profiles without uniqueness: ambiguity semantics fail.
- Make session/event IDs nondeterministic or perform append-only duplicate writes: replay counts/query equality fail.
- Remove the repository target filter: `matched_only` contains unrelated rows and fails.

## Existing regression tests adjusted or exercised

`tests/unit/sources/test_parsers_local_agent.py` continues to prove that the real binary sidecar is terminal/non-session, now with the more precise `verification_evidence` kind.

`tests/unit/core/test_sampling.py` continues to prove that sampling records the sidecar as an intentional typed exclusion rather than attempting UTF-8/JSON decoding.

Neighbor suites cover existing Hermes import explain, OriginSpec parity, CLI import explain, local-agent dispatch behavior, sampling, and source-emitter/raw-capture contracts.

## Commands and honest results

Executed from the supplied snapshot worktree using its `.venv`:

```text
.venv/bin/pytest -q tests/unit/sources/parsers/test_hermes_verification.py
```

Result: `5 passed in 2.28s`.

```text
.venv/bin/pytest -q \
  tests/unit/sources/parsers/test_hermes_verification.py \
  tests/unit/sources/test_parsers_local_agent.py \
  tests/unit/core/test_sampling.py \
  tests/unit/sources/test_hermes_import_explain.py \
  tests/unit/sources/test_origin_specs.py \
  tests/unit/cli/test_import_explain.py
```

Result: `82 passed in 3.71s`.

```text
.venv/bin/pytest -q tests/unit/sources/test_source_laws.py -k 'emitter or raw_capture'
```

Result: `11 passed, 113 deselected in 8.85s`.

```text
.venv/bin/ruff check <16 changed Python files>
.venv/bin/ruff format --check <16 changed Python files>
```

Result: all checks passed; `16 files already formatted`.

```text
.venv/bin/mypy \
  polylogue/archive/artifact_taxonomy/models.py \
  polylogue/archive/artifact_taxonomy/runtime.py \
  polylogue/archive/session/hermes_verification.py \
  polylogue/sources/dispatch.py \
  polylogue/sources/emitter.py \
  polylogue/sources/import_explain.py \
  polylogue/sources/origin_specs.py \
  polylogue/sources/parsers/hermes_verification.py \
  polylogue/sources/source_parsing.py \
  polylogue/storage/repository/archive/sessions.py \
  polylogue/storage/sqlite/async_sqlite_archive.py \
  polylogue/storage/sqlite/queries/session_events.py \
  polylogue/storage/sqlite/query_store_archive.py
```

Result: `Success: no issues found in 13 source files`.

```text
.venv/bin/python -m devtools render topology-projection
.venv/bin/python -m devtools render topology-status
.venv/bin/python -m devtools render topology-projection --check
.venv/bin/python -m devtools render topology-status --check
```

Result: render/check completed successfully; projection contains 1,036 rows.

```text
git diff --check --text -- <19 package paths>
```

Result: passed.

Apply validation:

```text
git clone polylogue-all-refs.bundle <clean-dir>
git checkout bf8191b3f56aa40da8f271df7f3385c712825497
git apply --check <reconstructed-supplied-dirty.patch>
git apply <reconstructed-supplied-dirty.patch>
git apply --check PATCH.diff
git apply PATCH.diff
git diff --check
```

Result: passed. All 19 touched files matched the implementation worktree byte-for-byte.

## Commands attempted but not certified as passing

A full run of `tests/unit/sources/test_source_laws.py` exceeded 300 seconds in this environment. It is marked unverified, not failed; the 11 affected emitter/raw-capture tests passed separately.

`devtools verify --quick` was attempted. It initially identified two formatting changes in modified existing tests; those were applied. The rerun reached repository-wide MyPy and exceeded 300 seconds. The package therefore does not claim a completed repository-wide quick gate. Targeted Ruff, formatting, strict MyPy, tests, topology checks, whitespace checks, and clean apply checks all passed afterward.

## Required local real-byte proof

Before this surface is called producer-exact, the paired local lane must retain a private SQLite snapshot and independently record its hash, size, schema version, table DDL, and row ordering. It must then derive the JSON envelope from actual columns without converting unknown values to defaults. A privacy-safe fixture/checksum derivation may replace or supplement the synthetic fixture.

The local proof must exercise at least: success, failure, null exit, nonterminal/unknown status, changed-path null versus empty, canonical-command presence/absence, profile-qualified and unqualified session identity, duplicate/order behavior, and a malformed/partial producer row if the producer permits one. Fidelity capabilities may be promoted only for mappings demonstrated by those retained bytes.
