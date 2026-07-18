# Hermes verification-ledger ingestion and fidelity harness — handoff

## Result

This package implements an integration-ready Hermes verification-evidence vertical slice against the supplied Polylogue snapshot. It makes verification evidence a typed archive artifact, admits a versioned producer/export JSON envelope through the existing raw → parser → normalized route, writes structural tool outcomes into the existing message/block/action substrate, retains lossless row events, correlates rows deterministically to archived Hermes sessions, and exposes a typed repository/backend query.

The implementation deliberately does not open `verification_evidence.db`, add a SQLite side reader, classify outcomes from prose, touch a live Hermes installation, or claim that the synthetic fixture came from real producer bytes. The binary sidecar is identified and retained as a terminal artifact; parsing requires the replaceable JSON acquisition envelope.

## Snapshot identity and authority

The authority was the Chisel snapshot archive supplied with this task. `polylogue-manifest.json` records:

- generated: `2026-07-18T013442Z`
- branch: `master`
- commit: `bf8191b3f56aa40da8f271df7f3385c712825497`
- dirty: `true`
- local `origin/master`: the same commit

The working-tree payload carried three real source modifications beyond that commit:

- `polylogue/archive/query/unit_results.py`
- `polylogue/daemon/http.py`
- `polylogue/hooks/__init__.py`

Their reconstructed dirty patch is 2,272 bytes with SHA-256 `f30102453e3576c9049eb65dcea7265fa99ac694c45531b06b0a34dd85279e9f`. It is part of the named snapshot authority but is not copied into this result ZIP and is not duplicated by `PATCH.diff`.

The extracted Chisel working tree omits many tracked hidden/generated files, so an ordinary unrestricted `git status` reports approximately 1,486 apparent deletions. Those are archive-exclusion artifacts, not proposed repository deletions. This package changes no such file and proposes no dominated deletion of an existing test or helper.

`PATCH.diff` is a 75,780-byte, 1,760-line unified diff over 19 files, with SHA-256 `5efdf46b81043d169fedbb4a5019851488888cff36917c88f135332c95801fcc`. It contains 1,269 insertions and 25 deletions. The topology files are emitted as textual unified hunks with `--text`, rather than opaque Git binary patches.

## Evidence inspected

Repository instructions and production dependencies were followed beyond the obvious parser files. The main anchors were:

- `AGENTS.md` for substrate ownership, retained raw authority, rebuildable normalized/index tiers, structural action outcome conventions, `NULL`-means-unknown semantics, focused verification, and topology projection requirements.
- Artifact admission: `polylogue/archive/artifact_taxonomy/{models,runtime}.py`, `polylogue/sources/origin_specs.py`, raw artifact inspection/sampling tests, and source walking.
- Raw routes: `polylogue/sources/source_acquisition.py`, `source_acquisition_components.py`, `source_parsing.py`, `emitter.py`, `polylogue/archive/raw_payload/decode.py`, `polylogue/pipeline/services/{acquisition,parsing,archive_ingest,ingest_worker}.py`, and BlobStore/publication seams.
- Parser conventions: Hermes state and ATIF parsers, dispatch lowering, parsed message/block/session-event models, schema resolution, import explain, and fidelity payloads.
- Normalized persistence/read route: message/block/action materialization, `session_events`, `sessions`, query-store/backend/repository mixins, and existing async ingest helpers.
- Tests: local-agent/Hermes classification, raw-capture source laws, sampling terminal ledgers, Hermes import explain, origin specs, CLI import explain, pipeline acquisition/parsing behavior, and archive action/event persistence.
- Beads: `polylogue-fs1`, `fs1.1`, `fs1.2`, `fs1.2.1`, `fs1.3`, `fs1.14`, `fs1.15`, `polylogue-segf`, and `polylogue-1xc.14.1.1`.
- History: the reproducible Hermes state import, fidelity declaration, lifecycle/ATIF work, real ATIF proof, workload-profile work, and terminal sidecar classification commits listed in `EVIDENCE.md`.

## Implemented mechanism

### Artifact boundary and acquisition

`ArtifactKind.VERIFICATION_EVIDENCE` distinguishes the verification ledger from generic metadata. A Hermes path named `verification_evidence.db`, `.sqlite`, or `.sqlite3` is typed as verification evidence but remains `parse_as_session=False` and `schema_eligible=False`; its reason explicitly requires a versioned JSON acquisition export.

The admitted envelope is structurally versioned:

- `schema_version = "polylogue.hermes.verification-evidence/v1"`
- `artifact_kind = "hermes_verification_evidence"`
- `rows = [...]`

The normal provider detector, artifact classifier, payload lowering, and Hermes dispatch recognize that envelope. No filename-only parser shortcut was introduced.

Both production raw routes preserve the source artifact. The canonical `AcquisitionService` already streams the whole file into BlobStore before `ParsingService` decodes it. For the direct/re-ingest `parse_one_source_path(..., capture_raw=True)` route, a bounded `ijson` top-level probe identifies the envelope without materializing its row array, writes the whole source file to BlobStore, and parses the retained blob. `_SessionEmitter` now honors a precomputed whole-artifact raw identity for ordinary JSON documents, so all observer sessions share the exact source blob instead of reserialized logical JSON.

### Normalization and outcome semantics

Rows are normalized into deterministic observer sessions grouped by producer-positive session identity. Every object row produces:

- one assistant message with paired `tool_use` and `tool_result` blocks;
- tool name `hermes_verification`;
- structured input for command, producer-supplied canonical command, kind, scope, status, identity, and changed paths;
- structural result fields for output summary, exit code, and `is_error`;
- one lossless `hermes_verification_evidence` session event containing the normalized row and row-quality state.

A non-object row is not dropped. It becomes a deterministic `malformed` event with unknown normalized fields and no fabricated tool message.

Outcome rules are structural:

- an explicit exit code is authoritative: `0 → is_error=False`, nonzero → `True`;
- with `exit_code=NULL`, a recognized explicit success/failure status may establish `is_error`;
- unrecognized or nonterminal statuses such as `running` and `skipped` preserve `is_error=NULL`;
- a recognized status conflicting with an exit code is retained as `outcome_conflict`, while exit code remains the outcome authority;
- `output_summary` is never scanned to infer failure.

The parser never derives `canonical_command` from `command`. Changed paths preserve `NULL` as unknown, preserve an explicit empty list as `()`, retain first-seen order, and remove exact duplicates.

### Deterministic identity and correlation

Observer session IDs derive from the export ID and correlation key using repository hashing helpers. Generated evidence/message/tool IDs derive from stable export, row-position, and row-content inputs. Replaying identical bytes therefore reproduces the same normalized identities.

The typed read route returns `HermesVerificationEvidence` with:

- `complete`, `partial`, `malformed`, or `outcome_conflict` row state;
- `matched`, `missing_identity`, `unmatched_identity`, or `ambiguous_identity` correlation state;
- producer-native identity, raw session identity, and resolved archive session ID kept separate.

Correlation first tries an exact profile-qualified Hermes native ID. A raw unqualified session ID may resolve only when exactly one profile-qualified candidate exists. Zero matches remain unmatched; multiple candidates remain ambiguous. The query does not guess or merge destructively.

### Query/read surface

The read query starts from `session_events` and left-joins the existing `actions` relation on source message ID plus `tool_name='hermes_verification'`. Command, output summary, exit code, and `is_error` therefore come from the archive-wide structural action view rather than duplicate prose parsing. Canonical command, kind, scope, status, changed paths, identity, and row state come from the lossless typed event.

The route is exposed through:

- `SQLiteQueryStoreArchiveMixin.get_hermes_verification_evidence`
- `SQLiteArchiveMixin.get_hermes_verification_evidence`
- `RepositoryArchiveSessionMixin.get_hermes_verification_evidence`

It supports optional observer-session and resolved target-session filters. No schema migration or new database is required because the implementation reuses `sessions`, `messages`, `blocks`, `actions`, and `session_events`.

### Fidelity declaration

The synthetic fixture declares `fixture_provenance: "synthetic; no real Hermes producer bytes"`, with `source_sha256` and `producer_schema_version` set to `null`.

Import explain declares exact reproducibility only for the retained JSON export artifact and deterministic replay of those bytes. Every producer-field capability remains `inferred`, `degraded`, or `absent`. The declaration states that the upstream SQLite-to-JSON mapping is unverified and names the real-byte checks required before exactness:

- real table and column names;
- SQLite value types and nullability;
- status vocabulary and terminal semantics;
- canonical-command meaning;
- changed-path encoding and ordering;
- session identity/profile rules;
- row ordering/deduplication behavior;
- producer schema/artifact revision and hash rules.

A later local fixture can replace or supplement the synthetic envelope without changing parser architecture.

## Changed files

Production domain and parser files:

- `polylogue/archive/artifact_taxonomy/models.py`
- `polylogue/archive/artifact_taxonomy/runtime.py`
- `polylogue/archive/session/hermes_verification.py` (new)
- `polylogue/sources/dispatch.py`
- `polylogue/sources/emitter.py`
- `polylogue/sources/import_explain.py`
- `polylogue/sources/origin_specs.py`
- `polylogue/sources/parsers/hermes_verification.py` (new)
- `polylogue/sources/source_parsing.py`

Read route:

- `polylogue/storage/repository/archive/sessions.py`
- `polylogue/storage/sqlite/async_sqlite_archive.py`
- `polylogue/storage/sqlite/queries/session_events.py`
- `polylogue/storage/sqlite/query_store_archive.py`

Fixtures and tests:

- `tests/fixtures/hermes/verification_evidence/synthetic-v1.json` (new)
- `tests/unit/sources/parsers/test_hermes_verification.py` (new)
- `tests/unit/core/test_sampling.py`
- `tests/unit/sources/test_parsers_local_agent.py`

Generated topology projections:

- `docs/plans/topology-target.yaml`
- `docs/topology-status.md`

## Acceptance matrix

| Requirement | Result | Concrete evidence |
|---|---|---|
| Identify/acquire verification artifact without bespoke DB reader | Implemented | Typed binary sidecar boundary plus versioned JSON envelope; boundary test |
| Reuse raw → parser → normalized route | Implemented | Direct retained-file route and canonical `AcquisitionService → ParsingService` end-to-end test |
| Normalize command/canonical/kind/scope/status/exit/output/paths | Implemented | Parser event/tool pair and typed query assertions for every field |
| Preserve unknown as `NULL` | Implemented | Running row retains null canonical command, exit code, outcome, output, and paths |
| Structural status/exit semantics | Implemented | pass, failure, unknown, and status/exit conflict assertions; skipped prose does not become error |
| Deterministic session correlation | Implemented | exact profile-qualified match, unmatched identity, missing identity, and fail-closed query logic |
| Typed degraded row states | Implemented | partial, malformed, and outcome-conflict rows retained and queried |
| Idempotent replay | Implemented | second canonical parse writes no changed session IDs; table counts and typed rows are unchanged |
| Query/read route | Implemented | backend and repository methods; target-session filter assertion |
| Synthetic producer-shaped corpus | Implemented | seven rows covering pass/fail/partial/conflict/unmatched/missing/malformed, explicitly synthetic |
| Fidelity declaration | Implemented | import-explain test rejects exact field capabilities and requires local real-byte validation |
| No watcher/full state.db expansion | Preserved | no watcher, exporter, SQLite schema reader, or new DB added |
| Topology projection | Updated | render and check commands pass |

## Apply order

`PATCH.diff` is based on the exact dirty snapshot, not a clean approximation.

1. Start from the supplied Chisel working tree, or check out `bf8191b3f56aa40da8f271df7f3385c712825497` and restore the three supplied dirty modifications from the original snapshot.
2. Run `git apply --check PATCH.diff`.
3. Run `git apply PATCH.diff`.
4. Run the focused commands in `TESTS.md`.
5. Run the paired local real-byte validation before promoting any producer mapping from inferred/degraded to exact.

An independent clean-clone apply check was performed by cloning `polylogue-all-refs.bundle`, checking out the named commit, applying the reconstructed supplied dirty patch, checking and applying `PATCH.diff`, running `git diff --check`, and comparing all 19 package-touched files byte-for-byte with the implementation worktree. All 19 matched.

## Verification performed

Successful checks:

- New vertical-slice file: `5 passed in 2.28s`.
- Focused and neighboring source/CLI suite: `82 passed in 3.71s`.
- Relevant emitter/raw-capture source-law slice: `11 passed, 113 deselected in 8.85s`.
- Ruff: all changed Python files passed.
- Ruff format check: 16 Python files already formatted.
- Strict MyPy: no issues in 13 changed production files.
- Topology projection/status render and check: passed.
- Unified patch whitespace check: passed.
- Clean clone + supplied dirty patch + package apply check: passed; 19/19 files matched byte-for-byte.

Two broad commands were attempted but are not represented as passing:

- Full `tests/unit/sources/test_source_laws.py` exceeded 300 seconds in this environment; the affected emitter/raw-capture slice above passed.
- `devtools verify --quick` reached repository-wide MyPy and exceeded 300 seconds. Its two initial format findings in modified existing tests were fixed; targeted Ruff, format, strict MyPy, focused tests, and generated topology checks subsequently passed.

No live daemon, live browser, operator archive, secrets, NixOS deployment, current operator worktree, or real `~/.hermes` installation was inspected or modified.

## Risks, remaining work, and iteration value

The principal risk is semantic fidelity at the producer boundary, not archive integration. No real `verification_evidence.db` bytes or producer export were supplied, so table names, columns, value encodings, statuses, identity rules, and ordering remain unproven. The JSON contract is intentionally versioned and replaceable; it is not evidence that Hermes currently emits that shape.

The query assumes archived Hermes target sessions retain their producer native identity in the existing `sessions.native_id` convention. It fails closed for ambiguous unqualified identities, but the local lane must verify which producer ledger identity actually corresponds to that field.

A small repair iteration without new producer evidence could improve ergonomics or add extra corruption cases, but would add limited value because the production route, query, replay, and typed degradation are implemented and checked. A second pass with retained real verification-ledger bytes would add substantial value: it could define the actual exporter, lock the versioned fixture to producer schema/hash evidence, adjust field normalization from observed types, and promote only byte-proven fidelity capabilities to exact.
