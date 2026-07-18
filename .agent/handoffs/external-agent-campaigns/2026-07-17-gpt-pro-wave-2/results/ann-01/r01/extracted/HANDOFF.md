# HANDOFF — Seed annotation ontologies and governed archive-local bootstrap

## Mission and outcome

This patch implements `polylogue-dve1` on Polylogue's existing typed-annotation substrate. It adds five immutable v1 seed schemas, data-only registration into `user.db`, an archive-local candidate/governance lifecycle, and an annotator-facing batch route that can consume operator-promoted schemas without adding archive-specific vocabulary to the process-global registry.

The implementation deliberately does not create a second annotation store or a tag-to-fact shortcut. Schema definitions live in `annotation_schemas`; run provenance lives in `annotation_batches`; labels remain `assertions` rows with `kind="annotation"`; autonomous labels are candidate/non-injected; operator judgments remain separate durable assertions. Informal tags and affinity are nomination evidence only.

No user-tier DDL change or `USER_SCHEMA_VERSION` bump is required. `annotation_schemas` is already the immutable versioned registry and `assertions.kind` is plain `TEXT`. Packaged vocabulary growth is therefore replayed as immutable rows on fresh and already-current `user.db` files. `USER_SCHEMA_VERSION` remains 9.

## Snapshot identity

- Project: `polylogue`
- Snapshot source recorded by the manifest: `/realm/project/polylogue`
- Snapshot generated: `2026-07-17T180950Z`
- Branch: `master`
- Commit: `536a53efac0cbe4a2473ad379e4db49ef3fce74d`
- Commit subject: `fix(repair): harden raw authority convergence (#3046)`
- Upstream state in reconstructed checkout: `master` at `origin/master`, ahead 0 / behind 0
- Manifest dirty flag: `true`
- Packaged branch delta: empty file list, empty log, and zero-byte patch at merge base `536a53ef...`
- Reconstructed source state before this work: clean. The manifest dirty flag therefore appears to describe excluded/local snapshot state rather than a source patch.

The patch is generated directly against the commit above. It was independently applied to a detached clean worktree at the same commit with `git apply --check`, then compiled there.

## Evidence inspected

Repository rules and architecture:

- `CLAUDE.md`
- `.agent/CONVENTIONS.md`
- `docs/data-model.md`
- `polylogue/storage/sqlite/archive_tiers/user.py`
- `polylogue/storage/sqlite/archive_tiers/bootstrap.py`
- `polylogue/storage/sqlite/archive_tiers/user_annotations.py`
- `polylogue/storage/sqlite/archive_tiers/user_write.py`
- `polylogue/storage/sqlite/archive_tiers/user_audit.py`
- `polylogue/annotations/schema.py`
- `polylogue/annotations/batch.py`
- `polylogue/annotations/write.py`
- `polylogue/annotations/importer.py`
- `polylogue/annotations/join.py`
- `polylogue/api/archive.py`
- CLI and MCP annotation bindings
- OpenAPI and CLI-output renderers and their contract tests

Full Beads records:

- `polylogue-dve1`: seed families, tag/ontology authority boundary, multi-view bootstrap evidence, governed accept/rename/split/reject lifecycle, and formal-label gate
- `polylogue-7yk5`: `unresolved_inactive(H)` ownership, named horizon/frame/evaluation receipt, right-censoring, and explicit-declaration precedence
- `polylogue-41ow`: `BEGIN IMMEDIATE` preserve/read/write requirement and explicit conflict semantics for competing operator judgments

Relevant history:

- `bf94704c059dcb05a782e7ac8090d39f8022ab08` — typed annotation foundation
- `246c48d085e2af33ca893c1f5c16075e15eb1e56` — durable schema/batch provenance
- `f4504cb4df3b5687454fc2809686d109e9f23642` — provenance-stamped JSONL batch import
- `4ed0cf2dc1a8d115fd2e59cd3448ba3e84fd6df2` — structural join for typed labels

## Production mechanism

### 1. Packaged immutable seed schemas

`polylogue.annotations.schema` now registers all five active v1 schemas in the existing `ANNOTATION_SCHEMA_REGISTRY`. `BUILTIN_ANNOTATION_SCHEMAS` contains the pre-existing `delegation.discourse@v1` schema plus the five seeds.

`persist_builtin_annotation_schemas()` reuses `persist_annotation_schema()`. Replaying an identical definition is idempotent; reusing a `(schema_id, version)` identity for a different canonical definition fails closed with `AnnotationSchemaError`.

`initialize_archive_tier(..., USER)` seeds built-ins on a fresh database. `initialize_archive_database(..., USER)` also replays missing rows when the on-disk database already has the current `USER_SCHEMA_VERSION`. This is data-only bootstrap, not a migration.

### 2. Seed schema catalog

All seed rows are `status="active"`, `evidence_policy="required"`, and include required `confidence` plus optional `abstain` and `rationale` fields.

| Qualified schema | Construct and grain | Fields | Authority model |
| --- | --- | --- | --- |
| `seed.activity@v1` | Primary activity. Session grain for `session:` targets; structural segment grain for `phase:`, `message:`, and `block:` targets. | `activity` = `debugging`, `design`, `implementation`, `research`, `writing`, `ideation`, `ops`, or `procurement`; `confidence`; optional `abstain`, `rationale`. | Evidence-linked annotation. Agent batch writes remain candidate/non-injected until operator judgment. The target ref is the declared grain, avoiding a second grain field that could disagree with identity. |
| `seed.goal-event@v1` | Prospective actor-declared goal lifecycle event on `message:`, `block:`, `work_event:`, or `observed-event:` evidence. | `event_type` = `opened`, `blocked`, `resumed`, `declared_resolved`, `superseded`, or `explicitly_abandoned`; `goal_ref`; `declared_by_ref`; `declaration_authority` = `actor_declared`; optional `opening_event_ref`, `related_goal_ref`; `confidence`; optional `abstain`, `rationale`. | Declaration-only. It cannot emit inferred abandonment or `unresolved_inactive(H)`. Closure/block/resume/supersession can retain opening/replacement refs. Goal-graph derivations remain owned by `polylogue-7yk5`. |
| `seed.outcome-evidence@v1` | Observed outcome evidence on `session:`, `work_event:`, `observed-event:`, `commit:`, `check-run:`, `github-pr:`, or `delegation:` targets. | `outcome_type` = `test_passed`, `commit_observed`, `deployment_observed`, `user_accepted`, `answer_declared`, or `unknown`; `authority` = `structural`, `rule`, or `judged`; `authority_ref`; `temporal_mode` = `observed` or `historical_backfill`; `confidence`; optional `abstain`, `rationale`. | Structural, rule-derived, and judged evidence remain distinct. Historical reconstruction is explicitly marked and cannot share schema identity with prospective goal events. |
| `seed.knowledge-artifact@v1` | Evidence-linked knowledge artifact on `session:`, `message:`, `block:`, `work_event:`, or `assertion:` targets. | `artifact_type` = `decision`, `lesson`, `preference`, `fact_candidate`, `fact_established`, or `commitment`; `statement`; `authority` = `agent_candidate`, `actor_declared`, `structural`, `rule`, or `operator_judged`; `authority_ref`; `confidence`; optional `abstain`, `rationale`. | Candidate versus established fact status is explicit in the construct, while named authority remains a separate field. Assertions still pass through the ordinary candidate/judgment lifecycle. |
| `seed.reusability@v1` | Purpose-specific reuse judgment on `session:`, `phase:`, `message:`, `block:`, `work_event:`, or `assertion:` targets. | `purpose` = `snippet`, `recipe`, or `demo`; `worthy` boolean; `authority` = `agent_candidate` or `operator_judged`; `authority_ref`; `confidence`; optional `abstain`, `rationale`. | Each reuse purpose is judged independently. Agent classification is not operator acceptance. |

Affect/stance is not added to the default seed set.

### 3. Archive-local ontology nomination

`OntologyCandidateNomination` captures one draft `AnnotationSchema` and preserves the source axes separately:

- source informal tag refs
- affinity score
- nomination confidence
- classifier ref and full classifier definition
- version crosswalk
- frame and archive epoch refs
- independent content, action-pattern, temporal-cost, and outcome view proposals
- each view's confidence and evidence refs
- cross-view agreement/disagreement/insufficient state
- unclassified residue refs
- rare-category sample refs
- general evidence refs
- privacy policy and excluded refs

`nominate_ontology_candidate()` requires an `agent:` author, writes `AssertionKind.ONTOLOGY_CANDIDATE`, forces `status=candidate`, uses private visibility, and sets `context_policy_json` to `{"inject": false, "promotion_required": true}`. It never inserts an `annotation_schemas` row and never writes a formal annotation membership assertion.

Candidate identity is content-addressed over the normalized nomination and evidence. Nested JSON provenance is detached and canonicalized; non-finite JSON, non-string object keys, and NFC-normalized key collisions fail closed. An exact retry returns the existing lifecycle row, including accepted/rejected terminal state, rather than resurrecting a candidate.

### 4. Operator governance state machine

`OntologyCandidateGovernance` requires a `user:` actor and one of four decisions:

```text
NOMINATED (candidate, inject:false)
  ├─ accept  ──> candidate ACCEPTED; exact draft definition registered active
  ├─ rename  ──> candidate SUPERSEDED; one operator-authored active output schema registered
  ├─ split   ──> candidate SUPERSEDED; two or more operator-authored active output schemas registered
  └─ reject  ──> candidate REJECTED; no schema registered
```

For `accept`, the active definition must equal the nominated draft except for `status="active"`. `rename` must actually change the definition. `split` requires at least two unique active schema identities. `reject` cannot carry output schemas.

`govern_ontology_candidate()` executes the complete read/preserve/judge/register/receipt sequence under one `BEGIN IMMEDIATE` transaction and requires an idle connection. It reuses `judge_assertion_candidate()` for the durable lifecycle and explicit idempotent/conflict behavior. If schema persistence fails, the candidate transition, generic judgment, and governance receipt roll back together.

Every terminal decision writes a deterministic, private, non-injected `AssertionKind.ONTOLOGY_GOVERNANCE` receipt containing the decision, candidate and generic judgment refs, resulting lifecycle ref when present, active schema identities/fingerprints, source tag refs, affinity, confidence, classifier ref, version crosswalk, frame, epoch, cross-view state, privacy-policy ref, and whether a subsequent annotation batch is required.

Accept/rename/split activates a construct definition; it does not manufacture archive membership. Formal ontology queries remain empty until an annotation batch writes labels and an operator separately judges those agent-authored label candidates.

### 5. Annotator-facing API for ann-02 and ann-03

The landed interface is the existing product route, extended to resolve governed archive-local schemas:

```python
from polylogue.annotations.importer import AnnotationBatchImportRequest

request = AnnotationBatchImportRequest(
    jsonl=jsonl_text,
    batch_id="stable-run-id",
    schema_id="seed.activity",          # or an operator-promoted archive-local id
    schema_version=1,
    target_ref="session:...",
    source_result_ref="result-set:...",
    actor_ref="agent:ann-02-labeler",
    model_ref="agent:model-version",
    prompt_ref="block:prompt-session:0",
    metadata={"calibration": "ann-02"},
    created_at_ms=optional_nonnegative_epoch_ms,
)
result = await poly.import_annotation_batch(request)
```

The lower-level operation is:

```python
await polylogue.annotations.importer.import_annotation_batch(poly, request)
```

The archive facade method is `PolylogueArchiveMixin.import_annotation_batch(request, *, registry=None)`. Existing CLI and MCP bindings call the same operation.

Each non-empty JSONL line has this shape:

```json
{
  "row_key": "stable-row-key",
  "value": {"schema_field": "value"},
  "evidence_refs": ["session:..."],
  "body_text": "optional rationale text",
  "confidence": 0.9
}
```

The batch route:

1. initializes/replays user-tier built-in schema rows;
2. resolves the schema from the caller/global registry and/or immutable archive-local `annotation_schemas` row;
3. rejects registry/durable fingerprint disagreement;
4. uses a one-schema local registry for custom archive vocabulary, avoiding process-global pollution;
5. validates target grain, value fields, evidence policy, live target/evidence refs, duplicate row keys, and bounded payload sizes;
6. enters `BEGIN IMMEDIATE` on the canonical write connection;
7. persists immutable schema and batch provenance;
8. writes each admitted row through `upsert_annotation_assertion()` with `author_kind="agent"`;
9. commits the batch and label assertions atomically.

Agent rows therefore remain candidate/non-injected. An exact batch replay after an operator judgment sees and preserves the terminal label lifecycle instead of downgrading it. This route satisfies the `polylogue-41ow` transaction shape for the newly landed production path; the repository-wide repair of unrelated direct `upsert_assertion()` callers remains owned by that separate open bead.

Ann-02 and ann-03 may assume:

- all five packaged schema ids/versions are durable after opening `user.db`;
- no migration or schema-version bump is needed;
- the request/result models above are the shared Python/CLI/MCP contract;
- archive-local active schemas can be imported without supplying a custom process-global registry;
- invalid rows are recorded in immutable batch validation provenance and valid rows are imported atomically;
- agent labels are candidate/non-injected;
- operator judgment is a separate required step for active formal queries;
- exact batch replay cannot silently overwrite a terminal operator judgment on this route.

## Generated contract surfaces

Adding `ONTOLOGY_CANDIDATE` and `ONTOLOGY_GOVERNANCE` to `AssertionKind` required the full typed registration path:

- Python enum in `polylogue/core/enums.py`
- candidate-review queue admission in `user_write.py`
- exhaustive user-overlay audit registration in `user_audit.py`
- OpenAPI assertion-kind enum
- CLI output JSON Schema assertion-kind enum

The exact renderer modules executed were:

```text
python -m devtools.render_openapi
python -m devtools.render_cli_output_schemas
```

Because this container lacks unrelated optional/runtime imports, the renderer modules were invoked through an import-only shim that bypassed eager package initializers while executing each renderer's real `main()` function. Both subsequent `--check` runs reported synchronized artifacts:

- `docs/openapi/search.yaml`
- `docs/schemas/cli-output/query-unit-envelope.schema.json`

The CLI status snapshot changed the fresh `annotation_schemas` count from 1 to 6.

## Changed files

- `polylogue/annotations/schema.py` — five seed declarations and built-in schema catalog
- `polylogue/annotations/write.py` — candidate nomination, deterministic ids, governed state machine, atomic receipts
- `polylogue/annotations/importer.py` — canonical connections and durable archive-local schema resolution
- `polylogue/annotations/__init__.py` — public exports
- `polylogue/core/enums.py` — two typed assertion kinds
- `polylogue/storage/sqlite/archive_tiers/bootstrap.py` — fresh and same-version data-only schema replay
- `polylogue/storage/sqlite/archive_tiers/user_annotations.py` — built-in schema persistence helper
- `polylogue/storage/sqlite/archive_tiers/user_write.py` — ontology candidate review admission
- `polylogue/storage/sqlite/archive_tiers/user_audit.py` — exhaustive diagnostic surfaces
- `docs/data-model.md` — seed catalog, governance, and batch interface
- `docs/openapi/search.yaml` — regenerated enum
- `docs/schemas/cli-output/query-unit-envelope.schema.json` — regenerated enum
- `tests/unit/annotations/test_seed_ontology.py` — real-route acceptance coverage
- `tests/unit/cli/__snapshots__/test_plain_cli_snapshots.ambr` — six built-in schema rows

No production module was added under `polylogue/`, so topology projection regeneration is not required.

## Acceptance matrix

| Requirement | Implementation evidence | Test evidence | Status |
| --- | --- | --- | --- |
| Versioned seed schemas | Five active `AnnotationSchema` constants; immutable built-in row replay | Exact enum/authority catalog, fresh registration, drift rejection, additive v2 coexistence, same-version replay | Satisfied |
| No user-tier bump | `USER_SCHEMA_VERSION` untouched; row replay uses existing table | Test asserts on-disk version remains current before and after replay | Satisfied |
| Prospective goals distinct from outcomes | Separate schema ids and disjoint event/outcome enums; authority/temporal provenance | Invalid `unresolved_inactive`; no generic abandonment; identity differs for same target/author/row key | Satisfied |
| No inferred abandonment | Goal schema admits only `explicitly_abandoned` under `actor_declared` | Validation rejects `unresolved_inactive`; enum assertions pin allowed set | Satisfied |
| High-affinity tag at most candidate | Nomination writes candidate assertion only, with affinity separate from confidence | Rejection fixture starts at affinity 0.97; no schema or annotation row appears | Satisfied |
| Governed accept/rename/split/reject | Atomic generic judgment + active schema rows + typed receipt | Rejection, accept, parameterized rename/split, and conflict rollback tests | Satisfied |
| Rejection preserves source/evidence | Candidate retains tags, views, disagreement, residue, rare refs, epoch, privacy, crosswalk | Test re-reads active source tag and terminal rejection/receipt | Satisfied |
| Formal query requires judged batch label | Schema activation does not write annotation membership | Active join empty before batch; candidate-only after agent batch; active only after operator accept | Satisfied |
| Operator judgment preserved on replay | Importer owns `BEGIN IMMEDIATE`; existing terminal assertion is returned | Exact agent batch replay after accept cannot restore candidate state | Satisfied for landed routes |
| Archive-local schema usable by annotators | Importer reads durable schema and builds local registry | Accepted custom schema imports without explicit registry | Satisfied |
| Generated surfaces | Enum propagated through both renderers and audit registration | Renderer sync, four OpenAPI tests, eighteen published CLI-schema checks, audit exhaustiveness | Satisfied |

## Apply order

1. Confirm the target checkout is exactly `536a53efac0cbe4a2473ad379e4db49ef3fce74d` or rebase the patch deliberately.
2. Apply `PATCH.diff` with `git apply --binary PATCH.diff`.
3. Run `git diff --check`.
4. Run the native render checks:
   - `python -m devtools.render_openapi --check`
   - `python -m devtools.render_cli_output_schemas --check`
5. Run `devtools test tests/unit/annotations/test_seed_ontology.py` in the repository's fully provisioned environment.
6. Run the focused schema/storage/generated-contract tests listed in `TESTS.md`.
7. Run the normal broad pre-merge gate in the repository-managed environment.
8. No user-tier migration or backup-manifest operation is needed because no durable DDL version changes.

## Verification performed

Detailed commands and results are in `TESTS.md`. In summary:

- apply-ready binary patch checked and applied against a detached clean base worktree: passed
- compileall on changed Python/test modules in both working and apply-check trees: passed
- `git diff --check`: passed
- seed/governance acceptance file: 7 passed
- annotation schema tests: 69 passed
- assertion substrate tests: 32 passed
- user-overlay audit tests: 2 passed
- OpenAPI renderer contract tests: 4 passed
- published CLI-output schema checks: 18 passed, 14 deselected
- unaffected annotation-writer subset: 16 passed, 2 environment-gated tests deselected
- selected user-tier DDL/bootstrap checks: 7 passed, 1 pre-existing/environment skip, 21 deselected
- both generated renderer `--check` operations: synchronized

## Risks, limitations, and continuation value

The normal repository test entry points could not start in this bare container because the snapshot environment lacks `hypothesis`, `ijson`, `dateparser`, `aiosqlite`, and the `sqlite-vec` extension. Focused execution used import-only shims for unavailable eager imports; production functions, SQLite DDL, transactions, schema persistence, batch importer, joins, and judgments were not mocked. The two full annotation round-trip tests that bootstrap the embeddings tier remain unexecuted here because `sqlite-vec` is unavailable.

No live daemon, browser, operator archive, secrets, NixOS deployment, or concurrent multi-process production archive was available. The tests prove the immediate-transaction shape and terminal replay behavior in real SQLite connections, but the separate repository-wide `polylogue-41ow` forced-interleaving repair remains outside this bead.

Automatic same-version bootstrap now performs idempotent writes/checks for six built-in rows whenever a current `user.db` is initialized. This is intentional migration-free vocabulary replay; immutable-definition drift fails startup rather than silently replacing authority.

The low-level `persist_annotation_schema()` function remains available to trusted production code. The governed API is the authority-safe route for archive-specific promotion; this patch does not attempt to revoke lower-level storage access.

The generic candidate lifecycle creates a non-injected promoted/superseding audit assertion in addition to the dedicated governance receipt. That row is not a formal `kind="annotation"` membership fact. Consumers should use active `annotation_schemas`, `annotation_batches`, annotation assertions, and the governance receipt for ontology semantics.

A further iteration is most likely a small repair/certification pass: run the native managed suite with all dependencies, exercise the CLI/MCP command against a disposable real archive, and adjust any type/lint formatting found by unavailable tooling. A substantial second implementation pass is unlikely to add much unless live integration reveals a missing operator UX or the owner chooses to fold the repository-wide `polylogue-41ow` concurrency repair into this scope.
