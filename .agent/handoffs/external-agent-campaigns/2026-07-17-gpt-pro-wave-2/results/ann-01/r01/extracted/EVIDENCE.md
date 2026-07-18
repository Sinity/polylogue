# EVIDENCE — Authority, source findings, and decisions

## Snapshot authority

The attached project-state archive recorded:

```text
project: polylogue
source: /realm/project/polylogue
generated_at: 2026-07-17T180950Z
branch: master
commit: 536a53efac0cbe4a2473ad379e4db49ef3fce74d
dirty: true
```

The snapshot's branch-delta artifacts all identify `origin/master` with merge base `536a53ef...`; the changed-file list, commit log, and patch are empty. The all-refs bundle reconstructed `master` at the same commit, with upstream `origin/master` ahead 0 / behind 0, and the reconstructed checkout was clean before implementation. This is the base named by `PATCH.diff`.

The manifest's dirty flag conflicts with its zero-byte branch delta and the reconstructed clean source state. The most defensible reading is that the flag reflects excluded/local snapshot state, not an undisclosed source patch. No input archive or snapshot artifact is copied into the result package.

## Beads authority

### `polylogue-dve1`

The current description and design establish these binding constraints:

- use the existing annotation/batch/judgment substrate;
- ship versioned seed schemas for activity, prospective goal events, observed outcome evidence, knowledge artifacts, and reusability;
- keep goal events prospective and outcome evidence structural/observed;
- never infer timeless abandonment;
- treat `unresolved_inactive(H)` as a separate horizon/evaluation-world derivation;
- preserve structural/rule/judged authority for outcomes;
- let informal tags and affinity nominate candidates only;
- preserve source tag membership, affinity, classifier definition, confidence, frame, and evidence as separate axes;
- retain cross-view disagreement, residue, rare samples, epoch drift scope, version crosswalks, privacy/excision, and rejection history;
- require operator/shared judgment to accept, rename, split, or reject;
- create formal ontology membership only through a versioned schema, annotation batch, and judged annotation assertion;
- stop autonomous work at candidates.

The acceptance criteria specifically require prospective events and outcome evidence to remain distinct, prevent absence-of-closure from creating abandonment, keep formal queries empty after high-affinity tagging alone, preserve a rejected candidate's tag/evidence trail, retain multi-view disagreement and boundary evidence, and make historical backfill authority visible.

### `polylogue-7yk5`

The corrective contract owns goal state and says:

- goal state is `open | explicitly_closed | explicitly_blocked | unresolved_inactive(H)`;
- `H` is an as-of horizon with an evaluation receipt;
- explicit close/block events outrank inactivity inference;
- missing future/capture frame yields censored/unknown, not abandoned;
- historical reconstruction must preserve named proxy/judged authority;
- marker absence is unknown/no-declaration, never a protocol failure.

This is why `seed.goal-event@v1` contains only actor declarations and rejects `unresolved_inactive`, while `seed.outcome-evidence@v1` has a separate `historical_backfill` mode and authority fields.

### `polylogue-41ow`

The reproduced race is a non-atomic read followed by upsert that can downgrade a concurrently accepted assertion. Its design requires the full preserve/read/write decision under `BEGIN IMMEDIATE`, canonical connection profiles, and explicit conflict rather than last-writer-wins for competing operator judgments.

The new governance route owns one `BEGIN IMMEDIATE` transaction from candidate read through generic judgment, immutable schema registration, and governance receipt. The batch importer already had the correct immediate transaction boundary; this patch switches it to the canonical write connection and keeps schema/batch/assertion writes inside that transaction. Exact replay after operator acceptance is tested.

The separate repository-wide repair of every unrelated `upsert_assertion` caller is not claimed here.

## Current source findings

### Durable annotation substrate already exists

`polylogue/storage/sqlite/archive_tiers/user.py` already defines:

- unified `assertions` with plain `TEXT` kind vocabulary;
- immutable `annotation_schemas` keyed by `(schema_id, schema_version)` with canonical JSON and SHA-256;
- immutable `annotation_batches` with a foreign key to schema identity and complete source/actor/model/prompt/count provenance;
- a built-in `delegation.discourse@v1` row at registration time 0.

Therefore adding five schema rows is data-only vocabulary growth. No table, column, index, constraint, or durable version transition is needed.

`polylogue/storage/sqlite/archive_tiers/user.py` currently sets `USER_SCHEMA_VERSION = 9`. `CLAUDE.md` still says user.db version 6, so the source constant is authoritative and the documentation table is stale. The patch does not alter the version.

### Typed declaration boundary already exists

`polylogue/annotations/schema.py` provides:

- `AnnotationField` and `AnnotationSchema` with normalized immutable definitions;
- target ObjectRef-grain declarations;
- evidence policy and abstention rules;
- canonical definition JSON and fingerprint;
- `AnnotationSchemaRegistry` with active-status enforcement and drift rejection;
- packaged registration via `register_annotation_schema`.

The five seeds are added directly to this registry, not to a new framework.

### Storage persistence already enforces immutability

`polylogue/storage/sqlite/archive_tiers/user_annotations.py` provides `persist_annotation_schema`, `read_durable_annotation_schema`, `persist_annotation_batch`, and `read_annotation_batch`. Reusing an identity with identical canonical JSON is idempotent; conflicting JSON raises `AnnotationSchemaError`.

The new `persist_builtin_annotation_schemas` is a loop over this existing authority.

### Label writing route already exists

`polylogue/annotations/importer.py` already defines:

- `AnnotationBatchImportRequest` and `AnnotationBatchImportResult`;
- bounded JSONL row parsing;
- target/evidence resolution;
- schema validation;
- `BEGIN IMMEDIATE` around schema, batch, and assertion writes;
- immutable batch retry semantics;
- per-row outcomes.

`polylogue/api/archive.py` already exposes `PolylogueArchiveMixin.import_annotation_batch`; existing CLI and MCP bindings call the same operation. The missing capability was resolving an operator-promoted archive-local durable schema without registering it globally. The patch adds that resolution and preserves global archive isolation.

### Agent candidate authority already exists

`polylogue/annotations/write.py` delegates labels to the shared `upsert_assertion` chokepoint. Non-user authors are forced to candidate/non-injected status. `judge_assertion_candidate` already supplies accept/reject/defer/supersede transitions, terminal retry detection, promoted assertion provenance, and explicit conflicting-prior-judgment errors.

The ontology governance implementation composes this lifecycle rather than inventing a new judgment table or status vocabulary.

### Query path already distinguishes status and schema

`polylogue/annotations/join.py` reads `kind="annotation"` assertions, filters by schema stamp and lifecycle status, and joins them to structural targets. The end-to-end test uses this path to prove:

- accepted schema alone yields zero active memberships;
- agent batch yields candidate membership only;
- operator judgment yields active membership;
- replay does not restore candidate status.

### AssertionKind propagation has multiple registration points

Adding an enum value affects more than `core/enums.py`:

- candidate-review admission is a manual tuple in `user_write.py`;
- user-overlay diagnostics require a manual name-to-kind entry and an exhaustive every-kind test;
- OpenAPI serializes the enum;
- CLI-output JSON Schema serializes the enum.

Both ontology kinds were added at all relevant points. Renderer and contract tests are synchronized.

## History findings

The implementation follows the production route established in four July 12 commits:

1. `bf94704c059dcb05a782e7ac8090d39f8022ab08` — introduced typed annotation declaration/validation over assertions.
2. `246c48d085e2af33ca893c1f5c16075e15eb1e56` — introduced immutable schema and batch provenance in user.db.
3. `f4504cb4df3b5687454fc2809686d109e9f23642` — introduced bounded JSONL batch import with transaction/provenance stamping.
4. `4ed0cf2dc1a8d115fd2e59cd3448ba3e84fd6df2` — introduced typed label joins to structural targets.

This history is why the patch extends `schema.py`, `user_annotations.py`, `importer.py`, `write.py`, and `join` consumers rather than creating a new ontology package or storage tier.

## Contradictions and resolutions

### Manifest dirty flag versus source delta

- Manifest: `dirty=true`.
- Branch-delta files/log/patch: empty.
- Reconstructed base checkout: clean and identical to `origin/master` at the named commit.

Resolution: patch against the named commit; disclose the discrepancy; do not invent an unseen dirty patch.

### `CLAUDE.md` user.db version versus source

- `CLAUDE.md`: user.db version 6.
- `polylogue/storage/sqlite/archive_tiers/user.py`: `USER_SCHEMA_VERSION = 9`.

Resolution: current source wins. No version change is made.

### Legacy outcome ontology versus corrective Beads design

The legacy notes inside `polylogue-dve1` proposed a single outcome vocabulary containing solved/partial/abandoned/question-opened/question-closed. The later authoritative corrective description/design separates prospective goal declarations from observed outcome evidence and forbids inferred abandonment.

Resolution: implement the current corrective design. Goal events and outcome evidence are separate schema identities with disjoint enums and explicit temporal/authority provenance.

### Stale annotation package wording versus current importer

The package documentation still described JSONL/CLI/MCP import as future work even though `polylogue/annotations/importer.py`, the archive facade, CLI, and MCP bindings exist.

Resolution: update the package wording and use the current importer as the ann-02/ann-03 interface.

## Key implementation decisions

### Migration-free built-in replay

Decision: seed built-ins on fresh initialization and replay missing rows on same-version USER initialization.

Reason: this makes packaged vocabulary available to existing archives without violating durable-tier migration rules. It changes data only, uses immutable insertion semantics, and fails closed on drift.

### Target ref as activity grain

Decision: session versus segment grain is declared by allowed target ObjectRef kinds instead of a duplicate value field.

Reason: target identity is already authoritative; a second `grain` field could disagree with it.

### Explicit-only abandonment

Decision: expose only `explicitly_abandoned` with `declaration_authority="actor_declared"`.

Reason: the goal graph owns derived inactivity and censoring. A schema enum must not make absence look like timeless fact.

### Historical outcomes remain separate

Decision: give outcome evidence `authority`, `authority_ref`, and `temporal_mode` fields, including `historical_backfill`.

Reason: parser structure, named rules, operator judgments, and historical reconstruction are different authorities and cannot overwrite prospective events.

### Affinity and confidence are independent

Decision: persist `affinity` and nomination `confidence` separately.

Reason: high tag affinity is evidence for nomination, not classifier certainty and not ontology membership.

### Multi-view evidence is preserved, not averaged into truth

Decision: store each declared view, label, confidence, and evidence refs independently, plus an explicit cross-view state.

Reason: disagreement is validation/boundary evidence. Collapsing it into one score would erase the bead's acceptance evidence.

### Operator governance composes generic judgments

Decision: accept/reject use the generic lifecycle; rename/split use generic supersession; all decisions add a typed governance receipt and immutable schema persistence in the same immediate transaction.

Reason: this preserves one judgment substrate, existing retry/conflict semantics, and auditable provenance.

### Schema activation is not membership activation

Decision: governance registers active schema definitions but does not write `kind="annotation"` membership rows.

Reason: the tag/ontology boundary requires a later batch and label judgment. This also lets ann-02/ann-03 calibrate/backfill under explicit run provenance.

### Archive-local schemas stay archive-local

Decision: the importer reads immutable durable custom definitions and creates a one-schema local registry for the call.

Reason: registering a custom archive taxonomy in the process-global registry would pollute other archives and make authority depend on process history.

### New durable receipt kinds use plain TEXT vocabulary

Decision: add `ontology_candidate` and `ontology_governance` to the typed `AssertionKind` boundary without changing user.db DDL.

Reason: the database column is intentionally plain `TEXT`; typed surfaces and generated contracts, not a SQL enum, own vocabulary closure.

## Evidence not available

The attached authority did not provide a running operator daemon, live archive, browser surface, secrets, NixOS deployment, or a fully provisioned dependency environment. No claim is made that those were exercised. Native managed test commands were attempted and their dependency failures are recorded in `TESTS.md`.
