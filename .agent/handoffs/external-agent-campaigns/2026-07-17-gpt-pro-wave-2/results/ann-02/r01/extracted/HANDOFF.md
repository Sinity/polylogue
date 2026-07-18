# Judge calibration infrastructure handoff

## Operator summary

This package is a cohesive implementation draft for blinded archive-annotation calibration on Polylogue's existing `user.db` substrate. It adds private operator gold answers, deterministic blind injection receipts, exact judge/context identity, atomic recording through the existing annotation writer, and durable per-construct agreement/calibration/drift products. It does not add a labels table, a judge registry, or a durable-tier migration.

The continuation pass repaired and strengthened the first implementation before packaging it. The material improvements are:

1. Removed an eager package re-export that created a real `user_annotations -> annotations.batch -> annotations.__init__ -> calibration -> user_annotations` import cycle under normal imports.
2. Added a true two-connection forced interleaving test around `BEGIN IMMEDIATE`, proving that a blocked automated replay observes and preserves an operator's committed terminal judgment.
3. Made ordinary-item ordering, peer-batch sets, gold selection, and judge order deterministic without relying on Python runtime shuffle behavior.
4. Enforced one active gold key and one active gold target per `(gold_set_ref, annotation schema version)`, while retaining byte-identical retry behavior.
5. Bound opaque item IDs back to `(batch_ref, target_ref)` and reject forged assignment IDs before recording.
6. Prevented envelope confidence and schema `value.confidence` from disagreeing; calibration metrics use the validated resolved confidence.
7. Made identical measure recomputation idempotent, kept the original measurement timestamp, and prevented later re-analysis of an older judged batch from becoming the latest trust result.
8. Required drift baselines to use the same exact judge identity, schema version, gold set, and sealed gold assertion refs; a clearly newer batch cannot be used as a baseline.
9. Hardened recursive blinding checks against nested, case, separator, and camelCase provenance-key aliases.
10. Removed an unrelated importer edit from the final patch and regenerated only the affected checked-in contract/topology surfaces.

## Mission covered

The implementation covers the requested gold-question protocol, composition-layer blinding, Krippendorff agreement, confidence calibration, batch drift, stable judge identity, queryable trust summaries, and synthetic proof of authority preservation. The implementation extends the existing `assertions`, `annotation_schemas`, and `annotation_batches` contracts and remains computable offline from `user.db`; archive evidence presentations can be loaded from `index.db` by the trusted composition caller.

## Snapshot identity

- Project: `polylogue`
- Branch: `master`
- Commit: `536a53efac0cbe4a2473ad379e4db49ef3fce74d`
- Subject: `fix(repair): harden raw authority convergence (#3046)`
- Commit timestamp: `2026-07-17T18:55:47+02:00`
- Snapshot manifest generated: `2026-07-17T180950Z`
- Manifest state: `dirty: true`
- Branch delta patch/files/log: empty
- Merge base/HEAD used for the patch: the commit above

The supplied working-tree archive was treated as supporting evidence, not a different patch base. Among the tracked paths present in that overlay, no bytes differed from the named commit. Some tracked paths were absent because the snapshot archive intentionally omitted portions of local/agent state; therefore the apply authority is the named Git commit, not an assertion that the tar overlay was a complete checkout.

## Inspected authority

Repository instructions and architecture:

- `AGENTS.md`
- `.agent/CONVENTIONS.md`
- `docs/data-model.md`
- `polylogue/storage/sqlite/archive_tiers/user.py`
- `polylogue/storage/sqlite/migrations/user/006_annotation_schemas_batches.sql`
- `polylogue/annotations/schema.py`
- `polylogue/annotations/batch.py`
- `polylogue/annotations/write.py`
- `polylogue/annotations/importer.py`
- `polylogue/storage/sqlite/archive_tiers/user_annotations.py`
- `polylogue/storage/sqlite/archive_tiers/user_write.py`
- `polylogue/insights/judgment/types.py`
- `polylogue/insights/judgment/blinding.py`
- `polylogue/insights/judgment/calibration.py`
- relevant annotation, assertion-authority, schema, durable-storage, and judgment tests

Tracker authority:

- `polylogue-rxdo.9`
- `polylogue-rxdo.9.6`
- `polylogue-rxdo.9.12`
- `polylogue-h10`
- `polylogue-41ow`
- `polylogue-rxdo.7`
- `polylogue-rxdo.7.2`

Relevant history:

- `bf94704c0 feat(annotations): add enforced typed annotation foundation (#2757)`
- `246c48d08 feat(annotations): persist durable labeling provenance (#2765)`
- `f4504cb4d feat(annotations): import provenance-stamped JSONL batches (#2767)`
- `4ed0cf2dc feat(annotations): join typed labels to structural targets (#2768)`
- `5aa34e6c5 feat(assertions): add reviewed candidate judgment flow (#2791)`
- `866dab24d feat(insights): comparative judgment core for rxdo.9.11-.9.15 (#2889)`
- `9163d0134 feat(query): bound agent-facing archive reads (#3018)`
- `672786a07 feat(identity): declare actor and execution context refs`

## Durable vocabulary

No SQL migration is required. The existing `assertions.kind TEXT` vocabulary is extended by `AssertionKind`; `annotation_schemas` and `annotation_batches` remain the only schema/batch registries.

### `annotation_gold`

- Payload format: `polylogue.annotation-gold/v1`
- Assertion key: operator-defined `gold_key`
- Scope: `gold_set_ref`
- Target: concrete annotation target
- Author: `author_kind=user`, `author_ref=user:*`
- Status: `active` or operator-transitioned `accepted`
- Visibility: `private`
- Context policy: `{ "inject": false, "promotion_required": false }`
- Schema binding: `_schema = <schema_id>@v<version>`
- Payload fields: `format`, `_schema`, `gold_key`, `gold_set_ref`, `label`, `strata`, and optional `rationale`
- Identity: content-addressed over gold set, qualified schema, gold key, and target
- Invariant: at most one active/accepted key and one active/accepted target per gold set and schema version

The label is validated against the existing `AnnotationSchema`, including target kind and evidence policy. Gold content is sealed into an assignment via a digest; post-assignment mutation fails measure materialization.

### `annotation_assignment`

- Payload format: `polylogue.annotation-calibration-assignment/v1`
- Assertion key: `composition`
- Scope/target: `annotation-batch:<batch_id>`
- Author: `insight:annotation-calibration@v1`, `author_kind=system`
- Status: `active`
- Visibility: `private`
- Context policy: non-injectable, no promotion queue
- Identity: deterministic from `batch_ref`

The receipt stores the exact schema version, source result set, prompt/rubric refs, seed, exact judge triple, source item IDs, randomized order hash, complete masked-field set, deterministic selection policy, opaque item IDs, item targets/evidence, presentation digests, hidden gold refs, and hidden gold content digests. It is never returned as the judge DTO.

### `annotation`

Judge labels continue to use the existing assertion kind and `upsert_annotation_assertion` chokepoint. The calibration route does not define another label row. Agent labels remain `candidate` and non-injectable; human labels follow the existing user-author authority behavior.

### `annotation_measure`

- Payload format: `polylogue.annotation-calibration-measure/v1`
- Algorithm: `polylogue.annotation-calibration-algorithm/v1`
- Assertion key: construct field name
- Scope/target: current `annotation-batch:<batch_id>`
- Author: `insight:annotation-calibration@v1`, `author_kind=system`
- Status: `active`
- Visibility: `private`
- Context policy: non-injectable, no promotion queue
- Identity: deterministic from batch, construct, algorithm parameters, immutable current/peer/baseline inputs, and sealed gold refs

Each row contains:

- agreement: Krippendorff alpha, measurement level, judge count, pairable units, pairable ratings, peer batches;
- gold: assigned/scored/correct/abstained/missing counts, accuracy, abstention rate, unscored rate, observed label distribution;
- calibration: Brier score, fixed reliability buckets, expected calibration error, confident-gold count;
- drift: exact baseline ref, accuracy delta, Brier delta, distribution total variation, thresholds, reasons, alert;
- exact judge identity, schema version, input digest, and measurement timestamp.

### Calibration batch metadata

`AnnotationBatch.metadata` gains the optional key `item_target_refs`. It is an ordered, unique array of concrete ObjectRefs and must have length `valid_count`. A heterogeneous batch retains the source `result-set` as its collection `target_ref`/`source_result_ref`; label assertions target the concrete entries. Other calibration metadata includes:

- `format = polylogue.annotation-calibration-batch/v1`
- `assignment_ref`
- `execution_context_ref`
- `rubric_ref`
- `gold_count`
- `item_order_hash`
- ordered `item_ids`

## Production API

The implementation is in `polylogue.annotations.calibration` and intentionally imported directly to avoid a package-initializer/storage cycle.

```python
author_gold_annotations(
    conn,
    *,
    schema,
    registry,
    gold_set_ref,
    items,
    author_ref="user:local",
    now_ms,
) -> tuple[ArchiveAssertionEnvelope, ...]
```

Authors immutable, schema-validated gold assertions inside an owned `BEGIN IMMEDIATE` transaction.

```python
compose_calibration_batch(
    conn,
    *,
    request: CalibrationCompositionRequest,
    item_loader: Callable[[str, tuple[str, ...]], CalibrationSourceItem],
) -> ComposedCalibrationBatch
```

Selects gold deterministically, loads presentation evidence through the trusted caller, applies composition-side blinding, persists the private assignment receipt, and returns a separately shaped `JudgeBatchView`.

```python
record_calibration_batch(
    conn,
    *,
    schema,
    registry,
    batch_id,
    labels: Sequence[CalibrationLabelInput],
    now_ms,
) -> AnnotationBatch
```

Requires a complete response keyed only by opaque item IDs. Target/evidence refs are recovered from the sealed assignment. Batch and label writes are atomic under `BEGIN IMMEDIATE`.

```python
materialize_calibration_measures(
    conn,
    *,
    batch_id,
    peer_batch_ids=(),
    baseline_batch_id=None,
    constructs=None,
    measurement_levels=None,
    bucket_count=10,
    accuracy_drop_alert=0.10,
    brier_increase_alert=0.05,
    distribution_tv_alert=0.15,
    now_ms,
) -> tuple[ArchiveAssertionEnvelope, ...]
```

Computes typed per-construct products under one owned immediate transaction. Peer order is canonical. Drift requires exact anchors and exact judge identity.

```python
query_judge_trust(
    conn,
    *,
    schema_id,
    schema_version,
    construct,
    judge=None,
    minimum_gold=20,
    minimum_confident_gold=20,
    minimum_accuracy=0.80,
    maximum_brier=0.25,
    maximum_expected_calibration_error=0.10,
    maximum_gold_unscored_rate=0.25,
) -> tuple[JudgeTrustSummary, ...]
```

Returns the latest batch-provenance result for each exact judge triple, or an explicit `unknown/no_measure` result when an exact requested context has no measure.

## Exact judge-facing query surface

The judge receives only this shape:

```json
{
  "batch_ref": "annotation-batch:...",
  "qualified_schema_id": "schema.id@v1",
  "rubric_ref": "insight:...",
  "item_order_hash": "...",
  "items": [
    {
      "item_id": "item-...",
      "target_ref": "session:...",
      "evidence_refs": ["session:..."],
      "presentation": {"...": "sanitized archive material"}
    }
  ]
}
```

The composition route reads only `annotation_gold` assertions for gold selection. Ordinary items are explicit source DTOs supplied by a trusted archive-read layer. It does not join prior `annotation` or `judgment` rows. Before the view is returned, top-level blinding and recursive key checks reject actor/model/provider/provenance fields, prior labels/judgments/scores, assignment refs, seed, gold set/ref/label/marker fields, and naming aliases such as `Prior-Labels` or `goldLabel`.

The security boundary is the trusted composer. A judge must receive `JudgeBatchView`, not `ComposedCalibrationBatch`, the private assignment assertion, or unrestricted database access. The implementation prevents structural field leakage; it cannot detect a malicious trusted caller encoding prior-label semantics inside innocently named free text.

## Judge identity

Calibration is partitioned by the exact tuple:

```text
(actor_ref, model_ref, execution_context_ref)
```

- `actor_ref`: `agent:*` or `user:*`
- `model_ref`: versioned model/config ref for agents; for humans, a stable ref selected by the caller
- `execution_context_ref`: required `context-snapshot:*` ref binding prompt/tools/runtime/config

No sibling context inherits calibration. A missing exact tuple returns `state=unknown`, not a pooled score. This follows the actor/execution-context identity direction while making model/version an explicit annotation-batch provenance dimension.

## Trust states

- `eligible`: sufficient scored/confident gold, no drift alert, and all thresholds pass
- `below_threshold`: sufficient data but one or more quality thresholds fail
- `drifting`: sufficient data and the selected measure contains a drift alert
- `unknown`: no exact measure or insufficient scored/confident gold

The default thresholds are implementation defaults, not a governance decision. They should be made operator-configured policy before automated routing depends on them.

## First real archive calibration batch

Recommended first campaign per judge/context:

- 200 total items
- 40 gold items (20%)
  - 24 fixed anchor items retained across campaigns
  - 16 rotating stratified gold items
- 160 ordinary archive items
- at least three distinct exact judge identities over the same target set for agreement

Recommended strata:

- construct class and expected prevalence, with rare classes deliberately oversampled;
- evidence/provider/source family;
- session length or token quartile;
- archive recency cohort;
- ambiguity/difficulty;
- presence of structurally missing or weak evidence;
- terminal-state or claim-verification subtypes relevant to the construct.

The initial run should be descriptive, not an automatic production gate. Require at least two batches for the exact actor/model/context tuple before treating drift as operational. Direct drift comparison in this implementation requires the exact same sealed gold assertion refs. Therefore materialize drift over the 24 fixed anchors, or compose dedicated fixed-anchor batches; the rotating 16 should inform coverage and fresh accuracy but should not be compared as though they were identical anchors.

For each campaign, retain:

- one shared ordinary target frame across judges for alpha;
- exact prompt, rubric, model, and context snapshots;
- class-stratified anchor counts;
- adjudicated gold authorship and rationale;
- a documented trust-policy threshold set external to this algorithm version.

## Decisions made

1. Reuse `assertions` for gold, assignments, labels, and measures; reuse `annotation_batches` for packet provenance.
2. Avoid a durable schema migration because the kind column and immutable schema registry already support vocabulary growth.
3. Treat confidence as the judge's probability that the emitted construct label is correct; Brier and reliability metrics score that event on gold items.
4. Use Krippendorff alpha because it supports missing/incomplete raters and nominal or interval constructs. Units with fewer than two observed ratings are unpairable.
5. Require complete packet recording. Partial/resumable packets remain an open product decision rather than silently changing count semantics.
6. Keep gold and measure assertions private and permanently non-injectable.
7. Use deterministic hash ranking for selection/order instead of runtime-dependent PRNG shuffling.
8. Use batch composition time as trust recency; re-running analysis over an older judged batch does not make the judge look newly calibrated.
9. Require the same exact fixed anchors for drift rather than comparing unmatched rotating gold.
10. Fail closed on malformed or policy-mutated durable receipts/products.

## Changed files

- `docs/data-model.md` — records the durable calibration vocabulary and exact judge partition.
- `docs/openapi/search.yaml` — regenerates the assertion-kind enum.
- `docs/plans/topology-target.yaml` — regenerates the realized module projection, including the new module.
- `docs/schemas/cli-output/query-unit-envelope.schema.json` — regenerates the assertion-kind enum.
- `docs/topology-status.md` — regenerates topology counts.
- `polylogue/annotations/__init__.py` — documents the new module without importing it eagerly.
- `polylogue/annotations/batch.py` — validates and exposes immutable heterogeneous item target refs.
- `polylogue/annotations/calibration.py` — complete gold/composition/recording/measurement/query implementation.
- `polylogue/annotations/write.py` — admits concrete item targets declared by a heterogeneous batch.
- `polylogue/core/enums.py` — adds the three assertion kinds.
- `polylogue/storage/sqlite/archive_tiers/user_annotations.py` — validates concrete item targets at durable batch persistence.
- `tests/unit/annotations/test_calibration.py` — seven real-route synthetic tests, including cross-connection interleaving.

No complete replacement files are included because `PATCH.diff` unambiguously represents all changes, including generated files through Git binary patches where repository attributes suppress textual diffs.

## Apply order

1. Start from commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d`.
2. Run `git apply --check PATCH.diff`.
3. Run `git apply PATCH.diff`.
4. Install the repository's complete locked development environment.
5. Run the focused test and generated-surface commands in `TESTS.md`.
6. Run the full repository verification lane before merge.
7. Wire an authorized CLI/MCP/web composition/recording surface only after its capability boundary is designed; do not expose assignment assertions directly.
8. Pilot the recommended calibration campaign against a copy of the real archive before enabling any automated trust gate.

## Acceptance matrix

| Mission requirement | Implementation/proof | Status |
| --- | --- | --- |
| Gold authoring and storage | `annotation_gold` under immutable schema version, private/non-injectable, key/target uniqueness and content sealing | Implemented and focused-tested |
| Stratified blind injection | deterministic quotas/hash ranking and private `annotation_assignment` receipt | Implemented and focused-tested |
| Per-judge gold accuracy | `annotation_measure.gold` plus exact-context `query_judge_trust` | Implemented and focused-tested |
| Composition-layer blinding | explicit `JudgeBatchView`, no prior-label query, recursive provenance block | Implemented and focused-tested |
| Inter-judge agreement | nominal/interval Krippendorff alpha per construct and batch | Implemented; hand fixture tested |
| Confidence calibration | Brier, reliability buckets, ECE, confidence-authority validation | Implemented and focused-tested |
| Drift across batches | exact-anchor accuracy/Brier/distribution deltas and alert reasons | Implemented and focused-tested |
| Typed/queryable durable rows | existing assertions/batches, no ad-hoc report | Implemented and SQLite-tested |
| Stable judge identity | exact actor/model/context tuple; no context pooling | Implemented and focused-tested |
| No operator overwrite | route-owned `BEGIN IMMEDIATE`, existing authority chokepoint, real two-connection interleaving | Implemented and focused-tested |
| Offline computation | materializer uses durable user-tier rows; evidence projection supplied from archive read layer | Implemented at library layer |
| Live CLI/MCP/web judge flow | no production adapter in this patch | Unverified/not implemented |
| Real 13k-session pilot | no live operator archive access | Unverified/not performed |
| Global `upsert_assertion` TOCTOU repair | calibration paths avoid the race; generic chokepoint remains open under `polylogue-41ow` | Outside this patch |

## Risks and open decisions

### Product/API decisions

- Select the live capability owner for composition, assignment access, recording, reveal-after-verdict, and measure/query operations. `polylogue-rxdo.9.6` indicates that the generic blinding primitive is not yet fully wired to CLI/MCP/web surfaces.
- Define a stable human `model_ref` convention. The code requires an ObjectRef but deliberately does not invent a universal JudgeSpec.
- Define gold lifecycle: draft review, activation, retirement/supersession, multi-operator adjudication, leakage monitoring, and anchor refresh.
- Decide whether partial/resumable packets are required. This implementation requires a complete exact item set and records one immutable batch.
- Externalize threshold governance and algorithm versions before trust states drive automated work allocation.
- Decide tie, incomparable, and insufficient-evidence conventions per construct. Current annotation schemas can encode them, but the calibration module does not invent categories.
- Decide whether multiple measure policies for the same batch/construct need a first-class policy ref. The query currently treats the most recently materialized measure on the latest judged batch as current for that batch.

### Security/blinding limits

- Structural blinding is enforced at composition. A malicious trusted loader could still place semantic hints in an allowed free-text key.
- Target/evidence refs remain visible. Repeated fixed anchors can eventually become recognizable to a judge across campaigns; rotate presentation/order and monitor anchor exposure.
- The patch does not add a capability gate or HTTP/CLI adapter. Exposing `annotation_assignment` rows to a judge would break the protocol.

### Statistical/measurement limits

- Gold accuracy is only as valid as gold authoring and construct clarity.
- Reliability metrics require enough confident gold outcomes; empty/low-count buckets are represented rather than smoothed.
- Drift compares identical anchors. This is deliberate but means rotating-gold campaigns require a fixed-anchor subset or a future matched/cohort-aware drift algorithm.
- Alpha quantifies agreement, not correctness. It can be high when all judges share the same bias.
- The initial threshold defaults are not empirically calibrated to the operator archive.

### Storage/concurrency limits

- The generic shared `upsert_assertion` function still has the open TOCTOU defect described by `polylogue-41ow`. Every calibration preserve/read/write path owns an immediate transaction, so this implementation does not depend on fixing that global route.
- The focused interleaving covers two SQLite connections in one process. It exercises the same database locking semantics required across processes, but an actual two-process repository test remains valuable when the complete environment is available.

## Verification completed

Detailed commands and anti-vacuity explanations are in `TESTS.md`. The final patch was also applied to a fresh clone of the exact snapshot commit; all 12 changed paths matched the implementation tree byte-for-byte and the focused suite passed there.

Verified:

- focused calibration routes: 7 passed;
- existing judgment blinding tests: 18 passed;
- existing comparative calibration tests: 9 passed;
- existing annotation schema tests: 69 passed;
- Python compilation;
- Python 3.10 grammar parsing for changed Python/test files;
- normal import smoke without preloading `polylogue.annotations`;
- generated OpenAPI and CLI schema synchronization;
- topology projection/status synchronization and topology verification;
- generated JSON/YAML parsing;
- `git diff --check`;
- fresh-clone `git apply --check`, apply, byte comparison, focused tests, import/compile/grammar/generated checks.

Unverified or blocked:

- the repository-native `devtools test` dispatcher cannot start in this container because `ijson` is absent;
- broad durable/write test modules cannot collect because optional dependencies such as `dateparser` and parser exports are unavailable in this snapshot environment;
- full repository suite, lint, static typing, layering verification, and Nix checks;
- live daemon, CLI, MCP, browser, secrets, NixOS deployment, or operator archive;
- a real 13k-session calibration campaign;
- an operating-system two-process interleaving test;
- the global generic assertion TOCTOU repair.

## Value of another iteration

A small repair pass has low expected value: the implementation, documentation, packaging, fresh application, and focused correctness proofs are complete.

A substantial second implementation pass has meaningful value only with the full development/runtime environment and archive access. The highest-value work would be: design and implement the authorized CLI/MCP/web judge queue; add reveal-after-verdict and capability tests; run the first real fixed-anchor/rotating-gold campaign; tune thresholds from observed distributions; exercise a true two-process race; run the full verification lane; and fix `polylogue-41ow` globally rather than only avoiding it in calibration routes.
