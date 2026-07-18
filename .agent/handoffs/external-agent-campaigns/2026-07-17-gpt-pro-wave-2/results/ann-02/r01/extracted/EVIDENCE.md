# Judge calibration evidence and adjudication

## Evidence method

The named snapshot commit and current repository source were treated as primary authority. Repository instructions were read before implementation. Complete relevant Beads records and history were then used to resolve design intent and stale descriptions. Observed facts, source-supported inferences, recommendations, and unresolved uncertainty are separated below.

## Observed snapshot facts

- Snapshot manifest: `polylogue-manifest.json`
- Manifest project: `polylogue`
- Manifest branch: `master`
- Manifest commit: `536a53efac0cbe4a2473ad379e4db49ef3fce74d`
- Manifest generated: `2026-07-17T180950Z`
- Manifest dirty flag: true
- Git subject at the named commit: `fix(repair): harden raw authority convergence (#3046)`
- Commit timestamp: `2026-07-17T18:55:47+02:00`
- Snapshot branch delta patch, file list, and log: empty
- Among tracked files present in the supplied working-tree overlay, no bytes differed from the named commit. The overlay omitted some tracked/local-state paths, so the named commit remains the patch base.

## Source findings

### Existing durable annotation substrate

`polylogue/storage/sqlite/archive_tiers/user.py` and migration `006_annotation_schemas_batches.sql` establish:

- one unified `assertions` table with a text `kind` column;
- immutable/versioned `annotation_schemas` rows;
- immutable `annotation_batches` rows carrying actor/model/prompt/source/count/assertion provenance;
- no need for a new table merely to add assertion vocabulary.

`polylogue/annotations/schema.py` provides the typed schema registry and row validation, including target kinds, field types/ranges/enums, abstention, and evidence policy.

`polylogue/annotations/batch.py` is the typed immutable batch provenance object.

`polylogue/annotations/write.py` is the schema-aware single-row write route. It stamps `_schema` and optional `_batch`, validates batch/schema/target/actor provenance, creates deterministic IDs, and delegates authority behavior to `user_write.upsert_assertion`.

`polylogue/annotations/importer.py` already defines a confidence-authority rule: when a schema declares `confidence`, top-level confidence may repeat but may not disagree with `value.confidence`. Calibration recording now follows that source contract.

### Existing assertion authority model

`polylogue/storage/sqlite/archive_tiers/user_write.py` establishes:

- non-user assertion writers are coerced to candidate/non-injectable state unless a call site explicitly opts out;
- terminal operator judgments should be preserved on automated retries;
- `judge_assertion_candidate` is the explicit operator adjudication route;
- the generic upsert currently reads existing status before the upsert statement.

The calibration recorder continues to use `upsert_annotation_assertion` and therefore does not create a new authority path. Its deterministic annotation IDs include schema, target, author, row/item key, and batch, so independent judges/batches do not collapse into one row.

### Existing blinding and comparative calibration

`polylogue/insights/judgment/blinding.py` already provides:

- a fixed provenance mask;
- caller-supplied order;
- a receipt binding order hash, masked fields, rubric, and seal time;
- reveal only after verdict;
- a top-level leak assertion.

The new annotation composer reuses that primitive but adds an independent frozen baseline mask, gold/prior-label fields, and recursive nested-key enforcement. It constructs a narrower DTO after blinding rather than exposing arbitrary visible mappings.

`polylogue/insights/judgment/calibration.py` already provides comparative-judgment calibration functions. It is an in-memory comparative result layer, not a durable annotation-batch product. The new module does not replace it and does not create a second label store; it persists annotation-specific measures as assertion vocabulary.

## Beads findings

### `polylogue-rxdo.9`

Observed intent:

- analysis rigor is graph/provenance validity, not a generic statistics framework;
- frame-exact claims require named definitions;
- blinding, sparse operator gold, judge calibration, and actor plus execution-context identity are intended mechanisms;
- no universal JudgeSpec or separate receipt table should be invented;
- no automatic injection of findings.

Implementation consequence:

- private non-injectable assertion receipts/products;
- existing substrate reused;
- explicit construct/schema/batch/judge input identity;
- no broad statistics library.

### `polylogue-rxdo.9.6`

Observed intent:

- judge surfaces must mask detector/model/agent/actor identity until verdict;
- blinding is enforced by the surface/view mechanism, not voluntary judge behavior.

Current-source contradiction/adjudication:

- the generic blinding primitive is present and tested, but a complete live CLI/MCP/web judgment path is not established by this snapshot.
- This patch therefore reuses the primitive in a real annotation composition route but does not claim live surface wiring.

### `polylogue-rxdo.9.12`

Observed intent:

- calibration attaches to actor plus execution context;
- context must be content-addressed/stable;
- absent context/gold should produce unknown rather than false certainty;
- do not introduce a universal JudgeSpec.

Implementation consequence:

- exact `(actor_ref, model_ref, execution_context_ref)` partition;
- required `context-snapshot:*` execution context;
- exact-context unknown result;
- no judge table or universal registry.

The explicit model ref is retained because `annotation_batches` already treats model as first-class provenance and the mission asks for agent+model+version identity. It is an additional strict partition, not a replacement for actor/context identity.

### `polylogue-h10`

Observed intent:

- Brier score and reliability buckets;
- calibration partitioned by model/config;
- calibration should inform trust in unverified agent claims.

Implementation consequence:

- confidence is scored as probability the emitted gold label is correct;
- Brier, reliability buckets, and ECE are durable per construct;
- exact model and execution context partitioning.

### `polylogue-41ow`

Observed reproduced defect:

- generic `upsert_assertion` performs a separate status read before its `INSERT ... ON CONFLICT` write;
- an operator judgment between those operations can be reverted by automation;
- the prescribed correction is one `BEGIN IMMEDIATE` preserve/read/write transaction using the canonical connection profile, with a second connection/process forced interleaving.

Implementation consequence:

- every gold, assignment, batch-label, and measure preserve/write route in calibration owns `BEGIN IMMEDIATE`;
- nested/deferred transactions fail closed;
- recording test uses a second connection, an operator-held immediate transaction, a blocked agent replay, terminal preservation, and changed-replay rollback.

Scope adjudication:

- the global generic chokepoint remains open and is not silently claimed fixed;
- calibration avoids the defect locally by owning the required transaction shape.

### `polylogue-rxdo.7` and `polylogue-rxdo.7.2`

Observed source/history state:

- the durable schema/batch foundation and JSONL import/query/adjudication flow are already merged;
- external agent rows must remain candidate/non-injected and go through the existing annotation writer;
- independent batches must remain distinct;
- confidence authority is already tested in the importer.

Implementation consequence:

- calibration labels are ordinary `annotation` assertions;
- no parallel label persistence;
- independent judge/batch IDs;
- schema confidence rule preserved.

## History findings

- `bf94704c0` introduced the typed annotation foundation.
- `246c48d08` established durable schema/batch provenance.
- `f4504cb4d` completed the JSONL annotation import route.
- `4ed0cf2dc` joined typed labels to structural targets.
- `5aa34e6c5` established reviewed candidate judgment behavior.
- `866dab24d` merged the comparative judgment core; older notes treating it as unmerged are stale.
- `9163d0134` established bounded agent-facing archive reads, supporting a trusted composition-side projection rather than unrestricted database access.
- `672786a07` declared actor and execution context refs.

## Contradictions resolved

### Stale plan versus current route names

Older notes describe annotation import/CLI/MCP work as future. Current source and merged history show those foundations exist. The implementation extends current `polylogue.annotations.*` and USER-tier APIs rather than inventing the stale planned surface.

### “Population” language versus frame-exactness

Later `rxdo.9` notes supersede unqualified “archive is the population” language. Measures in this patch are batch-/schema-/gold-set-exact and do not claim population validity.

### Blinding primitive versus live product wiring

The primitive exists; complete live judge surfaces do not. The patch implements a composition library boundary and exact DTO but leaves CLI/MCP/web capability wiring unverified.

### Comparative calibration versus annotation calibration

The existing comparative module is not a durable annotation measure store. The new assertion vocabulary is a typed product over existing annotation rows/batches, not a competing label or judge model.

### Dirty snapshot flag versus patch base

The manifest reports dirty, but branch delta artifacts are empty and included tracked overlay files match the named commit. The honest apply base is the named commit; no uncommitted source delta was copied into the result.

## Source-supported inferences

1. `assertions.kind TEXT` plus `AssertionKind`/generated enums is the intended extension point for gold/assignment/measure vocabulary; a migration would add risk without substrate value.
2. A heterogeneous calibration packet needs concrete item targets in immutable batch provenance because the existing batch target is singular. `metadata.item_target_refs` is a minimal additive contract that preserves collection scope and existing tables.
3. Composition must return a dedicated DTO because merely documenting that callers should ignore hidden fields does not enforce blinding.
4. Confidence calibration should use schema-validated confidence when declared, because allowing a side-channel value would make durable labels and measures disagree.
5. Drift should use identical fixed anchors. Comparing different rotating gold sets would confound judge change with item change.
6. Trust recency should follow judgment/batch provenance, not analysis execution time. Otherwise rerunning an old batch could falsely appear to recalibrate a judge today.

## Recommendations

- Pilot 200 items per exact judge context with 20% gold: 24 fixed anchors and 16 rotating stratified gold.
- Run at least three exact judge identities over the same ordinary target frame for alpha.
- Keep the first run descriptive and inspect class/stratum coverage before setting gates.
- Add a governed policy ref/registry if multiple trust or drift policies will coexist operationally.
- Add live capability-gated adapters that expose only `JudgeBatchView` and opaque-ID recording.
- Add reveal-after-verdict without revealing gold before the entire packet is sealed/recorded.
- Fix `polylogue-41ow` globally and add an actual two-process test in the complete development environment.

## Unresolved uncertainty

- Real archive class prevalence, difficult strata, and appropriate gold ratio have not been measured.
- Human `model_ref` convention is not settled by current source.
- Gold adjudication/retirement and leakage-monitoring workflows are not defined.
- Trust thresholds have not been fitted to operator outcomes.
- Multiple measure policies for the same batch/construct currently rely on latest materialization within that batch.
- Full integration behavior cannot be established without the missing development dependencies and live product surfaces.
