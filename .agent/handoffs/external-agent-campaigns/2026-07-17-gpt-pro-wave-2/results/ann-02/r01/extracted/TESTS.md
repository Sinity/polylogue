# Judge calibration test design and execution

## Test philosophy

The focused suite exercises production modules and real SQLite USER-tier routes. It does not replace the calibration implementation with a fake repository. A small import harness was used only because this container lacks unrelated optional dependencies imported by the repository's global package/test bootstrap. The final harness deliberately lets the real `polylogue.annotations.__init__` execute, which is how the continuation pass found the original eager-export import cycle.

Every focused test below names the production dependency it protects and a representative implementation mutation/removal that should make it fail.

## Focused tests

### 1. `test_mask_checklists_are_independent_and_blind_view_has_exact_surface`

Production dependencies:

- `CalibrationSourceItem.__post_init__`
- `CALIBRATION_MASKED_FIELDS`
- `compose_calibration_batch`
- `polylogue.insights.judgment.blinding.blind_items`
- `assert_no_leak`
- private `annotation_assignment` persistence

Proof:

- freezes the existing generic blinding mask and the independently copied calibration baseline mask;
- inserts a real prior annotation containing `PRIOR_LABEL_SENTINEL` and proves it never reaches the judge projection;
- rejects nested exact, case/separator, and camelCase aliases (`prior_labels`, `Prior-Labels`, `goldLabel`);
- asserts the exact top-level judge DTO keys and exact per-item keys;
- proves actor, model, execution context, gold marker/label/ref, and prior-label sentinel are absent;
- inspects the private assignment to verify hidden gold receipts and deterministic composition metadata.

Anti-vacuity mutation:

Removing the no-prior-label source boundary, shrinking either mask, returning the assignment payload directly, omitting recursive key checks, or making the assignment public/injectable fails the test.

### 2. `test_gold_identity_and_composition_inputs_are_canonical`

Production dependencies:

- `author_gold_annotations`
- `_gold_records`
- `_gold_assertion_id`
- deterministic source canonicalization/hash ranking
- `_item_id`
- `_read_assignment`

Proof:

- rejects a new gold key for an already-gold target;
- rejects an existing gold key moved to another target;
- confirms rollback leaves exactly the original three gold rows;
- composes the same ordinary source set in opposite caller order and obtains the same assignment and judge view;
- tampers with a stored assignment item ID and proves recording fails closed.

Anti-vacuity mutation:

Removing key/target uniqueness, preserving caller sequence as source identity, using nondeterministic shuffle, or trusting stored item IDs without recomputation fails the test.

### 3. `test_krippendorff_alpha_matches_hand_computed_nominal_fixture`

Production dependency:

- `krippendorff_alpha`

Proof:

For units `A/A`, `A/B`, `B/B`, observed disagreement is `1/3`, expected disagreement is `3/5`, and alpha is `1 - (1/3)/(3/5) = 4/9`. The test also checks insufficient pairable data returns `None` and a degenerate all-equal interval fixture returns `1.0`.

Anti-vacuity mutation:

Using raw percent agreement, omitting the unit-size coincidence denominator, including single-rater units in the coincidence pool, or changing interval distance away from squared numeric distance fails the fixture.

### 4. `test_measures_are_batch_scoped_queryable_and_context_exact`

Production dependencies:

- `record_calibration_batch`
- `materialize_calibration_measures`
- `_construct_gold_metrics`
- `_reliability`
- `query_judge_trust`
- exact batch/judge/assignment validation

Proof:

- materializes a private batch-scoped measure;
- verifies alpha `4/9`, gold accuracy `1.0`, Brier `0.01`, ECE `0.1`, pairable counts, and label distribution;
- repeats identical materialization with a later wall-clock time and proves assertion identity and original timestamp remain unchanged;
- returns `eligible` for the measured exact context;
- returns explicit `unknown/no_measure` for the same actor/model under an unseen context;
- materializes a later exact-context batch against the same anchors and verifies accuracy/Brier drift reasons;
- re-analyzes the older batch later under a different bucket parameter and proves the newer judged batch remains the trust result;
- policy-mutates a durable measure public and proves the query fails closed.

Anti-vacuity mutation:

Pooling contexts, using analysis time instead of batch time for trust recency, omitting drift checks, accepting mutable public products, or always inserting a fresh measure on identical inputs fails this test.

### 5. `test_measure_peer_sets_are_canonical_and_future_baselines_fail_closed`

Production dependencies:

- peer-set canonicalization in `materialize_calibration_measures`
- deterministic measure input digest/identity
- baseline chronology validation

Proof:

- materializes the same peer set in opposite orders and obtains one assertion identity with stable original timestamp;
- verifies stored peer refs are sorted;
- rejects a later-created batch as the drift baseline for an older current batch.

Anti-vacuity mutation:

Hashing caller peer order directly or omitting the future-baseline check fails the test.

### 6. `test_recording_derives_schema_confidence_and_rejects_dual_value_drift`

Production dependencies:

- `_resolved_calibration_confidence`
- schema validation
- atomic `record_calibration_batch`
- confidence-based measure materialization

Proof:

- submits conflicting top-level and schema `value.confidence` and verifies the whole record operation rolls back before an annotation batch exists;
- records schema confidence without a duplicate top-level value;
- verifies assertion envelopes carry the resolved confidence;
- verifies Brier and confident-gold counts use that confidence.

Anti-vacuity mutation:

Allowing the two confidence channels to diverge, ignoring schema confidence, or committing the batch before row validation fails the test.

### 7. `test_recording_is_immediate_and_cannot_overwrite_operator_judgment`

Production dependencies:

- `_immediate_transaction`
- `record_calibration_batch`
- `persist_annotation_batch`
- `upsert_annotation_assertion`
- `user_write.judge_assertion_candidate`
- canonical SQLite connection profile/busy timeout

Proof:

- captures SQL trace and proves the recording route's first statement is `BEGIN IMMEDIATE`;
- records agent labels as candidates;
- opens a second real SQLite connection;
- has the operator acquire `BEGIN IMMEDIATE`, accept a candidate, and hold the transaction;
- starts an agent replay and proves its own `BEGIN IMMEDIATE` blocks behind the operator;
- releases/commits the operator transaction and proves the byte-identical agent replay preserves `accepted` status/value;
- submits a changed replay and proves immutable-input drift rolls back without modifying the accepted judgment.

Anti-vacuity mutation:

Removing `BEGIN IMMEDIATE`, beginning after the preserve read, bypassing the existing authority chokepoint, or allowing changed replay inputs to overwrite a terminal row fails the test.

## Existing regression tests run

- Existing judgment blinding tests: `18 passed`
- Existing comparative judgment calibration tests: `9 passed`
- Existing annotation schema tests: `69 passed`

These protect the generic blinding receipt/reveal behavior, the pre-existing in-memory comparative calibration functions, and immutable annotation schema validation from regressions caused by the new vocabulary and imports.

## Commands and results

### Focused production-route suite

Command executed through the import-only harness against the implementation tree and again against a fresh applied tree:

```text
python /mnt/data/ann02-continue/run_calibration_tests_normal_annotations.py
python /mnt/data/ann02-continue/run_calibration_tests_apply.py
```

Result each time:

```text
7 passed, 2 warnings
```

The two warnings are Pytest configuration warnings because `pytest-timeout` is not installed in the container; they are not test failures.

### Existing suites

```text
python /mnt/data/ann02-continue/run_blinding_tests.py
# 18 passed, 2 warnings

python /mnt/data/ann02-continue/run_existing_judgment_calibration_tests.py
# 9 passed, 2 warnings

python /mnt/data/ann02-continue/run_selected_tests.py tests/unit/annotations/test_schema.py
# 69 passed, 2 warnings
```

### Repository-native test dispatcher

```text
python -m devtools test tests/unit/annotations/test_calibration.py -q
```

Result: blocked before test collection with:

```text
ModuleNotFoundError: No module named 'ijson'
```

The global dispatcher imports source decoding and scenario modules unrelated to calibration before selecting the focused test.

### Broader annotation durable/write modules

Attempted:

```text
python /mnt/data/ann02-continue/run_selected_tests.py \
  tests/unit/annotations/test_schema.py \
  tests/unit/annotations/test_durable_storage.py \
  tests/unit/annotations/test_write.py
```

Collection was blocked by unavailable environment dependencies/imports:

- `ModuleNotFoundError: No module named 'dateparser'`
- unavailable `ParsedMessage` export in the reduced parser import setup

The schema module was then run independently and passed all 69 tests. No claim is made that the durable/write modules passed.

### Compilation and compatibility

Executed on the implementation tree and fresh applied tree:

```text
python -m compileall -q <changed Python/test files>
```

Result: pass.

Changed Python/test files were also parsed with:

```python
ast.parse(source, feature_version=(3, 10))
```

Result: `python-3.10-grammar: OK (6 files)`. This check caught and drove removal of a Python-version-sensitive multiline f-string during the continuation pass.

### Normal import smoke

The smoke preloaded only heavy storage parent packages, not `polylogue.annotations`, then imported:

```python
import polylogue.storage.sqlite.archive_tiers.user_annotations
import polylogue.annotations.calibration
```

Result: pass in both implementation and fresh applied trees. This is specifically designed to fail if the removed calibration re-export/import cycle returns.

### Generated surfaces and topology

```text
python -m devtools.build_topology_projection
python -m devtools.render_topology_status
python -m devtools.render_topology_status --check
python -m devtools.verify_topology --json
```

Result:

```json
{
  "blocking": false,
  "counts": {
    "orphans": 0,
    "missing": 0,
    "conflicts": 0,
    "kernel_rule": 0,
    "tbd": 9
  }
}
```

Realized/declared module count: `1024/1024`. The nine advisory TBDs are pre-existing storage placement entries, not introduced by calibration.

OpenAPI and CLI schema checks were run with import-only stubs for unavailable optional dependencies:

```text
python /mnt/data/ann02-continue/run_module_with_stubs.py devtools.render_openapi --check
python /mnt/data/ann02-continue/run_module_with_stubs.py devtools.render_cli_output_schemas --check
```

Results:

```text
render openapi: sync OK: docs/openapi/search.yaml
render cli-output-schemas: sync OK: docs/schemas/cli-output
```

Generated JSON/YAML parse checks passed.

### Patch and fresh application

```text
git diff --check
git apply --check PATCH.diff
git apply PATCH.diff
```

Results: pass.

The patch was applied to a fresh clone at the exact snapshot commit. All 12 changed paths were compared byte-for-byte against the implementation tree and matched. Focused tests, compilation, Python 3.10 grammar, normal imports, topology, generated contract checks, and `git diff --check` passed again in the applied tree.

## Checks not performed

- full repository test suite;
- Ruff, Mypy, or Pyright (not installed; network unavailable for fetching them);
- full layering/import-boundary verification;
- Nix/NixOS build or deployment checks;
- live daemon, CLI, MCP, browser, archive, or secret-backed integration;
- actual 13k-session campaign;
- OS-level two-process interleaving (the test uses two real connections and threads);
- mutation-testing runner, although representative mutations are listed above.
