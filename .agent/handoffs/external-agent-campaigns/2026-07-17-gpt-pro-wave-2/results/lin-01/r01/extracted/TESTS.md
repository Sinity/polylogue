# Verification and test design for polylogue-866e

## Scope and environment

All successful behavior tests exercise the repository’s production SQLite writer, canonical index DDL, direct SQL state, real `aiosqlite`, and independent composed readers. No mock lineage engine, alternate schema, or read-time repair shim was used.

Executed environment:

```text
Python           3.13.5
SQLite           3.46.1
pytest           9.0.2
pytest-asyncio   1.4.0
pytest-timeout   2.4.0
Hypothesis       6.156.1
aiosqlite        0.22.1
Ruff             0.15.20
mypy             1.20.0 (attempted; no passing claim)
uv               0.10.0
```

The managed `devtools test` wrapper refused this container’s 64 MiB `/dev/shm`. Successful focused runs therefore disabled automatic plugin loading, cleared repository-wide addopts, and loaded only the plugins required by each selected lane. This avoids accidental dependence on unrelated globally installed plugins while still exercising production code.

The final `PATCH.diff` was also applied to a fresh detached worktree at the exact base commit. Principal and adjacent suites were rerun there.

## Three-oracle contract

Every convergence fixture observes three independent surfaces:

1. **Physical-row oracle** — direct SQL over `messages`, ordered by `(position, variant_index, message_id)`, proves the canonical rows physically owned by parent and child.
2. **Link-row oracle** — direct SQL over `session_links` and `lineage_prefix_witnesses`, proves resolved parent, branch point, inheritance, typed status, witness length, and digest.
3. **Composed-read oracle** — `read_archive_session_envelope` and the asynchronous production message reader prove logical text and `LineageCompleteness` independently of the state-machine model.

Projection regressions add direct SQL over `action_pairs` and `delegation_facts`. Rebuild cleanup adds `PRAGMA foreign_key_check`. A property-model adjustment is accepted only when the direct SQL and production read oracles establish the same fact.

## Baseline reproduction

The supplied snapshot contains no `.hypothesis/examples` directory. There were therefore no saved Hypothesis blob IDs to enumerate or literally replay. `POLYLOGUE_HYPOTHESIS_REUSE_FAILURES=1` was still set for final property runs, but no unavailable corpus is claimed as executed.

Before applying the patch, a deterministic production-route harness ran the same direct and nested histories now represented by named committed fixtures.

### Direct semantic parent replacement

Transitions: `old-parent`, `child`, `final-parent`; all six permutations. Old and final parents had equal semantic content and sibling variants but wholly different native message identities.

Clean master produced two fixed points:

```text
Healthy fixed point — 4 orders
  child -> final-parent -> old-parent
  final-parent -> child -> old-parent
  final-parent -> old-parent -> child
  old-parent -> final-parent -> child

  parent rows: q0, q1, q1-alt
  child rows:  c2 only
  link cut:    codex-session:parent:q1-alt
  composed:    root / primary / sibling / child tail
  complete:    true

Stale fixed point — 2 orders
  child -> old-parent -> final-parent
  old-parent -> child -> final-parent

  parent rows: q0, q1, q1-alt
  child rows:  c2 only
  link cut:    codex-session:parent:p1-alt
  composed:    child tail only
  complete:    false, dangling_branch_point
```

Baseline trace SHA-256: `3455e5c74c63b38cf60182e68859f2c3f79113be167acb24ebc069b6076054d6`.

After the patch, all six orders produce final cut `q1-alt`, witness length 3, child tail `c2` only, and complete text `root / primary / sibling / child tail`.

### Nested root → parent → child arrival order

Transitions: `root`, `parent`, `child`; all six permutations. Parent and child replayed the root prefix, so delayed parent normalization changed the composed identity visible to the grandchild.

Clean master produced two fixed points:

```text
Dangling fixed point — 2 orders
  child -> parent -> root
  parent -> child -> root

  parent rows: p2 only
  child rows:  c3 only
  parent cut:  codex-session:root:r1
  child cut:   codex-session:parent:p1  (deleted replay row)
  composed:    child tail only
  complete:    false, dangling_branch_point

Healthy fixed point — 4 orders
  child -> root -> parent
  parent -> root -> child
  root -> child -> parent
  root -> parent -> child

  parent rows: p2 only
  child rows:  c3 only
  parent cut:  codex-session:root:r1
  child cut:   codex-session:root:r1
  composed:    root prompt / root reply / child tail
  complete:    true
```

After the patch, all six orders produce the root-owned cut and one complete fixed point.

Patched trace SHA-256: `f8510a65861d12f74c0db8a3485d3bb2c20739eb39e84bd502bd036bb1b8c731`. The primary tree and clean-applied tree produced this trace byte-for-byte.

## Deterministic regression design

### `test_semantic_prefix_witness_converges_all_sibling_replacement_orders`

Production dependencies:

- full-replace prefix extraction;
- canonical sibling-variant signature order;
- compact witness persistence;
- parent-replacement descendant reconciliation;
- production composed envelope reader.

Oracles:

- final parent physical rows;
- child physical tail;
- link identity/inheritance/status;
- witness length/hash shape;
- composed transcript/completeness.

Representative killed mutations:

- disable semantic witness matching;
- stop reconciling descendants after parent replacement;
- retain the old branch-point ID after semantic identity churn.

The direct mutation run that disabled witness rebinding left `p1-alt` with `dangling_branch_point` instead of final `q1-alt` and failed this test.

### `test_nested_prefix_witness_converges_all_root_parent_child_orders`

Production dependencies:

- delayed child-before-parent normalization;
- composed parent signatures;
- parent-first subtree traversal;
- composition-cache invalidation;
- independent composed reader.

Representative killed mutation: do not enqueue a reconciled child as the next parent. The child then retains deleted `parent:p1` instead of root `r1`, and the fixture fails.

### `test_merge_append_preserves_prefix_lineage_edge_and_witness`

Production dependencies:

- `_write_session_link` merge-append preservation branch;
- existing edge/witness ownership;
- composed read after appended child-owned material.

Representative killed mutation: remove `preserve_existing_lineage=merge_append`. The append rewrites the established prefix-sharing relation to empty lineage values, and the edge assertion fails.

### `test_child_before_parent_reextract_refreshes_action_pair_ranks`

Production dependencies:

- delayed prefix deletion in `_reextract_prefix_tail_db`;
- `refresh_action_pairs` after child physical rows change.

Representative killed mutation: omit the refresh call. The surviving use row remains rank 2 instead of rebasing to rank 1, and the direct SQL oracle fails.

### `test_ancestor_rebind_refreshes_descendant_delegation_fact`

Production dependencies:

- semantic ancestor rebinding;
- impacted descendant set returned by subtree reconciliation;
- `refresh_delegation_facts_for_session` for affected descendants.

Representative killed mutation: remove the descendant delegation refresh loop. The materialized fact retains deleted `root:r1` instead of current `root:q1`, and the SQL oracle fails.

### `test_parent_shortening_degrades_then_semantic_reingest_restores_cut`

Production dependencies:

- witness retention when the parent is too short;
- typed `dangling_branch_point` persistence;
- bounded composed reader behavior;
- later semantic rebinding to new IDs.

Representative killed mutation: clear the witness during degradation. The expected witness row becomes absent, preventing exact later recovery, and the test fails immediately.

The fixture also kills shorter-prefix substitution and untyped degradation by requiring unchanged child physical rows, tail-only incomplete output while shortened, and exact complete restoration after reingest.

### `test_full_replace_graph_failure_rolls_back_then_retries_idempotently`

This existing fault-injection fixture was strengthened with the witness oracle. It faults immediately after the real `_resolve_session_graph` returns, when parent rows and descendant edge/witness state have already changed inside the transaction.

It compares physical rows, edge, witness, composed text, completeness, and truncation reason before and after the exception, then proves retry and repeated retry converge. It would fail if message replacement, edge repair, witness persistence, or projection refresh committed separately.

### `test_full_replace_without_parent_clears_witness_when_foreign_keys_suspended`

Production dependencies:

- `_clear_session_projection_rows`;
- caller-owned bulk transaction;
- `PRAGMA foreign_keys=OFF` behavior;
- explicit witness cleanup before edge deletion.

Representative killed mutation: remove `DELETE FROM lineage_prefix_witnesses`. Converting a normalized child into a standalone session then leaves an orphan witness because SQLite does not execute the edge cascade while foreign keys are disabled. The test observes the row directly and fails; it also checks the final complete standalone read and `foreign_key_check`.

### `test_lineage_cut_evidence_is_non_fk_but_edge_owned`

Production dependency: canonical index DDL.

The test protects the load-bearing absence of a message FK on `branch_point_message_id`, while requiring a strict edge-owned witness table, positive prefix length, and fixed 64-character digest constraint.

### `test_v39_lineage_witness_requires_authoritative_semantic_reparse`

Production dependencies: `INDEX_SCHEMA_VERSION`, lifecycle delta registry, and fast-forward eligibility.

The test fails if v39 is classified as clone-safe SQL or if the declaration chain omits v38/v39. Normalized tails cannot reconstruct inherited semantics, so replay is mandatory.

### Typed downstream status tests

`test_session_link_upsert_persists_dangling_branch_point_status` exercises the asynchronous `upsert_session_links` production route. `test_session_digest_preserves_dangling_branch_point_link_status` exercises the downstream insight transform. Removing the enum mapping or closed allowlist entry fails the respective test.

### Cycle test

`test_session_link_resolver_quarantines_cycle` now asserts the link row, unchanged physical rows, and both production composed reads. It proves a quarantined cycle remains typed, bounded, and non-fabricating.

## Stateful and permutation properties

### Added state-machine transitions

`replace_parent_with_equivalent_new_message_identities` selects a root parent with a prefix-sharing child, rewrites every parent message with fresh identities and equal semantic material, and verifies each child through direct SQL and the production reader. Exact identity cannot rescue the cut, so the rule forces witness-based rebinding.

`shorten_parent_then_semantically_reingest` removes the branch-point-bearing suffix, requires typed tail-only incomplete composition, then reingests equivalent parent content under new IDs and requires complete restoration.

The direct branch-point deletion rule was corrected after a generated nested sequence exposed a model defect: deleting an earlier ancestor row can move a surviving immediate-parent branch point earlier in the composed transcript without changing its ID. The model now rebases only by locating that exact ID in a complete production parent envelope. Missing cuts keep their previous logical bound, preserving the strict incomplete-tail oracle.

The state machine continues to assert:

- content-hash and stale-replacement laws;
- physical message identity uniqueness;
- FTS equality with indexable blocks;
- pending/resolved link shape;
- exact transcript equality for complete lineages;
- owned-tail equality for incomplete lineages;
- no modeled prefix beyond a complete parent;
- the closed typed-status vocabulary;
- quarantined-cycle bounded behavior.

### `test_composed_reads_converge_over_permuted_equivalent_histories`

This property is intentionally independent of the state-machine model. Hypothesis exhausts all six permutations of old parent, child, and final semantically equivalent parent. Every example directly requires:

- child physical tail `child-2` only;
- final branch point `final-1-alt`;
- prefix-sharing inheritance and `NULL` status;
- witness length 3 and 64-character digest;
- complete logical transcript `root`, `primary`, `sibling`, `tail`.

## Executed mutation experiments

Each experiment used a separate worktree, removed one load-bearing behavior, and ran the named targeted test through the same production route. All seven produced the intended assertion failure rather than a syntax/import/harness failure.

| Removed behavior | Targeted test | Observed failure |
| --- | --- | --- |
| Merge-append edge preservation | `test_merge_append_preserves_prefix_lineage_edge_and_witness` | Edge tuple lost prefix-sharing cut/inheritance. |
| Deferred `action_pairs` refresh | `test_child_before_parent_reextract_refreshes_action_pair_ranks` | Remaining use rank was `2`, expected `1`. |
| Descendant delegation refresh | `test_ancestor_rebind_refreshes_descendant_delegation_fact` | Fact retained deleted ancestor message identity. |
| Descendant enqueue in subtree walk | `test_nested_prefix_witness_converges_all_root_parent_child_orders` | Child retained immediate-parent cut instead of canonical root cut. |
| Semantic witness rebinding | `test_semantic_prefix_witness_converges_all_sibling_replacement_orders` | Old cut remained typed dangling instead of rebinding to final identity. |
| Witness retention during dangling degradation | `test_parent_shortening_degrades_then_semantic_reingest_restores_cut` | Witness row became `NULL`, expected retained length/hash. |
| Explicit cleanup with foreign keys off | `test_full_replace_without_parent_clears_witness_when_foreign_keys_suspended` | Orphan witness row remained after edge removal. |

Targeted command shape:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -o addopts='' -q \
  tests/unit/storage/test_lineage_normalization.py::<target-test>
```

## Successful commands and results

### Final focused storage, DDL, and lifecycle suite

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -o addopts='' -q \
  tests/unit/storage/test_lineage_normalization.py \
  tests/unit/storage/test_archive_tiers_ddl.py \
  tests/unit/storage/test_index_fast_forward_lifecycle.py
```

Primary tree: `75 passed, 3 warnings in 22.77s`.

Clean-base applied worktree: `75 passed, 3 warnings in 21.50s`.

The warnings are expected unknown-config notices because automatic loading of the repository-wide asyncio/timeout plugins was intentionally disabled for this lane.

### Final topology/status projection suite

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -o addopts='' -p pytest_asyncio.plugin -q \
  tests/unit/insights/test_topology_cycle_rejection.py \
  tests/unit/insights/test_transforms.py
```

Primary tree: `44 passed, 2 warnings in 2.71s`.

Clean-base applied worktree: `44 passed, 2 warnings in 2.95s`.

### Property campaign, Bead-derived seed 866039

```bash
POLYLOGUE_HYPOTHESIS_REUSE_FAILURES=1 \
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -o addopts='' \
  -p hypothesis.extra.pytestplugin -q \
  --hypothesis-seed=866039 \
  tests/property/test_write_path_state_machine.py
```

Primary tree: `4 passed, 3 warnings in 20.20s`.

Clean-base applied worktree: `4 passed, 3 warnings in 21.32s`.

Separate statistics runs on seed 866039 reported:

```text
permutation property: 6 passing, 0 failing, 0 invalid; exhausted domain
state machine:        100 passing, 0 failing, 4 invalid
stateful steps:       18 per accepted example
```

### Independent property seed 866040

```bash
POLYLOGUE_HYPOTHESIS_REUSE_FAILURES=1 \
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -o addopts='' \
  -p hypothesis.extra.pytestplugin -q \
  --hypothesis-seed=866040 \
  tests/property/test_write_path_state_machine.py
```

Primary tree: `4 passed, 3 warnings in 20.14s`.

### Broad adjacent storage compatibility

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -o addopts='' \
  -p pytest_asyncio.plugin -p xdist.plugin -q \
  tests/unit/storage/test_archive_tiers_write.py \
  tests/unit/storage/test_delegations_view.py \
  tests/unit/storage/test_message_query_reads.py \
  tests/unit/storage/test_session_topology.py
```

Primary tree: `86 passed, 2 warnings in 26.65s`.

Clean-base applied worktree: `86 passed, 2 warnings in 22.21s`.

### Adjacent topology-edge compatibility

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -o addopts='' \
  -p pytest_asyncio.plugin -p xdist.plugin -q \
  tests/unit/storage/test_topology_edges.py
```

Primary tree: `14 passed, 2 warnings in 4.29s`.

Clean-base applied worktree: `14 passed, 2 warnings in 4.54s`.

### Content-hash, identity, and merge-append laws

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -o addopts='' -q \
  tests/unit/pipeline/test_pipeline_ids.py::test_session_hash_is_deterministic \
  tests/unit/pipeline/test_pipeline_ids.py::test_message_id_change_changes_session_hash \
  tests/unit/storage/test_archive_tiers_write.py::test_message_content_hash_tracks_same_identity_body_edits \
  tests/unit/storage/test_archive_tiers_write.py::test_merge_append_increments_session_counts_without_full_refresh
```

Primary tree: `4 passed, 3 warnings in 1.59s`.

These protect deterministic import identity, message-identity sensitivity, body-edit hashing, and merge-append count behavior. The patch does not modify `pipeline/ids.py` or `core/hashing.py`.

### Static checks

```bash
ruff check <12 changed Python files>
ruff format --check <12 changed Python files>
python -m py_compile <12 changed Python files>
git diff --check
```

Result:

```text
Ruff check:       All checks passed
Ruff format:      12 files already formatted
py_compile:       12 files passed
git diff --check: passed
```

The same Ruff and diff checks passed in the clean-applied worktree.

### Schema lifecycle policy

```bash
PYTHONPATH=. python -m devtools lab policy schema-versioning --json
```

Result on both finalized primary and clean-applied trees:

```json
{
  "upgrade_helpers": [],
  "invalid_migration_resources": [],
  "index_delta_declarations": {
    "compatibility_floor": 32,
    "declared_versions": [33, 34, 35, 36, 37, 38, 39],
    "missing_versions": [],
    "duplicate_versions": [],
    "invalid_versions": [],
    "ok": true
  },
  "ok": true
}
```

Generated checks that passed:

```bash
python -m devtools render docs-surface --check
python -m devtools render topology-status --check
```

`devtools render topology-projection --check` is not claimed: it rewrites unrelated baseline LOC counts even on untouched clean master while returning success. The generated target was reverted and omitted from the patch.

### Clean-base patch validation

```bash
git worktree add --detach /mnt/data/polylogue_applycheck_final \
  536a53efac0cbe4a2473ad379e4db49ef3fce74d
git -C /mnt/data/polylogue_applycheck_final apply --check PATCH.diff
git -C /mnt/data/polylogue_applycheck_final apply PATCH.diff
git -C /mnt/data/polylogue_applycheck_final diff --check
git -C /mnt/data/polylogue_applycheck_final diff --full-index --binary \
  > /tmp/PATCH.reproduced.diff
cmp /tmp/PATCH.reproduced.diff PATCH.diff
```

Result:

```text
apply check:          passed
apply:                passed
diff check:           passed
regenerated diff:     byte-for-byte identical
```

The applied worktree then passed the `75`, `44`, `4`, `86`, and `14` item lanes listed above. Its convergence trace was byte-identical to the primary trace.

### Synthetic fan-out probe

A non-gating local probe created one parent, N normalized children, and a semantically equivalent parent replacement with new IDs, then read every child:

```text
N=10:  replacement 0.023719s, 10 healthy links, 10 witnesses, 0 incomplete reads
N=100: replacement 0.049125s, 100 healthy links, 100 witnesses, 0 incomplete reads
```

This demonstrates functional fan-out behavior on a small local SQLite database. It is not representative production-scale benchmarking.

## Commands attempted but not completed

### Managed focused test wrapper

```bash
PYTHONPATH=. python -m devtools test \
  tests/unit/storage/test_lineage_normalization.py::test_semantic_prefix_witness_converges_all_sibling_replacement_orders \
  -q
```

Result: exit 125 before pytest. The wrapper reported only 64 MiB free in `/dev/shm` and refused disk-backed pytest containment.

### Managed quick verification

```bash
PYTHONPATH=. python -m devtools verify --quick --json
```

Observed progress:

```text
ruff format: passed
ruff check:  passed
mypy:        exceeded execution bound
```

No successful `devtools verify` claim is made.

### Direct production-file mypy

A direct mypy run over the changed production files exceeded the execution bound after reporting diagnostics in unrelated imported files `polylogue/cli/shell_words.py` and `polylogue/cli/machine_main.py`. This does not establish patch-local mypy success or failure.

### Frozen managed dependency synchronization

```bash
uv sync --frozen --extra dev
```

The locked plan began but DNS retries failed while fetching `prompt-toolkit==3.0.52`, pulled through `questionary`. Successful suites used the available exact versions listed at the top of this document.

### Full generated-surface check

`python -m devtools render all --check` exceeded the execution bound without leaving tree changes. Targeted `docs-surface` and `topology-status` checks passed; full render coverage remains unverified.

## Remaining operator verification

Run these in the complete managed environment with the original private Hypothesis database, when available:

```bash
POLYLOGUE_HYPOTHESIS_REUSE_FAILURES=1 \
  devtools test tests/property/test_write_path_state_machine.py

devtools test -k session_links
devtools test -k composed
devtools verify
```

Then validate representative live archives, especially deep lineage, very high fan-out, inherited event references, incomplete-lineage telemetry, and daemon process interruption around transaction boundaries.
