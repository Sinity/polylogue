# Evidence record for polylogue-866e

## Input custody

| Input | Bytes | SHA-256 | Use |
| --- | ---: | --- | --- |
| `/mnt/data/polylogue-all.tar(124).gz` | 128,314,788 | `ec3fa193b2f99a6daee0bc620af4f69b5728834e99f8196792a17c3fbae11155` | Supplied project-state authority |
| `/mnt/data/lin-01-lineage-law(1).md` | 8,688 | `3c4866aaf915dab57722f683ff229f3723a5c40bacca18f1cef48032f32eb299` | Mission and delivery contract |

Neither supplied input is copied into the result ZIP.

## Snapshot identity

The archive’s generated overview reports:

```text
Generated: 2026-07-17T180950Z
Git: master @ 536a53ef dirty=true
```

The authoritative branch-delta artifact reports:

```text
Base ref: origin/master
Merge base: 536a53efac0cbe4a2473ad379e4db49ef3fce74d
Diff Stat: empty
Commits: empty
```

The all-refs bundle resolves both `HEAD` and `master` to:

```text
536a53efac0cbe4a2473ad379e4db49ef3fce74d
fix(repair): harden raw authority convergence (#3046)
```

Resolution: `PATCH.diff` targets the clean tracked tree at that exact commit. The overview’s global dirty flag reflects omitted ignored/local runtime material, not a tracked branch patch.

## Bead authority and chronology

### Older tracked record

The tracked `.beads/issues.jsonl` row for `polylogue-866e` has:

```text
status: open
updated_at: 2026-07-16T13:04:56Z
last note: PR #2922 merged canonical sibling ordering and direct missing-cut safety;
           remaining broader acceptance criteria stay under this owner
```

### Newer archive export

The archive-side `polylogue-beads-export.jsonl` row has:

```text
status: in_progress
priority: 0
updated_at: 2026-07-17T14:44:09Z
```

It contains later notes absent from the tracked row:

- the implementation-readiness audit says not to redo PR #2922 and names nested dangling-ancestor propagation, persisted semantic-cut witnesses, and crash/rollback fault injection as remaining work;
- a later direct audit at commit `3b217d63` did not reproduce its then-assumed falsifier and required fresh production evidence before speculative normalization;
- the PR #3044 note says Claude arrival/replay witnesses landed but did not close the writer-level order-independent branch-point protocol.

Resolution: the archive export is later and agrees with the mission’s `in_progress` status. Its direct-audit caution is satisfied by fresh clean-master deterministic traces in this pass.

### Acceptance criteria extracted from current authority

The current Bead and mission require:

1. saved Hypothesis reuse on baseline when available, plus named deterministic physical/link/composed fixtures;
2. newest accepted sibling identity/text/order under repeated replacement;
3. convergence across child-first, parent-first, delayed parent, and replacement order;
4. exact branch-point repair or typed bounded degradation, never crash or fabricated prefix;
5. atomic rollback and idempotent retry;
6. no property weakening without direct SQL and composed-read proof;
7. exact property, session-link, composed-read, and verification evidence with seeds, counts, and mutation targets;
8. preservation of content-hash/import identity and rebuild-only index policy.

`HANDOFF.md` and `TESTS.md` map each criterion to implementation and executed evidence.

## Ratified architecture decision

The authoritative decision is:

```text
.agent/handoffs/external-agent-campaigns/2026-07-16-gpt-pro-wave/
  testdiet/context/testsuite_diet/architecture/
  02-lineage-composition-and-snapshots.md
```

Load-bearing requirements found there:

- children store only canonical divergent tails plus typed `session_links` edges;
- reads compose through `branch_point_message_id` in one deferred snapshot and return `LineageCompleteness`;
- arrival order must not alter the final transcript;
- missing or ambiguous authority may not invent a parent prefix;
- later parent arrival or replacement re-normalizes atomically;
- `dangling_branch_point` is a canonical incomplete state;
- `branch_point_message_id` remains a non-FK logical reference;
- exact surviving identity wins, and equivalent-prefix rebinding requires proof;
- never attach by nearest timestamp, ordinal, or similar heuristic;
- rollback retry must be idempotent;
- proof must cover physical tails, edges, composed content, completeness, cycle/depth safety, and the historical seed.

The patch follows the selected normalized representation. It does not duplicate complete parent transcripts into children, mutate during reads, reject child-first ingestion, or attach heuristically.

## Repository policy evidence

`AGENTS.md` classifies `index.db` as rebuildable derived state and states that derived tiers have no migration chain: schema changes edit canonical DDL and require reset/replay (`polylogue ops reset --index && polylogued run`) rather than an upgrade helper. The schema-version policy distinguishes semantic-reparse-required changes and rejects in-place derived migrations.

This supports v39 `SEMANTIC_REPARSE`: existing normalized child tails cannot reconstruct inherited semantic prefixes exactly.

## Prior landed history

### PR #2922

```text
b55f3fd9697083d44466613091604a21c7324ae6
fix(lineage): compose sibling variants canonically (#2922)
2026-07-16T04:17:17+02:00
```

Its commit message says it solved:

- canonical sibling-variant ordering;
- parent-first versus child-first convergence for that slice;
- bounded behavior when a direct branch cut is missing;
- no fabricated prefix on dangling-ancestor reingest.

It explicitly left open:

- persisted semantic cut witnesses;
- crash/rollback fault injection;
- exhaustive saved examples;
- complete nested-ancestor completeness propagation.

The current patch preserves that landed behavior and targets only the remaining transition-law gaps.

### PR #3044

```text
1d3145afa5f07a8f90ffceefc33ba425c877a8ad
feat(archive): admit Test Diet survivor laws (#3044)
2026-07-17T16:42:27+02:00
```

The newer Bead note explicitly states that #3044 did not close `polylogue-866e`. Inspection confirms it did not introduce a durable writer-side semantic branch-point witness or subtree reconciliation protocol.

### Index schema v38 history

V38 introduced writer-populated `action_pairs` and `delegation_facts` projections. Their accepted parser semantics cannot be reconstructed by cloning DDL alone. The patch therefore adds the missing v38 `SEMANTIC_REPARSE` declaration while adding the same classification for v39.

## Production source findings

### Non-FK branch point is deliberate

`polylogue/storage/sqlite/archive_tiers/index.py` documents why `session_links.branch_point_message_id` is not a foreign key: parent full replacement deletes and recreates message rows, and `ON DELETE SET NULL` would fire during the delete phase and destroy the splice before reinsertion.

The new witness table therefore owns proof through the edge’s composite natural key, not through volatile parent message rows.

### Canonical sibling signatures already existed

PR #2922’s current writer signature query orders by message position, `variant_index`, and block position and includes every sibling variant. The witness hashes this existing ordered semantic stream. It does not introduce a competing message-equivalence definition.

### Existing readers already degraded safely

The independent composed readers stop when the recorded branch-point ID is absent from the resolved parent snapshot. They return child-owned material and surface incomplete lineage with reason `dangling_branch_point`. The missing part was durable writer authority and typed persisted edge state, not read-time exception suppression.

### Graph reconciliation was not transitive

Before this patch, graph resolution repaired the arriving session and directly unresolved children but did not walk descendants whose composed parent identity changed indirectly. This explains the nested two-fixed-point trace where a grandchild retained a deleted immediate-parent replay ID.

### Legacy stale-cut mapping lacked semantic proof

`_replacement_for_stale_prefix_branch_point` can map narrow legacy suffix/ordinal shapes. It cannot prove that a wholly new identity set has the exact previously accepted semantic prefix. The new witness path takes precedence; the mapper remains only for witnessless legacy/manual rows.

### Merge append could reinterpret lineage

The full-replacement-derived branch point variables are empty during merge-append. Without an explicit preservation branch, `_write_session_link` can replace an established prefix-sharing edge and witness with empty lineage data. The continuation adds `preserve_existing_lineage=merge_append` and a direct mutation-killing fixture.

### Deferred normalization changed derived ranks

Deleting an inherited child prefix changes the rank of remaining action pairs. `_reextract_prefix_tail_db` refreshed counts but did not previously prove action-pair rank refresh. The new regression shows omission leaves rank 2 where canonical rank is 1.

### Ancestor rebinding changed descendant delegation authority

A grandchild’s materialized delegation fact can name an ancestor message identity reached through its composed parent. Rebinding the lineage edge therefore requires descendant delegation refresh, not only link/projection/thread updates. The continuation fixture proves omission retains a deleted `root:r1` reference instead of current `root:q1`.

### Bulk rebuilds suspend foreign keys

Existing rebuild tests use caller-owned transactions with `PRAGMA foreign_keys=OFF`. In that mode SQLite does not execute the witness table’s edge cascade. The continuation reproduced an orphan witness when a normalized child became standalone and added explicit cleanup in `_clear_session_projection_rows`.

## Clean-master failure evidence

A deterministic harness used the production writer, canonical DDL, direct SQL rows, `session_links`, and `read_archive_session_envelope` on untouched commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d`.

### Direct sibling/identity replacement family

All six histories end with parent rows `q0`, `q1`, `q1-alt` and child row `c2` only, yet clean master divides by whether the child normalized before final parent identities existed.

| Orders | Link cut | Composed result | Completeness |
| --- | --- | --- | --- |
| `child → final-parent → old-parent`; `final-parent → child → old-parent`; `final-parent → old-parent → child`; `old-parent → final-parent → child` | `parent:q1-alt` | `root`, `primary`, `sibling`, `child tail` | complete |
| `child → old-parent → final-parent`; `old-parent → child → final-parent` | stale `parent:p1-alt` | `child tail` | incomplete: `dangling_branch_point` |

This is direct evidence that equivalent final material did not converge.

### Nested root/parent/child family

All six histories end with parent row `p2` only and child row `c3` only. Clean master divides by whether the root arrives after the child has normalized against immediate-parent replay rows.

| Orders | Child cut | Composed result | Completeness |
| --- | --- | --- | --- |
| `child → parent → root`; `parent → child → root` | stale deleted `parent:p1` | `child tail` | incomplete: `dangling_branch_point` |
| remaining four permutations | canonical `root:r1` | `root prompt`, `root reply`, `child tail` | complete |

This is direct evidence that one-level normalization did not propagate to descendants.

### Trace hashes

```text
baseline JSON: 3455e5c74c63b38cf60182e68859f2c3f79113be167acb24ebc069b6076054d6
patched JSON:  f8510a65861d12f74c0db8a3485d3bb2c20739eb39e84bd502bd036bb1b8c731
```

The patched trace contains one fixed point across all six direct orders and one across all six nested orders. The same patched JSON was reproduced byte-for-byte after applying `PATCH.diff` to a fresh detached worktree.

## Implementation evidence

### Compact witness rather than copied prefix rows

The final schema stores one row per normalized edge, not one row per inherited message. Its digest input is the ordered existing semantic-signature sequence plus exact prefix length. This keeps proof size constant with prefix length and prevents copied parent transcript content from becoming a second authority.

### Exact authority precedence

The reconciliation order is:

1. exact current branch-point identity;
2. complete semantic witness matching the exact parent prefix at recorded length;
3. narrow legacy witnessless mapper;
4. typed dangling state.

No nearest-position, partial-prefix, timestamp, or sibling substitution exists in the new path.

### Parent-first subtree traversal

The reconciliation queue starts from the written/resolved session and follows indexed inbound prefix-sharing edges in deterministic natural-key order. A visited set bounds malformed cycles. Each child becomes a later parent in the queue, and composition caches are invalidated when its edge or witness changes.

### Atomic refresh set

The writer’s existing transaction encloses message replacement, link write, delayed normalization, descendant reconciliation, witness updates, session projection refresh, thread refresh, descendant delegation refresh, and final derived refresh guard removal. The strengthened rollback fixture faults after real graph resolution and proves no mixed state commits.

### Merge preservation

Merge-append passes `preserve_existing_lineage=True`. If the natural edge already exists, link and witness authority remain unchanged while new child-owned rows are appended. The dedicated mutation test fails when this guard is removed.

### Explicit suspended-FK cleanup

`_clear_session_projection_rows` deletes `lineage_prefix_witnesses` before `session_links`. This mirrors edge ownership when SQLite cascades are disabled and leaves `foreign_key_check` clean after the caller commits.

### Typed downstream status

`TopologyEdgeStatus`, canonical DDL, async session-link persistence, and insight projection all accept `dangling_branch_point`. Dedicated tests cover both persistence and transformed report output.

### Rebuild-only schema evolution

V39 evidence cannot be reconstructed from normalized v38 child tails. The lifecycle declaration blocks clone-safe fast-forward and requires authoritative replay. Schema policy reports valid complete declarations for versions 33 through 39 and no upgrade helper.

## Test anti-vacuity evidence

Seven explicit mutation executions produced intended assertion failures:

| Mutation | Failure evidence |
| --- | --- |
| Remove merge lineage preservation | Edge tuple lost prefix-sharing cut/inheritance in `test_merge_append_preserves_prefix_lineage_edge_and_witness`. |
| Remove deferred action-pair refresh | Remaining use rank was 2, expected 1. |
| Remove descendant delegation refresh | Materialized fact retained deleted ancestor identity. |
| Stop transitive subtree enqueue | Nested child retained immediate-parent cut instead of root cut. |
| Disable semantic witness matching | Direct history retained old dangling cut instead of final identity. |
| Delete witness during degradation | Expected retained `(prefix_length, hash)` row became absent. |
| Rely only on cascade with FKs off | Orphan witness remained after edge removal. |

Additional structural anti-vacuity:

- add a message FK to `branch_point_message_id` or remove the witness edge FK → DDL test fails;
- classify v39 clone-safe or omit a declaration → lifecycle test fails;
- drop dangling status from async mapping or transform allowlist → downstream tests fail;
- split graph resolution from the transaction → rollback/witness before-after oracle fails;
- weaken state-model prefix handling → direct SQL and production reader assertions remain independent.

No existing test/helper was deleted. The property generator was expanded rather than narrowed.

## Verification evidence summary

Successful final primary-tree evidence:

```text
75 focused storage/DDL/lifecycle items
44 topology/insight status items
4 property-module items under seed 866039
4 property-module items under independent seed 866040
100 passing state-machine examples at seed 866039, 18 steps each, 4 invalid
6 exhausted permutation examples at seed 866039
86 broad adjacent storage items
14 adjacent topology-edge items
4 targeted content-hash/identity/merge items
7 intentional mutation failures
Ruff check and format
12-file py_compile
git diff --check
schema declarations 33-39 complete and valid
docs-surface and topology-status render checks
small 10/100-child functional fan-out probe
```

Successful clean-applied evidence:

```text
git apply --check and apply
byte-identical regenerated full-index diff
75 focused storage/DDL/lifecycle items
44 topology/insight status items
4 property items under seed 866039
86 broad adjacent storage items
14 topology-edge items
schema policy and targeted generated checks
byte-identical patched convergence trace
```

Important negative evidence:

- no `.hypothesis/examples` corpus exists in the supplied snapshot;
- managed `devtools test` refused the 64 MiB `/dev/shm` environment;
- `devtools verify --quick` exceeded the bound in repository-wide mypy after Ruff passed;
- direct mypy did not finish and surfaced unrelated imported-file diagnostics;
- frozen `uv sync --frozen --extra dev` failed DNS fetching locked `prompt-toolkit==3.0.52`;
- full `devtools render all --check` exceeded the bound;
- no live daemon, private archive, NixOS deployment, secrets, or production-scale corpus was available.

## Generated-surface evidence

`devtools render docs-surface --check` and `devtools render topology-status --check` pass on the final and clean-applied trees.

`devtools render topology-projection --check` rewrites unrelated baseline LOC counts and exits successfully even on untouched clean master. The generated target was reverted in both trees and deliberately excluded from this surgical patch. This is pre-existing renderer drift, not hidden package incompleteness.

## Contradictions and resolutions

| Apparent contradiction | Resolution |
| --- | --- |
| Overview says `dirty=true`, but package claims a clean base | Branch-delta evidence has the same merge base and empty commits/diff; dirty state is omitted ignored/local material. |
| Tracked Bead says `open`; mission says `in_progress` | Archive-side Bead export is newer, contains later notes, and reports `in_progress`; it supersedes the tracked row for current intent. |
| The 2026-07-17 direct audit said the remaining falsifier was not reproducible | This pass produced fresh clean-master direct and nested traces with two fixed points each before changing production code. |
| PR #2922 claimed parent/child convergence | Its own commit scoped that slice and explicitly left semantic witnesses, rollback, and nested propagation open. |
| A stale branch-point reference needs repair but must not be an FK | Correct: the logical cut must survive delete/reinsert. Semantic proof and typed degradation restore authority without cascade nulling. |
| Witness table uses a foreign key while the branch point cannot | The witness FK targets the durable edge natural key, not volatile message rows. |
| Edge cascade should clean witnesses, yet explicit delete was added | Rebuild paths disable foreign-key enforcement; SQLite does not execute cascades in that mode. The explicit delete preserves the same ownership invariant. |
| Property model changed after a prefix-bound failure | Direct SQL and the production reader proved a surviving branch-point ID moved earlier after ancestor contraction. The model now derives that position from production state and keeps strict bounds/content assertions. |
| A generated topology file was not updated | Its renderer produces unrelated drift on clean master. Targeted relevant generated checks pass; the unrelated target is excluded to keep the patch surgical. |

## Package evidence

`PATCH.diff` is an apply-ready full-index unified diff against the named clean base:

```text
bytes:      88,808
lines:      2,054
files:      13
insertions: 1,482
deletions:  78
SHA-256:    5c122ad4b077a39bf5542811fd4a07c7e8bf69866bf9eeff73c7f3d5bdaac5dd
```

Changed paths:

```text
docs/internals.md
polylogue/core/enums.py
polylogue/insights/transforms.py
polylogue/storage/sqlite/archive_tiers/index.py
polylogue/storage/sqlite/archive_tiers/write.py
polylogue/storage/sqlite/lifecycle.py
polylogue/storage/sqlite/queries/session_links.py
tests/property/test_write_path_state_machine.py
tests/unit/insights/test_topology_cycle_rejection.py
tests/unit/insights/test_transforms.py
tests/unit/storage/test_archive_tiers_ddl.py
tests/unit/storage/test_index_fast_forward_lifecycle.py
tests/unit/storage/test_lineage_normalization.py
```

It contains no supplied archive, mission document, repository snapshot, binary blob, dependency environment, complete replacement file, or placeholder implementation text.
