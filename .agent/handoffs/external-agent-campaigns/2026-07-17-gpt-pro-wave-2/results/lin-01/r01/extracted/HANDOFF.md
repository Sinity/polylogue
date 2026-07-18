# polylogue-866e handoff: order-independent lineage writes and branch-point safety

## Operator report

This package implements the remaining writer-side convergence law for `polylogue-866e` against the supplied Polylogue snapshot at clean `master` commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d` (`fix(repair): harden raw authority convergence (#3046)`). PR #2922 had already unified sibling-variant ordering and bounded direct missing-cut reads. The remaining defect was durable authority: after a normalized child removed its inherited prefix, the edge retained only a volatile parent message ID. A semantically equivalent parent replacement with new message IDs could therefore leave one arrival order complete and another dangling despite identical final logical content.

The patch adds one compact semantic-prefix witness per normalized prefix-sharing edge, reconciles the complete affected descendant subtree parent-first inside the accepted write transaction, and persists `dangling_branch_point` when neither a surviving identity nor a complete witness proves the current cut. It keeps canonical divergent-tail storage, the deliberate non-FK branch-point reference, independent composed reads, content-hash semantics, and rebuild-only index evolution.

This continuation added four material hardening changes beyond the first iteration:

1. merge-append now explicitly preserves an established lineage cut and witness;
2. deferred child normalization refreshes `action_pairs` after inherited rows are deleted;
3. ancestor identity rebinding refreshes affected descendant `delegation_facts` in the same transaction;
4. `_clear_session_projection_rows` explicitly deletes edge-owned witnesses because bulk rebuilds intentionally suspend SQLite foreign-key enforcement, which otherwise bypasses `ON DELETE CASCADE` and can strand an orphan witness.

Each of those paths has a named production-route regression, and the four critical omissions were exercised as mutations. The final patch is:

```text
base commit: 536a53efac0cbe4a2473ad379e4db49ef3fce74d
files:       13
insertions:  1,482
deletions:   78
patch lines: 2,054
patch bytes: 88,808
patch SHA-256: 5c122ad4b077a39bf5542811fd4a07c7e8bf69866bf9eeff73c7f3d5bdaac5dd
```

No complete replacement files are needed; `FILES/` is intentionally omitted.

## Snapshot identity and authority

The supplied project-state archive is `/mnt/data/polylogue-all.tar(124).gz`, 128,314,788 bytes, SHA-256 `ec3fa193b2f99a6daee0bc620af4f69b5728834e99f8196792a17c3fbae11155`. Its generated overview identifies `master` at `536a53ef` and marks the archive globally dirty. The authoritative branch-delta artifact names merge base `536a53efac0cbe4a2473ad379e4db49ef3fce74d` and contains no commits or tracked diff. The package therefore targets the clean tracked tree at that exact commit; the archive-level dirty flag belongs to omitted ignored or local runtime state.

The mission document is `/mnt/data/lin-01-lineage-law(1).md`, 8,688 bytes, SHA-256 `3c4866aaf915dab57722f683ff229f3723a5c40bacca18f1cef48032f32eb299`.

Two Beads surfaces exist. The tracked `.beads/issues.jsonl` row is older, reports `open`, and stops at the 2026-07-16 PR #2922 note. The archive-side `polylogue-beads-export.jsonl` row is newer, reports `in_progress`, was updated at `2026-07-17T14:44:09Z`, and contains later implementation-readiness, direct-audit, and PR #3044 notes. The newer export and mission agree on current status and supersede the tracked row for execution intent.

Evidence inspected:

- repository guidance in `AGENTS.md` and `CLAUDE.md`, especially derived-index schema policy and test conventions;
- the full newer and older `polylogue-866e` Bead records;
- PR #2922 commit `b55f3fd9697083d44466613091604a21c7324ae6`;
- PR #3044 commit `1d3145afa5f07a8f90ffceefc33ba425c877a8ad`;
- the ratified lineage decision at `.agent/handoffs/external-agent-campaigns/2026-07-16-gpt-pro-wave/testdiet/context/testsuite_diet/architecture/02-lineage-composition-and-snapshots.md`;
- synchronous write authority in `polylogue/storage/sqlite/archive_tiers/write.py`;
- asynchronous topology persistence in `polylogue/storage/sqlite/queries/session_links.py`;
- independent composed-read behavior in `polylogue/storage/sqlite/queries/message_query_reads.py` and `read_archive_session_envelope`;
- canonical DDL and lifecycle declarations in `polylogue/storage/sqlite/archive_tiers/index.py` and `polylogue/storage/sqlite/lifecycle.py`;
- downstream typed projections in `polylogue/core/enums.py` and `polylogue/insights/transforms.py`;
- the modified property/unit tests and adjacent write, delegation, composed-read, topology, freshness, and idempotency suites.

## What PR #2922 already covered

PR #2922 (`fix(lineage): compose sibling variants canonically`) made writer signatures and composed reads share canonical sibling-variant order, included every variant during prefix alignment, and bounded composition when a direct branch-point ID was absent. It added real-route coverage for parent-first versus child-first normalization, parent replacement with stable IDs, missing sibling cuts, dangling-ancestor reingest, and deep lineage.

Its own acceptance section explicitly left open:

- persisted semantic cut witnesses;
- crash/rollback fault injection;
- exhaustive saved examples;
- complete nested-ancestor completeness propagation.

This patch preserves #2922’s ordering and read behavior and implements those remaining writer/link transition requirements. It does not add a second lineage representation or mutate lineage during reads.

## Implemented equivalence law

For accepted histories with the same final logical material, arrival and replacement order must not affect the final lineage state. This includes child-first ingestion, parent-first ingestion, delayed parent resolution, semantic parent replacement with wholly new message identities, sibling-variant races, and retry after rollback.

The fixed point consists of:

1. the same canonical parent physical rows;
2. the same child-owned divergent physical tail;
3. the same resolved parent, canonical branch-point identity, inheritance, and typed status;
4. the same compact semantic witness;
5. the same composed transcript and `LineageCompleteness` result;
6. refreshed projections, threads, and affected delegation facts consistent with the rebound graph.

An existing branch-point message ID is the strongest authority. If that identity survives, the cut remains bound and its witness refreshes to the newest accepted parent semantics. If the ID disappears, rebinding is allowed only when the complete persisted semantic-prefix witness equals the candidate parent prefix at exactly the recorded length. Otherwise the edge retains its logical cut and proof, becomes `dangling_branch_point`, and reads expose only provable child-owned material with explicit incompleteness. The writer never substitutes a nearest ordinal, timestamp neighbor, shorter prefix, or sibling.

Every normal replacement, delayed normalization, descendant reconciliation, edge mutation, witness mutation, projection refresh, thread refresh, and affected delegation refresh remains inside the existing SQLite transaction. An exception rolls the transition back as one unit; retrying the same accepted envelope converges idempotently.

## Mechanism

### Compact semantic witness

Index schema v39 adds `lineage_prefix_witnesses`, keyed by the same natural edge identity as `session_links`:

- `src_session_id`;
- `dst_origin`;
- `dst_native_id`;
- `link_type`.

Each row stores only `prefix_length` and `semantic_prefix_hash`, a 64-character SHA-256 digest over the ordered per-message semantic signatures. It stores no parent transcript text, message blocks, or message foreign key. Proof size is fixed per edge rather than proportional to inherited prefix length.

The witness row is owned by the edge through a composite foreign key with `ON DELETE CASCADE`. `session_links.branch_point_message_id` deliberately remains non-FK: parent full replacement deletes and reinserts message rows inside one transaction, and `ON DELETE SET NULL` would destroy an otherwise stable splice during the delete phase.

### Parent-known child writes

During an accepted full child replacement, `_extract_prefix_tail` compares the complete parsed child stream with the parent’s composed semantic signatures. It returns the divergent child tail, inherited message-reference mapping, and the exact accepted parent prefix. `_write_session_link` persists the edge and reduces that prefix to the compact witness.

Merge-append is not a full lineage reinterpretation. When the natural parent edge already exists, `preserve_existing_lineage=True` leaves the established branch point, inheritance, status, and witness untouched while appending new owned rows. Content hashing and import identity generation are unchanged.

### Child-before-parent resolution

When a parent arrives after a child, `_reextract_prefix_tail_db` aligns the already stored child against the parent’s composed transcript, remaps inherited event references where evidence exists, deletes inherited child rows and their dependent projections, updates the edge, and writes the witness. After physical normalization it refreshes session counts and `action_pairs`, preventing stale ranks left by deleted inherited actions.

### Parent replacement and descendant reconciliation

After accepted graph resolution, `_reconcile_prefix_sharing_subtree` walks the complete reachable prefix-sharing descendant subtree in deterministic breadth-first, parent-first order. For each edge:

- exact branch-point identity survives: keep it, clear dangling state, and refresh the witness;
- identity changed and the complete witness matches the exact current prefix: rebind to the message at `prefix_length - 1`, clear dangling state, and refresh the witness;
- neither proof succeeds: retain branch point and witness and set `dangling_branch_point`;
- quarantined edges remain untouched.

The parent-first walk is load-bearing. Normalizing an immediate parent can delete replay rows named by a grandchild even when the root write never directly touches that grandchild. One-level repair left that descendant stale; the subtree walk closes the transitive gap and invalidates composition caches as each changed child becomes a later parent in the queue.

The graph resolver then refreshes all impacted session projections, old and new thread roots, and descendant `delegation_facts` within the same transaction. The continuation regression proves a rebound grandchild delegation no longer names the deleted ancestor message identity.

### Typed degradation and recovery

The DDL and `TopologyEdgeStatus` vocabulary now admit `dangling_branch_point`. The async session-link writer and insight transform preserve the token rather than reducing it to `NULL`.

The independent readers already implement the required bounded behavior: if the recorded cut cannot be found in the resolved parent snapshot, they return the child-owned tail, set completeness false, and report `dangling_branch_point`. The patch fixes writer authority; it does not mask exceptions or fabricate content in the read path.

A later semantically equivalent parent can restore the edge because the witness survives degradation. Rebinding is exact and atomic.

### Suspended-FK rebuild cleanup

Bulk index rebuild paths can set `PRAGMA foreign_keys=OFF` while replacing parsed-session projections. In that mode SQLite does not execute the witness edge’s cascade. `_clear_session_projection_rows` therefore explicitly deletes `lineage_prefix_witnesses` for the child before deleting `session_links`.

The regression changes a normalized child into a standalone session inside a caller-owned transaction with foreign keys suspended, then proves there is no edge or witness row, the accepted standalone transcript reads completely, and `PRAGMA foreign_key_check` is empty after commit. Removing the explicit delete leaves an orphan witness and fails the test.

### Schema lifecycle

`INDEX_SCHEMA_VERSION` becomes 39. Version 39 is declared `SEMANTIC_REPARSE`: normalized child tails intentionally no longer contain the inherited semantics needed to reconstruct an exact witness, so a v38 index cannot be upgraded correctly through DDL-only mutation. The supported route is authoritative replay into canonical v39 DDL.

The patch also fills the previously missing v38 lifecycle declaration. V38 introduced writer-populated `action_pairs` and `delegation_facts`; those parser-dependent projections likewise require semantic replay rather than clone-safe SQL fast-forward.

## Failure classes and repair mechanisms

### Semantic identity replacement with sibling variants

On clean master, all six histories ended with the same final parent physical rows and child tail, but split into two logical fixed points. Four orders normalized against final `q*` identities and composed completely. `child → old-parent → final-parent` and `old-parent → child → final-parent` retained `p1-alt`, returned only `child tail`, and reported incomplete dangling lineage.

Root cause: the edge retained only a volatile message ID. The final parent was semantically equivalent but used an entirely new identity set, and the legacy mapper could not prove the cut.

Fix: persist the accepted semantic prefix and use it to rebind exactly to `q1-alt`. The deterministic six-order fixture and independent Hypothesis permutation property prove identical parent rows, child rows, link row, witness, transcript, and completeness for every order.

### Nested root → parent → child propagation

On clean master, six root/parent/child permutations also split into two fixed points. `child → parent → root` and `parent → child → root` left the child naming deleted immediate-parent replay row `parent:p1`, returned only `child tail`, and reported dangling lineage. The other four orders bound through root-owned `r1` and composed completely.

Root cause: delayed normalization repaired the immediate child of the arriving root but did not propagate the resulting composed identity change to that child’s descendants.

Fix: reconcile the complete prefix-sharing subtree parent-first. Every permutation now stores only `parent:p2` and `child:c3`, binds both cuts through `root:r1`, and composes `root prompt`, `root reply`, `child tail` completely.

### Parent branch-point deletion or shortening

A parent replacement can remove the referenced branch point while leaving a normalized child tail. Continuing to index the old logical prefix can exceed the surviving parent; heuristic repair can fabricate history.

Fix: retain the child physical tail and semantic witness, mark the edge `dangling_branch_point`, and let reads return bounded child material with explicit incompleteness. A later semantically equivalent reingest with new IDs proves and restores the cut. The state-machine model rebases only when direct SQL and the production composed reader show a surviving branch-point identity.

### Projection consistency after physical normalization

Deleting inherited rows changes the rank and source-message authority of derived projections. The continuation added two specific protections:

- child-before-parent normalization refreshes `action_pairs`; without it, the remaining use row keeps rank 2 instead of rebasing to rank 1;
- ancestor rebinding refreshes descendant `delegation_facts`; without it, the materialized fact retains deleted `root:r1` rather than current `root:q1`.

Both updates remain inside the graph-resolution transaction.

### Transaction rollback

The rollback fixture injects a fault immediately after the real `_resolve_session_graph` has replaced parent rows and reconciled descendants. It compares parent/child physical rows, edge row, witness row, composed transcript, and completeness before and after the exception. All state rolls back at the SQL-oracle level. Removing the fault and retrying converges, and a second retry is idempotent.

### Suspended foreign-key orphan

The continuation audit found that edge ownership through `ON DELETE CASCADE` was insufficient while rebuild paths suspended foreign keys. Without explicit cleanup, a child becoming standalone left an orphan witness after its edge was deleted.

Fix: explicit witness deletion in `_clear_session_projection_rows`, covered under a caller-owned `BEGIN IMMEDIATE` with `foreign_keys=OFF`, post-commit row assertions, complete composed read, and `foreign_key_check`.

## Composition and degradation semantics

| Parent/cut state | Edge state after write | Child physical rows | Composed read | Completeness |
| --- | --- | --- | --- | --- |
| Exact branch-point ID survives | same ID, status `NULL`, witness refreshed | divergent tail only | parent prefix through exact ID + tail | complete |
| ID changed, full witness matches exact current prefix | rebound to current ID, status `NULL`, witness refreshed | unchanged tail | proved current prefix + tail | complete |
| Parent exists, ID absent, witness mismatches or prefix is too short | old logical ID retained, `dangling_branch_point`, witness retained | unchanged tail | child-owned tail only | incomplete, reason `dangling_branch_point` |
| Semantically equivalent parent later returns | exact witness rebinding, status `NULL` | unchanged tail | restored proved prefix + tail | complete |
| No shared prefix at normalization | `spawned-fresh`, no branch point, no witness | complete child transcript | child transcript only | complete |
| Merge-append on an established child | existing cut/status/witness preserved | previous tail + appended rows | existing proved prefix + owned rows | unchanged completeness |
| Cycle is quarantined | quarantined edge unchanged | unchanged | bounded owned material under existing cycle rules | no fabricated prefix |
| Reader depth guard reached | writer state unchanged | unchanged | bounded snapshot | incomplete with depth reason |

## Oracle design

The regressions intentionally observe three independent layers:

1. **Physical-row oracle** — direct SQL over `messages`, ordered by `(position, variant_index, message_id)`, proves what each session physically owns.
2. **Link-row oracle** — direct SQL over `session_links` and `lineage_prefix_witnesses` proves resolved parent, branch point, inheritance, typed status, prefix length, and digest shape.
3. **Composed-read oracle** — the production envelope and async message readers prove logical transcript and `LineageCompleteness` independently of the state-machine model.

Projection-specific tests add direct SQL over `action_pairs`, `delegation_facts`, and `PRAGMA foreign_key_check`. A test is not accepted merely because the property model agrees with itself. The model correction after ancestor contraction was made only after the SQL edge and production reader proved that a surviving branch-point ID had moved earlier in the composed parent transcript.

## Changed files

| File | Purpose |
| --- | --- |
| `docs/internals.md` | Documents v38/v39 replay policy and the closed dangling status vocabulary. |
| `polylogue/core/enums.py` | Adds `TopologyEdgeStatus.DANGLING_BRANCH_POINT`. |
| `polylogue/insights/transforms.py` | Preserves the typed dangling status in subagent child-link projection. |
| `polylogue/storage/sqlite/archive_tiers/index.py` | Bumps index schema to v39; adds dangling status and compact edge-owned witness table. |
| `polylogue/storage/sqlite/archive_tiers/write.py` | Persists witnesses; preserves merge lineage; normalizes delayed children; reconciles descendant subtrees; degrades/restores cuts; refreshes projections/delegations; explicitly cleans witnesses with FKs suspended. |
| `polylogue/storage/sqlite/lifecycle.py` | Declares v38 and v39 semantic-reparse lifecycle policy. |
| `polylogue/storage/sqlite/queries/session_links.py` | Persists dangling status through async topology upsert. |
| `tests/property/test_write_path_state_machine.py` | Adds semantic-ID replacement and shorten/reingest transitions, strict prefix bounds, cycle oracles, and permuted-history equivalence. |
| `tests/unit/insights/test_topology_cycle_rejection.py` | Exercises async persistence of dangling status and physical/read safety for quarantined cycles. |
| `tests/unit/insights/test_transforms.py` | Exercises downstream preservation of dangling status. |
| `tests/unit/storage/test_archive_tiers_ddl.py` | Protects the non-FK branch point and edge-owned witness DDL. |
| `tests/unit/storage/test_index_fast_forward_lifecycle.py` | Requires v39 authoritative replay and complete v33–v39 declarations. |
| `tests/unit/storage/test_lineage_normalization.py` | Adds direct/nested convergence, merge preservation, projection refresh, typed degradation/recovery, witness-aware rollback, and suspended-FK cleanup fixtures. |

No existing test or helper is deleted. No dominated deletion is proposed.

## Acceptance matrix

| Mission criterion | Result | Evidence |
| --- | --- | --- |
| Reuse saved Hypothesis failures on baseline and commit named three-oracle fixtures | Saved corpus unavailable; deterministic production-route reproduction complete | The snapshot contains no `.hypothesis/examples` database. Baseline traces reproduce two fixed points in both direct and nested six-order families. Named fixtures assert physical rows, link/witness rows, and composed reads. |
| Newest accepted sibling identity/text/order survives replacement races | Satisfied | All six orders converge to final `q0/q1/q1-alt`, child `c2` only, `q1-alt` cut, witness length 3, and complete final text. |
| Child-first, parent-first, and later replacement converge in every oracle | Satisfied | Direct and nested six-order fixtures plus standalone Hypothesis permutation property and state-machine transition. |
| Missing/replaced cut exactly rebinds or becomes typed and bounded | Satisfied | Parent shortening yields `dangling_branch_point`, unchanged child tail, retained witness, and tail-only incomplete read; semantic reingest restores exact composition. |
| Child prefix never exceeds surviving parent | Satisfied | State-machine invariant checks the bound for complete parents; missing cuts remain explicit incomplete states rather than indexing beyond parent content. |
| Transition is atomic and retry-safe | Satisfied in the production SQLite writer route | Fault after graph resolution rolls back physical, edge, witness, transcript, and completeness state; retry and repeated retry converge. Process-level daemon termination remains unverified. |
| Quarantined cycles remain typed/readable | Satisfied | Cycle fixture proves unchanged physical rows, quarantined relation, bounded complete owned reads, and no fabricated prefix. |
| Content-hash idempotency/import identity survives | Satisfied in adjacent focused tests | Equal-material hash, changed-content hash, stale replacement, and append replay tests pass; hashing/ID code is untouched. |
| Rebuildable index policy is respected | Satisfied | Canonical v39 DDL plus `SEMANTIC_REPARSE`; no in-place migration helper. Schema policy reports complete valid declarations 33–39. |
| Every changed hot-path branch has mutation-killing evidence | Satisfied for the new load-bearing branches | Seven explicit mutation runs fail the intended named tests; details are in `TESTS.md`. |

## Verification summary

Executed on the finalized primary tree:

- focused storage/DDL/lifecycle: `75 passed` in 22.77s;
- focused topology/insight transforms: `44 passed` in 2.71s;
- property module, seed `866039`: `4 passed` in 20.20s;
- property module, independent seed `866040`: `4 passed` in 20.14s;
- seed-866039 statistics: six permutation examples; 100 passing state-machine examples, four invalid draws, 18 transitions per accepted example;
- broad adjacent archive-write/delegation/composed-read/topology lane: `86 passed` in 26.65s;
- adjacent topology-edge lane: `14 passed` in 4.29s;
- adjacent content-hash/freshness/append laws: `4 passed` in 1.59s;
- Ruff 0.15.20 check and format check: passed for all 12 changed Python files;
- `py_compile`: passed for all 12 changed Python files;
- schema policy: versions 33–39 complete, no missing, duplicate, or invalid declaration;
- `devtools render docs-surface --check`: passed;
- `devtools render topology-status --check`: passed;
- `git diff --check`: passed;
- deterministic patched convergence trace SHA-256: `f8510a65861d12f74c0db8a3485d3bb2c20739eb39e84bd502bd036bb1b8c731`.

Executed from a fresh detached worktree at the exact base after applying the packaged patch:

- `git apply --check`: passed;
- apply: passed;
- resulting `git diff --check`: passed;
- regenerated full-index binary diff is byte-identical to `PATCH.diff`;
- focused storage/DDL/lifecycle: `75 passed` in 21.50s;
- focused topology/insight transforms: `44 passed` in 2.95s;
- property module, seed `866039`: `4 passed` in 21.32s;
- broad adjacent storage lane: `86 passed` in 22.21s;
- topology-edge lane: `14 passed` in 4.54s;
- schema policy and generated docs/status checks: passed;
- convergence trace is byte-identical to the primary patched trace.

The directly changed test files contribute 123 unique passing pytest items in the principal lanes (`75 + 44 + 4`). Adjacent suites are reported separately because some targeted identity checks may overlap with broader files; no inflated aggregate is claimed.

Seven representative mutation runs produced assertion failures in their intended tests:

- remove merge lineage preservation;
- remove deferred `action_pairs` refresh;
- remove descendant delegation refresh;
- stop subtree traversal after one level;
- disable semantic witness rebinding;
- drop witness retention during dangling degradation;
- remove explicit witness cleanup while foreign keys are suspended.

`TESTS.md` records the exact commands, results, and failed assertions.

## Apply order

1. Check out exactly `536a53efac0cbe4a2473ad379e4db49ef3fce74d`, or a descendant where the 13 touched files have not diverged.
2. Run `git apply --check PATCH.diff`.
3. Run `git apply PATCH.diff`.
4. Rebuild the derived index from authoritative durable input. V39 is intentionally `SEMANTIC_REPARSE`; do not attempt an in-place witness backfill from normalized v38 child tails.
5. Run the managed property, session-link, composed-read, and full verification lanes in the operator environment. Exact focused commands are in `TESTS.md`.
6. Before rollout, validate representative live lineages: direct and nested child tails, edge/witness rows, incomplete-lineage counts, action/delegation projections, inherited event references, high-fanout write latency, and retry behavior around daemon/process failure.

## Important limitations and residual risks

### Original saved Hypothesis database unavailable

The snapshot contains no `.hypothesis/examples` directory or saved-example IDs. `POLYLOGUE_HYPOTHESIS_REUSE_FAILURES=1` was set for final property runs, but there was no private corpus to reuse. Deterministic baseline traces reproduce the direct and nested order failures, and two fresh exact seeds pass; this is not literal replay of unavailable blobs.

### Managed repository environment incomplete

The repository’s `devtools test` wrapper refused the container’s 64 MiB `/dev/shm`, below its containment threshold. `devtools verify --quick` completed Ruff gates and then exceeded the execution bound in repository-wide mypy. A direct production-file mypy attempt also exceeded the bound after unrelated imported-file diagnostics. No successful full mypy or `devtools verify` claim is made.

A frozen `uv sync --frozen --extra dev` attempt failed after DNS retries while fetching locked `prompt-toolkit==3.0.52`. Successful focused suites used Python 3.13.5, SQLite 3.46.1, pytest 9.0.2, pytest-asyncio 1.4.0, pytest-timeout 2.4.0, Hypothesis 6.156.1, real `aiosqlite` 0.22.1, and Ruff 0.15.20 available in the container.

### Live daemon and archive unverified

No operator daemon, private archive, secrets, browser, NixOS deployment, or production-scale dataset was available. Real SQLite transaction rollback is exercised; process termination, daemon restart, and durable replay recovery are not.

### High-fanout and deep-tree cost

Reconciliation is proportional to the reachable prefix-sharing descendant subtree. The witness is fixed-size and the walk shares signature/composition caches, but only a small local 10/100-child probe was run. Deep and very high-fanout live archives remain the principal performance-validation item.

### Inherited event message references

Child `session_events.source_message_id` values referring to inherited parent rows can be nulled by existing foreign-key behavior during later parent identity churn. Initial and delayed normalization remap inherited references when evidence exists, but this patch deliberately does not persist a per-message semantic identity map. Transcript, edge, witness, projection, and read correctness are covered; historical event-reference continuity should be audited on live data.

### Maintenance helper scope

Normal writer routes refresh affected projections, threads, and delegation facts. The public stale-prefix maintenance helper remains a narrow relation repair and does not independently reproduce the full writer refresh set. V39 authoritative replay is the preferred operational route.

### Legacy witnessless rows

A deterministic legacy mapper remains subordinate to exact identity and semantic witnesses for manually created or pre-v39 witnessless rows. A correctly rebuilt v39 index should not depend on this fallback.

### Generated topology projection drift

`devtools render topology-projection --check` rewrites unrelated baseline LOC counts even on an untouched clean master and exits successfully. The generated target was therefore not included in this surgical patch. `docs-surface` and `topology-status` checks pass; the unrelated renderer drift is recorded in `EVIDENCE.md` rather than hidden.

## Value of another iteration

A **small repair** would add little unless integration finds an apply conflict, a managed-test regression attributable to this patch, or a package/document defect. The implementation now has clean-base apply identity, deterministic baseline/fixed traces, two property seeds, seven mutation experiments, rollback, schema, downstream-status, projection-refresh, merge-preservation, suspended-FK, and adjacent compatibility coverage.

A **substantial second pass** remains valuable only with evidence unavailable here: the original saved Hypothesis database, the complete managed/Nix environment, and a representative live archive. The highest-value work would be literal saved-example replay, full `devtools verify`, deep/high-fanout benchmarking, daemon process-fault injection around commit boundaries, and an audit of inherited event references through parent identity churn. That pass would materially improve operational confidence; current evidence does not point to a redesign of the implemented law.
