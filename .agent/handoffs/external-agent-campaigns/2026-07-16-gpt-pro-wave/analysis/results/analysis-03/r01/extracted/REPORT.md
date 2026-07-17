# Adversarial acceptance-criteria review of active large diffs

**Job:** `analysis-03`  
**Snapshot generated:** 2026-07-17T04:32:02Z  
**Review baseline:** bundled `refs/remotes/origin/master` at `0d081e5bc0b5e5bedcb3256d1779da8c0f091c65`  
**Checked-out snapshot head:** local `master` at `f654480cadb7cc4c194704e24dfd483199547b35`

## Executive judgment

No currently visible substantial branch, worktree head, recovery ref, or open PR should be merged wholesale.

The apparent active-diff set is mostly repository hygiene debt rather than unintegrated product value:

- Three large feature branches are patch-equivalent to squash commits already in current master: workload profiles to `c20286459` (#2934), raw-authority plan conservation to `593ef3c62` (#2961), and raw-authority unification to `d17ded51b` (#2962).
- The raw-authority proof-runner branch has the exact same tree as current master. Its predecessor scale-proof branch is superseded by the merged runner and later metering/host-admission work.
- The capture-job checkpoint branch is stale and strictly superseded by the larger receiver-authoritative implementation merged as `e6698a74e` (#2953).
- Two worktree heads are direct ancestors of current master and carry no active delta: schema journal regeneration at `067c87e49` (#2968) and fable/query execution control at `fd7b35492` (#2964).
- The only clearly unmerged broad production ref, `origin/worktree-wf_54d4fb2e-841-7` for `polylogue-48h`, is not salvageable as a merge candidate: a compilation check found 22 parse/indentation failures, 226 remaining calls to removed private helpers, no replacement aliases, no parity tests, and behavior changes that violate the owning Bead's no-semantics-change requirement.
- The unmerged `polylogue-pf1` companion test branch is merge-clean but does not repair the merged false-green contract. The closed Bead requires a generated AST/SQL inventory of every write path and a test that regenerates it. Current master instead validates a hand-written ten-row list and source/documentation strings. The companion test explicitly calls its SQL-semantic check a placeholder. `polylogue-pf1` should be reopened.
- `polylogue-hjpx.2` remains open correctly. Current master contains a real production-route scale-proof runner, but defaults are 16 components/24 raws, tests use 3/5, and the source says only an explicit July-15-sized invocation is production-shaped evidence. The snapshot contains no successful full-shape receipt.
- `polylogue-lkrc` also remains open correctly. The shared reconciler and status-truth corrections are in current master, but the stopped-daemon live postflight and the separately authorized `yla8` gate are not evidenced here.
- Testsuite Diet planning treats the complete `polylogue-1xc.14.1` outcome as a realized prerequisite even though that Bead and children `.1`, `.2`, and `.3` remain in progress. Those consumers must refresh against merged source and block on absent correlated generation, named C-03 canaries, full-scale receipts, privacy regeneration, and family/default semantics rather than invent parallel substitutes.

The integration strategy is therefore: normalize to current master; retire landed/superseded refs; reopen and freshly implement the false-green `pf1` contract; complete current P0/P1 proof programs from fresh branches; defer the P4 `48h` refactor until the hot storage/schema programs stabilize; refresh routine automation PRs last.

## Authority and scope

The review used this order of authority:

1. source at bundled current remote master;
2. repository instructions in `CLAUDE.md`;
3. current Beads export and Bead history/notes;
4. merged/open PR records and all-ref Git history;
5. older proposals, recovery packets, and Testsuite Diet planning only where consistent with 1–4.

The bundle's checked-out local master is five commits behind the bundled remote master. Those five commits are material follow-ups:

| Commit | Purpose |
| --- | --- |
| `0dc5773a9` | preserve complete raw frontier counts (#2965) |
| `805d49286` | stream safe raw-authority replay (#2966) |
| `9ddfc8d81` | add raw-authority scale-proof runner (#2967) |
| `067c87e49` | bound schema journal replay transactions (#2968) |
| `0d081e5bc` | guard and meter raw-authority proofs (#2970) |

Using local `f654480c` as the merge baseline would erase the exact follow-up evidence needed to adjudicate the active raw/schema branches. All branch comparisons in this package therefore use `0d081e5b`.

## Active-ref disposition

`Behind/ahead` and diff sizes are measured against `0d081e5b` through each ref's merge base. Conflict counts come from `git merge-tree --write-tree --messages` and count conflict messages, not textual conflict-marker occurrences.

| Ref / worktree | Owning area / Bead | Behind / ahead | Delta | Conflict messages | Adversarial disposition |
| --- | --- | ---: | ---: | ---: | --- |
| `feature/feat/schema-workload-profiles` (`d0cf780128`) and `polylogue-test-harness` | `polylogue-1xc.14.1` | 12 / 31 | 88 files, +10,543/-947 | 7 | **Retire.** Cumulative patch equals squash `c20286459` (#2934). The Bead remains open because #2934 was only a first slice; continue residual ACs from current master, not this ref. |
| `feature/integration/capture-job-authority` (`8ecc34ecce`) | `polylogue-06zm.1` | 53 / 2 | 9 files, +875/-44 | 9 | **Retire as superseded checkpoint.** #2953 is a larger, receiver-authoritative implementation covering exact-account profile-loss recovery and cache rehydration. Do not merge the older fallback model. |
| `feature/storage/raw-authority-ledger` (`34ed698553`) | `polylogue-hjpx.1` | 16 / 13 | 21 files, +4,281/-477 | 10 | **Retire.** Cumulative patch equals squash `593ef3c62` (#2961); current master also includes correctness/scale follow-ups #2965–#2970. |
| `feature/storage/raw-authority-unification` (`a294128820`) | `polylogue-lkrc` | 10 / 12 | 20 files, +2,634/-5,050 | 3 | **Retire.** Cumulative patch equals squash `d17ded51b` (#2962). The parent Bead remains open for live proof, not branch integration. |
| `origin/feature/test/raw-authority-proof-runner` (`fa13be13c9`) | `polylogue-hjpx.2` | 1 / 3 | 2 files, +312/-37 | 0 | **Retire.** Ref tree equals current master exactly. |
| `origin/feature/test/raw-authority-scale-proof` (`329081fd7d`) | `polylogue-hjpx.2` | 3 / 5 | 6 files, +548/-0 | 0 | **Retire as superseded.** The merged runner plus #2970 contains the current mechanism. Absence of a full-shape receipt is work to execute, not a reason to revive this ref. |
| `origin/feature/fix/lkrc-v35-legacy-native-id` (`adda2b96ac`) | historical `polylogue-lkrc.1` / current `polylogue-lkrc` | 195 / 20 | 17 files, +7,459/-86 | 9 | **Do not merge.** Most behavior landed or was superseded through later repairs and the unified reconciler. Transpose only the still-relevant source-v7 compatibility fixture into the current reconciler model. |
| `origin/worktree-wf_54d4fb2e-841-20` (`4eca0b90fd`) | `polylogue-pf1` | 98 / 2 | 1 file, +213/-0 | 0 | **Delete after preserving findings.** It is a second false-green test, not the required generated inventory. Reopen the Bead and implement freshly. |
| `origin/worktree-wf_54d4fb2e-841-7` (`b4217917ce`) | `polylogue-48h` | 98 / 1 | 41 files, +462/-461 | 17 | **Quarantine and rebuild.** It does not compile, leaves removed helper names live, lacks Bead evidence, and changes error/view/schema behavior. Do not repair by conflict resolution. |
| `polylogue-schema-regen` (`067c87e49`) | `polylogue-1xc.14.1.1` | ancestor | no active delta | n/a | **Remove worktree.** Head is already merged as #2968. |
| `polylogue-fable-program` (`fd7b35492`) | query execution control | ancestor | no active delta | n/a | **Remove worktree.** Head is already merged as #2964. |
| `feature/chore/prune-dead-protocols` (`3796f5452`) | cleanup | ancestor | no active delta | n/a | **Retire ref.** Already in current history. |
| `refs/remotes/bundle/ap7` (`d9db53a672`) | recovery packet for `polylogue-ap7` | 352 / 3 | 91 files, +18,304/-36 | 3 | **Archive only.** Current production slices were selectively adapted via #2700/#2736; proof-only/self-referential material was rejected. |
| `refs/remotes/recovered/demo-packet-v2` (`55c7833cc8`) | recovery packet for demo packet work | 352 / 8 | 82 files, +5,906/-98 | 9 | **Archive only.** Current-relevant pieces were selectively ported via #2704/#2709. Do not restore the stale subsystem. |

### Routine open PRs

The snapshot records seven open PRs, all release/dependency automation rather than substantial product implementation:

| PR | Title | Decision |
| ---: | --- | --- |
| #2701 | `chore(release): 0.3.0` | Regenerate/refresh after product ref cleanup and current changelog state. |
| #2858 | setup-uv 8.3.0 → 8.3.2 | Refresh/rebase last; rerun workflow gate. |
| #2859 | Python patch group | Refresh/rebase last; inspect lockfile regeneration. |
| #2860 | coverage 7.14.1 → 7.15.1 | Refresh/rebase last; rerun coverage workflow. |
| #2861 | syrupy 5.1.0 → 5.5.3 | Refresh/rebase last; review snapshot format drift. |
| #2862 | tree-sitter 0.25.2 → 0.26.0 | Refresh/rebase last; this is the highest semantic-risk dependency update of the group. |
| #2863 | mypy 1.17.1 → 2.3.0 | Refresh/rebase last; expect typing-policy and plugin diagnostics rather than treating lock resolution as proof. |

None has an owning Bead in the snapshot, and none should be inserted into the product-branch integration sequence merely because it merges cleanly.

## Important falsifications

### `polylogue-pf1` is closed on unsupported evidence

The Bead requires three concrete outcomes: a committed artifact classifying every write-path method, zero unexplained divergences, and a test that regenerates the diff and fails on a new divergence. The close reason asserts that PR #2897 did this and reports nine passing tests.

Current source does not implement the claimed oracle:

- `tests/unit/storage/test_storage_twins.py` defines ten `StorageTwinDivergence` objects by hand.
- Its strongest assertions count those ten rows, validate field/status strings, search comments/docstrings, and check that expected modules/method-name substrings exist.
- It does not extract method inventories, SQL statements, tables written, pragma behavior, or error semantics from either lane.
- It cannot notice an eleventh write path unless a human also edits a string the test happens to read.
- `docs/plans/STORAGE_TWINS_DIVERGENCES.md` is a narrative ten-row classification, not output regenerated by a source-derived function.
- The unmerged `test_storage_twins_divergence.py` parses class names and documentation structure but explicitly labels same-table SQL consistency “a placeholder” and only asserts that two files exist.

Passing these tests, whether nine or any other count, does not satisfy the Bead's anti-vacuity criterion. The status should return to open before the downstream twin-collapse program treats it as a proven prerequisite.

### `polylogue-48h` is structurally broken, not merely stale

The branch commit claims strict typing verification, but a direct `python -m compileall -q polylogue` check failed in 22 production files. Typical failures are imports inserted at invalid indentation inside `try`/function bodies. Static source counts also show:

- 186 calls to `_table_exists(` and zero definitions;
- 34 calls to `_column_exists(` and zero definitions;
- 6 calls to `_index_exists(` and zero definitions.

The replacement `polylogue/storage/sqlite/introspection.py` is not a semantics-preserving normalization:

- `table_exists()` propagates `sqlite3.Error`, while at least `operations/archive_debt.py::_table_exists` previously returned `False` on SQLite errors.
- The new default excludes views unless `include_views=True`; some old helpers intentionally treated tables and views alike.
- `schema` and PRAGMA table identifiers are interpolated without a validated identifier policy.
- `column_exists()` has no schema parameter despite schema-sensitive old call sites.
- No new parity tests, ownership map, compatibility notes, public-model checks, or import/layering artifact are present.
- Topology files were regenerated even though the production tree cannot parse; generated-file presence is therefore not closure evidence.

Testsuite Diet's adjudicated policy also rejects the Bead design's proposed source-spelling/grep tripwire. The fresh implementation should enforce behavior and import ownership, not memorialize private helper names.

### “Landed” is not the same as “proved at required scale”

`devtools/raw_authority_scale_proof.py` is a real route: it builds an archive, writes source raws, calls `repair_raw_materialization`, records census receipts and resource counters, and requires two matching quiescent digests. That is valuable production wiring.

It is not the requested July-15 proof by existence alone. Defaults are 16 components and 24 raws; unit tests use 3 components and 5 raws; the receipt notes state that only an explicit July-15-sized invocation is production-shaped evidence. Current Bead notes record an admitted 10,163-component/15,264-direct-raw scheduler-shape run and a terminated medium run, but explicitly do not claim the 4.788 GiB / 21,398-expanded-candidate result. No successful full result artifact is included in the snapshot.

### Testsuite Diet has a stale realized-baseline assumption

The Diet archive correctly says not to create a second workload identity, profile schema, or shadow oracle. That architectural constraint remains sound. Its execution baseline is nevertheless ahead of the source authority: it says to assume correlated variants, named scale/selectivity tiers, C-03, shared receipts, and the complete `polylogue-1xc.14.1` outcome are realized. Current Beads explicitly keep the parent and three children in progress, and current notes identify live full-corpus replay economics, content-shaped property-key regeneration, 55-family default semantics, production-route canaries, and live promotion as residual.

Diet workers must consume only the subset actually present in current master and wait for the remaining upstream contracts. They must not fill the gap with Diet-owned equivalents.

## Uncommitted and stranded value

The extracted working tree is source-clean relative to checked-out local master except for `browser-extension/package-lock.json`. That file is explicitly ignored by `.gitignore`, is generated package-manager state, has no repository history, and is not stranded product value. The apparent mass deletion from reconstructing a Git worktree is an extraction artifact: the working-tree tar omits tracked `.agent`, `.beads`, and lock content.

Actual stranded value is narrower:

1. The old `lkrc-v35` branch contains intent for a legacy source-schema-v7 fixture. Its incident-specific mechanism is obsolete, but the compatibility question is still useful. Re-express it as a genuine current unified-reconciler fixture rather than cherry-picking the old command/repair stack.
2. The stale `pf1` branches expose the precise false-green failure and can seed a fresh test design, but no code should be carried over as the oracle.
3. The `48h` branch's call-site census is useful as navigation only. The replacement module and mass import edits are unsafe.
4. Recovery refs retain historical reasoning, but current Beads and selectively merged PRs already adjudicated which production pieces survived.

## Integration order

1. **Normalize repository state.** Fast-forward local master from `f654480c` to bundled `0d081e5b`; record or tag stale heads before deleting worktrees; do not merge them.
2. **Retire already-landed/superseded refs.** Remove workload-profile, raw-ledger, raw-unification, raw-proof, schema-regen, fable, capture checkpoint, and dead-protocol refs/worktrees after confirming no unpushed worktree-only files.
3. **Correct tracker truth.** Reopen `polylogue-pf1`, attach the source contradiction, and remove “regenerates the diff” from closure evidence. Keep `polylogue-hjpx.2`, `polylogue-lkrc`, `polylogue-1xc.14.1`, and children open.
4. **Repair `pf1` fresh from current master.** This is a prerequisite for any twin-collapse migration and is independent enough to land early.
5. **Finish the current P1 schema proof chain.** Close `.1` production-scale journal/replay economics and cleanup; then `.2` live privacy regeneration/promotion; then `.3` family/default semantics. Only then claim parent canaries/receipts/promotion complete.
6. **Run and retain the full `hjpx.2` proof.** Use the merged guarded runner on an admitted host, add fairness/conservation mutation variants, and retain exact receipts. Do not alter live archives.
7. **Close `lkrc`/`yla8` live boundary.** Only after fixed-point proof and the separate backup/quiescence/operator authorization gate; retain a stopped-daemon postflight showing no unreported frontier gaps.
8. **Continue CaptureJob parent slices.** Land `polylogue-06zm.2` event/timeline projection before `.3` retention/quota/migration so lifecycle policy consumes the durable event model rather than inventing one.
9. **Rebuild `polylogue-48h` last among these product refs.** It is P4 and touches hot storage/schema routes. Start from current master after P0/P1 churn, migrate semantic cohorts in small batches, and prove parity per cohort.
10. **Refresh routine automation PRs.** Recreate/rebase release and dependency updates on the resulting master, then run their owning workflow/type/snapshot gates.

There is no safe sequence in which the stale `48h`, `pf1`, capture, or raw-authority branch tips are merged first and cleaned up afterward.

## Limits and missing evidence

This review is bounded to the attached snapshot. It did not contact GitHub, read a live daemon, inspect a live archive, run a browser extension, execute deployment, or verify changes that may have occurred after the snapshot generation time.

Only one production-code execution check was run: `python -m compileall -q polylogue` on the isolated `polylogue-48h` branch, which failed with 22 errors. No pytest, JavaScript, mypy, `devtools verify`, live scale, daemon, or browser command was run by this audit. Test counts and green-gate statements elsewhere in the package are explicitly attributed to PR/Bead records, not independently reproduced.

Static evidence included bundle/ref enumeration, Git ancestry, patch-id comparison, tree comparison, diff statistics, merge-tree simulation, source/path inspection, current Beads, PR records, and working-tree hash comparison.

The highest-value next iteration would consume a newer project-state archive after tracker corrections and proof execution. It should specifically attempt to falsify: the new `pf1` inventory's completeness against all SQL construction forms; `hjpx.2` fairness/conservation mutations at full shape; the current reconciler's source-v7 and stopped-daemon fixtures; and the schema journal's live replay query plan/resource receipt. Another pass before those changes exist would mostly reconfirm stale-ref hygiene rather than reduce product uncertainty.
