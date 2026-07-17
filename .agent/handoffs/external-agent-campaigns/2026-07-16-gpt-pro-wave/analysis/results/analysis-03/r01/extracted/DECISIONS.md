# Decisions

## Decision register

### D01 — Use bundled remote master as the integration authority

**Decision:** compare every ref to `0d081e5bc0b5e5bedcb3256d1779da8c0f091c65`, not checked-out `f654480c`.

**Observed basis:** the bundle contains five newer merged commits, including corrections to complete raw-state counts, stream-safe replay, schema journal transaction bounds, and scale-proof metering.

**Falsifier:** a newer attached authoritative snapshot that shows a later remote master or proves these refs were deliberately rebased elsewhere.

### D02 — Merge none of the substantial visible refs wholesale

**Decision:** every large visible ref is either already landed, superseded, archival, false-green, or structurally broken.

**Observed basis:** exact patch/tree/ancestry results plus source/AC review.

**Falsifier:** a ref-local commit or untracked worktree file absent from the archive that is not in current master and independently satisfies an open AC. Preserve tags/ref names before deletion so such evidence can be recovered.

### D03 — Retire exact-landed and ancestor refs

**Refs:** workload profiles, raw ledger, raw unification, raw proof runner, raw scale predecessor, schema regen, fable program, dead-protocol cleanup.

**Action:** remove worktrees and branches only after checking for worktree-local files; do not make merge commits.

**Reason:** retaining equivalent pre-squash histories creates false active work and encourages conflict resolution against code already integrated.

### D04 — Retire the CaptureJob checkpoint as superseded

**Decision:** keep #2953/current master; delete `feature/integration/capture-job-authority` after preserving any review comments.

**Reason:** the stale branch's fallback scope model is weaker and its nine touched files all conflict with current receiver/extension routes. Its parent residuals belong to `.2`/`.3`, not to this branch.

### D05 — Reopen `polylogue-pf1`

**Decision:** change tracker status from closed to open and replace its close evidence.

**Observed basis:** current test does not regenerate any source-derived write-path/SQL/table diff; companion test explicitly contains a placeholder.

**Required closure:** one generator enumerates the defined lane universe, normalizes write operations, emits a stable committed artifact, and provides the test oracle. Adding/changing a write path must fail without updating classification through the generator.

**Not accepted:** a count of ten, docstring numbering, filename/method substring assertions, or an independently hand-authored second model.

### D06 — Quarantine `polylogue-48h`; do not repair the branch in place

**Decision:** retain the ref only as forensic evidence; create a new branch from current master when scheduled.

**Observed basis:** 22 compile failures, 226 unresolved private-helper calls, 17 merge conflicts, missing Bead artifacts/tests, and semantic changes.

**Reason:** conflict resolution would obscure whether failures came from stale base, automated edits, or API design. A mechanical re-run without a semantic inventory would reproduce the defect.

### D07 — Keep `polylogue-hjpx.2` open

**Decision:** runner landing is a prerequisite, not closure.

**Required closure:** successful admitted-host July-15-shape receipt, exact distribution/bytes, pass/resource/health data, interruption invariants, two matching quiescent digests, and fairness/conservation mutation failures. No live archive apply.

### D08 — Keep `polylogue-lkrc` open while accepting the merged unification code

**Decision:** retire the unification branch, retain the P0 Bead.

**Reason:** source unification and state-count truth landed; stopped-daemon live postflight and separate `yla8` authorization boundary remain absent.

### D09 — Refresh Testsuite Diet's baseline, but preserve its non-duplication rule

**Decision:** change “assumed realized” to a versioned prerequisite matrix tied to current merged symbols/receipts. Block clusters that require C-03, full correlated variants, or shared full receipts until those are landed.

**Not accepted:** Diet-owned workload identities, profile schemas, tier taxonomies, query/storage shadow oracles, or source-spelling gates.

### D10 — Transpose, do not cherry-pick, old `lkrc-v35` compatibility intent

**Decision:** add a current unified-reconciler source-schema-v7 fixture if current Bead ownership still requires the case.

**Reason:** the old branch's repair commands and incident-specific receipt lifecycle were intentionally removed by #2962. Cherry-picking would reintroduce parallel authority.

### D11 — Treat ignored extension lockfile as disposable state

**Decision:** do not include `browser-extension/package-lock.json` in a repair or merge unless repository policy changes explicitly.

**Reason:** `.gitignore` excludes it and no history indicates it is a committed authority surface.

### D12 — Keep recovery refs archival

**Decision:** do not merge `bundle/ap7` or `recovered/demo-packet-v2`.

**Reason:** later current-source PRs selectively assimilated valid pieces and Beads explicitly rejected stale/self-referential proof shapes.

### D13 — Refresh automation PRs only after product-state normalization

**Decision:** #2701 and #2858–#2863 are last in order.

**Reason:** release/changelog and dependency lock/type/snapshot outcomes should be generated from the settled product baseline. A clean textual merge is not enough for tree-sitter or mypy major/minor changes.

## Ordered integration plan

| Order | Change | Entry gate | Exit gate | Unsafe alternative rejected |
| ---: | --- | --- | --- | --- |
| 0 | fast-forward/normalize local master to `0d081e5b`; tag stale refs | bundle integrity; no worktree-local value hidden | one current baseline, clean worktree inventory | evaluating against `f654480c` |
| 1 | retire exact-landed/superseded refs and ancestor worktrees | patch/tree/ancestry evidence recorded | no duplicate active branches | merging equivalent histories |
| 2 | tracker correction for `pf1` and stale assumptions | source contradiction attached | Bead statuses match evidence | leaving false closure as downstream authority |
| 3 | fresh `pf1` generated inventory | current master; exact lane scope approved | generated artifact + mutation-sensitive test | merging either old twins test branch |
| 4 | schema `.1` replay/memory closure | #2968 current master | full receipt/query plan/cleanup | claiming small 1x/10x proof is production closure |
| 5 | schema `.2` privacy regeneration and `.3` family semantics | `.1` stable replay substrate | reviewed current packages and defensible catalog roles | promoting content-shaped keys or rare-latest default |
| 6 | parent workload canaries/receipts/promotion | children stable | C-03/related route mutations, shared receipts, no parallel scenarios | letting Diet fill missing upstream contracts |
| 7 | full `hjpx.2` proof | admitted host; merged #2970 runner | durable full-shape receipt plus mutations | using tiny unit run as scale proof |
| 8 | `lkrc` stopped-daemon/`yla8` boundary | full fixed-point proof, backup, quiescence, authorization | zero unreported gaps and retained postflight | live mutation before authorization |
| 9 | CaptureJob `.2`, then `.3` | #2953 core stable | events/timeline first, retention/quota/migration second | lifecycle policy on cache-local state |
| 10 | fresh `48h` semantic-cohort refactor | P0/P1 storage/schema churn settled | compile/type/parity/layering/public-contract proof | resolving 17 conflicts in the broken branch |
| 11 | release/dependency automation | settled master | owning workflow/type/snapshot gates | merging stale release/lock outputs earlier |

## Merge-order hazards

1. Merging schema/raw pre-squash branches before current master would overwrite later transaction, count, stream, and metering corrections.
2. Merging old capture code would reintroduce weaker scope/adoption semantics into a completed receiver-authoritative route.
3. Merging `48h` before current raw/schema proof completion would touch the same readiness, repair, archive, FTS, and source surfaces while lacking behavior parity.
4. Closing or consuming `pf1` before a real source-derived oracle makes the planned storage-twin collapse depend on a circular proof.
5. Running Diet implementation against assumed C-03/tiers/receipts before they exist invites precisely the parallel product model the planning documents prohibit.
6. Running live `lkrc` repair before `hjpx.2` and `yla8` gates confuses synthetic fixed-point evidence with live authorization.

## Residual disputes

- Current Bead/PR records say `pf1` is independently verified. This report disputes closure based on direct source. A new test run cannot resolve the dispute unless the test's oracle is replaced.
- `hjpx.1` is closed, yet #2965 corrected a status-truth gap after #2961. This report accepts current-master satisfaction while rejecting the idea that the stale branch/initial close record alone was complete.
- Testsuite Diet labels its future substrate a “realized baseline” while explicitly saying those changes are still landing. This report treats it as a planned consumer contract, not current source truth.
- The old `48h` design permits a grep lint; the newer adjudicated test policy forbids source-spelling gates. Current policy and behavior-focused acceptance criteria control.

## Conditions for another adversarial pass

A second pass has high value after at least one of these artifacts exists:

- a fresh `pf1` generator/artifact/test;
- a full `hjpx.2` result receipt and mutation results;
- the schema `.1` production-scale replay receipt/query plan;
- a current source-v7 unified-reconciler fixture;
- a fresh staged `48h` semantic-cohort branch.

At that point, the next review should actively seek omitted dynamic SQL, decorator/generated methods, views/attached schemas, retry/starvation edge cases, receipt self-certification, and current call sites outside the declared ownership map.
