# Next actions and repair prompts

## 1. Normalize and retire stale repository state

**Owner:** repository integrator  
**Starting point:** bundled `origin/master @ 0d081e5b`

1. Record `git show-ref`, worktree paths, and any untracked/ignored files for each stale worktree.
2. Tag or archive ref names locally if historical review trace is required.
3. Fast-forward local master; do not merge old feature heads.
4. Remove worktrees/branches whose patch/tree/ancestor proof is recorded in `EVIDENCE.md`.
5. Delete ignored `browser-extension/package-lock.json` unless a separate policy decision makes npm lock authority explicit.
6. Leave recovery namespaces read-only or prune them under an explicit archival policy; never present them as pending product branches.

**Acceptance:** one clean current-master checkout; no substantial “active” ref that is patch-equivalent, tree-equivalent, ancestor-only, or explicitly superseded.

**Falsification:** any worktree-local file not represented by its Git ref and not ignored/generated must halt deletion and be classified against an owning Bead.

## 2. Correct `polylogue-pf1` and implement a real generated inventory

**Tracker action:** reopen `polylogue-pf1`; replace the close reason with the source contradiction and link downstream `polylogue-hiu`/`polylogue-a7xr` dependencies.

### Implementer prompt

> Create a fresh branch from current master for `polylogue-pf1`. Do not copy either existing twins test as the oracle. Define the authoritative async and sync storage lane roots from current architecture. Build one deterministic inventory function that parses Python AST and recoverable SQL string expressions, associates each write-capable method with normalized operation intent, tables/schema objects written, pragmas/transaction/error policy, and source locations, and marks unsupported/dynamic constructs explicitly rather than dropping them. Compare lanes by stable semantic operation and table intent, not by method spelling alone. Classify every row as equivalent, lane-only with rationale, divergent-to-fix, or unresolved-parser coverage. Generate `docs/plans/STORAGE_TWINS_DIVERGENCES.md` (or a compact JSON/CSV plus readable rendering) from this same function. The test must regenerate in a temporary directory and compare semantic content to the committed artifact. Adding a new write method, changing same-table SQL semantics, or making SQL unparsable without an explicit unresolved row must fail. Do not create a second hand-authored storage model or require source spellings merely to exist.

### Required owning paths

- async lane: current `polylogue/storage/sqlite/async_sqlite*.py` and delegated write mixins actually used by it;
- sync lane: current `polylogue/storage/sqlite/archive_tiers/` write paths and shared query/write modules reached by it;
- generated artifact: `docs/plans/STORAGE_TWINS_DIVERGENCES.md` and, only if useful, one compact machine-readable inventory;
- tests: `tests/unit/storage/test_storage_twins.py` rewritten around the generator, plus known-answer parser tests in a dedicated test module if needed.

### Closure gates

- every in-scope write method appears exactly once or has an explicit parser-coverage blocker;
- every normalized write table/intent is classified;
- no unexplained divergence remains;
- artifact is byte/content deterministic under shuffled file discovery;
- mutation: add a new same-table write with incompatible conflict/update semantics; test fails;
- mutation: hide SQL behind an unsupported construct; test fails as unresolved rather than silently omitting it;
- mutation: alter transaction/error handling where the lanes claim equivalence; test fails;
- targeted storage-twin command and repository quick gate are run by the local implementer and exact receipts are attached.

**Do not accept:** a fixed row count, docstring-number check, filename existence check, method-name grep, or test that only validates its own constant table.

## 3. Finish `polylogue-1xc.14.1.1` production-scale replay/memory proof

### Implementer prompt

> Continue from current master with #2968 present. Enumerate every remaining full-corpus retention site in schema inference/replay and prove no list/set/map grows with samples, memberships, scopes, paths, tool IDs, or distinct values. For the live representative archive, capture `EXPLAIN QUERY PLAN` and phase receipts proving membership replay is selective and does not globally scan samples per package. Define safe journal checkpoint/transaction boundaries that bound WAL and replay while preserving deterministic indexed passes. Run 1x/10x generated scaling and the representative production-scale generation under the shared `polylogue-1xc.14` receipt: peak RSS/PSS, journal/WAL bytes, I/O, timings, cancellation/progress, and cleanup. Exercise success, exception, cancellation, and abrupt worker death; remove DB/WAL/SHM and recover stale runs. Do not claim closure from small scaling alone.

### Closure gates

- source audit has no corpus-proportional retention path;
- 1x/10x counts scale by ten while peak Python memory stays within declared fixed overhead/buffers;
- query plan and timings demonstrate selective membership replay;
- exact/additive counts and approximation/loss metadata survive merge/order changes;
- cleanup proves scratch-root restrictions and stale abrupt-death recovery;
- known-answer/shuffled input output is deterministic;
- targeted and quick-gate receipts are retained.

## 4. Finish workload-profile privacy and catalog semantics

### `polylogue-1xc.14.1.2`

> Use one shared dynamic-key classifier across field-stat traversal, fingerprints, schema collapse, generation, and validation. Seed sentence/question, path/XML, control-character, overlength, secret/token, and review-only values. Stage a live Claude Code regeneration without mutating committed packages. The promotion scanner must hard-block secrets and content-shaped property keys, while separately inventorying enums, dates, domains, email/account-like values, paths, IDs, and rare strings with artifact/path/frequency. Replace current committed packages only after the review artifact is accepted.

**Falsifier:** any previously observed content-shaped key remains in a committed property name, or any subsystem classifies the same key differently.

### `polylogue-1xc.14.1.3`

> Represent `latest`, `recommended`, `default`, evidence family, and promoted release as distinct semantics. Retain all positive-value families and exact/profile/scope resolution. Select recommended/default by deterministic support/compatibility rules, not observation recency. Regenerate the live 55-family corpus, emit coverage/novelty/time-window/default rationale, and prove a new rare family cannot become default merely by being newer.

**Falsifier:** remove the dominant family, reorder input, or add a newer rare family; resolution must remain deterministic and every retained family reachable.

## 5. Complete parent workload canaries and receipts

**Owner:** `polylogue-1xc.14.1` after children stabilize.

- Implement correlated joint variant generation rather than independent marginals.
- Make provider-native wire artifacts reach production acquire, parser, materializer, index, and query routes.
- Define named scale/selectivity tiers from the profile mechanism.
- Land C-03: mixed archive plus exact-session action query, with planted independent facts and an explicit selective work bound.
- Add tool pairing, lineage replay, active-growing, and partial-convergence canaries from the same profile/receipt identity.
- Inventory performance tests and remove/redirect any independent corpus identity or resource envelope.
- Apply parser and query-pushdown mutations; the intended canary must fail.

**Testsuite Diet handoff:** publish a versioned availability manifest listing which workload identities, tiers, canaries, and receipt fields are actually merged. Diet workers consume this manifest and block on absent prerequisites.

## 6. Execute `polylogue-hjpx.2` at the required shape

### Implementer prompt

> Use current `devtools/raw_authority_scale_proof.py` with #2970's host-admission and resource metering. Extend the generated corpus, if necessary, to reproduce the July-15 direct/expanded candidate counts, authority component distribution, 4.788 GiB retained payload, revision/path/bundle skew, broken seeds, cursor-ahead sources, incomparable heads, missing blobs, and blocked cohorts within documented tolerances. Keep all content synthetic/private-free and bind the receipt to corpus bytes. Run on an admitted contained host. Record each bounded pass's complete state counts, executable backlog, carried/deferred/terminal states, retry identity, fairness/wait age, cursor/source/index/FTS/raw-authority invariants, RSS/PSS/swap, I/O, temp/database growth, wall/CPU time, and daemon health latency. Interrupt/resume at defined boundaries. Require two final identical quiescent census digests. Add two reversible variants: remove fair rotation and remove conservation/carry-forward accounting; retain the expected starvation/census-mismatch failures. Do not touch a live archive.

### Durable result surface

Retain:

- exact command and commit/tree id;
- admission/containment environment;
- corpus seed, byte digest, and shape tolerances;
- machine-readable per-pass ledger;
- shared workload receipt and resource envelopes;
- interruption/resume points;
- mutation commands/results;
- final result SHA-256 and artifact location.

**Closure prohibition:** the default 16/24 run, the unit 3/5 run, or the scheduler-cardinality-only 10,163/15,264 run cannot close the Bead.

## 7. Close current `polylogue-lkrc` residuals without reviving old mechanisms

### Source-v7 compatibility fixture

> Construct a genuine current source-schema-v7 archive/fixture through supported migration/setup code. Feed it to the current shared `raw_reconciler.py` census/plan/apply/postflight route. Verify legacy native-id/capture-mode shape is classified and converges or produces an explicit typed blocker. Do not restore old incident CLI commands, proof digests, or receipt types. Mutate legacy identity/bytes and assert fail-closed CAS/judgment/reacquisition behavior.

### Live gate

After `hjpx.2` succeeds and `yla8` prerequisites are met:

1. verified backup and restore proof;
2. daemon stopped and quiescent census;
3. explicit operator authorization;
4. apply through the single reconciler only;
5. complete postflight state inventory including `proven_current` and remediation refs;
6. two matching quiescent digests and zero unreported frontier gaps;
7. retained rollback/receipt evidence.

**Do not accept:** sidecar/index health alone, preflight counts reused as postflight, manual SQL, raw deletion, force replay, or a separate browser-origin repair lifecycle.

## 8. Continue CaptureJob parent in order

### `polylogue-06zm.2`

Land the durable receiver event/timeline projection over the #2953 job identity and receipts. Events must be monotonic/idempotent, queryable by stable job/scope, and reconstructible without extension-local caches.

### `polylogue-06zm.3`

Only after `.2`, implement retention/quota/migration/terminal lifecycle policy over durable jobs/events. Prove profile wipe, lease expiry, old-client reconnect, orphan migration, quota pressure, and terminal cleanup without acknowledged-page replay or cross-account disclosure.

**Do not reopen:** the stale `paired:<provider>` checkpoint model.

## 9. Rebuild `polylogue-48h` from current master

**Scheduling:** after current P0/P1 raw/schema changes; this is a P4 consolidation.

### Implementer prompt

> Start a fresh branch. First commit a before/after ownership and semantics map for every SQLite introspection helper and call site: sync/async connection type; main/attached schema; table/view/virtual-table/trigger/index meaning; quoting/identifier source; error policy; row-factory assumptions; transaction context; and whether absence versus query failure is distinguishable. Group call sites into semantic cohorts. Design a small shared module with validated schema/object identifier handling and explicit policy parameters or adapters; do not force unlike semantics into one default. Add known-answer behavior tests before migration for each cohort, including attached schemas, views, virtual tables, malformed/unavailable DB, and async cursor lifecycle. Migrate one cohort at a time; after each cohort run compile, strict typing for touched modules, parity tests, and public contract tests. Delete old helpers only when the call census reaches zero. Regenerate topology once at the end and retain an import/layering diff. Do not add a grep/source-spelling denylist.

### Mandatory gates

- ownership map covers all definitions and calls on current master;
- no interpolated schema/table identifier without validation/quoting policy;
- explicit error semantics preserve old behavior or carry a migration/release note;
- attached schema, view, virtual table, trigger/index, and async cases have parity fixtures;
- `rg` shows zero unresolved removed helper calls, but this census is review navigation rather than the test oracle;
- `python -m compileall -q polylogue` passes;
- targeted strict typing and tests pass;
- import/layering and public-model compatibility evidence is retained;
- generated topology is current.

**Do not reuse:** the mass import edits or replacement API from `b4217917ce` without independently re-deriving semantics.

## 10. Refresh automation PRs

After product integration:

- regenerate #2701 release metadata/changelog from current master;
- refresh #2858–#2863 instead of resolving stale lock/workflow output manually;
- for tree-sitter, run parser/provider/schema route tests, not just installation;
- for mypy 2.3, classify new diagnostics and preserve strictness rather than blanket-ignore;
- for syrupy, inspect intentional snapshot serialization diffs;
- for coverage, verify configured branch/source semantics and report generation;
- retain exact workflow receipts.

## Local verification checklist for the next archive

The next project-state export should include:

- current remote and local heads plus all worktrees;
- clean/ignored/untracked classification;
- current Beads with corrected `pf1` status;
- fresh branch diffs and PR bodies/reviews;
- generated artifacts and their producing command/version;
- targeted command outputs/receipts, not only prose summaries;
- full `hjpx.2` result or explicit failed/blocked receipt;
- schema journal full-scale result/query plan;
- any `48h` ownership map and parity matrix;
- exact archive timestamp so later changes are not silently treated as present.
