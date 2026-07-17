# Evidence and per-acceptance-criterion matrix

## Evidence labels

Every conclusion below uses one of four labels:

- **Observed:** directly present in source, Git objects, current Beads/PR snapshot, or output of a command run during this review.
- **Source-supported inference:** the source and history support the conclusion, but the complete runtime behavior was not executed by this review.
- **Unresolved:** evidence required by the acceptance criterion is absent, contradictory, or outside the snapshot.
- **Recommendation:** an integration or repair action, not a claim about current behavior.

A passing test record is not promoted to “Observed runtime behavior” unless this audit ran it. Bead/PR-reported commands remain provenance-tagged evidence.

## Commands and checks actually performed

| Check | Result | Use |
| --- | --- | --- |
| Bundle/ref enumeration and ancestry | completed | identify local heads, remote heads, pseudo-worktree heads, recovery refs, ancestors, and current bundled master |
| `git rev-list --left-right --count`, merge-base diff stats | completed | stale-base and size inventory |
| Stable patch-id comparison | completed | exact equivalence of three feature branches to merged squash commits |
| Git tree comparison | completed | raw-proof runner tree equals current master |
| `git merge-tree --write-tree --messages` | completed | conflict messages/path classes without modifying a checkout |
| Working-tree file hashing | completed | distinguish real untracked state from omitted archive content |
| Static source/Bead/PR inspection | completed | production route and test trace; AC adjudication |
| `_table_exists`/`_column_exists`/`_index_exists` call/definition census on `48h` | completed | prove removed-private-helper calls remain unresolved |
| `python -m compileall -q polylogue` on isolated `48h` checkout | **failed: 22 files** | falsify branch verification claim and merge readiness |
| pytest / JavaScript tests / mypy / `devtools verify` | **not run** | no independent green claim |
| live daemon/browser/archive/deployment/full-scale proof | **not accessed** | no live claim |

## Baseline and repository state

**Observed.** `polylogue-overview.md` identifies checked-out `master @ f654480c`, dirty state, and snapshot generation at 2026-07-17T04:32:02Z. The bundled all-ref repository has `refs/remotes/origin/master @ 0d081e5b`, five commits ahead. The remote tip contains the latest source in the attached authority and is used throughout.

**Observed.** Repository instructions require Bead AC honesty, fresh branch/squash integration, targeted verification, topology regeneration for new modules, and rejection of old/parallel devloop mechanisms. Those rules exclude “merge then discover intent” and source-spelling gates as substitutes for behavior.

**Observed.** The extracted working-tree source matched local `f654480c` except ignored `browser-extension/package-lock.json`. Tracked `.agent`, `.beads`, and lockfile omissions in the tar explain the false mass-deletion status produced by naively overlaying it onto a clone.

## Patch, tree, and ancestry proof

| Ref | Proof | Adjudication |
| --- | --- | --- |
| `feature/feat/schema-workload-profiles` | stable patch id `81739ae638f54f6902834da110321ca44aef0167`, equal to squash `c20286459` | all branch code landed; residual Bead ACs are new work |
| `feature/storage/raw-authority-ledger` | stable patch id `31aeb2456c68023932eac0385d3ff49027585f32`, equal to squash `593ef3c62` | branch fully landed |
| `feature/storage/raw-authority-unification` | stable patch id `daa985a58b834d6cceb82aeb83842d7892f3bb0a`, equal to squash `d17ded51b` | branch fully landed |
| `origin/feature/test/raw-authority-proof-runner` | branch tree id equals `origin/master` tree id | no unmerged tree value |
| `polylogue-schema-regen` | head `067c87e49` is in current master | no active delta |
| `polylogue-fable-program` | head `fd7b35492` is in current master | no active delta |
| `feature/chore/prune-dead-protocols` | ancestor of current master | no active delta |
| `feature/integration/capture-job-authority` | no patch equality to #2953; older 9-file checkpoint versus merged 16-file implementation | superseded, not additive |

## Conflict simulation

| Ref | Conflict messages | Important conflict classes/paths |
| --- | ---: | --- |
| workload profiles | 7 | add/add in `polylogue/scenarios/workload.py`, `observation_journal.py`, and receipt/journal tests; content in generated topology and `cluster_collection.py` |
| capture checkpoint | 9 | add/add in current CaptureJob production/tests; content in background, contracts, server, topology |
| raw ledger | 10 | add/add in `raw_authority.py`; content across maintenance, backfill, readiness, repair, and tests |
| raw unification | 3 | add/add in `raw_reconciler.py`; readiness/test content |
| old `lkrc-v35` | 9 | modify/delete old monolithic maintenance CLI; add/add current repair tests; content in live source, backfill, repair, archive store |
| `pf1` companion | 0 | merge-clean but semantically inadequate |
| `48h` | 17 | generated topology plus readiness, live source, blob, embedding, FTS, raw retention, search, source sessions, and archive write paths |
| semantic recovery packet | 3 | current archive session repository, facade contracts, semantic-card test |
| demo recovery packet | 9 | current demo tooling, registry, docs, tour, fixtures, and tests |

Conflict-free does not mean acceptable: `pf1` is the strongest counterexample in this set.

# Per-AC matrices

## `polylogue-1xc.14.1` — derive archive-scale workload profiles

The visible branch is exactly landed as #2934, but the owning Bead remains in progress. The matrix evaluates current master, not whether the old branch should merge.

| AC | Required outcome | Current source/routes and tests | Status | Falsification / missing proof |
| ---: | --- | --- | --- | --- |
| 1 | deterministic, versioned, privacy-classified `WorkloadProfile`; provenance; staging does not mutate committed packages | `polylogue/schemas/generation/workload_profiles.py`, `archive_workload_profile.py`, provider package/generation modules, `promotion_audit.py`; tests in `test_schema_workload_profiles.py`, `test_schema_promotion_audit.py`, schema generation/laws | **Partially satisfied; source-supported.** Core artifacts/provenance/staging landed. Live reviewed promotion remains open. | Demonstrate staged live generation leaves committed packages byte-identical until explicit promotion; then show reviewed artifact replacement. |
| 2 | bounded streaming rates/distributions/joints/relationships/lineage/growing/convergence/mix/sizes/selectivity; memory independent of corpus size | bounded sketch modules and `ObservationJournal`; #2968 bounds transactions/selective replay; memory probe and journal tests exist | **Partial.** Data structures and small scaling machinery exist; production-scale replay economics and residual retention sites are unresolved in child `.1`. | A committed representative `EXPLAIN QUERY PLAN`, 1x/10x RSS receipt, journal bytes, and cleanup across success/cancel/abrupt death; source audit must find no corpus-proportional list/set. |
| 3 | joint profile consumption; deterministic provider-native wire; real acquire→parse→materialize→index→query; parser/pushdown mutation fails | `polylogue/schemas/synthetic/*`, provider wire builders, scenario/workload modules, schema and source parser tests | **Partial.** Deterministic wire generation and some real ingest/convergence wiring landed; correlated joint variant selection and query/pushdown sensitivity are not closed. | Remove a production parser and selective query pushdown in reversible mutations; the same profile canary must fail for the intended reason. |
| 4 | named scale tiers preserving tails/selectivity; C-03 mixed archive + exact-session action query; related canaries | `polylogue/schemas/workload_tiers.py` and scenario structures exist | **Not demonstrated as the complete AC.** Presence of a tier module is not a C-03 production-route receipt. Bead notes keep named tiers/canaries open. | Retain a machine-readable named-tier definition and show C-03 ranking/selectivity failure when either ranking leg loses its bound; add tool/lineage/growing/partial-convergence canaries from the same profile. |
| 5 | shared `polylogue-1xc.14` receipts with identities, timings, resources, cancel/progress/cleanup; no parallel envelope | `polylogue/scenarios/workload.py`, `tests/unit/scenarios/test_workload_receipts.py`, infra workload artifacts | **Partial.** Shared receipt types/routes exist, but all performance lanes have not been shown to consume them. | Inventory every active perf/scale route; fail any route that invents a separate corpus identity/resource envelope. Retain cancellation/cleanup receipts from real routes. |
| 6 | promotion review for structural/distribution/privacy changes; automatic hard rejects and operator-review inventory | `polylogue/schemas/promotion_audit.py` and tests landed | **Partial.** Audit mechanism exists. Child `.2` proves current committed provider artifacts still require live regeneration/review for content-shaped property keys. | Stage live Claude Code artifacts; hard-fail seeded secrets/content-shaped keys; separately enumerate review-only values; promote only reviewed replacements. |
| 7 | supersede vague perf family; focused inference/generator/route/privacy/determinism/memory/receipt tests and quick gate | broad focused tests added by #2934; PR/Bead reports targeted and quick results | **Partial; not rerun.** Existing tests do not prove all residual ACs or full-scale closure. | Current master must remove/redirect parallel perf scenarios, run the complete targeted set, and attach exact quick-gate receipt after `.1`–`.3` completion. |

**Branch decision:** retire `d0cf780128`; do not equate exact landing of its patch with closure of the seven-AC Bead.

### Children that control residual closure

| Bead | Open residual | Current evidence | Required next proof |
| --- | --- | --- | --- |
| `polylogue-1xc.14.1.1` | full-corpus replay/memory, safe journal lifecycle, exact mergeable accumulators | #2968 commits bounded transactions and selective replay; Bead notes describe a 41.7 GiB WAL/live replay incident and repair | full production-scale receipt, selective query plan, cancellation/abrupt cleanup, no proportional retention |
| `polylogue-1xc.14.1.2` | prevent observed content becoming property names; scan/promote reviewed current packages | classifier/audit design and tests are in progress | live Claude Code staging, blocker/review inventory, reviewed replacement of committed schemas |
| `polylogue-1xc.14.1.3` | separate latest/recommended/default/evidence family semantics | Bead records 55 live families and rare-latest default failure | retain all families, deterministic coverage-first recommendation rationale, live regenerated catalog and resolution mutations |

## `polylogue-06zm.1` — receiver-authoritative CaptureJob identity

The stale checkpoint branch is not the implementation under review for closure; current master #2953 is.

| AC | Required outcome | Current production routes/tests | Status | Residual / falsification |
| ---: | --- | --- | --- | --- |
| 1 | receiver create/get/list/adopt/update with stable id/scope/intent/revision/checkpoint/lease/retry/hold/receipts/client policy | `polylogue/browser_capture/capture_jobs.py`, `route_contracts.py`, `server.py`; extension `backfill/capture_jobs.js`, coordinator/storage/background; Python and JS CaptureJob tests | **Satisfied in current source; PR-reported verification, not rerun.** | Remove CAS/revision enforcement or receipt idempotence; production-route fixtures must fail. |
| 2 | whole-profile wipe and new extension id discovers/adopts only exact compatible scope, with no credential/cross-account leak/replay | receiver scope model and extension recovery/adoption routes in #2953 | **Satisfied in current source; not rerun.** The old branch's `paired:<provider>` fallback is weaker and must not return. | Use two accounts under one provider and a wiped profile; only exact account scope may be discoverable/adoptable. |
| 3 | concurrent adoption, expired leases, incompatible clients, duplicate reconnects, checkpoint conflict are visible/idempotent; removing CAS/lease breaks fixture | receiver CAS/lease operations and Python/JS tests | **Satisfied in current source; not rerun.** | Reversible mutation of lease expiry/CAS should produce duplicate ownership or acknowledged-page replay and fail. |
| 4 | IndexedDB/chrome.storage are caches and rehydrate from receiver | extension storage/coordinator/background routes and tests | **Satisfied in current source; not rerun.** | Delete both caches between acknowledged pages and recover solely from receiver state. |
| 5 | old mirrored checkpoints migrate or surface as typed orphans; focused/quick gates | typed legacy orphan paths and tests merged in #2953; close reason reports receiver 7, daemon-auth 19, extension 313, lint/manifest, quick 16/16 | **Satisfied by source plus recorded verification; not independently reproduced.** | A malformed or cross-scope legacy checkpoint must become a typed orphan rather than be silently adopted. |

**Branch decision:** retire `8ecc34ecce`. **Parent decision:** keep `polylogue-06zm` open; event/timeline projection (`.2`) and lifecycle quota/retention/migration (`.3`) are separate residual work.

## `polylogue-hjpx.1` — immutable raw replay-plan conservation

The branch patch is exactly landed as #2961. The matrix evaluates current master, including later #2965/#2966 corrections.

| AC | Required outcome | Current source/routes/tests | Status | Adversarial note |
| ---: | --- | --- | --- | --- |
| 1 | moved-path singleton is fully censused before immutable multi-raw plan; preview/apply same id/inputs | `polylogue/storage/raw_authority.py`, `revision_backfill.py`, `repair.py`; raw ledger/backfill/repair tests | **Source-supported satisfied; not rerun.** | A pre-census plan or post-id widening must fail the moved-path fixture. |
| 2 | executable conservation algebra over every before plan id | durable authority ledger/migration and plan state accounting; `test_raw_authority_ledger.py` | **Source-supported satisfied; not rerun.** | Drop carried-forward/unselected inventory or duplicate terminalization; digest/algebra must fail. |
| 3 | logical keys, witnesses, raw ids, source/index preconditions, exact application/membership receipts | raw authority plan/receipt records and repair integration | **Source-supported satisfied.** | `parsed_at_ms`-only execution evidence must not satisfy postflight. |
| 4 | interrupted census resumes without duplicate/partial plans; incomplete census cannot mutate | ledger and restart paths/tests | **Source-supported satisfied.** | Kill/restart between census and plan publication; no partial component may apply. |
| 5 | rejected-stale writes durable blocker before event; automation halts until explicit repair | source-tier authority ledger, readiness/repair/daemon projection | **Source-supported satisfied.** | Suppress durable blocker while emitting event; readiness must fail closed. |
| 6 | two consecutive matching quiescent dry-run digests | fixed-point fields and current runner/ledger tests | **Source-supported satisfied at mechanism level.** | One empty pass or candidate count alone cannot mark fixed point. |
| 7 | bounded CLI/MCP/daemon inventory plus query handles | maintenance CLI, daemon CLI, MCP resources, readiness; #2965 adds complete frontier counts | **Satisfied only in current master after follow-up #2965.** | Omit `proven_current` or derive postflight blocking from preflight counts; contract tests must fail. |
| 8 | mutations to closure/slicing/carry-forward/application receipt/second census fail | focused raw-authority tests recorded in merged change | **Source-supported; not rerun.** | Keep exact mutation list in the targeted gate; a test that only reads constants is insufficient. |

**Adversarial correction:** the #2961 close reason is not sufficient evidence for current AC7 by itself. #2965 exists because complete frontier state counts were missing. Current master repairs that gap; the stale branch does not.

## `polylogue-lkrc` — one proof-driven raw authority reconciler

The unification branch is exactly landed as #2962, but the P0 Bead remains open.

| AC | Required outcome | Current source/routes/tests | Status | Residual / falsification |
| ---: | --- | --- | --- | --- |
| 1 | one typed census over origin mismatch, duplicate, quarantine, superseded, missing/replaced bytes, competing authority | `polylogue/storage/raw_reconciler.py`, `raw_authority.py`, `repair.py`; browser/duplicate/quarantine tests | **Source-supported satisfied for implemented state space.** | Add a known frontier case that falls through to an untyped/healthy state; census must fail. |
| 2 | one plan-authorize-apply-receipt-postflight lifecycle; no incident-specific parallel proof lifecycle | #2962 removes old incident routes and routes actuators through shared reconciler; CLI/readiness/daemon use shared contract | **Source-supported satisfied.** | Global consumer trace must find no separate browser-origin/duplicate repair digest or mutator. |
| 3 | equivalent/rekeyable/duplicate converge idempotently/restartably with CAS | shared planner/apply paths and focused tests | **Source-supported satisfied.** | Replay the same plan after restart and mutate a witness between plan/apply; stale write must be rejected. |
| 4 | conflicting content cannot auto-win; durable judgment blocker; assertion resumes same plan | current conflict disposition/judgment handling and browser origin repair tests | **Source-supported satisfied.** | Remove assertion or change competing head after assertion; plan must remain blocked/revalidate. |
| 5 | missing bytes create durable reacquisition obligation; promote only after origin/identity/hash proof; receiver artifact retained | shared reconciler typed state and repair routes | **Partially source-supported.** Mechanism exists; full current fixture/live evidence is not reproduced here. | Delete/replace bytes while retaining receiver source; promotion without exact hash/origin proof must fail and source must remain recoverable. |
| 6 | all known fixtures use one reconciler; stopped-daemon live postflight has zero unreported gaps | known fixture suites were merged/reported; current Bead explicitly says live stopped-daemon postflight remains | **Partial.** Fixture convergence is source-supported; live postflight is unresolved. | Run on reviewed stopped-daemon copy after proof/backup gates; retain complete pre/post census and zero unreported frontier states. |
| 7 | readiness/status expose complete state counts/remediation; sidecar/index alone cannot report healthy | `archive_readiness.py`, daemon status/CLI; #2965 adds `frontier_state_counts` including `proven_current` | **Satisfied in current master; not rerun.** | Omit a state or use preflight residuals for postflight blocking; contract must fail. |
| 8 | `OriginSpec` authority rules and `yla8` replay-order protections; mutations fail if bypassed | current authority/replay code and related tests preserve mechanisms; live authorized `yla8` gate is separate | **Partial.** Static integration is present; complete mutation/live gate evidence is absent from snapshot. | Bypass OriginSpec or replay ordering in a reversible mutation; then run separately authorized live gate only with verified backup/quiescence. |

**Old branch decision:** do not merge `adda2b96ac`. Its useful residual is a compatibility question, not its incident-specific architecture. Build a current source-schema-v7 fixture through `raw_reconciler.py` and public repair/readiness surfaces.

## `polylogue-hjpx.2` — July-15 raw replay convergence proof

| AC | Required outcome | Current mechanism/evidence | Status | Exact missing evidence |
| ---: | --- | --- | --- | --- |
| 1 | sanitized corpus matches candidate/component/byte/skew distributions and is private-free | runner creates deterministic synthetic components/raws and content-addresses corpus; Bead notes record a 10,163/15,264 scheduler-shape execution | **Not satisfied at full requested shape.** | Receipt matching ~21,398 expanded candidates, 10,163 components, 4.788 GiB payload, skew/conflict cohorts, with documented tolerances/privacy scan. |
| 2 | each bounded pass within RSS/PSS/swap/temp/write/wall envelopes; daemon health responsive | #2970 records process resources and rejects contended hosts | **Partial mechanism only.** No successful full-shape receipt; runner does not itself prove live daemon responsiveness in the snapshot. | Admitted-host full run with declared envelopes, health probe latency, and retained containment receipt. |
| 3 | backlog monotonically decreases modulo retry injection; finite retries resolve once; no starvation | bounded pass loop drains candidates on small/default shapes | **Partial.** No full-skew fairness receipt and no explicit retry-injection proof in retained artifact. | Per-pass backlog/state ledger at full shape, deterministic retry injection, bounded fairness/wait evidence. |
| 4 | cursor/source/index/FTS/raw authority never regress across interruption/resume | real repair route and ledger censuses are used | **Partial.** Snapshot lacks interruption/resume invariant artifact at required scale. | Kill/restart points with before/after invariant digests and FTS/readiness state. |
| 5 | two final equal quiescent digests, zero executable plans, identical residual debt | runner enforces two quiescent census digests; unit test exercises 3 components/5 raws | **Satisfied only for small mechanism, not Bead scale.** | Same invariant in successful July-15-shaped receipt. |
| 6 | removing fair rotation causes starvation; removing conservation causes mismatch; harness fails both | no such mutation variants were found in runner/test source | **Not satisfied.** | Two reversible mutation executions with exact expected failures and retained results. |
| 7 | exact commands, containment receipts, seed, result artifacts durable/reviewable | command catalog/docs and JSON receipt path exist; content-addressed seed/corpus id exists | **Partial.** No successful full result package in snapshot. | Commit or otherwise durably reference command line, environment/admission, seed, corpus digest, per-pass data, resource receipt, result SHA. |
| 8 | no live archive apply; `yla8` owns separate gate | runner constructs a generated workdir/archive and Bead notes explicitly reject live apply | **Satisfied.** | Keep this boundary; any live apply would invalidate this Bead's safety contract. |

The tests are not vacuous: they reach `ArchiveStore`, source raw writes, and `repair_raw_materialization`. They are insufficient because their scale and mutation coverage do not match the AC, not because they are mere mocks.

## `polylogue-pf1` — generated storage-twin divergence inventory

| AC clause | Required | Current master evidence | Status | Why the close claim fails |
| --- | --- | --- | --- | --- |
| classify every write-path method | AST + SQL-string extraction of method, SQL, tables written, lane classification | narrative `docs/plans/STORAGE_TWINS_DIVERGENCES.md`; hand-written `DOCUMENTED_DIVERGENCES` in test | **Contradicted.** | No extractor, source-derived inventory, SQL/table association, or proof that every write method is included. |
| zero unexplained divergences | every same-table/intent semantic difference fixed or rationale row | ten narrative rows derived from existing docstring | **Not demonstrated.** | The universe being classified is not generated, so “zero unexplained” is circular. |
| test regenerates diff and fails on new divergence | same generator produces committed artifact and test oracle; mutation/new route fails | test counts ten constants and checks strings/files; companion test says SQL analysis is a placeholder | **Contradicted / false-green.** | Adding a new write method or changing SQL can pass without touching any asserted string. |

**Observed contradiction:** the Bead close reason says the test “regenerates the diff” and reports nine passes. Source shows no regeneration function. Passing tests do not cure the missing oracle.

**Decision:** reopen `polylogue-pf1`; delete both stale test approaches after extracting only their rationale text as seed material.

## `polylogue-48h` — consolidate SQLite introspection helpers

| AC clause | Required | Branch evidence | Status | Falsification |
| --- | --- | --- | --- | --- |
| before/after ownership map | complete old helper definitions, call semantics, owner and final API mapping | no committed map; only broad mass edits/new module | **Missing.** | Static census finds 226 private-helper calls still present after definitions were removed. |
| public behavior preserved through parity tests | sync/async/schema/view/virtual/error semantics tested before and after | no new tests in 41-file branch | **Contradicted by source and compile failure.** | 22 files do not parse; old error/view behavior differs from replacement defaults. |
| old path deleted/redirected with compatibility notes | call sites use new API or aliases; notes where public/internal contract changes | old definitions removed, calls remain; no aliases/notes | **Contradicted.** | `_table_exists` 186 calls/0 defs; `_column_exists` 34/0; `_index_exists` 6/0. |
| no evidence semantics change without migration/release note | same error and object-existence meaning | replacement propagates errors and changes view inclusion; identifier/schema semantics differ | **Contradicted.** | `operations/archive_debt.py` previously returned `False` on `sqlite3.Error` and treated views as existing. |
| layering/import graph, parity, public-model suite | reviewable verification artifacts | generated topology changed, but no parity/public suite and code cannot parse | **Missing/invalid.** | A generated topology file cannot validate an unparsable tree. |

### Compile failures observed on `48h`

`python -m compileall -q polylogue` failed in:

`polylogue/cli/commands/status.py`, `tutorial.py`, `daemon/convergence_stages.py`, `embedding_backlog.py`, `fts_automerge.py`, `metrics.py`, `hooks/__init__.py`, `insights/readiness.py`, `operations/archive_debt.py`, `sources/live/hook_paste_enrichment.py`, `storage/blob_integrity.py`, `blob_repair.py`, `embeddings/materialization.py`, `embeddings/preflight.py`, `embeddings/status_payload.py`, `fts/fts_lifecycle.py`, `fts/session_repair.py`, `raw_retention.py`, `session_replacement.py`, `sqlite/archive_tiers/archive.py`, `sqlite/archive_tiers/write.py`, and `usage.py`.

The branch commit's “mypy --strict” claim is unsupported by the branch tree: Python cannot parse these files. This does not prove which command the author attempted; it proves the recorded verification cannot apply to the visible final tree.

## Recovery and archival refs

### `refs/remotes/bundle/ap7`

**Observed.** This is a recovery/assimilation ref, not an open worktree or PR. Current Bead `polylogue-ap7` is an open semantic-renderer invariant epic. Current master already contains selectively adapted core via #2700 and #2736. Bead/history notes reject the proof-only packet's self-corrupting snapshot as independent evidence and keep residual normalized-family/Origin coverage and bounded lineage parity open.

**Decision.** Do not merge or use recovery packet tests to close current ACs. Continue from current registry/document routes and current `polylogue-ap7.1` ownership.

### `refs/remotes/recovered/demo-packet-v2`

**Observed.** This is also archival recovery history. Current-relevant schema/validator/receipt behavior was selectively ported via #2704 and #2709, while the stale flagship subsystem was rejected.

**Decision.** Do not merge. Any missing current demo invariant must be reproduced against current `devtools/demo_packet.py`, CLI/tour routes, and current fixtures rather than resurrecting the old packet tree.

## Missing generated surfaces and invented parallel mechanisms

| Area | Finding | Decision |
| --- | --- | --- |
| topology docs | stale branches contain generated topology conflicts, but generated files do not confer correctness | regenerate once on the final fresh implementation after code/tests parse and targeted checks pass |
| `pf1` diff artifact | committed narrative is not generated from current source | one source-derived function must emit both test comparison and stable artifact; never maintain two hand-authored lists |
| Testsuite Diet workload baseline | assumes not-yet-realized C-03/tiers/correlated generation/receipts | block consumers or refresh to available subset; do not add a Diet-specific profile/receipt/canary model |
| raw scale proof | runner exists; successful full result surface absent | retain exact full-shape JSON receipt and command/environment metadata |
| `48h` grep lint | Bead design suggests a source-spelling tripwire; Diet adjudication rejects it | use semantic parity, import ownership/layering, compile/type checks; do not require/forbid private helper spellings |
| old `lkrc` incident tools | old branch contains independent repair/receipt lifecycle | transpose only compatibility fixture into current unified reconciler; no parallel command or plan schema |

## Evidence limitations

- The snapshot is self-contained but time-bounded; no network verification was performed.
- PR bodies and Bead close reasons can report commands, but this audit did not rerun them.
- Static comparison cannot prove all runtime invariants, resource envelopes, browser behavior, or live archive safety.
- `git merge-tree` predicts textual merge conflicts, not semantic compatibility; merge-clean refs were still rejected where AC evidence failed.
- Recovery namespaces do not expose whether an external operator still informally calls them “active”; repository ancestry/worktree/PR evidence classifies them as archival here.
