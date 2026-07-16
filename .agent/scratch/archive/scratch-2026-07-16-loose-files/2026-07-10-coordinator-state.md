---
created: 2026-07-10
purpose: Ephemeral cross-repo coordinator state for Polylogue-first, Sinex-secondary execution
status: active
project: polylogue
---

# Coordinator State

This is an ephemeral anti-drift dashboard. Beads owns durable tasks, decisions,
dependencies, and closure evidence. Update this file when lanes or priorities
change; do not turn it into another tracker.

## 2026-07-11 convergence update

- Interrupted work was reconciled and merged linearly through PRs #2679-#2686.
  Master is currently `3fe7837c8`; all feature worktrees and merged feature
  branches have been removed.
- Production writer containment remains in force: `polylogued.service` and the
  SQLite backup timer are stopped and runtime-masked while the archive is
  rebuilt. The stale packaged runtime caused the 2026-07-10 lock/replay incident
  and must not be restarted.
- The divergent home v30 and realm v24 indexes were backed up, hashed, and
  quarantined under
  `/realm/staging/polylogue-sqlite/recovery/20260710T225846Z`. The configured
  home index alias and `/realm/db/polylogue/index.db` now resolve to one
  canonical active inode. The active v30 index remains preserved until v32
  promotion succeeds.
- A current official `rebuildable_cache_exclude` backup was created and restore
  verified at
  `/realm/staging/polylogue-sqlite/migration-backup/polylogue-archive-20260711T003705Z`:
  source/user/embeddings plus 21,629 referenced blobs (45.17 GB), zero missing
  blob references. Source migrated additively from v4 to v7 with `quick_check`
  passing.
- PR #2684 (`6a579d090`) replaced force replay with durable revision evidence,
  bundle membership census, typed byte/semantic frontiers, application receipts,
  fixed-point selection, and canonical FTS verification. PR #2685
  (`a2bbd25d6`) added owned inactive index generations, exact readiness gating,
  canonical alias-safe atomic promotion, source-delta refusal, WAL fencing,
  rollback retention, and promotion recovery.
- First production v32 generation was safely aborted before promotion: the
  original census retained every parsed payload and grew from 2.4 to 6.9 GB RSS
  by only 1,452/18,013 memberships. This falsified the implementation's scale
  readiness while leaving the active v30 index untouched.
- PR #2686 (`3fe7837c8`) is the dogfood-derived repair: parsed session trees spill
  to temporary SQLite and replay retains one authority cohort at a time. Its
  anti-vacuity fixture proves peak retained count follows maximum cohort size,
  not total raw count. The restarted real v32 generation is now progressing
  with bounded RSS; do not start another archive mutation concurrently.
- Remaining sequence: finish and atomically promote the exact-ready v32
  generation; update/deploy the Sinnix Polylogue package; unmask/start only the
  daemon; prove CLI/daemon/MCP/direct identity agreement and root-session
  no-shrink after a subsequent catch-up; then close P0s and merge the final
  Beads/release bookkeeping.

## Objective

Rapidly improve and dogfood Polylogue as the evidence, perception, and bounded
coordination substrate for frontier agents while continuing correctness-first
Sinex progress. Test the possibility that the broader Polylogue roadmap is
misframed; archive/evidence integrity and falsifiable user value outrank feature
count.

## Priority order

1. Restore truthful, fresh Polylogue production ingestion and preserve evidence.
2. Integrate recent Fable/Codex/GPT Pro findings into evidence-graded notes.
3. Keep one local Polylogue implementation lane active; integrate verified cloud
   work rather than merely launching it.
4. Make Polylogue correct, usable, externally presentable, and demonstrable;
   treat the Web UI rewrite as structural workflow work, not cosmetic polish.
5. Repair agent perception/control through existing `s7ae`, `1hj`, `37t.11`,
   `d1y`, `ptx`, and proof-gap/testing programs; no conductor-2.0.
6. Advance Sinex through isolated Sinex-context lanes, emphasizing derivation
   durability, tracker truth, prod truth, and test/proof coverage.
7. Run bounded comparative/behavioral falsification before expanding upper rings.

## Resource authority

- Exactly one local heavy archive/database mutation at a time; test/build work
  may overlap only when it uses isolated scratch state and memory/PSI evidence
  shows headroom.
- The source-v3 -> index-v30 replay is complete: 17,814/17,814 raw rows were
  attempted, 6,243 changed sessions were materialized, and the explicit
  `session_insights` repair restored 73,998 rows. Exact readiness is 9/9 with
  17,156 sessions, 4,455,734 messages, zero blocked/actionable debt, and all
  five SQLite tiers returning an asserted `ok` integrity result.
- Current archive writer: transient `polylogued-v30-runtime.service`, sourced
  from detached worktree `/realm/tmp/worktrees/polylogue-runtime-current` at
  merged v30 master `f6b396bf6`. The packaged v29 `polylogued.service` remains
  stopped and runtime-masked. Startup held about 134 MiB; host evidence at
  17:37 showed 17 GiB available and zero memory PSI.
- Cloud sandboxes may execute synthetic/disjoint work concurrently.
- New ChatGPT Pro conversations: global maximum one per five minutes; after any
  rate-limit response, one per fifteen minutes. Prefer replies in extant chats.

## Active lanes

### Polylogue runtime recovery

- Stale packaged v29 `polylogued.service`: stopped and runtime-masked.
- Current v30 transient runtime: `polylogued-v30-runtime.service`, PID 1242633
  at startup, with watcher, API, and browser-capture components all ready.
- `/healthz/live` and `/healthz/ready` pass; FTS reports 2,672,652 indexed and
  indexable messages, zero missing/excess/duplicates, and trusted freshness.
  Browser capture returns its expected authenticated 401 on an anonymous probe.
- `polylogue status --json` resolves through the daemon and reports daemon API,
  watcher, browser capture, and search ready. Keep the package unit masked until
  Sinnix deploys v30; do not expose source-v3/index-v30 to installed v29 code.

### Polylogue implementation

- Authority correction is merged: Polylogue PR #2637 and Sinex PR #2465 now
  restore the intended integrated-mode boundary. Sinex owns exact raw and
  normalized materials, revision/observation history, provenance, and
  lifecycle; Polylogue owns conversation ontology and rebuildable read models,
  while standalone Polylogue remains supported. `polylogue-303r` and
  `sinex-4j2.1.1/.1.2` carry the executable program; the contradictory
  metadata-only mirror items were superseded.
- Source-backed Hermes adjudication is merged in Polylogue PR #2638. It exposed
  a false completion: current Hermes schema v16 lacks two columns the parser
  required, and automatic watcher/catch-up paths parsed a mutable live SQLite
  file instead of the bytes they retained. `polylogue-fs1.1` is reopened and
  claimed as the active P1 trust-floor implementation.
- Two isolated local lanes are active: `polylogue-hermes-parser` owns
  schema-capability/fidelity, inactive/compacted/observed history, cost
  provenance, and profile-qualified identity; `polylogue-hermes-acquisition`
  owns consistent SQLite backup, retained-blob parsing, and WAL/SHM watcher
  admission. Integration already found one cross-lane hazard: profile identity
  must derive from the original Hermes root, never the blob-store path.
- Local Sol: `codex-polylogue-d1y.service`, session
  `codex-session:019f4aad-3ad3-7470-902e-564dd02047ce`.
- Worktree/branch: `/realm/tmp/worktrees/polylogue-d1y`,
  `feature/feat/hooks-install-liveness`.
- Scope: full `polylogue-d1y`; source work only until heavy permit releases.
- Logs: `/realm/tmp/agent-runs/polylogue-d1y/`.
- Source work is committed at `9b6a02e95`. Its quota-burning rebuild poll was
  stopped after positive scope attestation; verification resumes as a separate
  bounded job after the permit releases.
- Resume fallback was repaired, its initially false fixture caught and made
  parser-shaped, and PR #2635 merged (`6 passed`; quick 13/13).
- Reasoning projection leakage found by PolyDrop was repaired and PR #2634
  merged (`26 passed`; quick 13/13).
- `v6vy`: 294 focused tests and quick 13/13 pass; PR #2636 is open pending CI.
- Fable campaign Beads surgery merged as PR #2633. The first implementation
  slice is the action-spined delegation attempt read operation under `y964`;
  do not retire the v28 view until a later batched index bump.
- Cloud `kj22`: fuzz discovery worked and exposed failures; seeded run was
  intentionally interrupted at 46% to release the heavy permit for production
  recovery. Do not merge a red harness change.
- Cloud storage scenario task `task_e_6a50935bcfa48320a747a87d6e57d4a1`
  was rejected: it built a parallel toy archive whose tests proved its own
  reimplementations. Corrected task
  `task_e_6a509765684c8320a8f2f67ea206ccd5` is ready and invokes five exact
  production pytest node IDs with missing-node/zero-collection failure. It is
  evidence worth salvaging, but its 235-line parallel registry/runner should
  not be applied unchanged: `hjwr` remains the stronger storage authority and
  `9e5.19` should be narrowed to a thin real-node adapter or superseded.
- Cloud `3v1.1` task `task_e_6a5095a424c88320b22cad8a0260bd61`
  returned a useful partial diff but is not integration-ready: it adds instance
  attribution and atomic temp cleanup, yet does not carry attribution through
  raw acquisition or prove hash dedup/final archive state, and its freshness
  test covers only favorable old-then-new arrival while writes remain
  unconditional. Local branch `feature/fix/browser-instance-attribution` now
  has commits `7ebcf3b7c` and `b495d8caf`: one parsed-evidence replacement
  policy drives spool and canonical ingest, declined ACKs distinguish requester
  from retained artifact, and the integration proof follows concurrent writes
  through raw acquisition and stable index/hash. Focused tests await the permit.

### PolyDrop export demo

- Reproducible generator: `/realm/inbox/polydrop/regenerate.py`; operator guide:
  `/realm/inbox/polydrop/README.md`.
- Latest authoritative private refresh completed at 2026-07-10T15:37:27Z for
  July 9-10 Europe/Warsaw: 145 raw CLI sessions (72 Claude including 60
  subagents, 73 Codex), six supplemental browser captures, and 151 successfully
  rendered sessions in full/dialogue/user-only forms. Parse/render failures are
  0/0; output is 575 MiB across 610 files.
- The manifest explicitly reports the remaining evidence gap: no distinct
  cloud-agent transcript source is provable in the live archive date slice.
  Browser chatlogs are present as provider-native captures, not inferred from
  local CLI activity.

### Durable-tier backup cleanup

- `/realm/inbox/polylogue-backups/pre-blob-cleanup-20260706` was removed after
  evidence review. It consumed 67 GiB, was source schema v2, had 36 missing
  referenced blobs, and was both older and less complete than the healthy
  source-v3/user-v4 durable tiers. The three small backup/audit manifests were
  retained; the directory is now 28 KiB.

### Integration / research

- Polylogue session synthesis:
  `.agent/scratch/2026-07-10-recent-agent-session-integration.md`.
- Sinex session synthesis:
  `/realm/project/sinex/.agent/scratch/024-recent-agent-session-integration-2026-07-10.md`.
- GPT Pro result synthesis: complete at
  `.agent/scratch/2026-07-10-gpt-pro-results-integration.md`. Complete outputs:
  Polylogue strategy amendment, Sinex semantic fingerprints, and Sinex coverage
  obligations. Kernel/replay/axes/Beads-surgery chats remain in progress; two
  report-shaped interim drafts are quarantined for nonexistent/wrong-project
  paths.
- Dogfood ledger:
  `.agent/scratch/2026-07-10-agent-control-dogfood-ledger.md`.
- Cross-project control/resource ledger:
  `~/.claude/scratch/2026-07-10-agent-control-capability-ledger.md`.
- Strategy/dialogue anchors:
  `.agent/scratch/2026-07-10-broad-project-strategy-and-verifiability.md` and
  `.agent/scratch/2026-07-10-agent-dialogue.md`.

### Cross-project execution control

- Sinnix program `sinnix-056` is committed/pushed at `e0660edf`; first slice
  source commit is `a4fdef4c` on `feature/feat/attested-agent-jobs`.
- Source review reopened `sinnix-056.1`: collected systemd scopes made `list`
  abort, and abnormal exits could leave false `running` manifests. Repairs and
  anti-vacuity coverage are uncommitted in the same worktree pending the
  verification permit.
- Scope guard: Beads stays task authority; the new substrate owns only execution
  handles, resource policy, lifecycle, and receipts.

### Web UI / external usability

- Read-only audit is complete at
  `.agent/scratch/2026-07-10-webui-rewrite-and-proof-audit.md`.
- Verified structural failures: the primary split-archive reader bypasses the
  executor claimed by closed `0hqs`; token-auth deployments serve a shell that
  never sends credentials; several routes materialize whole sessions/archives;
  and `tests/visual` never launches a browser or executes JavaScript.
- Preserve the ratified ASGI + `RouteSpec` + TS/Preact program. Correct existing
  Beads/ACs, stabilize direct reads and auth first, then ship one Playwright +
  a11y + responsive vertical reader slice. Do not create a parallel Web epic.

### Sinex leadership

- Every execution worker must start in a Sinex worktree and load Sinex
  `CLAUDE.md`; Polylogue coordinator context is not a substitute.
- Integrated closed Fable/Sol findings identify immediate truth work:
  reconcile merged `v7od`, re-review PR #2464, classify the 2gk cloud canary,
  and perform coherent replay/convergence/identity Bead surgery before broad
  derivation-kernel work.
- No local Sinex heavy execution while the Polylogue rebuild owns the permit.
- Active cloud integration: Sinex `2gk` is task
  `task_e_6a50857bf3ec8320a7cbebcc1f07e241` in
  `/realm/tmp/worktrees/sinex-2gk-cloud`. The previously remembered
  `task_e_6a50858a55a083208c3452983003812e` is Polylogue `v6vy`, not Sinex.
  Audit the recovered diff against current Sinex master/AC before another
  disjoint Sinex launch.
- Xtask cloud audit is committed at `4781feeb0` in
  `/realm/tmp/worktrees/sinex-xtask-truthful-diagnostics`. It rejected ancient
  cloud preimages and ports truthful `--bg --wait`, typed timeouts, fail-closed
  queued work, watchdog cleanup, and bounded stderr diagnostics. The worker
  nevertheless ran Sinex checks while the Polylogue rebuild owned the heavy
  permit; prompts are therefore not an adequate host-wide sequencing control.

## Control-plane findings to preserve

- `get_resume_brief` can hide an existing-but-empty profile and emit a UUID as
  the next action.
- Blackboard persists but does not deliver; `1hj`/`s7ae.3` own the delivery gap.
- Coordination payload is noisy/large and retains dead conductor handoff refs.
- Reverse posting works in no-submit mode but duplicate tabs are nondeterministic;
  attachments/model attestation still require CDP.
- Extension can capture all supported tabs without focus, but existing tabs did
  not auto-capture after pairing and the canary receiver split capture from prod.
- Process liveness is not writer readiness. Status must foreground tier
  compatibility, watcher admission, last successful ingest, and retry cost.
- PID/title/session-age are not sufficient agent identity. A near-interruption
  of this coordinator proved that destructive control needs positive binding of
  logical session, exec/terminal handle, cgroup, repo/worktree, and role.
- The actual Codex resource boundary is an inner `sinnix-agent-*.scope`, not the
  nearly-empty launcher service. Runtime caps now bound `agent.slice` at
  10/14 GiB and the local `d1y` scope at 3/5 GiB.
- A worker can still violate the coordinator's heavy permit because it is only
  prose. The next `sinnix-056` slice must acquire a host-wide build/test/rebuild
  lease at command launch and return a truthful queued/refused receipt.

## Next actions

1. Reconstruct source-v4 publication receipts on current master, excluding the
   already-landed Hermes Beads commit; run focused/quick/broad proof, review,
   publish, and perform a backup-gated live source-v4 rollout.
2. Close `w9wt` and `qkmd` with the merged broad receipt; keep `8jg9.2` open
   until source-v4 deployment and migration evidence are complete.
3. Verify the v30 watcher ingests this active session and keep the transient
   runtime pinned until the packaged v30 deployment replaces it.
4. Selectively port browser-instance identity after source-v4; do not import its
   superseded precedence policy or invalid attachment query.
5. Continue the test/verifiability program through real boundary scenarios and
   proof-gap enumeration; prioritize the Web UI Playwright vertical slice over
   more unit-count expansion.

## Stop / drop order

- First drop speculative surface cleanup and new Web scope.
- Then drop proof-compiler generalization beyond mandatory authority cells.
- Never drop production archive recovery, evidence integrity, the comparative
  baseline, or truthful negative results.
- Freeze any coordination slice that requires task authority, a new queue/state
  machine, stale/cross-scope delivery, or more than two implementation person-days
  before a paired behavior test.

## 2026-07-10 Hermes reproducibility closure

- PR #2639 now closes the real `fs1.1` surface rather than the earlier
  parser-only claim: consistent SQLite/WAL snapshots, retained-blob parsing,
  profile-scoped raw identity plus blob deduplication, v16/later structural
  contracts, durable active/observed/rewound/compacted state, exact cost
  provenance, and provider-positive compression composition.
- Five cold adversarial iterations were necessary. They successively found
  byte-identical profile collisions, empty-row loss, retained-artifact
  inspection drift, v17 support drift, parser-only cost proof, arbitrary event
  loss at the writer, zero-token cost loss, truncated compression children,
  late-parent projection pruning, provider-ref loss, and weak durable-read
  coverage. Iteration five found no legitimate gap.
- The reusable architecture correction is index v30: `session_events` is the
  lossless open normalized event stream; policy and usage relations are typed
  projections. Provider-local source refs survive independently while the
  canonical ref resolves/remaps to the physical owner of shared-prefix content.
- The final focused composition is 37 passing tests; quick verification is
  13/13. Fresh seed-testmon produced 13,201 pass / 12 fail / 1 skip. Eleven
  failures reproduce on clean pre-PR master; the twelfth passes exactly on both
  branch and baseline and is suite-order/shared-state pollution.
- Clipboard technical adjudication created `polylogue-fs1.12`, a compact Hermes
  evidence-and-continuity demo: tool run -> consistent snapshot -> fidelity
  report -> bounded recall -> exact delivery manifest -> claim/tool-evidence
  comparison. It is deliberately distinct from `fs1.6`'s air-gapped memory loop.

## Immediate continuation

1. Merge PR #2639 after final-head CI/review, close `fs1.1`, then rebuild the
   live derived index once at v30; do not repeatedly reset during adjacent work.
2. Create one baseline-restoration cluster for the eleven inherited failures
   and separately instrument the browser coalescing shared-state pollution.
3. Advance `fs1.3` fidelity declarations, then use `fs1.12` as the forcing
   integration of `fs1.11` recall and `fs1.4` forensics.

## 2026-07-10 source-v4 publication checkpoint

- The reviewed source-v4 branch is reconstructed on current master at
  `/realm/tmp/worktrees/polylogue-source-v4-current`, excluding the already
  merged Hermes Beads commit. It adds per-publication durable receipts,
  reserve-before-visible ordering, atomic receipt consumption with references,
  archive-root ownership, bounded destructive GC rechecks, writer exclusion,
  backup/restore inventory parity, and real process-pool interleavings.
- A final cold review found two more legitimate gaps: backup verification could
  omit a source-referenced blob and still self-certify from its copied-file
  inventory, and zero-byte inline attachments were treated as absent. Both now
  have mutation-sensitive real-boundary tests; the reviewer reran and passed.
- First seed-testmon broad run: 13,303 pass / 17 fail / 1 skip. It exposed one
  inherited #2644 integration regression plus three weak harness contracts:
  root hooks violated the kernel topology and snapshots were stale; parser
  roundtrips bypassed archive publication; live tests spied on a retired plain
  store call; convergence benchmarks statistically repeated one destructive
  state transition and asserted on the final no-op result.
- PR #2658 separately moved hooks to `polylogue/hooks/__init__.py`, tightened
  root topology owners to kernel-only, and refreshed exactly four help
  snapshots. Source-v4 now routes roundtrip laws through `ArchiveStore`, proves
  receipt consumption, observes private blob preparation, and uses one-shot
  scale probes across single/multi-provider benchmarks.
- Final receipts at head `a45998e0b`: integration cluster 35/35; quick 13/13
  (`20260710T161442Z-quick-1281529-d1d6c238`); seed-testmon 13,321 pass / 1
  skip / 0 fail in 250.09s
  (`20260710T161512Z-seed-testmon-1282104-018e6e2a`).
- `8jg9.5` now tracks a remaining proof-substrate gap: migration validation
  checks manifest presence/tier but not a content-bound successful verification
  receipt. Live rollout must still use an actually verified backup and preserve
  the command receipt; the new Bead prevents mistaking that operator discipline
  for an enforced invariant.

## 2026-07-10 v30 rebuild preflight

- The live derived index is v29, 25.4 GB plus a 160 MB WAL. Source/user tiers
  are healthy and FTS is logically complete; embeddings are independently
  partial (97.5%, 430 pending) and must not be mistaken for an index rebuild
  blocker.
- `polylogued-agent.service` runs from the detached
  `/realm/tmp/worktrees/polylogue-runtime` at master `4dabc85dd`, with its own
  continuous writer and 2.3 GB current / 6 GB peak cgroup footprint. The
  service must be stopped before replacing the index, and the runtime worktree
  must first advance to the merged v30 code so status/schema authority agrees.
- Host memory pressure is currently zero with 14 GB available, but a live
  `sinnix-build-*` Sinex test scope owns the shared heavy lane. Do not start the
  25 GB rebuild until that scope exits. This is transient sequencing evidence,
  not a reason to weaken permanent build or archive policy.

## 2026-07-10 inherited-suite ownership reconciliation

- Do not create a new baseline-restoration bead. `polylogue-w9wt` already owns
  all eleven inherited failures plus the browser coalescing order leak; enrich
  and retitle it around deterministic full-suite restoration.
- Execute `polylogue-qkmd` in the same branch: line-number keyed interpolated-
  SQL allowlists are the direct cause of two failures and must become content-
  stable keys rather than receive another line-number refresh.
- Reconcile `polylogue-8jg9.2` with `polylogue-v7e0`: its lease-row/sweeper AC
  became invalid when leases were deliberately removed in favor of the age
  gate. Preserve the concurrency proof, not the retired mechanism.
- Treat the browser-capture node as concrete shared-state leakage: it passes
  exactly on both branch and clean baseline but fails in broad order. Find the
  polluter and prove an ordered pair; do not quarantine or retry it.
- Review status snapshot changes explicitly, and repair stale expectations
  only after checking the production contracts introduced by `gnie`, `v7e0`,
  `9e5.24`, and `37t.15`.

## 2026-07-10 Hermes merge and live v30 transition

- PR #2639 merged at `9e92b6b6d`; PR #2640 closed `fs1.1`, upgraded `w9wt`,
  linked `qkmd`, and replaced `8jg9.2`'s removed lease premise with a
  falsifiable age-gate race proof. The baseline implementation cluster is now
  claimed on `feature/test/restore-deterministic-baseline`.
- The first v30 replay failed safely before row 1 because the coordinator
  stopped only transient `polylogued-agent.service`; installed
  `polylogued.service` remained enabled, restarted with packaged v29 code, and
  recreated a 1.7 MB v29 index. The stale installed unit is now stopped,
  disabled, and runtime-masked; its Home Manager activation will restore it on
  the next system deployment.
- Runtime worktree `/realm/tmp/worktrees/polylogue-runtime` is detached at
  merged v30. Canonical replay is running under
  `sinnix-background-1783682525566165079-991906.scope` with 4 workers and
  20-row batches: 17,788 raw rows / 47.16 GB planned. Early envelope: about
  2.1 GB cgroup memory, zero memory PSI, 3-10 raw rows/s depending on message
  volume. Restart a v30 transient daemon only after replay/convergence proof.

## 2026-07-10 blob publication race proof

- `8jg9.2` falsified the source-v3 lease-free GC premise on the real acquisition
  path. `iter_source_raw_stream` drains 128 items before its first yield and
  acquisition holds raw/reference writes in a bulk transaction, flushing only
  every 500 records. The first published blob can therefore remain invisible
  to `raw_sessions`/`blob_refs` for longer than `MIN_AGE_S` while later source
  artifacts stream.
- A deterministic provider-shaped harness advances a frozen clock by 61 seconds
  during the second input, runs real GC before raw persistence, and observes one
  unlink. Acquisition then commits a durable reference to the missing blob.
  Runtime is 3.21 seconds; no sleep, large corpus, or synthetic GC shortcut is
  involved.
- The exact fix is a separate source-v4 durability slice: reserve a hash in the
  durable source tier before final blob-path visibility; convert the reservation
  to a raw/blob reference atomically; make GC/doctor serialize reservation/ref
  recheck with unlink; retain crash debt without TTL expiry. BlobStore remains
  substrate-neutral through an injected publication hook. Reshape acquisition
  so pre-publication reservation is not blocked by an already-open bulk write.
- This must not be folded into the w9wt baseline PR or the in-progress live v30
  rebuild. The live replay reads existing retained bytes and remains safe from
  this acquire-versus-GC race because no GC or concurrent writer is running.

## 2026-07-10 publish and integration checkpoint

- The deterministic-baseline cluster is backed up as draft PR #2641 from
  `feature/test/restore-deterministic-baseline` at `e9add9c54`. Original cluster:
  13 passed in 20.94 seconds; strengthened browser/direct/security/SQL cluster:
  25 passed in 66.04 seconds; pre-push quick gate: all 13 steps green in 15.9
  seconds (`20260710T122745Z-quick-1061419-a0b3a390`). Keep the PR draft until
  `devtools verify --seed-testmon --skip-slow` runs after the replay releases
  the heavy-I/O lane.
- Five independent adversarial rounds all found real gaps. Every demonstrated
  reproduction is repaired, but the cap was reached without a no-gaps verdict;
  neither the PR nor Beads should claim reviewer convergence.
- The source-v4 reservation repair is parked clean on
  `feature/test/measure-blob-age-gate` as `374a1d0c8`, `38a54cce5`,
  `a28a10bbf`, and `e3df68640`. Its daemon startup ordering proof passes 6/6.
  Do not rebase or publish it until #2641 lands; afterward explicitly reconcile
  the source 3->4 status/snapshot and CLI skip-output changes, require a verified
  source.db backup manifest, and only then start any v4 daemon.
- The live v30 replay remained healthy at 9,864/17,788 sessions and 3,327,716
  messages with a 2.146 GiB cgroup peak and schema v30. Installed v29 daemon
  remains disabled and runtime-masked.
- `/realm/inbox/polydrop` was absent when rechecked, so the earlier requested
  private session export had not actually been delivered. A dedicated lane now
  owns a regenerable raw plus Polylogue-projected export with an explicit source
  coverage manifest. `/realm/inbox/polylogue-backups` is only 28 KiB but contains
  pre-destructive blob cleanup manifests; retain it as audit evidence.

## 2026-07-10 coordination dogfood and workspace reconciliation

- A real MCP/CLI call falsified s7ae.1's shipped "compact, bounded,
  agent-grade" projection claim without falsifying the envelope architecture.
  CLI status was 20,671 bytes at the ordinary bound and reported 10 peers plus
  10 resources: earlyoom, launcher/host/MCP/spare plumbing, systemd-timesyncd,
  resolved, udevd, oomd, UVM threads, dbus, and below. It missed the actual
  named v30 rebuild scope and emitted three nonexistent retired-conductor
  handoff paths. Correct fields included w9wt, dirty checkout, schema v30,
  daemon absence, and current Codex lineage.
- `polylogue-s7ae.7` now owns the repair under the existing program: <=8 KiB
  compact status plus opt-in detail, logical actor collapse, cgroup-aware
  resource evidence, live handoff sources, omission counts, and mutation tests.
  PR #2642 carries the Bead-only delta and passed the local 13-step quick gate.
- Worktree reconciliation removed 13 clean merged/superseded worktrees and 19
  verified-redundant local branches. Retained lanes are #2641, source-v4 GC,
  browser-instance attribution, hook liveness, dirty fuzz discovery, runtime,
  and the browser/qkmd source branches until #2641 merges.

## 2026-07-10 source-v4 independent review blockers

- The parked blob-publication-reservation branch is not publishable. A
  same-hash two-publisher interleaving defeats a hash-only reservation primary
  key/delete: one publisher consumes the shared row, its later ref is pruned,
  GC deletes while the other publisher remains in flight, and that publisher
  commits a durable ref to missing bytes.
- Existing browser-daemon and pure-source law tests fail because reservation
  reconciliation/construction assumes source schema v4 at APIs that do not own
  an archive migration boundary.
- Backup includes referenced hashes but not reservation-only bytes, producing a
  restored reservation whose blob is gone.
- Reconciliation can clear a committed reservation before a paused publisher
  makes the final path visible; daemon pidfile exclusivity does not prove that
  CLI/source writers are absent. One maintenance repair writer also bypasses
  reservations entirely. The design needs publication multiplicity,
  archive-owned capability injection, complete writer enumeration,
  reservation-aware backup, and genuine writer/exclusive-generation proof.

## 2026-07-10 polydrop delivered and independently verified

- `/realm/inbox/polydrop` now contains 132 primary raw CLI sessions (72 Claude,
  including 60 subagents, plus 60 Codex), six supplemental recently acquired
  ChatGPT captures, and 138 normalized sessions in each of full, authored
  dialogue, and user-only views. Metadata has 138 parseable JSONL rows.
- `regenerate.sh` defaults to yesterday+today in Europe/Warsaw and delegates to
  the private `regenerate.py`; both scripts are mode 0700, data is 0600, and
  directories are 0700. Manifest accounting excludes itself: 557 files,
  525,228,973 bytes, zero parse/render failures.
- Independent SHA-256 checks matched source, copied raw file, and manifest for a
  representative Claude session, Codex session, and ChatGPT browser capture.
  Cloud-agent transcript coverage remains explicitly unproven because the
  archive exposes no distinct cloud transcript source for this window.

## 2026-07-10 v30 replay lock failure and controlled retry

- The first replay failed at 10,362/17,788 raw rows with `database is locked`.
  Live evidence found both `polylogued-agent.service` and the installed
  `polylogued.service` writing/restarting against the archive. Runtime masking
  had not stopped an already-loaded restart loop; the maintenance command held
  no archive-wide exclusion capability and did not reject the competitors.
- Both writers are now stopped, the installed unit remains runtime-masked, the
  backup timer is paused, and only the browser-post canary remains. A full
  idempotent retry runs as `polylogue-index-rebuild-v30-retry.service` with log
  `/realm/tmp/polylogue-v30-rebuild-retry.log`, MemoryHigh=6 GiB and
  MemoryMax=10 GiB. Do not start any daemon until it completes and parity/read
  checks pass.
- `--only-missing` is not an exact crash resume: it selected 6,946 historical
  raw revisions (17.9 GB) because superseded raw ids no longer own the current
  session row. The full retry is safer; already completed rows hash-skip.
- `polylogue-b5l.1` now owns writer exclusion plus commit-aligned raw cursor and
  delta replay. `polylogue-rii.4` owns first-class Codex Cloud/Claude Code Web
  transcript intake proven missing by polydrop. PR #2643 carries both Beads.

## 2026-07-10 browser-instance lane adjudication

- Preserve but do not merge `feature/fix/browser-instance-attribution`.
  Valuable, still-unique work: stable extension-profile instance IDs,
  capture provenance/state fields, honest `persisted=false` plus retained
  `artifact_instance_id` acknowledgments, and atomic temp cleanup.
- Drop its parallel capture-precedence implementation. PR #2641's structural
  native-payload evidence and ownership-before-freshness rule supersede it.
  The branch also queries nonexistent `attachments.session_id`; richness must
  use `attachment_refs.session_id`.
- A raw-attribution proof currently fails on missing retained blob evidence, so
  integration order is #2641, corrected source-v4 publication receipts, then a
  fresh selective port. Extend canonical precedence so equal-message arrivals
  cannot discard attachments, and prove simultaneous requester IDs, retained
  owner, one uncorrupted spool artifact, one indexed identity, and durable raw
  provenance. Decide explicitly whether losing-instance attribution must be
  durable server-side or may remain only in the extension receipt.

## 2026-07-10 demo shelf clean-checkout falsification

- Do not commit the four dirty canonical shelf projections as a generated-docs
  sync. They claim 185 total/148 readable files because generation included
  ignored or untracked demo artifacts. The same patch on a clean current-master
  worktree sees 69/59 and `devtools.demo_shelf --check` marks all four outputs
  stale, even though the eight unit tests pass.
- `polylogue-r3o3` owns repository-closure-aware generation: committed indexes
  use tracked/declared inputs; private artifacts go to an explicitly untracked
  projection; dirty and clean checkout generation must be byte-identical with
  omission accounting and mutation tests.

## 2026-07-10 source-v4 receipt redesign checkpoint

- The rejected hash-only reservation implementation was rewritten as four clean
  commits on `feature/test/measure-blob-age-gate`: per-publication UUID receipts,
  archive-owned reserve-many/publish-many, exact source and post-index
  consumption, pure source/parser defaults, receipt-aware repair/backup/GC, and
  inspect plus confirmed abandon operations.
- Audit-focused receipt is 106 passed in 91.37s; strict mypy is green for 13
  touched modules; `8jg9.2` carries all ten findings and remains in progress.
  No rebase/push/PR/live access occurred. An independent static re-review is in
  progress before the branch is allowed to consume another managed gate.

## 2026-07-10 hook liveness integrated

- PR #2644 merged the hook installation/liveness implementation at
  `7e6123cac`; PR #2645 closed `polylogue-d1y` at `0f8580d39`. The lane updated
  current Claude/Codex event catalogs, topology/CLI publication, typed provider
  compatibility, and honest unknown/unconfigured/no-eligible-session metrics.
- Receipts: 82 focused tests, six absence-semantics tests, two full 13-step
  quick gates, clean Beads graph lint, and applicable CI green. Worktree and
  merged branches are removed. Residual is post-deployment real-harness
  dogfood, not unfinished implementation.

## 2026-07-10 accounting and Nous/Hermes integration

- `polylogue-f2qv.2` is complete: implementation PR #2647 merged as
  `8c9dfbb00`, closure PR #2648 as `77554289b`. The parser-to-writer-to-event/
  model-row-to-pricing proof keeps Codex cached/uncached lanes disjoint, retains
  reasoning as event evidence, and never re-adds reasoning to inclusive output.
  Focused groups were 76/76 and 33/33, controls 58/58 and 14/14, with two green
  quick gates. The live copied five-session ratios never overcounted.
- The report the operator meant was recovered from
  `polylogue_nous_followup_2026-07-10-v2.zip`, not the three Fable markdowns.
  PR #2649 merged the four-record Bead delta as `3f0e786e0`: refinements to
  `fs1.4`, `fs1.7`, and `fs1.11`, plus new `fs1.13` for longitudinal,
  evidence-backed memory/skill revision evaluation. Integrated-mode evidence
  authority remains governed by `303r`/Sinex.

## 2026-07-10 replay amplification re-opened

- The v30 retry crossed the prior failure point and reached 17,631/17,814, but
  cold replay exposed the still-open half of closed `3wb`: historical snapshots
  of one logical session repeatedly perform full replacement. Measured examples
  include 408-422 MiB Codex rows parsed for about 12 seconds only to report
  `changed=0`, and Claude full-replace DELETE work growing through 15, 85, 103,
  and 349 seconds per revision.
- A privileged `py-spy` stack proved the apparent pause was not deadlock: the
  asyncio main loop waits while `asyncio_0` is active inside
  `_replace_full_session_messages_and_blocks`; all four parse workers are idle.
  Files, log, and resources therefore reflect one long SQLite writer operation.
- `polylogue-3wb` is reopened with proof-carrying prefix-coverage design and
  anti-vacuity AC, related to `hjwr`, `1xc.8`, and `20d.15`. The optimization
  must preserve exhaustive replay as reference and may cover only
  cryptographically proven prefix-monotone rows; append/bundle/divergent/
  truncated/unsupported evidence stays exhaustive.

## 2026-07-10 source-v4 second rereview checkpoint

- The second static audit found four more production gaps: ingest-batch inline
  attachments bypassed receipts; dedup maintenance skipped receipts for an
  already-existing target; backup/CLI used a global blob root while writers used
  the archive root; and GC bounded affected deletions but not inspected rows
  under `BEGIN IMMEDIATE`. It also lacked a positive reconciliation-exclusion
  anti-vacuity case.
- The source branch now has separate commits repairing all four and tests for
  the real ingest route, existing-hash race, referenced-heavy bounded lock,
  exclusion clearing, and archive-root override. Ruff and strict mypy are green;
  managed focused tests are running sequentially under the live resource gate.

## 2026-07-10 late-session closure checkpoint

- Demo-shelf repository closure is complete. PR #2651 merged the production
  fix as `c6aa6a05d2a`, PR #2652 closed `polylogue-r3o3` as `ab30ac3645`, and
  both temporary worktrees/branches were removed. Committed projections now
  use Git-tracked inputs, private artifacts require a private output, and clean
  clone bytes match.
- Source-v4 is clean and unpublished at `2832f5a37`. The five cold-audit gaps
  were repaired in separate commits; a final archive/browser acquisition sweep
  passed 10/10 and the exact browser authority node passed 1/1. `8jg9.2` holds
  the complete receipt plus the inherited demo-policy failure classification.
  A second independent production-diff review is active before publication.
- Replay remains exclusive with packaged writers and the backup timer disabled.
  It completed batch 1040 at 17,670/17,814 after four measured full replacements
  (60.59s, 438.51s, 89.12s, 89.61s) and entered batch 1041. CPU time continues
  increasing with zero host memory PSI, so this is slow replacement work rather
  than a deadlock. Do not restore writers until replay exit plus direct archive
  integrity/readiness and the deterministic baseline gate.
- Draft PR #2641 remains clean and conflict-free against current master, with
  CI green but the test job intentionally skipped. Its broad local gate waits
  for the replay resource exclusion; a fresh independent cold review is active.

## 2026-07-10 replay completion and readiness gate

- Exclusive v30 replay exited success with 17,814/17,814 raw rows, 6,243 parsed
  and materialized sessions, 1,048 batches, zero failures, and 5,589.4 seconds
  elapsed. The direct archive reports source v3, index v30, embeddings v1,
  user v4, ops v1, 17,156 sessions, 4,455,734 messages, and 17,814 raw rows.
- Exact readiness correctly refused a green result: only 4/9 surfaces were
  ready. Session profiles, work events, phases, threads, and latency profiles
  had missing/stale materialization, plus 814 non-critical raw-materialization
  debt rows. `rebuild-index` nevertheless printed success, so a read-only agent
  is tracing the misleading completion/test contract and will enrich or create
  the owning Bead rather than hiding the postcondition.
- Runtime-v30 `ops maintenance run --target session_insights` is executing as
  isolated unit `polylogue-v30-session-insights.service`, operation
  `e306b8ea-c145-4906-bb6b-31d838f875b1`. Writers and backup timer remain off.
  After it exits: repeat exact readiness, run the baseline broad seeded gate,
  run coordination live dogfood, then restore the packaged daemon only after
  all archive gates are honest.

## 2026-07-10 baseline and source-v4 adversarial closure

- Cold review of draft PR #2641 found two P1 false greens and one P2: direct
  hash-skip retained stale same-size attachment bytes; it skipped FTS repair;
  and the token scanner missed the daemon's actual `_send_json`/`_send_error`
  sinks. Commits `63a663973` and `9edb87062` repair all three. Focused 51/51,
  quick 13/13, push and CI are green; the broad seeded gate still waits for the
  live archive workload to finish.
- Source-v4 independent rereview found no remaining P0-P2 production issue.
  The one residual static-only process-pool path now has a real-route
  reservation/final-path/ref-commit/GC interleaving test (`5c04aec87`, 1/1 in
  5.58s) and Bead evidence commit `1c8d018b3`. Branch remains unpublished and
  will be reconstructed on current master after baseline merge, excluding the
  already-merged Hermes Bead commit from its old history.

## 2026-07-10 Sinex-Polylogue report capture

- Operator clipboard was captured before delegation at
  `/realm/tmp/clipboard-sinex-polylogue-20260710T162317.md`, 97,223 bytes,
  SHA-256 `dbb056f5aaaa065357f32e02e6ca8c71af978238af72113aee4de62f29d30a4c`.
- A cross-repo agent is reconciling the report against both live codebases and
  Beads graphs. The fixed authority decision is: integrated mode uses Sinex as
  durable raw/normalized evidence, lifecycle, and model-effect substrate;
  Polylogue owns AI-work ontology, interpretation, offline/local projections,
  standalone behavior, and product UX. Concurrent operator-authorized Beads
  and design-file edits are intentional and must be preserved/reconciled.

## 2026-07-10 source-v4 live rollout complete

- Every archive writer was stopped before the durable boundary. The packaged
  `polylogued.service` remains runtime-masked because the installed package is
  still source-v3/index-v29; the browser canary on 8876 writes only to its
  private `/realm/tmp` spool.
- A verified backup completed at
  `/realm/inbox/polylogue-backups/polylogue-archive-20260710T162633Z` under
  transient unit `polylogue-source-v4-backup.service`, invocation
  `c3a1967697a046dbb01d55a568e09678`. Scratch restore and exact SQLite
  integrity passed for source/user/embeddings; 21,457 blobs totaling
  44,748,652,907 bytes matched the inventory. The operator receipt SHA-256 is
  `e4770e283302d779206282249789727fdd84c7189f0779cae4315a402cf3f480`.
- The current migration runner cannot enforce that receipt; `8jg9.5` owns the
  content-bound receipt gate. With the archive still quiescent, transient unit
  `polylogue-source-v4-migrate.service` applied exactly source v3 -> v4 from
  merged PR #2660 in 1.49s, exit 0, 83.5 MiB peak, zero swap.
- Independent post-migration checks: source `PRAGMA integrity_check=ok`; tier
  versions source/index/embeddings/user/ops = 4/30/1/4/1; the reservations
  table and hash index match canonical DDL; outstanding publication receipts
  = 0; convergence debt = 0.
- Live authority is now `polylogued-current-runtime.service`, invocation
  `276fa7bbef7345ccb16ddb3db529cbf4`, running merged commit
  `a0ef2fa8d479a0168db36fe09f3752de6311b26e` from the retained detached
  worktree `/realm/tmp/worktrees/polylogue-runtime-v4`. API 8766 and capture
  8765 are ready. Readiness reports schema match, eight sources, zero WAL,
  2,672,652/2,672,652 FTS rows, and 100% lexical coverage. Startup catch-up
  scanned 14,760 paths, needed one 41.1 MiB Codex file, used the append route,
  and finished one success/zero failures with about 0.01x initial read
  amplification. Memory peaked near 1.2 GiB and host memory PSI remained zero.
- Dogfood proved this session (`019f49d8-0185-7c43-8793-db6e57db13e1`) is in
  the index with 7,023 messages and a matching source row with passed
  validation. Subsequent append snapshots still expose `parsed_at_ms=NULL`,
  the already-filed `kwlu` bookkeeping defect, without preventing index
  refresh.
- Archive-debt's two actionable rows are not migration/storage failures: they
  are 430 embedding-pending sessions plus two historical catch-up failures.
  Retrieval remains ready at 97.5% session coverage. Treat this as embedding
  work, not a reason to roll back source v4.
- The source-v4 feature worktree and two obsolete detached runtime worktrees
  were removed after PR #2660 was verified merged. Keep the v4 runtime worktree
  while the transient service uses it. `s8q` owns replacing transient commit
  pinning with trustworthy installed-package deployment/attestation.
- `/realm/inbox/polydrop` was refreshed again after rollout through merged v4
  code: 151 raw CLI sessions (72 Claude including 60 subagents, 79 Codex), six
  browser captures, 157 Polylogue-rendered sessions, zero parse/render failures,
  634 files and about 597 MiB. The run completed in 49.83s with 1007.9 MiB peak.
  `regenerate.sh` now separates the code root from interpreter discovery, so a
  detached worktree can use the canonical virtualenv instead of failing on a
  nonexistent per-worktree `.venv`.

## 2026-07-10 post-rollout integration

- Sinnix PR #1 merged as `431fc11e6903e3daef6ac6bdc559a329f94acec2`,
  pinning the Polylogue flake input to source-v4 merge `a0ef2fa8d`. The exact
  package built successfully at
  `/nix/store/3ywdr3ql81wr2fjpbanm1j8na7w5asdl-python3.13-polylogue-0.1.0`;
  both binaries report `0.1.0+a0ef2fa` and constants source/index 4/30. The
  live `switch` is intentionally waiting for the scheduled realm Borg job to
  leave its high-IO start phase. `sinnix-4e2` owns the unrelated hermetic test
  input omission; Polylogue `6rvt` owns full-SHA runtime attestation.
- Embedding truth PR #2661 merged as
  `69990dcc873c2fc0a9c900861bb10db94b75f434`. Approximate `sqlite_stat1`
  candidate counts can no longer produce impossible message coverage, and
  bounded archive debt preserves unknown pending-message counts instead of
  printing zero. Evidence: 48 focused tests, quick 13/13 twice, all substantive
  CI/security checks green. `b2r9` is closed.
- Live embedding audit also found 11,348 metadata rows for messages absent from
  the rebuilt index, six status rows for absent sessions, and two intentionally
  terminal HTTP-400 rows that remain permanently critical but are not
  inspectable by identity. `1dk1` owns generation-aware orphan reconciliation;
  `egm8` owns terminal failure inspect/acknowledge/supersede lifecycle.
- `8jg9.2` is closed with the full migration/live receipt. `8jg9.5` remains the
  enforced content-bound backup-receipt gate. Coordination dogfood's false
  self-PID is `8k91`; the 13.5s latency sample is appended to `s7ae.8`.
- Beads graph lint remains green. The canonical JSONL is intentionally staged
  with the other agent's authority/design rewrite plus these live DB changes;
  do not split, overwrite, or commit that shared file without reconciling the
  concurrent agent's branch/intent.
- Web verifiability audit is captured at
  `.agent/scratch/2026-07-10-webui-verifiability-audit.md`. The key diagnosis is
  zero browser-engine coverage: existing `tests/visual` is valuable but
  explicitly browserless. Sequence `1ilk` Playwright characterization against
  the current UI before v2, then an enumerable <=40-cell proof compiler, auth/
  request-order/failure/bounded-read blockers, renderer structure, and only
  then the first `bby.11` vertical using the same journeys.
- Raw parse-state PR #2663 merged as `938c671e5`: live append/full routes now
  retain raw evidence once, index every derived session, then finalize one
  typed raw-level success; sync and async writers share the SQLite compiler;
  crash-after-index retry is idempotent and duplicate blob publication is gone.
  Verification: live protocol 7/7, raw state 49/49, ArchiveStore 30/30, quick
  13/13, substantive CI green.
- Independent cold review after #2663 caught two real gaps missed by those
  tests: full-route result accounting was published after session 1, so a
  session-2 failure could still advance cursor eligibility, and raw failure
  warnings were plain text in a `_json` column, making envelope readback raise
  `JSONDecodeError`. Follow-up PR #2664 merged as
  `0cccef1df3b8618c6e6dbe3e3a4b6860ab8cebc3`; per-record results now publish
  only after all sessions plus the success marker, and warnings serialize as
  JSON arrays. Focused 8/8 + 49/49, quick 13/13, green CI, independent PASS.
- `kwlu` remains open only for a sanitized live catch-up proof after the
  packaged final commit is deployed. Sinnix is being repinned to `0cccef1df`;
  the current transient source-v4 daemon remains healthy meanwhile.

## 2026-07-10 final runtime catch-up proof

- Sinnix PR #2 merged as `ab75743bb9df9a97fdf5a0051a2945dab12e760f`,
  pinning final Polylogue `0cccef1df3b8618c6e6dbe3e3a4b6860ab8cebc3`.
  Exact package `/nix/store/xy53wnwypy013x203489bp57j5ips4ff-python3.13-polylogue-0.1.0`
  reports `0.1.0+0cccef1`, source/index 4/30, NAR
  `sha256-EZ6sUa9ORtvW+cRv3IFZv+UPOnWZedvKtxDF6+kwqKY=`.
- The transient runtime was cut from old `a0ef2fa8` to final `0cccef1df` under
  unit `polylogued-final-runtime.service`, invocation
  `eacf5185c4684d48b3b0902ac096f40a`. Startup catch-up scanned 14,765 paths,
  selected one 42.8 MiB Codex append, and completed 1 success/0 failures with
  0.000145x read amplification, 0.039s parse, and 0.076s convergence.
- A service-start cutoff isolated final-runtime evidence from two old-runtime
  NULL rows: acquired=1, parsed=1, failed=0, unclassified=0, exact index links=1.
  Raw `946c8b809b1bad9171d900b64b8726e54aa96d2c6bbfc7735e949996418deccc`
  is the exact current session `raw_id` (7,516 messages); cursor offset equals
  stat size with failure_count=0. Readiness is ready/actionable=0, receipts=0,
  convergence debt=0, FTS exact-count parity 2,672,652/2,672,652.
- `kwlu` is now closed. The old inactive runtime worktree was removed; keep
  `/realm/tmp/worktrees/polylogue-runtime-final` until packaged cutover. Nix
  activation remains queued behind the still-running high-IO realm Borg job.
- Migration-backup retention audit: `/realm/inbox` is included in realm
  btrbk/Borg (only `**/inbox/monero` is excluded). All 21,457 backup blobs also
  exist in the canonical Polylogue corpus but are distinct inodes, so Borg
  deduplicates stored chunks while still traversing/hashing the duplicate 46
  GiB tree. Keep it until archive `realm-realm.20260710T183000+0200` exits
  success and membership/extraction of receipt, source.db, inventory, and sample
  blobs is verified. Then request explicit deletion confirmation; do not move a
  second full copy to `/outer-realm`. The one-off Borg checkpoint has a certain
  seven-day rollback window; long-term recovery belongs to managed rolling
  backups and open restore-drill bead `4be`.

## 2026-07-10 live contention and backup checkpoint correction

- The final runtime later became live-but-unready while periodic raw
  materialization processed 23 rows / 121.3 MiB with message FTS triggers
  suspended. `py-spy` proved multiple in-process SQLite writers overlapped:
  raw materialization replaced session content while watcher and periodic
  convergence independently rewrote session insights. Live Codex append writes
  then exhausted the 30-second busy timeout repeatedly with `database is
  locked`; readiness remained `fts_not_fresh` and the index WAL reached 160
  MiB. `n2wy` now owns a deterministic concurrency harness and one explicit
  daemon write coordinator. This falsifies the weaker assumption that a sole
  writer process is sufficient without serializing its connections.
- The 18:30 btrbk snapshot contains the migration backup's `source.db`, blob
  inventory, manifest, and representative blob, but not
  `verification-receipt.json`: the receipt was created at 18:31:49. Therefore
  Borg archive `realm-realm.20260710T183000+0200` cannot satisfy the deletion
  gate even if the job succeeds. Keep the local backup until a later coherent
  snapshot/archive contains the receipt and payload proof together; do not
  infer snapshot completeness from the directory creation time.

## 2026-07-10 replay-authority incident and containment

- The active packaged archive is the split file set rooted at
  `/home/sinity/.local/share/polylogue/index.db` with durable tier symlinks into
  `/realm/db/polylogue`; `/realm/db/polylogue/index.db` is a separate obsolete
  writable index. Verifying the latter produced false confidence while the
  former was being truncated. P0 `polylogue-nkmy` now owns one typed archive
  identity and split-root startup/preflight rejection.
- One-shot acquisition and parse of only the current Codex raw
  `6a74735ec...` restored root session
  `019f49d8-0185-7c43-8793-db6e57db13e1` to 8,076 physical messages in the
  active index. FTS repair inserted 7,131 missing rows and readiness returned
  true. The staged recovery raw remains under the active archive inbox.
- `polylogued.service` remains enabled but inactive, with ports 8765/8766
  closed. Do not restart it until containment and the write coordinator are
  merged, packaged, and proven against the active file set.
- P0 child `polylogue-yla8.1` was corrected after adversarial review. An empty
  index is not revision authority: replaying several historical full snapshots
  can still end on the wrong one. The containment PR therefore keeps read-only
  plan/backlog inspection but blocks every source-to-index replay executor,
  removes ambient `--force-write`, reports structured blocked debt, and proves
  zero session/message/FTS/parser mutation. Parent `yla8` retains the typed
  per-session revision ledger, baseline/suffix ordering, and crash-resume work.
- Two cloud yla8 candidates and two cloud n2wy candidates were rejected. They
  sorted on non-authoritative metadata, built self-confirming toy archives, or
  introduced event-loop deadlocks. Cloud throughput is useful only behind
  real-route fixtures and independent review; generated volume is not accepted
  evidence.

## 2026-07-11 cleanup and recovery correction

- PR #2679 merged the structured Beads reconciliation as `e1b049193`. The
  canonical checkout was then restored from its evidence-backed recovery patch,
  fast-forwarded to `origin/master`, and left clean. The interrupted ChatGPT
  extraction is complete: 22 unique captures, 68 readable files, with corrected
  `INDEX.md` and `VALIDATION.md` under
  `.agent/scratch/gpt-fork-deliveries-2026-07-10/`.
- The stale packaged daemon restarted before containment and falsified the prior
  recovery snapshot. It ran source/index schema 3/29 code against live 4/30
  databases, accumulated 75 lock failures and 944 raw/index gaps, read 250.9
  GiB, and shrank the active root session again from 8,076 to 7,936 messages.
  All three active-index FTS surfaces are stale with triggers suspended. The
  daemon and weekly backup timer are now inactive and runtime-masked; their
  persistent enable symlinks were removed. Do not restart or run Sinnix switch
  without rechecking the mask.
- Live identity remains split: home index v30 has 17,224 sessions and the 7,936
  root; realm index v24 has 17,096 sessions and a 360-message root. The weekly
  backup job targeted only the obsolete realm index. `polylogue-nkmy` is being
  implemented as a write/startup identity preflight plus status projection; it
  must not claim blue-green promotion or live quarantine ACs without actual
  wiring and receipts.
- `polylogue-yla8` cannot honestly order existing v4 raw history: raw rows carry
  full/append kind but no predecessor, baseline revision, byte range, capture
  generation, or authority status. A source-v5 additive durable migration and
  acquisition-time revision envelope are prerequisites. Historical v4 cohorts
  must be classified from provable byte relations or quarantined, never ordered
  by timestamps, paths, provider time, or content-hash lexical order.
- Recovery after nkmy + yla8: online-backup home/realm indexes and durable tiers
  into `/realm/staging/polylogue-recovery/<stamp>`, quick-check/hash/count them,
  preserve both originals in quarantine, promote the validated home-v30 backup
  into `/realm/db/polylogue/index.db`, and replace the home index with a symlink
  to that same inode. Then update the Sinnix Polylogue flake pin, switch while
  masked, and run bounded daemon ticks with an immediate abort on root shrink,
  wrong baseline, lock error, identity conflict, or persistent FTS suspension.
  `b5l.1/.2` are not prerequisites for guarded in-place salvage, but are required
  before trusting a fresh destructive rebuild.
- Verified-redundant worktrees for #2675, n2wy/write-coordinator, qkmd,
  browser-coalescing, kj22, blob-age, and runtime-final were removed. The browser
  instance-attribution branch is rejected wholesale: it queries nonexistent
  `attachments.session_id`, regresses native-over-DOM precedence, and has a
  false-green concurrency proof. Preserve only its stable extension instance-ID
  and atomic-temp-file ideas in `polylogue-3v1.1` before deleting the branch.

## 2026-07-11 packaged replay rollout and two new P0 findings

- PRs #2680-#2689 landed archive identity containment, source-v7 revision
  evidence, one daemon write coordinator, offline v32 generation promotion,
  bounded component-aware replay, and exact actions-view readiness. The live
  archive now resolves both `/realm/db/polylogue/index.db` and the home alias to
  device/inode `3a:913945`. The promoted v32 root session
  `codex-session:019f49d8-0185-7c43-8793-db6e57db13e1` has 9,298 messages,
  4,270 tool-use blocks, and content hash
  `63B19AD16B7AFF4C01C3BC82A02473861E40737CC9B5FE1306BEAB8C9097EC7E`.
  Full verified backup:
  `/realm/staging/polylogue-sqlite/migration-backup/polylogue-archive-20260711T003705Z`.
- PR #2688 initially shipped ordinary replay but independent review found an
  archive-wide 45 GiB census on each daemon tick and incomplete resource
  guards. It was not deployed. Forward correction #2689 uses fixed-point
  authority components, aggregate component bounds, terminal DEFERRED receipts
  for incomparable adoption, and real-route anti-starvation/resource tests.
  The packaged runtime is Nix store path
  `/nix/store/4cw3icbjig2lq29ljaralw13vhqndk90-python3.13-polylogue-0.1.0`.
- The packaged daemon then proved root no-shrink across three bounded passes,
  but exposed a new infinite-work defect: passes of 200.7s, 190.8s, and 192.7s
  each reported 15 replayed logical sources while remaining candidates rose
  391 -> 393 -> 395 and quarantine rose 176 -> 178 -> 180. The candidate query
  recognizes only DEFERRED receipts; immutable selected/superseded/ambiguous
  receipts do not retire their exact raw IDs. The daemon was stopped before a
  fourth pass. P0 `polylogue-yla8.2` owns terminal-receipt exclusion and a
  two-call real-route fixed-point proof.
- Six `unknown-export` JSON decode failures are not malformed inputs. They are
  ChatGPT browser-capture envelopes staged through the inbox watcher. The live
  append guard checked watcher name `browser-capture`, so ordinary rewritten
  `.json` files under `inbox` were sliced at obsolete offsets and persisted as
  `source_index=-1` suffix blobs. For all six, the archived blob is exactly the
  current file suffix while the old archived full blob is not a prefix of the
  current file. Their ops cursors nevertheless record current EOF/hash with
  `failure_count=0`, suppressing full acquisition. The first three current
  captures have 105/120/117 turns while the index retains only 17/14/14.
  P0 `polylogue-yla8.3` owns append-safe allowlisting, cursor non-vacuity, six
  exact full-file reimports, and terminal classification of the bad suffix
  evidence. Do not restart production until both P0s are merged, packaged, and
  their production recovery receipts exist.

## 2026-07-11 terminal production convergence

- PRs #2690-#2695 completed terminal-receipt retirement, JSONL-only append
  planning, semantic-frontier preservation, governed bundle retirement, and
  readiness projection from the shared durable authority census. The final
  packaged replay backlog has zero executable candidates. Governed ambiguity
  remains visible as quarantine rather than being replayed or called absent.
- The exact quiesced archive scan is green on all 9 read surfaces: 17,519
  sessions, 4,419,819 messages, exact FTS parity, exact action/tool-use parity,
  and zero missing insight materializations. Receipt:
  `/realm/staging/polylogue-sqlite/recovery/20260710T225846Z/receipts/final-exact-readiness-green.json`.
- Installed CLI, direct SQLite, daemon HTTP, and MCP all resolve the protected
  root `codex-session:019f49d8-0185-7c43-8793-db6e57db13e1` at 9,298 messages.
  The home and realm index paths resolve to the same device/inode and promoted
  generation. HTTP readiness is `overall=ok` with exact FTS counts; the daemon
  is active with zero restarts. Cross-surface receipts live beside the exact
  scan as `final-daemon-root-session.json`, `final-mcp-root-session.json`, and
  `final-live-packaged-status.json`.
- #2696 durably closed the recovered P0 bead family, `polylogue-nkmy`, and
  `polylogue-b5l.2`. Release PR #2102 then merged and published `v0.2.0`.
  Master is clean at `2f220e9b1`; all product PRs are merged, no P0 or
  in-progress beads remain, and all Polylogue worktrees/feature branches were
  removed.
- Eleven append fragments created by continuing live capture remain explicitly
  pending byte-authority adjudication. They are non-executable, visible debt
  owned by the still-open suffix/killed-batch resume work; this is the honest
  residual rather than a hidden convergence claim.

## 2026-07-11 strict completion audit after release

- Git inventory: canonical checkout only, clean `master` at the `v0.2.0`
  release commit; no stash, local feature branch, open PR, P0 bead, or
  in-progress bead. `origin/gh-pages` is the intentional published-site branch.
  The unmerged automation branch
  `release-please--branches--master--components--polylogue--release-notes`
  contained only a stale July 5 pre-release note, had no PR, and was deleted
  after `CHANGELOG.md` plus the GitHub `v0.2.0` release became authoritative.
- Archived Fable disposition: original session
  `claude-code-session:3347cf34-ca12-45ae-918f-781c7f96a704` and fork
  `fa4df7c3-7fc7-449c-bbd0-b42aec839c40` produced Lane A (#2671), Lane B
  (#2673), Lane C (#2674), and Lane D (#2675). The failed ten-chat extraction
  was later completed as 22 unique captures / 68 readable files. Its 24-file
  browser canary spool was proven byte-identical to durable source/inbox
  escrow; the one unique dry-run post-command receipt was added to escrow.
  The 17-hour Fable supervisor, private browser, transient canary receiver, and
  redundant temporary spool were then stopped/removed.
- Recent Codex error-left sessions were reconciled by mechanism, not terminal
  label: `019f4f64-853d-7a30-a28d-c703bf8b8640` delivered #2691;
  `019f4f5f-ab06-70a1-a4ae-163d9e1969d8` diagnosed the six browser JSON
  suffix failures consumed by #2691/#2695; the authority/readiness reviewers
  either delivered #2690/#2692 or returned no-blocker adjudications. No unique
  uncommitted code remained in their checkouts.
- Interrupted Lane E had been proposed in Fable prose but never implemented.
  A late-created `/realm/worktrees/lane-api` was clean at master with zero
  commits/diff and no process or PR, so it was removed. The source-verified
  residual is now durable as `polylogue-bby.17`: privacy-safe public HTTP
  projections, a bounded cockpit overview aggregate, and a bounded structural
  session-evidence summary. This explicitly defers real product work instead
  of pretending the empty lane was integrated.
- Beads correction: `polylogue-bby.1` was implementation-complete in #2673
  but still open; it was closed with the merged fault/visual proof. Other open
  beads referenced by recent PRs were audited and retain explicit unmet scope
  or are umbrella programs. Example: `polylogue-5k5l` remains open because its
  successful live sandbox-byte acquisition AC is not yet proven, despite the
  expired-file negative path being proven.
