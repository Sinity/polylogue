---
created: 2026-07-11
purpose: Preserve the integrated GPT Pro/code-assimilation audit and production convergence evidence
status: active
project: polylogue
---

# Assimilation and production convergence

## Scope

This note continues `2026-07-10-coordinator-state.md` and
`2026-07-10-gpt-pro-results-integration.md`. It records what the recovered
"Branch / Project / Explanation / Relevance" work actually contributed, what
landed, what was rejected, and the live-production evidence that now gates
completion. Beads remains the durable task authority.

## Landed results

- PR #2708 (`a21b907`) authenticated durable-migration receipts. The live
  user tier migrated from v4 to v5 only after an independently keyed, verified
  backup was written under
  `/realm/staging/polylogue-sqlite/migration-backup/user-v5-a21b907-20260711T154133Z/`.
  Postflight was `integrity_check=ok`, one assertion, zero settings, zero
  deliveries, and both user-v5 indexes present. `polylogue-8jg9.5` is closed.
- Sinnix master `7fe17bd0` pins the installed Polylogue build to `a21b907`.
- PR #2709 (`885b46d`) is the one recovered code branch that was directly
  portable after repair. Demo Packet v2 now binds claims to receipt bytes and
  digests, enforces exact section/control/falsifier contracts, and contains
  six production-route false-green mutations. `polylogue-212.12` is closed.

## Recovered work disposition

The ten recovered branches were not ten merge candidates. Treating them that
way would import stale architectures and self-proving test worlds.

- The Demo Packet evidence-binding work was selectively ported and landed.
- Branch 20's useful context-delivery, bulk-SAVEPOINT, injection, payload/API,
  preview, and MCP-role ideas are now execution-grade design/AC in
  `polylogue-37t.12` and `polylogue-37t.22`. Its weaker migration, forgiving
  JSON reads, self-asserted authority, and stale root `judge` command were
  explicitly rejected.
- Recovered Incident, Cockpit, proof-packet, Markdown-only web, and static
  cockpit artifacts supplied constraints or counterexamples to
  `polylogue-212.11`, `polylogue-1ilk`, `polylogue-ap7`, `polylogue-37km`, and
  `polylogue-bby.11`; they were not code-complete against current architecture.
- The rejected synthetic archive test branch was a false-green exemplar: it
  reimplemented Polylogue behavior in a toy schema and proved the toy agreed
  with itself. Future delegated tests must name the production dependency
  exercised and the implementation mutation/removal that makes them fail.

## Append/CAS P0: corrected causal model

`polylogue-yla8.6` supersedes the earlier named-session diagnosis.

1. Deterministic full + append1 + append2 + append3 reproduction at
   `/realm/tmp/yla8-6-repro.2wOrUT` showed post-ingest legacy compaction deletes
   append1 after append2. Append3 is then correctly rejected by strict CAS as
   an older/disconnected frontier. Disabling compaction makes the same route
   pass; CAS is the guard, not the defect.
2. `_record_full_cursor` ignored the acquired raw byte size whenever
   `source_revision` existed and committed later `stat.st_size`. A live
   Claude/Sinex source accepted 748,295 bytes but recorded a 766,042-byte
   cursor while the file later reached 3,304,375 bytes.
3. Read-only production census found seven byte-proven append raws with
   missing predecessors, six current accepted heads, five excluded paths, and
   one separate cursor-ahead path. Repair must derive its cohort from queries,
   not a frozen list of three session IDs.
4. The implementation under review protects both `sessions.raw_id` and
   `raw_revision_heads.accepted_raw_id`, traverses and validates the complete
   source predecessor closure, fails destructive cleanup closed, and commits
   full cursors only through captured blob bytes. The two real live-batch
   regressions pass.
5. A read-only live run of the new protection helper failed closed in 374.7 ms
   on missing accepted raw
   `203af5524be8d1abc13e9f7d35fd9013ad8eb0ff3c4ce4c4be23c29a2f0df3ea`.
   `polylogued.service` was stopped at 16:09 local time because deployed
   `a21b907` still contains the unsafe compactor.

## Completion gate

Do not call production converged until all of the following are evidenced:

- Independent adversarial review converges on the append/CAS diff.
- Focused live-batch, retention, repair, and revision-authority tests pass;
  quick verification and Beads graph lint pass.
- The PR is squash-merged and Sinnix deploys that exact commit.
- With the daemon stopped, a verified source/user backup is taken and the
  dynamic broken-head/cursor-ahead cohort is captured.
- Only disposable cursor rows for the reviewed cohort are removed. Durable raw
  rows, blobs, heads, receipts, sessions, and user data remain untouched.
- Ordinary full reacquisition converges every path; postflight has no missing
  accepted predecessor, no cursor beyond accepted material, no exclusions or
  failures, source/index parity, and no bounded-journal CAS regressions.
- A controlled sanitized append advances exactly once.

## Strategic refinement

The useful Polylogue trajectory remains archive-first, with coordination and
execution affordances built only where the evidence model gives them a clean
home. Current dogfooding supports three near-term priorities:

1. Make workflow proof compositional and anti-vacuous: real-route journey
   registry, declared evidence sources, gap enumeration, and mutation checks.
2. Finish the `s7ae` coordination spine instead of introducing a parallel
   execution vocabulary. Observation, addressed delivery, ack/wakeup, and
   bounded payloads precede scheduling ambitions.
3. Treat production convergence as a typed, queryable proof surface. A green
   process is insufficient; accepted-head closure, cursor/material parity,
   and source/index parity should become standing readiness invariants.

The third item is the most immediate leverage discovered by this incident:
the code can be process-healthy while its evidence frontier is unreplayable.
The live integrity queries used for `yla8.6` should become a maintained health
projection after the P0 is closed, rather than remaining an operator-only
repair script.

## Service-backed execution dogfood

The retired conductor remains retired. Beads owns work/dependencies; these
units are disposable process containment and telemetry, not another scheduler
or packet vocabulary.

Each external Codex lane now has one explicit task handle, branch/worktree,
model/effort receipt, isolated `POLYLOGUE_ARCHIVE_ROOT`, isolated `TMPDIR`,
bounded cgroup, journal, and file outputs under
`/realm/tmp/polylogue-agent-runs/<task>/`. No lane can use the production
archive by inherited configuration, and workers are forbidden to mutate Beads
or merge their PRs. The coordinator owns claims, integration, live proofs, and
closure.

Active receipts at 2026-07-11 18:56 CEST:

- `polylogue-yla8-review.service`: `gpt-5.6-sol`, high, read-only adversarial
  review of immutable `bb3df3dde`, 4 GiB high / 7 GiB max. Output:
  `yla8-review/review.{log,last.md}`.
- `polylogue-xy95-terra.service`: `gpt-5.6-terra`, high, branch
  `feature/perf/provider-usage-stale-diagnostics`, 4 GiB high / 6 GiB max.
- `polylogue-iyew-luna.service`: `gpt-5.6-luna`, high, branch
  `feature/fix/workload-probe-boundary-table`, 3 GiB high / 4 GiB max.

Monitor with `systemctl --user show <unit> -p ActiveState -p SubState -p
MemoryCurrent -p MemoryPeak -p CPUUsageNSec -p TasksCurrent`, `journalctl
--user-unit <unit>`, and the per-task log/last-message files. Launch receipts
confirmed all three requested models; never trust the configured default.

Dogfood findings:

- Native collaboration workers share the coordinator's broad agent scope, so
  its cgroup memory can include tests and several child lanes. Per-task
  transient services give cleaner attribution and independent limits.
- Fresh worktrees must disable checkout hooks during creation, or Beads can
  import that branch's stale JSONL and reset other live claims. The coordinator
  claims centrally and later ports only the owning Bead record into each PR.
- `codex exec` emitted a harmless system-skill installation warning because the
  system skill directory is read-only. Future launches should pass the
  runner's `--skip-agents-render`/`SINNIX_SKIP_AGENTS_RENDER=1` option to remove
  this noise.
- Heavy verification still belongs to each worker, but concurrent lanes use
  distinct worktrees, archives, temp roots, and verify caches. Production DBs
  remain stopped/off-limits; live benchmarks run serially under coordinator
  control.

## Append/CAS publication and production repair

PR #2710 squash-merged as `8a68241809d1cfa218612f54d014c5e0c5436a01`
after seven adversarial iterations. The final correction added a versioned
cursor-authority envelope binding the complete accepted-prefix digest, bounded
tail digest, and `ctime_ns`; archive reconciliation now checks ctime as well as
device, inode, size, and mtime. The final real-route matrix was 18 passed in
29.76 seconds and quick verification run
`20260711T204508Z-quick-956614-ed758411` passed all 13 steps. The deliberate
tradeoff is O(accepted-prefix bytes) verification; `polylogue-yla8.8` owns the
measured sublinear follow-up without weakening authority.

Sinnix master `2350daec8b4654ece9d219e6414c002810c4b015` pins that exact
merge. `nix develop --command switch` built it successfully; a pre-existing
seven-hour `dbus-broker` reload pathology required the wrapper's exact-toplevel
activation fallback. The active package is
`/nix/store/5m9rx8p4rfpvrwjdhd4vyhaw84d7bc7c-python3.13-polylogue-0.1.0`,
and direct installed-source inspection confirmed the ctime guard and observable
cursor-invalidation failure.

With the daemon stopped, the authority census selected 252 cursor rows: 9
missing-parent append raws, 8 accepted broken heads, and 244 cursor-ahead rows.
The authority input is
`/realm/tmp/polylogue-yla8-6-repair-census.json` (SHA-256
`e78915a7e5451e99a9a29b2fec70a68cc761637d6409eaeb86f340f23032d44d`).
The verified durable backup is
`/realm/staging/polylogue-sqlite/yla8-6-authoritative-pre-repair-20260711T2059Z/polylogue-archive-20260711T205907Z/`;
it contains source/user plus 24,289 blobs and was independently verified before
repair. The exact 252 disposable cursor rows were removed, leaving zero selected
rows. Durable raw rows, blobs, heads, receipts, sessions, and user data were not
deleted. The repair receipt is
`/realm/staging/polylogue-sqlite/recovery/20260711T2104Z-yla8-6/cursor-repair.json`
(SHA-256
`951a7afc...`; full digest is recorded alongside the receipt), with the prior
ops tier preserved as `ops-before.db`.

A stopped-writer targeted production reacquisition then ran all 9 dynamically
selected source paths through the installed `LiveBatchProcessor`: 9 succeeded,
0 failed, 95,736,118 input bytes, 551.2203 seconds. Receipt:
`/realm/tmp/polylogue-yla8-6-targeted-reacquire.json` (SHA-256
`303bb9de6111d48c6342826699bc87cb28bdac99fe75847565be20f5c6dbef13`).
The stopped post-target census reports zero current heads with missing parents
and zero cursor-ahead rows; the 9 historical missing-parent raw rows remain as
durable incident evidence. Receipt:
`/realm/tmp/polylogue-yla8-6-post-targeted-census.json` (SHA-256
`d5ad39b36edea36ce61b4ef4c4e4e07e1867d2715e04ac5746dcb7588fa19ae8`).

The daemon restarted at 2026-07-11 23:36:33 CEST with zero restarts and no CAS
error. It is completing the expected one-time modern-cursor reauthentication
backlog: 669 files / 5.1255 GB after archive-authenticated skipping of 14,248
files. Keep `polylogue-yla8.6` open until this bounded catch-up, the final
integrity/census/journal checks, resource-override restoration, and a controlled
sanitized append are evidenced.

## Typed cross-route actuator

The first packaged catch-up disproved the assumption that append/CAS repair was
the final production blocker. Chunk 1 rejected a ChatGPT browser capture as a
conflicting semantic head and an older Gemini full as an older frontier. The
strict rejections preserved the index, and failed cursors correctly retained
`failure_count=1` plus retry time. The dangerous inverse was in the same path:
single-session full replay compared an existing semantic head by frontier
cardinality, so a larger divergent capture could overwrite it without
strict-prefix proof.

PR #2716 (`0653ebdeb0a189d8a8207e3bb33bd08f9e70f5f1`) closed the
cross-route authority split after three independent adversarial rounds:

- Full-only, append-independent byte cohorts migrate to the durable membership
  census. Their byte-revision columns are retired, making rebuild selection
  stable rather than dual-governed.
- Bundle-first, single-first, and already-bound failed retries use the same
  semantic classifier. A related byte head transitions inside the same index
  transaction as the semantic write and receipts; failure rolls the head back.
- Proven prefixes become terminally superseded; larger divergence remains
  ambiguous and leaves the prior head/index intact.
- Metadata-only variants collapse only when every distinct metadata state has
  a unique direct provider `updated_at`. Missing or equal timestamps stay
  ambiguous.
- Browser capture no longer launders `provenance.captured_at` into provider
  `updated_at`.

The live ChatGPT conflict was measured directly: both retained revisions have
4,368 messages, 251 attachments, identical message/event/attachment hashes,
equal title/created time, and provider updates of 2026-04-11 versus 2026-07-01.
Only session metadata differed, so the direct provider timestamp is sufficient
to choose the later metadata revision without weakening content authority.

Verification: focused authority/timestamp/browser matrix 7 passed; membership
classifier file 5 passed; existing semantic CAS rollback 1 passed; final-head
quick run `20260711T220233Z-quick-1445786-3516ef36` passed 13/13. Six
broader live-batch failures reproduced identically on detached `origin/master`
and were classified baseline. GitHub-hosted jobs failed before allocation under
the account billing lock; GitGuardian passed. CodeRabbit remained in-progress
without findings beyond a bounded wait; the three independent reviews had
already found and driven correction of substantive release blockers.

Sinnix master `2eb95c68` pins and deployed the exact #2716 squash commit. Normal
activation again hit the known `dbus-broker` reload failure; the exact-toplevel
fallback succeeded. Installed package:
`/nix/store/gh1jv54vgdwzkr0873ibbak8scfrfjjk-python3.13-polylogue-0.1.0`.
Direct installed-source inspection confirms the governance-retirement flag,
provider timestamp classifier, and removal of the capture-time fallback. The
post-deploy daemon began with 666 files / 5.0591 GB pending after skipping
14,252, down from 669 / 5.1255 GB before deployment.

## Byte-head recovery, testing discipline, and bounded parallel lanes

Subsequent production catch-up exposed three more authority defects, each fixed
without weakening the general CAS rule. PR #2717 authorized classifier-proven
semantic head transitions. PR #2718 preserved an accepted byte head during
membership replay unless the membership evidence selected that exact head; a
divergent or merely newer raw still fails closed. PR #2719 reconciled immutable
receipts when membership replay re-selected an equivalent representative raw,
while limiting the compatibility lookup to terminal `superseded` decisions.
The first formerly failing 45.7 MB production snapshot then completed in
652.2 seconds, proving these fixes on the installed route.

That success exposed a measured performance defect rather than a correctness
guess: py-spy attributed 498/754 samples to deleting a session's messages while
foreign-key backreferences lacked leading indexes. `polylogue-rgbj` records the
derived-index schema and benchmark follow-up; no live schema bump was attempted
during recovery.

Two isolated Terra lanes were useful only after independent review:

- PR #2722 parses real Claude background completion protocol shapes, including
  failed, queue-operation, attachment, and historical optional-tool-id forms.
  Synthetic `completed (exit code 1)` fixtures initially false-greened the work;
  live-corpus review corrected them. Correlation now fails closed for duplicate
  task IDs and duplicate exact task/tool pairs. The final route suite passed 19
  tests.
- PR #2721 adds a quick-gate policy for explicit pytest timeout overrides. The
  first AST scanner missed aliases, parameter marks, class/module marks,
  source-order rebinding, and mutable marker lists. Repeated mutation review
  narrowed the supported contract to bounded static syntax and fail-closed
  aliases; 50 registered-command tests and all 14 quick steps passed.

The main production CAS failure at 2026-07-12 01:40:58 was not divergent data.
For `codex:019f4f5f-ab06-70a1-a4ae-163d9e1969d8`, full raw `adfe162b...`
contained exactly the accepted baseline `81051219...` plus append `b653d95b...`
at byte frontier 2,645,672. Recomputing `append_source_revision` over the full
snapshot's tail exactly produced the stored append revision. Split and full
Codex parsing nevertheless produced different normalized hashes because event
positions and message metadata are segmentation-sensitive.

PR #2723 (`3423d3cf0dc3a0d6f1c01285e585ba85f48827d2`) therefore adds a
one-shot, in-memory fold authorization rather than an equal-frontier exception.
Inside the existing index transaction it proves the full baseline prefix,
contiguous append offsets, raw predecessor links, predecessor revisions, each
tail hash/append revision, and the exact final byte frontier. The capability is
bound to the precise old head and incoming full receipt; every other
equal-frontier normalized-hash change remains a conflict.

The first tests again looked stronger than they were: arbitrary byte strings
and hand-built sessions did not exercise Codex segmentation, attachment state
was empty, and FTS docsize equality did not prove term rollback. Final tests
route a reduced Codex JSONL stream through the real parser in full and segmented
modes, assert distinct normalized hashes over identical folded bytes, and use
non-empty block/event/attachment state plus a candidate-only FTS term. Single
and multi-append success and seven proof mutations all pass; adversarial review
converged with no implementation or verification blocker. The exact SQL audit's
five failures were reproduced unchanged on detached `origin/master` and are
baseline allowlist drift, not introduced SQL.

Sinnix master `0d6b128` pins #2723. The normal switch hit the known
`dbus-broker` reload failure, and exact-toplevel fallback succeeded. Installed
package `/nix/store/ah41rqf4j348qnr62m4mavgwqzd1m8c6-python3.13-polylogue-0.1.0`
contains `FullSnapshotFoldAuthorization`; production restarted at 02:18:54 CEST
with zero restarts and began a reduced 394-file / 3.1113 GB retry backlog.
Keep `polylogue-yla8.9`, `polylogue-yla8.6`, and parent `polylogue-yla8` open
until the exact failed raw transitions, catch-up completes, final census and
SQLite integrity pass, temporary resource overrides are restored, and the
Beads reconciliation lands.
