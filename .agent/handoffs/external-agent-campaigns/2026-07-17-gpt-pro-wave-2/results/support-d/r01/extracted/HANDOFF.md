# Lane D raw-authority restart proof harness — handoff

## Mission outcome

This package adds an integration-ready, local-only proof command for raw-authority crash recovery. It builds fresh production-format archives, drives the production `repair_raw_materialization` reconciler through three real durability boundaries, resumes from the durable `source.db` ledger, and proves convergence through two consecutive matching quiescent dry-run census receipts.

The implementation does not touch a live archive, authorize an apply, add an operational bypass, or lower production limits. It introduces no alternative replay model. The synthetic setup is only fixture construction; planning, replay, outcome recording, interrupted-census recovery, blocker creation, finalization, postflight, census reads, and application-receipt validation all use the existing production route.

## Snapshot identity

The attached Chisel snapshot was generated at `2026-07-18T013442Z` from `/realm/project/polylogue`.

- Branch: `master`
- Commit: `bf8191b3f56aa40da8f271df7f3385c712825497`
- Commit date: `2026-07-18T02:39:50+02:00`
- Subject: `feat: land WebUI v2 scaffold, design system, and generated client (#3074)`
- Branch delta from the captured remote default branch: empty
- Captured working tree: dirty

The captured dirty baseline contains three pre-existing tracked edits, all outside this package:

- `polylogue/archive/query/unit_results.py`: replaces a redundant literal cast with explicit `count`/`key` branches.
- `polylogue/daemon/http.py`: narrows the object passed to `select.select` to `socket.socket`.
- `polylogue/hooks/__init__.py`: replaces a redundant provider cast with explicit `claude-code`/`codex` branches.

Those edits are preserved as snapshot authority and excluded from `PATCH.diff`. They matter only because focused mypy on clean `HEAD` reports the three pre-existing errors that the captured dirty patch fixes; focused mypy against the actual captured snapshot passes.

## Inspected authority

The implementation followed the production route beyond the obvious repair entry point.

Primary source anchors:

- `polylogue/storage/raw_authority.py:31-123`: typed statuses, immutable `RawReplayPlan`, per-plan outcome, and durable census receipt.
- `polylogue/storage/raw_authority.py:357`: bounded durable census reader.
- `polylogue/storage/raw_authority.py:759`: census/plan publication.
- `polylogue/storage/raw_authority.py:1052`: exact application-receipt postcondition validator.
- `polylogue/storage/raw_authority.py:1159`: guarded one-row outcome commit.
- `polylogue/storage/raw_authority.py:1255`: finalization, pending-outcome refusal, postflight conservation, and atomic lifecycle transition.
- `polylogue/storage/raw_authority.py:1382`: restart recovery from exact durable source/index postconditions, including fail-closed stale rejection.
- `polylogue/storage/repair.py:80-82`: production 1 GiB application envelope and 25-component parser-census limit.
- `polylogue/storage/repair.py:3589`, `:3871`, `:5563`: production candidate closure, component ordering, and reconciler entry point.
- `polylogue/storage/repair.py:5579-5598`: interrupted-census recovery and finalization before new planning.
- `polylogue/storage/repair.py:5984-6202`: production outcome commits, finalization, and conservation-error metric.
- `polylogue/product/raw_authority.py:52-69`: product repair/read wrappers.
- `polylogue/daemon/cli.py:614-646`: daemon drain path into the product repair API.
- `polylogue/storage/sqlite/archive_tiers/source.py:116-226`: durable census, immutable plan, census-plan, post-plan, and blocker tables.

Existing tests inspected and retained include `tests/unit/storage/test_raw_authority_ledger.py` exact interruption recovery and stale-membership cases, `tests/unit/storage/test_repair.py` raw-materialization route coverage, and `tests/unit/devtools/test_raw_authority_scale_proof.py` typed deferred/terminal and two-census scale proof coverage. Repository architecture and test guidance were read from `CLAUDE.md` and `TESTING.md`. Relevant Beads and history are enumerated in `EVIDENCE.md`.

## Implemented mechanism

### Synthetic topology

Each fault case receives its own fresh active archive under the supplied work directory. Six ChatGPT raw bundles are parsed through the production parser census into twelve durable membership rows and four immutable replay components:

| Role | Raw members | Membership shape | Expected conserved endpoint |
| --- | ---: | --- | --- |
| `solo-one` | 1 | independent bundle | `executed` |
| `solo-two` | 1 | independent bundle | `executed` |
| `membership-terminal` | 2 | siblings linked by one shared logical source | `terminal` |
| `membership-deferred` | 2 | siblings linked by one shared logical source; one durable ordinary-replay deferred application receipt | `deferred` |

The production component order is asserted as `[1, 1, 2, 2]`. The first dry run completes parser census. A second dry run publishes the immutable preview inventory; its four plan IDs must be identical to the first apply census and every preview row must be conserved as `carried_forward`.

The deferred fixture row follows the repository's existing scale-proof/test precedent. It is fixture setup in a disposable `index.db`, not an operational repair action or live SQL procedure.

### Fault matrix

| Boundary | Injection point | Boundary evidence required before raising | Restart behavior proved |
| --- | --- | --- | --- |
| `before_outcome_commit` | Patched call to production `repair.record_raw_replay_outcome` for `solo-one` | Replay application already satisfies the immutable plan's production application-receipt validator; selected ledger row remains `outcome_recorded=0` | Recovery reads exact durable postconditions, records one recovered `executed` outcome, finalizes the interrupted census, and drains the remaining components |
| `after_outcome_commit_before_census_finalization` | Patched call to production `repair.finalize_raw_authority_census` | All four selected rows have committed outcomes; the two executed receipts validate against their immutable plans; lifecycle remains `planned` | Recovery finalizes the already-complete census without replaying or duplicating outcomes |
| `during_resumed_batch` | First crash before `solo-one` outcome commit, then a second crash before `solo-two` outcome commit in the next production repair invocation | The second crash observes at least one already finalized `interrupted` census and validates the newly applied `solo-two` receipt before its outcome commit | A subsequent restart recovers the second interrupted census, preserves the first recovery, drains all remaining work, and leaves two interrupted census receipts rather than losing either attempt |

Fault injection is at Python call boundaries that directly surround real SQLite transactions. It is not a simulated state machine. The crashes are raised after production writes but before the selected production outcome/finalization transaction, then the patch is removed and a new production reconciler invocation performs recovery.

### Conservation law

For every finalized (`completed` or `interrupted`) census `C`:

1. `C.plan_count` equals the number of distinct durable `raw_authority_census_plans` rows for `C`.
2. Each selected immutable plan has exactly one recorded outcome in `executed`, `retryable`, `deferred`, `terminal`, or `rejected_stale`, and is never `carried_forward`.
3. Each unselected immutable plan has exactly one recorded `carried_forward` outcome.
4. Across retry/recovery censuses in this finite topology, each of the four initial immutable plan IDs reaches exactly one endpoint in `executed`, `deferred`, or `terminal`; `retryable` may exist only as an intermediate selected outcome.
5. Every `executed` endpoint carries a production v2 application receipt that validates against the durable immutable plan's exact membership, source-head, index-application, session, and content-hash postconditions.
6. No `planned` census or unresolved durable blocker remains.
7. Fixed point requires two final, consecutive, predecessor-linked, completed, quiescent dry-run censuses with zero plan/executable/residual/post-plan counts and equal inventory digest, residual digest, scope, and parser fingerprint. The first must not claim fixed point; the second must.

The harness reads the durable rows directly for algebraic completeness and also resolves every census through the production `read_raw_authority_census` API so the query surface is part of the proof.

### Deliberate failure proofs

Two anti-vacuity tests demonstrate that the harness does not merely bless its own execution:

- Conservation mutation: after a successful fault case, one finalized selected census-plan row is changed from `outcome_recorded=1` to `0`; the durable audit fails with `exactly one recorded outcome per plan`.
- Postcondition mutation: after production applies `solo-one` and crashes before its outcome commit, one durable applied membership witness is deleted. Production restart validation records `rejected_stale`, opens one durable blocker, and the harness fails because convergence is fail-closed.

## Changed files

- `devtools/raw_authority_restart_proof.py` — new proof runner, topology fixture, fault injection, restart drain, fixed-point confirmation, and durable audit.
- `tests/unit/devtools/test_raw_authority_restart_proof.py` — successful matrix, conservation mutation, postcondition mutation, and CLI/catalog tests.
- `devtools/command_catalog.py` — registers `devtools workspace raw-authority-restart-proof`.
- `docs/devtools.md` — synchronized generated command reference.

No production storage, reconciler, schema, daemon, or limit file is changed. `FILES/` is omitted because the unified diff fully disambiguates every change.

## Acceptance matrix

| Mission requirement | Package evidence | Status |
| --- | --- | --- |
| Multiple immutable replay components | Four preview/apply-identical production plan IDs over component sizes `[1,1,2,2]` | Met |
| Membership siblings | Two independent two-raw logical-source sibling components; 12 durable membership rows total | Met |
| Deferral and terminal conservation | Final partition is exactly `2 executed + 1 deferred + 1 terminal` in every fault case | Met |
| Durable census receipts | All attempts and both fixed-point passes are read from `source.db` and through the production reader | Met |
| Crash before outcome commit | Exact executed receipt validated while `outcome_recorded=0`, then recovered on restart | Met |
| Crash after outcome commit/pre-finalization | Four committed outcomes retained and finalized on restart without duplicate replay | Met |
| Crash during resumed batch | Two successive interrupted attempts retained; second occurs after the first recovery | Met |
| Two-census quiescent fixed point | Final matching dry-run receipts are consecutive and second has `fixed_point=true` | Met |
| Exactly-once terminal/conserved representation | Per-census algebra plus one terminal endpoint per initial plan ID | Met |
| Broken conservation fails | Durable outcome bit mutation is rejected | Met |
| Postcondition mutation fails | Production recovery emits `rejected_stale` and blocker; harness rejects convergence | Met |
| Real APIs/storage, no parallel model | Production repair, ledger, recovery, finalization, reader, receipt validator, and archive format are exercised | Met |
| No live action/bypass/limit reduction | Fresh disposable archives only; default `raw_artifact_limit=None`, 1 GiB payload limit, 25-component census limit | Met |
| July-15 scale/resource/daemon-health proof | Existing separate scale lane; not claimed by this compact restart package | Not in scope of this package |
| Live preflight and operator decision | Inputs listed below; paired local lane retains authority | Unverified and intentionally not performed |

## Apply order

1. Confirm the target is the named snapshot or a descendant where the four patch paths have not diverged. Preserve or separately account for the three captured dirty baseline edits listed above.
2. From repository root, run `git apply --check PATCH.diff`.
3. Apply with `git apply PATCH.diff`.
4. Verify generated reference sync with `python -m devtools render devtools-reference --check`.
5. Run the focused commands in `TESTS.md` using the repository's locked environment.
6. Run the proof through the registered control plane, for example:

   `python -m devtools workspace raw-authority-restart-proof --workdir /path/to/disposable-work --keep --json`

7. Inspect the emitted JSON and retained disposable case archives. Do not redirect this command at an existing or live archive; it always creates/removes only `<workdir>/raw-authority-restart-proof`.

## Local-only preflight inputs still required before any live-lane decision

This package is not live authorization. The paired local lane must independently collect and bind all of the following to the exact intended action:

- Exact installed package/build commit, binary/unit identity, target source revision, and source/index/user/ops schema versions; confirmation that the reviewed patch and packaged command are identical.
- Actual current worktree/package dirty state and any divergence from `bf8191b3f56aa40da8f271df7f3385c712825497`.
- Daemon PID/unit/journal identity, writer lease and queue state, stopped or otherwise proven-quiescent maintenance window, and proof that no competing writer can resume.
- A new verified durable full-evidence backup covering `source.db`, `user.db`, retained blobs and required active derived/ops inputs, with manifest hashes, verification receipt, and tested read/restore evidence.
- Current exact raw-authority frontier/profile: direct candidates, expanded raws, authority components, joint component/raw/byte distribution, largest aggregate component, missing blobs, parse failures, quarantines, deferred/terminal debt, and unresolved blockers.
- Current cursor/source/head/application/content-hash/session/message/FTS state for every implicated path and any historical no-shrink witness; unknown rows remain red.
- Free disk, WAL/temp budget, RSS/PSS/swap, I/O pressure, cgroup limits, wall-time envelope, and daemon health latency for the intended bounded action, especially the largest aggregate authority component.
- Two consecutive stopped/quiescent dry-run census IDs and full digests under the same scope and parser fingerprint, with zero executable plans, no untyped residual debt, no `planned` census, and no unresolved blocker.
- Bounded action and rollback/stop criteria, journal capture destination, postflight witness comparisons, and append/no-op validation where required by `polylogue-yla8`.
- Explicit operator authorization after reviewing all current evidence. No historical count, prior backup, synthetic result, or this handoff can substitute for that authorization.

## Verification performed

The final patch was generated only from the four changed files and apply-checked against both clean commit `bf8191b3...` and the captured dirty snapshot. It was then applied to a separate copy of the captured snapshot and its new tests, Ruff, and mypy were rerun there.

The final proof command produced deterministic proof ID `raw-authority-restart-proof:a114d2360d08b2d323281ca0`. Every matrix case produced terminal counts `{deferred: 1, executed: 2, terminal: 1}`, no planned census, no open blocker, and a second matching dry-run fixed point. Exact commands and results are in `TESTS.md`.

The repository-managed `devtools test` wrapper could not run in this container because it refuses disk-backed pytest when `/dev/shm` has only 64 MiB; it exited 125 before collecting tests. The same locked virtual environment was used for focused raw pytest with explicit disk-backed `--basetemp`. Full `devtools verify`, full non-integration pytest, `nix flake check`, packaged/Nix execution, daemon-process kill testing, and all live-archive checks remain unverified.

## Risks and iteration value

The proof is deterministic and production-route, but compact. It does not reproduce the July-15 archive's tens of thousands of components, multi-gigabyte retained payload, resource pressure, or daemon-health envelope. That remains `polylogue-hjpx.2` work. It also injects exceptions at exact Python call/transaction boundaries rather than delivering SIGKILL, power-loss, or filesystem-fault semantics; SQLite crash/fsync behavior and process-supervisor recovery require a contained process-level pass.

A small follow-up iteration would likely add only integration repairs caused by local path/API drift, packaging checks, or documentation polish. The core conservation mechanism is already exercised end to end. A substantial second pass could add real subprocess/SIGKILL boundaries, packaged/Nix command execution, and an admitted July-15-shaped resource/daemon-health run. That pass would add material operational confidence, but it requires the paired local lane's containment, adequate shared memory/disk, packaged daemon environment, and live-gate evidence; it should not weaken or replace this proof.
