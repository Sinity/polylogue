# Lane D raw-authority restart proof — evidence

## Snapshot provenance

The archive's `polylogue-overview.json`, `polylogue-overview.md`, `polylogue-manifest.json`, all-refs bundle, and working-tree tarball agree on the following authority:

- Generated: `2026-07-18T013442Z`.
- Source: `/realm/project/polylogue`.
- Branch: `master`.
- Commit: `bf8191b3f56aa40da8f271df7f3385c712825497`.
- Dirty working tree: true.
- Branch delta files/log/patch: empty, so the branch itself matched the captured remote default branch.

`git show` in the restored repository reports commit date `2026-07-18T02:39:50+02:00` and subject `feat: land WebUI v2 scaffold, design system, and generated client (#3074)`.

The working-tree tarball contains three tracked dirty edits. They are type-narrowing fixes in `polylogue/archive/query/unit_results.py`, `polylogue/daemon/http.py`, and `polylogue/hooks/__init__.py`; they are unrelated to raw authority. They were retained as baseline authority but excluded from the package diff. The patch changes only the four files listed in `HANDOFF.md`.

## Repository contracts inspected

`CLAUDE.md` establishes that substrate/storage owns semantics (`:17-38`), `source.db` is durable authority (`:93-99`), the daemon owns writes (`:140`), and durable schema changes require explicit additive migrations and backup discipline (`:164-176`). The package therefore keeps the proof in a devtool/test surface while calling the existing storage semantics, and introduces no schema or duplicate lifecycle.

`TESTING.md` prefers `devtools test` for contained focused runs (`:16-24`) but explicitly permits raw pytest for ad hoc needs (`:26-31`). It documents `/dev/shm` managed basetemps (`:89-103`) and full `devtools verify --all`/`nix flake check` gates (`:30-34`). The container's 64 MiB shared-memory mount triggered the documented wrapper refusal, so focused raw pytest used a disk basetemp and the full gates remain unverified.

## Production source findings

### Durable types and storage

- `polylogue/storage/raw_authority.py:31` defines the closed status set: planned, executed, retryable, deferred, terminal, rejected-stale, and carried-forward.
- `:41` defines immutable `RawReplayPlan` identity and witnesses.
- `:65` defines a typed per-plan outcome with optional application receipt.
- `:103` defines the census receipt, including before/post digests, mode/lifecycle, quiescence, fixed point, and predecessor.
- `:759` persists census/plan inventories before apply.
- `polylogue/storage/sqlite/archive_tiers/source.py:116`, `:157`, `:168`, and `:201` create durable census, immutable plan, census-plan, and blocker tables in `source.db`.

### Outcome and restart boundaries

- `polylogue/storage/raw_authority.py:1159` records an outcome in one source-tier transaction with `WHERE selected = 1 AND outcome_recorded = 0`; it requires exactly one updated row. This is the before-outcome-commit fault boundary.
- `:1255` finalizes only while lifecycle is `planned`; it refuses any pending selected outcome, verifies persistent retryable/carried plans survive postflight, inserts post-plan rows, stores post digests, and atomically moves the census to `completed` or `interrupted`. This is the after-outcome/pre-finalization boundary.
- `:1382` scans planned apply censuses on restart. For each unrecorded selected plan it reconstructs the application receipt from durable state and validates it. Exact postconditions become recovered `executed`; stale preconditions invoke durable rejected-stale/blocker handling; otherwise the same immutable plan becomes `retryable`.
- `polylogue/storage/repair.py:5579-5598` runs that recovery before selecting new candidates and finalizes recovered censuses as interrupted.

### Production reconciler and limits

- `polylogue/storage/repair.py:80` sets the real 1 GiB application payload envelope.
- `:82` sets the default parser-census limit to 25 components.
- `:3589` computes production candidate and membership expansion state.
- `:3871` computes ordered independent authority components.
- `:5563` is the production `repair_raw_materialization` entry point. The proof never supplies a smaller `raw_artifact_limit` or `max_payload_bytes`.
- `:5984-6185` records typed outcomes at each production route branch and finalizes the census after postflight.
- `:6202` exposes the plan-conservation error metric; the result fails closed when conservation or stale rejection is nonzero in the surrounding success predicate.
- `polylogue/product/raw_authority.py:52-69` exposes product repair and census reads.
- `polylogue/daemon/cli.py:614-646` shows the daemon's bounded drain calls the product wrapper rather than owning separate semantics.

### Receipt and census observation

- `polylogue/storage/raw_authority.py:1052` validates v2 application receipts against exact plan-bound source memberships, accepted revision heads, index application rows, materialized sessions, accepted raw IDs, and content hashes.
- `:357` is the bounded durable census reader used by operator/product surfaces. The proof audits SQL algebra and separately requires this reader to return the same census count with no unexpected pagination.

## Existing tests inspected

- `tests/unit/storage/test_raw_authority_ledger.py:442` proves interrupted apply recovery from exact durable postconditions.
- `tests/unit/storage/test_raw_authority_ledger.py:587` proves partial expanded-membership postconditions are rejected.
- The same file's finalization/recovery tests around `:737` prove planned censes with all outcomes can be finalized after restart.
- `tests/unit/storage/test_repair.py:695`, `:757`, `:838`, and `:1916` cover interrupted index rebuild, terminal/deferred/executable partitioning, complete governed membership replay, and deferred receipt reuse.
- `tests/unit/devtools/test_raw_authority_scale_proof.py:28` covers two matching quiescent censes; `:433` covers explicit deferred cohort convergence.

The new package composes these individual properties into one multi-component, three-boundary restart matrix and adds direct conservation/postcondition mutations.

## Beads findings

The complete records were read from `polylogue-beads-export.jsonl`; later notes were treated as authoritative over older path/line descriptions.

- `polylogue-hjpx` — P0, in progress, “Make RawAuthorityReconciler execute accepted replay plans to fixed point.” Its acceptance criteria require exact per-plan conservation, fair complete-component scheduling, explicit retry/defer/terminal outcomes, two quiescent dry runs, mutation failures, and live build/schema/backup/dry-run/resource/operator gates. This package addresses the compact restart/conservation proof slice only.
- `polylogue-hjpx.1` — P0, closed by PR #2961. It requires immutable before/after plan census, exact application/membership receipts, interruption without duplicate/partial visibility, durable rejected-stale blockers, carried-forward accounting, and two-census fixed point. The implementation reuses precisely those landed contracts.
- `polylogue-hjpx.2` — P1, in progress, “Prove raw replay convergence at the July-15 archive shape.” It requires a sanitized current-shape corpus, measured RSS/PSS/swap/I/O/time/daemon health, cursor/head/FTS invariants, and fairness/conservation mutations. This compact six-raw proof does not claim that scale/resource closure.
- `polylogue-hjpx.3` — P0, closed by PR #3011. It made read-only raw-authority scale-profile capture bounded and complete. The package does not alter or bypass that profile route.
- `polylogue-lkrc` — P0, in progress, “Converge raw evidence authority through one proof-driven reconciler.” It forbids independent proof/receipt lifecycles. The package therefore invokes the existing reconciler/ledger rather than creating a parallel authority engine.
- `polylogue-yla8` — P0, in progress, live authority-safe closure gate. It requires exact packaged build/schema identity, daemon state, a new verified backup, current frontier and cursor/head/application/hash/message/FTS evidence, explicit authorization, ordinary bounded actuators only, and fail-closed handling. No live phase was attempted here.

The most recent yla8 notes explicitly refused prior live action because current evidence was red and because a current verified backup/quiescent preflight was absent. This reinforces the package boundary: synthetic success cannot authorize live replay.

## Relevant history inspected

- `593ef3c626b89a4f9e02d400b1aad0fb69a03c55` — `feat(storage): conserve raw authority replay plans (#2961)`: introduced the durable immutable plan/census/outcome/blocker foundation.
- `5b40d57000c5592e288fc03281e187f832164c4a` — `fix(repair): fail closed on plan conservation drift (#3029)`: made conservation error non-success.
- `d108e9431eafa3d76c9d8fee42892fdfeb2bd67d` — `test(repair): preserve authority state across replay recovery (#3034)`: added interruption/recovery authority-state evidence.
- `536a53efac0cbe4a2473ad379e4db49ef3fce74d` — `fix(repair): harden raw authority convergence (#3046)`.
- `bce7336d3bb2e493080b37fd0bc76b429b0c1cbd` — `feat: harden archive continuity closure (#3051)`.
- `fd556c5d9f8e9416316c79c8c4a1d988acae0f9d` — `fix: harden raw-authority repair and scale proof (#3072)`: current product-wrapper and proof hardening immediately before the snapshot.
- `9b801a7ccbd219ef6a6f900189c90f205b468a77` — bounded aggregate raw-authority profile (#3011).
- `314bcb011c7c75ffb51056e348adb0e8ac31f4ce`, `695fedbc7094531cf994cc7ad97cf40a09048a5f`, and `0829f65e06ec656c7bd210644634da15e1b8ddb2` — parameterized/bounded/direct-publish scale corpus evolution (#3009, #3038, #3040).

History confirms that the current route is intentionally one production reconciler plus durable source-tier ledger, with scale proof as a separate concern. No stale plan naming an older repair line range was used as API authority.

## New implementation anchors

- `devtools/raw_authority_restart_proof.py:53` — fault-boundary enum.
- `:186` — fresh compact topology and durable preview plan inventory.
- `:401` — before-outcome commit injection and exact receipt validation.
- `:484` — after-all-outcomes/pre-finalization injection and committed receipt validation.
- `:588` — production restart/drain loop with conservation/blocker checks.
- `:612` — two matching quiescent dry-run confirmation.
- `:638` — three-case matrix orchestration.
- `:704` — durable census/plan algebra, production-reader check, exactly-once endpoint audit, receipt validation, blocker/planned-census check, and final fixed-point audit.
- `:944` — isolated proof runner/report.
- `tests/unit/devtools/test_raw_authority_restart_proof.py:16`, `:59`, `:94`, `:145` — success, conservation mutation, postcondition mutation, and CLI/catalog tests.
- `devtools/command_catalog.py:642` and `docs/devtools.md:215` — registered/synchronized command surface.

## Final local proof evidence

Final proof ID: `raw-authority-restart-proof:a114d2360d08b2d323281ca0`.

All cases produced the same deterministic terminal partition and final digests:

- Terminal partition: `executed=2`, `terminal=1`, `deferred=1`.
- Inventory digest: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`.
- Residual digest: `6ab22a5de6879f31b9aed726d238eccdb85d4bab186bf0b5dcca01b664c79e69`.
- Open blockers: `0`.
- Planned censuses after restart: `0`.
- Second quiescent census fixed point: true in all cases.

Boundary-specific durable evidence:

- Before outcome commit: selected `4`, recorded `0`, one executed application receipt validated before crash, one interrupted census after recovery.
- After outcomes/pre-finalization: selected `4`, recorded `4`, both executed receipts validated before crash, one interrupted census after recovery.
- During resumed batch: first selected `4`/recorded `0`; second selected `3`/recorded `0` while one interrupted census already existed; one executed receipt validated at each crash; two interrupted censuses retained after final recovery.

The evidence JSON and case archives were generated only in disposable `/mnt/data` paths for verification and are intentionally not copied into the result ZIP. The command regenerates them deterministically.

## Contradictions and resolution

- The assignment says to use the exact Result ZIP “named near the top,” but the supplied prompt contains no explicit filename. The delivered name is therefore derived from the attachment/mission identifier: `support-d-repair-proof-harness.zip`.
- Older Bead text cites former `repair.py` line ranges and foundation APIs. Current snapshot source is authoritative; all anchors in this package use the actual restored files.
- Clean commit `bf8191b3...` alone has three unrelated focused-mypy errors. The attached snapshot explicitly declares `dirty=true` and carries their fixes, so the integration baseline is the captured dirty tree. The package still apply-checks against clean HEAD, and none of those baseline fixes is copied into `PATCH.diff`.
- A compact synthetic proof cannot satisfy the separate July-15 scale/resource/live gate. The package marks that work unverified instead of treating terminal counts as operational authorization.
