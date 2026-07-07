---
created: 2026-06-28
purpose: SINGLE consolidated worklog/backlog for the dogfood-feedback-engine mandate. Gathers ALL threads
  so none is lost across context resets. Newer ideas do NOT cancel earlier ones.
status: active — THE index. Update status inline as threads progress.
read-with: STATE-OF-UNDERSTANDING-2026-06-28.md (evidence), project_standing_goal_2026_06_28 (mandate)
---

# Master backlog — Polylogue dogfood engine

Legend: [DONE] shipped/verified · [WIP] in progress · [NEXT] queued, ready · [DESIGN] needs design/sign-off ·
[BLOCKED] waiting on something · [BATCH-REINGEST] needs the one definitive v16+ reingest.

## 0. Live state
- **2026-06-29 overnight session → see [HANDOFF-2026-06-29.md](HANDOFF-2026-06-29.md) (read first).**
  Shipped: parser 139-retry fix, DSL `with` slice, canonical-URL projection, recovery-digest honesty
  (text-mined events → candidates). Separate branches: FTS cargo-cult removal, browser-capture POSTING
  channel (gated OFF). Browser capture works end-to-end; project "a" sessions captured. New audit:
  [audit-construct-algebraic-2026-06-29.md](audit-construct-algebraic-2026-06-29.md) (v17 epoch plan).
  Dev daemon WITH capture: relaunch `/realm/tmp/polylogue-dev/relaunch.sh`.

### (earlier 2026-06-28 live state, superseded above)
- **2026-06-29 ~00:01: forced v16 reingest.** Keystone commit bumped code to v16; fresh daemon rejected the
  v15 dev archive (fresh-first schema model). Wiped dev archive, reingesting FRESH on v16 (gets keystone
  structured outcomes) with POLYLOGUE_LIVE_FULL_INGEST_WORKERS=8 (parse-pool experiment — default was 1=serial).
  A later insights-cut (v17) will reingest again; dev archive is disposable, multiple reingests are fine.
- Dev daemon ingesting `/realm/tmp/polylogue-dev/archive` (XDG_DATA_HOME=/realm/tmp/polylogue-dev/xdg),
  log `/realm/tmp/polylogue-dev/converge.log`, flags POLYLOGUE_SKIP_RUN_PROJECTION=1 +
  COMMIT_BATCH_MESSAGES=8000. setsid-detached. ~chunk 290/526. Relaunch cmd in STATE §11.
- Branch `feature/dogfood/parallel-parse`. Uncommitted: rebuild.py (skip stopgap, dev-only), AGENTS.md (regen).

## 1. Shipped this session [DONE]
- DOGFOOD ARTIFACT (2026-06-29): keystone analytics on the live v16 archive (via `actions` view) —
  113k tool calls, 11,348 errors. Error rate by tool: MultiEdit 17.3%, shell 12.7%, Bash 8.4%, Edit 7.4%,
  Read 2.1%, Grep 0.8%. Codex exit codes: 0×63k, 1×6.5k, 2×1k, 101×641, 124(timeout)×496, 127×97. Proves the
  keystone enables tool-mechanics/claim-vs-evidence analytics from structure (no regex).
- `7f9ac2ac5` status live ingest-workload surface (ops.db projection).
- `19490eadd` KEYSTONE structured tool-result capture, schema v16 (blocks.tool_result_is_error/exit_code,
  actions view). The enabler for reading outcomes from structure.
- `cdf3c1eb1` recovery digest workdirs noise fix.

## 2. BIG architectural redesign: Insights collapse (substrate-not-interpreter) [DESIGN/NEXT]
The core construct-validity work. Operator pre-signed-off "CUT the bad heuristics". Detail:
insight-redesign-PLAN.md, STATE §4/9/10, construct-validity-audit-2026-06-28.md.
- [NEXT] CUT the regex prose-miner: `_events_from_text` → `session_work_events`, `session_phases` dead cols.
- [NEXT] Inline deterministic stats onto `sessions`; drop `session_profiles` separate table + deferred pass +
  read join + materialization tracking → sessions become atomic, most live_convergence_debt evaporates.
- [DESIGN] Rebuild run-projection trio from STRUCTURED outcomes (now possible via keystone) OR drop the
  `run`/`observed-event`/`context-snapshot` query units. Remove env-stopgap once digest gone.
- [NEXT] Cost as a VIEW over token cols × price catalog (reprice-free, never stale).
- [DESIGN] "Full missing scope" — define target analytics taxonomy; FLAGSHIP = claim-vs-evidence calibration
  (assistant claim vs structured tool outcome — only possible here). insight-redesign-PLAN §"OPEN PIECE".
- [BATCH-REINGEST] schema changes here batch with keystone → ONE definitive v16+ reingest into final shape.

## 3. BIG architectural redesign: Context-pack/compile_context collapse → query algebra [DESIGN, sign-off GIVEN]
Operator gave implicit sign-off. Detail: context-pack-gratuitous-2026-06-28.md, dsl-vs-flags-audit-2026-06-28.md.
- [WIP] DSL `with <units>` projection clause (e.g. `sessions where repo:x with assertions, actions`). Agent
  a68dadb848e1876ec mapping parse→spec→execute→render touchpoints (read-only). THEN implement vertical slice
  (`with assertions`) + verify on dev archive + generalize. Additive — NO reingest.
- [NEXT] General read modifiers: `--max-tokens` (token-bounded accumulation + omission accounting),
  multi-`--view`, `--include-assertions` (→ subsumed by `with`).
- [NEXT] Remove redundant SELECTION flags (`--project-path/-repo`, `--since/--until`, `--pack-origin/-query`,
  `--max-sessions`, `--message-role`, `--material-origin`, `--message-type`); route via the `find` query.
  Smoking gun: cli/read_views/context.py `run_read_context_pack` does `del request` (discards the query!).
- [NEXT] Re-express/retire compile_context + build_context_pack + mcp/context_pack.py (15 DTOs) + the
  ContextSpec/ContextImage/compiler trio as thin shims over `find … then read`.

## 4. Recovery digest redesign (continuity substrate) [WIP/DESIGN]
Detail: recovery-digest-dogfood-2026-06-28.md. NOTE: operator deferred the automated SessionStart
self-recovery LOOP (days-out, operator-orchestrated) — do NOT build that now. Digest QUALITY is in scope.
- [DONE] workdirs noise collapse (cdf3c1eb1).
- [NEXT] Lead "# Resume:" with semantic state (session summary / recent authored-user asks), not the title.
- [DESIGN] Rebuild execution-projection on structured outcomes (keystone) — drops default-to-`failed`,
  regex-guessed events. Same dependency as §2 run-projection.
- [NEXT] Collapse verbatim subagent-prompt dumps to role + 1-line intent + outcome.
- [NEXT] Fix the BROKEN SessionStart hook (~/.claude/hooks/sessionstart-polylogue-recall.sh uses removed
  `list` verb → emits nothing). Low-risk maintenance, independent.

## 5. Perf threads — CONCLUSION (2026-06-29): well-engineered, no significant win available.
Verified 3 ways (redundancy checks; parse-worker A/B = 0 gain; bench ingest-throughput = 2894 msg/s, write
stages well-distributed, no fixable hotspot). DECISIONS (operator principle: don't keep/add complexity that
doesn't measurably help): ABANDON parallel-parse WIP (commit f73180222) — drop before PR. Index-deferral &
append-full-replace leads = NOT pursued (expected marginal vs real complexity; revisit only if a clean bench
shows a real gain). Already-taken wins remain: catch-up skip, read_amp=1.0, prefix-sharing dedup, FTS
bulk-suspend+atomic-rebuild, commit batching #2492, regex-insight skip flag.

## 5b. Perf threads (historical detail; incremental OK)
 (incremental OK; -25% counts. The obvious wins are ALREADY taken by existing engineering.)
- [RESOLVED-ALREADY-DONE] FTS triggers ARE suspended during bulk drain (ingest_batch/_core.py:746-752
  `_drain_ingest_results_sync`) and restored+rebuilt ATOMICALLY with the commit
  (`_commit_sync_ingest_side_effects`, #1242 — single atomic write, no silent drift). INVARIANT SAFE: FTS
  current at batch commit; transient-only inside the write; daemon health auto-restore is the backstop
  (daemon/health.py:_auto_restore_fts_triggers). NOT the win — already optimal + invariant-correct.
- [VERIFIED NEGATIVE — EMPIRICAL A/B (2026-06-29)] parse parallelism does NOT help. Same chunks, serial
  (POLYLOGUE_LIVE_FULL_INGEST_WORKERS default=1) vs workers=8: parse_s identical (chunk3 10.55→10.37, chunk5
  14.48→14.55, etc; pool engaged, 0→2 child procs, scales by batch blob size). Reason: the serial SQLite
  WRITE (`index_parsed_write`, 75-85% of parse_s) is the wall; provider_parse is tiny → overlap can't touch it.
  => the parallel-parse WIP (branch f73180222) is a DUD for this workload; deprioritize/abandon it. Write cost
  ∝ rows (messages+blocks) written = genuine.
- [NEXT PERF HYPOTHESIS — test via bench] Speed the WRITE itself: defer secondary B-tree index maintenance
  during catch-up (DROP idx_blocks_*/idx_messages_*, bulk-insert, rebuild once — same suspend-during-bulk
  pattern already proven for FTS; same atomic-restore invariant). Classic bulk-load win, UNTESTED. Measure with
  `devtools bench ingest-throughput` (clean A/B), NOT the noisy live daemon. Catch-up-only (live incremental
  keeps indexes). This is the real remaining write lever.
- [SUPERSEDED] parse‖write overlap. Machinery ALREADY EXISTS and is default-ON for the API
  path: `archive_ingest.py:parse_sources_archive` parses across a ProcessPool (`_parse_worker_count` default
  min(8,cpus-1)=8 here) explicitly to overlap parse CPU with the serial SQLite writer (its docstring says so).
  `ingest_batch/_core.py:_iter_ingest_results_chunk` also streams via submit+wait(FIRST_COMPLETED). BUT the live
  dev daemon shows **0 child processes** + serial parse_s→convergence_s chunk timings → the catch-up path is
  very likely NOT engaging the pool (parsing serially). So the parallel-parse "WIP" (branch f73180222) is
  largely ALREADY in archive_ingest.py — the remaining work is to VERIFY which ingest path the daemon watcher
  catch-up uses and ENABLE the existing pool there (or set POLYLOGUE_INGEST_PARSE_WORKERS), NOT write new
  parallelization. RESUME: trace daemon watcher → ingest call; confirm pool engagement; measure with workers on.
- [LOW] attachments per-row execute → executemany (rare: 1667 archive-wide; micro). User Q: are per-row
  executes "cheapest"? batching them is a micro-win at best (0.187s in profile, dominated by big sessions).
- Evidence: profiler (.agent/scratch/exp_profile_write.py) — moderate-session ingest CPU is small; cost scales
  with session size (the 33k-tool Codex monster = real work). executescript in profile = bootstrap artifact.
- Already-taken wins (NOT redundancy bugs): catch-up cursor skip, read_amp=1.0, prefix-sharing dedup, FTS
  bulk-suspend+atomic-rebuild, commit batching #2492, regex-insight skip flag.
- Ruled out (STATE §2): synchronous=OFF, page_size, autocheckpoint, bigger cache.

## 6. Construct-validity fixes [NEXT] (apply: default UNKNOWN, assert positive class only on positive evidence)
Detail: construct-validity-audit-2026-06-28.md (+ addendum).
- [DONE 2026-06-30 AUDITED] `material_origin=HUMAN_AUTHORED` is no longer a
  fall-through default. Current `classify_material_origin` explicitly returns
  `UNKNOWN` for plain `Role.USER + MessageType.MESSAGE`; `HUMAN_AUTHORED` is
  reserved for parser/hook paths with positive evidence. Added
  `test_plain_user_message_does_not_imply_human_authorship` as an executable
  regression guard. Live v18 archive still has historical/user-positive rows
  (`role=user`: 128 unknown, 55,044 human_authored, 23,375 explicit non-human
  lanes); a full reingest remains the data refresh path if older rows need to
  reflect the current classifier.
- [DONE 2026-06-30 AUDITED] old regex `_events_from_text` work-event miner is
  gone in current source. `polylogue/insights/transforms.py:_extract_events`
  builds command/test outcome events only from paired tool-result structure
  (`tool_result_exit_code` / `tool_result_is_error`) and skips `unknown`
  outcomes. `polylogue/insights/pathology.py:_detect_wasted_loops` consumes
  typed `ObservedEvent` metadata; focused tests in
  `tests/unit/insights/test_pathology.py` already pin that generic/file-read
  failures are ignored and legacy fallback requires a concrete command.
- [DONE 2026-06-30 FIXED] Topology edge type overclaim: current Codex parser
  already left `forked_from_id` as `branch_type=None` because the marker proves a
  parent but not fork-vs-resume. This slice fixed the remaining storage fallback:
  unclassified parent links now store generic `link_type=branch`, and projection
  only copies true `BranchType` values into `sessions.branch_type`. Added
  `test_codex_unclassified_parent_uses_generic_link`; active v18 audit found 75
  Codex `continuation` links and no `forked_from_id`/`thread_spawn` markers in
  their source files.
- [MED] Remaining construct-validity cleanup: `has_paste` still unions runtime
  marker/history/hash/heuristic evidence into one boolean; `paste_boundary`
  projection is now surfaced, but query/filter/display semantics still need the
  boolean collapse audited and likely split/renamed.
- [DONE 2026-06-30 FIXED] Codex stored token-lane semantics: writer already
  treats `total_token_usage` as session-global, subtracts cached input into a
  disjoint cache-read lane, and avoids per-model cumulative summing. The stale
  part was the provider-usage audit, which reconstructed expected rollups per
  `(session, model)`. Fixed `_expected_provider_model_rollups`, updated the
  coverage/docs wording to "session-global", and added
  `test_provider_usage_report_treats_codex_cumulative_as_session_global`. Live
  v18 report for `codex-session` now shows `stale_rollup_session_count=0`;
  residual 4 acquired-but-not-materialized Codex raw rows are convergence debt,
  not token-lane semantics drift.
- [DONE-ANALYSIS] pathology_report inherits regex invalidity → re-base on keystone (§2 dependency). Leave in place.

## 7. Correctness to verify [NEXT]
- [DONE 2026-06-30 VERIFIED] `session_tag_rollups` does not miss the tested
  late/backfilled provider-day move in current source. Targeted sync rebuild
  already records previous profile groups and refreshes
  `previous_profile_groups | refreshed_profile_groups`; async incremental
  refresh likewise carries both old and new `profile_provider_day` groups.
  Added `test_targeted_session_insight_rebuild_moves_tag_rollup_between_days`
  to prove a session moved from 2026-04-03 to 2026-04-01 removes the old bucket
  and leaves only the new bucket. No implementation fix was needed.

## 8. Ingest robustness [NEXT]
- [NEXT] 139 retry-pending = antigravity(116) + gemini-cli(23) parser failures (failure_count=1). claude/codex
  clean. Investigate parser errors (converge.log / re-run a failing file).
- [DONE 2026-06-30 FIXED] Provider-usage `acquired_not_materialized` false
  debt for 4 Codex raw rows: `ops debt list --kind raw-materialization`
  correctly classified these as parsed metadata-only/non-session artifacts, but
  the provider-usage report counted every raw row without `sessions.raw_id` as
  acquired-not-materialized. `storage/usage.py` now excludes parsed Codex
  `session_meta`-only raw blobs from actionable acquired-not-materialized counts
  and samples. Live v18 provider-usage report for `codex-session` now shows
  `acquired_not_materialized_count=0`; coverage is partial only because some
  materialized sessions lack provider usage event rows.

## 9. Architecture cleanup (STATE §3) [DESIGN]
- [READY TO REMOVE — verdict: unnecessary cargo-cult] FTS auto-restore + #1613 drift machinery
  (daemon/health.py `_auto_restore_fts_triggers`, `_check_fts_trigger_drift_fast`, `_active_bulk_ingest_attempt`,
  `_BULK_ATTEMPT_FRESHNESS_S`; `health.fts_auto_restore` config; journald backend ref). CLINCHER: the health
  check uses a SEPARATE connection → under WAL it can never observe the writer's uncommitted DROP TRIGGER;
  committed state always has triggers; post-SIGKILL a fresh conn rolls back to triggers-present (SQLite atomic
  DDL, trusted). So the guarded state is unobservable+unreachable. #1613 was a symptom-patch; #1242 (atomic
  restore in the single ingest transaction) made it unreachable at the root → now vestigial. KEEP the
  full_parse/full_worker_wait phase tracking (used by catchup_status/progress, not just this). Blast radius:
  health.py, config.py, notification_backends/journald.py; tests test_health_contract.py + test_health_check_paths.py;
  docs internals.md/configuration.md/daemon.md. Fold into a cleanup pass.
- [NEXT] convergence_stages.py dead single-file `else` branches (~half of 1490 lines post-#1787); vestigial
  run_stages / INGEST_STAGE_SEQUENCES pre-daemon vocabulary.
- [DESIGN] cross-tier non-atomicity (index.db + source.db two-connection commit → raw-without-index half-state
  + its repair machinery). Pre-existing.

## 10. Operator vision / rawlog-sourced (bigger, longer-horizon) [DESIGN]
Source: rawlog 2026-06-18..28 (esp 22:14, 22:53, 21:55). These are the ambient-context arc.
- [DESIGN] DSL annotation overlay: agents emit structured plans/goals/todos/observations/conclusions inline in
  prose; polylogue parses → reflects back queryably → optional ambient context injection. "More ergonomic than
  tool calls; doesn't interrupt flow." Pairs with §3 `with`-projection + Workflows.
- [DESIGN] Ambient past-session access WITH noise-filtering (agents "absurdly hobbled re: history"); injection
  content AGENT-TUNABLE over time; dogfood by cutting sessions + unprompted recovery. (Operator: this is the
  intense-dogfooding priority, but the self-recovery LOOP is days-out/operator-orchestrated.)
- [DESIGN] polylogue wrap/understand CLAUDE.md + skills as sources (query/edit/version them; maybe obsolete CLAUDE.md).
- [DESIGN] vendor agent skill/protocol definitions into polylogue/sinex as optimizable sources, tight-loop dogfood.
- [IDEA] 'dreaming' — background processing of agent memory.

## 11. Deferred / external
- [LOW — dev-loop ergonomics] MCP dev-loop staleness: the loaded `mcp__polylogue__*` tools are pinned to
  PROD archive (default root) + session-start code → can't loop on the MCP surface during dev. Dogfood via CLI/
  API/sqlite on the dev archive instead (current code + dev data). Fix (enabler for agent-ambient-context vision
  §10): MCP resolves POLYLOGUE_ARCHIVE_ROOT at request-time + a dev MCP instance restarted on current code
  pointed at the dev archive. Until then MCP dogfooding only at session boundaries.
- [BATCH-REINGEST] Definitive v16+ reingest into final shape — AFTER §2 + §3 schema epoch complete (reingest ONCE).
- [NEXT] Import 3.8G exports (/realm/data/exports/chatlog/raw) once steady.
- [NEXT] Verify recovered parallel-parse WIP (branch f73180222) — see §5.
- Issue parts-bin (GH): #2482 topic-pack(=§3/§4 P0), #2467 lineage umbrella, #2468 attachments-not-preserved
  (fold into #2467), #2391 ingest-latency/WAL(=§5), #1807 evidence-cockpit epic, #2480/#2316 forensics,
  #2472 Codex per-model partition residual, #2470 fork read-paths.

## Sequencing intuition (not rigid)
Parallel-safe NOW (additive, no reingest): §3 DSL `with`+collapse, §4 recovery-digest quality + hook fix,
§5 FTS-suspension + parse‖write perf, §8 robustness, §7 correctness verify.
Batch into ONE schema epoch + reingest: §2 insights cut + §6 construct-validity (material_origin, has_paste,
edge-types) + §2 run-projection rebuild + §4 execution-projection rebuild.
Longer-horizon design: §10 ambient/annotation vision, §9 convergence cleanup.
