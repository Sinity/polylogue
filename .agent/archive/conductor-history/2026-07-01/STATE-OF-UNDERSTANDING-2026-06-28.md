---
created: 2026-06-28
purpose: Consolidated, verified understanding — perf, schema, architecture, and especially "insights"
status: active
---

# Polylogue — Established Understanding (2026-06-28)

Everything below is evidence-backed (code read + measured), not assumed. Scratch
experiments live in `.agent/scratch/exp_*.py`, `profile_ingest.py`,
`insights-dissection-2026-06-28.md`, `convergence-architecture-analysis-2026-06-28.md`.

## 2026-06-30 Baseline Correction

The 16K-session archive scale mentioned below is the old pre-dedup/stale-index
shape. Current active archive probe: 2,397 raw rows, 2,390 indexed sessions,
159,956 indexed messages. Preserve the old numbers only as historical
stress-test data; use the current deduplicated baseline for new estimates.

## 1. Shipped this session (merged to master)
12 PRs: #2469 lineage+cost, #2484 credit-5x, #2485 fork read-paths (#2470),
#2486 server_tool_use capture, #2487 cache-rate guard, #2488 lineage memoize +
expr sortkey index (schema v15), #2489 Codex per-session cumulative (#2472),
#2490 bench ingest-throughput, #2491 ingest instrument (cpu/mem/io/stage),
#2492 commit batching, (+#2466/#2464 earlier). Issue backlog triaged 39→37.
WIP recovered (branch `feature/dogfood/parallel-parse`, commit f73180222):
parallel parse across processes — UNVERIFIED (no tests run, agent hit quota).

## 2. Performance — measured facts
- **Method**: a representative ingest benchmark MUST apply `WRITE_CONNECTION_PROFILE`
  (WAL). Raw `sqlite3.connect` shows commit at 41% — a rollback-journal fsync
  artifact; WAL is 2.5× faster.
- **Ingest is I/O-wait bound** (cpu_utilization ~0.60), not CPU/parallel. Root:
  per-session commits — `write_parsed_session_to_archive` AND
  `write_source_raw_session` each `with conn:` = 2 fsync/session.
- **Commit batching (#2492, merged)**: work-based, COMMIT_BATCH_MESSAGE_THRESHOLD=8000
  (env `POLYLOGUE_INGEST_COMMIT_BATCH_MESSAGES`, ≤0=per-session). Real path: 355→1712
  msg/s (~4.8×), 375MB→22MB written (~16.8×). Message-count beat session-count
  (uneven WAL) and one-shot (slower). Atomic at session boundaries; rollback covers batch.
- **THE DOMINANT COST — deferred insight pass = ~9× the write** (write 1.3s vs
  insights 11.8s for 17.2K msgs; 57ms/session; ≈15 min on the real 16K-session
  archive). 76% is `re.Pattern.search` (292K calls) from
  `insights/transforms.py:1810 _events_from_text` regex prose-mining, invoked by
  `compile_recovery_digest()` per session in `rebuild.py:615`. **The deferral HID
  this from the write-path instrument** — I optimized the 10% while the 90% sat behind
  the "deferred" label.
- **Ruled out with data**: synchronous=OFF (1.45× but corruption risk; batching better),
  page_size (4096 optimal, bigger=worse), autocheckpoint tuning (irrelevant once WAL~14MB),
  bigger cache (hurt). WAL 160MB cap (#1614) = steady-state safety, NOT a re-ingest bottleneck.
- **Parallelism**: parse = 44% of ingest, CPU-bound, embarrassingly parallel per-file.
  Gates validated: pickle 119K msg/s (6× cheaper than parse); blob_store write_from_path
  atomic (tempfile+os.replace, content-addressed → process-safe); writes order-independent.
  Ceiling ~1.77×+ (higher post-batching). WIP recovered, needs verification.

## 3. Architecture / schema — verdicts
- **5-tier split** (source/index/embeddings/user/ops) by durability class — EARNED. Don't flatten.
- **Convergence stages** mostly earned (crash-safety, #1498 I/O budget). Accidental complexity =
  dual-path residue: dead single-file `else` branches in convergence_stages.py (post-#1787,
  ~half of 1490 lines) + vestigial `run_stages`/`INGEST_STAGE_SEQUENCES` pre-daemon vocabulary.
  (cleanup agent didn't run — not yet done.)
- **Cross-tier non-atomicity**: `archive.commit()` commits index.db then source.db as TWO
  connections — crash between = raw-without-index half-state, "handled" by raw-materialization-debt
  detection. Pre-existing (not from batching). This half-state + its repair machinery IS the
  complexity; it exists because raw & derived are split across files/passes.

## 4. INSIGHTS — the core finding (the operator was right on every axis)
"insights" is an **incoherent category** = three different products under ONE deferred stage +
ONE `materializer_version` (`_PER_SESSION_INSIGHT_TABLES`, rebuild.py):

**A. Cheap deterministic stats** — `session_profiles` (evidence: counts/durations/tokens/cost),
`session_latency_profiles`, `session_tag_rollups`. Pure arithmetic over a session's own rows.
Sub-ms. **Doubly stored**: `sessions` ALREADY has ~18 derived stat columns (message_count,
word_count, role counts, tool_use_count, ... via `_refresh_session_counts` inline) AND there's a
separate deferred `session_profiles` table. Pointless schema elaboration + double write + read join.
→ Should be inline columns on `sessions`, one home, atomic. Recompute via in-place UPDATE.

**B. Structural lineage** — `threads`/`thread_sessions`. Topology, not "insight". Cheap. Keep, rename.

**C. Expensive heuristic prose-mining** — `session_work_events`, `session_phases`, and the
`session_runs`/`session_observed_events`/`session_context_snapshots` run-projection trio.
- `_events_from_text` (transforms.py:1810): ~16 regexes line-by-line over EVERY message guessing
  dev-workflow events (PR merged, tests passed, issue closed, review acted-on).
- **Construct-invalid (PROVEN)**: "should we merge PR #123" → recorded as `pr_merged: PR #123 merged`
  (false fact). Cannot distinguish action vs intention/question/quote/negation. review_kind decided
  by substring sentiment ("fixed"/"addressed"). English+GitHub-specific → noise on ChatGPT/Gemini/non-English.
- **Misleading**: work_event `confidence` (0.4–0.75) are author priors, stored in a `CHECK BETWEEN 0
  AND 1` column reading like calibrated model output.
- **Dead columns**: `session_phases.phase_type` is constant "phase"; `session_phases.confidence`
  always 0.0.
- **The run-projection trio IS READ** (CORRECTION to dissection's "unread"): `run`/`observed-event`/
  `context-snapshot` are SQL-backed terminal query units in the `find` language
  (`archive/query/expression.py`). Materializing them via `compile_recovery_digest` per session is
  the 9× cost; the recovery/postmortem MCP tools SEPARATELY recompute the digest at query time. So:
  redundant compute path, but the materialized tables do serve the query-unit surface — cannot delete
  outright. Fix = the regex root (source from structure), or make query-units recompute lazily, or
  drop the units if low-value.

## 5. Capture gap (keystone)
`blocks` stores `tool_name`, `tool_input` (structured JSON, generated `tool_command`/`tool_path`) but
the tool RESULT only as `text` — **no structured `is_error`/exit/outcome column**. Claude/Codex source
JSONL carry `is_error`/exit; dropped at parse. So in-session outcomes ("tests passed", "command
succeeded") cannot be READ — must be inferred (the regex). Fixing this enables reading outcomes from
structured tool calls+results instead of prose-guessing.

## 6. Correctness concerns to verify
- `session_tag_rollups` uses `input_high_water_mark` → likely MISSES late/backfilled data below the
  mark (import a session into a past bucket_day → stale rollup). Verify + fix (dirty-tracking/content-hash).
- day/week summaries are computed ON-READ (not materialized) → correctly reflect late data. Good.

## 7. External truth boundary
PR-merged / issue-closed / CI-passed are authoritative in GitHub/CI, NOT a session transcript.
polylogue should store only what the session OBSERVED (structured tool calls+results). Cross-source
truth = Lynchpin's job (it has get_github_pr/issue, git). Don't make polylogue guess external state from prose.

## 8. Principled analytics we DON'T compute but could (all from structured data, deterministic)
Structured outcome events (from tool calls+results, w/ exit status); tool success/failure/retry rates +
error taxonomy; timing/latency (turn gaps, tool durations, stuck/idle, time-to-completion); loop/
abandonment detection (repeated identical calls, repeated errors, mid-task end); cost-efficiency
($/outcome, cache-hit ratio, trends); conversation structure (correction/restart/interrupt, compaction
pressure); cross-session trends (model evolution, productivity, fork/resume/abandon, topic clustering via
existing embeddings).

## 9. Redesign direction (needs operator sign-off on sequencing)
GUIDING STANCE (operator, 2026-06-28): the current heuristics (work_events/phases) are simply BAD —
misleading and confusing — so **CUT them**, don't dress them up. Heuristics aren't banned forever, but
we add a *good* one deliberately later if we find one. Critically: do **NOT** build confidence/provenance/
uncertainty machinery now — once the bad heuristics are gone, everything remaining is DETERMINISTIC, so
there is nothing probabilistic to tag. No premature uncertainty layer (YAGNI).

1. **Capture gap first** — persist structured tool-result outcomes (is_error/exit). Keystone; enables reading.
2. **Inline deterministic stats** onto `sessions`; drop `session_profiles`-as-separate-table; collapse its
   materialization tracking + deferred pass + read join. Atomic sessions.
3. **CUT the regex prose-miner** (`_events_from_text` / work_events / dead phase cols). Where a real outcome
   is wanted, READ it from structured tool-calls+results (deterministic, cheap, gated to tool-use sessions) —
   not as a heuristic, as a fact derived from structure. If structure can't give it, leave it out for now.
4. **Run-projection**: it's read by `find` query units but built on the regex digest. Either rebuild it from
   structured data (deterministic) if the units are worth keeping, or drop the units if low-value. No
   regex-digest materialization either way.
5. **(removed)** — no confidence/provenance tier. Everything kept is deterministic; reintroduce heuristics
   (and only then their handling) deliberately, later, if a genuinely valuable one appears.
6. **Cross-source truth → Lynchpin.** **Embeddings stay deferred** (genuinely expensive/external — the one
   thing the convergence machinery is actually for).
Net: deletes the ~9× regex pass, removes double-storage + most of live_convergence_debt (sessions become
atomic), less schema, fewer concepts, faster — simultaneously.

## 10. Construct-validity audit (verdict — full table in construct-validity-audit-2026-06-28.md)
ROOT PATTERN: validity fails wherever a **text-shape heuristic or an absence-of-evidence is promoted to an
asserted positive fact** without positive/speaker gating. Two anti-patterns: "regex-match → fact" and
"default-to-a-positive-class".

HIGH (construct-invalid, load-bearing):
- **work_events miner** — regex text-shape → asserted "what happened"; no speaker gating. (insights/transforms.py)
- **`material_origin=HUMAN_AUTHORED` / `authored_user_message_count`** — it's the FALL-THROUGH DEFAULT after a
  hardcoded robot-marker allowlist fails (`archive/message/artifacts.py:85-112,141`). "human authored" = "no known
  robot marker matched" = absence-of-evidence, NOT positive evidence. Overcounts human authorship. This was the
  supposed FIX for the role=user conflation, and it's invalid the same way — so it can't even serve as the gate for
  other constructs. Propagates into any human-vs-AI attribution analytic (incl. Lynchpin's ai_assist_density).

MEDIUM (conflation / proxy):
- **TopologyEdgeType** advertises 7 types but Codex collapses fork+resume→FORK; RESUME/REPAIRED never assigned;
  BranchType lacks RESUME. Over-claims distinctions it doesn't make.
- **branch_type=CONTINUATION** guessed from "2nd embedded session_meta exists" (codex.py) — proxy.
- **has_paste** unions the real `[Pasted text #N]` marker with shape proxies (len>4000 or >70% code-fence) — boolean
  conflates ground-truth with "looks big".
- **Codex token lanes** — input passed through incl. cached while cache_read is a separate additive lane (double-count
  risk; honest `uncached_input_tokens` normalizer exists). (cost path fixed #2489; stored lane semantics still suspect.)
LOW: timestamp sentinels (1970-01-01, sub-86400 epoch → None).

SOUND (record what they claim — CORRECTIONS to my earlier over-claims):
- attachment honesty: #2468 FIXED this session — `acquisition_status` + nullable `blob_hash`; no longer fabricates.
  (My "attachment content_hash is a lie" was the pre-fix state.)
- **CostBasis/CostUnavailableReason taxonomy is SOUND** — honest about *which basis* (api_equivalent vs
  subscription_equivalent) rather than presenting one "cost". (So cost is honest where the taxonomy is used; risk is
  only an unlabeled surface.)
- message content_hash + attachment metadata subhash; identity_law IDs; Origin + public read filters;
  logical_session_count; matched (non-default) MaterialOrigin lanes; TopologyEdgeRecord storage.
- session_phases dead `phase_type`/`confidence` — likely already removed in #1743 (stale note).

DESIGN PRINCIPLE: default to UNKNOWN/UNCLASSIFIED; assert a positive class ONLY on positive evidence. Record
structure, name honestly. Lynchpin already consumes polylogue's inference layer WITH documented caveats →
fixing these improves Lynchpin's analytics (esp. attribution) for free.

## 11. LIVE EXECUTION STATE (dev steady-state reingest — 2026-06-28 ~17:45)
MANDATE now active (goal hook): dogfood-as-feedback-engine; prod down; manual daemon/devloop; reach
steady-state dev archive; architecturally-right; artifacts on real data; move fast/batch/parallel; no
ceremony (disregard git/CI/issue-boundary cleanliness — everything is a source of info). Goal text:
`memory/project_standing_goal_2026_06_28.md`. Demo ladder: `/realm/inbox/download/demo-and-agent-plan.md`.

- **prod polylogued.service STOPPED** (was v11; code is v15 → incompatible anyway). Prod DBs irrelevant
  (operator: do whatever; datasets that matter = ~/.claude, ~/.codex, + exports).
- **DEV archive**: `/realm/tmp/polylogue-dev/archive` (v15, isolated via XDG_DATA_HOME=/realm/tmp/polylogue-dev/xdg
  + POLYLOGUE_ARCHIVE_ROOT). Convergence daemon running: `nix develop --command bash -c 'export XDG_DATA_HOME=... POLYLOGUE_ARCHIVE_ROOT=...; polylogued run --no-api --no-browser-capture'`,
  log `/realm/tmp/polylogue-dev/converge.log`. It does a catch-up BACKFILL of existing files (watcher
  `_catch_up`), then watches. ~4.7 sessions/s; `slow_batch` warnings (likely the 9× insight pass). Backlog
  13,050 claude+codex files + 3.8G exports (`/realm/data/exports/chatlog/raw`, one-shot via `polylogue import`).
  ETA steady state ~1-2h.
- To RESUME/CHECK reingest: `sqlite3 /realm/tmp/polylogue-dev/archive/index.db "SELECT COUNT(*) FROM sessions"`;
  daemon pid via `pgrep -af "polylogued run"`; if dead, relaunch the command above.

## 12. SESSION 2 PROGRESS (2026-06-28 ~22:45, after session-limit reset)
- **Daemon relaunched** detached (setsid, survives harness reaping) with SKIP_RUN_PROJECTION=1 +
  COMMIT_BATCH_MESSAGES=8000. Confirmed the skip flag WORKS: insights stage 13.5s→0.95s, convergence
  14.6s→1.0s. Parse (`index_parsed_write` ~5s) is now the bottleneck → validates parallel-parse WIP as
  next perf lever. Reingest climbing well (954→2009+ sessions, chunk 63/526).
- **SHIPPED: `polylogue status` live ingest-workload surface** (commit 7f9ac2ac5 on branch
  feature/dogfood/parallel-parse). `_ops_workload_status()` reads ops.db (ingest_attempts/cursor/
  convergence_debt) read-only → throughput, in-flight phase+heartbeat, coverage, debt. Direct text+JSON
  paths render it; header now "daemon ingesting" when a running attempt has fresh heartbeat (was the
  misleading "daemon not running" under --no-api). No new tables — projection over existing substrate.
  Also fixed pre-existing facade-route-catalog drift (4 pathology/postmortem/portfolio methods missing).
  31 status tests pass, mypy clean, dogfooded live.
- **FINDING (ingest robustness)**: 139 retry-pending files = antigravity(116)+gemini-cli(23), failure_count=1.
  claude/codex ingest clean. The antigravity/gemini-cli parsers fail on some files — non-urgent (not the
  datasets that matter) but real. Investigate later.
- **SHIPPED: KEYSTONE — structured tool-result capture (schema v16)** (commit 19490eadd). blocks gains
  `tool_result_is_error`(0/1 null)+`tool_result_exit_code`; `actions` view exposes is_error/exit_code.
  Claude captures seg `is_error`; Codex parses `metadata.exit_code`→is_error. NULL=unknown (no fabricated
  positive). E2E proven on real data (.agent/scratch/exp_keystone_e2e.py): 106 is_error=1, exit codes
  0/1/2/22/123/124, actions view per-tool (shell 101err/602ok). mypy+449 parser tests+render-all-check green.
  internals.md v16 note added. THIS unblocks reading outcomes from structure (kills need for regex miner).
- **OPERATOR FLAG + CONFIRMED (construct validity)**: pathology_report (insights/pathology.py) detects
  wasted_loop/missed_review/stale_context over RunProjection, whose test_failed/check_failed events come
  from the SAME regex `_events_from_text` → pathology inherits the invalidity via its INPUT. Concept is
  sound + SALVAGEABLE: re-base run-projection on the keystone's structured outcomes (= redesign step 4).
  DECISION: leave pathology/postmortem in place for now (downstream of an already-targeted bad class, not
  a new one). Full note: construct-validity-audit addendum.
- **KEYSTONE map (reference, now implemented)**: blocks DDL + INDEX_SCHEMA_VERSION 15→16
  (storage/sqlite/archive_tiers/index.py:36,169-194); ParsedContentBlock (sources/parsers/base_models.py:51-77);
  BlockRecord (storage/runtime/archive/records.py:92-120); ArchiveBlockRow + INSERT (storage/sqlite/archive_tiers/write.py:42-54,1403-1444);
  Claude capture `seg.get("is_error")` DROPPED at base_support.py:51-69; Codex function_call_output
  status/error at codex.py:344-365; consumer regex `_tool_status` transforms.py:2164.
  SEQUENCING DECISION: keystone is additive (safe). Defer the full dev v16 reingest until the insight
  CUT (regex miner removal) is also ready, so reingest ONCE into the final shape, not repeatedly.

- **SHIPPED: recovery digest noise fix** (commit cdf3c1eb1): workdirs collapsed (150+ subagent worktrees
  → primary dirs + count). Dogfood found `read --view recovery` is noise-dominated + built on construct-invalid
  run-projection — full findings: recovery-digest-dogfood-2026-06-28.md.
- **ARCHITECTURE FINDING: context-pack/compile_context are GRATUITOUS** (operator-flagged). 3 parallel surfaces
  (context/compiler.py ContextSpec/Image trio; mcp/context_pack.py 15 DTOs; cli read --view context-pack).
  compile_context seed_query does search(limit=1) — degenerate vs general find. Collapse path: add GENERAL read
  modifiers --max-tokens (token-bounded accumulation + omissions), multi --view, --include-assertions → packs
  become `find <seed> then read --view a,b --max-tokens N`. Full design+sequencing: context-pack-gratuitous-2026-06-28.md.
  NEEDS operator sign-off (scoped refactor). Same family as insights redesign (algebra-not-hardcoded).
- **OPERATOR STEERS (session 2)**: (a) self-recovery proof is days-out/operator-orchestrated, DON'T build the
  SessionStart loop now; (b) make general capability more expressive to OBVIATE bespoke surfaces; (c) reingest
  perf bottleneck = index_parsed_write (SQLite write: messages/fts/attachments), single-core CPU-bound, NOT
  parse — so parallel-parse WIP has limited upside (write serializes).

NEXT (parts-bin, ordered by value; most need the populated archive):
1. **`polylogue status` daemon-workload surface** (operator idea, independent of populated archive,
   dogfoodable on the running reingest NOW): surface live convergence state — backlog remaining,
   throughput (files/s, msg/s), current stage, debt, ETA — from ops.db `live_ingest_attempt`/
   `live_convergence_debt` (data already exists; capability emerges from substrate). I need this for the reingest.
2. **Insights redesign** (the construct-validity work, §4/9/10) — cut regex miner, inline deterministic
   stats onto sessions, fix material_origin-default-to-human, structured tool-result capture (keystone).
   This also fixes the slow_batch/9× reingest tax. Doesn't need populated archive.
3. **P1 self-context-pack / session-continuity** — once archive populated: use Polylogue to pull prior-session
   prose so a fresh session continues losslessly (the indefinite-continuity goal). Generalize to `topic-pack --topic X` (P0).
4. Import exports; verify recovered parallel-parse (branch feature/dogfood/parallel-parse, unverified).
