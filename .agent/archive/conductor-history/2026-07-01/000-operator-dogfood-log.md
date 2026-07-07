---
created: "2026-06-27T16:20:00+02:00"
purpose: "Long-running operator-dogfood hardening log for Polylogue"
status: active
project: polylogue
branch: feature/operator-dogfood-hardening
---

# Operator Dogfood & Hardening — Polylogue

## Current Baseline Correction

2026-06-30: the old 16K-session / 5.7M-message archive figures below are
historical pre-dedup numbers. They counted fork/continuation replay that was
later collapsed by lineage/fork-dedup work. Current active archive probe:
2,397 raw rows, 2,390 indexed sessions, 159,956 indexed messages, latest raw
acquisition 2026-06-30T03:04:58.856+02:00. Use this deduplicated baseline for
new status reports and demos.

## Charter (my mirror of the sinex operator mandate)

Operate as the **operator-side owner** of Polylogue. Independent of the issue
set, on this long-term branch, dynamically find and fix what is architecturally
broken, incoherent, under-observable, or rough in UX. **Dogfood relentlessly**
against the real archive:
drive every surface as a real operator would — query-first CLI, daemon HTTP
reader, MCP tools, web reader, TUI — record what actually works vs. what is
broken / confusing / slow, then fix the breakage. Bias to nimble, verified,
source-material-respecting changes. Respect product intent (chatlog ≠ spec).

Verification posture: from the **operator's perspective**. A change is "done"
when I have driven the actual surface against the real archive and seen it
behave well — not when a unit test passes in isolation.

### Thrusts (dynamic, not a fixed checklist)

1. **Source-material defrag / archive hygiene** — blob GC (#818),
   raw-materialization debt, FTS trigger drift, WAL hygiene, re-ingest/reset
   ergonomics, split-tier archive-layout coherence. Make the archive
   self-healing and legible.
2. **Observability** — daemon `/metrics`, health tiers, convergence/debt
   visibility, catch-up progress legibility. The operator must be able to SEE
   what the daemon is doing and whether it is healthy and converged.
3. **UX** — query-first CLI, read views, MCP ergonomics, error messages, web
   reader. Ensure surfaces function and are well-designed for a cold operator.
4. **Robustness / responsiveness** — read-path tuning (busy-timeout, count
   path), scale-hardening (#2465), run_ref debt healing, ingest catch-up
   speed & resource efficiency.
5. **Deploy + dogfood loop** — keep it deployed and working well for the
   operator on sinnix-prime.

## Operating Log (newest first)

### 2026-06-27 — session start

**Deploy in flight.** flake pin bumped 15f4f21b → 2eee22a9f (#2464 run_ref
upsert + #2466 WAL-bound rebuild). Toplevel build running background
(`bbm1j8ij6`, heavy sinex Rust). Will `switch` once cached.

**Daemon state.** polylogued active 1h24m, 788M RSS, healthy. Deployed binary
is still pre-#2464 (969751e) → emitting recoverable run_ref UNIQUE debt on
subagent ingests. Deploy fixes this.

**Read-path audit triage** (verify-then-fix; the prior audit over-ranked all
as CRITICAL):
- Item "health MEDIUM on every poll" → **FALSE POSITIVE.** Default monitored
  path `/api/health` does only `SELECT 1 FROM sqlite_master` + `stat()`. Only
  the explicit CI-facing `/api/health/check` runs FAST+MEDIUM tiers.
- Item "list count materializes up to 1M rows"
  (`archive/query/archive_execution.py:471`) → **REAL but bounded.**
  Post-filtered / semantic counts can't use SQL `COUNT` (computed in Python),
  so they materialize matching summaries to count them (default_limit=1M cap).
  At 16K sessions this is wasteful (full hydration just to take `len()`) but
  not catastrophic; the 1M cap is a guardrail, not a bug. Candidate: an
  ID-only count variant that skips summary hydration.
- Item "daemon read busy_timeout = 0.25s" (`daemon/http.py:82`) → **plausibly
  real.** Reads are `mode=ro` WAL with `busy_timeout=250ms`. WAL readers don't
  block writers except in a narrow checkpoint/WAL-index window; at 28.8GB with
  periodic TRUNCATE checkpoints that window can spuriously fail interactive
  reads with "database is locked". Needs live lock-failure evidence before
  changing the constant.

### 2026-06-27 — embeddings dogfood

**Real archive reachable** via `POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue`
(daemon uses XDG default; my devshell inherited the cloud `/tmp/polylogue-archive`
sandbox override — itself a minor dev-UX trap). index.db 36 GiB, ops.db 481 MiB
(large telemetry), embeddings.db 60 KiB (was disabled).

**Cost reality:** full corpus = 5.72M msgs ≈ 2.86B tokens ≈ **$286** at voyage-4
$0.10/1M. Operator clarified: **no billing configured, 200M free tokens/model.**
So full embed is impossible on free tier (14× over); Voyage just rejects when
free exhausted (no charge). Realistic target: newest ~200M tokens (~400K msgs,
most recent ~1K+ sessions).

**BUG FOUND + FIXED — pending-embedding windows were oldest-first.**
Two near-duplicate selection fns in `storage/embeddings/materialization.py`:
- `select_pending_session_window` (ordered `updated_at_ms` ASC) — used by
  `polylogue embed`, pipeline, api.
- `select_pending_archive_session_window` (ordered `sort_key_ms` ASC) — used by
  the operator-facing `ops embed backfill`.
Both oldest-first. On a partial budget that wastes the entire window on the
oldest sessions, which are **empty stubs** (sort_key_ms=0, 0–3 msgs).
Fixed both to newest-first (`DESC`, nulls last). Verified: backfill now selects
real recent sessions (today's claude-code sessions, 841/453/124 msgs) and
embeds them via real Voyage calls. 59 affected tests pass.
*Architectural note:* the two fns are duplicated logic differing only in sort
column (`updated_at_ms` vs `sort_key_ms`) + a where-clause term — a unification
candidate (follow-up, not done here to keep the fix scoped).

**Minor bug (noted, not fixed):** `ops embed backfill` preflight under-counts
window messages (showed "3 messages" for a 1418-msg window) — it reads stale
`sessions.message_count`. The **run-time cost cap is unaffected** (it sums real
`embedded_message_count`), so budget safety holds; only the pre-run estimate is
wrong. Candidate fix: preflight should count real message rows for the window.

**Minor bug (noted):** `ops embed enable` raises raw `OSError: Read-only file
system` on this HM-managed deployment (user `polylogue.toml` is a read-only
nix-store symlink). Activation flow #1217 assumes a writable user config.
Backfill works off env key regardless. Candidate: detect read-only config and
print actionable guidance instead of a raw traceback.

**Throughput:** ~13 msg/s (BATCH_SIZE=128 already at Voyage's per-request limit;
~10s/batch is Voyage latency). Full free budget ≈ 8.5h. Launching bounded
background backfill (`--max-cost-usd 20` ≈ 200M tokens), resumable.

**Next:** finish deploy; dogfood remaining surfaces (daemon HTTP, MCP, web,
TUI); backup embeddings.db after backfill.

### 2026-06-27 — daemon read responsiveness under load

Daemon web reader on :8766 (8765 also bound; :7778 is unrelated ASR server).
Probed under concurrent backfill IO, then after it went network-bound:

| Endpoint | Under IO load | Quiescent |
|----------|--------------|-----------|
| /healthz/live,ready, /api/health, /api/status | <10ms | <10ms |
| /metrics | 5.0s | 0.28s |
| /api/facets | 19.5s | 1.18s |

**Item 1 (busy_timeout=0.25s) RESOLVED as non-issue.** Under heavy concurrent
IO, reads got slow (1–20s) but never *failed*/timed-out — exactly WAL semantics
(readers snapshot, don't block on the writer). The 0.25s was about write-lock
contention, which WAL readers avoid. Leave the constant; document why.

**Real systemic finding:** daemon reads degrade 16–20× under concurrent
heavy IO (disk-bandwidth saturation on the shared 36 GiB index.db). Trigger
here was the backfill preflight's full scan, but the daemon does similar scans
during convergence/ingest. `idx_messages_role` exists and role-count is a
covering-index scan (v11 confirmed), so facets' baseline 1.18s is structural
fan-out, not a missing index. Mitigation space: cap/nice daemon background IO;
trim preflight's per-session COUNT scan. Logged, not yet fixed.

### 2026-06-27 — CLI read robustness + enable UX (committed)

**BUG FOUND + FIXED — `polylogue find` intermittently "database is locked".**
The primary query surface hard-failed during daemon ingest. Root cause: global
`READ_DB_TIMEOUT = 1s` (connection_profile.py) — the busy_timeout for every read
connection. WAL readers don't block on writers *except* the brief
commit+TRUNCATE-checkpoint window, which on a 36 GiB archive can exceed 1s →
hard "database is locked" instead of a short wait. Confirmed the read path is
mode=ro (doesn't block on BEGIN EXCLUSIVE), so the failure is specifically the
checkpoint window. Fixed: READ_DB_TIMEOUT 1→5 (still ≪ 30s writer timeout).
Commit 19df2f39b. Bandwidth-bound reads unaffected (they already succeed slowly).

**BUG FOUND + FIXED — `ops embed enable` raw OSError on read-only config.**
Translated to a click.ClickException with three actionable paths. Commit
93d56393e.

**Backfill running**: newest 769 sessions (~400K msgs ≈ free budget),
network-bound, ~6h. Pending backlog visibly shrinking (16,410→16,294 sessions).

**Branch state** (feature/operator-dogfood-hardening, on origin/master 2eee22a9f):
- 2fa82c92c perf(embeddings): newest-first pending windows
- 19df2f39b fix(storage): read busy_timeout 1s→5s
- 93d56393e fix(cli): read-only embedding-config guidance

**Deferred deliberately:** the #2464+#2466 deploy (needs a ~30min sinex Rust
build that would degrade the live host's daemon reads — measured 16–20×; not
urgent since run_ref debt is recoverable). Do it in a quiet window.

**Still open / candidates:** preflight stale message_count; daemon
read-degradation-under-IO (nice/cap background IO); the two duplicated
pending-window fns (unify); MCP/web-reader/TUI surfaces not yet dogfooded.

### 2026-06-27 — MCP surface dogfood

Drove the agent-facing MCP tools (stats, readiness_check, archive_debt) — the
surface other agents actually consume.

**FINDING #1 (significant) — MCP server silently serves an EMPTY archive.**
stats → total_sessions:0, db_size_mb:0.6; readiness archive_root =
`/tmp/polylogue-archive`. The MCP server inherited
`POLYLOGUE_ARCHIVE_ROOT=/tmp/polylogue-archive` from the repo's
`.claude/settings.json` (cloud-lane default) because Claude Code spawned it
while cwd=the polylogue repo. So an operator running `claude`/`codex` *inside
the repo* gets an MCP pointed at an empty archive — and every tool returns
empty results with **no error**. Worse: `readiness_check` reports an empty
archive as all-green ("messages indexed: 0" → ok). Silent-wrong-answer trap.
Candidate fixes (design, needs operator input): (a) don't let the cloud
`.claude/settings.json` archive override apply to local in-repo sessions;
(b) readiness should flag "archive_root under /tmp + 0 sessions" as a warning,
not ok; (c) MCP server should resolve the canonical archive independent of the
sandbox env. Not a one-liner — logged for decision.

**FINDING #2 (clean bug) — readiness_check emits duplicate entries.**
`orphaned_messages`, `empty_sessions`, `fts_sync` each appear TWICE in the
`checks` list (summary over-counts; `component_readiness` dict hides it via
last-wins). Root cause: `polylogue/readiness/__init__.py:529-531` directly
append these three, AND `_archive_debt_checks` (line 540) re-emits the same
names from the maintenance-target catalog. `archive_debt` is computed fresh in
the same pass (line 495), so the catalog versions aren't stale — the direct
checks are redundant. CAVEAT before removing 529-531: confirm the catalog's
`fts_sync` spec is the same "FTS present" semantic as direct `_fts_sync_check`
(line 343) and not a distinct drift check sharing the name; if distinct, rename
one instead of deleting. Reviewable follow-up.

**Also noted:** readiness schemas_freshness warns 8 provider schemas >30d stale
(claude-code 103d, chatgpt 103d, …) + grok schema missing — informational
(parsing schemas, regenerated via `devtools lab schema generate`).

### 2026-06-27 — bare-uuid resolution + selective embedding (committed)

- `feat(archive) 62d13f2f6` — `read <bare-uuid>` now resolves via unique-suffix
  fallback (was "session not found"; the UUID is the source filename but a
  suffix of the stored `claude-code-session:<uuid>`). +test.
- `feat(embeddings) 248767cb4` — `ops embed backfill --min-messages N` quality
  floor (skip trivial sessions on a partial budget). Realizes the operator's
  "enable selective embedding for real users" steer. **Bonus:** found+fixed a
  THIRD divergent pending-window copy in the preflight (oldest-first) — unified
  onto the canonical newest-first selector, so the preflight now estimates the
  same window the backfill embeds. +test. Verified on prod archive
  (`--max-sessions 10 --min-messages 200` → 21,228 msgs, no stubs).

**Embedding backfill diagnosis (corrected):** it was NOT hung — it was working
(RN states) but slow on large sessions with no per-session progress + RSS
growth to 1.5 GiB. Stopped it (182 sessions / 247 MB persisted), restarted under
a 4 GiB MemoryMax scope, resuming. Real issues = no inner progress + memory
growth (logged, §5 of report).

**Deploy blocked:** sinnix-prime toplevel build fails on `sinex-0.4.2` (Rust) +
`sinex-*-database-auth` derivations — unrelated to polylogue, sinex territory.
run_ref debt is recoverable so deploy is not urgent.

**Completions:** issue #1844 "Build completion on shared query grammar" + #2006
DSL substrate already exist (operator was right). Noted in report §6, not yet
worked.

**Branch (feature/operator-dogfood-hardening): 5 commits.** Operational-state
report delivered to /realm/inbox/polylogue-operational-state-2026-06-27.md.
