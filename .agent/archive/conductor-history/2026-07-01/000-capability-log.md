---
created: "2026-06-27T18:10:00+02:00"
purpose: "Capability-ownership log — drive one externally-legible capability to done"
status: active
project: polylogue
branch: feature/operator-dogfood-hardening
---

# Capability Log — make one valuable capability real and shown

## Current Archive Baseline Correction

The old 16K-session / 5.7M-message / 37 GiB numbers in this log are a
historical pre-dedup snapshot. They counted fork/continuation duplication that
the lineage/fork-dedup work later resolved. Do not use them in new status
reports, demo prompts, or product claims except as explicitly historical
evidence of the old index shape.

Current live archive probe, 2026-06-30 03:05 CEST, from
`/home/sinity/.local/share/polylogue`:

- `source.db.raw_sessions`: 2,397 rows, latest acquisition
  2026-06-30T03:04:58.856+02:00.
- `index.db.sessions`: 2,390 rows, latest indexed session update
  2026-06-30T02:55:55.095+02:00.
- `index.db.messages`: 159,956 rows.

Operational reading: the live baseline is now a deduplicated archive with about
2.4K indexed sessions and 160K indexed messages, while the dev daemon is still
reconciling source-file catch-up. A large catch-up file count is not evidence
that the archive is empty or that the old 16K count is current.

## The pick: (a) AI-Agent Forensics — corpus-mining findings report

**Chosen capability:** a reproducible analysis that mines the real archive and
produces a findings report with real numbers and charts on how AI-agent usage
actually looks over time — token economy, cost, model evolution, session
failure/abandonment, workflow shape — across the operator's full corpus.

### Why this, over the other candidates

Scored on value × closeness-to-working × skeptic-impressiveness, from a live
survey of the real `index.db` (read-only):

| Candidate | Closeness | Impressiveness | Verdict |
|-----------|-----------|----------------|---------|
| **(a) forensics report** | **High** — analytics already materialized | **High** — data only this archive can show | **PICK** |
| (b) topic context-pack | Medium — tools exist, need end-to-end proof | Medium — a utility, less "wow" | later |
| (c) Atropos/Nous export | Low — external format + viewer round-trip | Niche | later |
| (d) MCP end-to-end | Medium — I found the empty-/tmp bug | Medium — "infra works", and goal says *not* surface-hardening | later |

**Closeness evidence (surveyed live, real archive):**
- Span: **2022-12-11 → 2026-06-27 (~3.5 years)**, 16,099 timestamped sessions.
- 16,410 sessions; **5,722,201 messages**; 5,422,417 blocks.
- Origins: claude-code-session 10,203 · chatgpt-export 2,539 · codex-session
  2,436 · claude-ai-export 1,010 · aistudio-drive 222.
- Materialized analytics already present: `session_profiles` (16,410),
  `session_work_events` (32,050), `session_phases` (45,348),
  `session_model_usage` (15,739, with **`cost_usd`**), and
  **`session_provider_usage_events` (2,972,798)** with per-event
  input/output/cached/cache-write/**reasoning** token columns + `model_name` +
  `occurred_at_ms`.

This is genuinely "only this data can show": 3.5 years of one operator's real
multi-provider AI-agent usage, already materialized into queryable rollups.
A skeptic handed real charts of token burn, cost trends, model evolution, and
failure rates over a 5.7M-message corpus would say "this is real and useful."

### Scope / definition of done

1. A reproducible script that runs **against any archive** (no mocks), reads
   read-only, and emits a markdown findings report + chart PNGs.
2. Run it against the **real 37 GiB archive** → deliver the full report to the
   operator (`/realm/inbox`, private — may include any detail).
3. A README/docs section showing the command and an example.
4. Red-team: every headline number cross-checked against an independent raw-SQL
   query; charts inspected for artifacts; privacy reviewed.

### Privacy boundary (hard constraint, overrides the goal's "commit real output")

The archive is the operator's personal 3.5-year AI history. Committed artifacts
must contain **no verbatim private content** — no session titles, message text,
file paths, or identifying strings. Only **aggregate statistics** (counts,
rates, distributions, token/cost totals, time series) may be committed, and
only **locally** (no push without explicit operator approval). The full,
detailed report goes to `/realm/inbox` for the operator's eyes. The committed
example is reproducible by a stranger against the synthetic demo archive
(`polylogue demo seed`) so the tool is verifiable without private data.

### Build plan

- `scripts/agent_forensics.py` — self-contained, `--archive <root>` (defaults to
  `POLYLOGUE_ARCHIVE_ROOT`), `--out <dir>`. Efficient aggregate SQL only.
- Findings: scale/span; token economy (input/output/cache/reasoning, cache-read
  ratio); total + per-session cost distribution; model evolution over time;
  sessions-per-month rhythm; failure signals (abandoned/stuck, work-event
  outcomes, phases).
- Charts: sessions/month, cumulative cost, token-mix, model-over-time,
  cost-per-session histogram.
- Deliver `/realm/inbox/agent-forensics-2026-06-27.md` (+ charts) to operator.
- Commit script + README section + demo-archive example output.

## Log

- 2026-06-27 18:10 — surveyed real archive, chose (a), wrote this. Building next.

## DONE — capability #1 shipped (commit 2d5ac3c0a)

`scripts/agent_forensics.py` + `docs/agent-forensics.md` + README section.
Real report delivered to `/realm/inbox/agent-forensics-2026-06-27/` (report.md
+ 9 SVG charts). Pure stdlib, read-only, reproducible against the demo archive
(verified: clean `demo seed` → report + 6 charts). Charts validated as
well-formed XML and visually inspected (clean dark-theme growth curve + bars).

### Real findings (operator-private; aggregate, no content)
- Span 2022-12 → 2026-06 (3.5y), 16,410 sessions, 5.72M messages, 5 origins.
- Adoption curve: peak 2026-03 = 3,293 sessions/month.
- Priced (Claude Code): 760.8M fresh input vs **164.0B cache-read = 216×
  cache amplification**; cost **$89,278 API-list-equiv** at **$0.525/M** blended.
- Other providers (origin_reported): 376.6B tokens, uncosted.
- Subscription reality: ~**3.5B credits ≈ ~10 Max-20× plan-months**; cache reads
  cost ZERO credits → plan beats API ~13–37×.
- Per-session cost: median $0.84, p90 $14.53, max $3,192.

### Red-team (caught real errors — the value of running against real data)
1. Reasoning over-counted 3000× (summed cumulative `total_*` not `last_*`
   delta): 955.6B → 317.3M. FIXED.
2. Paired $89K priced cost with 546.5B all-provenance tokens — split by
   provenance; cost now scoped to priced subset. FIXED.
3. "$89K list-price" ignored subscription cache-read-free economics — added
   Subscription reality section (she-llac.com credit model + operator's
   ChatGPT token-accounting export as cross-refs). FIXED.
4. Demo reproduction hit a hot-journal `mode=ro` failure → `connect_ro` now
   fails with guidance, never writes. FIXED. Clean repro verified.

### Methodology cross-refs (saved to memory)
[[reference_claude_subscription_credit_pricing]] — cost_usd is API-equiv,
overstates subscription spend; cache reads free on plans.

## Next capability (per goal: "then pick the next")
Candidates re-ranked now that forensics exists: (d) MCP end-to-end (I found the
empty-/tmp-archive bug — making an external agent search/retrieve live is high
value + I know the fix) or (b) topic context-pack. Also pending from operator
steer: embedding content filter (skip tool-call results, prioritize prose).

## PROGRAM: accurate, integrated cost/usage analytics (multi-session)

Operator direction (2026-06-27, many msgs): build analysis INTO polylogue proper
(not standalone scripts), cumulatively; use extant libs (solved problems); price
EVERYTHING; don't throw away source info; cross-verify derived vs authoritative;
keep re-materialization FAST/low-RAM/low-IO. The agent_forensics.py script is to
be folded in and deleted.

### KEY FINDINGS (evidence, real archive)
- **Pricing coverage gap = the whole $89K→$150K story.** `PRICING`
  (archive/semantic/pricing.py, effective 2026-04-24) is a frozen ~30-model
  subset (Claude+Gemini+GPT-4/o1/o3) — **no gpt-5.x / gpt-5-codex**, so all
  recent OpenAI/Codex sessions are `cost_provenance=origin_reported` / uncosted.
  gpt-5.4 alone = 231.6B tokens. Provenance is literally
  "polylogue-curated-litellm-shaped-seed" → was always meant to track litellm.
- **Cost numbers:** existing $89.6K (Claude-only). Naive litellm all-provider
  all-class = $695K (depends heavily on cache-read pricing policy). Operator .md
  triangulated ~$150K ("preferred" 248.7B cache-inclusive). The spread is
  coverage + cache-class policy → must use a maintained source + ONE explicit
  documented cache policy.
- **DATA-FIDELITY (cross-verification, the important catch):** Codex tokens —
  authoritative `~/.codex/state_5.sqlite threads.tokens_used` = **139.3B**
  (2,289 threads); polylogue `session_model_usage` = **376.6B** (2.7×!);
  polylogue provider-events `last_*` deltas = **270.8B**. Polylogue's two
  internal measures disagree (376.6B vs 270.8B) → likely cumulative-vs-delta
  over-count in the cost materializer. INVESTIGATE before trusting any total.
- **`server_tool_use`** (web_search/web_fetch request counts, separately billed)
  is in Claude JSONL but NOT captured by polylogue parsers. Minor data loss.
- Token/cache classes ARE mostly captured (Claude: in/out/cache_read/cache_write;
  Codex: in/cached_input/out/reasoning) — the gap is cost, not capture.

### AUTHORITATIVE CROSS-VERIFY SOURCES (currently uningested)
- Claude: `~/.config/claude/stats-cache.json` (per-model in/out/cacheRead/
  cacheCreation + `dailyModelTokens` time-series; costUSD=0 subscription) +
  `~/.config/claude/usage-data/` + `ccusage` tool installed.
- Codex: `~/.codex/state_5.sqlite` (`threads.tokens_used`), `logs_2.sqlite`
  (493M event log), `history.jsonl` (prompt history), `goals_1`/`memories_1`.
- Antigravity: `~/.gemini/antigravity/code_tracker/history`.
- she-llac/claude-counter (GitHub) — exact credit/limit reverse-engineering;
  mine for the subscription credit model. [[reference_claude_subscription_credit_pricing]]

### DEPS (operator green-lit "use extant work")
- DONE (pip into .venv, verified): tokencost (gpt-5 priced), tiktoken.
- TODO pyproject + nix: tokencost, tiktoken; litellm price JSON (vendor — has
  gpt-5.4/5.5; 2918 models) since tokencost lags latest. Consider duckdb/polars
  (analytics), a real charting lib (replace hand-rolled SVG).

### EXECUTION ORDER (perf = first-class: re-price via in-place SQL UPDATE over
materialized token cols, NOT re-parse/re-ingest; measure RAM/IO/time)
A. Pricing: source catalog from litellm JSON (+tokencost), one documented
   cache-class policy → re-price in place → all providers costed.
B. Fix the 376.6B vs 270.8B internal Codex over-count (cumulative vs delta).
C. Capture `server_tool_use`.
D. Cross-verification feature: derived vs stats-cache.json / state_5.sqlite.
E. Fold forensics analytics into polylogue insights+CLI; delete the script.
F. Atropos ScoredDataGroup export over whole archive (scores from pathologies).
G. Install Hermes in sinnix to test E/F on real sessions.

### ARCHITECTURE PRINCIPLE (operator, 2026-06-27)
Materialize analyses INTO the schema — compute at ingest/read into derived
read-model tables (like session_model_usage/session_profiles/run projection),
NOT dynamically at query time. Dynamic stats only when accurate materialization
isn't feasible. So: (1) agent_forensics.py is doubly wrong (standalone + dynamic)
→ fold its analytics into materialized rollups + thin read surfaces; (2) duckdb/
polars is a fallback, not the default. Perf: re-price via in-place SQL UPDATE.

### STATUS
- Deps added to pyproject + uv.lock (clean resolution): tiktoken 0.13.0,
  tokencost 0.1.26. litellm JSON still to vendor for latest gpt-5.4/5.5.
- Branch feature/operator-dogfood-hardening: 6 commits incl. the (to-be-folded)
  agent_forensics.py. Forensics report already delivered to /realm/inbox.

### CORRECTIONS (operator)
- PERF focus = **initial re-ingestion snappiness**, NOT re-pricing. When a
  pricing/schema change forces reset + re-ingest, DOGFOOD the ingestion:
  measure throughput/RAM/IO, ensure snappy (ties to #2391 catch-up latency).
- ccusage (`~/.config/claude/ccusage`) is OBSOLETE — not a source.
- WebUI analytics (surface H): materialized rollups → daemon JSON endpoint →
  client-side charts (Chart.js/Plotly.js) in the web reader.

### PROGRESS LOG (resumed session 2026-06-27, after session-limit reset)
- **Step A LANDED (commit 67dd9e64c).** Vendored LiteLLM catalog (3558 models)
  replaces frozen ~30-model PRICING; curated overrides win; date-snapshot
  normalization. gpt-5.x/codex/deepseek now priced; unknowns stay unpriced
  (not silent-zero). cost_compute._PRICE_SNAPSHOT_VERSION now derives from
  pricing.py canonical constants. 200 cost/pricing tests pass.
  - Dep tension resolved: tokencost/tiktoken were in pyproject but NEVER
    imported (dead). tokencost superseded by vendored JSON -> dropped. tiktoken
    (better token-estimate fallback) is OFF the critical path (real archive uses
    provider-reported EXACT tokens) -> deferred, not wired.
  - Seed mechanics: catalog_id = f"{PROVENANCE}-{DATE}"; both changed -> next
    daemon-start seeds a NEW catalog row idempotently on existing archives, no
    reset. NOT a schema bump (DDL unchanged).
- **NEXT = Step B (truth layer): Codex token over-count.** Three measures
  disagree: authoritative state_5.sqlite threads.tokens_used=139.3B vs
  session_model_usage=376.6B (2.7x) vs provider-events last_* deltas=270.8B.
  Until reconciled, NO cost total is trustworthy. Plan: evidence-harness first
  (measure all three over real archive + reconcile against state_5.sqlite),
  THEN fix the materializer. Demonstrable capability #2 core: a cross-verified,
  all-provider cost ledger a skeptic trusts.

### STEP B LANDED (commit 3938bc6c2) — Codex cost double-count fixed + cross-verified
Two confirmed bugs in the token_count rollup (archive_tiers/write.py), both inflating cost:
1. **Cached double-count**: Codex input_tokens INCLUDES cached_input (verified:
   cached<=input on 100% of 1.84M real events). Rollup stored input=total_input
   AND cache_read=total_cached separately; cost bills both as additive lanes ->
   cached billed twice. Cached is ~96% of Codex input -> **7.69x inflation**.
2. **Reasoning double-count**: output stored as total_output+total_reasoning, but
   Codex output already includes reasoning (total==input+output 98.9%).
Fix: `_provider_usage_disjoint_lanes()` — fresh_input=input-cached (clamp 0),
output passthrough, cache_read/write unchanged. Applied at 4 rollup sites +
usage.py drift-check. 3 old-behavior fixtures corrected + helper regression test.
218 tests pass; mypy/ruff clean.

**CROSS-VERIFICATION (the killer evidence, real archive vs ~/.codex/state_5.sqlite):**
- Corrected Codex cost $76,856 vs buggy $591,103 (7.69x).
- **Per-thread poly/auth ratio: median=1.00, p10=1.00, p90=1.00** (1986/2110
  within 10%). Corrected formula matches Codex's authoritative token store
  thread-by-thread.
- 1.43x aggregate residual FULLY EXPLAINED + shows polylogue's value:
  - 96 threads / **36.18B tokens in polylogue but PRUNED from live state_5**
    (archive retains history the live tool discarded).
  - 114 resumed-session threads (14B) where rollout cumulative > state_5's
    last-recorded tokens_used.
- Harness: .agent/scratch/cost-reconcile/{measure,corrected_ledger}.py

### REMAINING for P2 demonstrable artifact (memo "agent 2: Forensics & cost"):
- Fold forensics analytics INTO polylogue (delete agent_forensics.py) [step E, big].
- Re-materialize real archive so stored cost reflects the fix (operator step).
- Ship: private-safe demo-archive sample output + git-ignored real output +
  README section on which cost claims are externally safe.
- The state_5 cross-verify is architecturally a devtools/lab probe (validates
  materialization), not an archive surface.

## DONE — capability #2 shipped (cost truth-layer, memo P2)
Commits on feature/operator-dogfood-hardening:
- 67dd9e64c feat(pricing): vendor full LiteLLM catalog (every provider priced)
- 3938bc6c2 fix(cost): stop double-billing Codex cached input + reasoning (7.69x)
- f6d455264 docs(cost): runnable demo + cross-verify + docs + README + sample

**Demonstrable artifact** (`scripts/cost_accounting_demo.py`):
- Stranger-reproducible: `uv run python scripts/cost_accounting_demo.py`
  (exit 0 in clean env, no POLYLOGUE_ARCHIVE_ROOT needed). Ingests a crafted
  Codex session through the REAL writer, reads back materialized rollup, prices
  with real catalog, shows corrected vs pre-fix (7.10x per-session). No mocks.
- SELF-GUARDING: readback assert `fresh_input + cache_read == provider input`
  fails (AssertionError) if the double-count ever regresses.
- Operator cross-verify: `--archive ~/.local/share/polylogue
  --codex-state ~/.codex/state_5.sqlite` -> per-thread median 1.000, 36.18B
  pruned-but-retained tokens.
- Committed sample: docs/examples/cost-accounting-demo.txt (synthetic, safe).
- docs/cost-model.md "Codex disjoint billing lanes" + README pointer.

**Red-team passed**: real-data (real writer/pricing, not mocked logic) + operator
cross-verify on real archive; baseline/counterfactual (buggy vs correct); no
overstatement ($591K buggy is conservative — omits reasoning re-add); reusable;
private clearly fenced. Gates: ruff+mypy clean, doc-commands (77 files), render
all --check (all sync OK).

## NEXT capability (per goal "then pick the next")
Memo ranks P0 (Sinex topic-lineage context-pack) the #1 first demo. With cost
truth-layer done, P0 is the natural next: iterative multi-channel retrieval
(lexical seeds -> term/file/branch extraction -> embedding -> time/topology
expansion -> classify), staged artifacts. Forbidden to conclude "no results"
from exact-string search. Also available: re-materialize real archive so stored
cost reflects the fix (operator step), and fold agent_forensics.py into polylogue
proper (operator's step E, deletes the script).

## PERF: ingest commit batching (data-driven, 2026-06-28)
Built permanent instrument `devtools bench ingest-throughput` (cpu/mem/io/stage,
PR #2491). Findings against synthetic WAL ingest:
- Ingest is I/O-WAIT bound (cpu_utilization ~0.60), NOT cpu/parallel bound.
- write_parsed_session_to_archive AND write_source_raw_session each commit per call
  (`with conn:`) → 2 fsyncs/session. Per-session commit = the cost.
- synchronous=OFF gives 1.45x but risks corruption; commit BATCHING is strictly
  better (faster AND fewer bytes AND crash-safe at sync=NORMAL).
- Policy sweep (mixed 200 small + 6x2000-msg workload): per-session 9622 msg/s /
  120MB written; session-count N=20 13195/30MB (peak_wal 27.9MB); **message-count
  M=8000 12745 msg/s / 25MB / peak_wal 13.9MB**; one-shot SLOWER (11335) + big WAL.
- CHOICE: work-based (message-count) M≈8000, NOT session-count — uniform txn work
  → ~half the peak WAL of session-count when a batch catches large sessions; ~the
  best throughput; bytes near the one-shot floor. Knee is ~M=8000; bigger is worse.
- WAL cap (160MB #1614) / 40MB autocheckpoint are NOT re-ingest bottlenecks under
  batching (peak WAL ~14MB << both); tuning autocheckpoint on top = noise. Leave them.
IMPL: manage_transaction param added to write_parsed_session_to_archive; need same
on write_source_raw_session + write_raw_and_parsed passthrough + parse_sources_archive
batch driver (commit every ~8000 accumulated messages, commit tail, rollback batch on error).
