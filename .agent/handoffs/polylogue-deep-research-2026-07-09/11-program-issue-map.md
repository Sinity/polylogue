---
created: "2026-06-28"
purpose: "Map open cost/usage/lineage/analytics issues onto the A–G cost-analytics program; record dependencies and post-#2469 status"
status: "complete"
project: "polylogue"
---

# Cost/Usage Analytics Program ↔ Issue Map

## Program steps (recap)
- **A** — pricing catalog + one cache policy + re-price in place
- **B** — fix Codex token over-count
- **C** — capture `server_tool_use`
- **D** — cross-verification vs authoritative sources
- **E** — fold `agent_forensics.py` into materialized insights + delete script
- **F** — Atropos `ScoredDataGroup` export
- **G** — install Hermes in sinnix to test on real sessions

## What PR #2469 already merged (2026-06-28, "normalize session lineage and correct cross-provider cost accounting")
- **Cost double-billing fixed at parse/compute**: Codex disjoint billing lanes
  (`fresh_input = input_with_cached - cache_read`), vendored full LiteLLM price
  catalog (`archive/semantic/data/litellm_model_prices.json`), `pricing.py`/`cost_compute.py`,
  runnable `scripts/cost_accounting_demo.py`.
- **Lineage normalization**: index schema v11→v14; parsers detect Codex
  `forked_from_id` + `thread_spawn`, reclassify `agent-acompact-*`; `session_links`
  gains `branch_point_message_id` + `inheritance`; writer stores divergent tail;
  `get_messages` + `read_archive_session_envelope` compose; Codex compaction summary
  materialized; out-of-order re-extraction.
- **Attachment bytes**: `attachments.blob_hash` nullable + real SHA-256 + `acquisition_status`;
  inline export bytes written to blob store; fabricated hash removed.
- Shipped `scripts/agent_forensics.py` (719 lines) as a stopgap.
- **Crucial caveat**: #2469 fixed cost *compute paths* but did NOT recompute the
  materialized `session_model_usage` rollups, and lineage composition was wired into
  only 2 of ~5 read surfaces. Re-ingest (`polylogue ops reset --database && polylogued run`)
  is REQUIRED to realize storage/cost benefits.

## Program ↔ Issue Map

| Issue | Title (short) | Program step | Depends-on | Status after #2469 |
|---|---|---|---|---|
| #2467 | Session lineage duplication (parent design issue) | underpins B/D/E (foundation) | — | **Partially resolved.** Detection+storage+2 read surfaces landed. Open residue: #2470/#2471/#2472/#2475/#2476/#2478. Keep open as tracking, or close in favor of children. |
| #2468 | Attachments metadata-only (parent design issue) | orthogonal (data-loss) | — | **Partially resolved.** Inline-bytes path landed. Remaining = #2479. Close in favor of #2479. |
| #2472 | Count tokens on logical-session basis (rollups still double-count) | **B** (completes it) | #2467 storage (done); re-ingest | **NOT resolved — the real remaining B work.** #2469 fixed compute, not `session_model_usage` rollups sliced to tail. Highest-priority cost issue. |
| #2470 | Compose forks on all read surfaces (paginated/batch/iter) | B/D correctness (read side) | #2467 (done) | **New, unresolved.** CLI pagination + exports return truncated fork transcripts. Correctness blocker for any export (incl. F). |
| #2471 | Distinguish subagent acompact from main-session acompact | B/D correctness (lineage) | #2467 (done) | **New, unresolved.** ~39/187 acompact files get wrong parent → composition prepends wrong transcript. Data-correctness bug. |
| #2316 | Complete + auditable provider token accounting (parent) | **B + D** (umbrella) | #2472, #2481 | **Partially advanced.** Disjoint lanes + pricing landed; subscription-vs-API split, coverage/caveats, server_tool_use (C) still open. Umbrella for B/C/D. |
| #2481 | devtools lab probe cost-reconciliation (vs state_5.sqlite + stats-cache) | **D** | #2472 (else mismatch is "expected") | **New, unresolved.** This IS step D. Should land after #2472 re-ingest so reconciliation passes. |
| #2480 | Fold agent_forensics.py into analyze + delete script | **E** | #2472 (reasoning lane), #2469 (script exists) | **New, unresolved.** This IS step E. ~70% already materialized; needs reasoning-token lane + usage_timeline insight, then delete the 719-line script. |
| #2478 | Compaction boundary-range columns + effective-context | B/D refinement | #2467 (done); schema v14→v15 | **New, unresolved.** Enables effective-context vs full-prefix distinction; refines token attribution + stale_context pathology. |
| #2473 | MCP scoped aggregates silently capped at page limit | **D** (trust of aggregates) | — (independent) | **New, unresolved.** facets/aggregate/cost_rollups return wrong totals. Cheap, high-value correctness fix for any analytics surface. |
| #2475 | Memoize lineage composition + fix per-write FTS docsize scan | **perf for re-ingest** (enables B/D/E) | #2467 (done) | **New, unresolved.** Old pre-dedup re-ingest hit this 16K×; current dedup baseline is ~2.4K sessions, but this should still land before any large source-root re-ingest that #2472/#2481 need. |
| #2476 | Wrap lineage composition reads in single read txn | correctness (lineage reads) | #2467 (done) | **New, unresolved.** Torn-transcript race during concurrent re-ingest. Low-severity, self-healing. |
| #2479 | Attachment byte acquisition for non-inline sources | orthogonal (extends #2468) | #2468 inline path (done) | **New, unresolved.** Drive/zip/local bytes. Successor to #2468. |
| #2391 | Optimize live full-ingest catch-up latency + WAL | perf (enables re-ingest) | — | **Unchanged.** parse_s ~280s/chunk, WAL → 5.9 GiB. Re-ingest pain; pairs with #2475. |
| #2474 | ChatGPT image-only + Antigravity non-UTF-8 silently dropped | orthogonal (parser robustness) | — | **New, unresolved.** Per-source single-session loss. Not cost; data-completeness. |
| #2477 | Remove dead session-commit stubs / unused row / stale fuzz README | orthogonal (cleanup) | — | **New, unresolved.** Single surgical-renewal PR. |
| #2483 | Harden blob hash validation + drop symlink check | orthogonal (security defense-in-depth) | — | **New, unresolved.** Low-severity; touches blob_store (relevant to #2479). |
| #2482 | topic-pack staged multi-channel retrieval | orthogonal (retrieval feature) | composition correctness (#2470) | **New, unresolved.** Flagship retrieval; not cost program. |
| #2461 | Materialize hook events + OTLP spans | feeds **C** (capture) | — | Capture-completeness; adjacent to C (server_tool_use). |
| #2316/C | capture server_tool_use | **C** | parser work | No dedicated issue yet — C lives under #2316. Consider filing. |
| #818 | Blob orphan GC/integrity | orthogonal | — | **CLOSED.** (was reopened scope; now closed). |
| #1503 | Semantic embeddings operational | orthogonal (embeddings) | — | **CLOSED.** Embedding activation flow shipped (#1217). |

## Dependency / ordering analysis

```
#2475 (perf: memoize composition + FTS docsize) ─┐
#2391 (ingest latency/WAL)                       ├─► makes the required re-ingest tolerable
                                                 │
#2472 (logical-session token rollups)  ──────────┼─► RE-INGEST ──► #2481 (cost reconciliation D passes)
#2470 (compose all read surfaces)      ──────────┘                 │
#2471 (acompact parent correctness)    ──────────┘                 ▼
                                                            #2480 (fold forensics → analyze, delete script = E)
```

- **Cache policy / re-price (A)**: largely landed via vendored LiteLLM catalog +
  disjoint lanes in #2469. Residual A = subscription-vs-API split presentation (#2472 asks
  for it; #2316 specifies it). One cache policy is implicit in the disjoint-lane fix.
- **B before D**: #2472 (rollups attributed to logical root) must land + re-ingest
  *before* #2481, or the reconciliation probe will keep reporting the stale-rollup
  mismatch (#2472 quantifies it: Codex 2.12× pre, ~1.0 post-reingest).
- **#2475/#2391 before the big re-ingest**: composition is recomputed unmemoized per
  fork-child inside the ingest txn; #2475 prevents super-linear blowup. The old
  stress baseline was 16K physical sessions, but the current dedup baseline is
  ~2.4K indexed sessions.
- **#2470/#2471 are read-correctness** and should land alongside #2472 so post-reingest
  exports/aggregates are trustworthy (also unblocks F export and #2482).
- **E (#2480) last** in the cost arc — it consumes the corrected reasoning lane (#2472)
  and deletes the stopgap script #2469 shipped.
- **C (server_tool_use)** is independent of the lineage chain; can proceed in parallel,
  lives under #2316; no dedicated issue — candidate to file.

## Close/merge candidates
- **#818** and **#1503** are already CLOSED — ignore (listed in task but not open).
- **#2467** → demote to tracking umbrella or close once #2470/#2471/#2472/#2475/#2476/#2478
  are filed (they are) — its core landed in #2469. Don't re-implement.
- **#2468** → close in favor of **#2479** (inline path done; non-inline is the only remainder).
- No true duplicates among #2470–#2483; they are clean decompositions of #2467/#2468/#2316
  authored from the same agent findings.
