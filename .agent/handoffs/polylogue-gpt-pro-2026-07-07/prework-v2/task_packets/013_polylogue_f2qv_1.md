# 013. polylogue-f2qv.1 — Per-model token rollup double-count: session totals partitioned once (#2472)

Priority/type/status: **P2 / bug / open**. Lane: **02-usage-cost-honesty**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

PROBLEM. Memory (project cost/usage analytics research 2026-06-28) records a residual per-model partitioning bug filed as GH #2472: a session's token totals are attributed under EACH model row it touched, so a multi-model session is counted more than once in per-model rollups. This is distinct from the fork/resume lineage double-count (owned by 4ts) — it is a within-session partition error in the model-usage rollup.

FILES. storage session_model_usage rollup builder and its SQL/aggregation (the path that groups session_provider_usage_events by model); any cost_rollups / get_stats_by grouping that partitions by model. Cross-check provider_usage_report_from_connection per-model detail.

ALGORITHM. Attribute each provider usage EVENT's tokens to exactly the model named on that event; a session's total = sum over its events, and per-model totals must partition (sum of per-model = session total). Add a synthetic fixture: one session with events across two models, assert sum(per_model_tokens) == session_total and neither model row carries the full session total.

PITFALLS. GROUP BY model over a table where a session-total column is repeated per event row re-sums the total; join/aggregate must be at event grain. Watch stale session_model_usage rows (xy95) masking the fix.

## Acceptance criteria

On a synthetic two-model session, per-model rollups partition the session total exactly (sum of per-model == session total, no model row holds the full total); a regression test locks this. Live-archive per-model Codex/Claude rollups no longer exceed the session-grain totals. #2472 is cited by the test.

## Static mechanism / likely defect

Mechanism from bead: a session’s token totals are attributed under each model row it touched. Correct behavior: per-model totals are sums of provider usage events whose model equals that model; sum(per-model) == session total. Static anchors: `session_model_usage` materialization in `polylogue/storage/sqlite/archive_tiers/write.py:618`, table/rollup logic around `:2696+`, `:2904+`, `:2953+`.

## Source anchors to inspect first

- `polylogue/storage/usage.py:478` — stale provider rollup stats are materialized/read here.
- `polylogue/storage/usage.py:797` — full stale diagnostics path can become expensive.
- `polylogue/storage/usage.py:891` — comments indicate corrected rows with cached/reasoning partitions.
- `polylogue/storage/usage.py:1033` — model rollup stats aggregate input/output/cache lanes.
- `polylogue/archive/semantic/cost_compute.py` — Inspect pricing/provenance computation before changing cost views.
- `polylogue/archive/semantic/subscription_pricing.py` — Subscription-credit view belongs here, distinct from API-list-equivalent cost.
- `scripts/cost_accounting_demo.py` — Existing demo captures usage/cost accounting expectations.

## Implementation plan

1. Implementation shape:
2. 1. Locate the builder that groups `session_provider_usage_events` into `session_model_usage`.
3. 2. Change grouping to `GROUP BY session_id, model` over event rows; each event contributes only to its own model.
4. 3. Preserve session-grain totals separately if needed; do not copy them into every model row.
5. 4. Add a rollup invariant helper used by tests: for each session, sum(model rows lanes) equals sum(provider event lanes) within integer exactness/tolerance.
6. 5. Add a live-diagnostics query that reports any sessions where per-model > session total.

## Tests to add

- one session with two model events: model A gets only A event tokens, model B gets only B event tokens, sum equals session total.
- mixed cache/reasoning lanes partition independently.
- regression named with GH #2472 or bead id.
- existing single-model session behavior unchanged.

## Verification commands

- ``devtools test tests/unit/storage/test_provider_usage*.py tests/unit/storage/test_session_model_usage*.py -k 'partition or multi_model or f2qv'``

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
