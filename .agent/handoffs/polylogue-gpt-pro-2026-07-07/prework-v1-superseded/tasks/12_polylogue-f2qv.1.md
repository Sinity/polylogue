# 12. polylogue-f2qv.1 — Make per-model rollups partition usage events instead of duplicating session totals

Priority: **P2**  
Lane: **usage-cost-correctness**  
Readiness: **ready-now / storage-rollup**

Depends on packet(s): polylogue-f2qv.2

## Why this is urgent / critical-path

Per-model charts and reports become false when a multi-model session contributes its whole total to every model row it touched.

## Static diagnosis / likely mechanism

Mechanism from bead: a session’s token totals are attributed under each model row it touched. Correct behavior: per-model totals are sums of provider usage events whose model equals that model; sum(per-model) == session total. Static anchors: `session_model_usage` materialization in `polylogue/storage/sqlite/archive_tiers/write.py:618`, table/rollup logic around `:2696+`, `:2904+`, `:2953+`.

## Implementation plan

Implementation shape:
1. Locate the builder that groups `session_provider_usage_events` into `session_model_usage`.
2. Change grouping to `GROUP BY session_id, model` over event rows; each event contributes only to its own model.
3. Preserve session-grain totals separately if needed; do not copy them into every model row.
4. Add a rollup invariant helper used by tests: for each session, sum(model rows lanes) equals sum(provider event lanes) within integer exactness/tolerance.
5. Add a live-diagnostics query that reports any sessions where per-model > session total.

## Test plan

Tests:
- one session with two model events: model A gets only A event tokens, model B gets only B event tokens, sum equals session total.
- mixed cache/reasoning lanes partition independently.
- regression named with GH #2472 or bead id.
- existing single-model session behavior unchanged.

## Verification command / proof

`devtools test tests/unit/storage/test_provider_usage*.py tests/unit/storage/test_session_model_usage*.py -k 'partition or multi_model or f2qv'`

## Pitfalls

Do this after the disjoint lane normalizer, so the partition invariant is defined over the right token fields.

## Files/functions to inspect or touch

- `polylogue/storage/sqlite/archive_tiers/write.py:618`
- `polylogue/storage/sqlite/archive_tiers/write.py:2696+`
- `polylogue/storage/sqlite/archive_tiers/write.py:2904+`
- `polylogue/storage/usage.py`
