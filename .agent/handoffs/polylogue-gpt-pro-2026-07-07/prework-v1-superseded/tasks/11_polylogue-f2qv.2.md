# 11. polylogue-f2qv.2 — Normalize Codex/Claude token lanes into disjoint uncached/cache/reasoning/completion fields

Priority: **P2**  
Lane: **usage-cost-correctness**  
Readiness: **ready-now / parser+tests**

## Why this is urgent / critical-path

Codex/Claude token accounting underpins usage reports, cost reports, and public forensics. Double-counting cached or reasoning tokens creates order-of-magnitude false claims.

## Static diagnosis / likely mechanism

Mechanism from bead: Codex `input` includes cached tokens and `output` includes reasoning tokens; naive input+output inflated cost by 7.69x in a prior fix. Static source anchor: `polylogue/sources/parsers/codex.py` has token_count/cached/reasoning logic around `:161-193` and `:278+`, but the invariant is not locked.

## Implementation plan

Implementation shape:
1. Create/centralize a provider usage normalizer returning disjoint lanes: `input_uncached`, `cache_read`, `cache_write`, `output_completion`, `output_reasoning`.
2. For Codex: derive uncached input as reported input minus cached input, reasoning as a separate output sublane, completion output as reported output minus reasoning. Clamp/report inconsistencies loudly; do not silently negative-clamp without diagnostics.
3. For Claude: map cache creation/read lanes and output/reasoning fields into the same disjoint schema.
4. Add a helper that asserts lane sum equals provider-reported total where the provider reports a total.
5. Ensure `session_provider_usage_events` writer stores lanes separately and downstream rollups consume those lanes, not raw input/output totals.

## Test plan

Tests:
- synthetic Codex payload where cached is 96% of input: disjoint lanes sum to reported total; naive input+output would fail the regression guard.
- synthetic output with reasoning: completion+reasoning equals reported output.
- Claude cache_creation/cache_read payload maps to cache_write/cache_read.
- malformed payload with inconsistent totals is reported/classified, not silently accepted.
- optional live scratch cross-check against `state_5.sqlite` if available.

## Verification command / proof

`devtools test tests/unit/sources/test_codex*.py tests/unit/storage/test_provider_usage*.py -k 'token_count or cache or reasoning or disjoint'`

## Pitfalls

Do not change cost formulas before the lane invariant is locked. Cost packets should depend on this packet.

## Files/functions to inspect or touch

- `polylogue/sources/parsers/codex.py:161-193`
- `polylogue/sources/parsers/codex.py:278+`
- `polylogue/sources/parsers/claude/code_parser.py`
- `polylogue/storage/sqlite/archive_tiers/write.py:provider_usage writer`
- `polylogue/storage/usage.py`
