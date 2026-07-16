# 012. polylogue-f2qv.2 — Codex disjoint-lane normalizer: decompose cached/uncached and reasoning/completion with a regression guard

Priority/type/status: **P2 / task / open**. Lane: **02-usage-cost-honesty**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

PROBLEM. Codex token_count records report 'input' INCLUDING cached tokens (~96% in practice) and 'output' INCLUDING reasoning tokens; naive input+output summation caused a 7.69x cost inflation (fixed in commit 3938bc6c2 on operator-dogfood-hardening). 38x's seed finding 'Codex token lane normalizer divergence' flags this normalizer as needing current-source reconciliation — the fix has no regression guard, so it can silently regress. docs/internals.md asserts 'Cache read/write token lanes remain labelled and are not merged into generic input/output' as a contract with no executable enforcement.

FILES. sources/parsers/codex.py token_count normalizer; storage session_provider_usage_events writer (the lane columns); the equivalent Claude usage extraction (cache_creation/cache_read lanes). Cross-verify against ~/.codex/state_5.sqlite (per-thread median ratio 1.00; copy to scratch first, it is live-locked).

ALGORITHM. Normalizer must emit four disjoint lanes per event: input_uncached = input_total - cached; input_cached = cached; output_completion = output_total - reasoning; output_reasoning = reasoning. Store each lane distinctly; never fold cache into a generic input column. Add an invariant test over synthetic Codex/Claude token_count payloads asserting lanes are disjoint, sum to the reported totals, and that a raw input+output sum would exceed the corrected billable sum (the 7.69x repro stays green as a guard).

PITFALLS. Provider field naming differs (Codex cached_input_tokens vs Claude cache_read_input_tokens); missing lane fields default to 0, not to the total. Reasoning tokens absent on non-reasoning models must not subtract.

## Acceptance criteria

Synthetic Codex and Claude token_count payloads normalize into four disjoint labelled lanes that sum to reported totals; an invariant test asserts disjointness and that the naive input+output sum would double-count (7.69x-class guard). docs/internals.md's cache-lane contract is backed by this test. Live Codex accounting cross-verifies against a scratch copy of state_5.sqlite within tolerance. 38x's Codex-token-lane leg is classified fixed with this test cited.

## Static mechanism / likely defect

Mechanism from bead: Codex `input` includes cached tokens and `output` includes reasoning tokens; naive input+output inflated cost by 7.69x in a prior fix. Static source anchor: `polylogue/sources/parsers/codex.py` has token_count/cached/reasoning logic around `:161-193` and `:278+`, but the invariant is not locked.

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
2. 1. Create/centralize a provider usage normalizer returning disjoint lanes: `input_uncached`, `cache_read`, `cache_write`, `output_completion`, `output_reasoning`.
3. 2. For Codex: derive uncached input as reported input minus cached input, reasoning as a separate output sublane, completion output as reported output minus reasoning. Clamp/report inconsistencies loudly; do not silently negative-clamp without diagnostics.
4. 3. For Claude: map cache creation/read lanes and output/reasoning fields into the same disjoint schema.
5. 4. Add a helper that asserts lane sum equals provider-reported total where the provider reports a total.
6. 5. Ensure `session_provider_usage_events` writer stores lanes separately and downstream rollups consume those lanes, not raw input/output totals.

## Tests to add

- synthetic Codex payload where cached is 96% of input: disjoint lanes sum to reported total; naive input+output would fail the regression guard.
- synthetic output with reasoning: completion+reasoning equals reported output.
- Claude cache_creation/cache_read payload maps to cache_write/cache_read.
- malformed payload with inconsistent totals is reported/classified, not silently accepted.
- optional live scratch cross-check against `state_5.sqlite` if available.

## Verification commands

- ``devtools test tests/unit/sources/test_codex*.py tests/unit/storage/test_provider_usage*.py -k 'token_count or cache or reasoning or disjoint'``

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
