# 017. polylogue-5hf — Provider token accounting: honest cross-provider usage ledger

Priority/type/status: **P2 / feature / open**. Lane: **02-usage-cost-honesty**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Coverage, caveats, cached-vs-uncached splits, reasoning tokens, current-window + cumulative session usage. Companions: lineage-tokens (double-count), cost reconciliation probe. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

## Existing design note

SCOPE. The honest cross-provider usage ledger surface: given a session, logical session, day, or origin, return coverage, caveats, cached-vs-uncached input split, reasoning-vs-completion output split, and both current-window and cumulative token totals. This is the READ surface that consumes the corrected lane/pricing substrate from the sibling children; it is not the place to fix the underlying decomposition (that is the disjoint-lane child) or pricing (the LiteLLM child).

FILES. Read models: storage session_provider_usage_events (exact provider events) and session_model_usage (per-model rollup); provider_usage_report_from_connection and the analyze-usage CLI path; MCP provider_usage / cost_rollups / session_costs tools. Coverage/caveat states already enumerated in docs/internals.md 'Provider usage accounting is audited as a source-derived read model' (exact event rows vs text-only estimates vs unsupported origins vs acquired-not-materialized vs stale rollups) are the caveat vocabulary to surface, not to invent.

ALGORITHM. For each origin, prefer exact provider usage events; fall back to text-estimate only with an explicit caveat flag; expose per-lane totals (input_uncached, input_cached, output_completion, output_reasoning) sourced from the disjoint-lane child; attach the LiteLLM-resolved API-equivalent cost and the subscription-credit cost as the two-view child provides them. Report cumulative session usage AND the current provider window separately.

PITFALLS. Do not re-sum raw provider fields here; consume already-decomposed lanes. Do not paper over missing coverage as zero — a source acquired-but-not-materialized is a distinct caveat, not $0. Respect logical-session grain (4ts) so inherited-prefix tokens are not re-counted at the ledger.

## Acceptance criteria

Given a session/day/origin, the ledger returns per-lane token totals (cached/uncached input, reasoning/completion output), a coverage class and caveat set drawn from the documented vocabulary, and both API-equivalent and subscription-credit cost figures. Text-only-estimate and unsupported-origin rows are labelled, never silently zeroed. A test asserts the ledger consumes decomposed lanes (no raw input+output sum) and that logical-grain totals do not re-count inherited-prefix tokens. Verify on the live archive: analyze usage over codex-session and claude-session emit labelled lanes and dual cost views without the 7.69x-class inflation.

## Static mechanism / likely defect

Issue description localizes the mechanism: Coverage, caveats, cached-vs-uncached splits, reasoning tokens, current-window + cumulative session usage. Companions: lineage-tokens (double-count), cost reconciliation probe. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: SCOPE. The honest cross-provider usage ledger surface: given a session, logical session, day, or origin, return coverage, caveats, cached-vs-uncached input split, reasoning-vs-completion output split, and both current-window and cumulative token totals. This is the READ surface that consumes the corrected lane/pricing substrate from the sibling children; it is not the place to fix the underlying decomposition (that …

## Source anchors to inspect first

- `polylogue/storage/usage.py:478` — stale provider rollup stats are materialized/read here.
- `polylogue/storage/usage.py:797` — full stale diagnostics path can become expensive.
- `polylogue/storage/usage.py:891` — comments indicate corrected rows with cached/reasoning partitions.
- `polylogue/storage/usage.py:1033` — model rollup stats aggregate input/output/cache lanes.
- `polylogue/archive/semantic/cost_compute.py` — Inspect pricing/provenance computation before changing cost views.
- `polylogue/archive/semantic/subscription_pricing.py` — Subscription-credit view belongs here, distinct from API-list-equivalent cost.
- `scripts/cost_accounting_demo.py` — Existing demo captures usage/cost accounting expectations.

## Implementation plan

1. SCOPE.
2. The honest cross-provider usage ledger surface: given a session, logical session, day, or origin, return coverage, caveats, cached-vs-uncached input split, reasoning-vs-completion output split, and both current-window and cumulative token totals.
3. This is the READ surface that consumes the corrected lane/pricing substrate from the sibling children
4. it is not the place to fix the underlying decomposition (that is the disjoint-lane child) or pricing (the LiteLLM child).
5. FILES.
6. Read models: storage session_provider_usage_events (exact provider events) and session_model_usage (per-model rollup)
7. provider_usage_report_from_connection and the analyze-usage CLI path

## Tests to add

- Acceptance proof: Given a session/day/origin, the ledger returns per-lane token totals (cached/uncached input, reasoning/completion output), a coverage class and caveat set drawn from the documented vocabulary, and both API-equivalent and subscription-credit cost figures.
- Acceptance proof: Text-only-estimate and unsupported-origin rows are labelled, never silently zeroed.
- Acceptance proof: A test asserts the ledger consumes decomposed lanes (no raw input+output sum) and that logical-grain totals do not re-count inherited-prefix tokens.
- Acceptance proof: Verify on the live archive: analyze usage over codex-session and claude-session emit labelled lanes and dual cost views without the 7.69x-class inflation.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
