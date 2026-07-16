# 133. polylogue-f2qv — Provider usage & cost honesty: disjoint token lanes, one pricing source, dual cost view

Priority/type/status: **P2 / epic / open**. Lane: **02-usage-cost-honesty**. Release: **A-trust-floor**. Readiness: **epic-needs-child-closure**.

## What the bead says

WHY: token/cost accounting is a correctness surface with a track record of silent large errors (7.69x Codex inflation; per-model partition double-count #2472) — and cost numbers are exactly what operators quote publicly, so wrong numbers are reputational. Four invariants define honest accounting (full doctrine in design): disjoint token lanes; one pricing source (vendored LiteLLM catalog, last-path-segment match); dual view (API-list-equivalent vs subscription-credit); stale-row hygiene (376.6B-token class artifacts re-ingest away). ENABLES: credible cost analytics (9l5.4), provider comparisons, the flight-recorder byte-resolution promise applied to money. Epic members carry the per-surface work; this epic owns the invariants staying true across new providers/models.

## Existing design note

PROBLEM / DOCTRINE. Token and cost accounting is a correctness surface, not a nicety: prior bugs produced a 7.69x Codex cost inflation and a residual per-model partition double-count (#2472). Four invariants define 'honest' and are the spine of this epic:

1. DISJOINT LANES. Provider-reported token fields overlap and must be decomposed before summing. Codex 'input' INCLUDES cached tokens (~96% of input in practice) and Codex 'output' INCLUDES reasoning tokens; summing raw input+output double-counts. Claude cache-read tokens are effectively free on subscription. The archive must store and report cached/uncached and reasoning/completion as SEPARATE labelled lanes and never fold cache lanes into generic input/output (docs/internals.md 'Provider usage accounting' already asserts this contract; regression guard is missing).

2. SINGLE PRICING SOURCE. LiteLLM's vendored price catalog (committed 67dd9e64c under the LiteLLM catalog module) is the sole source of per-model $/1M rates. tokencost must be dropped; model-name resolution matches the LAST path segment of the model id. No second pricing table may drift against it.

3. DUAL COST VIEW. cost_usd today is API-list-equivalent and OVERSTATES actual subscription spend (cache reads free on Max/Pro; credit formula differs). Surfaces must report BOTH an API-list-equivalent view AND a subscription-credit view, never conflate them, and must not carry the credit-rate 5x-output error.

4. RECONCILIATION AGAINST GROUND TRUTH. Archive accounting is validated against external provider state (Codex ~/.codex/state_5.sqlite; Claude stats-cache.json) via the cost-reconciliation probe, with lineage-replay residuals classified separately from external-state/accounting-grain drift.

SURFACES / MODULES. Provider usage read models: session_provider_usage_events (exact event rows), session_model_usage (per-model rollup), sessions authored-user aggregates; provider_usage_report_from_connection and the analyze usage CLI/MCP path; cost rollup surfaces (cost_rollups / session_costs / cost_outlook MCP tools); the LiteLLM price-catalog module; devtools lab probe cost-reconciliation. Parsers: Codex token_count normalizer (sources/parsers/codex.py) and Claude usage extraction.

RELATION TO OTHER EPICS. This epic OWNS the token/cost-honesty leg that is currently a relates-to leaf off 38x (archived-audit reconciliation) and 4ts (session lineage truth). Lineage double-counting of INHERITED-PREFIX tokens across fork/resume/compaction stays owned by 4ts (logical-session high-water accounting); this epic owns WITHIN-session lane decomposition, cross-provider pricing, and reconciliation. 38x's 'Codex token lane normalizer divergence' seed finding is adopted here as a concrete child. Keep 38x itself as a relates-to meta-reconciliation task, not reparented.

PITFALLS. (a) Summing raw provider fields re-introduces the 7.69x class. (b) Per-model partition SQL that sums a session's total under each model row double-counts multi-model sessions (#2472). (c) Merging cache-read lane into input hides the free-on-subscription reality. (d) Stale session_model_usage rows read as live drift (xy95). (e) A second hardcoded price map silently drifts from LiteLLM.

## Acceptance criteria

1. A cross-provider usage ledger reports cached/uncached input and reasoning/completion output as separate labelled lanes for Codex, Claude, ChatGPT; a property/invariant test asserts no lane is double-summed and cache lanes are never folded into generic input/output (repro of the 7.69x-class inflation stays green).
2. Per-model rollups sum to the session total with no multi-model double-count; #2472 has a regression test on a synthetic multi-model session.
3. All model->price resolution goes through the single LiteLLM catalog (last-path-segment match); grep shows tokencost is gone and no second price table exists.
4. Cost surfaces expose an API-list-equivalent view AND a subscription-credit view distinctly; the credit-rate 5x-output error is fixed and covered by a test.
5. The cost-reconciliation probe distinguishes lineage-replay residuals from external-state/accounting-grain drift and passes against the live 38GB archive with documented remaining outside-tolerance rows.
6. The provider-usage full diagnostic returns within an interactive budget or gates expensive sections separately (no D-state hang) on the live archive.
7. docs/internals.md 'Provider usage accounting' contract is backed by executable checks, not prose alone.

## Static mechanism / likely defect

Issue description localizes the mechanism: WHY: token/cost accounting is a correctness surface with a track record of silent large errors (7.69x Codex inflation; per-model partition double-count #2472) — and cost numbers are exactly what operators quote publicly, so wrong numbers are reputational. Four invariants define honest accounting (full doctrine in design): disjoint token lanes; one pricing source (vendored LiteLLM catalog, last-path-segment match); dual view (API-list-equivalent vs subscription-credit); stale-row hygiene (376.6B-token class artifac… Design direction: PROBLEM / DOCTRINE. Token and cost accounting is a correctness surface, not a nicety: prior bugs produced a 7.69x Codex cost inflation and a residual per-model partition double-count (#2472). Four invariants define 'honest' and are the spine of this epic: 1. DISJOINT LANES. Provider-reported token fields overlap and must be decomposed before summing. Codex 'input' INCLUDES cached tokens (~96% of input in practice) a…

## Source anchors to inspect first

- `polylogue/storage/usage.py:478` — stale provider rollup stats are materialized/read here.
- `polylogue/storage/usage.py:797` — full stale diagnostics path can become expensive.
- `polylogue/storage/usage.py:891` — comments indicate corrected rows with cached/reasoning partitions.
- `polylogue/storage/usage.py:1033` — model rollup stats aggregate input/output/cache lanes.
- `polylogue/archive/semantic/cost_compute.py` — Inspect pricing/provenance computation before changing cost views.
- `polylogue/archive/semantic/subscription_pricing.py` — Subscription-credit view belongs here, distinct from API-list-equivalent cost.
- `scripts/cost_accounting_demo.py` — Existing demo captures usage/cost accounting expectations.

## Implementation plan

1. PROBLEM / DOCTRINE.
2. Token and cost accounting is a correctness surface, not a nicety: prior bugs produced a 7.69x Codex cost inflation and a residual per-model partition double-count (#2472).
3. Four invariants define 'honest' and are the spine of this epic:
4. 1.
5. DISJOINT LANES.
6. Provider-reported token fields overlap and must be decomposed before summing.
7. Codex 'input' INCLUDES cached tokens (~96% of input in practice) and Codex 'output' INCLUDES reasoning tokens

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: A cross-provider usage ledger reports cached/uncached input and reasoning/completion output as separate labelled lanes for Codex, Claude, ChatGPT
- Acceptance proof: a property/invariant test asserts no lane is double-summed and cache lanes are never folded into generic input/output (repro of the 7.69x-class inflation stays green).
- Acceptance proof: 2.
- Acceptance proof: Per-model rollups sum to the session total with no multi-model double-count
- Acceptance proof: #2472 has a regression test on a synthetic multi-model session.
- Acceptance proof: 3.
- Acceptance proof: All model->price resolution goes through the single LiteLLM catalog (last-path-segment match)

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
