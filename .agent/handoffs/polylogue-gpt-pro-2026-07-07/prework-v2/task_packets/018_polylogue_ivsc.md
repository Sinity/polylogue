# 018. polylogue-ivsc — Classify Codex state_5 token drift outside lineage replay

Priority/type/status: **P2 / task / open**. Lane: **02-usage-cost-honesty**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

After logical-session high-water token accounting, the live Codex reconciliation probe still shows 78 logical outside-tolerance threads. New residual classification shows 62/78 have zero replay gap and all sampled residuals come from external state_5.sqlite thread rows with archived=0 and has_user_event=0, while archive sessions contain real user/assistant messages. This is no longer the fork/resume replay double-count class; classify whether state_5 tokens_used is stale, sentinel/default, or a different accounting grain, and update the reconciliation probe/status semantics accordingly.

## Existing design note

Use /realm/tmp/polylogue-cost-reconciliation/codex-logical-probe-current-max100.json as the seed artifact. Compare sampled thread rows against provider token_count events, session_model_usage, Codex rollout paths where available, and any current Codex state schema docs/source. Produce a bounded classifier in the probe rather than making the whole check fail as undifferentiated token drift. Keep logical-session replay-gap diagnostics separate from external-state drift.

## Acceptance criteria

The Codex reconciliation report distinguishes lineage replay residuals from external-state/accounting-grain drift; live active archive artifact explains the remaining outside-tolerance rows without implying replay double-counting; any adjusted pass/fail status is backed by tests and live evidence.

## Static mechanism / likely defect

Issue description localizes the mechanism: After logical-session high-water token accounting, the live Codex reconciliation probe still shows 78 logical outside-tolerance threads. New residual classification shows 62/78 have zero replay gap and all sampled residuals come from external state_5.sqlite thread rows with archived=0 and has_user_event=0, while archive sessions contain real user/assistant messages. This is no longer the fork/resume replay double-count class; classify whether state_5 tokens_used is stale, sentinel/default, or a different accountin… Design direction: Use /realm/tmp/polylogue-cost-reconciliation/codex-logical-probe-current-max100.json as the seed artifact. Compare sampled thread rows against provider token_count events, session_model_usage, Codex rollout paths where available, and any current Codex state schema docs/source. Produce a bounded classifier in the probe rather than making the whole check fail as undifferentiated token drift. Keep logical-session repla…

## Source anchors to inspect first

- `polylogue/storage/usage.py:478` — stale provider rollup stats are materialized/read here.
- `polylogue/storage/usage.py:797` — full stale diagnostics path can become expensive.
- `polylogue/storage/usage.py:891` — comments indicate corrected rows with cached/reasoning partitions.
- `polylogue/storage/usage.py:1033` — model rollup stats aggregate input/output/cache lanes.
- `polylogue/archive/semantic/cost_compute.py` — Inspect pricing/provenance computation before changing cost views.
- `polylogue/archive/semantic/subscription_pricing.py` — Subscription-credit view belongs here, distinct from API-list-equivalent cost.
- `scripts/cost_accounting_demo.py` — Existing demo captures usage/cost accounting expectations.

## Implementation plan

1. Use /realm/tmp/polylogue-cost-reconciliation/codex-logical-probe-current-max100.json as the seed artifact.
2. Compare sampled thread rows against provider token_count events, session_model_usage, Codex rollout paths where available, and any current Codex state schema docs/source.
3. Produce a bounded classifier in the probe rather than making the whole check fail as undifferentiated token drift.
4. Keep logical-session replay-gap diagnostics separate from external-state drift.

## Tests to add

- Acceptance proof: The Codex reconciliation report distinguishes lineage replay residuals from external-state/accounting-grain drift
- Acceptance proof: live active archive artifact explains the remaining outside-tolerance rows without implying replay double-counting
- Acceptance proof: any adjusted pass/fail status is backed by tests and live evidence.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
