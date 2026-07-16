# 019. polylogue-xy95 — Speed up provider usage full stale diagnostics

Priority/type/status: **P2 / bug / open**. Lane: **02-usage-cost-honesty**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

During polylogue-4ts.2, polylogue analyze usage --origin codex-session --detail full --limit 20 --format json entered D-state and had to be terminated. A targeted SQL audit over the same archive completed in about 30s and showed Codex stale rollups were actually clean after the reasoning-only predicate fix. The full report path likely does avoidable broad Python reconstruction/source sampling work and is too slow for routine devloop evidence.

## Existing design note

Profile provider_usage_report_from_connection(detail='full', origin='codex-session') by stage. Replace the stale-rollup path with bounded SQL/window aggregates or add planner-supporting indexes if needed. Keep raw/source debt and sample collection separate so stale-rollup diagnostics can be requested cheaply. Add a regression/perf smoke that prevents full detail from silently doing unbounded row materialization on large archives.

## Acceptance criteria

On the active archive, the Codex full usage diagnostic either completes within an agreed interactive budget or exposes separately selectable expensive sections; no D-state wait in the normal stale-rollup path; tests cover reasoning-only rows and the optimized stale-rollup result.

## Static mechanism / likely defect

Issue description localizes the mechanism: During polylogue-4ts.2, polylogue analyze usage --origin codex-session --detail full --limit 20 --format json entered D-state and had to be terminated. A targeted SQL audit over the same archive completed in about 30s and showed Codex stale rollups were actually clean after the reasoning-only predicate fix. The full report path likely does avoidable broad Python reconstruction/source sampling work and is too slow for routine devloop evidence. Design direction: Profile provider_usage_report_from_connection(detail='full', origin='codex-session') by stage. Replace the stale-rollup path with bounded SQL/window aggregates or add planner-supporting indexes if needed. Keep raw/source debt and sample collection separate so stale-rollup diagnostics can be requested cheaply. Add a regression/perf smoke that prevents full detail from silently doing unbounded row materialization on l…

## Source anchors to inspect first

- `polylogue/storage/usage.py:478` — stale provider rollup stats are materialized/read here.
- `polylogue/storage/usage.py:797` — full stale diagnostics path can become expensive.
- `polylogue/storage/usage.py:891` — comments indicate corrected rows with cached/reasoning partitions.
- `polylogue/storage/usage.py:1033` — model rollup stats aggregate input/output/cache lanes.
- `polylogue/archive/semantic/cost_compute.py` — Inspect pricing/provenance computation before changing cost views.
- `polylogue/archive/semantic/subscription_pricing.py` — Subscription-credit view belongs here, distinct from API-list-equivalent cost.
- `scripts/cost_accounting_demo.py` — Existing demo captures usage/cost accounting expectations.

## Implementation plan

1. Profile provider_usage_report_from_connection(detail='full', origin='codex-session') by stage.
2. Replace the stale-rollup path with bounded SQL/window aggregates or add planner-supporting indexes if needed.
3. Keep raw/source debt and sample collection separate so stale-rollup diagnostics can be requested cheaply.
4. Add a regression/perf smoke that prevents full detail from silently doing unbounded row materialization on large archives.

## Tests to add

- Acceptance proof: On the active archive, the Codex full usage diagnostic either completes within an agreed interactive budget or exposes separately selectable expensive sections
- Acceptance proof: no D-state wait in the normal stale-rollup path
- Acceptance proof: tests cover reasoning-only rows and the optimized stale-rollup result.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
