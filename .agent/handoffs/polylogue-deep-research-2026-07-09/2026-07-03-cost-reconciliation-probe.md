---
created: "2026-07-03T08:22:00+02:00"
purpose: "Research synthesis for Beads issue polylogue-lpl"
status: "complete"
project: "polylogue"
---

# Cost Reconciliation Probe

## Context

Research lane for `polylogue-lpl`: turn the cost-reconciliation idea into an
executable implementation issue rather than a broad analysis prompt.

## Findings

- Codex token disjoint-lane semantics are implemented in
  `polylogue/storage/sqlite/archive_tiers/write.py`:
  `_provider_usage_disjoint_lanes` subtracts cached tokens from fresh input and
  does not add reasoning tokens into output again.
- Codex cumulative `token_count` events are already treated as session-global
  latest totals, not per-model rows.
- `polylogue diagnostics usage` audits internal event-vs-rollup drift, but it
  intentionally does not compare against external provider billing/state.
- `scripts/cost_accounting_demo.py` contains a one-off text demo for Codex
  cross-checking `~/.codex/state_5.sqlite`, but there is no reusable structured
  probe.
- No first-class Claude `stats-cache.json` parser exists yet. The intended
  semantics are lane-by-lane `modelUsage` comparison for input, output,
  cacheRead, and cacheCreation. Do not fold cache into input; `costUSD=0` means
  skip dollar reconciliation for that source.

## Product Shape

Implement `devtools lab probe cost-reconciliation`, registered through
`devtools/command_catalog.py` and backed by `devtools/cost_reconciliation_probe.py`.
This should be a lab probe because it validates local private external stores
against archive accounting with tolerance bands; it should not be a normal query
surface.

Required flags:

- `--archive-root`
- `--codex-state`
- `--claude-stats-cache`
- `--scratch-dir`
- `--json`
- `--check`
- `--require-codex`
- `--require-claude`
- tolerance flags for ratio checks

Codex path:

- Copy `state_5.sqlite` to scratch before opening it read-only.
- Join `sessions.native_id` / `codex-session:<uuid>` to `threads.id`.
- Compare archive `MAX(session_provider_usage_events.total_tokens)` to
  `threads.tokens_used`.
- Report compared/missing counts, median, p90, p99, sample disagreements, and
  a separate disjoint-lane decomposition ratio.

Claude path:

- Parse `stats-cache.json` `modelUsage` lanes independently.
- Compare input/output/cacheRead/cacheCreation counts separately.
- Skip cost reconciliation when the source reports zero dollars.

Internal axis:

- Where archive rows carry both provider-reported and catalog-priced basis
  values, report disagreement distributions by model to catch catalog/pricing
  drift.

## Acceptance Criteria

- Command is registered as `devtools lab probe cost-reconciliation`.
- Stable JSON payload supports `--json` and `--check`.
- Missing external stores produce structured skip reasons unless required.
- `--check` exits nonzero only for required missing stores, unreadable schemas,
  or tolerance failures.
- Tests cover synthetic Codex DB, synthetic Claude stats cache, missing and
  malformed stores, JSON shape, check exits, and command catalog/docs
  registration.

## Verification Plan

```bash
devtools test tests/unit/devtools/test_cost_reconciliation_probe.py
devtools test tests/unit/devtools/test_devtools_main.py tests/unit/devtools/test_render_devtools_reference.py
devtools render all --check
devtools verify --quick
```
