# 016. polylogue-f2qv.5 — Version-gate provider-usage projection so it self-heals like session_profiles

Priority/type/status: **P2 / bug / open**. Lane: **02-usage-cost-honesty**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

PROBLEM: session_model_usage (provider token/cost rollup) is materialized once at ingest (polylogue/storage/sqlite/archive_tiers/write.py:618) and is NOT in the insight rebuild path (absent from storage/insights/session/rebuild.py and insights/registry.py). The materializer_version self-heal gate in storage/insights/session/status.py:211-368 covers session_profile/logical/work/threads but NOT provider usage. So when the provider-usage materializer improves or a zero-token bug is fixed, stale rows persist and archive_debt._provider_usage_rows (operations/archive_debt.py:847-912) can only offer a manual full 'Rebuild the index' — no convergence stage or periodic loop re-derives it. This violates the automagic-invariants doctrine: derived read-model staleness belongs to daemon convergence, and provider usage is the one insight rollup left as manual operator maintenance. DESIGN: give provider-usage a materializer_version (or reuse insight_materialization with a 'provider_usage' insight_type), have the session-insight rebuild path re-derive session_model_usage from blocks/usage events when the version differs, and add a stale-provider-usage check to the periodic session-insight drain (daemon/cli.py _drain_session_insights_once / _schema_archive_session_ids_missing_profiles) so a version bump auto-refreshes existing rows. Coordinate with f2qv.1 (fix the double-count first so the re-derivation is correct). PITFALL: cache read/write token lanes must stay disjoint (see reference_codex_token_semantics). PITFALL: page the refresh; do not fetchall all sessions.

## Acceptance criteria

1) Provider-usage rollups carry a materializer version and a stale check reachable from the periodic session-insight convergence loop. 2) Bumping the provider-usage materializer version auto-refreshes existing session_model_usage rows on a daemon run without any manual `maintenance rebuild-index` (test: seed rows at an old version, run drain, assert rows re-derived). 3) archive_debt provider-usage 'zero-token' rows drain to zero after a daemon run on an archive whose source blocks carry usage, instead of requiring a full index rebuild. 4) devtools test covering the new stale-provider-usage path passes.

## Static mechanism / likely defect

Root cause from bead and source anchors: provider usage rows are written during ingest (`polylogue/storage/sqlite/archive_tiers/write.py:618`) but are not in the session insight rebuild/convergence path. The status gate covers profiles/logical/work/threads, not provider usage. Archive debt can only recommend manual rebuild (`polylogue/operations/archive_debt.py:854+`, zero-token rows `:899+`).

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
2. 1. Add a materializer version for provider-usage/session-model-usage projection.
3. 2. Store that version in an existing insight materialization table or a new lightweight table keyed by session id.
4. 3. Extend session-insight status/staleness detection to include provider usage.
5. 4. Extend the periodic convergence loop to refresh stale provider usage rows for sessions whose source blocks carry usage.
6. 5. Archive-debt should report stale provider usage as drainable by convergence, not only full rebuild.

## Tests to add

- old-version `session_model_usage` row becomes stale.
- daemon/convergence drain refreshes it without `maintenance rebuild-index`.
- zero-token debt over sessions with source usage drains to zero after convergence.
- archive_debt message changes from manual-only to convergence-drainable.

## Verification commands

- ``devtools test tests/unit/storage/insights tests/unit/operations/test_archive_debt.py -k 'provider_usage or materializer_version or stale'``

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
