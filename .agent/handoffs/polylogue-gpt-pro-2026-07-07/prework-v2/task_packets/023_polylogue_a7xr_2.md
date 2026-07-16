# 023. polylogue-a7xr.2 — Converger and repair disagree on session_profile staleness for NULL-sort-key sessions

Priority/type/status: **P2 / bug / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

VERIFIED LIVE 2026-07-06 (divergence audit): daemon/convergence_stages.py:829-836 and storage/repair.py:566-584 encode DIFFERENT staleness predicates for the same derived rows. For sessions with sort_key_ms IS NULL the converger compares strftime of source_updated_at vs updated_at_ms/1000 as strings, while repair COALESCEs the NULL to 0.0 and applies the 1e-6 epsilon against source_sort_key — a NULL-sort-key session with non-zero source_sort_key is permanently stale to repair and possibly fresh to the converger. Consequence: repeated repair churn or missed rebuilds, and the two paths also source the materializer version differently (constant vs helper call). The two derived-model maintenance paths can disagree about the same row indefinitely.

## Existing design note

One session_profile_stale_predicate(sessions_alias, profile_alias) -> str SQL-fragment builder in storage/insights/session/runtime.py (next to SESSION_INSIGHT_MATERIALIZATION_TYPES); both convergence_stages.py and repair.py compose their queries from it; repair's UNION arms for session_latency_profiles reuse the same fragment with the lp alias. Materializer version comes from one accessor. Decide the NULL-sort-key semantics ONCE (the converger's updated_at comparison is the better-considered branch) and encode it in the fragment. Ties into the cpf temporal doctrine (timeless sessions).

## Acceptance criteria

rg shows exactly one definition of the staleness predicate; a fixture with sort_key_ms NULL + source_sort_key set is classified identically by a convergence pass and an ops repair pass (regression test asserting agreement); no repair churn on a converged archive (idempotence test: repair immediately after convergence selects zero rows). Verify: devtools test -k 'staleness or repair'.

## Static mechanism / likely defect

Issue description localizes the mechanism: VERIFIED LIVE 2026-07-06 (divergence audit): daemon/convergence_stages.py:829-836 and storage/repair.py:566-584 encode DIFFERENT staleness predicates for the same derived rows. For sessions with sort_key_ms IS NULL the converger compares strftime of source_updated_at vs updated_at_ms/1000 as strings, while repair COALESCEs the NULL to 0.0 and applies the 1e-6 epsilon against source_sort_key — a NULL-sort-key session with non-zero source_sort_key is permanently stale to repair and possibly fresh to the converger. C… Design direction: One session_profile_stale_predicate(sessions_alias, profile_alias) -> str SQL-fragment builder in storage/insights/session/runtime.py (next to SESSION_INSIGHT_MATERIALIZATION_TYPES); both convergence_stages.py and repair.py compose their queries from it; repair's UNION arms for session_latency_profiles reuse the same fragment with the lp alias. Materializer version comes from one accessor. Decide the NULL-sort-key s…

## Source anchors to inspect first

- `polylogue/storage/sqlite/archive_tiers/index.py:296` — messages_fts table DDL lives in archive_tiers/index.py.
- `polylogue/storage/fts/sql.py:11` — messages_fts DDL copy also exists in storage/fts/sql.py.
- `polylogue/storage/fts/fts_lifecycle.py:292` — ensure_fts_triggers_sync owns runtime repair/recreation.
- `polylogue/storage/fts/fts_lifecycle.py:512` — Thread FTS rebuild logic is duplicated in lifecycle repair path.
- `polylogue/daemon/convergence_stages.py:988` — daemon convergence has another repair/readiness path for archive messages_fts.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.
- `polylogue/core/dates.py:10` — parse_date has no injected clock parameter.
- `polylogue/core/dates.py:37` — RELATIVE_BASE uses ambient datetime.now.
- `polylogue/archive/query/expression.py:2440` — Query grammar recognizes relative-date literals.
- `polylogue/archive/query/spec.py:498` — SessionQuerySpec.from_params is the central query-spec constructor.
- `polylogue/insights/temporal_source.py:66` — classify_profile_hwm_source promotes any updated_at to provider_ts.
- `polylogue/insights/temporal_source.py:97` — classify_aggregate_hwm_source currently collapses all non-empty source updates to provider_ts.

## Implementation plan

1. One session_profile_stale_predicate(sessions_alias, profile_alias) -> str SQL-fragment builder in storage/insights/session/runtime.py (next to SESSION_INSIGHT_MATERIALIZATION_TYPES)
2. both convergence_stages.py and repair.py compose their queries from it
3. repair's UNION arms for session_latency_profiles reuse the same fragment with the lp alias.
4. Materializer version comes from one accessor.
5. Decide the NULL-sort-key semantics ONCE (the converger's updated_at comparison is the better-considered branch) and encode it in the fragment.
6. Ties into the cpf temporal doctrine (timeless sessions).

## Tests to add

- Acceptance proof: rg shows exactly one definition of the staleness predicate
- Acceptance proof: a fixture with sort_key_ms NULL + source_sort_key set is classified identically by a convergence pass and an ops repair pass (regression test asserting agreement)
- Acceptance proof: no repair churn on a converged archive (idempotence test: repair immediately after convergence selects zero rows).
- Acceptance proof: Verify: devtools test -k 'staleness or repair'.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
