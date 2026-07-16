# 011. polylogue-cpf.6 — Temporal correctness: clock seam for relative-date parsing + targeted sort_key_ms audit

Priority/type/status: **P1 / bug / open**. Lane: **00-trust-floor**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

(1) core/dates.py:37 sets RELATIVE_BASE = datetime.now(tz=utc) PER CALL inside parse_date (verified live 2026-07-06 — the earlier 'frozen at import' claim was wrong; a long-lived daemon does NOT drift). The real defect: relative-date parsing has no clock seam, so frozen_clock cannot reach it, since:7d is untestable deterministically, and query-time now() is uncontrolled. Fix = route parse_date + query lowering through a core/clock.py seam. (2) sort_key_ms COALESCE(...,0): SOME read paths epoch-pin timeless sessions; others handle NULL explicitly — this needs a targeted audit of every ordering/window path (classify each: fixed / safe / intentionally synthetic), not a blanket claim. Timeless sessions excluded from lower-bounded timed windows by default but reachable via include_timeless with explicit time_confidence. The wider four-time-kinds doctrine lives in the cpf epic; this bead is the clock seam + the audit + the two concrete fixes.

## Acceptance criteria

parse_date and query lowering accept an injected clock; since:7d under frozen_clock is deterministic and shifts only with the injected clock; no direct datetime.now in query-time parsing outside the seam (lint or grep gate); audit table enumerates every sort_key_ms/COALESCE ordering+window path with a fixed/safe/synthetic verdict; timeless sessions appear with time_confidence=synthetic instead of vanishing or pinning to 1970. Verify: focused date/query tests + the audit artifact.

## Static mechanism / likely defect

`parse_date` uses ambient `datetime.now()` as RELATIVE_BASE, making relative query semantics nondeterministic and hard to test. `sort_key_ms` fallback-to-zero paths need audit for synthetic time leakage.

## Source anchors to inspect first

- `polylogue/core/dates.py:10` — parse_date has no injected clock parameter.
- `polylogue/core/dates.py:37` — RELATIVE_BASE uses ambient datetime.now.
- `polylogue/archive/query/expression.py:2440` — Query grammar recognizes relative-date literals.
- `polylogue/archive/query/spec.py:498` — SessionQuerySpec.from_params is the central query-spec constructor.
- `polylogue/insights/temporal_source.py:66` — classify_profile_hwm_source promotes any updated_at to provider_ts.
- `polylogue/insights/temporal_source.py:97` — classify_aggregate_hwm_source currently collapses all non-empty source updates to provider_ts.
- `polylogue/insights/audit.py:173` — build_insight_rigor_audit_report is the audit entry point.
- `polylogue/insights/audit.py:194` — Current code iterates list_rigor_contracts, not the product registry.
- `polylogue/insights/audit.py:216` — Registry lookup is secondary and skipped for products without contracts.
- `polylogue/insights/rigor.py:85` — _RIGOR_MATRIX declares only a subset of registered products.
- `polylogue/insights/registry.py:294` — INSIGHT_REGISTRY is the universe the audit should iterate.

## Implementation plan

1. Add an injected clock parameter or context object to `parse_date`, query parser lowering, and all surfaces that accept relative dates.
2. Thread the frozen clock through CLI/daemon/MCP tests; default only at the outer boundary.
3. Inventory `COALESCE(sort_key_ms, 0)` and similar ordering fallbacks; replace silent epoch ordering with explicit synthetic-time fields or loud caveats.
4. Add operator audit output listing timeless/synthetic sessions and where they affect ordering.

## Tests to add

- Frozen clock: `since:7d` lowers to identical absolute bound in CLI, daemon, MCP.
- Changing wall clock during test does not change parsed result.
- Timeless rows do not masquerade as 1970/provider time; render confidence as synthetic/unknown.

## Verification commands

- ``devtools test tests/unit/core/test_dates.py tests/unit/archive/test_query_dates.py -k 'relative or clock or sort_key or timeless'` plus audit artifact review.`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
