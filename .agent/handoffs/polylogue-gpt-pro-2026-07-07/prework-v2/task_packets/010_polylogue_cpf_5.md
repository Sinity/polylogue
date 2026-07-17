# 010. polylogue-cpf.5 — Temporal provenance laundering: aggregates collapse to provider_ts; propagate the weakest source

Priority/type/status: **P1 / bug / open**. Lane: **00-trust-floor**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

classify_aggregate_hwm_source (temporal_source.py) launders weak timestamp provenance into provider_ts, so freshness/staleness surfaces look better-grounded than they are. Fix is two-level: aggregate inputs become typed TemporalSource values with weakest_source over the provenance lattice threaded through summaries/rollups/materializer payloads; AND audit the LEAF classifier (classify_profile_hwm_source) — an aggregate fix over already-laundered leaves is half a fix. Truth surfacing may legitimately change recency sorting and staleness UX; that is the point.

## Acceptance criteria

Table-driven tests over every TemporalSource pair (weakest wins); provider_ts + fallback_date aggregate emits fallback_date; leaf audit reports unjustifiable provider_ts paths. Verify: focused temporal tests.

## Static mechanism / likely defect

`classify_aggregate_hwm_source` currently returns provider_ts for any non-empty update list, laundering weak timestamps into strong provenance.

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

1. Define a TemporalSource lattice: provider_ts > capture_ts > file_mtime > fallback/synthetic > unknown, with explicit ordering.
2. Carry source provenance beside every high-water-mark value; aggregates choose the weakest source among contributors, not the strongest.
3. Audit callers in archive summaries/rollups/profile sources and update payload/renderers to expose the source.
4. Add migration/default handling for older rows with unknown source.

## Tests to add

- Aggregate(provider_ts, file_mtime) => file_mtime/weakest, not provider_ts.
- All-provider aggregate remains provider_ts.
- Fallback-only aggregate is synthetic/fallback and rendered with caveat.

## Verification commands

- ``devtools test tests/unit/insights/test_temporal_source_taxonomy.py tests/unit/insights/test_archive_summaries.py -k 'temporal or high_water or weakest'``

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
