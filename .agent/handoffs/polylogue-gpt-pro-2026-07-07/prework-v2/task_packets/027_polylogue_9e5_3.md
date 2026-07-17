# 027. polylogue-9e5.3 — Column honesty audit: null/unknown density for key semantic columns

Priority/type/status: **P2 / task / open**. Lane: **00-trust-floor**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

For material_origin, tool_result_is_error/exit_code, message_type, branch_type, session_kind: null/unknown density per origin per month on the live archive. Tells you which structural fields are populated well enough to replace keyword heuristics — the go/no-go gate for the heuristic->structural sweep bead and the coverage-caveat source for outcome analytics.

## Existing design note

Read-only column-honesty census over the live index.db (promoted from the 2026-07-04 notes sidecar; no product-code mutation). For each of material_origin, tool_result_is_error, tool_result_exit_code, message_type/block_type, branch/link type, and session_kind, compute NULL + 'unknown'-sentinel density as a fraction of ELIGIBLE rows, grouped by (origin, YYYY-MM). Columns are CHECK-constrained in polylogue/storage/sqlite/archive_tiers/index.py (e.g. material_origin DEFAULT 'unknown', tool_result_is_error IN (0,1)), so 'unknown'/'NULL' means structure-absent. Emit one row per (column, origin, month) plus a per-column rollup: total, null_count, unknown/empty_count, populated_count, populated_pct, top-5 non-null values, and a per-field recommendation (structural-ready / structural-with-caveat / keep-heuristic). Pitfall: tool_result_* denominator is tool_result blocks only, not all blocks; material_origin denominator is authored messages only. Save the exact SQL alongside the artifact so it reruns after schema rebuilds. Read via SQLite URI mode=ro against a copy or the live archive.

## Acceptance criteria

1. A committed evidence artifact (CSV/JSON matrix + short markdown, under .agent/scratch/research/ or demo-shelf) reports, per (column, origin, month), the null/unknown/populated counts, populated_pct, and top-5 values, with denominators correct (tool_result_* over tool_result rows, material_origin over authored messages). 2. Each column carries a go/no-go verdict — structural-ready vs structural-with-caveat vs keep-heuristic — that the b0b heuristic->structural sweep bead consumes, plus a per-origin coverage-caveat sentence for outcome analytics. 3. The exact SQL is saved with the artifact and reconciles to SELECT COUNT(*) on each source table. 4. No product-code mutation; follow-up beads are filed only where populated_pct + value distribution justify a heuristic replacement. Verify: the SQL runs read-only (mode=ro) against POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue and its totals match the per-table COUNT(*).

## Static mechanism / likely defect

This is a read-only audit packet. It supports 9e5.29 by finding numeric/default columns whose meaning is ambiguous, especially across insight products, usage, attachments, and lineage.

## Source anchors to inspect first

- `polylogue/insights/audit.py:173` — build_insight_rigor_audit_report is the audit entry point.
- `polylogue/insights/audit.py:194` — Current code iterates list_rigor_contracts, not the product registry.
- `polylogue/insights/audit.py:216` — Registry lookup is secondary and skipped for products without contracts.
- `polylogue/insights/rigor.py:85` — _RIGOR_MATRIX declares only a subset of registered products.
- `polylogue/insights/registry.py:294` — INSIGHT_REGISTRY is the universe the audit should iterate.
- `polylogue/archive/actions/followup.py` — Action/followup classification is a real structural analytics input.
- `polylogue/archive/actions/fields.py` — Action fields determine what can be measured without prose heuristics.
- `polylogue/insights/registry.py:294` — Insight registry should become measure/product registry input.
- `scripts/agent_forensics.py` — Existing forensics script is a proof artifact and candidate product surface.
- `polylogue/core/dates.py:10` — parse_date has no injected clock parameter.
- `polylogue/core/dates.py:37` — RELATIVE_BASE uses ambient datetime.now.
- `polylogue/archive/query/expression.py:2440` — Query grammar recognizes relative-date literals.
- `polylogue/archive/query/spec.py:498` — SessionQuerySpec.from_params is the central query-spec constructor.
- `polylogue/insights/temporal_source.py:66` — classify_profile_hwm_source promotes any updated_at to provider_ts.
- `polylogue/insights/temporal_source.py:97` — classify_aggregate_hwm_source currently collapses all non-empty source updates to provider_ts.

## Implementation plan

1. Implementation shape:
2. 1. Generate a table of public payload fields/DB columns with type, nullable, default, sample null density, sample zero density, and known evidence source.
3. 2. Classify each as true-zero-safe, unknown-when-absent, not-applicable, text-derived, or needs-contract.
4. 3. Emit JSON + Markdown artifact under docs/audits or `.agent/reports`.
5. 4. File follow-up beads only for confirmed high-risk fields.

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: A committed evidence artifact (CSV/JSON matrix + short markdown, under .agent/scratch/research/ or demo-shelf) reports, per (column, origin, month), the null/unknown/populated counts, populated_pct, and top-5 values, with denominators correct (tool_result_* over tool_result rows, material_origin over authored messages).
- Acceptance proof: 2.
- Acceptance proof: Each column carries a go/no-go verdict — structural-ready vs structural-with-caveat vs keep-heuristic — that the b0b heuristic->structural sweep bead consumes, plus a per-origin coverage-caveat sentence for outcome analytics.
- Acceptance proof: 3.
- Acceptance proof: The exact SQL is saved with the artifact and reconciles to SELECT COUNT(*) on each source table.
- Acceptance proof: 4.
- Acceptance proof: No product-code mutation

## Verification commands

- `Run the census command/artifact generation on the active fixture/live copy read-only. Review artifact against 9e5.29 field-contract work.`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
