# 132. polylogue-38x — Reconcile archived audit residue against current source

Priority/type/status: **P2 / task / open**. Lane: **00-trust-floor**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Older archived audits under .agent/archive/conductor-history/2026-07-01 still contain valuable findings that are not all represented as executable Beads. This task is to re-check the remaining concrete findings against current source and either close them as stale/fixed or split/link them to the owning subsystem bead. Seed findings: construct-validity audit flags Codex FORK vs RESUME conflation, multi-meta CONTINUATION as proxy, scalar paste detection flattening exact vs fallback, timestamp fallback to epoch-zero, Codex token lane normalizer divergence; fanout audit flags transcript pagination/batch/stream reads bypassing prefix composition, child usage rollups counting inherited prefix, MCP scoped aggregates capped by page limit, ChatGPT image/asset-only nodes dropped before block construction, Antigravity non-UTF-8 drop; insights dissection flags dead phase_type/confidence and heuristic confidence/provenance flattening. Some classes are already covered by lineage, provider-usage, insights-as-declared-views, attachment, and DSL Beads; this task exists to make the residual current/stale classification explicit rather than leaving it buried in archived markdown.

## Existing design note

Run this as a source-grounded reconciliation pass, not as implementation by memory. For each seed finding: inspect current source and tests; classify fixed, still-live, subsumed-by-existing-bead, or split-needed; cite file paths/functions and the owning bead. Live bugs should be turned into narrow child/linked Beads under the relevant parent (lineage, provider usage, insights-as-declared-views, provider parsers, MCP/query surface). Stale/fixed findings should name the commit/test or current source behavior that invalidates the old audit. The final artifact can be a concise markdown note under .agent/scratch/research plus Beads notes; do not leave decisions only in chat.

## Acceptance criteria

A current-source reconciliation table exists for every seed finding from the archived construct-validity/fanout/insights audits; every still-live finding is linked to an owning executable Beads issue or split into one; every stale/fixed finding cites current source or tests; no archived audit item in the seed list remains only as untriaged markdown; bd ready no longer depends on reading .agent/archive to discover these issues.

## Static mechanism / likely defect

Issue description localizes the mechanism: Older archived audits under .agent/archive/conductor-history/2026-07-01 still contain valuable findings that are not all represented as executable Beads. This task is to re-check the remaining concrete findings against current source and either close them as stale/fixed or split/link them to the owning subsystem bead. Seed findings: construct-validity audit flags Codex FORK vs RESUME conflation, multi-meta CONTINUATION as proxy, scalar paste detection flattening exact vs fallback, timestamp fallback to epoch-zero,… Design direction: Run this as a source-grounded reconciliation pass, not as implementation by memory. For each seed finding: inspect current source and tests; classify fixed, still-live, subsumed-by-existing-bead, or split-needed; cite file paths/functions and the owning bead. Live bugs should be turned into narrow child/linked Beads under the relevant parent (lineage, provider usage, insights-as-declared-views, provider parsers, MCP…

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

1. Run this as a source-grounded reconciliation pass, not as implementation by memory.
2. For each seed finding: inspect current source and tests
3. classify fixed, still-live, subsumed-by-existing-bead, or split-needed
4. cite file paths/functions and the owning bead.
5. Live bugs should be turned into narrow child/linked Beads under the relevant parent (lineage, provider usage, insights-as-declared-views, provider parsers, MCP/query surface).
6. Stale/fixed findings should name the commit/test or current source behavior that invalidates the old audit.
7. The final artifact can be a concise markdown note under .agent/scratch/research plus Beads notes

## Tests to add

- Acceptance proof: A current-source reconciliation table exists for every seed finding from the archived construct-validity/fanout/insights audits
- Acceptance proof: every still-live finding is linked to an owning executable Beads issue or split into one
- Acceptance proof: every stale/fixed finding cites current source or tests
- Acceptance proof: no archived audit item in the seed list remains only as untriaged markdown
- Acceptance proof: bd ready no longer depends on reading .agent/archive to discover these issues.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
