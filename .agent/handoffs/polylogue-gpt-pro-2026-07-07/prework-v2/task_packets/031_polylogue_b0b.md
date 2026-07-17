# 031. polylogue-b0b — Replace remaining keyword outcome/pathology heuristics with structural evidence

Priority/type/status: **P2 / task / open**. Lane: **10-analytics-experiments**. Release: **I-analytics-experiments**. Readiness: **blocked-hard**.

Hard blockers: polylogue-9e5.3, polylogue-9e5.9

## What the bead says

Includes High-Value backlog: wherever detectors/insights still regex prose for outcomes, consume tool_result_is_error/exit_code instead, with per-origin coverage caveats where structure is absent. Inventory first (grep detector modules for prose-pattern matching), then convert or explicitly label each as heuristic-tier. The construct-validity moat depends on this stratum staying honest.

## Existing design note

Inventory first: rg detector/insight modules (insights/, schemas/code_detection/, pathology surfaces) for prose-pattern matching — regexes over message/block text that infer outcomes ('error', 'failed', 'fixed', success words). For each hit, one of three verdicts: (1) CONVERT — a structural field exists (tool_result_is_error, tool_result_exit_code, action-view pairing) -> consume it, with per-origin coverage caveat where the origin lacks structure (web exports have no exit codes); (2) LABEL — no structural equivalent -> keep but tag the emitting measure evidence_tier=text_derived (9e5.30's provenance contract) so consumers see the tier; (3) DELETE — the heuristic feeds nothing load-bearing. The recovery-digest fabrication (regex _events_from_text inventing 'PR #123 merged') is the standing cautionary fixture — no prose claim without tier labeling.

## Acceptance criteria

Inventory table committed (module:line -> verdict); every CONVERT lands with a coverage caveat; every retained heuristic emits evidence_tier=text_derived; the fabrication fixture (prose claiming an event that structure contradicts) does not surface as fact on any public payload. Verify: devtools test -k 'pathology or outcome' + the inventory script re-run.

## Static mechanism / likely defect

Bead design says keyword outcome/pathology heuristics should be replaced by structural evidence. Static source likely contains regex/keyword classifiers in insight transforms/forensics/report scripts. These are useful as fallback, not as primary evidence.

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

## Implementation plan

1. Implementation shape:
2. 1. Inventory outcome/pathology fields and their current evidence source: tool result exit code, test runner structured status, command result, PR/commit state, assistant prose, keyword heuristic.
3. 2. Define evidence precedence: structured tool/test status > persisted action outcome > explicit user/agent annotation > text-derived prose > keyword fallback.
4. 3. Add evidence_class/source fields to outcome/pathology records.
5. 4. Convert the highest-impact classifiers first: failed tool call, test failure, silent proceed, retry loop, abandonment/resume.
6. 5. Keep keyword fallback but mark `text_derived`/`heuristic` and exclude it from strong numeric claims unless caveated.

## Tests to add

- structured failed test beats optimistic prose.
- success prose does not override nonzero exit status.
- keyword-only case still produces fallback but labelled heuristic/text-derived.
- aggregate report footnotes counts by evidence tier.

## Verification commands

- ``devtools test tests/unit/insights tests/unit/archive -k 'outcome or pathology or tool_result or evidence_class'``

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
