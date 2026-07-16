# 032. polylogue-b0b.1 — Fix substring false-positives in work-event keyword classifier + inventory activity-type label as heuristic-tier

Priority/type/status: **P2 / task / open**. Lane: **10-analytics-experiments**. Release: **I-analytics-experiments**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

polylogue/archive/session/extraction.py:_text_signal_from_lowered_text (241-262) matches keyword tables with naive `pattern in lowered_text`, so tokens embed into unrelated words: 'fix'->prefix/suffix, 'test'->latest/contest, 'plan'->explanation/airplane, 'data'->update/validate/metadata, 'spec'->respect/inspect, 'move'->remove, 'config'->reconfigured. This silently mislabels work-event heuristic_label and the noise is invisible in the hardcoded confidence float. Two changes: (1) match on word boundaries (compile the pattern tables to `\b(?:...)\b` regexes or tokenize+set-membership) so substring collisions stop; keep multiword phrases ('stack trace','should we') as phrase matches. (2) b0b's inventory is scoped to 'outcome/pathology heuristics' — the work-event activity-TYPE classifier (planning/debugging/testing/...) is neither, so explicitly record it in the b0b heuristic-tier inventory with a per-origin coverage caveat, since unlike outcomes there is no structural ground truth to convert it to (it stays heuristic-tier by nature). Feeds 9e5.9's labeled corpus as a before/after precision point.

## Acceptance criteria

1. _TEXT_SIGNAL_TABLE matching uses word boundaries; a regression test asserts 'prefix'/'latest'/'explanation'/'metadata'/'remove' do NOT trigger fix/test/plan/data/move signals while genuine 'fix the bug'/'run pytest'/'let us plan' do. 2. The work-event activity-type classifier appears in the b0b heuristic-tier inventory with an explicit 'stays heuristic (no structural ground truth)' note and a coverage caveat. 3. No change to the confidence literals in this bead (calibration is 9e5.9); this is the correctness fix only. Verify: devtools test on tests covering extraction._classify_range / _text_signal_from_lowered_text.

## Static mechanism / likely defect

Design direction: polylogue/archive/session/extraction.py:_text_signal_from_lowered_text (241-262) matches keyword tables with naive `pattern in lowered_text`, so tokens embed into unrelated words: 'fix'->prefix/suffix, 'test'->latest/contest, 'plan'->explanation/airplane, 'data'->update/validate/metadata, 'spec'->respect/inspect, 'move'->remove, 'config'->reconfigured. This silently mislabels work-event heuristic_label and the noise…

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

1. polylogue/archive/session/extraction.py:_text_signal_from_lowered_text (241-262) matches keyword tables with naive `pattern in lowered_text`, so tokens embed into unrelated words: 'fix'->prefix/suffix, 'test'->latest/contest, 'plan'->explanation/airplane, 'data'->update/validate/metadata, 'spec'->respect/inspect, 'move'->remove, 'config'->reconfigured.
2. This silently mislabels work-event heuristic_label and the noise is invisible in the hardcoded confidence float.
3. Two changes: (1) match on word boundaries (compile the pattern tables to `\b(?:...)\b` regexes or tokenize+set-membership) so substring collisions stop
4. keep multiword phrases ('stack trace','should we') as phrase matches.
5. (2) b0b's inventory is scoped to 'outcome/pathology heuristics' — the work-event activity-TYPE classifier (planning/debugging/testing/...) is neither, so explicitly record it in the b0b heuristic-tier inventory with a per-origin coverage caveat, since unlike outcomes there is no structural ground truth to convert it to (it stays heuristic-tier by nature).
6. Feeds 9e5.9's labeled corpus as a before/after precision point.

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: _TEXT_SIGNAL_TABLE matching uses word boundaries
- Acceptance proof: a regression test asserts 'prefix'/'latest'/'explanation'/'metadata'/'remove' do NOT trigger fix/test/plan/data/move signals while genuine 'fix the bug'/'run pytest'/'let us plan' do.
- Acceptance proof: 2.
- Acceptance proof: The work-event activity-type classifier appears in the b0b heuristic-tier inventory with an explicit 'stays heuristic (no structural ground truth)' note and a coverage caveat.
- Acceptance proof: 3.
- Acceptance proof: No change to the confidence literals in this bead (calibration is 9e5.9)
- Acceptance proof: this is the correctness fix only.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
