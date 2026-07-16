# 26. polylogue-b0b — Replace keyword-only outcome/pathology heuristics with structural evidence where available

Priority: **P2**  
Lane: **evidence-honesty**  
Readiness: **spec-first then targeted code**

Depends on packet(s): polylogue-9e5.30

## Why this is urgent / critical-path

Behavioral analytics over agent work should not infer outcomes purely from words when structured tool/test/exit data exists.

## Static diagnosis / likely mechanism

Bead design says keyword outcome/pathology heuristics should be replaced by structural evidence. Static source likely contains regex/keyword classifiers in insight transforms/forensics/report scripts. These are useful as fallback, not as primary evidence.

## Implementation plan

Implementation shape:
1. Inventory outcome/pathology fields and their current evidence source: tool result exit code, test runner structured status, command result, PR/commit state, assistant prose, keyword heuristic.
2. Define evidence precedence: structured tool/test status > persisted action outcome > explicit user/agent annotation > text-derived prose > keyword fallback.
3. Add evidence_class/source fields to outcome/pathology records.
4. Convert the highest-impact classifiers first: failed tool call, test failure, silent proceed, retry loop, abandonment/resume.
5. Keep keyword fallback but mark `text_derived`/`heuristic` and exclude it from strong numeric claims unless caveated.

## Test plan

Tests:
- structured failed test beats optimistic prose.
- success prose does not override nonzero exit status.
- keyword-only case still produces fallback but labelled heuristic/text-derived.
- aggregate report footnotes counts by evidence tier.

## Verification command / proof

`devtools test tests/unit/insights tests/unit/archive -k 'outcome or pathology or tool_result or evidence_class'`

## Pitfalls

Coordinate with 9e5.30 so prose-derived fields share the same evidence-class vocabulary.

## Files/functions to inspect or touch

- `polylogue/insights/transforms.py`
- `scripts/agent_forensics.py or promoted forensics surface`
- `work-event/outcome classifiers`
- `report renderers`
