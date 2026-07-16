# 08. polylogue-9e5.30 — Tag prose-mined forensic fields as text-derived

Priority: **P1**  
Lane: **evidence-honesty**  
Readiness: **ready-now / model-and-renderer**

Depends on packet(s): polylogue-9e5.28

## Why this is urgent / critical-path

The system already prevents fabricating events from prose, but still surfaces prose-mined refs and decisions as ordinary fields. Users and downstream agents need to know when a fact came from text interpretation rather than structured evidence.

## Static diagnosis / likely mechanism

Mechanism: forensic transform models (`ForensicIndexEntry`, `ToolSummary`, `DecisionCandidate`) carry extracted refs/labels/test evidence/decisions without per-field evidence class. Source anchors: `polylogue/insights/transforms.py:140`, `:192`, `:280`, extraction functions `:1573+` and `:1923+`.

## Implementation plan

Implementation shape:
1. Define a small evidence-class vocabulary: `raw_evidence`, `structured_tool_result`, `text_derived`, `synthetic`.
2. Add either explicit fields or `field_evidence: dict[str, EvidenceClass]` to `ToolSummary`, `DecisionCandidate`, and `ForensicIndexEntry`.
3. Mark refs extracted from command/output/message prose (`commit_refs`, `pr_refs`, `issue_refs`, `test_evidence`, prose decisions) as `text_derived` unless the parser has structured provider/tool metadata.
4. Structured tool result status/exit code can be `structured_tool_result`.
5. Update report/render surfaces so text-derived fields get a visible caveat. Do not suppress them; label them.
6. Add a downgrade rule: a finding backed only by text-derived fields cannot render as a hard fact without caveat.

## Test plan

Tests:
- synthetic session with a SHA/PR/test mention in assistant prose yields extracted refs with `text_derived` evidence class.
- structured tool-result exit status remains structured, not text-derived.
- rendered report includes a text-derived caveat.
- serialization remains backward-compatible for old rows where evidence class is absent, defaulting to `text_derived` for prose-mined fields.

## Verification command / proof

`devtools test tests/unit/insights/test_transforms.py -k 'text_derived or prose or forensic or evidence_class'` plus renderer tests for the affected report surface.

## Pitfalls

Do not treat text-derived as useless. The important invariant is label + caveat, not deletion.

## Files/functions to inspect or touch

- `polylogue/insights/transforms.py:140`
- `polylogue/insights/transforms.py:192`
- `polylogue/insights/transforms.py:280`
- `polylogue/insights/transforms.py:1573+`
- `polylogue/insights/transforms.py:1923+`
- `render/report modules that display forensic fields`
