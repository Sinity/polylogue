Title: "Claude Workflow artifact admission through OriginSpec (2qx.2)"

Result ZIP: `mandate-01-claude-workflow-admission-r01.zip`

## Mission

Implement the mandate-critical `polylogue-2qx.2` slice. Read its full Bead
record and `polylogue-2qx.1.2` before touching code. Extend the existing
OriginSpec/admission path; do not invent a Workflow-only archive or second
registry. The configured Claude source must admit coordinator invocations,
workflow state, journal revisions, paired metadata sidecars, and adopt
manifests, with raw evidence, coverage, parsing, materialization, provenance,
and semantic-reparse implications in one contract.

The synthetic `wf_54d4fb2e-841` fixture must prove one run, four coordinator
invocations, 50 content-keyed calls, 91 attempt transcripts plus 91 metadata
sidecars, 65 results across 49 completed keys, one unresolved key, and the
final result. Exclude 38 unrelated coordinator children. Generated prompts are
not human-authored; direct prompts remain positively human-authored. Missing
members are degraded/unresolved, never fabricated.

## Constraints

- Preserve raw revisions in source.db and follow the derived-tier rebuild plan.
- Every count/link carries evidence refs; membership cannot be inferred from
  parent-child counts.
- Include a quantified reparse plan without accessing private live data.

## Deliverable emphasis

HANDOFF.md maps every 2qx.2 AC to a production route/test. PATCH.diff is
apply-ready. TESTS.md names real admission dependencies and failure mutations.
