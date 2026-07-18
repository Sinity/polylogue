Title: "Observed repository effects and evaluated satisfaction (1vpm.6.2)"

Result ZIP: `mandate-02-observed-work-effects-r01.zip`

## Mission

Implement `polylogue-1vpm.6.2` on the provider-neutral work topology/claim
substrate and 2qx.2 Claude artifacts. Connect claims to observed git commits,
GitHub PRs, Beads history/state, artifacts, and verification receipts, while
keeping evaluated acceptance satisfaction distinct from claims and effects.

Support direct identifiers/evidence, repository snapshots, squash merges,
branch-local Beads state, many-to-many links, corrections, and uncertainty.
Time/file overlap is candidate-only. A seeded query must answer which sessions
created, edited, claimed, or closed a requested Bead via direct archived refs.
Mutations must fail if claims become effects or time overlap becomes causality.

## Constraints

- Reuse ObjectRef, EvidenceRef, session events, assertions, ObservedEvent,
  ProjectedRun, and existing generic graph/query infrastructure.
- Use synthetic/committed fixtures only; mark live checks unverified.
- Missing 2qx.2 evidence is degraded/unresolved, not invented.

## Deliverable emphasis

Provide cohesive PATCH.diff and real-route tests. HANDOFF.md includes AC and
edge matrices, fixture identities, claim/effect/evaluation separation, and the
exact residual for z9gh.7.
