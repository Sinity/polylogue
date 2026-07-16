# 178. polylogue-1vpm — Work-graph units: delegation, episode, artifact edges — the derived units between lineage and analysis

Priority/type/status: **P2 / epic / open**. Lane: **10-analytics-experiments**. Release: **I-analytics-experiments**. Readiness: **epic-needs-child-closure**.

## What the bead says

Three convergent derived units that make "what work actually happened" queryable, sitting ABOVE within-provider lineage (session_links stays the leaf truth) and BELOW analysis runs. (1) DELEGATION: provider-neutral rows mined from Claude Task tool_use blocks (excluding agent-acompact-* which is compaction, not delegation), Codex source.subagent.thread_spawn (forked_from_id alone proves parentage, NOT subagent-ness), session_runs.role=subagent, and SubagentReport extraction — parent/child refs, instruction/result block refs, link_status incl. unresolved (failed delegation must not vanish), evidence refs. Assertions ANNOTATE delegations (stern/failed/under-specified); they must not BE the delegation graph — keeping the central noun implicit defeats the DSL/composer direction. (2) EPISODE: one logical task stitched across sessions/tools/time via 4-signal edge scorer (repo/cwd hard prior; repo-CONDITIONED time kernel — 6h same-repo plausible, 5min cross-repo is a context switch; session-summary embedding as soft recall; shared-hard-artifact as the strongest signal) with a HARD false-merge floor: candidate edges (embedding+time only) never default-render as one episode; only linked (topology-proven) and corroborated (>=2 independent signals incl. one hard non-semantic) merge. False merges are expensive, missed stitches are cheap — deliberately under-stitch and calibrate from operator confirm/split/reject assertions. Commits/PRs with no in-session window can still join via artifact overlap (produced_refs). (3) ARTIFACT EDGES: generic produced/consumed/mentioned/reported_by/derived_from edges from sessions/actions/runs/delegations to artifacts — no special-casing .agent/scratch paths; reports, evidence packs, sidecars all one relation. All three become query units + projection presets (delegation-card, layout:episode interleaved cross-tool transcript with stitch-evidence badges at boundaries). Verbatim wave specs preserved: .agent/scratch/corpus-gpt-pro-2026-07-06/bundles/ (delegation: bundle-4 L723; episode: bundle-6 L466; units: bundle-2 L466, bundle-5 L2154, bundle-6 L715).

## Acceptance criteria

delegations where / episodes where work as terminal units with set-algebra participation; fixtures prove no false subagent from bare forked_from_id and no acompact false-delegation; episode default render includes only linked+corroborated tiers; per-edge signal contributions auditable in evidence_json; operator stitch decisions round-trip as assertions and constrain rebuilds.

## Static mechanism / likely defect

Issue description localizes the mechanism: Three convergent derived units that make "what work actually happened" queryable, sitting ABOVE within-provider lineage (session_links stays the leaf truth) and BELOW analysis runs. (1) DELEGATION: provider-neutral rows mined from Claude Task tool_use blocks (excluding agent-acompact-* which is compaction, not delegation), Codex source.subagent.thread_spawn (forked_from_id alone proves parentage, NOT subagent-ness), session_runs.role=subagent, and SubagentReport extraction — parent/child refs, instruction/result b…

## Source anchors to inspect first

- `polylogue/archive/actions/followup.py` — Action/followup classification is a real structural analytics input.
- `polylogue/archive/actions/fields.py` — Action fields determine what can be measured without prose heuristics.
- `polylogue/insights/registry.py:294` — Insight registry should become measure/product registry input.
- `scripts/agent_forensics.py` — Existing forensics script is a proof artifact and candidate product surface.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.

## Implementation plan

1. Inventory open child beads and map them to the invariant named by the epic.
2. Add/verify a terminal acceptance checklist for the epic rather than landing broad code.
3. Close only after child beads are closed or explicitly split out with new blockers.

## Tests to add

- Acceptance proof: delegations where / episodes where work as terminal units with set-algebra participation
- Acceptance proof: fixtures prove no false subagent from bare forked_from_id and no acompact false-delegation
- Acceptance proof: episode default render includes only linked+corroborated tiers
- Acceptance proof: per-edge signal contributions auditable in evidence_json
- Acceptance proof: operator stitch decisions round-trip as assertions and constrain rebuilds.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.
- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
