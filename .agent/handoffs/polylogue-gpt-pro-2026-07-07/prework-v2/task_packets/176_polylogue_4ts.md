# 176. polylogue-4ts — Session lineage truth: shared content stored once, counted once, composed correctly

Priority/type/status: **P1 / epic / open**. Lane: **03-lineage-compaction-truth**. Release: **F-lineage-compaction**. Readiness: **epic-needs-child-closure**.

## What the bead says

Fork/resume/compaction share content; storage+aggregates ignored it. v12-v14 landed prefix-dedup + composition; this program owns the residuals. Design doc docs/design/session-lineage-model.md. Operator: 'broken unless modeled correctly' — correctness > demo-ladder. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

## Acceptance criteria

Terminal state: shared content stored once (prefix dedup verified on live archive), counted once (4ts.2), composed correctly (read paths serve full logical transcripts across the branch-point matrix); external citation of archive counts uses logical grain with the physical figure footnoted.

## Static mechanism / likely defect

Issue description localizes the mechanism: Fork/resume/compaction share content; storage+aggregates ignored it. v12-v14 landed prefix-dedup + composition; this program owns the residuals. Design doc docs/design/session-lineage-model.md. Operator: 'broken unless modeled correctly' — correctness > demo-ladder. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

## Source anchors to inspect first

- `polylogue/archive/session/threads.py` — Session/thread lineage read and composition model.
- `polylogue/insights/topology.py` — Topology/lineage derived insight code.
- `polylogue/daemon/lineage_startup.py` — Daemon lineage startup/convergence path.
- `polylogue/archive/coverage.py` — Completeness/truncation cues live here.
- `polylogue/insights/postmortem.py` — Compaction/continuation postmortem evidence is mined here.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.

## Implementation plan

1. Inventory open child beads and map them to the invariant named by the epic.
2. Add/verify a terminal acceptance checklist for the epic rather than landing broad code.
3. Close only after child beads are closed or explicitly split out with new blockers.

## Tests to add

- Acceptance proof: Terminal state: shared content stored once (prefix dedup verified on live archive), counted once (4ts.2), composed correctly (read paths serve full logical transcripts across the branch-point matrix)
- Acceptance proof: external citation of archive counts uses logical grain with the physical figure footnoted.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
