# 080. polylogue-4ts.3 — Distinguish subagent auto-compaction from main-session acompact

Priority/type/status: **P2 / bug / open**. Lane: **03-lineage-compaction-truth**. Release: **F-lineage-compaction**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

agent-acompact-* also fires for Task-subagent self-compaction (~39/187 files <90% overlap; 9 at 0%) — parser assigns wrong parent; composition prepends the wrong transcript. Test prefix content/UUID membership before assigning the main session as parent. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

## Existing design note

Code-confirmed (gh#2471): the agent-acompact-* prefix classifier assigns parent=main-session unconditionally, but ~39/187 such files are Task-subagent self-compactions (<90% content overlap with the main session; 9 at 0%). Fix in the Claude parser's compaction-classification: before assigning the main session as parent, test prefix content/UUID membership against the main session (or detect a fresh task-prompt head); on mismatch treat as a fresh subagent (sidechain topology, no inherited prefix). Regression fixtures: one true main-session acompact, one subagent self-compact (e.g. modeled on 1796b263's '12-commit testing overhaul' subagent). Acceptance: composition never prepends the main transcript onto a subagent's compaction.

## Acceptance criteria

1. The Claude parser's compaction classifier tests prefix content/UUID membership (or detects a fresh task-prompt head) before assigning the main session as parent; on mismatch it treats the record as a fresh subagent with sidechain topology and no inherited prefix. 2. Regression fixtures cover both cases: (a) a true main-session agent-acompact-* whose prefix content/UUIDs are members of the main session -> parent=main; (b) a subagent self-compaction (<90% overlap, modeled on 1796b263's '12-commit testing overhaul' subagent) -> sidechain, no inherited prefix. 3. A composed read of the (b) session never prepends the main transcript (explicit assertion). Verify: focused Claude-parser tests pass (`devtools test` selection); a live re-ingest of the ~39 mismatched files drives mis-parented subagent-acompacts to zero, confirmed by a probe count over session_links/topology_edges.

## Static mechanism / likely defect

Bead hints point to Claude parser behavior around `agent-acompact-*` parent assignment. Static next step is to inspect parser code with `rg 'agent-acompact|acompact|compaction|parent' polylogue/sources/parsers` and find where parent/lineage kind is assigned.

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

1. Implementation shape:
2. 1. Locate parser branch that classifies Claude Code auto-compaction/session ids.
3. 2. Add an explicit lineage event kind or flag: `main_compaction`, `subagent_auto_compaction`, `subagent_spawn`, etc.
4. 3. Parent assignment: a subagent auto-compact event should attach to the subagent/session tree, not become a main-session compaction boundary.
5. 4. Downstream lineage/composed-session renderers should show subagent compaction separately and not truncate/restart main session effective context.
6. 5. Migration/backfill: old parsed rows may need a derived-lineage rebuild, not source mutation.

## Tests to add

- fixture with main session + subagent `agent-acompact-*`: main compaction count remains zero or unchanged; subagent compaction count increments.
- composed lineage tree renders subagent compaction under subagent.
- existing main compaction fixture still works.
- aggregate counts distinguish both classes.

## Verification commands

- ``devtools test tests/unit/sources/test_claude* tests/unit/lineage -k 'acompact or compaction or subagent'``

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
