# 23. polylogue-4ts.3 — Separate subagent auto-compaction from main-session compaction in lineage

Priority: **P2**  
Lane: **lineage-truth**  
Readiness: **needs-source-confirmation then parser patch**

## Why this is urgent / critical-path

Compaction events affect what context was available. Subagent auto-compaction should not be represented as a main-session compaction boundary.

## Static diagnosis / likely mechanism

Bead hints point to Claude parser behavior around `agent-acompact-*` parent assignment. Static next step is to inspect parser code with `rg 'agent-acompact|acompact|compaction|parent' polylogue/sources/parsers` and find where parent/lineage kind is assigned.

## Implementation plan

Implementation shape:
1. Locate parser branch that classifies Claude Code auto-compaction/session ids.
2. Add an explicit lineage event kind or flag: `main_compaction`, `subagent_auto_compaction`, `subagent_spawn`, etc.
3. Parent assignment: a subagent auto-compact event should attach to the subagent/session tree, not become a main-session compaction boundary.
4. Downstream lineage/composed-session renderers should show subagent compaction separately and not truncate/restart main session effective context.
5. Migration/backfill: old parsed rows may need a derived-lineage rebuild, not source mutation.

## Test plan

Tests:
- fixture with main session + subagent `agent-acompact-*`: main compaction count remains zero or unchanged; subagent compaction count increments.
- composed lineage tree renders subagent compaction under subagent.
- existing main compaction fixture still works.
- aggregate counts distinguish both classes.

## Verification command / proof

`devtools test tests/unit/sources/test_claude* tests/unit/lineage -k 'acompact or compaction or subagent'`

## Pitfalls

Do not key solely on id string if structured provider metadata exists; prefer provider event shape first and id prefix as fallback.

## Files/functions to inspect or touch

- `polylogue/sources/parsers/claude/*`
- `polylogue/lineage or session lineage modules`
- `lineage composition/render tests`
