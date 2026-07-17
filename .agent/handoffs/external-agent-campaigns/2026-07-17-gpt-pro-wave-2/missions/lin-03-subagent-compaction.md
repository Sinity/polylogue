Title: "Stop mis-parenting Task-subagent self-compactions: prefix-membership test before parent assignment (4ts.3)"

Result ZIP: `lin-03-subagent-compaction-r01.zip`

## Mission

Implement bead `polylogue-4ts.3` (P1 — read its full record; root cause is
code-confirmed). The Claude Code parser's `agent-acompact-*` prefix
classifier assigns `parent = main-session` UNCONDITIONALLY, but ~39 of 187
such files in the live corpus are Task-SUBAGENT self-compactions (<90%
content overlap with the main session; 9 at 0% overlap). Consequence: the
parser asserts the wrong parent, and lineage composition prepends the WRONG
transcript — an agent reading the composed session sees a main-session
prefix that never belonged to it. This corrupts exactly the continuity
evidence the archive exists to preserve.

Fix (bead design is authority):

1. In the Claude parser's compaction classification (find it via the
   `agent-acompact` handling in `polylogue/sources/parsers/` —
   claude-code streaming/session parser), BEFORE assigning the main
   session as parent: test prefix content/UUID membership against the main
   session (or detect a fresh task-prompt head). On mismatch: treat as a
   fresh subagent — sidechain topology, NO inherited prefix.
2. Regression fixtures for both cases (bead AC): (a) a TRUE main-session
   `agent-acompact-*` whose prefix genuinely belongs to the main session →
   parent assignment + prefix-sharing inheritance preserved exactly as
   today; (b) a Task-subagent self-compaction (build from the 0%-overlap
   shape) → sidechain topology, spawned-fresh inheritance, no prefix.
   Derive fixture structure from real shapes but fully synthetic content
   (public repo).
3. **Reclassification path for existing data**: index.db is rebuildable —
   the fix takes effect on reparse. Deliver the verification query the
   integrator runs post-rebuild (count of agent-acompact sessions by
   parent-assignment class before/after; the ~39 misparented files should
   flip) and note that `session_links` rows re-resolve on save (writer
   stores divergent tails; `resolve_session_links_for_session` runs per
   save — confirm the reparse route re-evaluates them).
4. Sibling-route check (the repo's recurring failure shape is
   fixed-one-path-missed-the-twin): confirm whether the memory-bounded
   streaming path for multi-GiB Claude Code JSONL shares the classifier or
   duplicates it — if duplicated, fix both and add the divergence test.

## Constraints

- `Role`/`MaterialOrigin` semantics untouched; this is topology/parentage
  only.
- No schema changes; `session_links` vocabulary (prefix-sharing vs
  spawned-fresh, TopologyEdgeStatus) already expresses both outcomes.
- Detection must be shape/structural (content/UUID membership), never
  filename-only (detection tightness discipline in
  `sources/dispatch.py`).

## Deliverable emphasis

HANDOFF.md: classifier change mechanism, membership-test cost note (it
must not blow up streaming-path memory), fixture derivations, the
post-rebuild verification queries with expected count movement, and the
sibling-route audit result.
