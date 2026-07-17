# 177. polylogue-gjg — Compaction lifecycle: pre-compaction snapshot, loss forensics, post-compaction re-grounding

Priority/type/status: **P2 / epic / open**. Lane: **03-lineage-compaction-truth**. Release: **F-lineage-compaction**. Readiness: **blocked-hard**.

Hard blockers: polylogue-4ts.5, polylogue-d1y

## What the bead says

Compaction is where the OS-like context-management vision meets the harness's own memory management, and today Polylogue only observes its AFTERMATH (acompact lineage edges, v12). Three gaps: nothing snapshots the full pre-compaction context (the harness summarizes-and-discards; what was lost is unknowable after the fact); nothing measures what compaction costs (which facts/decisions/refs present before are absent after — the construct behind every 'the agent forgot' complaint); and re-grounding after compaction is left to the harness's own summary instead of the archive's evidence (37t.4 deliberately injects nothing on compact — right call for volume, but it leaves the re-grounding opportunity unused: the archive HAS the full pre-compact transcript).

## Existing design note

(1) SNAPSHOT: wire the PreCompact hook (VERIFY availability/payload in current Claude Code — the hook catalog moves; Codex equivalent via app-server events ox0) to capture the full context state as an artifact (session-linked, blob-stored, content-addressed — dedup makes repeated compactions cheap). Fallback without the hook: the JSONL up to the compaction boundary IS the pre-state; the snapshot adds what JSONL lacks (the exact assembled context, if the payload provides it). (2) FORENSICS: a compaction-loss measure (9l5.7-registered): diff pre-snapshot against the post-compact continuation's early context — structurally extractable items (file paths, refs, tool outcomes, decisions marked via 37t.2 notation) present-before/absent-after; corpus-level epidemiology ('compaction loses X% of marked decisions, median') is finding-grade material. (3) RE-GROUNDING: an opt-in SessionStart(source=compact) lane that injects a compact delta-restoration: the top-K lost-but-referenced items as refs (resolve_ref expandable), budget ~200 tokens, only items the loss-forensics ranks high — jgp-compliant because it is keyed to measured loss, not generic recap. Arm-able as an experiment (stc): compact sessions with vs without re-grounding, outcome comparison. (4) This bead + 37t.3 (reboot-with-refs) + yps (freshness) together are the 'controlled handoff' story: voluntary handoff (37t.3), involuntary compaction (this), cross-session resumption (briefs) — document the triad in the 37t epic as the OS-vision map.

## Acceptance criteria

PreCompact snapshots land for real compactions on the operator machine (or the JSONL-boundary fallback is implemented and labeled); the loss measure runs corpus-wide with tier=structural and renders an epidemiology table; re-grounding injects only under the flag and its arm comparison is defined as an ExperimentSpec; 37t epic description carries the handoff-triad map.

## Static mechanism / likely defect

Issue description localizes the mechanism: Compaction is where the OS-like context-management vision meets the harness's own memory management, and today Polylogue only observes its AFTERMATH (acompact lineage edges, v12). Three gaps: nothing snapshots the full pre-compaction context (the harness summarizes-and-discards; what was lost is unknowable after the fact); nothing measures what compaction costs (which facts/decisions/refs present before are absent after — the construct behind every 'the agent forgot' complaint); and re-grounding after compaction i… Design direction: (1) SNAPSHOT: wire the PreCompact hook (VERIFY availability/payload in current Claude Code — the hook catalog moves; Codex equivalent via app-server events ox0) to capture the full context state as an artifact (session-linked, blob-stored, content-addressed — dedup makes repeated compactions cheap). Fallback without the hook: the JSONL up to the compaction boundary IS the pre-state; the snapshot adds what JSONL lack…

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

1. (1) SNAPSHOT: wire the PreCompact hook (VERIFY availability/payload in current Claude Code — the hook catalog moves
2. Codex equivalent via app-server events ox0) to capture the full context state as an artifact (session-linked, blob-stored, content-addressed — dedup makes repeated compactions cheap).
3. Fallback without the hook: the JSONL up to the compaction boundary IS the pre-state
4. the snapshot adds what JSONL lacks (the exact assembled context, if the payload provides it).
5. (2) FORENSICS: a compaction-loss measure (9l5.7-registered): diff pre-snapshot against the post-compact continuation's early context — structurally extractable items (file paths, refs, tool outcomes, decisions marked via 37t.2 notation) present-before/absent-after
6. corpus-level epidemiology ('compaction loses X% of marked decisions, median') is finding-grade material.
7. (3) RE-GROUNDING: an opt-in SessionStart(source=compact) lane that injects a compact delta-restoration: the top-K lost-but-referenced items as refs (resolve_ref expandable), budget ~200 tokens, only items the loss-forensics ranks high — jgp-compliant because it is keyed to measured loss, not generic recap.

## Tests to add

- Acceptance proof: PreCompact snapshots land for real compactions on the operator machine (or the JSONL-boundary fallback is implemented and labeled)
- Acceptance proof: the loss measure runs corpus-wide with tier=structural and renders an epidemiology table
- Acceptance proof: re-grounding injects only under the flag and its arm comparison is defined as an ExperimentSpec
- Acceptance proof: 37t epic description carries the handoff-triad map.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
