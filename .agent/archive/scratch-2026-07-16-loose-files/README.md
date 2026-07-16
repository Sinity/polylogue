# Archived loose scratch files (2026-07-16)

40 top-level files from `.agent/scratch/`, moved here after confirming zero
open-or-closed bead references (path-anchored grep against
`.beads/issues.jsonl`). All represent completed, one-off work:

- `bd-*.py` (17 files): one-off bead-graph mutation scripts from the
  2026-07-06/07 tech-tree integration and P3/P4 fleshing passes — already
  executed, their mutations are reflected in the live beads state, kept as
  provenance of what changed and why.
- `wave1-campaign.js`, `wave1-fixups.js`, `wave2-campaign.js`,
  `fanout-sessions.json`, `fanout-prep-2026-07-12.md`: completed
  fanout/campaign orchestration artifacts.
- `2026-07-11-yla8-*.py`: one-off production-census/repair scripts for a
  since-closed investigation.
- Various dated `.md` audit/design/RnD notes (2026-07-04 through
  2026-07-13) whose findings were already digested elsewhere (beads,
  `codebase-grok-history-2026-07-06-to-16.md`, or superseded design docs).

Not moved: files still referenced by open beads
(`2026-07-10-webui-verifiability-audit.md`, `2026-07-15-wiring-closure-census.md`,
`archive-intelligence-design-2026-07-13.md`, `closed-loops-design-2026-07-13.md`,
`dsl-pattern-matching-design-2026-07-13.md`, `gpt-pro-demo-review-analysis.md`,
`merge-conductor-prompt.md`, `pattern-analysis-grounding-2026-07-13.md`,
`rigor-mechanisms-proposal-2026-07-13.md`), the active living grok doc and its
paired chatlogs/history file, `README.md`, and the protected
`dogfood/`, `dogfood-2/`, `testsuite_diet/` directories.

Two files were left in place despite having only *closed* references
(lower urgency, not re-verified this pass — safe to archive later):
`2026-07-09-affordance-usage-review.md`, `2026-07-10-agent-control-dogfood-ledger.md`.
