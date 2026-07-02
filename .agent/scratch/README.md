# Scratch Index

Scratch is not the active conductor. The active devloop packet lives at
`.agent/conductor-devloop/`. This README is tracked; scratch contents are
ignored supporting research.

Use this directory for supporting research that should not be startup state.
Generated runtime/process evidence belongs under `.agent/task-history/`, demos
belong under `.agent/demos/`, compact archaeology belongs under
`.agent/archive/`, and current loop state belongs under
`.agent/conductor-devloop/`. The 2026-06-30 cleanup moved the active loop out
of `scratch/current` so a contextless agent has a single resume target.

## Research

`research/` contains targeted analysis that may feed future product slices:

- `00-INDEX.md` — entrypoint for the June research wave.
- `01-*` through `12-*` — cost/pricing, token accounting, server-tool-use,
  cross-verification, forensics folding, Atropos export, and lineage validation.
- `2026-06-29-browser-capture-devloop-intel.md` — browser-capture/project-a
  context useful for live capture work.

## Rule

New active conductor/devloop notes go under `.agent/conductor-devloop/`, not
under scratch. New durable research goes under `research/`. Generated proof,
raw dumps, runtime baselines, and execution ledgers do not go under scratch.
