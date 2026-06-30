# Scratch Index

Scratch is not the active conductor. The active devloop packet lives at
`.agent/conductor-devloop/`. This README is tracked; scratch contents are
ignored local state/evidence.

Use this directory for supporting material that should not be startup state:
research, bulky/generated artifacts, and archived notes. The 2026-06-30 cleanup
moved the active loop out of `scratch/current` so a contextless agent has a
single resume target.

## Research

`research/` contains targeted analysis that may feed future product slices:

- `00-INDEX.md` — entrypoint for the June research wave.
- `01-*` through `12-*` — cost/pricing, token accounting, server-tool-use,
  cross-verification, forensics folding, Atropos export, and lineage validation.
- `2026-06-29-browser-capture-devloop-intel.md` — browser-capture/project-a
  context useful for live capture work.

## Artifacts

`artifacts/` is for generated or bulky evidence such as live baselines,
probe outputs, JSON/JSONL/CSV/log dumps, and scripts that are evidence rather
than reusable tooling.

## Archive

`archive/` preserves older notes. The 2026-06-30 cleanup moved older loose
markdown notes into `archive/2026-06-30-root-notes/` with
`MANIFEST-original-root-md.txt`.

## Rule

New active conductor/devloop notes go under `.agent/conductor-devloop/`, not
under scratch. New durable research goes under `research/`. Generated proof and
raw dumps go under `artifacts/`. Completed or stale notes go under `archive/`.
