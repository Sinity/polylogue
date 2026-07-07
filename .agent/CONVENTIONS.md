# Polylogue Agent Conventions

Repo-local conventions for agents working in this checkout. Always-loaded
operating rules live in `CLAUDE.md` (= `AGENTS.md`); this file holds the
repo-agent conventions that do not need to be in every context window.
Kept deliberately parallel to `sinex/.agent/CONVENTIONS.md` — divergences are
intentional and marked.

**The devloop substrate is Beads.** The former bespoke conductor packet
(`conductor-devloop/`, `DEVLOOP.md`, `devloop-*` scripts) is archived at
`.agent/archive/devloop-2026-07/` — see its README for what subsumed each
piece. Do not resurrect packet files or `devloop-*` script names; the loop is:
`bd prime` → `bd ready` → claim → work → PR → close with reasons.

## Directory Shape

```text
.agent/
  README.md          # orientation
  CONVENTIONS.md     # this file
  scripts/           # small repo-agent helpers (non-devloop)
  demos/             # curated demo shelf
  reports/           # tracked report artifacts
  task-history/      # historical task notes
  scratch/           # gitignored thinking space
  tools/             # helper tooling
  archive/           # retired scaffolds kept as evidence
```

## Beads Task Substrate

Beads (`bd`, workspace at `.beads/`, prefix `polylogue`) is the durable task
substrate: ready work, claims, blockers, dependencies, deferred work, and
discovered follow-ups. Run `bd prime` for workflow context.

- **Beads** owns work items; anything that should survive the current slice
  becomes a bead, not a bullet in a markdown file.
- **Current focus** is the claimed bead; its notes field is the running trail;
  `bd close <id> --reason "…"` with verification commands ends the slice.
  There is no active-loop file.
- **Operator directives** that must survive compaction are beads, never a
  queue file.
- **`bd remember`** holds durable cross-session insights; search with
  `bd memories <keyword>` before re-deriving anything expensive.
- Discovered work is linked: `bd create ... --deps discovered-from:<bead>`.
- `blocks` only for true ordering, `related` for affinity; keep
  `bd dep cycles` clean.
- No Dolt remote: `.beads/issues.jsonl` in git IS the sync surface; ship
  bead-state deltas in PRs (`chore(beads):`).
- Graph lint: run `.agent/scripts/bd-graph-lint` before shipping bead deltas
  (cycles, missing acceptance criteria, duplicate `wave:`/`area:` labels,
  wave inversions). INTENTIONAL DIVERGENCE from sinex: polylogue does not
  (yet) enforce exactly-one-wave/exactly-one-area — its label taxonomy
  (`lane:`, `delivery:`, `horizon:`) evolved separately; unify deliberately
  or not at all.

Execution-grade bar for ready beads at priority ≤ 2 (same as sinex):
description states the problem + current verified state with dated
`file:line` cites; design carries the settled decision, exact targets, and
interacting beads; acceptance names repo-native VERIFY commands — here that
means `devtools test` (managed pytest-testmon path) or a targeted
`pytest tests/... -k`, never "tests pass". Reconcile-on-claim: re-verify a
bead's cited facts against master before coding.

## Execution Tactics

- **Async where the harness allows**: long test runs and imports go to
  background execution; do light work (reads, bead edits) while they run.
- **Serial heavy, parallel light.** Serialize anything sharing the pytest
  temp DBs (`/realm/tmp/polylogue-pytest`) or the archive DB; parallel tool
  calls are for reads/searches only.
- **Proof ladder.** Narrowest proof while iterating (`devtools test` with
  testmon affected-selection or a targeted `-k` filter); the broad suite once
  per publishable phase. Never blanket `pytest tests/unit` as an iteration
  step (see CLAUDE.md).

## Greedy Batch / PR Cadence

Identical policy to sinex (canonical text in
`sinex/.agent/CONVENTIONS.md` — Greedy Batch section; do not fork the
wording): one complete bead per branch/PR by default; widen to the largest
coherent acceptance-criteria phase before splitting; a green substep is a
checkpoint, not a publishing trigger; splitting requires a real boundary
(risk, reviewability, dependency, ownership, deployment, failure isolation);
before publishing, record the satisfied/deferred/follow-up AC matrix in the
PR body and bead notes.

## Scratch, Demos, Git Boundary

`.agent/scratch/` is gitignored thinking space (README + research notes);
promote durable insight out: rules → CLAUDE.md/CONVENTIONS.md, work items →
beads, gotchas → `bd remember`. `.agent/demos/` is a curated shelf, not a
dump. The tracked `.agent` surface stays small (this file, README, scripts,
demos, reports, task-history, tools, archive); everything else is ignored
live state. New tracked files get a deliberate `.gitignore` allowlist entry.
