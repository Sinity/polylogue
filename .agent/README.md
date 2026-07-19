# .agent — Polylogue Repo Agent Surface

Orientation for agents. Always-loaded rules live in `CLAUDE.md` (= `AGENTS.md`);
repo conventions in [`CONVENTIONS.md`](CONVENTIONS.md).

- **Task substrate**: Beads. `bd prime` → `bd ready` → claim → work → PR →
  close with reasons. There is no separate devloop scaffold; the former
  conductor packet lived at `archive/devloop-2026-07/` (its README maps each
  retired piece to what subsumed it). Retired design notes from the old
  `includes/` tree lived at `archive/includes-2026-07/`. Both `archive/` and
  `reports/` are **gitignored, not git-tracked** (polylogue-ocby) — they are
  evidence/scratch history that stays on disk in a working checkout but is
  not shipped in the public repo or a fresh clone; do not resurrect them as
  scaffold.
- `CONVENTIONS.md` — bead content bar, graph lint, execution tactics, PR
  cadence (kept deliberately parallel to sinex's; divergences are marked
  intentional).
- `scripts/bd-graph-lint` — bead-graph invariant lint; run before shipping
  bead-state deltas.
- `scratch/`, `archive/`, `reports/`, `task-history/` — gitignored, present
  only in a working checkout (thinking space, retired scaffold/evidence,
  report artifacts, local task-history JSONL respectively); `demos/`,
  `handoffs/`, `tools/` — tracked shelves.
