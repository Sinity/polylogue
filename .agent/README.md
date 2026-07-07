# .agent — Polylogue Repo Agent Surface

Orientation for agents. Always-loaded rules live in `CLAUDE.md` (= `AGENTS.md`);
repo conventions in [`CONVENTIONS.md`](CONVENTIONS.md).

- **Task substrate**: Beads. `bd prime` → `bd ready` → claim → work → PR →
  close with reasons. There is no separate devloop scaffold; the former
  conductor packet is archived at `archive/devloop-2026-07/` (its README maps
  each retired piece to what subsumed it). Retired design notes from the old
  `includes/` tree live at `archive/includes-2026-07/`.
- `CONVENTIONS.md` — bead content bar, graph lint, execution tactics, PR
  cadence (kept deliberately parallel to sinex's; divergences are marked
  intentional).
- `scripts/bd-graph-lint` — bead-graph invariant lint; run before shipping
  bead-state deltas.
- `scratch/` — gitignored thinking space; `demos/`, `reports/`,
  `task-history/`, `tools/` — tracked shelves; `archive/` — retired scaffolds
  kept as evidence, never resurrected.
