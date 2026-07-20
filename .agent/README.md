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
- Repo-agent helper scripts live under `devtools/` as proper commands, not in
  `.agent/` — `devtools lab policy bead-graph` (bead-graph invariant lint; run
  before shipping bead-state deltas), `devtools workspace
  bead-reimport-guard`, `devtools workspace delivery-gate-status`, `devtools
  workspace bead-batch-show` (polylogue-kapb: `.agent/scripts/` and
  `.agent/tools/` no longer carry tracked tooling).
- `scratch/`, `archive/`, `reports/`, `task-history/` — gitignored, present
  only in a working checkout (thinking space, retired scaffold/evidence,
  report artifacts, local task-history JSONL respectively); `demos/`,
  `handoffs/` — tracked shelves.
