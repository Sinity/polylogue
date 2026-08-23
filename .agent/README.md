# .agent — Polylogue Repo Agent Surface

Orientation for agents. Always-loaded rules live in `CLAUDE.md` (= `AGENTS.md`);
repo conventions in [`CONVENTIONS.md`](CONVENTIONS.md).

- **Task authority**: external to this checkout. Do not carry or publish task
  state through branches, worktrees, commits, or PRs. Historical interaction
  ledgers are archive evidence, not task control.
- `CONVENTIONS.md` — execution tactics and Git boundary guidance.
- `scratch/`, `archive/`, `reports/`, `task-history/` — gitignored, present
  only in a working checkout (thinking space, retired scaffold/evidence,
  report artifacts, local task-history JSONL respectively); `demos/`,
  `handoffs/` — tracked shelves.
