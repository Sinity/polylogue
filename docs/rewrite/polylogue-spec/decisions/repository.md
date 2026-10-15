# Decision: fresh workspace, continuous repository history

Status: **revised delivery strategy, 2026-10-13.** Preserve verified pre-rewrite refs and recoverable WIP, then replace the source tree through a coherent PR into the existing repository's verified default branch. Subsequent vertical slices use ordinary PRs there, not a long-lived `rust` integration branch. The joint [realization plan](../../polylogue-sinex-joint/REALIZATION-PLAN.md) owns preservation and milestone criteria.

The operator’s 2026-10-14 sequencing correction requires existing worktrees and branches to be reconciled, intended changes completed and integrated on the default branch, and completed worktrees cleaned before M0 preparation. Preservation is refreshed from the resulting heads; the [joint realization plan](../../polylogue-sinex-joint/REALIZATION-PLAN.md) owns that gate.

Git history, issues and references remain continuous. The old implementation stays recoverable through preserved refs and a verified private bundle; running/parked packages remain pinned to it until separate cutover. A branch is not a data backup. Raw inputs, spools, purchased vectors and private evidence have independent conservation requirements (`plan/migration.md`).

The source-replacement PR proves an executable bootstrap, not a useful archive product. Existing std-only/todo skeletons prove composition; actual ownership types, build/check routes, neutral fixtures and explicit unsupported capabilities must be ready before replacement. Retire old callers, Python tooling and instructions together; no parallel product tree in the new workspace. Current source hooks/rules govern until usable successors replace them in the PR.

The earlier choice below preserves its reasoning. Its continuity goal survives; a separate long-lived integration branch is unnecessary once old deployments are pinned and source history is preserved.

## Earlier choice, 2026-10-11

Status: **decided: option D**, the same repository with the rewrite on a
long-lived branch. Decided by the spec authors from the operator's statement
that continuity of version control matters (2026-10-11); the operator may
overrule.

### Chosen

- **D. Same repository, continuous history, rewrite on a branch.** A
  long-lived `rust` branch from `main` holds the new workspace (`crates/`,
  `xtask/`) while `main` keeps the Python version running. Slices land on
  that branch as squash-merged PRs. At switchover the branch merges into
  `main` in one change that also deletes the Python tree, so history, issues,
  Beads references and blame stay continuous. Costs: CI must build two
  toolchains while both exist, and Python tooling in the tree must not leak
  into the Rust workspace (separate directories, the Rust CI ignores the
  Python paths).

### Rejected

- **A. New repository** (the Python repository archived): breaks history
  continuity.
- **B. Same repository, fresh history** on a new default branch, the old
  history on an archived branch: breaks history continuity.
- **C. Same repository, Python and Rust side by side** until the switch: two
  implementations in one tree is the parallel path the operator rules out.
