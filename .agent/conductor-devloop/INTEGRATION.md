# Integration Protocol

Polylogue uses a long-lived devloop workbench branch for fast dogfood work, but
master integration must stay PR-shaped and reviewable. This file is the durable
protocol for turning the workbench into coherent pull requests while object work
continues.

## Current Model

- The workbench branch may contain many small, atomic, proven commits.
- Integration branches start from current `origin/master`.
- Related workbench commits are clustered by product/change intent, not by
  chronology alone.
- Each cluster becomes a PR-shaped replay branch with a durable title, body,
  verification record, and explicit dependency notes.
- Push and PR creation are normal authorized devloop work. Do not treat either
  as a blocking approval step. Open PRs ready-for-review by default; use draft
  only when a concrete reason is recorded in the replay ledger or PR body.

## Required Command

Use the integration report before publication decisions:

```bash
.agent/scripts/devloop-integration
```

Use the subagent prompt when the branch has accumulated enough unrelated work
that clustering should be read-heavy and parallel:

```bash
.agent/scripts/devloop-integration --subagent-prompt
```

The report must show both sides of the lane:

- workbench state: branch, head, merge-base, commits ahead of `origin/master`;
- replay state: local PR-shaped worktrees, branches, heads, dirty-path counts,
  and the replay-plan artifact when present.

## Replay Discipline

1. Fetch current `origin/master` before replaying a group.
2. Create a branch named for the intended PR, not the original workbench slice.
3. Cherry-pick or replay the clustered commits in dependency order.
4. Squash only inside the coherent group when that produces a better PR history.
5. Repair generated surfaces in the replay worktree, not on the main workbench.
6. Run the narrow proof for the group plus generated-surface checks when the
   group touches docs/topology/agent surfaces.
7. Push the branch and open a PR with Summary, Problem, Solution, and
   Verification.
8. Record the remote branch, PR URL, head commit, verification, and residual
   dependency in the replay plan or operating log.

## PR Shape

Each PR body should stand alone without chat context:

- Summary: what changed and user/operator impact.
- Problem: what evidence or constraint made this necessary.
- Solution: modules/contracts touched and why the grouping is coherent.
- Verification: exact commands and meaningful result lines.
- Residual scope: dependencies, deferred work, or caveats when applicable.

Do not claim convergence, unification, or removal unless the replay diff proves
the old path is gone or the new shared path is actually used by every claimed
surface.

## Subagent Boundary

Subagents are appropriate for:

- clustering ahead commits;
- identifying dependencies between groups;
- drafting PR bodies and verification plans;
- flagging overbroad claims;
- generating dry-run replay scripts.

The main devloop normally owns:

- branch creation;
- conflict resolution;
- final verification;
- pushing;
- PR creation;
- updating the replay ledger.

The operator may explicitly delegate any of those, but absent that instruction
the main devloop keeps the write-authority lane.

## First-Wave Ledger

The current first-wave replay ledger lives in ignored local state:

```text
.agent/task-history/integration-first-wave-replay-plan.md
```

That file is intentionally not part of the cold-start protocol. It records the
current replay branches, PR URLs, and remaining candidate groups; regenerate or
replace it as the workbench changes.
