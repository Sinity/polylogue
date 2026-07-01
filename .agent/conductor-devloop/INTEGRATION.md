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
- Prefer fewer, fatter PRs when the work is one coherent product or substrate
  phase. A good default target is roughly 50 workbench commits per PR-shaped
  group when dependency boundaries allow it. Use branch-local commits as review
  waypoints; do not burn PR overhead on tiny adjacent slices that will be
  understood, tested, and merged together.
- Push and PR creation are normal authorized devloop work. Do not treat either
  as a blocking approval step. Open PRs ready-for-review by default; use draft
  only when a concrete reason is recorded in the replay ledger or PR body.

## Current Semantic Ledger

Captured 2026-07-02 against:

- workbench branch: `feature/dogfood/parallel-parse`
- workbench head: `30cccc675`
- base: `origin/master` at `f41e07890`
- raw graph: `6` commits only on master, `321` commits only on the workbench

The raw `321` is not the remaining integration scope. Master already contains
known workbench semantics through these merged PRs:

| PR | Master commit | Workbench commits semantically covered | Status |
| --- | --- | ---: | --- |
| #2494 browser post commands | `79a8ac097` | 2 | landed from workbench |
| #2495 query with units | `4a9c7a07a` | 2 | landed from workbench |
| #2496 structured evidence stack | `e55c675aa` | 17 | landed from workbench |
| #2499 context image collapse | `0f64a0759` | 4 | landed from workbench |
| #2500 pages link repair | `a9985f411` | 0 | master-only side fix |
| #2501 squash worktree detection | `f41e07890` | 0 | master-only integration fix |

Known semantic subtraction from the workbench: `25` commits. First-pass
semantic remaining count: `321 - 25 = 296` branch commits to classify. Treat
that as a conservative planning number, not a promise that exactly 296 PR-sized
units remain.

Current next grouping bias:

- Aim for about 50 workbench commits per PR-shaped group when the topic is
  coherent. Smaller PRs require a concrete reason: high blast radius, urgent
  isolated fix, incompatible dependency ordering, or a review boundary that
  actually helps.
- Do not split adjacent schema/query/read-model fixes when they share the same
  archive rebuild story and verification lane.
- Keep `.agent` scaffold/process changes together unless a product PR genuinely
  depends on one of them.
- Keep demos/current demo shelf changes with the product capability they prove
  when that makes the PR easier to review; split them only when the demo is an
  independent artifact refresh.
- Keep test harness IO/resource work together as one performance/verification
  infrastructure PR, not scattered repair commits.
- Do not wait on CI as a planning activity. Run focused local proof, publish the
  ready PR, inspect substantive automated feedback when it appears, and keep
  classifying/replaying the next group instead of idling.

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
