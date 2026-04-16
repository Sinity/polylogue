---
created: "2026-04-09T21:18:00+02:00"
purpose: "Assess retroactive LICENSE history rewrite risk versus current remote/PR state"
status: "complete"
project: "polylogue"
---

# LICENSE History Rewrite Investigation

## Context

User noticed the AGPL header in `LICENSE` names `Simon Ohara` as copyright holder and wants
to know whether retroactively removing that name from git history would compromise the remote
repository state or existing PRs.

## Findings

- Current working tree is clean on `master`, tracking `origin/master`.
- `LICENSE` currently contains:
  - `Copyright (C) 2025 Simon Ohara`
- The `LICENSE` file appears in git history only starting from the initial repository commit:
  - `3e3432bb chore: initialize repository infrastructure`
- The string `Simon Ohara` appears only in the current `LICENSE` file in the live tree.
- GitHub currently reports:
  - `81` open PRs
  - `1` closed PR (`#2`)
- Every currently open PR number appears in a `master` commit subject as `(#NNN)`.
- For many PRs, the `master` commit subject exactly matches the PR title. For some older PRs, the
  PR number matches while the squash commit title was edited before landing on `master`.
- Representative patch-identity checks:
  - PR `#164` patch matches `master` commit `8a1e1a02` exactly by `git patch-id --stable`
  - PR `#152` does **not** match `master` commit `b9db6fbb` by patch-id, despite number alignment
- This means the repository behaves like an open-PR registry paired with squash commits on
  `master`, but the current PR branch heads are not guaranteed to be frozen exact content mirrors
  of their corresponding `master` squash commits.
- PR `updatedAt` timestamps cluster on 2026-04-07/08, consistent with prior broad repo/branch
  manipulation rather than ordinary independently maintained PR review flow.
- Later repo state change:
  - all open PRs were closed, yielding `82` closed PRs and `0` open PRs
  - the `82` `origin/feature/*` branches still exist
- Reachability checks after closure:
  - all `83` remote refs checked (`origin/master` + `82` feature branches) still descend from
    initial commit `3e3432bb`
  - all `83` checked refs still contain `LICENSE` with `Simon Ohara`
  - GitHub still exposes `82` `refs/pull/*/head` refs for the closed PRs

## Outcome

- Rewriting the initial commit to remove the bad copyright line would rewrite every commit SHA
  in the repository, because the offending blob is rooted at the initial commit.
- This would not destroy file contents if handled correctly, but it would invalidate SHA-based
  relationships across `master`, remote branches, and historical references.
- If the goal is to preserve GitHub PR objects as much as possible, the rewrite must update all
  remote refs together, not just `master`. GitHub PRs track branch names, so leaving feature
  branches untouched while rewriting only `master` would badly distort the open PR diffs.
- Even with a full-ref rewrite, any SHA-anchored review context becomes stale and some open PRs may
  display changed diffs because current branch heads are not all exact patch mirrors of their
  numbered `master` commits.
- For a *complete* purge from GitHub reachability, cleaning `master` and branch refs is still not
  sufficient: the closed PR refs (`refs/pull/*/head`) also reference the old commits. Those refs
  are read-only from normal pushes. GitHub's own guidance for sensitive-data cleanup says that pull
  request refs can keep the old data reachable and may require provider-side dereferencing or PR
  deletion to allow garbage collection.
- Lowest-risk path: fix `LICENSE` in-place now, do not rewrite history.
- Resolution taken: fix the live license and keep history intact.
- If retroactive purge is still desired, treat it as full-repo surgery with a repo backup and a
  mirror-style force push of all surviving refs.
