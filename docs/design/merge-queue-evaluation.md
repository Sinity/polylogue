# GitHub Merge Queue Evaluation

Status: deferred, evaluated 2026-08-26. Owner bead: `polylogue-f6t51`.

## Decision

Do not enable GitHub merge queue for Polylogue yet. Keep `harvest_queue2` as the publish path until the repository has a usable required-check policy and enough successful CI observations to measure queue behavior. The phase-B rebase, local quick gate, push, pull request, and auto-merge flow therefore remains in place. No harvest script or repository flock is removed by this decision.

## Evidence

The checked-in workflow runs on `pull_request`, `push` to `master`, and manual dispatch. It does not run on `merge_group`. GitHub documents that a required GitHub Actions check must also be triggered by `merge_group`; otherwise a queued pull request cannot report the required check and the merge fails. See [Managing a merge queue](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue#triggering-merge-group-checks-with-github-actions).

As observed through the GitHub API on 2026-08-26:

| Fact | Observation | Consequence |
| --- | --- | --- |
| Merge queue | `repository.mergeQueue` is `null` | There is no queue to exercise or measure. |
| Required checks | `GET .../branches/master/protection/required_status_checks` returns HTTP 404, `Required status checks not enabled` | GitHub cannot currently be the authoritative check sequencer. |
| Rulesets | `GET .../rulesets` returns an empty list | No ruleset currently establishes queue or check requirements. |
| Merge policy | Auto-merge and squash are enabled; merge commits and rebase merges are disabled; head branches are deleted after merge | The current repository settings already match the intended squash publication shape. |
| CI observations | The two available CI runs, 31647014564 and 29610303427, failed during job setup in 3-5 seconds; failed logs were unavailable | These are not valid end-to-end latency samples. Queue latency remains unknown. |

The recent merged-PR sample also shows a high-throughput sequence: PRs 4270-4280 merged between 16:29:13 and 18:06:22 UTC on 2026-08-26. That measures merge activity, not CI completion or queue wait, so it is supporting context rather than a queue performance result.

## Release-please and squash titles

Release-please reads conventional commit subjects from the squash-merged history. Its configured release PR title is `chore(release): ${version}`. A future queue rollout must retain squash merging and must verify that the queue does not rewrite that title before it becomes the master commit subject. There is no `strip-title-suffix` setting in `release-please-config.json`, and no repository ruleset currently defines one. If a future queue policy introduces such a rule, it must preserve the conventional prefix and release version or be rejected as incompatible.

## Adoption gates

Re-evaluate after hermetic lane gates are green and the following changes can be made together:

1. Add `merge_group` to every workflow that reports a required check, then prove that the checks are emitted for the merge-group SHA.
2. Establish a master ruleset or branch-protection policy requiring the selected CI checks and the pull request, with squash as the only merge method.
3. Run a bounded pilot and record per-entry queue wait, CI completion latency, queue position, and the result of a gate-red ejection followed by a retry. Do not infer these values from PR merge cadence.
4. Exercise a release-please PR in the pilot and verify the exact squash subject and resulting conventional-commit parsing.
5. Only after the pilot is green, change `harvest_queue2` to push, open the PR, and add it to the queue. Delete phase B and the repository flock in that same change, with tests or a direct script-level proof that no local merge-train authority remains.

Until these gates pass, a gate-red result is handled by the existing harvest path and lane retry process. A queued rollout must instead treat gate-red as queue ejection, preserve the failed check and SHA for diagnosis, and require an explicit retry after the cause is fixed.
