---
created: "2026-04-10T00:00:00+02:00"
purpose: "Track the remaining GitHub-admin follow-ups after in-repo governance and CI hardening"
status: "complete"
project: "polylogue"
---

# GitHub Governance and CI Hardening

## Context

Audit the repository's GitHub-facing surface after the license correction:

- GitHub Actions reliability
- issue and PR templates
- branch and PR policy
- branch protection and adjacent repo settings

## Findings

- `master` had no branch protection configured.
- There were no repository rulesets configured.
- Recent Actions history showed systematic failures in both `CI` and `Nix`, while `CodeQL` succeeded.
- `CI` failure root cause:
  - `astral-sh/setup-uv` already provisions and activates a virtual environment
  - workflow steps then ran `uv venv` again
  - this failed with `A virtual environment already exists`
- `Nix` failure root causes:
  - workflow attempted `nix build .#polylogue` while the flake exported only `packages.default`
  - `checks.default` used `uv` inside a sandboxed derivation, which failed trying to discover
    managed Python interpreters via host paths such as `/bin/sh`
- Workflow pins were stale relative to current upstream majors:
  - `actions/checkout` latest: `v6.0.2`
  - `astral-sh/setup-uv` latest: `v8.0.0`
  - `cachix/install-nix-action` latest: `v31.10.4`
  - `actions/upload-artifact` latest: `v7.0.0`
  - `github/codeql-action` tag line currently on `v4`
- Issue templates were assigning nonexistent GitHub labels before the earlier cleanup.
- PR template was still structured around history-rewrite arcs rather than ordinary feature-branch PRs.
- Repo settings at audit time:
  - `allow_squash_merge = true`
  - `allow_merge_commit = false`
  - `allow_rebase_merge = false`
  - `delete_branch_on_merge = false`
  - `allow_update_branch = false`
  - `allow_auto_merge = false`
  - `default_branch = master`

## In-Repo Resolution

The repository-side work from this audit has already landed:

- workflow fixes and pin updates
- `flake.nix` export/check cleanup
- public contribution and PR policy docs
- issue/PR template cleanup
- branch/PR policy enforcement from the repo side

## Remaining GitHub-Admin Follow-Ups

- Enable branch protection or a ruleset for `master` after the workflow changes are pushed.
- Recommended branch protection posture:
  - require pull requests before merge
  - require status checks to pass
  - include administrators
  - restrict direct pushes
  - optionally require one approval
- Consider enabling:
  - `delete_branch_on_merge = true`
  - `allow_update_branch = true`
  - `allow_auto_merge = true` once required checks are stable
