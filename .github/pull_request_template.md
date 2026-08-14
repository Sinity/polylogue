## Summary

_What changed and why it matters (1-3 sentences)._

## Problem

_What the codebase needed. Why this work was necessary._

Use neutral issue references such as `Ref #NNN`. Do not use GitHub
issue-state keywords in PR text unless the operator explicitly requested that
this PR change that exact issue state.

## Solution

_What was done. Key modules, contracts, and boundaries touched._

## Verification

_Exact commands run and any manual validation performed._

## Bead disposition matrix

_Required for Bead-scoped PRs. Delete this section when the carrier uses `scope_kind=self_contained`._

| Assigned Bead | Whole-Bead disposition | Evidence refs | Named successor for residual work |
| --- | --- | --- | --- |
| `polylogue-...` | satisfied / partial / deferred / superseded | `test:...`, `command:...` | `polylogue-...` or n/a |

<!-- polylogue-pr-scope:v2
Replace this comment with the output of:
devtools workspace pr-scope render --input .agent/pr-scope.json > /tmp/pr-scope.md

The stable intent input declares scope_kind, assigned_beads, mutated_beads, and
one disposition with typed evidence for each assigned Bead. Use
scope_kind=self_contained with empty Bead lists for a self-contained PR.
mutated_beads declares every Bead record changed by this PR. Partial, deferred,
and superseded dispositions require an existing open successor Bead. The body
does not contain a head SHA or Bead digest. After each push, inspect the live
attestation with:
devtools workspace pr-scope sync --pr <PR-number>
-->

## Changelog

_If user-visible (new flags, renamed/removed commands, output changes, breaking migrations, security fixes), add a one-line entry to the `Unreleased` section of `CHANGELOG.md`. Skip for refactors, internal renames, and test-only PRs._

## Risks and Follow-ups

_Remaining risks, migrations, rollout concerns, or deferred work. Delete this section if none._
