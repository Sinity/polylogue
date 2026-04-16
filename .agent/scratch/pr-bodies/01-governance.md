## Summary

First of 9 PRs carving the 245-commit work from the former `feature/chore/repo-cleanup-governance` branch into reviewable units. This PR is the repo-governance base layer; subsequent PRs stack on top.

Highlights:

- Promote `devtools` to a first-class repo control plane with consistent CI integration (`uv run devtools …`).
- Consolidate governance surfaces: PR template, CLAUDE.md/AGENTS.md rendering, docs map, devshell status, local-state hygiene.
- Refresh the generated quality-workflow reference from live registries.
- Remove dead config harness, stale xfail drift, and personal session workflow docs.
- Normalize ruff style and import ordering across the repo.
- Pin `astral-sh/setup-uv` to `v8.0.0` in CI since the floating `v8` tag alias was removed upstream.

## Problem

The repo accumulated governance drift over the preceding branch wave:

- `devtools` was reachable only via `python -m devtools …` in CI and the PR template, making it feel like an internal module rather than a first-class control plane.
- `README.md` duplicated much of `docs/README.md`, fragmenting the docs map.
- The PR template still advised `python -m devtools render-all`, which was out of sync with how developers actually run the tool.
- `.envrc` emitted the compact MOTD only on the first direnv load, so interactive reloads sometimes showed nothing.
- The generated quality-workflow reference had drifted from the live validation/mutation/benchmark registries.
- CLAUDE.md/AGENTS.md carried personal-session workflow text that wasn't reusable for new contributors.
- Ruff formatting and import ordering were inconsistent across the repo, which was causing noisy diffs.
- Dead test scaffolding (config harness, prompt env hook, bogus semantic property xfails) remained staged.
- **CI was already broken upstream**: `astral-sh/setup-uv@v8` no longer resolved because GitHub removed the floating `v8` alias. Every CI job was failing at action resolution.

## Solution

- Expose `devtools` as a first-class package entry point so CI and the PR template can call `uv run devtools …`.
- Shorten the generated README docs section, restore `docs/README.md` as the full docs map, and tighten release/version wording.
- Add `devtools render-quality-reference` (live-registry-driven) and wire it into `devtools render-all --check`.
- Regenerate CLAUDE.md and AGENTS.md from a shared tight template; remove personal session workflow drift.
- Apply ruff format and import normalization repo-wide.
- Remove the dead config harness, prompt env hook, and stale xfail drift.
- Tighten devshell status, stderr routing, local-state hygiene (stale `result` symlinks, root `__pycache__/`), and restore the interactive direnv MOTD.
- Pin `astral-sh/setup-uv@v8.0.0` in `.github/workflows/ci.yml`.

## Verification

- `pytest -q --ignore=tests/integration`
- `devtools render-all --check`
- `ruff check polylogue tests devtools`
- `nix flake check`

Commits on this branch: 19 (18 governance/docs/control-plane + 1 CI pin).

## Stack

This is PR 1/9 in a stacked series. Each subsequent PR's base is the previous branch:

1. `feature/chore/stack-01-governance` → `master`
2. `feature/fix/stack-02-cli-contracts` → `feature/chore/stack-01-governance`
3. `feature/fix/stack-03-runtime-repair` → `feature/fix/stack-02-cli-contracts`
4. `feature/perf/stack-04-parse-hardening` → `feature/fix/stack-03-runtime-repair`
5. `feature/refactor/stack-05-artifact-graph` → `feature/perf/stack-04-parse-hardening`
6. `feature/refactor/stack-06-scenario-substrate` → `feature/refactor/stack-05-artifact-graph`
7. `feature/refactor/stack-07-corpus-execution-root` → `feature/refactor/stack-06-scenario-substrate`
8. `feature/refactor/stack-08-final-unification` → `feature/refactor/stack-07-corpus-execution-root`
9. `feature/fix/stack-09-runtime-fixes` → `feature/refactor/stack-08-final-unification`

Merge order: squash-merge #1, rebase #2 onto new master (drop merged commits), change base to master, merge; repeat through #9.
