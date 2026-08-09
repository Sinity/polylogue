# Repository Layout

The first table is the complete tracked root inventory for this repository.
The second lists common local-only roots created by Git, direnv, uv, tests, and
repository tools. For the Python package itself, see
[Code Navigation](code-navigation.md).

## Tracked repository roots

| Entry | Purpose | Managed by |
|-------|---------|------------|
| `.agent/` | Agent conventions, checked-in helper state, and repository-owned agent metadata; transient runs remain ignored | repo / agents |
| `.beads/` | Versioned Beads issue, dependency, and memory export used as the Git-side task authority | Beads / repo |
| `.beads-hooks/` | Canonical Beads-aware composite Git hooks; chains repository checks with `bd hooks run` | repo |
| `.circleci/` | CircleCI jobs, images, and pull-request verification entrypoints | CI |
| `.claude/` | Claude Code project settings, hooks, and generated agent definitions | repo / Claude Code |
| `.coderabbit.yaml` | CodeRabbit review policy | repo |
| `.codex/` | Codex project configuration and repository skills | repo / Codex |
| `.dockerignore` | Container build exclusions | repo |
| `.envrc` | direnv entrypoint for the Nix devshell | repo |
| `.gitattributes` | Git path and diff attributes | repo |
| `.githooks/` | Fallback repository hooks when the Beads composite hooks are unavailable | repo |
| `.github/` | GitHub Actions, issue/PR templates, Dependabot, and project-page assets | GitHub / repo |
| `.gitignore` | Git ignore policy | repo |
| `.release-please-manifest.json` | Release Please component-version manifest | release tooling |
| `.tokeignore` | Repository-owned source-attribution exclusions | repo |
| `AGENTS.md` | Symlink-compatible agent entrypoint generated from `CLAUDE.md` | repo |
| `CHANGELOG.md` | Release history maintained by Release Please | release tooling |
| `CLAUDE.md` | Primary standalone agent instructions and working rules | repo |
| `CONTRIBUTING.md` | Contribution, branch, schema, verification, and PR workflow | repo |
| `LICENSE` | Project license | repo |
| `README.md` | Public project entrypoint | repo |
| `SECURITY.md` | Supported versions, trust boundary, and reporting policy | repo |
| `TESTING.md` | Test-suite contracts, selection, and resource guidance | repo |
| `browser-extension/` | Chrome MV3 browser-capture extension | repo |
| `contrib/` | Standalone integration prototypes and operator-facing helpers | repo |
| `devtools/` | Repository control plane: verification, generators, audits, campaigns, and release checks | repo |
| `docs/` | Guides, references, architecture, evidence, plans, and historical records | repo |
| `flake.lock` | Pinned Nix dependency graph | Nix |
| `flake.nix` | Devshell, package, checks, and NixOS/Home Manager outputs | repo / Nix |
| `nix/` | Nix modules and packaging expressions | repo / Nix |
| `packaging/` | Distribution metadata and build helpers, including the container image | packaging |
| `polylogue/` | Application source; package roles and change paths are in [Code Navigation](code-navigation.md) | repo |
| `pyproject.toml` | Python package metadata, dependencies, entrypoints, and tool configuration | repo |
| `release-please-config.json` | Release Please policy | release tooling |
| `scripts/` | Narrow standalone commands useful outside the `devtools` registry | repo |
| `tests/` | Unit, integration, property, scale, benchmark, and fixture suites | repo |
| `uv.lock` | Reproducible Python dependency lock | uv |
| `webui/` | TypeScript/Preact web client and build configuration | repo |

## Expected local-only roots

These may exist in a working checkout but are not repository authority.

| Entry | Purpose | Owner |
|-------|---------|-------|
| `.cache/` | Disposable pytest, testmon, verify, render, benchmark, and type-check state | tools |
| `.direnv/` | direnv/Nix shell state | direnv |
| `.git/` | Git object database and worktree metadata | Git |
| `.local/` | Generated reports, demo outputs, proof material, and recovered local artifacts | repository tools |
| `.serena/` | Project-local semantic-code-tool state | Serena |
| `.venv/` | Primary uv-managed Python environment | uv |
| `.venv-freethreaded/` | Optional free-threaded Python validation environment | uv |
| `result` / `result-*` | Disposable Nix build-result symlinks | Nix |
| `__pycache__/` | Disposable Python bytecode | Python |

Local state is never evidence merely because it exists in a checkout. A
release, migration, or live archive operation must bind the selected Git SHA,
Beads digest, built package, and receipt rather than relying on an ambient
working directory.
