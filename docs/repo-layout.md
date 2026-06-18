# Repository Layout

Every top-level entry has a reason.

| Entry | Purpose | Managed by |
|-------|---------|------------|
| `.agent/` | Project-local agent scratch and task history | agents |
| `.cache/` | Disposable caches (hypothesis, pytest, mypy, ruff, site) | tools |
| `.claude/` | Claude Code project config (commands, agents, skills) | user |
| `.coderabbit.yaml` | CodeRabbit review bot configuration | repo |
| `.direnv/` | direnv layout (devshell activation) | direnv |
| `.dockerignore` | Container build exclusion rules | repo |
| `.envrc` | direnv entry point (`nix develop`) | repo |
| `.git/` | Git repository data | git |
| `.gitattributes` | Git attribute overrides | repo |
| `.githooks/` | Git hooks (format, lint, quick verify) | repo |
| `.github/` | GitHub Actions workflows and templates | repo |
| `.gitignore` | Git ignore rules | repo |
| `.local/` | Untracked local outputs (campaigns, showcases, build artifacts) | tools |
| `.release-please-manifest.json` | Release Please component version manifest | release tooling |
| `.venv/` | Python virtual environment (uv) | uv |
| `AGENTS.md` | Generated agent instructions from CLAUDE.md | devshell |
| `CHANGELOG.md` | Release changelog | repo |
| `CLAUDE.md` | Primary project instructions | repo |
| `CONTRIBUTING.md` | Contribution workflow and conventions | repo |
| `Containerfile` | Container image definition for runtime packaging | packaging |
| `LICENSE` | Project license | repo |
| `README.md` | Project readme | repo |
| `TESTING.md` | Test suite documentation | repo |
| `browser-extension/` | Chrome MV3 browser capture extension | repo |
| `contrib/` | Community-contributed integrations | repo |
| `devtools/` | Developer tooling (verify, render, campaigns) | repo |
| `docs/` | Architecture, internals, plans, design documents | repo |
| `flake.lock` | Nix flake lockfile (pinned dependencies) | nix |
| `flake.nix` | Nix flake definition (devshell, package, checks) | repo |
| `hatch_build.py` | Hatchling build hook (version injection) | build system |
| `nix/` | Nix packaging expressions | repo |
| `packaging/` | Distribution packaging helpers and metadata | repo |
| `pages.toml` | GitHub Pages site configuration | build system |
| `polylogue/` | Application source code | repo |
| `pyproject.toml` | Python project metadata and tool config | repo |
| `release-please-config.json` | Release Please configuration | release tooling |
| `systemd/` | Systemd service units for daemon | repo |
| `tests/` | Test suite | repo |
| `uv.lock` | uv dependency lockfile | uv |

## Ignored Local State

These entries can appear at the repository root during local work but are not
part of the tracked layout:

| Entry | Purpose / policy |
|-------|------------------|
| `.testmondata*` | pytest-testmon dependency database used by `devtools verify`; keep when it is useful for affected-test selection. |
| `.benchmarks/` | pytest-benchmark local output. |
| `.serena/` | local Serena project state. |
| `MagicMock/` | pytest pollution from a bad mock path; safe to delete when present. |
| `__pycache__/` | Python bytecode cache; safe to delete when present. |
