# What replaced the Python tree

The Rust rewrite replaced the Python Polylogue source tree at
`b4b61c7eab8daf740bfdccb146e530906839726d`. It was published in PR 6181 as
`99972871d8ee48614a6206a10761e5b703c21f54`. The Python code is still in Git
history; nothing below is kept as a compatibility path.

## Replaced

| Old path | Now |
|---|---|
| `.agentctl` | Nix-based AgentCTL operations for this workspace. |
| `.envrc`, `flake.nix`, `flake.lock` | The Rust package, checks and development shell. |
| `.githooks` | Whitespace and Rust check hooks. |
| `.gitignore` | Rust build exclusions. |
| `AGENTS.md`, `README.md` | Rewritten for the Rust workspace. |
| `docs` | `docs/development.md`; the design is in the architecture document named in `AGENTS.md`. |
| `CHANGELOG.md` | Kept as [docs/history/PYTHON-CHANGELOG.md](history/PYTHON-CHANGELOG.md). |
| `LICENSE` | Unchanged. |

## Removed

`.agent`, `.circleci`, `.claude`, `.codex`, `.config`, `.dockerignore`,
`.gitattributes`, `.github`, `.release-please-manifest.json`, `.tokeignore`,
`CONTRIBUTING.md`, `SECURITY.md`, `TESTING.md`,
`contrib`, `devtools`, `experiments`, `nix`, `packaging`, `polylogue`,
`pyproject.toml`, `release-please-config.json`, `scripts`, `tests`, `uv.lock`
and `webui`. The browser extension now lives in `extension/`, and the web
reader in `crates/polylogue`. There is no hosted CI;
checks run locally ([development.md](development.md)).

## Consumers

The Python package is still deployed. Its known consumers pin it by source
revision: [Sinnix e9bdd3a75](https://github.com/Sinity/sinnix/commit/e9bdd3a750227d02baf8df851c95945b6a25b282)
and [Lynchpin 0383336](https://github.com/Sinity/sinity-lynchpin/commit/03833361456999be6c84536fb6e0936b62d27ccb).

The content-addressed file store and credential masking now live in
[polylogue-sinex-common](https://github.com/Sinity/polylogue-sinex-common), which both Polylogue and Sinex
pin by tag.

Switching the deployment to the Rust version, migrating the existing archive
and its data, and removing the Python package are separate operator decisions.
