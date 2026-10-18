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
| `docs` | Notes under `docs/rewrite/`. |
| `CHANGELOG.md` | Kept as [docs/history/PYTHON-CHANGELOG.md](history/PYTHON-CHANGELOG.md). |
| `LICENSE` | Unchanged. |

## Removed

`.agent`, `.circleci`, `.claude`, `.codex`, `.config`, `.dockerignore`,
`.gitattributes`, `.github`, `.release-please-manifest.json`, `.tokeignore`,
`CONTRIBUTING.md`, `SECURITY.md`, `TESTING.md`, `browser-extension`,
`contrib`, `devtools`, `experiments`, `nix`, `packaging`, `polylogue`,
`pyproject.toml`, `release-please-config.json`, `scripts`, `tests`, `uv.lock`
and `webui`. None of them has a Rust counterpart yet. There is no hosted CI;
checks run locally ([development.md](development.md)).

## Consumers

The Python package is still deployed. Its known consumers pin it by source
revision: [Sinnix e9bdd3a75](https://github.com/Sinity/sinnix/commit/e9bdd3a750227d02baf8df851c95945b6a25b282)
and [Lynchpin 0383336](https://github.com/Sinity/sinity-lynchpin/commit/03833361456999be6c84536fb6e0936b62d27ccb).

The shared file layer is released as the annotated tag `hashed-blobs-v0.1.0`
on the same commit, and Sinex depends on it from this repository
([release process](shared-file-layer-release.md)).

Switching the deployment to the Rust version, migrating the existing archive
and its data, and removing the Python package are separate operator decisions.
