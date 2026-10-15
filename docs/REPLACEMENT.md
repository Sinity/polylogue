# Foundation replacement disposition

Predecessor base: `b4b61c7eab8daf740bfdccb146e530906839726d`. This replacement exists only in the isolated branch. Primary master, hosted publication, installed consumers, task routing, services and archives remain unchanged.

## Exact predecessor top-level paths

| Path | Disposition |
|---|---|
| `.agent` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `.agentctl` | Replaced with declared isolated Nix foundation operations. |
| `.circleci` | Replaced with the same quick-gate job name running actual Nix foundation checks; hosted execution remains unrun. |
| `.claude` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `.codex` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `.config` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `.dockerignore` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `.envrc` | Replaced with the target Nix development shell. |
| `.gitattributes` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `.githooks` | Replaced with target staged-whitespace and Rust check hooks. |
| `.github` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `.gitignore` | Replaced with Rust build/development exclusions. |
| `.release-please-manifest.json` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `.tokeignore` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `AGENTS.md` | Replaced with the concrete target owner and supported-scope instructions. |
| `CHANGELOG.md` | Retained under docs/history/PREDECESSOR-CHANGELOG.md as historical release evidence. |
| `CONTRIBUTING.md` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `LICENSE` | Retained unchanged, including predecessor copyright. |
| `README.md` | Replaced with actual foundation commands and limits. |
| `SECURITY.md` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `TESTING.md` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `browser-extension` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `contrib` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `devtools` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `docs` | Predecessor docs retired with their callers; selected current target contracts copied under docs/rewrite, history preserved separately. |
| `experiments` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `flake.lock` | Replaced with the reviewed target Nixpkgs lock. |
| `flake.nix` | Replaced with standalone Rust package, test and development outputs. |
| `nix` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `packaging` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `polylogue` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `pyproject.toml` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `release-please-config.json` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `scripts` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `tests` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `uv.lock` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `webui` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |

The replacement is the target `Cargo.toml`/lock, `crates/`, `xtask/`, Nix package and checks, successor hooks/CI/instructions and synthetic tests. Old Python package, extension, web UI, tooling and release configuration are not compatibility paths beside it. LICENSE remains unchanged; history and restore-tested recovery material retain removed implementation knowledge.

## Operational boundary

This is a foundation source replacement, not a usable M1 product. Known predecessor package consumers have explicit immutable source pins published in [Sinnix e9bdd3a75](https://github.com/Sinity/sinnix/commit/e9bdd3a750227d02baf8df851c95945b6a25b282) and [Lynchpin 0383336](https://github.com/Sinity/sinity-lynchpin/commit/03833361456999be6c84536fb6e0936b62d27ccb). Their locked graphs remain unchanged. These source declarations do not prove installed state, and explicit input overrides remain outside their scope. The current branch does not change those consumers. The successor CircleCI job definition keeps the existing quick-gate name and invokes actual target checks, but it has not run on the forge. Required protections and hosted review must be rechecked at publication.

The shared file layer is the canonical Polylogue owner candidate plus a reviewed read-only capability; source/adaptation digests are recorded in shared-crate-source.json. Package metadata and a disposable local-tag consumer prepare `hashed-blobs-v0.1.0`; [the release procedure](shared-file-layer-release.md) names the exact publication and vendor-replacement gates. The remote owner tag and real downstream pin remain open. No remote tag has been invented.

M1 acquisition, interpretation, retained revision/inline protocols, full query/read/export/MCP and index recovery remain later work. Physical data preservation, purchased vectors, provider inputs and task stores are separate operational responsibilities; this source branch neither imports nor deletes them.
