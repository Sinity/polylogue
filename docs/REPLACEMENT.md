# Foundation disposition

Predecessor base: `b4b61c7eab8daf740bfdccb146e530906839726d`. The Rust foundation is published as the repository source. Runtime activation, archive migration and consumer routing require separate qualification.

## Exact predecessor top-level paths

| Path | Disposition |
|---|---|
| `.agent` | Removed from the target workspace; predecessor implementation/configuration remains in the preserved Git history. No successor capability is advertised from this path. |
| `.agentctl` | Replaced with declared isolated Nix foundation operations. |
| `.circleci` | Replaced with the same quick-gate job name running actual Nix foundation checks. |
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

The Rust foundation was published through protected PR 6181 as `99972871d8ee48614a6206a10761e5b703c21f54`. Publication does not qualify runtime activation, archive migration, full canonical convergence or M1 behavior. Predecessor deployment consumers and retained data have separate operational custody. Known predecessor package consumers have explicit immutable source pins in [Sinnix e9bdd3a75](https://github.com/Sinity/sinnix/commit/e9bdd3a750227d02baf8df851c95945b6a25b282) and [Lynchpin 0383336](https://github.com/Sinity/sinity-lynchpin/commit/03833361456999be6c84536fb6e0936b62d27ccb). These declarations do not prove installed state; explicit input overrides remain outside their scope.

The shared file layer is published under the annotated `hashed-blobs-v0.1.0` tag targeting that same commit. The independent extracted-package and Git-consumer checks passed before the tag was pushed; Sinex now consumes the owner repository rather than a vendored copy. [The release procedure](shared-file-layer-release.md) records the release boundaries. Current source changes still require their own package/client verification; shared-library publication does not qualify parser, search or export behavior.

Retained Codex acquisition, parser-state restart and a limited selected-message terminal read have component and actual-client evidence. Complete canonical convergence, Claude/Morrow, JSON read/export, search/MCP, provenance and resource acceptance remain open. Physical data preservation, purchased vectors, provider inputs and task stores are separate operational responsibilities; synthetic verification neither imports nor deletes live provider data.
