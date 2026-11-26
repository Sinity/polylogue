# Development

## Set up

Enter with `nix develop`, or run `direnv allow` to load the same shell. Keep local settings in the ignored `.envrc.local`. Cargo uses `target/` unless `CARGO_TARGET_DIR` is set or the shell selects a per-checkout target through `sinnix-cargo-target`. The shell supplies the check tools, SQLite, the time zone database, and coverage and fuzzing tools. `nix develop .#format` supplies just the pinned toolchain, Git and diff. Both shells select `.githooks` if Git has no hooks path. Rustup, the shells and the package take Rust and its components from `rust-toolchain.toml`. Change the compiler in polylogue-sinex-common and both products together.

With `SCCACHE_DIR` and `XDG_RUNTIME_DIR` set and no custom Rust compiler wrapper, the development shell uses sccache. The host target helper can set the cache location beside the checkout targets. Set `SCCACHE_CACHE_SIZE` to override the 4 GB default. Checkouts share cached compilations and have separate server sockets. Look for hits, misses and uncacheable compilations in the xtask summary.

## While writing

Type-check a package and its tests with `cargo check --locked -p PACKAGE --tests` (`compile` on a job runner). Run the tests you write or change with `cargo xtask test -p PACKAGE --lib -E 'test(NAME)'` (`focused`). Use `--test NAME` for an integration target or `--bin polylogue` for the application's unit tests. The default targets are those marked for testing, including opted-in examples. A scoped build compiles the selected targets and their dependencies with the same features as the workspace test build. Declare shared dependency features in `[workspace.dependencies]` and run `cargo hakari generate` after changing dependencies or features.

Use `-p` and `--exclude` with package names or quoted globs. This package selection intersects the packages selected by `--changed` (against `origin/master`) or `--base REF`. Both compare against the merge base and include local changes. Use `-E` to choose tests within those packages. Without package or base filters, every package is selected. Put xtask options before test-name filters and other nextest arguments. Use `--partition hash:K/N` to run one shard, and `--junit PATH` for results even when tests fail. Empty shards pass. `cargo xtask scenarios` selects the real-daemon integration target. See [Testing](testing.md) for fixtures and test design.

Run `cargo fmt --all` (`fmt`) and stage the formatted files. The commit hook checks staged whitespace and Rust formatting, including partial staging and renames. When the origin registry changes, use `cargo xtask extension-providers` to regenerate `extension/src/background/providers.js`. A sources test compares the table with the registry.

## Hosted check

GitHub runs these jobs in parallel for pull requests, pushes to master and manual runs. The required `check` job requires success from every job below.

| Job | Command |
|---|---|
| `lint` | `cargo xtask fast` checks formatting, the workspace-hack crate and Clippy with and without the application's `sinex` feature. |
| `test` | `cargo xtask test --partition hash:K/4` runs one of four nextest shards. |
| `nix` | `cargo xtask nix` checks Nix formatting, settings rendering and module evaluation. |
| `browser` | `npm --prefix extension run check` runs lint, manifest validation and the JavaScript suite, including generating and checking release archives. |

On a pull request, the test and Nix jobs add `--base HEAD^1` to compare GitHub's merge commit with the base branch tip it was tested against. Rust tests cover changed packages and their dependents, with narrower selection where the changed files or test functions allow it. Workspace-wide inputs can select every test. Markdown outside packages selects tests only when a test declares it as an input. Nix checks run when changes touch `nix/`, `flake.nix` or `flake.lock`. Master pushes and exact-revision manual runs check everything.

Read the test job summary for failed tests and tests taking at least 10 s. Logs include failure output, and artifacts `junit-K` record outcomes and durations when nextest writes a report. Use `gh run view RUN --log-failed` to open the failing logs and compare with the same test on master. For local results use `cargo xtask status` and `cargo xtask history runs`.

Use `cargo xtask check` locally to reproduce a hosted failure. It runs formatting, workspace-hack verification, Nix checks, Clippy and Rust tests in sequence. Add `--changed` or `--base REF` to narrow the Nix checks and tests. With either flag a diff confined to Nix inputs selects the Nix checks in place of Clippy and Rust tests.

Hosted Rust jobs use rustup and the flake's `check-tools` package for nextest, cargo-hakari, zsh, fish, zip and Node. They can reuse master's saved build for unchanged packages. Only `lint` on a master push saves it, after `cargo xtask test --no-run` builds the test binaries and `cargo xtask prune-build` removes artifacts unused by Clippy, tests and xtask. Use `cargo xtask warm` to prepare a target with both Clippy configurations and the test binaries.

## Landing

Resolve review threads after answering their findings, then mark the pull request ready. Apply `hold` and comment with the reason to keep it open. `dev/land --queue` considers open pull requests targeting master that are ready for review and clear of holds, conflicts and unresolved threads. It merges eligible green heads oldest first, checking each against master's tip after the previous merge.

The required check records the `tested base`. If master and the pull request changed any of the same packages, the script merges master into the branch and requires a fresh check. A missing `tested base` also requires an update for changes beyond Markdown. Directories with a `Cargo.toml` count as packages, other files count by top-level directory, and Markdown is ignored. The root manifest, lockfile, toolchain, flake, `.cargo/`, `.config/` and `.github/` count as touching every package. Master can move again before the squash merge, and the post-merge check also covers interactions across packages.

The queue runs one pass a minute. It updates branches after merging, one per set of overlapping packages. A branch waits if another pull request touching those packages has check jobs queued or running or was just updated. It reports actions and reasons for waiting after merges and updates or every ten minutes. Failed checks and conflicts stay visible while the queue continues. Add `--once` for one pass, which exits unsuccessfully on a failed check or conflict. Restart the queue as needed using GitHub's current state.

Use `dev/land NUMBER` to wait for and land one pull request. It stops on a draft, a hold, unresolved review, conflicts or failed checks and names failing jobs. It reports the head, tested base, base tip, changed package sets and check runs at the landing decision. With either form, `--dry-run` reports planned actions. For a single pull request it exits successfully only when it would merge. Both forms squash-merge the tested head. The script uses Git with `origin`, gh and awk.

## A daemon to probe

`cargo xtask run` builds and starts a standalone daemon on a scratch root under `$TMPDIR` and removes the temporary root at shutdown. Give `--root DIR` to reuse a root and reopen its archive. Put session files in the printed home, open the printed web address, or use the printed CLI script with `status`. The testkit sandbox passes through `PATH`, sets home, XDG, source and temporary directories under the root, and chooses a free web port.

Use `--watch` to rebuild on source changes and restart on success. A failed build leaves the previous daemon serving. Each checkout builds and serves one development daemon at a time from its own sources. Use `--binary PATH` instead of `--watch` to serve a prebuilt binary. Stopping the command stops the daemon, including on Ctrl-C or SIGTERM.

## Measuring

Use paired benchmarks on the same host and sample. Draw with `cargo xtask bench sample`, measure the first build with `cargo xtask bench run` and add `--incremental` to measure appends. Use `cargo xtask bench pair --a BASE=PATH --b CHANGE=PATH` with two prebuilt `first_build` executables. Use `cargo xtask bench estimate` to fit results from multiple sample sizes and `cargo xtask bench validate RESULT` to compare a prediction with a full result.

Use `cargo xtask bench tails` for the last pages of archived sessions or `cargo xtask bench reads` for synthetic cold and warm reads. Choose where to keep benchmark manifests, sample copies, archives and reports with `--root DIR`. By default the location comes from `POLYLOGUE_BENCH_HOME` or the `polylogue-bench` directory under XDG data. See [Testing](testing.md#benchmarks) for figures to report.

On the host runner, use `measure` for benchmarks, index rebuilds with a prebuilt binary and measurement drivers. It admits one job at a time with 10 GiB free, runs at low CPU and I/O priority and removes the paid embedding key. Coordinate runs sharing a disk before comparing timings.

Set `POLYLOGUE_TIMING_WORKERS` to print pipeline stage and worker CPU times at daemon exit. On GNU/Linux this also prints C allocator statistics.

## Shared crates

Start shared changes in polylogue-sinex-common. Use `cargo xtask link-common PATH` to build Polylogue against that checkout. With no path it uses `POLYLOGUE_SINEX_COMMON` or the sibling checkout. Run `cargo xtask unlink-common` before committing locks. Common's `products` checks exercise the products with the proposed revision. After common merges and Sinex adopts the revision, the `Update product locks` workflow opens or updates Polylogue's `update-polylogue-sinex-common` pull request. It moves the Sinex pin and the common lock together. Convert callers there. See the [shared dependency guide](https://github.com/Sinity/polylogue-sinex-common#depending-on-this-repository).

## Package

`nix build` builds the release package with the `sinex` feature for event kinds. The packaged daemon opens standalone archives. `polylogue --version` reports the package revision, and version and help work before deployment. `nix flake check` also checks extension packaging, Rust and Nix formatting, settings and both modules. Its `behavior` VM exercises the packaged daemon, CLI, MCP and native messaging. Its `services` VM exercises the Home Manager service and aliases, literal paths and Git discovery.

`nix build .#polylogue-extension` produces Chrome and Firefox archives and an unpacked Chrome extension. The overlay exports both packages. Standalone setups using the bare package need Git on `PATH` for repository identity and commit capture. For a local artifact use `npm --prefix extension run build -- --out DIR`, with `--version VERSION` after the separator to override the artifact version only. The builder strips development suffixes and validates the browser version. Output includes `build-manifest.json`. For a source release use `npm --prefix extension version VERSION --no-git-tag-version`, then update `extension/manifest.json` to the browser version and review it with the package and lock. Test the unpacked extension in a browser to qualify runtime behaviour.

The `package` workflow publishes each master commit's runtime closure and the `polylogue-dependencies` compiler artifacts to `sinity.cachix.org` for deployment and subsequent builds. For a separate revision use `gh workflow run check -f rev=SHA` with its full commit hash, then `gh workflow run package -f rev=SHA`. The check verifies the requested commit. Deploy a revision after its package is published. The package log records the source path, revision, Nix archive hash and closure size.

## Deployment modules

Import `homeModules.polylogue` or `nixosModules.polylogue` and declare the archive in `services.polylogue`.

```nix
services.polylogue = {
  enable = true;
  archives.work = {
    path = "/home/alice/archive";
    settings.http.listen = "127.0.0.1:7777";
  };
};
```

Each archive gets a `polylogue-NAME` command and service sharing explicit archive and XDG paths. With several archives, set `defaultArchive` for the `polylogue`, `plg` and `plog` aliases. NixOS also requires an existing `user` and runs its commands with that account's home and configured XDG paths. Wrappers put the caller's Git first and supply their own when needed.

Set `settings`, `sinex` and `signals` as described in [configuration](installing.md#configuration). Omitted settings use the binary defaults. Supply credential file paths. For deployments on Sinex the module requires the day setting to come from Sinex. Activation restarts services on a package change. Restart after a configuration-only change to apply it. Services use readiness notification, a watchdog and CPU and I/O weights below interactive work. After a storage failure or archive refusal the service stays stopped. Register hooks, MCP, the browser host and shell completion separately with [`polylogue install TARGET`](installing.md#installing).

## Host job runner

[AgentCTL operations](../.agentctl/project.toml) clear the inherited `CARGO_TARGET_DIR` so the shell can select and seed the checkout's target. `fast` and `compile` share the compile queue, `focused` uses the normal queue, and `check` and `package` use the bulk queue. `package` runs the full flake checks. `fmt` uses the formatting shell, and `lock` updates selected lock entries. Use `extension` for npm commands, `extension_package` for the extension build and Nix formatting, and `land_queue` for the landing queue. For a separate build target use `exec` with `env CARGO_TARGET_DIR=DIR` before the command, and use that directory exclusively for this checkout.
