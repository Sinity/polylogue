# Development

`rust-toolchain.toml` selects Rust and its components for rustup, the Nix
shell and the packages; the flake reads it through its pinned rust-overlay
input. Change the compiler version in common and both products together.

| Command | Does |
|---|---|
| `cargo xtask check --changed` | formatting, Clippy (unchanged crates reuse their result) and the tests your changes since the merge base with `origin/master` can affect: the loop while editing |
| `cargo xtask check [--changed-base <ref>]` | the same with every test, or with the tests the changes since the merge base with `<ref>` can affect |
| `cargo xtask fast` | formatting and clippy, without tests |
| `cargo xtask test -p <crate> -E 'test(<name>)'` | the selected tests under nextest |
| `cargo xtask test --changed` | tests selected through changed packages and their dependents (`--changed-base=<ref>` for another base than `origin/master`) |
| `cargo xtask test --no-run` | build every test binary without running tests |
| `cargo xtask warm` | build every test binary and both Clippy configurations without running tests |
| `cargo xtask scenarios` | end-to-end tests of the real daemon, CLI and MCP server |
| `cargo xtask bench ...` | corpus builds and interactive read measurements |
| `cargo xtask extension-providers` | rewrite the browser extension's provider table (`extension/src/background/providers.js`) from the origin registry; a sources test fails while the checked-in table differs |
| `cargo xtask status`, `cargo xtask history ...` | what failed in this checkout's last run, and the recorded run history |
| `cargo fmt --all` | format |
| `nix flake check` | locked package, source, module and VM checks |
| `nix build` | the package |

`cargo xtask check` also builds the flake's `nix-fmt`, `settings` and `modules` checks (formatting, deployment-file rendering and module evaluation) when every test runs, and when the changes touch `nix/`, `flake.nix` or `flake.lock`. A diff confined to those files runs them instead of Clippy and the Rust tests.

The Nix package is built with the Cargo feature `sinex` (off in a plain `cargo build`), which holds Polylogue's Sinex event kinds; primary mode is not built in this version ([primary-mode.md](primary-mode.md)). `packages.polylogue-extension`
builds the Chrome and Firefox archives with the extension's Node builder and
provides an unpacked Chrome extension; the overlay adds both packages. Package
builds set `POLYLOGUE_REVISION` at compile time, and `--version` shows it.
`--version` and `--help` load no deployment file.

Extension packaging reads the committed registration in `extension/manifest.json` and the explicit runtime inputs in `extension/scripts/package.mjs`. `npm --prefix extension run build -- --out DIR` writes Chrome ZIP and Firefox XPI archives plus `build-manifest.json`; it never rewrites source versions. `--version VERSION` changes only the artifact version, stripping development suffixes while rejecting malformed browser versions. The packager derives the Firefox background scripts and Gecko identity from that manifest. Artifact checks do not qualify execution in a real browser.

To change the source release version, run `npm --prefix extension version VERSION --no-git-tag-version`, update the committed manifest version to its browser version, then review the package, lock and manifest together. `npm --prefix extension run check` runs lint, manifest validation and the JS suite, including real zip/Python archive generation, the released members and source-byte preservation. The required hosted check runs it too. The declared AgentCTL `extension` operation runs these npm commands without a Rust test build. `nix build .#polylogue-extension` checks packaging from read-only Nix source and provides the unpacked Chrome files. The declared `extension_package` operation builds that output and the Nix-format check only.

An explicit `-p PACKAGE` selects tests to execute while keeping the
workspace feature graph. Cargo target flags limit what builds: use
`--lib` for library unit tests, `--test NAME` for an integration target, or
`--bin NAME` for a binary's unit tests. Without a target flag, targets marked
`test = true` are selected, including opted-in examples. Package, exclusion,
filterset and changed-package predicates still intersect when selecting which
tests run.

All test selections use the primary workspace feature list in
`xtask/src/main.rs`, including test-only dependencies. A package filter changes
execution rather than resolving a separate dependency feature union. Named
`--bin` and `--test` targets build only matching targets; `--lib` builds the
workspace's library test targets. Without a target flag, all test
targets are built. The required check and warm command keep that full graph;
Clippy covers all targets both with and without the `sinex` feature. For an
intentional isolated dependency investigation, invoke Cargo directly with its
own package and feature flags; that creates a separate compilation graph.

Shared dependency features are declared once in the root manifest's
`[workspace.dependencies]`; workspace crates add no further features, except
test-only dev-dependencies. Focused runs, checks and warm-up use that same
graph so they can reuse compiler units.

The package uses a Crane dependency build with the package's own feature
and package selection and without test compilation. Clippy and nextest share
a separate workspace dependency build. The `polylogue-dependencies` package
exposes the production compiler artifacts for the package cache.
Clippy checks the primary and standalone configurations separately. The
nextest check runs the standalone workspace with the same engine and store
fault features as `cargo xtask test`, without the real-process
`polylogue::e2e` cases that need the operating system's machine identity. Its
offline shell-completion cases remain included. The behavior VM
runs the packaged daemon, CLI status, MCP initialization and a native-messaging
exchange in a guest with private XDG and archive paths. Formatting, extension packaging, deployment-file rendering and Nix formatting
are flake checks too. The `modules` check evaluates both deployment modules,
and the `services` VM starts the Home Manager user service and reaches its
daemon through every installed alias and MCP. Its regular account home lets
Home Manager activate; literal spaces, dollar signs and percent signs in the
archive and XDG data, state and cache paths are checked against the running
daemon's arguments and environment. Module evaluation separately checks those
characters in the Home Manager profile command path.

The development shell installs the check tools and points Git at `.githooks`
when no other hooks path is configured; it builds no VM. It carries coverage and
fuzzing tools, LLVM tools matching the compiler, SQLite, jq and the time zone
database. The formatting shell, `nix develop .#format`, uses the same pinned
Rust toolchain with Git and diff, without the check inputs and other development
tools. It installs the same Git hooks.

The commit hook checks staged whitespace and refuses staged Rust that rustfmt would
change (format it and stage again), including partial staging and renames. It enters the formatting shell once for
the complete staged Rust set; the declared `fmt` operation uses that shell too.
Formatting supplies an unused local target directory and selects no compiler
cache. The lock operation likewise supplies an unused local target before shell
entry because dependency resolution uses no compiler artifacts. There is no push
hook. GitHub Actions runs `cargo xtask check --changed-base` for every pull request
(affected tests) and `cargo xtask check` for every master push.

`direnv allow` loads the same shell through `.envrc`, which reloads when the
toolchain or Nix files change. The git-ignored `.envrc.local` holds personal
settings such as build-output placement, job counts and compiler wrappers.
Cargo's output directory is `target/` unless `CARGO_TARGET_DIR` is set.

Declared AgentCTL operations clear an ambient `CARGO_TARGET_DIR` before entering
the target workspace's shell. The `measure_run` operation sets an unused local
target so a prebuilt measurement skips compiler cache selection and seeding.
Untimed `import_preserved` runs share the measurement queue with benchmarks,
so these memory-heavy operations run one at a time. Their memory budget is
admitted before they start; the parent job-plane protections remain in force.
Coordinate runs that use the same disk before comparing timings.
When the operation leaves the target unset, the shell can select that checkout's
isolated host target and seed its dependency artifacts through `sinnix-cargo-target`.
For an explicit target experiment, pass
`exec -- env CARGO_TARGET_DIR=... cargo ...`; this assignment runs after shell
selection. Keep the experimental target exclusive to the same checkout.

To build against a local checkout of the shared crates, run
`cargo xtask link-common PATH`, and `cargo xtask unlink-common` before
committing a lock file, as the shared
[dependency guide](https://github.com/Sinity/polylogue-sinex-common#depending-on-this-repository)
describes.

For an early type-check of one package, `cargo check --locked -p
polylogue-sinex --tests` checks its library target without compiling a test
executable. The declared AgentCTL `compile` operation accepts those Cargo check
arguments in the same one-slot compile queue. `fast` and the required hosted
checks keep their complete feature matrices.

The `polylogue` package is binary-only: its daemon, CLI, MCP and web unit tests
live in `--bin polylogue`. For example, run its reference-query tests with
`cargo xtask test -p polylogue --bin polylogue -E 'test(daemon::query::tests::references)'`.

A host may queue these commands through its own job runner. AgentCTL routes
`fast` and `focused` through the same one-slot `compile` queue. Selected tests
therefore wait for source checks rather than long runtime checks, and their
builds do not overlap a metadata check. Full checks and Nix checks keep
their `bulk` queue. The `package` operation runs all locked Nix checks;
`nix build` builds the package alone.

## Published package

The `package` workflow builds `.#default` for every master commit and pushes
its runtime closure and production compiler artifacts to `sinity.cachix.org`.
The next source-only package build reuses those compiler artifacts. Sinnix deploys a revision only
after fetching that package, so its switches never compile Polylogue. To
deploy a revision master does not contain, such as a deployment branch's head,
qualify that tree with `gh workflow run check -f rev=<full-commit-sha>`,
then publish it with `gh workflow run package -f rev=<full-commit-sha>`. The manual
check records and asserts its checked commit and runs the full check. Pull requests
continue to check their merge with the default branch. Manual checks may reuse
master's build cache but never replace it or its recorded source commit. Master pushes
also restore that cache and date source files against its recorded commit, so
Cargo rebuilds changed units while the full test suite still runs. Only master
pushes save the target directory, labeled with the checked source commit. The package
log records the fetched flake source path, revision and NAR hash before building.

## Deployment modules

Import `homeModules.polylogue` into Home Manager or `nixosModules.polylogue`
into NixOS. Both take the same settings:

```nix
services.polylogue = {
  enable = true;
  archives.work = {
    path = "/home/alice/archive";
    settings.http.listen = "127.0.0.1:7777";
  };
};
```

Every archive has its own `polylogue-NAME` command and service. With several
archives, `defaultArchive` names the one also installed as `polylogue`, `plg`
and `plog`. The command, the daemon and its clients share explicit archive and
XDG paths. The NixOS module also needs `user`, an existing account, and its
client commands take that account's home and XDG directories (which can be set)
whoever runs them.

`settings`, `sinex` and `signals` render the files described in
[installing](installing.md#configuration). Omitted values keep the binary's
defaults. Credentials are file paths, never Nix strings holding their contents.
Primary mode refuses a local `settings.day`.

Activation restarts a service when the package changes; a configuration-only
change takes effect at the next service restart. The units use notify
readiness, a watchdog, no restart after a storage failure or a refused
archive ([operations.md](operations.md#the-daemon)), and CPU and I/O weights below interactive work, with no memory cap.

The Home Manager module also installs the harness hooks and MCP registration,
the browser extension's host manifest, and each archive command's shell
completion with its Alt-p key. Each is the output of
`polylogue install TARGET --dry-run` for that command
([installing](installing.md#installing)), so the module writes exactly what
`install` would.

The `POLYLOGUE_TIMING_WORKERS` profiling environment variable records pipeline
stage times and completed worker CPU times. The daemon prints them at exit
with SQLite's peak memory in bytes and, on GNU/Linux, the C allocator's
statistics. `MIMALLOC_SHOW_STATS=1` separately prints Rust allocator statistics;
allocation totals require building mimalloc with `MI_STAT` enabled.
