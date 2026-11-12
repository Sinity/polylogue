# Development

`rust-toolchain.toml` selects Rust and its components for rustup, the Nix
shell and the packages; the flake reads it through its pinned rust-overlay
input. Change the compiler version in common and both products together.

| Command | Does |
|---|---|
| `cargo xtask check --changed` | formatting, crate boundaries, Clippy (unchanged crates reuse their result) and the tests your changes since the merge base with `origin/master` can affect: the loop while editing |
| `cargo xtask check [--changed-base <ref>]` | the same with every test, or with the tests the changes since the merge base with `<ref>` can affect |
| `cargo xtask fast` | formatting, boundaries and clippy, without tests |
| `cargo xtask test -p <crate> -E 'test(<name>)'` | the selected tests under nextest |
| `cargo xtask test --changed` | tests selected through changed packages and their dependents (`--changed-base=<ref>` for another base than `origin/master`) |
| `cargo xtask test --no-run` | build every test binary without running tests |
| `cargo xtask warm` | build every test binary and both Clippy configurations without running tests |
| `cargo xtask scenarios` | end-to-end tests of the real daemon, CLI and MCP server |
| `cargo xtask bench ...` | corpus builds and interactive read measurements |
| `cargo xtask status`, `cargo xtask history ...` | what failed in this checkout's last run, and the recorded run history |
| `cargo fmt --all` | format |
| `nix flake check` | locked package, source, module and VM checks |
| `nix build` | the package |

The Nix package includes the Sinex adapter (the Cargo feature `sinex`, off in a plain `cargo build`). `packages.polylogue-extension`
builds the Chrome and Firefox archives with the extension's Node builder and
provides an unpacked Chrome extension; the overlay adds both packages. Package
builds set `POLYLOGUE_REVISION` at compile time, and `--version` shows it.
`--version` and `--help` load no deployment file.

An explicit `-p PACKAGE` limits Cargo builds to the selected packages with all
of their own features. Native target flags further limit what builds: use
`--lib` for library unit tests, `--test NAME` for an integration target, or
`--bin NAME` for a binary's unit tests. Without a target flag, targets marked
`test = true` are selected, including opted-in examples. Package, exclusion,
filterset and changed-package predicates still intersect when selecting which
tests run.

Without `-p`, or with explicit `--workspace`, tests retain the primary workspace
feature list in `xtask/src/main.rs`. Package checks can have narrower transitive
features than that full graph. The required check and warm command keep the
full graph; Clippy covers all targets both with and without the Sinex client.
Use `--workspace -p PACKAGE` to retain that graph while filtering execution.

Shared dependency features are declared once in the root manifest's
`[workspace.dependencies]`; workspace crates add no further features, except
test-only dev-dependencies. Focused runs, checks and warm-up use that same
graph so they can reuse compiler units.

The package uses a Crane dependency build with the package's exact feature
and package selection and without test compilation. Clippy and nextest share
a separate workspace dependency build. The `polylogue-dependencies` package
exposes the production compiler artifacts for the package cache.
Clippy checks the primary and standalone configurations separately. The
nextest check runs the standalone workspace with the same engine and store
fault features as `cargo xtask test`, without the real-process
`polylogue::e2e` cases that need the operating system's machine identity. Its
offline shell-completion cases remain included;
`cargo xtask check` still runs every e2e and Sinex client test. The behavior VM
runs the packaged daemon, CLI status, MCP initialization and a native-messaging
exchange in a guest with private XDG and archive paths. Formatting, crate
boundaries, extension packaging, deployment-file rendering and Nix formatting
are flake checks too. The `modules` check evaluates both deployment modules,
and the `services` VM starts the Home Manager user service and reaches its
daemon through every installed alias and MCP. Its ordinary account home lets
Home Manager activate; literal spaces, dollar signs and percent signs in the
archive and XDG data, state and cache paths are checked against the running
daemon's arguments and environment. Module evaluation separately checks those
characters in the Home Manager profile command path.

The development shell installs the check tools and points Git at `.githooks`
when no other hooks path is configured; it builds no VM. The commit hook checks
staged whitespace and formats staged Rust files, including partial staging and
renames. Formatting-only shell entry supplies an unused local target directory
so it does not select or seed a compiler cache. The hook enters the shell once
for the complete staged Rust set; the declared `fmt` operation also skips cache
selection. The lock operation likewise supplies an unused local target before shell entry because dependency resolution uses no compiler artifacts. There is no push hook: crate boundaries run in `cargo xtask check`,
which GitHub Actions runs for every pull request. The shell also carries coverage and fuzzing tools, LLVM tools matching
the compiler, SQLite, jq and the time zone database.

`direnv allow` loads the same shell through `.envrc`, which reloads when the
toolchain or Nix files change. The git-ignored `.envrc.local` holds personal
settings such as build-output placement, job counts and compiler wrappers.
Cargo's output directory is `target/` unless `CARGO_TARGET_DIR` is set.

Declared AgentCTL operations clear an ambient `CARGO_TARGET_DIR` before entering
the actual workspace's shell. The shell selects that checkout's isolated host
target when `sinnix-cargo-target` is available, with dependency artifacts seeded
by the host helper. For an explicit target experiment, pass
`exec -- env CARGO_TARGET_DIR=... cargo ...`; this assignment runs after shell
selection. Keep the experimental target exclusive to the same checkout.

To build against a local checkout of the shared crates, run
`cargo xtask link-common PATH`, and `cargo xtask unlink-common` before
committing a lock file, as the shared
[dependency guide](https://github.com/Sinity/polylogue-sinex-common#depending-on-this-repository)
describes.

For an early type-check of one package, `cargo check --locked -p
polylogue-sinex --tests` checks its native target without compiling a test
executable. The declared AgentCTL `compile` operation accepts those Cargo check
arguments in the same one-slot compile queue. `fast` and the required hosted
checks retain their complete feature matrices.

A host may queue these commands through its own job runner. AgentCTL routes
`fast` and `focused` through the same one-slot `compile` queue. Selected tests
therefore wait for source checks rather than long runtime proofs, and their
builds do not overlap a metadata check. Full checks and Nix checks retain
their `bulk` queue. The `package` operation runs all locked Nix checks;
`nix build` builds the package alone.

## Published package

The `package` workflow builds `.#default` for every master commit and pushes
its runtime closure and production compiler artifacts to `sinity.cachix.org`.
The next source-only package build reuses those compiler artifacts. Sinnix deploys a revision only
after fetching that package, so its switches never compile Polylogue. To
deploy a revision master does not contain, such as a deployment branch's head, build
it first with `gh workflow run package -f rev=<commit>`.

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
client commands run as that account; its home and XDG directories can be set.

`settings`, `sinex` and `signals` render the files described in
[operations](operations.md#configuration). Omitted values keep the binary's
defaults. Credentials are file paths, never Nix strings holding their contents.
Primary mode refuses a local `settings.day`.

Activation restarts a service when the package changes; a configuration-only
change takes effect at the next service restart. The units use notify
readiness, a watchdog, the exit code for a refused schema, and CPU and I/O
weights below ordinary interactive work, with no memory cap.

Designed and not built yet: the hooks, MCP registration and browser extension
host that the Home Manager module installs into harnesses
([operations](operations.md#installing-into-harnesses)). Completions for the
archive commands come from the application.
