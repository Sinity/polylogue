# Development

`rust-toolchain.toml` selects Rust and its components for rustup, the Nix
shell and the packages; the flake reads it through its pinned rust-overlay
input. Change the compiler version in common and both products together.

```sh
nix develop
cargo xtask check
cargo xtask test -p polylogue-store -E 'test(name)'
nix build
nix run . -- --help
nix flake check
nix fmt
```

The Nix package includes the Sinex adapter (the Cargo feature `sinex`, off in a plain `cargo build`). `packages.polylogue-extension`
builds the Chrome and Firefox archives with the extension's Node builder and
provides an unpacked Chrome extension; the overlay adds both packages. Package
builds set `POLYLOGUE_REVISION` at compile time, and `--version` shows it.
`--version` and `--help` load no deployment file.

Native test commands select targets marked `test = true` by default, so examples without tests do not add links. An example with unit tests opts in through its Cargo manifest. Explicit Cargo target flags such as `--lib` or `--example NAME` retain their native selection. Every test command builds the whole workspace with the one feature list in `xtask/src/main.rs` (the Sinex client and the fault features), so a focused run reuses the landing check's units; selectors choose which tests run. Clippy compiles every target in that configuration and again without the Sinex client.

The package, Clippy and nextest outputs share one Crane dependency build.
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
renames. There is no push hook: crate boundaries run in `cargo xtask check`,
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

A host may queue these commands through its own job runner.

## Published package

The `package` workflow builds `.#default` for every master commit and pushes
its runtime closure to `sinity.cachix.org`. Sinnix deploys a revision only
after fetching that package, so its switches never compile Polylogue. To
deploy a branch other than master, build its head first with
`gh workflow run package --ref <branch>`.

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
