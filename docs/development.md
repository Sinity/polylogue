# Development

`rust-toolchain.toml` selects Rust and its components for rustup, the Nix
shell and packages. The flake reads it through its pinned rust-overlay input.
Change the compiler version in common and both products together.

```sh
nix develop
cargo xtask check
cargo xtask test -p polylogue-store -E 'test(name)'
nix build
nix run . -- --help
nix flake check
nix fmt
```

The default package includes the Sinex adapter. `packages.polylogue-extension`
builds Chrome and Firefox archives through the extension's existing Node
builder and provides an unpacked Chrome extension. The overlay adds both
packages. Package builds supply `POLYLOGUE_REVISION` at compile time; the
behavior check requires `--version` to expose that revision. Version and help
remain early exits that load no deployment file.

The flake's package, Clippy and nextest outputs reuse one Crane dependency
artifact build. Clippy checks primary and standalone configurations separately.
The sealed nextest check uses standalone workspace selection and the same
engine/store fault features as the native runner. It excludes the real-process
`polylogue::e2e` binary, whose startup requires OS machine identity. The behavior
VM exercises the real packaged daemon, CLI status, MCP initialization and a
native-messaging exchange in a guest OS, with private XDG and archive paths.
The native full check retains every application e2e and Sinex client test.
Formatting, crate boundaries, extension packaging, shared deployment-file
rendering and Nix formatting are also flake checks. The deployment renderer
omits runtime-owned defaults and keeps credential contents out of the store.
The `modules` check evaluates both deployment modules and their restart
identity. The `services` VM starts the Home Manager user unit and reaches its
owned daemon through every installed alias and MCP. The package behavior VM
also starts a separate daemon directly for its transport checks.

The development shell installs check tools and configures `.githooks` when Git
has no explicit hooks path and preserves a configured custom path. It does not
build a VM. Repository commit hooks check staged whitespace and format the
staged Rust blobs, including partial staging and renames. Push hooks check
formatting and crate boundaries for changed Rust inputs. The xtask command owns
any services its run needs. The shell also carries coverage/fuzz tools,
compiler-matched LLVM tools, SQLite, jq and the timezone database.

`direnv allow` loads the same shell through `.envrc`, which watches the declared
toolchain and Nix source files. The ignored `.envrc.local` holds personal
build-output placement, job counts and compiler wrappers. Cargo's default output
directory is `target/`; callers may set `CARGO_TARGET_DIR` explicitly.

For local shared-crate development, run `cargo xtask link-common PATH`
explicitly. Run `cargo xtask unlink-common` before committing workspace
locks, as described in the shared
[dependency guide](https://github.com/Sinity/polylogue-sinex-common#depending-on-this-repository).

A host may queue these commands through its own job runner.

## Deployment modules

Import `homeModules.polylogue` into Home Manager or `nixosModules.polylogue`
into NixOS. Both use the same deployment schema:

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
archives, select `defaultArchive`; its wrapper is also installed as
`polylogue`, `plg` and `plog`. Wrapper, daemon and clients share explicit
archive and XDG paths. The NixOS module also requires `user`, an existing
account; run its client commands as that account. Its home and writable XDG
directories can be set explicitly.

`settings`, `sinex` and `signals` render the files described in
[operations](operations.md#configuration). Omitted values keep the binary's
defaults. Credentials are file paths, never Nix strings containing their
contents. Primary mode refuses a local `settings.day`.

Activation restarts a service when the application package changes.
Configuration-only changes take effect on the next explicit service restart.
The units use notify readiness, a watchdog, schema-refusal exit handling and
CPU/I/O weights below ordinary interactive work, with no memory cap.

Managed harness and browser registrations still need their product
declaration adapters; these module delivery criteria remain open. Archive
command completion uses the application's command-name-aware producer.
