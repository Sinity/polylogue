# Polylogue Rust foundation

This branch prepares the standalone Rust replacement in isolated state. It implements foundation identity/value contracts, durable SQLite schema checks, shared content files, one archive writer, read-only handles and a real daemon/status client. Session acquisition, parsing, search, read/export, MCP and HTTP are not registered yet. Status explicitly reports `product_path: "not_implemented"`; an initialized empty schema is not a completed ingestion or M1 demonstration.

## Build and run

```sh
nix flake check
nix build .#polylogue
./result/bin/polylogue --help
./result/bin/polylogue --version
./result/bin/polylogue config schema
./result/bin/polylogue --archive /path/to/isolated-archive daemon
./result/bin/polylogue --archive /path/to/isolated-archive status
```

Use fresh isolated state. `XDG_RUNTIME_DIR` must be absolute. The archive defaults to `$XDG_DATA_HOME/polylogue`, or `$HOME/.local/share/polylogue`; an explicit `--archive` takes precedence. No product-specific environment variable selects an archive. Static help/version/schema do not open or create state. Unsupported commands and bad arguments return a non-success outcome instead of an empty answer.

The foreground daemon owns the archive lock, writable schema and its derived socket. Ctrl-C stops admission, closes clients and joins handlers before releasing the writer. A second archive can run independently; competing archive aliases and occupied endpoints are refused. Startup never adopts a foreign listener. Missing/empty database files with retained recovery sidecars require explicit recovery, and newer durable schemas are refused before mutation.

## Development

The declared AgentCTL operations run the Nix build/check and development-shell commands. `cargo run -p xtask -- check` checks target formatting, actual Cargo dependency boundaries and workspace tests. A synthetic Cargo-edge mutation and compile-fail examples demonstrate forbidden access. `.githooks` contains successor hooks; in linked worktrees select them through per-worktree Git configuration, never by changing another checkout's shared hook configuration.

`crates/model` is pure. Only `crates/store` owns SQLite connections; public handles expose typed operations. `crates/hashed-blobs` is the reviewed shared file layer plus an explicit read-only capability, documented in [source provenance](docs/shared-crate-source.json). The local owner snapshot is provisional until the reviewed shared-crate publication/pin is chosen. [Store limits](crates/store/README.md) distinguish these primitives from complete input acquisition and recovery.

## Delivery boundary

[Replacement disposition](docs/REPLACEMENT.md) records predecessor removal and preserved legal/history material. [Contract snapshot](docs/rewrite/README.md) links current semantic owners; canonical DDL provenance lives beside `crates/store/schema`.

This candidate does not change master, deployed packages, consumers, archives or task routing. Shared dependency publication, external consumer pins, hosted successor checks and the full replacement review remain open. Checksum failures on main-database page reads report the actual file and page; Cold durable WAL startup uses a cancellable bounded-memory integrity guard under archive ownership; ambiguous, partial and stale-generation tails require controlled recovery without changing retained bytes. This can refuse legitimate crashes and does not prove an external expected history head. Full WAL recovery remains open. Index rebuild/swap, cross-file read views and the first source-to-client product path require further work. No broad M0/M1 or deployment acceptance is claimed.
