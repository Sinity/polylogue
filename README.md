# Polylogue Rust foundation

This repository contains the standalone Rust foundation. It implements foundation identity/value contracts, durable SQLite schema checks, shared content files, one archive writer, read-only handles and a real daemon/status client. Explicit immutable Codex regular-file import and provider-state progress are registered. Selected uncontested Codex Messages have a streaming terminal reference read. Full canonical finalization, session JSON read/export, search, MCP and HTTP remain unfinished. Status reports `product_path: "partial"`; an initialized empty schema is not a completed ingestion or M1 demonstration.

## Build and run

```sh
nix flake check
nix build .#polylogue
./result/bin/polylogue --help
./result/bin/polylogue --version
./result/bin/polylogue config schema
./result/bin/polylogue --archive /path/to/isolated-archive daemon
./result/bin/polylogue --archive /path/to/isolated-archive status
./result/bin/polylogue --archive /path/to/isolated-archive import /path/to/rollout.jsonl --format codex-rollout
./result/bin/polylogue --archive /path/to/isolated-archive progress INPUT
```

Use fresh isolated state. `XDG_RUNTIME_DIR` must be absolute. The archive defaults to `$XDG_DATA_HOME/polylogue`, or `$HOME/.local/share/polylogue`; an explicit `--archive` takes precedence. No product-specific environment variable selects an archive. Static help/version/schema do not open or create state. Unsupported commands and bad arguments return a non-success outcome instead of an empty answer.

The foreground daemon owns the archive lock, writable schema and its derived socket. Ctrl-C stops admission, closes clients and joins handlers and its capture/compute workers before releasing the writer. A second archive can run independently; competing archive aliases and occupied endpoints are refused. Startup never adopts a foreign listener. Missing/empty database files with retained recovery sidecars require explicit recovery, and newer durable schemas are refused before mutation.

## Development

The declared AgentCTL operations run the Nix build/check and development-shell commands. `cargo run -p xtask -- check` checks target formatting, actual Cargo dependency boundaries and workspace tests. A synthetic Cargo-edge mutation and compile-fail examples demonstrate forbidden access. `.githooks` contains successor hooks; in linked worktrees select them through per-worktree Git configuration, never by changing another checkout's shared hook configuration.

`crates/model` is pure. Only `crates/store` owns SQLite connections; public handles expose typed operations. `crates/hashed-blobs` is the reviewed shared file layer plus an explicit read-only capability, documented in [source provenance](docs/shared-crate-source.json). The owner publishes the reviewed file layer as `hashed-blobs-v0.1.0`; consumer pins require their own verification. [Store limits](crates/store/README.md) distinguish these primitives from complete input acquisition and recovery.

## Delivery boundary

[Replacement disposition](docs/REPLACEMENT.md) records predecessor removal and preserved legal/history material. [Contract snapshot](docs/rewrite/README.md) links current semantic owners; canonical DDL provenance lives beside `crates/store/schema`.

The foundation source and standalone `hashed-blobs-v0.1.0` release are published. Consumer pins, runtime activation and archive migration require their own qualification. Checksum failures on main-database page reads report the actual file and page. Cold durable WAL startup uses a cancellable bounded-memory integrity guard under archive ownership; ambiguous, partial and stale-generation tails require controlled recovery without changing retained bytes. This can refuse legitimate crashes and does not prove an external expected history head. Full WAL recovery remains open. Index rebuild/swap, cross-file read views and the first canonical source-to-query product path require further work. Canonical M1 and deployment acceptance remain open.

An import reply certifies durable raw retention, independently of provider parse
progress. The client prints its UUIDv7 request ID and acquisition time before
sending; reuse both with `--request-id ID --seen-at MICROS` to retry the same
action, including after the original path disappears. A reused ID with changed
metadata is refused. Provider work runs outside the writer actor, which admits
one provider installation transaction between peer requests. `progress` reports
raw offsets, accounting, failure and pending finalization; EOF does not make
`canonical_complete` true. `retry INPUT` explicitly retries a failed disposable
parse. `--wait`, `--dry-run`, archives/directories, symlinks and other source
formats remain unavailable. Claude context is retained by the store primitive
but is not yet admitted by this public source route.

The actor uses fixed record/cell/request admission allowances. Oversized records
fail explicitly with owed work retained; configurable larger allowances and full
resource qualification remain unfinished. Its current source scheduling follows
one imported input to provider EOF; cross-input fair parse rounds and canonical
finalization are still owed. Public search refuses even after provider EOF.

The first canonical worker path prepares uncontested Codex Text/Thinking
Messages and typed source fields through durable private roots, then selects a
whole certified Message body. Only the terminal Message-reference read is exposed; session JSON read, search
and export remain unavailable. Competing inputs, unsupported operations and unresolved
mapping remain pending; `canonical_complete` remains false. Hash validation is
cooperative off actor but can still process a whole Message/value per worker
call, and normalized-leaf windowing can repeat serialization. These costs have
not earned bounded worker-turn or append-local resource acceptance.

Read one selected Message with `polylogue --archive DIR message:<opaque-id>
read --format terminal`. This uses the stored Session relation, preserves one
whole selected body and its interleaved reasoning/text order, and streams UTF-8
payload ranges without collecting the full Message. Control characters are
escaped for terminal safety. The command explicitly exits 5 (degraded) because
history and source-locator coverage remain incomplete. Pending references,
truncated streams and unsupported formats fail explicitly; `read --format json`
is not a substitute export shape.

Import accepts paired `--machine ID --host LABEL` acquisition-observation metadata. These identify the collector, not the original session machine or the immutable interpretation context. Without the pair, the client reads `/etc/machine-id` and the local hostname. Explicit retries must reuse the original observation metadata, request ID and timestamp.

Selected terminal reads mask supported credential shapes in the daemon before socket encoding. The streaming adapter covers GitHub/GitLab/npm/Hugging Face/Google/Stripe/AWS/Anthropic/OpenAI prefix families, six PEM private-key headers and lowercase HTTP(S) credentials. It preserves matches across text pieces and merges overlapping matches; undecided URL regions above 512 bytes return an explicit error. Retained bytes and canonical values are unchanged. Generic secret assignments, JWT/Bearer patterns, derived `has:secret`/status counts and reveal remain unsupported; this is not comprehensive credential-policy qualification.
