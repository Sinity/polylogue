# Rust workspace

A fresh workspace in the existing repository after verified preservation and an executable-bootstrap
replacement PR (`decisions/repository.md`). Product slices then target its verified default branch.
Composition skeletons alone do not satisfy bootstrap. Existing packages stay pinned until cutover. Stable Rust, edition 2024, built with Nix (crane)
and plain `cargo`.

## Crates

```
polylogue/
  Cargo.toml                 workspace: shared dependency versions and lint table
  crates/
    model/      polylogue-model     plain types and pure rules; no IO
    hashed-blobs/ hashed-blobs      large blobs as seekable zstd files named by BLAKE3; no Polylogue dependency; shared with Sinex
    sources/    polylogue-sources   provider detection and parsers
    store/      polylogue-store     SQLite files, blob store, writes, reads
    query/      polylogue-query     query language: parse, plan, compile to SQL; reference evaluator
    engine/     polylogue-engine    the convergence loop, derived data, analysis
    render/     polylogue-render    terminal, Markdown, HTML, JSON output
    polylogue/  polylogue           the binary: CLI, daemon, MCP server, HTTP reader, hook command
  xtask/                            developer commands
  tests/                            end-to-end tests
  examples/                         example inputs and expected outputs (also the test data)
  docs/rewrite/                     this spec package, copied whole (spec/, plan/, decisions/, outline/,
                                    skeleton/, examples/archive/); the check scripts run from it and
                                    copy-from-spec.py finds its spec/storage/ by walking up
  flake.nix                         crane build, devshell
```

### Pure core, thin shell

`model`, `sources`, `query` and `render` are pure: no files, no SQLite, no
clock, no environment, no async. They compile and test from bytes and values
alone. `store`, `engine`'s loop and the binary are the shell that does IO.

| Crate | Kind | May depend on (internal) | Must never depend on |
|---|---|---|---|
| `polylogue-model` | pure | – | rusqlite, tokio, pest, rayon, any IO crate |
| `hashed-blobs` | shell | – (no Polylogue crate) | rusqlite, tokio |
| `polylogue-sinex` | shell, not in the default build | model, store | – |
| `polylogue-sources` | pure | model | rusqlite, tokio |
| `polylogue-query` | pure | model | rusqlite, tokio, store |
| `polylogue-render` | pure | model | rusqlite, tokio, store, query |
| `polylogue-store` | shell | model, hashed-blobs | tokio, sources, query, engine |
| `polylogue-engine` | core + shell | model, sources, query, store | tokio, render |
| `polylogue` (binary) | shell | all | – |

`query` compiles a query to SQL text with parameters; `store` executes SQL
text and returns model types. SQL lives only in `store` (fixed statements)
and in `query`'s compiler; `engine` reads and writes through `store`.

**Enforced mechanically**, not by prose:

1. Cargo dependency edges list only the allowed internal crates.
2. `deny.toml` wrapper bans allow `rusqlite` only in `store`, `tokio` only in
   the binary (and the HTTP/MCP crates it uses), `pest` only in `query`,
   `store` only in `engine` and the binary, `render` only in the binary, and
   HTTP client crates (`reqwest`, `hyper` clients) only in `engine` (which
   fetches embeddings and Drive files) and the binary, never in `model`,
   `sources`, `query`, `render` or `store`. The writer lives in `store` and
   receives finished batches over a channel, so no network call can run
   inside a write transaction (`plan/rewrite-rules.md` R36).
3. Each pure crate has a `clippy.toml` with `disallowed-methods` for
   `std::fs`, `SystemTime::now`, `env::var` and `process::Command`;
   `xtask check` runs clippy with warnings as errors.
4. No `pub use` re-exports across crates, so real edges stay visible, and
   no module inside a crate that only re-exports others; only a crate root
   re-exports its public API (rule R80).
5. Domain types live only in `model`; another crate adds no type for a
   concept `model` already has, and each crate has one error enum (rules
   R78, R79). `xtask surface` checks both.
- `clippy.toml` `disallowed-types`: `rusqlite::Connection` in every crate except `store`, so no second writer or lookup is written elsewhere (rule R27). `disallowed-methods` also lists `jiff::tz::TimeZone::system` and `jiff::Zoned::now` in every crate except `render`, so nothing but display uses the local time zone (rule R32).

### What each crate owns

| Crate | Owns | Depends on |
|---|---|---|
| `hashed-blobs` | The file layer of the blob store: writing, reading and range-reading seekable zstd files named by BLAKE3 in one storage root (a second root can be added and is searched in order), with hash checks and a file-mode parameter (`spec/storage/blobs.md`, Blobs). Shared with Sinex's content store. | zstd, blake3 |
| `polylogue-sinex` | The Sinex adapter: publishing to Sinex over its NATS ingress under the producer protocol, the `polylogue.have` holdings query, reading back for a rebuild (`decisions/sinex.md`). Built only with the `sinex` feature or as its own binary; the default build has no NATS dependency. | model, store, `sinex-protocol` (pinned by tag), async-nats |
| `model` | `Session`, `Message`, `Block`, `Attachment`, `Link`, `ToolCall`, `Usage`, typed IDs (`SessionId`, `MessageId`, `BlockId`), timestamps, the identity and hashing rules, the export format. Every rule in `spec/model.md` lives here and nowhere else. | serde, blake3 |
| `sources` | One module per provider: where its files live, how to recognize them, how to parse them into `model` types. Parsers are pure functions of bytes (plus companion files and, for append-only files, the carried state). Raw provider record types as Rust structs (the typed parse is the validation). Unknown fields and record types are collected, not dropped. | model, serde_json, zip |
| `store` | The SQLite schema (`spec/storage/`), inputs, revisions, extents and observations on top of `hashed-blobs`, writing a parsed session, reading rows, full-text and vector search primitives. Plain SQL strings; no ORM. | model, hashed-blobs, rusqlite (bundled) |
| `query` | The query grammar, parsing to a syntax tree, planning, compiling to SQL against `store`'s schema, and a small in-memory reference evaluator used only by tests. | model, pest |
| `engine` | The convergence loop (`spec/convergence.md`): discover inputs, store bytes, parse, write, keep derived data current. Derived data and analysis (`spec/analysis.md`): usage and cost, tool outcomes, profiles, embeddings, schema inference. The library the daemon runs (and tests call in-process, law 83); other programs use the socket. | model, sources, store, query |
| `render` | Turning query results and sessions into terminal output, Markdown, HTML and JSON. No IO beyond writing to a given sink. | model |
| `polylogue` | Argument parsing, the daemon process (file watching, scheduling, the one Unix socket that answers every request as newline-delimited JSON, and the local HTTP mapping of the same requests that serves the web reader), the socket client every command uses, the MCP stdio proxy, the hook producer subcommand. Thin: a request is a call into `engine` or `query` plus `render` inside the daemon. | everything above, clap, tokio (daemon and HTTP only), notify, rmcp, axum |

Rules that keep this shape:

- **Parsing style** (measured, `evidence/ideas-rust.md`): each provider
  record is read into a flat "wire" struct with optional fields, then a plain
  `match` on its type field maps it to model types. No serde untagged or
  internally tagged enums and no `flatten` on provider records: on real data
  they lost 14–16% of records and borrowed strings. `serde_ignored` reports
  skipped fields for the unrecognized counts. Record type names are `strum`
  enums so the compiler forces every known type to be handled.
- **Field tables, not macros**: each query unit's fields are one `const`
  table (name, kind, SQL expression, description, reference-evaluator
  accessor, and the slice that adds it) that drives the compiler, completion,
  errors ("available in slice N"), `explain` and the docs table. No custom proc macros, no `build.rs` code generation.
- **The clock and the time zone are arguments**; `now()` is read once per
  request by the daemon, so every spec example runs with a fixed clock and
  no mocks.
- **A trait needs two real implementations today.** Providers are
  `dyn Provider` in a static registry; there is no `Store` trait until a
  second backend exists.
- **Counts, sums and joins over index rows are SQL**; Rust computes only what
  needs Rust-side data (prices, text, embeddings).
- **Money** is `Usd`, a newtype over integer nano-USD (`i64`), and provider
  credits are a separate `Credits` type (`spec/model.md`, Value types), so sums are exact; rounding happens
  only when displayed.

- `model` has no IO and almost no dependencies (no rusqlite, tokio or pest),
  builds on stable Rust with a published minimum version, so other projects
  (Sinex) can depend on it. The Sinex adapter (`polylogue-sinex`) is an
  optional dependency of the `polylogue` binary behind the Cargo feature
  `sinex`, off by default and on in the operator's Nix build, so the default
  build has no Sinex code; the skeleton check compiles with all features so
  the adapter's wiring is still checked (`decisions/sinex.md`).
- Parsers never touch the database or the filesystem beyond the bytes they are
  given. This makes them testable from examples alone and parallelizable.
- There is exactly one SQL compiler (in `query`) and one schema (in `store`).
- Async is used only where it pays: the daemon's socket and HTTP server,
  MCP, file watching. Parsing and SQLite work run on plain threads (`rayon` for
  parallel parsing, one writer thread).
- Eight crates is a starting point, not a commitment. Split a crate only when
  compile times or a real second consumer demand it.

Sources: joint #11, joint #8.

## Main dependencies

| Need | Crate | Note |
|---|---|---|
| SQLite | `rusqlite` with `bundled` | bundled SQLite 3.53 or newer (FTS5 security fixes); FTS5 included |
| Vector search | none (f32 vectors held in the daemon's memory and scanned in Rust) | `spec/storage/embeddings.md`; sqlite-vec is pre-1.0 with one maintainer and is not needed at this scale |
| JSON | `serde`, `serde_json` | `simd-json` only if measurement says parsing dominates |
| Unknown fields | `serde_ignored` | reports every field the typed parse ignored, by path: the "unknown fields are counted" rule |
| Parse error paths | `serde_path_to_error` | the JSON path of a field that failed to parse, for `parse_state` reasons |
| Time | `jiff` | time zones and RFC 3339; rusqlite supports it directly |
| Hashing | `blake3` | one hash for blobs, inputs, texts and IDs (`spec/storage/blobs.md`); SIMD-accelerated, and its tree structure allows verifying a byte range without hashing a whole blob |
| CLI | `clap` with `clap_complete` | dynamic completion for query fields and values |
| Interactive selection | `nucleo-picker` | fuzzy pickers |
| File watching | `notify-debouncer-full` | plus the periodic rescan, which catches what notifications miss |
| MCP | `rmcp` | official Rust MCP SDK |
| HTTP | `axum` | loopback only |
| Web pages | `askama` templates with htmx | server-rendered; no frontend build. htmx is served by the daemon itself (the page's CSP allows only its own scripts); files written with `--format html` contain no scripts (`spec/rendering.md`) |
| Async runtime | `tokio` | daemon only |
| Parallelism | `rayon` | parse workers |
| Logging | `tracing`, `tracing-subscriber` | structured, with spans for timing |
| Errors | `thiserror` in libraries, `anyhow` in the binary | |
| ZIP | `zip` | export archives, read as streams |
| `archive.db` migrations | `rusqlite_migration` | ordered SQL steps (`spec/storage/archive-db.md`) |
| Drive | `reqwest` with `yup-oauth2` | AI Studio via Drive (slice 5) |
| Extension types | `ts-rs` | TypeScript types for the browser extension generated from the model crate |
| Compression | `zstd` | blob store |
| Query grammar | `pest` | `decisions/query-parser.md`; error messages with a caret under the problem are written by hand (a few dozen lines) |
| Export schema | `schemars` | JSON Schema of the export format generated from the types (`spec/export-format.md`) |
| Git (read-only) | `gix` | commit details for git correlation (slice 11) |
| Testing | `insta`, `proptest`, `trycmd`, `cargo-nextest`, `criterion`, `cargo-fuzz` | see `plan/testing.md` |

## Lints

Borrow Sinex's workspace lint table, trimmed: deny `unwrap_used`,
`expect_used`, `todo`, `dbg_macro` outside tests; deny `dead_code`,
`unused` and `unreachable_pub`, so code nothing calls fails the build and
deleting a caller surfaces what can go with it; clippy `pedantic` as
warnings with a short list of switched-off lints. Those lints cannot see a
`pub` item of a library crate that only tests call, so `xtask surface`
also lists `pub` items with no non-test caller anywhere in the workspace;
the list goes in the PR description beside the surface diff (code kept
alive only by tests was about 1,500 lines in the Python version; `audit
bloat-patterns-rerun` §6).

Added from the closed Beads (`plan/rewrite-rules.md`; each item names its
rule):

- **Errors are not discarded** (R3): `unused_must_use` and
  `clippy::let_underscore_must_use` are denied; `clippy::disallowed-methods`
  in `store`, `engine` and the binary lists `Result::ok`,
  `Option::unwrap_or_default` and `Result::unwrap_or_default` on fallible
  reads, with a short allow-list naming each permitted site and why.
  `std::panic::catch_unwind` is allowed only in the parse pool's per-input
  capture (R39) and at the binary's top level.
- **No made-up identity or zero** (R1, R21): `unwrap_or(0)` and
  `unwrap_or_default()` on IDs, counts and times are in the same
  disallowed list.
- **SQL rules** (R8, R24, R27, R63) are checked by the SQL inventory test
  (`plan/testing.md`, "The spec and the code cannot drift apart") rather
  than a text lint, because the statements are collected there anyway.
- **Test runner** (R60): the repository's nextest config sets
  `--no-tests=fail`, a `slow-timeout` with `terminate-after` for every test,
  and fails on any skipped test in CI.
- **Configuration** (R53): `clippy.toml` in every crate except the binary
  disallows `std::env::var` and `std::env::var_os`. The binary reads its
  settings and environment into one declared `Config` struct in `main`; a
  drift check compares `Config`'s fields with the settings and variables the
  documentation lists, so an undocumented or never-read variable fails. The
  archive directory comes from the XDG default or `--archive`.
- **Function length** (R81): `clippy::too_many_lines` is denied with
  `too-many-lines-threshold = 150` in `clippy.toml`.

## Developer commands (xtask)

A few hundred lines in total, written with `xshell`. Each is a thin wrapper
over cargo or over the binary. (Sinex's xtask grew to about 126k lines; this
one must not.)

| Command | Does |
|---|---|
| `xtask check` | fmt, clippy, build |
| `xtask test` | `cargo nextest run` plus the example (trycmd) tests; records the commit it tests and fails if the tree changes during the run (rule R64) |
| `xtask golden` | re-record expected outputs after an intended change, as their own commit for review; refuses to re-record a format's parser examples unless that format's parser version is higher than at the last recording (rule R42) |
| `xtask gen` | build a generated corpus from the inferred schemas (`--size small|real`) |
| `xtask compare-old` | compare new parser output with the recorded output of the Python parsers on the operator's real files |
| `xtask full-run` | fresh archive from all real sources into a scratch directory; report time, memory, counts |
| `xtask bench` | criterion benchmarks plus the end-to-end generated-corpus benchmark |
| `xtask field-scan` | report, per provider, fields in real files above 0.1% presence that are not mapped, plus the share of null, unknown or trivial values per field (rule R10); informational only |
| `xtask history` | query the build and test history (slowest and flaky tests, when a test started failing, durations over time) |
| `xtask surface` | print the public surface (tables and columns, commands and flags, MCP tools, `pub` items per crate, dependencies) and product and test lines per crate, so a PR can show its diff (rule R82); fails only on the code-shape rules R78–R80 (a domain type outside `model`, two copies of one concept, more than one error enum per crate, a re-export-only module) |

No health scores, no closure checks, no gates. Build and test history is
kept (operator, 2026-10-11: "half the reason for having xtask"): every
`xtask` run appends its structured result (command and arguments, git
revision and whether the tree was dirty, start time and duration, outcome
and duration per package and per test, toolchain, machine) to a history
store outside the repository (`$XDG_STATE_HOME/polylogue-xtask/history.db`),
and `xtask history` queries it (slowest tests, tests that started failing at
a revision, flaky tests, how long `xtask test` takes over time). The
recording comes from the shared xtask core. History is data, never a gate:
nothing refuses to run or merge because of it. Agent test runs are also in
Polylogue's own archive as tool calls; the two complement each other.

No benchmark baselines are committed: `xtask bench` results go to the same
history, and the history is the baseline. Decided by the spec authors from
the history design above (2026-10-11); the operator may overrule.

Sources: joint #13.

## Nix

- `flake.nix` with crane: a cached dependency build, the package, a devshell
  with the stable toolchain, the `mold` linker, `cargo-nextest`,
  `cargo-insta`, `cargo-deny`, `cargo-machete`, `sqlite`.
- The NixOS module and Home Manager module live in the operator's Sinnix
  repository, as today; this repository may carry a minimal module for
  outside users. The binary prints its `Config` as a JSON schema
  (`polylogue config schema`), the flake exports it, and the modules'
  options are generated from it, so a module cannot set a key the binary
  does not read or miss one it does (the same rule Sinex follows for its
  typed configuration).
- Release artifacts: the Nix package, a static Linux binary, and later
  packages for other distributions.

## CI

CI runs on GitHub's free runners for the public repository, on every push
and PR: `xtask check` and `xtask test` inside the Nix devshell, with the
crane dependency build cached, plus the browser extension's JavaScript tests
from slice 4. CI also runs the privacy filter's check over every committed
inferred-schema file and fails on a value the filter would have dropped;
runs the same filter and the `has:secret` patterns over `examples/` and
`docs/`, so fixtures stay synthetic (no operator home paths, real names,
e-mail addresses or tokens); checks with `git check-ignore` that
agent-output directories (`.claude/`, `.codex/`, `.agent/`) inside the
repository are ignored (rule R57); and rejects a commit that adds a file
over 1 MB outside `examples/` (no caches or generated output, rule R64).
Nothing else is gated (slice 1).

Sources: `polylogue-x7du`, `polylogue-stzx`.

## Rules held by construction

The operator's standard (shared with Sinex, `sinex-spec/spec/derivation.md`
§2.5): design so the error cannot happen, rather than giving a component a
duty and testing afterwards whether it met it. These types and interfaces
make rules from `rewrite-rules.md` hold by construction; the laws named
there stay as checks that the construction works.

| Rule | Construction |
|---|---|
| R11 every input value reaches the archive | The framework hands a parser each record wrapped so that every field read is recorded. After `record()` returns, fields the parser did not read go to the message's `attrs` as given, unless the provider's declared ignore list names them. A field is dropped only by being named. |
| R12 parsing is total | The per-record interface (`sources/README.md`, the framework owns iteration): every record gets a `RecordOutcome`; ignoring needs a kind from the provider's closed enum. |
| R29 every parameter honoured or refused | Request types are generated from the verb and command tables with unknown fields refused at decoding. Handlers destructure the request without `..` (a lint over the handler module), so a new field that no handler uses fails to compile. |
| R34 one cost function | `Cost` has a private constructor; the only way to make one is `price(message, table)`, and totals are sums of `Cost` values. A provider's total is a separate `ProviderTotal` type that can be compared with a sum but not added to one. |
| R5 status figures | Every figure in the status result is `Figure::{Measured(v), Absent, Unmeasured(reason)}`; there is no plain number to leave at zero. |
| R38 provider databases read-only | Provider databases are opened only through `ProviderDb::open`, which sets the read-only flags and refuses `immutable` unless given the read-only-directory proof (`storage/blobs.md`); `rusqlite::Connection::open` is a disallowed method outside the store crate. |
| R45 no `IN` lists from data | SQL text is only `&'static str` fragments (`model::types::SqlText`, whose one constructor is `const`); a `model::types::Sql` is a list of such fragments plus bound parameters, and sets go through `json_each`. The query compiler emits `Sql` from fixed fragments. |
| R56 HTML escaped | Pages are rendered with a template engine that escapes by default; raw HTML needs a named wrapper type, used only for Polylogue's own markup. |
| R66 calendar windows | Day, week and month arithmetic is the shared logical-day module (joint #15); calendar methods of the date library are disallowed elsewhere (`clippy.toml` `disallowed-methods`). |
| R69 URL encoding once | An ID enters a URL only through `UrlId`, which percent-encodes once; route parameters decode into the ID type once. |

## Sets of IDs in SQL

Code passes a set of IDs to SQL through `json_each(?)` or a temporary table,
never as an `IN (...)` list built from data, which breaks above SQLite's
limit on parameters (`plan/testing/laws.md`, law 35).

Sources: `polylogue-1tyq5.187`.

## Crates shared with Sinex

Agreed in the joint phase (`decisions/sinex.md` 10; joint #9, #11):

- `hashed-blobs` and `polylogue-model` live here and are used by Sinex;
  `sinex-protocol` lives in the Sinex repository and is used only by
  `polylogue-sinex`. Each is pinned by a semver git tag in the other
  project; no monorepo, no third repository, no crates.io.
- Shared crates build on stable Rust and never depend on tokio, rusqlite,
  sqlx, async-nats or pest; `deny.toml` enforces this for `hashed-blobs` and
  `polylogue-model`.
- Each shared crate ships examples that the other project runs as a
  conformance test.
