# Repository Guidelines

This is the Rust rewrite of Polylogue. The README and CLI help describe what
works today. The target architecture is
`/realm/project/polylogue/analysis/architecture-20261018/polylogue-architecture.md`;
planned work is in Beads (`bd` from this checkout). The Python predecessor
is kept for reference; the rewrite uses its own archive.

## Structure

- `crates/model`: identity, value and outcome types; no I/O.
- `crates/sources`: one declaration and one parser per origin on the shared
  framework (`origin.rs`, `parse.rs`, `document.rs`).
- `crates/store`: every SQLite connection and write; the blob store.
- `crates/engine`: the convergence loop and the parse pool.
- `crates/query`: the query language and its compiler to SQL.
- `crates/render`: one block stream in four formats.
- `crates/polylogue`: configuration, the daemon, CLI, MCP, web reader and
  browser host. Clients get typed store handles, never raw connections.
  Every command is one entry in the command table (`commands.rs`); the CLI,
  MCP tools and HTTP API are built from it, so a new operation is a table
  entry, not code on each surface.
- `crates/sinex`: the Sinex backend (`primary` mode), behind the `sinex`
  feature.
- `crates/testkit`: test archives built through the real loop (the synthetic
  example archive in `examples/morrow/`, and a Claude Code home).
- `xtask`: developer checks (`fast`, `check`, `boundaries`), on `devkit`.

## Development loop

Optimize for fast integration. See `docs/development.md`.

- Run focused tests for the change while editing.
- The pre-push hook runs the fast check, `cargo run -p xtask -- fast`
  (seconds). There is no hosted CI and no required check; open a PR and merge
  it when it is ready.
- Run the full suite (AgentCTL operation `check`) on batches of merged work,
  not before every merge. Fix what it finds afterwards.
- Merge small PRs often. Rebase before pushing; do not keep long-lived
  branches.
- Report what you ran and what you did not. A merge is not evidence of testing.

## Code

- Rust, four-space indent, `rustfmt`.
- Comments state constraints and reasons, briefly. No status, history or spec
  narration in code; Git and Beads hold those.
- A module is `name.rs`, its child modules in `name/`; no `mod.rs`.
- Tests live in their own files, never inline: a module's unit tests in
  `name/tests.rs`, declared last with `#[cfg(test)] mod tests;` (more test
  files go in `name/tests/`). Integration tests are one binary per area,
  `crates/<crate>/tests/<area>/main.rs` with its other files beside it. Use
  synthetic inputs, temporary directories and `polylogue-testkit`; helpers
  that need a crate's private items stay in that crate behind its test
  feature (`polylogue-store`'s `fixture`). CLI and MCP tests go through the
  public commands.
- Unsupported operations return a typed error, never an empty success.
  Unknown times stay unknown. IDs are opaque strings and may contain colons.
  Check the schema version before writing.

## Data

The repository is public. Keep credentials, real captures, private reports
and real databases out of Git; fixtures are synthetic. Reading the operator's
real files on this machine for development and measurement is fine.
