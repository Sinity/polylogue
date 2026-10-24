# Repository Guidelines

This is the Rust rewrite of Polylogue. The README and CLI help describe what
works today. The target architecture is
`/realm/project/polylogue/analysis/architecture-20261018/polylogue-architecture.md`;
planned work is in Beads (`bd` from this checkout). The Python predecessor
is kept for reference; the rewrite uses its own archive.

## Structure

[docs/architecture.md](docs/architecture.md) describes current module boundaries.

- `crates/model`: identity, value and outcome types; no I/O.
- `crates/sources`: one declaration and one parser per origin on the shared
  framework (`origin.rs`, `parse.rs`, `document.rs`).
- `crates/store`: every SQLite connection and write; the blob store.
- `crates/engine`: the convergence loop and the parse pool.
- `crates/query`: the query language and its compiler to SQL.
- `crates/render`: one block stream in four formats.
- `crates/polylogue`: application operations (`daemon/service.rs`), socket
  framing (`daemon/server.rs`), CLI, MCP, web reader and browser host.
  Clients get typed store handles, never raw connections.
  Every command is one entry in the command table (`commands.rs`); the CLI,
  MCP tools and HTTP API are built from it, so a new operation is a table
  entry, not code on each surface.
- `crates/sinex`: primary mode on Sinex (row batches sent to Sinex, restore
  from it), behind the `sinex` feature.
- `crates/testkit`: test archives built through the real loop (the synthetic
  example archive in `examples/morrow/`, and a Claude Code home).
- `xtask`: developer checks (`fast`, `check`, `boundaries`), on `devkit`.

## Development loop

Optimize for fast integration. See `docs/development.md`.

- Every cargo build or test runs as an AgentCTL job, which queues it against
  the rest of the machine's work: `agentctl job start polylogue check
  --workspace <worktree>`, or `focused -- -p <crate> <filter>` for focused tests.
- The pre-push hook checks formatting and crate boundaries only (seconds).
  There is no hosted test CI. Open a PR when ready and follow the forge's
  current review and status requirements.
- Run the full suite (AgentCTL operation `check`) on batches of merged work,
  not before every merge. Fix what it finds afterwards.
- Give each writable Cargo target directory one active build owner. Coordinate
  sequential warm-cache reuse; concurrent worktrees use separate targets.
- Merge small PRs often. Rebase before pushing; do not keep long-lived
  branches.
- Report what you ran and what you did not. A merge is not evidence of testing.
- Keep the loop fast: compare build, setup and test times separately. When a
  run slows down, inspect its slow tests, resource waits and cleanup before
  increasing timeouts or reducing coverage. Record measurements in Beads.

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
