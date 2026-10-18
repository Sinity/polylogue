# Repository Guidelines

This is the Rust rewrite of Polylogue. The README and CLI help describe what
works today. The design lives in the spec repository
(`/realm/inbox/spec/polylogue-spec/`); `docs/rewrite/` holds notes on parts
already built. The Python Polylogue is still the deployed version until the
operator switches over.

## Structure

- `crates/model`: pure identity, value and outcome types; no IO, no storage.
- `crates/hashed-blobs`: the content-addressed file store shared with Sinex.
- `crates/store`: every SQLite connection and write. Nothing else touches SQLite.
- `crates/engine`: parses provider files outside the writer and hands the
  results to the store.
- `crates/polylogue`: configuration, the daemon and the CLI. Clients get typed
  store handles, never raw connections.
- `xtask`: developer checks (`fast`, `check`, `boundaries`).

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
- Tests go in separate files (`src/tests.rs` or `tests/`) using `#[cfg(test)]`
  modules when they need private access; small tests may stay inline. Use
  synthetic inputs and temporary directories. CLI and MCP tests go through the
  public commands.
- Unsupported operations return a typed error, never an empty success.
  Unknown times stay unknown. IDs are opaque strings and may contain colons.
  Check the schema version before writing.

## Data

The repository is public. Keep credentials, real captures, private reports
and real databases out of Git; fixtures are synthetic. Reading the operator's
real files on this machine for development and measurement is fine.
