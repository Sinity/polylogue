# Repository Guidelines

This is the Rust rewrite of Polylogue. The README and CLI help describe what
works today. The target architecture is
`/realm/project/polylogue/analysis/architecture-20261018/polylogue-architecture.md`;
planned work is in Beads (`bd` from this checkout). The Python version
is kept for reference; the rewrite uses its own archive.

## Structure

[docs/architecture.md](docs/architecture.md) describes current module boundaries.

- `crates/model`: identity, value and outcome types; no I/O.
- `crates/sources`: one declaration and one parser per origin on the shared
  framework (`origin.rs`, `parse.rs`, `document.rs`).
- `crates/store`: every SQLite connection and write; the blob store.
- `crates/engine`: the loop and the parse pool.
- `crates/query`: the query language and its compiler to SQL.
- `crates/render`: one block stream in four formats.
- `crates/polylogue`: application operations (`daemon/service.rs`), socket
  framing (`daemon/server.rs`), CLI, MCP, web reader and browser host.
  Clients get typed store handles, never raw connections.
  Every command is one entry in the command table (`commands.rs`); the CLI,
  MCP tools and HTTP API are built from it, so a new operation is a table
  entry, not code on each surface.
- `crates/sinex`: primary mode on Sinex (inputs, records and parsed sessions
  sent to Sinex; the local copy rebuilt from it), behind the `sinex` feature.
- `crates/testkit`: test archives built through the real loop (the synthetic
  example archive in `examples/morrow/`, and a Claude Code home).
- `xtask`: developer checks (`fast`, `check`, `boundaries`), on `devkit`.

## Development loop

Optimize for fast integration. See `docs/development.md`.

- Every cargo build or test runs as an AgentCTL job, which queues it against
  the rest of the machine's work: `agentctl job start polylogue <operation>
  --workspace <worktree>`. `compile -- -p <crate>` checks that code builds,
  `focused -- -p <crate> [-E <filterset>] [<name>]` runs selected tests,
  `fmt` formats, and `check` runs formatting, boundaries, clippy and every
  test. All test runs share one build of the whole workspace, so `focused`
  and `check` reuse each other's artifacts.
- Start a job, then wait for it in the background with `agentctl job wait
  <id>` and keep working. A failed wait prints the failing tests and the
  `xtask history` pointer.
- Before treating a failure as yours, read the newest `polylogue:baseline`
  job (`agentctl job list --project polylogue`): the host runs `check` on
  each new master head.
- The pre-push hook checks formatting and crate boundaries only (seconds).
  There is no hosted test CI.
- Land a few large, coherent PRs. Rebase onto `origin/master` before
  pushing; `check` must pass on the rebased head before merging. Merge with
  `gh pr merge --auto --squash --match-head-commit <sha>`; auto-merge waits
  for the `stacked-review-threads` status.
- Report what you ran and what you did not. A merge is not evidence of testing.
- Keep the loop fast: compare build, setup and test times separately. When a
  run slows down, inspect its slow tests, resource waits and cleanup before
  increasing timeouts or reducing coverage. Record measurements in Beads.

## Code

- Rust, four-space indent, `rustfmt`.
- A name says what the thing is in ordinary words: types, functions, local
  variables, SQL, JSON keys, CLI help, errors, comments, tests and docs.
  [docs/vocabulary.md](docs/vocabulary.md) lists each concept's one name and
  the words not used. Before naming something, look for it there: use the
  listed word, or add the new one in the same PR.
- Comments state constraints and reasons, briefly. No status, history, Bead
  IDs or spec section numbers in code; Git and Beads hold those.
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
- A change to the index schema, the row encoding or what a parser produces
  from the same bytes raises `INDEX_VERSION` in `crates/store/src/schema.rs`;
  the index is then rebuilt from the stored inputs.
- `archive.db` is deployed data. A change to `archive.sql` appends a step to
  `ARCHIVE_MIGRATIONS` in the same file, with a test that an archive filled
  at the version before keeps every row (`archive_at` in the schema tests).

## Data

The repository is public. Keep credentials, real captures, private reports
and real databases out of Git; fixtures are synthetic. Reading the operator's
real files on this machine for development and measurement is fine.

## Cloud sessions and changes to polylogue-sinex-common

Cloud sessions (Claude Code on the web, Codex cloud) run on a fresh
VM without AgentCTL, `/realm` or the operator's services; run `cargo xtask`
commands there directly. The environment's setup script runs
`bash dev/cloud/setup.sh`, after which every shell is in the development
shell.

- A Polylogue session must include the polylogue-sinex-common and sinex
  repositories: Cargo fetches Git dependencies from both, and a session
  reaches only the repositories it includes.
- `cargo xtask fast`, `cargo xtask check` and `cargo xtask test [-p <crate>]
  [-E <filterset>]` work as on the host. The first workspace build takes
  minutes; run it in the background. `xtask bench` and anything that reads
  the operator's data do not run there; report them as not run.
- Push a branch and open a pull request; do not merge.

A change that needs polylogue-sinex-common changed too starts there.
`cargo xtask link-common [<path>]` builds this workspace against a checkout
of it (the path, else `$POLYLOGUE_SINEX_COMMON`, else
`../polylogue-sinex-common`), and entering the development shell does the
same when that checkout exists. It writes the git-ignored
`.cargo/local.toml`, and Cargo.lock then names the local crates, so the lock
is not committed in that state. `cargo xtask unlink-common` restores the lock and
keeps the shell from linking again. Merge the common pull request first;
its workflow then opens a pull request here that moves Cargo.lock to the new
commit. Convert the callers on that branch, or run `cargo update -p devkit
-p hashed-blobs -p credential-masking` on the branch that converts them, so
the lock change and the callers land together.
