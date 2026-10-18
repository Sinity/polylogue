# Development checks

Checks run on the development machine; there is no hosted CI. Run commands
inside `nix develop`.

| Check | Command | When | Typical time |
| --- | --- | --- | --- |
| Focused tests | `cargo test -p <crate> <filter>` | while editing | seconds |
| Fast check | `cargo run -p xtask -- fast` | every push (pre-push hook) | 2-10 s warm, 40 s cold |
| Full suite | `cargo run -p xtask -- check` | a batch of merged work | 1.5 min warm, 9 min cold |
| End-to-end scenarios | `cargo run -p xtask -- scenarios` | after daemon, CLI or MCP changes | about a minute warm |
| Package build | `nix flake check` | after dependency or packaging changes | about 2.5 min |

- **Fast check:** formatting, crate boundaries (`xtask boundaries`) and
  `cargo check` of every target, tests included.
- **Full suite:** formatting, boundaries and `cargo nextest run --workspace`
  (configuration in `.config/nextest.toml`). Each test runs in its own
  process, all binaries in parallel; a test still running after two minutes
  is terminated and fails.
- **End-to-end scenarios:** the integration tests in `crates/polylogue/tests`
  start the real `polylogue` binary as a daemon over a temporary archive and
  drive it through the CLI and the MCP server (`tests/session_read/mcp_client.rs`).
  They also run as part of the full suite; `xtask scenarios` runs only them.
- **Package build:** builds the `polylogue` binary from a clean, locked
  source. It does not run tests; the full suite covers them with incremental
  builds.

`xtask check` and `xtask scenarios` set `TMPDIR` to a fresh directory under
`/dev/shm` and remove it afterwards. Tests write fsync'd archives, and on the
btrfs work disk those syncs dominated test time. The daemon `World` harness in
`crates/polylogue/tests/support` puts its archive on tmpfs even when run
directly. Two tests deliberately keep a real filesystem, the one named by
`TEST_DURABLE_TMPDIR` (`target/tmp`): the hashed-blobs batch durability test
and the store killed-writer WAL reopen test.

The pre-push hook is `.githooks/pre-push`; enable it with
`git config core.hooksPath .githooks`. It skips pushes that change no Rust
build inputs.

## AgentCTL operations

Long runs go through AgentCTL so they queue and keep their logs:

```sh
agentctl job start polylogue check --workspace <worktree>      # full suite
agentctl job start polylogue scenarios --workspace <worktree>  # end-to-end tests
agentctl job start polylogue package --workspace <worktree>    # nix flake check
agentctl job start polylogue exec --workspace <worktree> -- cargo test -p polylogue-store
```

## Merging

A PR merges as soon as it is ready; nothing waits on a hosted check. Run the
full suite on batches of merged work and fix what it finds in follow-up PRs.
The integration tests use fixed deadlines (40 s per CLI call, 60 s for daemon
startup), so a heavily loaded machine can fail them spuriously; rerun before
treating such a failure as a regression.
