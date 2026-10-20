# Development checks

Checks run on the development machine; there is no hosted CI. Run commands
inside `nix develop`.

| Check | Command | When | Typical time |
| --- | --- | --- | --- |
| Focused tests | `agentctl job start polylogue exec --workspace <worktree> -- cargo nextest run -p <crate> <filter>` | while editing | seconds |
| Fast check | `agentctl job start polylogue exec --workspace <worktree> -- cargo run -p xtask -- fast` | before pushing a change | 2-16 s warm, minutes in a fresh worktree |
| Full suite | `agentctl job start polylogue check --workspace <worktree>` | a batch of merged work | 30 s warm, 2-3 min after an edit, 9 min cold |
| Changed packages | `cargo run -p xtask -- test --changed` | before pushing a focused change | depends on the change |
| End-to-end scenarios | `cargo run -p xtask -- scenarios` | after daemon, CLI or MCP changes | about a minute warm |
| Package build | `nix flake check` | after dependency or packaging changes | about 2.5 min |

- **Fast check:** formatting, crate boundaries (`xtask boundaries`) and
  clippy of every target, tests included, with warnings denied. The pre-push
  hook runs only its formatting and boundaries steps; clippy runs in the
  AgentCTL job so it queues with the rest of the machine's work.
- **Full suite:** formatting, boundaries, clippy and `cargo nextest run --workspace`
  (configuration in `.config/nextest.toml`). Each test runs in its own
  process, all binaries in parallel; a test still running after two minutes
  is terminated and fails. Then the laws that inject failures (laws 7, 45,
  63): `cargo nextest run -p polylogue-engine --features faults --test
  faults`. The `faults` feature arms the store's fault points
  (`polylogue_store::fault`); nothing that ships enables it.
- **End-to-end scenarios:** the `e2e` tests in `crates/polylogue/tests/e2e/`
  start the real `polylogue` binary as a daemon over a temporary archive and
  drive it through the CLI, and as the browser extension's native messaging
  host. They also run as part of the full suite; `xtask scenarios` runs only
  them.
- **Package build:** builds the `polylogue` binary from a clean, locked
  source. It does not run tests; the full suite covers them with incremental
  builds.

`xtask check` and `xtask scenarios` set `TMPDIR` to a fresh directory under
`/dev/shm` and remove it afterwards. Tests write fsync'd archives, and on the
btrfs work disk those syncs dominated test time.

The dev profile builds dependencies at `opt-level = 2` and workspace crates
at `opt-level = 1`. Tests spend most of their CPU in SQLite, JSON and the
credential masker. Measured on 926 tests at load average 25-70: suite CPU fell
from about 780 s to about 190 s and test execution from 63-76 s to 13-24 s.
A cold build costs more (9 min instead of 3.5 min); an incremental test build
after a one-line edit in `store` takes about 2 min under load.

`xtask` is built on `devkit` from
[polylogue-sinex-common](https://github.com/Sinity/polylogue-sinex-common).
Every `xtask` run is recorded in the machine-wide history store
(`$XDG_STATE_HOME/devkit/history.db`): its stages and times, each test's
outcome with output for failures, and compiler warnings and errors.
`cargo run -p xtask -- status` shows what failed in this checkout's last run;
`history failures [RUN]`, `history diagnostics [RUN]` and `history runs` read
more, each with `--json`. `test --changed` runs the tests of packages changed
since the merge base with `origin/master` and of everything that depends on
them.

Tests are laid out as AGENTS.md says: unit tests in `name/tests.rs` beside
their module, integration tests as one binary per area
(`crates/<crate>/tests/<area>/main.rs`). The engine's are `convergence` (the
loop over files a test writes), `laws` (properties every archive keeps, as
property tests over generated archives and over the example archive) and
`faults`.

Tests that need realistic data use `polylogue-testkit`: `Morrow::build(Set::Main)`
lays the Morrow example archive (`examples/morrow/`, facts in its
`FACTS.md`) out where each tool writes, converges the engine over it and
imports its exports. `view()` and `stored_bytes()` give what laws compare.
The spec's query examples run against it in
`crates/polylogue/src/daemon/query/tests/spec_examples.rs`; examples waiting on unbuilt
features are listed there with the reason.

The Sinex rebuild proof (`crates/sinex/tests/proof.rs`) runs a real `sinexd`,
so it is ignored by default. Build `sinexd` and `sinex-schema` in a Sinex
checkout, then run it from Sinex's dev shell, which has PostgreSQL with
TimescaleDB and `nats-server`:
`SINEXD=<sinex>/target/debug/sinexd SINEX_SCHEMA=<sinex>/target/debug/sinex-schema
nix develop <sinex> --command cargo nextest run -p polylogue-sinex --run-ignored all`.

The pre-push hook is `.githooks/pre-push`; enable it with
`git config core.hooksPath .githooks`. It checks formatting and crate
boundaries, and skips pushes that change no Rust build inputs.

## AgentCTL operations

Long runs go through AgentCTL so they queue and keep their logs:

```sh
agentctl job start polylogue check --workspace <worktree>      # full suite
agentctl job start polylogue scenarios --workspace <worktree>  # end-to-end tests
agentctl job start polylogue package --workspace <worktree>    # nix flake check
agentctl job start polylogue exec --workspace <worktree> -- cargo test -p polylogue-store
agentctl job start polylogue bench --workspace <worktree> -- run
```

The benchmark runs its builds and measurements inside that job. It does not
use a separate build-slot script or queue nested jobs. The real-data lock
still prevents overlapping corpus benchmarks.

## Keeping checks fast

Use the recorded stages to distinguish compilation, service setup and test
execution. A cold build is not a warm-suite timing. If the suite slows down,
inspect its slowest tests and CPU, memory and I/O waits; check for orphaned
services and repeated fixture setup before increasing limits. Keep assertions
and populations unchanged while diagnosing. Record the candidate, command,
stage times and remaining failures in the owning Bead. Recheck after changes
to the runner, fixtures or dependency profiles, and at integration batches.

## Merging

A PR merges as soon as it is ready; nothing waits on a hosted check. Run the
full suite on batches of merged work and fix what it finds in follow-up PRs.

## Timing in tests

Tests do not assert on elapsed time. A test that checks a deadline arranges
for the peer to stay silent or keep trickling until the client gives up, so
only the deadline can end the wait; other tests wait for an event (a reply, a
child's acknowledgement file, a closed connection) with no deadline of their
own. A hung test is ended by nextest's slow-timeout (`.config/nextest.toml`),
so run the suite through `xtask check` or `cargo nextest run`; under plain
`cargo test` a hung test waits forever.
