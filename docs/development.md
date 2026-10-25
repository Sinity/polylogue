# Development checks

Checks run on the development machine; there is no hosted CI. Every build and
test runs as an AgentCTL job, which queues it with the rest of the machine's
work and keeps its log.

| Check | Command | When |
| --- | --- | --- |
| Compile | `agentctl job start polylogue compile --workspace <worktree> -- -p <crate>` | while editing |
| Focused tests | `agentctl job start polylogue focused --workspace <worktree> -- -p <crate> <filter>` | while editing |
| Fast check | `agentctl job start polylogue fast --workspace <worktree>` | before pushing |
| Full suite | `agentctl job start polylogue check --workspace <worktree>` | before merging a batch |
| End-to-end scenarios | `agentctl job start polylogue scenarios --workspace <worktree>` | after daemon, CLI or MCP changes |
| Package build | `agentctl job start polylogue package --workspace <worktree>` | after dependency or packaging changes |
| Extension | `agentctl job start polylogue extension --workspace <worktree> -- ci`, then `-- test` | after extension changes |

- **Fast check:** formatting, crate boundaries (`xtask boundaries`) and
  Clippy over every target, tests included, with warnings denied. It enables
  `polylogue/sinex`, so the Sinex code compiles too.
- **Full suite:** the fast check, then `cargo nextest run --workspace
  --features polylogue/sinex,polylogue-engine/faults,polylogue-store/faults`
  (configuration in `.config/nextest.toml`). Each test runs in its own
  process; one still running after two minutes fails.
- **Fault tests:** the `faults` features add a fault registry that starts
  empty; only a test's own `polylogue_store::fault::arm` call activates a
  point, in that process. They stop work between an excision's database
  commits, during a restore and at other write steps. Shipped builds never
  enable them.
- **End-to-end scenarios:** the `e2e` tests in `crates/polylogue/tests/e2e/`
  start the real `polylogue` binary as a daemon over a temporary archive and
  drive it through the CLI, and as the browser extension's native messaging
  host. The full suite runs them too.
- **Package build:** `nix flake check`, the `polylogue` binary from a clean,
  locked source, without tests.

Test jobs get their own tmpfs scratch directory, removed when the job ends.
Tests that check durability use a real filesystem. The pre-push hook
(`.githooks/pre-push`, enable with `git config core.hooksPath .githooks`)
checks formatting and crate boundaries, and skips pushes that change no Rust
build inputs.

## Tests

Tests are laid out as AGENTS.md says: unit tests in `name/tests.rs` beside
their module, integration tests as one binary per area
(`crates/<crate>/tests/<area>/main.rs`). The engine's are `convergence` (the
loop over files a test writes), `laws` (properties every archive keeps, as
property tests over generated archives and the example archive) and `faults`.

Tests that need realistic data use `polylogue-testkit`:
`Morrow::build(Set::Main)` lays out the Morrow example archive
(`examples/morrow/`, facts in its `FACTS.md`) where each tool writes, runs the
loop over it and imports its exports. `view()` and `stored_bytes()` give what
laws compare. Query examples run against it in
`crates/polylogue/src/daemon/query/tests/spec_examples.rs`.

Tests do not assert on elapsed time. A test that checks a deadline makes the
peer stay silent until the client gives up, so only the deadline can end the
wait; other tests wait for an event (a reply, a file, a closed connection).
nextest's slow timeout ends a hung test; under plain `cargo test` it would
wait forever.

The Sinex rebuild test in `crates/sinex/tests/` runs a real `sinexd`, so it
is ignored by default. Build `sinexd` and `sinex-schema` in a Sinex checkout,
then run it from Sinex's dev shell, which has PostgreSQL with TimescaleDB and
`nats-server`:
`SINEXD=<sinex>/target/debug/sinexd SINEX_SCHEMA=<sinex>/target/debug/sinex-schema
nix develop <sinex> --command cargo nextest run -p polylogue-sinex --run-ignored only -E 'test(an_archive_restored_from_sinex_gives_the_same_answers)'`.

## Build speed

The dev profile builds dependencies at `opt-level = 2` and workspace crates
at `opt-level = 1`, because tests spend most of their CPU in SQLite, JSON and
the credential masker. Keep the package and feature set the same across a
connected edit and narrow tests with `-E`: a different package set can change
Cargo's unified features and rebuild libraries. For store and engine work,
`focused -- -p polylogue-store -p polylogue-engine --features faults -E '<filter>'`
reuses one build.

AgentCTL takes `CARGO_TARGET_DIR` from the caller when it queues a job. Give
each target directory one building job at a time.

When the suite slows down, read the recorded stage times to separate
compilation, setup and test execution, then the slowest tests and their CPU,
memory and I/O waits. Keep assertions unchanged while diagnosing, and record
the measurements in the owning Bead.

## xtask

`xtask` is built on `devkit` from
[polylogue-sinex-common](https://github.com/Sinity/polylogue-sinex-common).
Every run is recorded in `$XDG_STATE_HOME/devkit/history.db`: its stages and
times, each test's outcome with output for failures, and compiler
diagnostics. `cargo run -p xtask -- status` shows what failed in this
checkout's last run; `history failures [RUN]`, `history diagnostics [RUN]`
and `history runs` show more, each with `--json`. `test --changed` runs the
tests of packages changed since the merge base with `origin/master` and of
everything that depends on them.

## Benchmarks

`agentctl job start polylogue bench --workspace <worktree> -- run` measures a
first build and incremental updates; `-- reads --repetitions 3` measures
interactive reads. `agentctl job start polylogue measure --workspace
<worktree> -- <command>` runs another performance driver. Both use the normal
queue, which has no I/O bandwidth cap; record the pool and resource limits
with the timings.

A first-build report gives wall time, time per stage (pool time in
thread-seconds, which do not add up to wall time), input bytes, physical I/O,
peak memory and final storage sizes. A reading the runner could not take stays
unknown. The runner refuses a run with ingestion errors, unreadable roots or
no message writes. Load is sampled throughout; reports show its peaks.

The read benchmark generates one 16 MiB session and eight small ones with the
testkit's Claude Code builders. It checks that a complete read holds every
message once in order, that two-page windows at the first, middle and last
positions match it, and that repeated windows render the same bytes. It
reports daemon start, the first request after each restart and later
requests (samples, median, p95), page counts and daemon memory. `--full-walk`
also reads every page of the large session. It has no elapsed-time
assertions.

The full-corpus estimate assumes time is linear in input bytes.
`bench validate` compares that estimate with a full run's result.

## Row dictionary

`crates/store/src/encoding/rows.dict` compresses index rows. It is generated
only from the synthetic examples in
`crates/store/examples/row_dictionary/samples.rs`. Check that it reproduces:

```sh
agentctl job start polylogue exec --workspace <worktree> -- nix develop --command cargo run --locked --quiet -p polylogue-store --example row_dictionary -- --check crates/store/src/encoding/rows.dict
```

To regenerate, omit `--check`, then raise `INDEX_VERSION` and update the
format digests in `schema/tests.rs`. Archive records never depend on the
dictionary.
