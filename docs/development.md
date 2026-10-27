# Development checks

Checks run on the development machine; there is no hosted CI. Every build and
test runs as an AgentCTL job, which queues it with the rest of the machine's
work and keeps its log.

| Check | Command | When |
| --- | --- | --- |
| Compile | `agentctl job start polylogue compile --workspace <worktree> -- -p <crate>` | while editing |
| Formatting | `agentctl job start polylogue fmt --workspace <worktree>` | before committing |
| Focused tests | `agentctl job start polylogue focused --workspace <worktree> -- -p <crate> <filter>` | while editing |
| Fast check | `agentctl job start polylogue fast --workspace <worktree>` | before pushing |
| Full suite | `agentctl job start polylogue check --workspace <worktree>` | on a PR's rebased head, before merging |
| End-to-end scenarios | `agentctl job start polylogue scenarios --workspace <worktree>` | after daemon, CLI or MCP changes |
| Package build | `agentctl job start polylogue package --workspace <worktree>` | after dependency or packaging changes |
| Extension | `agentctl job start polylogue extension --workspace <worktree> -- ci`, then `-- test` | after extension changes |
| Lockfile | `agentctl job start polylogue lock --workspace <worktree> -- -p <crate>` | when updating a dependency |

Start a job, then wait for it in the background with `agentctl job wait <id>`.
A failed wait prints the failing tests, the compiler errors and the `xtask
history failures N` pointer. The host runs the `baseline` operation (the full
suite) on each new master head in the reserved checkout
`/realm/worktree/polylogue-master`; read the newest `polylogue:baseline` job
(`agentctl job list --project polylogue`) to see whether a failure is already
on master.

- **Fast check:** formatting, crate boundaries (`xtask boundaries`) and
  Clippy over every target, tests included, with warnings denied: once with
  `polylogue/sinex`, so the Sinex code compiles too, and once over the
  `polylogue` package without it.
- **Full suite:** the fast check, then `cargo nextest run --workspace
  --features polylogue/sinex,polylogue-engine/faults,polylogue-store/faults`
  (configuration in `.config/nextest.toml`). Each test runs in its own
  process; one still running after two minutes fails. `focused` and
  `scenarios` build exactly the same: `xtask test` always selects the whole
  workspace with these features and turns `-p` and `-E` into one filterset
  that chooses which tests run. It refuses feature flags.
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
(`crates/<crate>/tests/<area>/main.rs`). The engine's are `passes` (the
loop over files a test writes), `properties` (properties every archive keeps, as
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

A test that waits for the daemon to read a change waits until the change
shows in `status` and `busy` is false. Readers see a pass's rows before the
pass has derived titles and other products from them, so a count alone does
not mark the end of a pass. `passes` counts finished passes, for a test that
needs the next one.

The Sinex rebuild test in `crates/sinex/tests/` runs a real `sinexd`, so it
is ignored by default. It runs from a checkout of the Sinex revision this
workspace pins, in Sinex's dev shell, which has PostgreSQL with TimescaleDB
and `nats-server`. Sinex's queries are checked against its schema database
at build time, so build its binaries under `xtask db exec`:

```
agentctl job start sinex exec --workspace <sinex> --wait -- \
  cargo run --locked -q -p xtask -- db exec -- cargo build --locked -p sinexd -p sinex-schema
agentctl job start sinex exec --workspace <sinex> --wait -- \
  CARGO_TARGET_DIR=<polylogue>/target/sinex-shell \
  SINEXD=<sinex target>/debug/sinexd SINEX_SCHEMA=<sinex target>/debug/sinex-schema \
  cargo nextest run --locked --manifest-path <polylogue>/Cargo.toml -p polylogue-sinex \
  --run-ignored only -E 'test(the_morrow_example_rebuilt_from_sinex_gives_the_same_answers)'
```

`<sinex target>` is the dev shell's `CARGO_TARGET_DIR`.

## Build speed

The dev profile builds dependencies at `opt-level = 2` and workspace crates
at `opt-level = 1`, because tests spend most of their CPU in SQLite, JSON and
the credential masker. Every test run builds the whole workspace with one
feature set, so `focused`, `scenarios` and `check` reuse each other's
artifacts whatever packages or filters they select. The first test run in a
checkout builds every test binary; a new checkout's target starts as a
reflink copy of the master baseline's warm target, so that build compiles
only what differs from master.

The dev shell links with mold (`CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS`):
on a cold test build the workspace's test binaries spent 169 s linking with
the default linker and 37 s with mold. The Nix package build keeps the default.

Every operation that runs Cargo holds a place under the host's `cargo` cap:
at most three Cargo builds run at once across the normal and bulk queues,
each with an 8 GiB `MemoryHigh`. A job that waits for a place logs it, and
the wait does not count against its timeout. `check` stops after 20 minutes.

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
<worktree> -- <command>` runs another performance driver. Both use the
one-slot measure queue, so they never hold a slot a check needs. They run
under whatever load the host has: a run records its load, and a change is
judged by a paired run (`bench pair`), both builds at once over one sample.
Record the pool and resource limits with the timings.

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

`POLYLOGUE_BENCH_HOME` selects where sample copies and output archives go;
`POLYLOGUE_BENCH_MANIFEST_DIR` selects the manifest directory. Defaults are
`/realm/project/polylogue/bench` and `/realm/project/polylogue/analysis/bench`.
Use private directories for operator data.

A single sample's estimate assumes time is linear in input bytes. Samples
of two or more sizes (`bench sample --name NAME --mb MB`, then `bench run
--name NAME`) let `bench estimate` fit each stage as fixed seconds plus
seconds per input MB and predict the whole corpus stage by stage; `bench
validate RESULT` compares that prediction with a full run, stage by stage.
`--binary PATH --build LABEL` runs a `first_build` built elsewhere, such as
the commit a full run used. `bench tails --name NAME` times the last pages of
the largest sessions in a sample run's archive, and of any `--session`.

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
