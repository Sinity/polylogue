# Development checks

Checks run on the development machine; there is no hosted CI. Run commands
inside `nix develop`.

The shell uses Cargo's incremental builds directly. A persistent compiler
cache daemon must have its own lifecycle owner before it is enabled: a daemon
started inside a queued job keeps that job occupied after the command exits.

| Check | Command | When | Typical time |
| --- | --- | --- | --- |
| Focused tests | `agentctl job start polylogue focused --workspace <worktree> -- -p <crate> <filter>` | while editing | seconds |
| Fast check | `agentctl job start polylogue fast --workspace <worktree>` | before pushing a change | 2-16 s warm, minutes in a fresh worktree |
| Full suite | `agentctl job start polylogue check --workspace <worktree>` | a batch of merged work | 30 s warm, 2-3 min after an edit, 9 min cold |
| Changed packages | `cargo run -p xtask -- test --changed` | before pushing a focused change | depends on the change |
| End-to-end scenarios | `cargo run -p xtask -- scenarios` | after daemon, CLI or MCP changes | about a minute warm |
| Package build | `nix flake check` | after dependency or packaging changes | about 2.5 min |

- **Fast check:** formatting, crate boundaries (`xtask boundaries`) and
  clippy of every target, tests included, with warnings denied. It enables
  `polylogue/sinex` so the optional daemon publisher is compiled too. The pre-push
  hook runs only its formatting and boundaries steps; clippy runs in the
  AgentCTL job so it queues with the rest of the machine's work.
- **Full suite:** formatting, boundaries, clippy and
  `cargo nextest run --workspace --features
  polylogue/sinex,polylogue-engine/faults,polylogue-store/faults`
  (configuration in `.config/nextest.toml`). Ordinary tests and every gated fault
  control share one test build. Each test runs in its own process, all binaries
  in parallel; a test still running after two minutes is terminated and fails.
  The fault controls cover failure and recovery, including restore interruption
  after canonical commit, vector cleanup and index replacement. Enabling
  `faults` adds an initially empty in-memory registry; only a test's explicit
  `polylogue_store::fault::arm` call activates a point in that process. The
  preceding strict Clippy step still compiles every ordinary target without
  fault features. Shipped builds do not enable fault injection.
- **End-to-end scenarios:** the `e2e` tests in `crates/polylogue/tests/e2e/`
  start the real `polylogue` binary as a daemon over a temporary archive and
  drive it through the CLI, and as the browser extension's native messaging
  host. They also run as part of the full suite; `xtask scenarios` runs only
  them.
- **Package build:** builds the `polylogue` binary from a clean, locked
  source. It does not run tests; the full suite covers them with incremental
  builds.

Test operations declare AgentCTL-owned tmpfs scratch. Devkit creates its test
child inside that root; normal Drop removes the child, and the native owner
removes retained files after cancellation or SIGKILL. Long Linux paths use a
short held-directory `/proc` view, so Unix sockets fit without putting files
outside the owner. Standalone runs fall back to a temporary child with Drop
cleanup only. Durability tests retain their separate real-filesystem root.
Tests write fsync'd archives; on the btrfs work disk those syncs dominated time.

Run throughput measurements with the declared `bench` operation, or
`agentctl job start polylogue measure --workspace <worktree> -- <command>`
for a performance driver. Both use the normal queue. The background bulk
pool has a hard I/O bandwidth cap that can dominate disk-backed measurements;
normal retains queue admission, memory controls and lower I/O priority without
that cap. Record the actual pool and resource limits with timing results.
Keep background package builds and integrated checks in their existing pool.

Full-text indexing decodes one stored body at a time. First-build indexing
streams marked rows; live indexing retains up to 2,000 pending IDs, then
fetches their bodies individually. Transactions stop after 2,000 entries or
16 MiB of decoded text, whichever comes first. One large text may exceed that
byte limit; it is still decoded whole. These limits bound batches, not total
process memory or the time needed to drain all pending work.

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

The Sinex rebuild proof (`crates/sinex/src/tests.rs`) runs a real `sinexd`,
so it is ignored by default. Build `sinexd` and `sinex-schema` in a Sinex
checkout, then run it from Sinex's dev shell, which has PostgreSQL with
TimescaleDB and `nats-server`:
`SINEXD=<sinex>/target/debug/sinexd SINEX_SCHEMA=<sinex>/target/debug/sinex-schema
nix develop <sinex> --command cargo nextest run -p polylogue-sinex --run-ignored only -E 'test(excision_restores_current_units_after_retired_payloads_and_unknown_ack)'`.

The ignored `cli_publication_and_plain_restore_cross_distinct_uids` control also
needs `POLYLOGUE_BIN` (built with `sinex`) and a fresh
`POLYLOGUE_CROSS_UID_PROOF_DIR`. It starts owned PostgreSQL/NATS fixtures, then
uses sudo and a private mount namespace for numeric UID actors. The command
driver reuses existing executables; same-inode pins keep them stable without
copying binaries. It exercises CLI import/note publication and exact plain
restore with the provider absent. It does not activate installed services or
create accounts. Use a declared operation and preserve its binary hashes.

The pre-push hook is `.githooks/pre-push`; enable it with
`git config core.hooksPath .githooks`. It checks formatting and crate
boundaries, and skips pushes that change no Rust build inputs.

Keep source-only worktrees free of build outputs. After qualifying a candidate
in a warm checkout, keep ownership of that checkout through the push and run
`git push origin HEAD:refs/heads/<branch>` there, with `HEAD` at the exact
qualified commit. This lets the preserved hook reuse its compiled tools and
check the candidate it publishes. Release the checkout after the hook finishes;
waiting for hosted review does not need the build slot.

## AgentCTL operations

AgentCTL snapshots an explicit `CARGO_TARGET_DIR` from the caller when queuing a job. Select an absolute target with exclusive writer ownership; omitting it keeps the existing Cargo default.

```sh
CARGO_TARGET_DIR=/realm/worktree/leased-build/target \
  agentctl job start polylogue focused --workspace /realm/worktree/polylogue-edit -- -p polylogue-store
```

Long runs go through AgentCTL so they queue and keep their logs:

```sh
agentctl job start polylogue check --workspace <worktree>      # full suite
agentctl job start polylogue scenarios --workspace <worktree>  # end-to-end tests
agentctl job start polylogue package --workspace <worktree>    # nix flake check
agentctl job start polylogue focused --workspace <worktree> -- -p polylogue-store
agentctl job start polylogue bench --workspace <worktree> -- run
```

The benchmark runs its builds and measurements inside that job. It does not
use a separate build-slot script or queue nested jobs. The real-data lock
still prevents overlapping corpus benchmarks.

Benchmark reports retain wall time, per-stage time (including pool thread-seconds),
input bytes, child-reported physical I/O and final storage sizes separately.
These counters do not measure the same work, and thread-seconds cannot be added
to obtain wall time. Load is sampled throughout the outer command, including
lock waits; reports show peaks, unknown readings and the largest sampling gap.
Passing load checks screens measured contention, not timing reliability.
The runner refuses ingestion errors, unreadable roots, missing required counters and an initial
build with no message writes. Incremental measurements can report zero new
messages. Missing child I/O or peak-memory readings remain unknown; a requested
memory limit requires an available peak measurement. First-build wall time
includes setup and writer shutdown, with setup and ingestion reported
separately. I/O deltas exclude corpus generation; peak memory covers the whole
child process. Logical bytes read and newly stored are reported separately from
physical I/O. Older records used a narrower ingestion interval.

`agentctl job start polylogue bench -- reads --repetitions 3` records native
interactive measurements through the existing CLI end-to-end test owner.
It generates one 16 MiB session and eight small sessions with the testkit's
Claude Code builders, then checks exact session/search answers and consumes
all bounded read pages. Every source message body must appear once in order;
repeated walks must retain the same rendered digest. The scenario is ignored
by routine suites and has no elapsed-time pass/fail assertions.

Build, corpus generation, archive ingestion/search readiness and daemon starts
are recorded separately from request latency. The first request after each
daemon restart and subsequent requests are reported with samples, median and
p95; filesystem caches are not flushed, and each CLI call starts a new client
process. Read results include page counts, source coverage and daemon current
and peak RSS before and after the repeated work. A native client is killed
after output begins, then verified status and read requests check recovery.
This checks abandoned client output, not cancellation of work already done by
the server. The normal benchmark load monitor and devkit history retain the
result; synthetic input, archive and child processes use the existing test
lifecycle. These measurements do not qualify the real archive's speed targets.

The full-corpus estimate explicitly assumes linear time per input byte and is
uncalibrated. Missing times or scale leave it unknown. `bench validate` compares
that estimate with a supplied full-run result; matching input/candidate identity,
full-run load and multi-size calibration remain unverified by that command.

## Keeping checks fast

Use the recorded stages to distinguish compilation, service setup and test
execution. A cold build is not a warm-suite timing. If the suite slows down,
inspect its slowest tests and CPU, memory and I/O waits; check for orphaned
services and repeated fixture setup before increasing limits. Keep assertions
and populations unchanged while diagnosing. Record the candidate, command,
stage times and remaining failures in the owning Bead. Recheck after changes
to the runner, fixtures or dependency profiles, and at integration batches.

Keep the package and feature set stable during a connected edit, and narrow
execution with `-E`. Changing the selected packages can change Cargo's unified
dependency features and force another library build even when the library's own
features appear unchanged. For connected Store/Engine work, reuse
`focused -- -p polylogue-store -p polylogue-engine --features faults -E '<filter>'`
instead of alternating package sets for each filter. Inspect recorded build
time before attributing a slow focused run to its tests.

Use `xtask test` for focused tests too: it provides the same temporary storage
and records results as the full suite. Raw `cargo test` or `cargo nextest`
inherits the caller's temporary directory; filesystem sync costs and long
socket paths can then change the test behavior.

The `fast` operation uses the normal development queue for formatting,
boundaries and strict Clippy. The `focused` operation requires a selection
and uses the same queue. Full checks use the bulk
queue, so they do not consume every slot for short development checks. A new target or dependency
profile may still need compilation; the stage times show that separately.

## Merging

Follow the forge's required review statuses when merging. There is no hosted
test CI. Run the full suite on batches of merged work and fix what it finds
in follow-up PRs.

The writer reuses prepared SQL for message-version and source writes. Its
work test counts SQLite preparation callbacks while checking exact message and
tool-result counts; it has no elapsed-time assertion. Writer connections keep
the library's existing statement-cache limit. A first build releases its cached
insert programs before the bulk search scan.

## Timing in tests

Tests do not assert on elapsed time. A test that checks a deadline arranges
for the peer to stay silent or keep trickling until the client gives up, so
only the deadline can end the wait; other tests wait for an event (a reply, a
child's acknowledgement file, a closed connection) with no deadline of their
own. A hung test is ended by nextest's slow-timeout (`.config/nextest.toml`),
so run the suite through `xtask check` or `cargo nextest run`; under plain
`cargo test` a hung test waits forever.

### Selected compile feedback

Run `agentctl job start polylogue compile -- -p <crate>` for a compiler check
in the normal development queue. Update `Cargo.lock` when adding a dependency
before invoking this locked operation. Full tests and package qualification
remain separate bulk jobs.

## Derived-row dictionary

`crates/store/src/encoding/rows.dict` is generated only from the typed neutral
examples in `crates/store/examples/row_dictionary/samples.rs`. Check its
reproducibility with the locked dependency versions:

```sh
agentctl job start polylogue exec --workspace <worktree> -- nix develop --command cargo run --locked --quiet -p polylogue-store --example row_dictionary -- --check crates/store/src/encoding/rows.dict
```

To regenerate, omit `--check`. Review the examples and asset together, bump
`INDEX_VERSION` and update the format digests in `schema/tests.rs`. A dictionary
belongs to one disposable index generation; original inputs and canonical
archive records never depend on it.
