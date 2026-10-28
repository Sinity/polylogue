# Testing

How Polylogue is tested: the test archives (Morrow and independent goldens), the test layout, law suites and how laws map to tests, fault points and crash tests, end-to-end scenarios, fuzzing, benchmarks and their figures, and the row dictionary.

The commands that run all of this, and the standard every test is held to, are in [AGENTS.md](../AGENTS.md). The parser contract that goldens check is [archive.md#parsing](archive.md#parsing).

## Test data

Tests use synthetic data only: the Morrow archive and hand-written golden cases. Real data appears only in counts-only measurements; tests never read anyone's real archive or home, and a test daemon dies with its test.

## Morrow

Morrow is a synthetic archive: a week of work (2026-10-01 to 10-07) on a fictional Rust library, in Claude Code, Codex, ChatGPT and claude.ai, plus a Claude-Code-only parents set, a live set, hook spools and an annotation session. It lives in `examples/morrow/`; `FACTS.md` there owns its message identities, counting rules and cost interpretation, and `expected/` holds the query examples with their answers. Paths inside it use the synthetic home `/home/op`.

- `polylogue-testkit` lays Morrow out in a temporary home where each tool writes, runs the real engine over it and imports its exports, so tests read an archive built by the real parsers and writer: `Morrow::build(Set::Main)`, with `view()` and `stored_bytes()` giving what laws compare.
- The files are edited directly; nothing regenerates them. A new behaviour that Morrow cannot show is added to Morrow, with its facts in `FACTS.md`, rather than to a one-off fixture.
- The same archive backs the tour in the [README](../README.md) (`polylogue demo tour DIR`), whose answers a public CLI test checks.

## Goldens

A golden is the expected parse of an input, written by hand from knowledge of the format, never produced by running the parser and saving its output, and never re-blessed when the parser changes. A changed format gets a new input and a new golden; the old pair stays, since old formats stay readable. End-to-end goldens of CLI and MCP answers follow the same rule.

## Test layout

- A module's unit tests are in `name/tests.rs` (more files in `name/tests/`), declared last with `#[cfg(test)] mod tests;`.
- Integration tests are one binary per area: `crates/<crate>/tests/<area>/main.rs`, with the area's other files beside it. For example the engine's `passes` (the loop over files a test writes), `properties` (properties every archive keeps, as property tests over generated archives and Morrow) and `faults`; the sources' `formats`; the daemon's `e2e`.
- Helpers that need a crate's private items stay in that crate behind its test feature. CLI and MCP behaviour is tested through the public commands.

**Waiting, never timing.** No test asserts elapsed time. A test of a deadline makes the peer stay silent until the client gives up, so only the deadline can end the wait; other tests wait for an event (a reply, a file, a closed connection). A test that waits for the daemon to read a change waits until `status` shows it up to date. nextest's slow timeout ends a hung test.

## Law suites

Every area page ends with its laws. Each law has a stable key unique in its page, and its full name is `<page>/<key>`, for example `operations/open-never-deletes`.

- **A test that checks a law carries the key in its name,** in snake case: `open_never_deletes`, with a suffix when several tests check one law. Finding a law's tests is a name search; no table maps them by hand.
- Integration test binaries are already one per area, so a law's tests sit in the binary of the area that owns it.
- **Shared laws live with the shared code:** the query-language law suite is in common's [query-language](https://github.com/Sinity/polylogue-sinex-common/blob/master/docs/query-language.md) and runs against Polylogue's catalog; the signal evaluator's conformance cases are in common's [signals](https://github.com/Sinity/polylogue-sinex-common/blob/master/docs/signals.md) and run against Polylogue's occurrence keys.
- **Coverage is a report:** `cargo xtask laws` (devkit) lists each law key with the tests whose names carry it, and the laws that have none. It fails nothing; a law without a test is a gap for the area's owner.
- Cross-cutting laws (crash states, parity across CLI, socket, HTTP and MCP) are in [architecture.md](architecture.md) and [interfaces.md](interfaces.md).

## Fault points and crash tests

Crash tests stop the writer at every point between append, sync, commit and truncate, and check that the next open resumes each state a crash can leave, losing only the uncommitted group (the law is [architecture.md#durability](architecture.md#durability)'s).

The store and engine have a `faults` feature that adds a fault registry. It starts empty: only a test's own call to arm a named point activates it, in that process. Points sit at write steps (between commits, inside a group, during migration and catch-up) and at readers whose deadlines a test exercises. Shipped builds never enable the feature. Tests that check durability use a real filesystem, not a memory one.

## Scenarios

One end-to-end harness: the real `polylogue` binary runs as a daemon over a temporary archive built from a seeded template, driven through the CLI, MCP and the browser extension's native-messaging host, with answers compared to goldens. Harness behaviour is tested with fixture harnesses: stub binaries on `PATH` that record their argv, cwd and stdin, so pings, resumes and installs are checked without a real agent.

**Primary mode** is tested against a real Sinex stack started by devkit (PostgreSQL with TimescaleDB, NATS and `sinexd` at the Sinex revision Polylogue pins): Morrow converged in primary mode must answer as standalone does, and again after a rebuild from the confirmed stream ([primary-mode.md](primary-mode.md)). These tests need that stack and run where it is available.

## Fuzzing

Fuzzing runs as tests. Ruling, 2026-03-12: fuzzing "MUST NOT be a separate command, it's all suposed to be work via test" (cc-queued:065c511c-e336-4e59-825b-eeefc71cdaae@2026-03-12T00:31:10). Each fuzz target's corpus is replayed, bounded, by the ordinary test run; longer runs are a test profile over the same harness, and a host may queue them through its own job runner.

## Benchmarks

Performance is measured by paired `bench` runs: the base and the change built and run together over one sample on the same host, so host load affects both alike. A change is judged by the pair, never by comparing runs from different times.

What is measured: the first build per stage; incremental updates; interactive reads; the CLI latency classes end to end ([interfaces.md#latency-classes](interfaces.md#latency-classes)); daemon start and restart; idle I/O over 10 minutes; the append-to-view latency of live sessions.

**Figures.** Each report gives hardware-independent figures: bytes read and written per input byte and per event, syncs per writer group, and stage utilisation (the share of time each stage is busy, and whether any stage idles with a queue). Absolute rates, wall time per stage, peak memory, final storage sizes and the host's load during the run are reported beside them for the host that ran it. A reading the runner could not take stays unknown. The targets these figures are checked against are in [architecture.md#pipeline](architecture.md#pipeline).

**First build.** A first-build report gives wall time, time per stage (pool time in thread-seconds), input bytes, physical I/O, peak memory and final sizes. The runner refuses a run with ingestion errors, unreadable locations or no message writes. Samples of two or more sizes let the bench fit each stage as fixed time plus time per input megabyte and predict a whole corpus stage by stage, and a full run validates that prediction.

**Reads.** The read benchmark generates one 16 MiB session and eight small ones with the testkit builders, checks that a complete read holds every message once in order, that windows at the first, middle and last positions match it, and that repeated windows render the same bytes, then reports daemon start, the first request after each restart and later requests (median and p95), page counts and daemon memory.

The whole suite runs in under 10 minutes, measured per stage (compile, setup, run).

## Row dictionary

`crates/store/src/encoding/rows.dict` compresses index rows. It is generated only from the synthetic samples in `crates/store/examples/row_dictionary/samples.rs`, and a check confirms that it reproduces from them. Changing it is an index format change. Archive records never depend on it.

## Laws

- **goldens-independent**: no golden file is written by the code it checks; regenerating parser output never changes a golden.
- **tests-use-synthetic-data**: no test opens a path outside its temporary home and the repository.
- **no-elapsed-time-assertions**: no test asserts on elapsed time.
- **fault-registry-empty-by-default**: with the `faults` feature built in and no point armed, every test behaves as in a build without it.
- **fuzz-corpus-replayed**: every fuzz target's corpus replays in the ordinary test run.
- **row-dictionary-reproduces**: the committed row dictionary equals the one generated from the synthetic samples.
