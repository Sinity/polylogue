# Testing

How Polylogue is tested: the test archives (Morrow and independent goldens), the test layout, the properties tested, fault points and crash tests, end-to-end scenarios, fuzzing, benchmarks and their figures, and the row dictionary.

The commands that run all of this, and the standard every test is held to, are in [AGENTS.md](../AGENTS.md). The parser contract that goldens check is [archive.md#parsing](archive.md#parsing).

## Test data

Tests use synthetic data only: the Morrow archive and hand-written golden cases. Real data appears only in counts-only measurements; tests never read anyone's real archive or home, and a test daemon dies with its test.

## Morrow

Morrow is a synthetic archive: a week of work (2026-10-01 to 10-07) on a fictional Rust library, in Claude Code, Codex, ChatGPT and claude.ai, plus a Claude-Code-only parents set, a live set, hook spools and an annotation session. It lives in `examples/morrow/`; `FACTS.md` there owns its message identities, counting rules and cost interpretation, and `expected/` holds the query examples with their expected replies. Paths inside it use the synthetic home `/home/op`.

- `polylogue-testkit` lays Morrow out in a temporary home where each tool writes, runs the real engine over it and imports its exports, so tests read an archive built by the real parsers and writer: `Morrow::build(Set::Main)`, with `view()` and `stored_bytes()` giving what tests compare.
- The files are edited directly; nothing regenerates them. A new behaviour that Morrow cannot show is added to Morrow, with its facts in `FACTS.md`, rather than to a one-off fixture.
- The same archive is what `polylogue demo DIR` builds ([installing.md#demo](installing.md#demo)); the lines it prints and the examples in the [README](../README.md) are checked against it by a public CLI test.

## Goldens

A golden is the expected parse of an input, written by hand from knowledge of the format, never produced by running the parser and saving its output, and never re-blessed when the parser changes. A changed format gets a new input and a new golden; the old pair stays, since old formats stay readable. End-to-end goldens of CLI and MCP replies follow the same rule.

## Test layout

- A module's unit tests are in `name/tests.rs` (more files in `name/tests/`), declared last with `#[cfg(test)] mod tests;`.
- Integration tests share one binary per crate, with areas as modules so the dependency graph links once. The engine's `tests/engine/main.rs` contains `passes`, `properties` and `faults`; the application's `tests/e2e/main.rs` contains CLI, browser-host, onboarding, deadline and completion cases. Each area's other files stay beside its module; the sources use `tests/formats/main.rs`.
- Examples with unit tests set `test = true` in their Cargo manifest. Test commands select runnable test targets by default; examples without tests are compiled by Clippy.
- Helpers that need a crate's private items stay in that crate behind its test feature. CLI and MCP behaviour is tested through the public commands.

**Waiting, never timing.** No test asserts elapsed time. A test of a deadline makes the peer stay silent until the client gives up, so only the deadline can end the wait; other tests wait for an event (a reply, a file, a closed connection). An import's reply means its rows are readable. A test that reads what the daemon found itself (at its start, or in a file the test wrote) first waits with `status --wait` until the daemon has caught up, and with `--after` for a change it has not seen yet; no test polls. nextest's slow timeout ends a hung test.

## Properties

A test sits in the module of the area that owns its behaviour. The query language's shared tests live in common's [query-language](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/query-language/README.md) and run against Polylogue's catalog; the signal evaluator's conformance cases are in common's [signals](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/signals/README.md) and run against Polylogue's occurrence keys.

These properties are required for every input; named tests identify existing coverage. A verification result states which cases ran:

- Appending to an input in steps gives the same rows as reading it whole (`append_equivalence`).
- The order in which inputs arrive does not change the result (`order_independence`, `session_facts_do_not_depend_on_arrival_order`).
- A crash after each of append, sync and commit leaves only a state the open path resumes (it truncates an uncommitted segment tail), with no repair step (`crash_leaves_only_a_tail_at_each_product_commit_boundary`).
- Rebuilding the index from stored bytes gives the same replies (`a_deliberate_index_rebuild_keeps_authored_rows_and_vectors`; in primary mode `the_morrow_example_rebuilt_from_sinex_gives_the_same_answers`).
- An export imported into a fresh archive gives the same sessions, IDs and user state (`an_export_imported_into_a_fresh_archive_gives_the_same_sessions_ids_and_user_state`, `read_to_bundle_import_round_trip_preserves_structured_values_and_unknown_tool_fields`).
- The current revision of an authored record is a pure function of its revision links (`current_revisions`, `annotation_revision_state`).
- Budgeted pages concatenate to the whole read (`fused_pages_concatenate_to_the_whole_reply`).
- Archived text renders inert in every format (`archived_html_and_dangerous_urls_stay_inert`, `archive_quotes_escape_roles_and_controls_without_changing_source`).
- Writes between pages neither repeat nor skip rows, and a rebuild retires the cursor (`writes_between_pages_neither_refuse_nor_repeat_rows`).
- Re-parsing the same bytes twice yields identical rows; the same question gives identical replies over CLI, socket, HTTP and MCP; every writer-kept counter equals the count it stands for.

## Fault points and crash tests

Crash tests stop the writer after each of append, sync and commit, and check that the next open resumes each state a crash can leave, losing only the uncommitted group ([architecture.md#durability](architecture.md#durability) owns the states).

The store and engine have a `faults` feature that adds a fault registry. Only a test's own call to arm a named point activates it, in that process. The names checked by product code are declared in `crates/store/src/fault.rs`; arming or checking an undeclared name is an error. Points sit at write steps (between commits, inside a group, during migration and catch-up) and at readers whose deadlines a test exercises. Shipped builds never enable the feature. Tests that check durability use a real filesystem, not a memory one.

## Scenarios

One end-to-end harness: the real `polylogue` binary runs as a daemon over a temporary archive built from a seeded template, driven through the CLI, MCP and the browser extension's native-messaging host, with replies compared to goldens. Harness behaviour is tested with fixture harnesses: stub binaries on `PATH` that record their argv, cwd and stdin, so sends, resumes and installs are checked without a real agent.

**Isolation.** A run over synthetic data finds its inputs under a synthetic home and nowhere else, because discovery roots come from `HOME` and the tools' own root variables (`CLAUDE_CONFIG_DIR`, `CODEX_HOME`, `HERMES_HOME`, `XDG_STATE_HOME`, and whichever the origins declare). The run command sets them, never its caller: every daemon and CLI a scenario, `cargo xtask run` or `bench tails` spawns runs in the testkit's sandbox, which clears the environment, points `HOME`, the XDG directories, every declared root variable and `TMPDIR` under its root, and names a free web port ([development](development.md#a-daemon-to-probe)). `polylogue demo` and the first-build bench construct their engine configuration with the synthetic home and no root overrides, so the environment is not consulted. The `sandbox` cases check it through the real binary: under a parent whose `HOME`, XDG directories, tool variables and `POLYLOGUE_LISTEN` point at a marker directory full of the tools' files, the daemon records only the sandbox's session and leaves the marker untouched, also while the default web port is taken, where its web reader listens on the sandbox's port; and `cargo xtask run` answers `status` from its scratch archive and leaves no process once stopped.

**Primary mode** uses a real Sinex stack (PostgreSQL with TimescaleDB, NATS and
`sinexd` at the revision Polylogue pins). Morrow converged in primary mode
must reply as standalone does, and again after rebuilding from the confirmed
stream; backup and restore must preserve the same observations
([primary-mode.md](primary-mode.md)). A result shows these laws hold only when
those real-stack cases ran. Test selection and service preparation belong to
[development](development.md); generated command help describes the available
runner options.

The native binary-recovery case uses `SINEXD` and `SINEXD_BEFORE`, with
`SINEXD_REVISION` and `SINEXD_BEFORE_REVISION` naming their verified source
revisions. It keeps one seeded PostgreSQL database, NATS server and content
store through normal stops and starts. It checks the older binary's explicit
refusal of a new declaration, then the newer binary's ordinary reads and
producer admission over the same originals. Its event payload and material
identity/hash/length checks do not alone verify original-byte retrieval.
PostgreSQL runs with fsync,
synchronous commits and full-page writes enabled. This is a normal-restart
recovery check; it does not establish crash or full-corpus restore behavior.

## Fuzzing

Fuzzing runs as tests. Each fuzz target's corpus is replayed, bounded, by the default test run; longer runs are a test profile over the same harness, and a host may queue them through its own job runner.

## Benchmarks

Performance is measured by paired `bench` runs: the base and the change built and run together over one sample on the same host, so host load affects both alike. A change is judged by the pair, never by comparing runs from different times.

`cargo xtask bench --root PATH ...` keeps manifests, sample copies, archives, reports and the benchmark lock under one directory; without the flag it uses `POLYLOGUE_BENCH_HOME`, then `$XDG_DATA_HOME/polylogue-bench`. Selected inputs are copied under that root before the manifest hashes them, and SQLite samples are read in one read transaction, so later changes at the source do not alter a sample. A corpus estimate that predicts a negative or non-finite duration is refused.

What is measured: the first build per stage; incremental updates; interactive reads; the CLI latency classes end to end ([latency.md#latency-classes](latency.md#latency-classes)); daemon start and restart; idle I/O over 10 minutes; the append-to-view latency of live sessions.

**Figures.** Each report gives hardware-independent figures: bytes read and written per input byte and per event, syncs per writer group, and stage utilisation (the share of time each stage is busy, and whether any stage idles with a queue). Absolute rates, wall time per stage, peak memory, final storage sizes and the host's load during the run are reported beside them for the host that ran it. A reading the runner could not take stays unknown. The targets these figures are checked against are in [architecture.md#pipeline](architecture.md#pipeline).

**First build.** A first-build report gives wall time, time per stage (pool time in thread-seconds), input bytes, physical I/O, peak memory and final sizes. The runner refuses a run with ingestion errors, unreadable locations or no message writes. Samples of two or more sizes let the bench fit each stage as fixed time plus time per input megabyte and predict a whole corpus stage by stage, and a full run validates that prediction.

**Reads.** The read benchmark generates one 16 MiB session and eight small ones with the testkit builders, checks that a complete read holds every message once in order, that windows at the first, middle and last positions match it, and that repeated windows render the same bytes, then reports daemon start, the first request after each restart and later requests (median and p95), page counts and daemon memory.

The suite runs in the hosted check on each pull request, as four nextest shards in parallel; locally an agent runs only the tests it writes or changes ([development](development.md)). The shards' stages (setup, build, run) are recorded per job in the run's summary lines. A pull request whose changes select every test took 8.8 minutes end to end on 26 November 2026, each shard 2 to 3 minutes of setup, 2 to 3.5 minutes of building and 2 to 3 minutes of running. Selected tests take less; recorded runs give the current figures (`gh run list --workflow check`).

## Row dictionary

`crates/store/src/encoding/rows.dict` compresses index rows. It is generated only from the synthetic samples in `crates/store/examples/row_dictionary/samples.rs`, and `row_dictionary_rebuilds_from_its_samples` confirms that it reproduces from them; `cargo run -p polylogue-store --example row_dictionary PATH` regenerates it. Changing it is an index format change. Archive records never depend on it.

The `first_build` example includes the final archive and index WAL checkpoints
in its elapsed time and I/O totals. A busy reader or nonempty final WAL makes
the measurement fail; engine shutdown timing remains separate.
