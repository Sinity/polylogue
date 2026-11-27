# Polylogue

Polylogue is an archive and workbench for AI work. It captures every session
its user has with AI tools (Claude Code, Codex, Gemini, ChatGPT, claude.ai, AI
Studio and others, through their files, exports, hooks and a browser
extension), keeps the original bytes forever, parses them into one model of
sessions and their lineage, and lets people and agents search, read, measure
and annotate that history. Agents are primary users: they get the same powers
as the operator, through MCP first and the CLI second.

Polylogue runs alone (standalone, on SQLite) or on Sinex (primary mode), where
its bytes live in Sinex's store and its record is shipped to Sinex. Primary
mode is the integration the operator runs; standalone is a complete product
for everyone else and for development.

This file explains how the product is designed and how work is done in this
repository. Claude Code, Codex and any other agent read it; it holds the
working rules that apply here.

## The design of record

The pages in `docs/` are Polylogue's design: behaviour, model and formats,
one page per area, each fact on one page. They describe the product as
designed, not the code as it is; when the code and a page differ, the page is
the target and closing the difference is the work. The selected Bead bounds
what the current delivery must finish: unrelated future capabilities and
other design gaps do not become its scope or release blockers. It does not
license leaving a wrong design or a known defect in the code the delivery
changes. `docs/README.md` indexes
the pages. Start with `docs/architecture.md` (purpose, principles, modes, stores,
durability, pipeline, configuration and clocks), then read the page that owns
the area you are changing; `docs/vocabulary.md` lists every surface word.
CLI help, MCP tool schemas and module documentation describe the current
build.

Plans, open questions, decisions and acceptance criteria live in Beads, the
task tracker, outside this repository. A Bead points at the doc section it
builds; docs never point at Beads.

When a page is silent, decide from the principles below and in
`docs/architecture.md`, record the decision on the Bead and in the PR
description, and write the resulting behaviour into the owning page in the
same change. A change to the design is a change to its page.

## The core model, in brief

Each concept is owned by the page named; this is orientation, not a second
copy.

- **The archive keeps the original inputs** (inputs, revisions, chains,
  extents, content-addressed blobs in an append-only segment log) and holds
  what cannot be recomputed; `index.db` holds everything parsed and derived
  and is a function of that record, whatever order inputs arrived in
  (`docs/archive.md`, `docs/architecture.md`).
- **Sessions and lineage**: a message is stored once; links between sessions
  come only from explicit markers; a compaction is a context boundary; the
  logical work session is the default unit (`docs/sessions.md`).
- **Authorship is computed, never supplied**: the daemon derives `author` and
  `via` from the channel and the harness (`docs/authorship.md`).
- **Annotations are the one record of everything authored**: notes, tags,
  judgments, comparisons, saved queries, watches and settings; the current
  revision is a function of the revision graph (`docs/annotations.md`).
- **Pages arrange the archive**: a page's sections and slots are annotations
  that hold references, never copies of captured text; page text is Markdown
  whose links, citations, embeds and inclusions become link rows; the same
  verbs build and read pages (`docs/pages.md`, `docs/page-text.md`).
- **Queries, runs, saved queries and kept results are identified objects**,
  and every run is recorded (`docs/query.md`, `docs/saved-queries.md`);
  **references** use natural keys and one parser (`docs/references.md`).
- **Derived data** is computed at session grain as data arrives, with its
  construct and validity evidence declared (`docs/measures.md`).
- **One operation table drives every surface**: the CLI, the socket, HTTP and
  MCP are generated from it. A line is one selection (a quoted query, a
  reference, a row number or a saved name) followed by verbs, or a command;
  agents use the same grammar through three MCP tools (`docs/interfaces.md`,
  `docs/agents.md`).
- **Polylogue has no deletion**; re-parse replaces a chain's reading with a
  new reading of the same bytes (`docs/architecture.md`).
- **Primary mode** keeps the same record and index, hands its bytes to
  Sinex's store, ships the record to Sinex, and applies what other hosts and
  Sinex's lifecycle decided (`docs/primary-mode.md`).

## Repository layout

| Path | Holds |
|---|---|
| `crates/model` | domain types, replies, `Plan` types, references and their parser, closed sets, `Origin`. No I/O |
| `crates/sources` | one declaration and one parser per origin, reading versions, markup extraction |
| `crates/store` | every SQLite connection and all SQL: inputs, chains, versioned rows, row codecs, plan execution, migrations |
| `crates/query` | the catalog and units, compilation to a `Plan`. No SQL |
| `crates/render` | text, markdown and HTML over typed replies |
| `crates/reading` | reading stored sessions and records as pages and streams: composition, paging both ways, the links walk, layout facts, the invocation context. No SQL |
| `crates/engine` | discovery, the overlapped ingest pipeline, tool-database capture, recognition, derivations |
| `crates/sinex` | Polylogue's kind declarations and its event encoders and decoders for Sinex (behind the `sinex` feature); primary mode's shipper and reader of Sinex's log are written here |
| `crates/polylogue` | daemon, CLI, MCP, web, browser host and brief builder |
| `crates/testkit` | test archives built through the real loop: the synthetic Morrow archive (`examples/morrow/`), independent goldens |
| `clients/python/` | standard-library Python client and real-daemon contract tests |
| `extension/` | the browser extension |
| `xtask/` | developer commands, built on `devkit` from polylogue-sinex-common |
| `nix/` | the NixOS and Home Manager modules and their shared settings library |

Authored records live in `crates/store` and `crates/polylogue`.
Only the store touches SQLite.

The shared crates (`hashed-blobs`, `devkit`, `query-language`,
`command-surface`, `web-shell`, `authored`, `signals`, `signal-delivery`) follow polylogue-sinex-common's `master`
branch, with the exact commit pinned in `Cargo.lock`. Its
[dependency guide](https://github.com/Sinity/polylogue-sinex-common#depending-on-this-repository)
describes lock updates and local development with `cargo xtask link-common`.
Code is shared only where the mechanics are identical; Polylogue's concepts
stay its own.

## Building, testing and measuring

[Development](docs/development.md) owns shell setup, command selection,
local overrides and build graphs. Enter with `nix develop`. While writing,
type-check with `cargo check --locked -p <package> --tests` (the `compile`
operation) and run only the tests being written or changed, scoped to their
package and target, with `cargo xtask test -p <package> --lib -E
'test(<name>)'` (the `focused` operation). The suite runs in the hosted check
on the pull request. A test run builds only the
selected packages' test targets, with the units the full test build uses;
Cargo target flags such as `--lib` or `--bin NAME` bound it further. After
changing a dependency or its features, run `cargo hakari generate`.

**What a test checks.** A test exercises behaviour a caller can observe
through a stable interface (a command, an operation, a reader, a stored
format) and compares it with an expected result known independently of the
code under test: a hand-written golden, a Morrow fact, an oracle such as the
original SQL. Code that was replaced is deleted, and its absence needs no
test; tests describe what the product does now. Unit tests live in
`name/tests.rs` as [docs/testing.md](docs/testing.md#test-layout) lays out.
Assertions are exact. When an expectation changes, the commit cites the doc
section whose behaviour changed; a failing assertion is a finding to fix at
its cause, never one to widen until it passes. A slow test has its slowness
fixed rather than its wait extended.

**Performance is a product goal.** The targets and the overlapped
pipeline are in `docs/architecture.md`. Measure with `cargo xtask bench pair`
(before and after, in turn, over one fixed sample under a memory limit) and
report hardware-independent figures (CPU-seconds, bytes written per input
byte, peak memory) before the absolute times. No test asserts elapsed time. Designs target the data and
the guarantees, not one machine; host tuning belongs in deployment
configuration.

## How work is done here

**Design first.** Before writing code, read the page in `docs/` that owns the
area, the pages it links, the current code of the area, and the earlier
Python Polylogue's implementation and tests for what you are building (it is
in this repository's Git history). Write the whole change from the design:
complete features, built on the architecture's abstractions, with the earlier
Python Polylogue's behaviour and test cases ported.

**Edit in bulk; type-check as you go.** Change whole modules, crates and call
sites in one pass. Type-check each layer you write with the `compile`
operation; it reports every type error at once. Run the tests you wrote or
changed once the change is coherent.

**Review your own work before verifying it.** Read the whole diff once
against the task, the owning page, `docs/architecture.md` and the
earlier Python Polylogue's behaviour, and list everything wrong in that one pass: a
requirement not met, a mechanism duplicated, a predecessor left in place, a
caller not converted, a limit that refuses valid input, a test that cannot
fail, a name that does not say what the thing is, a case the design covers
and the code does not. Fix the whole list, then read only what the fixes
changed. Then verify. The author makes the work good; a separate reviewer is not the mechanism for
that.

**Verify scoped to the change.** Commit first, so a result belongs to a
commit. Locally, run only the tests the change writes or changes: the
development host's build queue is shared, while the hosted check runs the
suite in four parallel shards in about ten minutes. A pull request is
verified by GitHub Actions, which runs the parts of `cargo xtask check` as
parallel jobs for its head (affected tests only); every test runs again on
master after the merge. Run a full local `cargo xtask check` only to
reproduce a failure the hosted check reported. Before treating a failure as yours,
compare with the same test on current master (`gh run list -b master`).

**Finish what the task names.** A task states its done condition before work
starts, as behaviour on master and the tests that prove it; a runtime task
also names the deployment, data and client observations that prove it. A
defect inside the area that stops that condition from holding is fixed in the
same work, and so is a wrong design in the code the change touches (*Full
features, built on a whole architecture*, below). Anything else found goes on
a Bead with what was found, and the current work lands without it; unselected
designs and globally ready tasks are not implicit prerequisites. Keep one
current-state note on the Bead with evidence pointers, and leave unmet
criteria open, naming the observation or blocker that decides them.

**Git and pull requests.** Work on a branch from `origin/master`; every pull
request targets `master`, and stacked pull requests are not used. Commit with
plain messages whose body says why. Stage explicit paths. Land a feature as a
sequence of pull requests, each of which compiles, passes its tests and
leaves master coherent: complete for what it covers, with no stub to replace
later and no compatibility path between steps. Small related fixes share one
pull request, since each pull request costs a full CI run. Master requires
the `check` status on the pull request's head; branches need not be up to
date with master, so rebase only to resolve a conflict, taking master's names
and structure and reapplying your change. The owner's part is to finish the
pull request: not a draft, hosted review findings answered with a fix or a
refutation and their threads resolved. The landing queue (`dev/land --queue`)
then tests it in a batch with other ready pull requests on master's tip and
merges the batch when that test is green; with the queue running an owner
does not land by hand. A pull request the queue sends back fails only
combined with master; a new head puts it back. The label `hold`, with the
reason in a comment, parks a finished one. GitHub auto-merge is not used, since
armed on a clean, green head it sometimes never fires. Development speed
comes before a perfectly green master: when two
separately green changes combine into a failing master, whoever's change
caused it fixes it next. Hooks stay enabled.

**PR descriptions** are plain prose: what changed and why, the design
decisions taken and their reasons, what was verified (commands and run IDs)
and what was not, and before and after figures for anything performance
related. The body ends with a line `Task: <id>` naming the task it lands.

**Decide technical details yourself.** Low-level design choices are yours,
made from the principles here and the pages in `docs/`, and recorded on the
Bead and in the PR.
The operator decides product scope and intent, privacy, money (paid APIs,
purchased vectors) and anything published outward. Deploying, activating,
restarting, rebuilding an archive and moving to a new release are yours: original
inputs stay where their sources keep them and an old archive is set aside, not
deleted. No stored format or archive is deployed in a sense that needs
compatibility or migration; target one current format. Original input bytes
and paid outputs such as embedding vectors are preserved; a one-time
operation on an archive is written, run, and leaves a result, not code that
stays. When an operator
question is open, take the conservative option, keep
working on the rest, and put the question in the report and on the Bead.

**A request to check is read-only.** When asked to inspect, analyse or
explain, report and recommend; change only what was asked for.

## Engineering values

These are the operator's standing positions, explained so they can be applied
to cases they do not name.

**Invariants instead of repair.** Code does not produce broken state, so
there is nothing to repair. There are no repair, reconcile, reindex or
verify-and-fix commands and no scheduled checks. A crash leaves only states
the open path treats as normal: an uncommitted segment tail, a
derived store behind its saved position, a source position behind its
committed rows. When something is found broken, fix the code that produced
it; often the cause is code that no longer exists and no new invariant is
needed.

**One mechanism per concept; a replacement removes its predecessor.** When
something is replaced, the old version, its callers, docs, tests and
everything that only served it go in the same change. There are no
compatibility shims, aliases, fallbacks or old-format readers for Polylogue's
own formats; a breaking change fixes every caller. Outside formats are
different: provider exports from any year are user data and stay supported.

**Full features, built on a whole architecture.** Build the complete feature
the design describes, on abstractions that make the next features cheap,
rather than a bare-bones version to be replaced later. Polylogue is meant to
do more than the earlier Python Polylogue did, without its machinery.
When an area you are changing works but its design is wrong, redesign it as
part of the task; an area you are not changing gets a Bead.

**Use the earlier Python Polylogue's knowledge, not its mechanisms.** It
had a working core under layers of repair and bookkeeping. Port
its format knowledge, parsers' cases, capabilities and planned work; leave
its repair passes, census and debt ledgers, revision authority and audit
tiers behind.

**Small codebase, plain names.** Bulk is the main enemy: the earlier
Python Polylogue and the earlier Sinex grew past half a million lines each through kludges on kludges. Prefer one
deep abstraction to several shallow ones, clean internal and external
interfaces, and no repeated logic. A name says what the thing is in ordinary
words, everywhere: types, functions, local variables, SQL, JSON keys, CLI
help, errors, comments, tests and docs. When an opaque term is found, replace
it throughout the codebase in one pass, so that nobody copies it from what is
already there.

**Good design instead of guard machinery.** Quality comes from the
architecture and from the author's own review, not from build-enforced bans,
line budgets, word checks or required PR fields. A pattern that is absent
from the codebase does not reappear by itself.

**Valid input is never refused or truncated for size.** Page sizes are
defaults, transfer units continue, and memory is bounded by streaming and
read-ahead, not by caps on what the user may store or ask for. A query that
runs past its deadline names a narrower query and the same command with the
deadline lifted.

**Tests that are better than the earlier Python Polylogue's.** Every specified behaviour
in the pages in `docs/` has a test
that fails when the production path breaks. The valuable kinds: independent
oracles (goldens written from format knowledge, never re-blessed from parser
output), tests over the Morrow archive, crash tests at real durability
boundaries, and property tests at serialization boundaries. Port the
earlier Python Polylogue's cases as cases (inputs and expected outcomes), then fill the
gaps both suites had. A test that restates a helper, asserts a mock's value,
checks what the type system already prevents, or matches SQL substrings where
rows could be compared adds cost without coverage. A test asserts behaviour
that exists, never that a name, variant or format stays absent: a removal is
verified by the build and by tests of the replacement's behaviour.

**A fast development loop.** Code quality comes first; within it, optimise
for speed: bulk edits, targeted test runs, short compile times, a small
memory footprint. When a run slows down, look at slow tests, resource waits
and cleanup before raising timeouts or cutting coverage.

## Code conventions

- Rust, four-space indent, `rustfmt`.
- A module is `name.rs` with its children in `name/`; no `mod.rs`.
- Tests live in their own files, never inline; `docs/testing.md` owns their
  layout. Use synthetic inputs,
  temporary directories and `polylogue-testkit`.
- Comments state constraints and reasons, briefly. History, status, Bead IDs
  and section numbers belong in Git and Beads.
- Configuration is loaded once in each binary's `main` and passed
  explicitly with a `Clock`; libraries never read the environment or the
  system zone (`docs/architecture.md`, configuration and clocks).
- Unsupported operations return a typed error, never an empty success.
  Unknown times stay unknown, and unknown costs are never zero. IDs are
  opaque strings and may contain colons.
- Each stored format has one current version (`docs/archive.md`). A format
  change rewrites the format, its writers and its readers together and
  regenerates test data; no reader for an earlier version of Polylogue's own
  formats is kept, and what is derived is re-parsed from the original inputs.
  Inputs are read where the outside tools write them and recognised by
  declared structure (`docs/capture.md`).

## Vocabulary

`docs/vocabulary.md` lists each concept and its one name. Before naming
something, look for it there; use the listed word, or add the new concept in
the same change. Names in the code, the docs and this file agree.

## Data and privacy

The repository is public. Fixtures are synthetic and neutral (the Morrow
archive, `/home/op` paths). Credentials, real captures, private reports and
real databases stay out of Git, and so do paths into a particular machine.
Reading the operator's real files on the development machine for development
and measurement is fine; report figures, not contents.
Public docs state design intent in neutral language; PR bodies, review
comments and commit messages carry no excerpts of captured data or private
analyses.

## Cloud sessions and changes to polylogue-sinex-common

In cloud sessions, check for `${CLOUD_PREFIX:-../.cloud}/env.sh` from the
checkout root before development commands. When present, source it in every
command shell:

```sh
. "${CLOUD_PREFIX:-../.cloud}/env.sh"
```

If absent, read the
[shared cloud setup guide](https://github.com/Sinity/polylogue-sinex-common/blob/master/dev/cloud/README.md)
before choosing a backend or running `bash dev/cloud/setup.sh`. The included
polylogue-sinex-common checkout owns that setup implementation and its pins.
Preserve a resumed task's branch, commits and uncommitted work; sourcing
initializes tools. Run `cargo xtask` directly; each run owns its services.

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
`../polylogue-sinex-common`). Linking is explicit. It writes the git-ignored
`.cargo/local.toml`, and Cargo.lock then names the local crates, so the lock
is not committed in that state. `cargo xtask unlink-common` restores the lock.
The common pull request's `products` jobs run this workspace's checks against
its head. Merge the common pull request first. Polylogue also pins Sinex by
revision (`crates/sinex/Cargo.toml`), and that Sinex revision's own common
dependencies must resolve against the same common commit, so Polylogue moves
after Sinex: once Sinex's master adopts the common commit, common's `Update
product locks` workflow opens or updates the pull request here, on the
`update-polylogue-sinex-common` branch, that moves the Sinex revision and the
common lock together. Convert the callers on that branch.
