# Polylogue

Polylogue is an archive and workbench for AI work. It captures every session
its user has with AI tools (Claude Code, Codex, Gemini, ChatGPT, claude.ai, AI
Studio and others, through their files, exports, hooks and a browser
extension), keeps the original bytes forever, parses them into one model of
sessions and their lineage, and lets people and agents search, read, measure
and annotate that history. Agents are primary users: they get the same powers
as the operator, through MCP first and the CLI second.

Polylogue runs alone (standalone, on SQLite) or on Sinex (primary mode), where
Sinex stores its bytes and records and Polylogue keeps a local index. Primary
mode is the integration the operator runs; standalone is a complete product
for everyone else and for development.

This file explains how the product is designed and how work is done in this
repository. Claude Code, Codex and any other agent read it; it holds the
working rules that apply here.

## The design of record

The pages in `docs/` are Polylogue's design: behaviour, model and formats,
one page per area, each fact on one page. They describe the product as
designed, not the code as it is; when the code and a page differ, the page is
the target and closing the difference is the work. `docs/README.md` indexes
the pages and defines how the operator's rulings are cited. Start with `docs/architecture.md` (purpose, principles, modes, stores,
durability, pipeline, configuration and clocks), then read the page that owns
the area you are changing. CLI help, MCP tool schemas and module
documentation describe the current build.

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
  (`docs/archive.md`, `docs/architecture.md`).
- **Sessions and lineage**: a message is stored once; links between sessions
  come only from explicit markers; a compaction is a context boundary; the
  logical work session is the default unit (`docs/sessions.md`).
- **Authorship is computed, never supplied**: the daemon derives `author` and
  `via` from the channel and the harness (`docs/assertions.md`).
- **Assertions are the substrate of everything authored**: notes, tags,
  judgments, saved queries, Pages and settings; the current revision is a
  function of the revision graph (`docs/assertions.md`, `docs/pages.md`).
- **Queries, results, findings and figures are identified objects**, and every
  run is recorded (`docs/query.md`); **references** use natural keys and one
  parser (`docs/references.md`).
- **Derived data** is computed at session grain as data arrives, with its
  construct and validity evidence declared (`docs/measures.md`).
- **One operation table drives every surface**: the CLI, the socket, HTTP and
  MCP are generated from it (`docs/interfaces.md`).
- **Polylogue has no deletion**; re-parse replaces a chain's rows with a new
  reading of the same bytes (`docs/architecture.md`).
- **Primary mode** sends bytes and records to Sinex and applies Sinex's
  confirmed stream to the local index (`docs/primary-mode.md`).

## Repository layout

| Path | Holds |
|---|---|
| `crates/model` | domain types, answers, `Plan` types, references and their parser, closed sets, `Origin`. No I/O |
| `crates/sources` | one declaration and one parser per origin, reading versions, markup extraction |
| `crates/store` | every SQLite connection and all SQL: inputs, chains, versioned rows, row codecs, plan execution, migrations |
| `crates/query` | the catalog and units, compilation to a `Plan`. No SQL |
| `crates/render` | text, markdown and HTML over typed answers |
| `crates/engine` | discovery, the overlapped ingest pipeline, tool-database capture, recognition, derivations |
| `crates/sinex` | primary mode: Sinex's producer client wiring, the confirmed-stream consumer, Polylogue's kind declarations (behind the `sinex` feature) |
| `crates/polylogue` | the daemon library and thin CLI, MCP, web and browser-host front ends generated from the command table; the pack builder |
| `crates/testkit` | test archives built through the real loop: the synthetic Morrow archive (`examples/morrow/`), independent goldens |
| `extension/` | the browser extension |
| `xtask/` | developer commands, built on `devkit` from polylogue-sinex-common |
| `nix/` | the NixOS and Home Manager modules and their shared settings library |

Authored records live in `crates/store` and `crates/polylogue`.
`cargo xtask boundaries` enforces which crates may depend on which; only the
store touches SQLite.

The shared crates (`hashed-blobs`, `devkit`, `query-language`,
`command-surface`, `web-shell`, `authored`, `signals`, `signal-delivery`) follow polylogue-sinex-common's `master`
branch, with the exact commit pinned in `Cargo.lock`. Its
[dependency guide](https://github.com/Sinity/polylogue-sinex-common#depending-on-this-repository)
describes lock updates and local development with `cargo xtask link-common`.
Code is shared only where the mechanics are identical; Polylogue's concepts
stay its own.

## Building, testing and measuring

Enter the development shell with `direnv allow` (the `.envrc` loads the
flake) or `nix develop`. It carries the toolchain and the development tools.
Put personal settings (`CARGO_TARGET_DIR`, job counts) in a git-ignored
`.envrc.local`.

| Command | Does |
|---|---|
| `cargo xtask check --changed` | formatting, crate boundaries, Clippy (unchanged crates reuse their result) and the tests your changes since the merge base with `origin/master` can affect: the loop while editing |
| `cargo xtask check [--changed-base <ref>]` | the same with every test, or with the tests the changes since the merge base with `<ref>` can affect |
| `cargo xtask fast` | formatting, boundaries and clippy, without tests |
| `cargo xtask test -p <crate> -E 'test(<name>)'` | the selected tests under nextest |
| `cargo xtask test --changed` | tests selected through changed packages and their dependents (`--changed-base=<ref>` for another base than `origin/master`) |
| `cargo xtask test --no-run` | build every test binary without running tests |
| `cargo xtask warm` | build every test binary and both Clippy configurations without running tests |
| `cargo xtask scenarios` | end-to-end tests of the real daemon, CLI and MCP server |
| `cargo xtask bench ...` | corpus builds and interactive read measurements |
| `cargo xtask status`, `cargo xtask history ...` | what failed in this checkout's last run, and the recorded run history |
| `cargo fmt --all` | format |
| `nix flake check` | locked package, source, module and VM checks described in [development](docs/development.md) |
| `nix build` | the package |

Every command builds one dependency graph: the whole workspace with the Sinex
client and the fault-test features. Shared dependencies declare their
features once, in the root manifest's `[workspace.dependencies]`, and crates
add none of their own (dev-dependencies may add test-only ones), because
Cargo builds a crate again for each feature set it sees, and everything above
it. So Clippy, `check`, `test`, `scenarios`, `warm` and every focused run reuse
the same units. Selectors (`-p`, `--exclude`, `-E`, `--changed`) choose which
tests run, never what is built, and feature flags are refused. Tests run in
the primary-mode build; the standalone Clippy pass compiles the code that the
Sinex client replaces. Every xtask run is recorded in devkit's history
database.

Designed and not built yet: `cargo xtask compile` and `coverage`.

Pull requests and master run `cargo xtask check` on GitHub Actions; read the
result with `gh pr checks`. A pull request runs the tests its changes can
affect: devkit maps each changed file to its package and selects that package
and every package depending on it, or only the changed tests when a change is
confined to test functions. A change to the lockfile, a manifest, the
toolchain, the Cargo or AgentCTL configuration or any file outside the crates
(documentation included, which tests read) selects every test. Master runs
every test on each push. A host may queue these same commands through its own job runner; the
operator's host does so with AgentCTL, whose descriptor in `.agentctl/`
carries only host tuning. Use it where your environment provides it; the
commands and their results are the same either way.

The pre-commit hook checks whitespace and formatting of staged files in
seconds. Crate boundaries and everything else run in `cargo xtask check`.

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
pipeline are in `docs/architecture.md`. Measure with paired `cargo xtask bench`
runs (before and after, same host) and report hardware-independent figures
(bytes written per input byte, syncs per group, stage utilisation) alongside
the absolute rates. No test asserts elapsed time. Designs target the data and
the guarantees, not one machine; host tuning belongs in deployment
configuration.

## How work is done here

**Design first.** Before writing code, read the page in `docs/` that owns the
area, the pages it links, the current code of the area, and the
predecessor's implementation and tests for what you are building (Polylogue's
earlier Python implementation is in this repository's Git history). Write the
whole change from the design: complete features, built on the architecture's
abstractions, with the predecessor's behaviour and test cases ported.

**Edit in bulk; type-check as you go.** Change whole modules, crates and call
sites in one pass. Run `cargo xtask check --changed` (formatting, boundaries,
clippy and the tests the change can affect) after each layer you write; it
reports every type error at once. Run the wider tests once the change is
coherent.

**Review your own work before verifying it.** Read the whole diff once
against the task, the owning page, `docs/architecture.md` and the
predecessor's behaviour, and list everything wrong in that one pass: a
requirement not met, a mechanism duplicated, a predecessor left in place, a
caller not converted, a limit that refuses valid input, a test that cannot
fail, a name that does not say what the thing is, a case the design covers
and the code does not. Fix the whole list, then read only what the fixes
changed. Then verify. The author makes the work good; a separate reviewer is not the mechanism for
that. "Review should not look for a single wrong thing, it should strive to
list everything wrong. so that there is no forever loop of fixing and finding
new issue" (operator, 2026-09-29, `cc:9ec284a7-6a6a-486b-949f-d8bbdb22ae1e:16cb3a25-318f-455b-82b5-bf0f6f732e30`).

**Verify scoped to the change.** Commit first, so a result belongs to a
commit. While editing, `cargo xtask check --changed` runs formatting, clippy
and the tests the change can affect. A pull request is verified by GitHub
Actions, which runs `cargo xtask check` for its head (affected tests only);
every test runs again on master after the merge. Run a full local check only
to reproduce a failure CI reported. Before treating a failure as yours,
compare with the same test on current master (`gh run list -b master`).

**Finish what the task names.** A task states its done condition before work
starts, as behaviour on master and the tests that prove it. A defect inside
the area that stops that condition from holding is fixed in the same work.
Anything else found goes on a Bead with what was found, and the current work
lands without it.

**Git and pull requests.** Work on a branch from `origin/master`; every pull
request targets `master`, and stacked pull requests are not used. Commit with
plain messages whose body says why. Stage explicit paths. Land a feature as a
sequence of pull requests, each of which compiles, passes its tests and
leaves master coherent: complete for what it covers, with no stub to replace
later and no compatibility path between steps. Small related fixes share one
pull request, since each pull request costs a full CI run. Master requires
the `check` status on the pull request's head; branches need not be up to
date with master, so rebase only to resolve a conflict, taking master's names
and structure and reapplying your change. Merge with
`gh pr merge --auto --squash --match-head-commit <sha>`. Development speed
comes before a perfectly green master (operator, 2026-11-09): when two
separately green changes combine into a failing master, whoever's change
caused it fixes it next. Hooks stay enabled.

**PR descriptions** are plain prose: what changed and why, the design
decisions taken and their reasons, what was verified (commands and run IDs)
and what was not, and before and after figures for anything performance
related.

**Decide technical details yourself.** Low-level design choices are yours,
made from the principles here and the pages in `docs/`, and recorded on the
Bead and in the PR. "IDK how these are for me, I do not know ~anything about
these details" (operator, 2026-09-29,
`cc:9ec284a7-6a6a-486b-949f-d8bbdb22ae1e:b6a95d83-9b7f-4478-a74e-a0a3d3b46683`).
The operator decides product scope and intent, privacy, money (paid APIs,
purchased vectors) and anything published outward. Deploying, activating,
restarting, rebuilding an archive and the release cutover are yours: original
inputs stay where their sources keep them and an old archive is set aside, not
deleted. No stored format or archive is deployed in a sense that needs
compatibility or migration; target one current format. When an operator
question is open, take the conservative option, keep
working on the rest, and put the question in the report and on the Bead.

**A request to check is read-only.** When asked to inspect, analyse or
explain, report and recommend; change only what was asked for.

## Engineering values

These are the operator's standing positions, explained so they can be applied
to cases they do not name. References in the form `cc:<session>:<message>`
locate the operator's words in his session history.

**Invariants instead of repair.** Code does not produce broken state, so
there is nothing to repair. There are no repair, reconcile, reindex or
verify-and-fix commands and no scheduled checks. A crash leaves only states
the ordinary open path treats as normal: an uncommitted segment tail, a
derived store behind its saved position, a source position behind its
committed rows. When something is found broken, fix the code that produced
it; often the cause is code that no longer exists and no new invariant is
needed. "We want to engineer things well, to maintain appropiate invariants,
which make "break glass" solutions redundant because the only way these would
be necessary is a striaght-up bug, which if it happens we could just fix"
(2026-07-19, `cc:af12164b-a2fc-42cb-a548-22277c0875a2:60fc6414-24ab-4628-81dc-66aae9282d5d`).

**One mechanism per concept; a replacement removes its predecessor.** When
something is replaced, the old version, its callers, docs, tests and
everything that only served it go in the same change. There are no
compatibility shims, aliases, fallbacks or old-format readers for Polylogue's
own formats; a breaking change fixes every caller. Outside formats are
different: provider exports from any year are user data and stay supported.
"No compatibility shims means things should be done right, not that there
shoudl be no compatiblity and things coudl break" (2026-09-28,
`cc:9ec284a7-6a6a-486b-949f-d8bbdb22ae1e:e645e5ee-d29a-402e-bd4a-a58c32ad0c88`).

**Full features, built on a whole architecture.** Build the complete feature
the design describes, on abstractions that make the next features cheap,
rather than a bare-bones version to be replaced later. Polylogue is meant to
do more than its predecessor did, without the predecessor's machinery. "Why
build some sort of bare-bones features which will need to be replaced by
actual full-fledged ones anyway?" (2026-10-27,
`cc:ee52794c-8a9e-43b6-9fbc-9749c7e78e8f:3c3dd59f-4560-4fee-95d4-5c43b6c94ca5`).
When an area you are changing works but its design is wrong, redesign it as
part of the task; an area you are not changing gets a Bead.

**Use the predecessor's knowledge, not its mechanisms.** The Python
predecessor had a working core under layers of repair and bookkeeping. Port
its format knowledge, parsers' cases, capabilities and planned work; leave
its repair passes, census and debt ledgers, revision authority and audit
tiers behind. "Don't make the opposite mistake of mindlessly porting things
over just becasue they were in the old codenbase either -- to not port over
all these terrible mechanisms, corrections on top of corrections to broken
code" (2026-10-27,
`cc:ee52794c-8a9e-43b6-9fbc-9749c7e78e8f:3f159a64-8c63-4851-850b-0db265a6998f`).

**Small codebase, plain names.** Bulk is the main enemy: the predecessors
grew past half a million lines each through kludges on kludges. Prefer one
deep abstraction to several shallow ones, clean internal and external
interfaces, and no repeated logic. A name says what the thing is in ordinary
words, everywhere: types, functions, local variables, SQL, JSON keys, CLI
help, errors, comments, tests and docs. When an opaque term is found, replace
it throughout the codebase in one pass, so that nobody copies it from what is
already there. "The jargon is not supposed to be anywhere, period. including
stuff like local variable names etc." (2026-10-24,
`cc:ee52794c-8a9e-43b6-9fbc-9749c7e78e8f:e52f56c3-a9ec-45d8-b719-6b881ee22367`).

**Good design instead of guard machinery.** Quality comes from the
architecture and from the author's own review, not from build-enforced bans,
line budgets, word checks or required PR fields. A pattern that is absent
from the codebase does not reappear by itself. "These sorts of checks dont'
actually work." (2026-10-27,
`cc:ee52794c-8a9e-43b6-9fbc-9749c7e78e8f:deed5fe0-c9a3-49e3-8f27-c35efdc08301`).

**Valid input is never refused or truncated for size.** Page sizes are
defaults, transfer units continue, and memory is bounded by streaming and
read-ahead, not by caps on what the user may store or ask for. A query that
runs past its deadline names a narrower query and the same command with the
deadline lifted. "Such arbitrary limits are themselves straight-up bugs."
(2026-09-27,
`cc:62be45f2-6c40-49ce-a88e-bc566948628c:4cc20d7c-9e62-4b7a-9550-7dd97f9d48e9`).

**Tests that are better than the predecessor's.** Every specified behaviour
in the pages in `docs/` has a test
that fails when the production path breaks. The valuable kinds: independent
oracles (goldens written from format knowledge, never re-blessed from parser
output), tests over the Morrow archive, crash tests at real durability
boundaries, and property tests at serialization boundaries. Port the
predecessor's cases as cases (inputs and expected outcomes), then fill the
gaps both suites had. A test that restates a helper, asserts a mock's value,
checks what the type system already prevents, or matches SQL substrings where
rows could be compared adds cost without coverage. A test asserts behaviour
that exists, never that a name, variant or format stays absent: a removal is
verified by the build and by tests of the replacement's behaviour. "We want to cover various
aspects of the projects, but we don't want to have tests which are
counterproductive or low-value just to have more tests." (2026-10-27,
`cc:ee52794c-8a9e-43b6-9fbc-9749c7e78e8f:bbb105d6-16f1-418e-b04a-0c5f105832df`).

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
Public docs state design intent in neutral language and cite rulings by date
and locator, never quoting the operator's private sessions; PR bodies, review
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
Merge the common pull request first. Polylogue also pins Sinex by revision
(`crates/sinex/Cargo.toml`), and that Sinex revision's own common dependencies
must resolve against the same common commit, so the order is: common merges,
Sinex adopts that common commit, then one Polylogue pull request moves the
Sinex revision and the common lock together and converts the callers.
