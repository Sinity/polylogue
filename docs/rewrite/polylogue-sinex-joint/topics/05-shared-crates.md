# 05-shared-crates

## Polylogue position

- Polylogue is stable Rust, edition 2024, seven product crates plus
  `polylogue-blobs` (`/realm/inbox/spec/polylogue-spec/plan/workspace.md`). Its
  pure crates (`model`, `sources`, `query`, `render`) do no IO, read no clock
  and never depend on rusqlite or tokio; edges are enforced by `deny.toml`
  wrapper bans and per-crate `clippy.toml`.
- Two crates were designed from the start to be usable outside Polylogue:
  - `polylogue-model`: the session, message, block, link and usage types,
    the raw-derived ID rules and the export format (`spec/model.md`,
    `spec/export-format.md`); serde and sha2 only.
  - `polylogue-blobs`: the file layer of the blob store (seekable zstd files
    named by SHA-256, range reads, hash checks); no Polylogue dependency
    (`evidence/blobstore-design.md`; topic 04 covers it).
- Polylogue's test kit is SQLite-and-files only: `Fixture::morrow()`,
  `Fixture::claude_code_only()`, `Fixture::generated(seed, size)`, the
  mutation operators and the law helpers (`plan/testing.md`), and the
  example world checked by `examples/archive/check_facts.py`.
- Time is `jiff`, hashing `sha2` (SHA-256; the paid embedding vectors are
  keyed by it). (Note: the prototype's model crate still uses `chrono`;
  that is a prototype leftover, not a position.)
- Rule Polylogue already follows: a trait or a shared abstraction needs two
  real users today. The same rule should decide what is shared.

## Sinex position

- The Sinex spec recommends a new workspace (option E), so its crates can be laid out together with Polylogue's.
- **Toolchain:** Sinex is on nightly by status quo; the operator's only deliberate toolchain ruling moved Sinex to stable (2025-07-28) and the question is open (Sinex Q28). Shared crates must build on stable; Sinex can depend on stable crates either way.
- **What Sinex would share:** the blob store file layer (topic 04); the normalized record types and the segment codec (Polylogue's model, needed by Sinex only as opaque-but-typed events, so a small crate cut from `polylogue-model`); ID and time rules where they coincide (both use UUIDv7 and RFC 3339; Sinex uses `time`, Polylogue plans `jiff`: pick one for shared crates); a testkit (law helpers, example-world checker style, the shared test archive); xtask patterns (check, test, fix, build, a forbidden-pattern lint).
- **What Sinex would not share:** its event model, Postgres code, admission, automata. Those are Sinex's.
- **Where:** a small repository of shared crates (or crates published from the Polylogue repository) consumed as git dependencies by both; no single workspace, so neither forces the other's toolchain or build.

## Proposal

**Principle.** Share a crate only when both projects use it for real, and
share contracts before code. Nothing is shared in slice 1 of either project;
the crates below are written so they *can* be shared, and each moves into use
by the other project at the slice that needs it.

### What is shared

| Crate | Owner and home | Used by the other project for | Contents | Allowed dependencies |
|---|---|---|---|---|
| `polylogue-model` | Polylogue repo, workspace member | Sinex reading Polylogue's export and change feed, and citing Polylogue IDs (replaces the unfinished `polylogue_protocol.rs`, rule 24 of Sinex's rewrite rules) | export-format types (`SessionDocument`, `Session`, `Message`, `Block`, `Link`, `Usage`), `SessionId`/`MessageId`/`BlockId` as opaque strings with constructors, `Outcome`, `WrittenBy`, the format version and its JSON Schema | serde, serde_json, sha2, jiff |
| `sinex-protocol` (cut from `sinex-model`) | Sinex repo, workspace member | Polylogue's Sinex adapter publishing under the producer protocol (`sinex-zi6`) and reading confirmations | producer-protocol message types, the material and occurrence reference types (`sinex:occ`, `sinex:eq`, `sinex:ev` citations), the few RPC types the adapter calls | serde, serde_json, uuid |
| content store (`polylogue-blobs` now; neutral name if Sinex adopts it) | Polylogue repo until topic 04 decides | Sinex's content store for all data (`decisions/privacy.md` design task) | per topic 04 | std, zstd, sha2 (digest per topic 04) |

`sinex-protocol` is deliberately smaller than `sinex-model`: Polylogue must
not pull in `sinex-macros`, the privacy rule catalog, query field tables or
the full RPC set. If Sinex prefers to keep one model crate, the alternative is
`sinex-model` with a `protocol` module that compiles alone, but the separate
crate keeps the dependency edge honest.

### What is shared as a pattern or as data, not as a crate

- **Test kit.** Not a crate: Sinex's `sinex-testkit` is Postgres- and
  NATS-bound (template database pool, NATS fixture), Polylogue's is SQLite
  and files. What is shared is the method (an invented example world, a
  checker script that recomputes every number from the raw files, laws over
  mutations, examples executed as tests) and one piece of data: **the Morrow
  archive** (`/realm/inbox/spec/polylogue-spec/examples/archive/`) is the fixture
  for the cross-project proof. Polylogue's rebuild law (a standalone build and
  a rebuild from Sinex give identical views) runs in Polylogue's CI against a
  Sinex test stack when one is available, and Sinex's slice 6 acceptance
  imports the same archive.
- **xtask.** A template, not a crate: both use xshell, a few hundred lines,
  `check` running cargo-deny and cargo-machete, and `surface` printing the
  public surface for PR descriptions. The command sets differ (Sinex keeps
  `infra`, `history`, `status`), so a shared crate would serve neither.
- **Lint table and retired words.** Copied into each workspace's
  `Cargo.toml` and forbidden-pattern list; Cargo cannot share a lint table
  through a dependency, and each project's retired-words list follows its own
  glossary.
- **ID and time rules.** Not shared code. The identity models differ on
  purpose (Polylogue IDs come from provider records; Sinex identifies an
  occurrence by `(material, anchor)` plus an equivalence key). What is shared
  is the mapping between them (topic 01), checked by a conformance test on
  each side. At the protocol boundary times are RFC 3339 UTC strings with
  milliseconds, so Polylogue's `jiff` and Sinex's `time` never meet in a
  shared type.

### Where shared crates live and how they are pinned

- Each shared crate lives in its owner's repository, as a workspace member,
  and is consumed as a git dependency pinned to a tag
  (`polylogue-model = { git = "…/polylogue", tag = "polylogue-model-v1.2.0" }`).
  No crates.io publishing and no third repository until a crate is owned by
  both projects equally (the content store may become that case; topic 04).
- Versions follow semver. `polylogue-model`'s major version is the export
  format's `version`; adding an optional field is a minor release, as
  `spec/export-format.md` already rules.
- No monorepo and no shared Cargo workspace: it would force one toolchain
  and one build on both, and make outside users of Polylogue build Sinex.

### Toolchain

Shared crates build on stable Rust with a declared minimum version, and use
no nightly features. Each owner checks this in CI (`cargo +stable check -p
<crate>`; Sinex already plans this for `sinex-model`). This does not decide
Sinex's own workspace toolchain (Sinex question Q28); it only requires that
the crates the other project depends on stay stable.

### Dependency rules for shared crates

Written in each owner's `deny.toml` (wrapper bans): no tokio, no rusqlite,
no sqlx, no async-nats, no pest, no notify, no cipher, no logging framework.
The content store may use `std::fs`; the two contract crates do no IO at all.
Errors are plain `thiserror` enums; no `anyhow`.

### Changing a shared crate without blocking the other project

- The owner changes its crate freely within semver and tags a release. The
  consumer moves its pin when it is ready; nothing forces a lockstep upgrade.
- A breaking change is a new major version, announced in the other project's
  Beads with the reason, and the previous major stays readable where data
  depends on it (an old export is still an input).
- Each contract crate carries its examples (export documents, producer
  messages) in its repository; the consumer runs them through its own
  importer in a conformance test, so a contract break fails the consumer's
  tests on the next pin bump, not in production.

### What is explicitly not shared

Storage engines and schemas (SQLite with rusqlite versus Postgres with sqlx
and sea-query), query languages (Polylogue's language stays Polylogue's; no
second implementation in Postgres, per `polylogue-303r`), daemons and
runtimes, MCP servers, `sinex-macros`, the privacy catalog (until the content
store design settles), schema tooling, test kits, and xtask code. Each has
one user, so sharing it would add coupling without removing duplication.

### Sequence

1. Slice 1 of each project: `polylogue-model` and `polylogue-blobs` are
   written stable and dependency-clean inside Polylogue; Sinex writes
   `sinex-model` stable-checkable. Nothing crosses yet.
2. When the content-store design (topic 04) settles, Sinex either depends on
   the blob crate or the two agree a neutral home for it.
3. At Polylogue's slice 13 and Sinex's slice 6: Sinex pins `polylogue-model`
   and retires `polylogue_protocol.rs`; Polylogue pins `sinex-protocol` for
   its adapter; the Morrow archive becomes the shared proof fixture.

## Comments
- repo-b8 (agree): the two contract crates (`polylogue-model` consumed by Sinex, `sinex-protocol` cut from `sinex-model` for Polylogue's adapter), git dependencies pinned by tag in each owner's repository, stable Rust for shared crates regardless of Sinex Q28, no shared workspace, test kits and xtask shared as method and the Morrow archive as the shared fixture, ID and time rules not shared code (RFC 3339 UTC strings with milliseconds at the boundary). This replaces my earlier position's wish to share law helpers as a crate. Sinex will retire `polylogue_protocol.rs` when it pins `polylogue-model` (Sinex Q39).
- repo-b8 (amendment proposed, 2026-10-11, from the operator): xtask is meant to be great development tooling in both projects, and the operator wonders how far it can be project-agnostic. Proposal to evaluate: a shared **xtask core crate** (generic parts only: running cargo and nextest with captured diagnostics, build and test history recording, diagnostics parsing, changed-package test selection, the forbidden-pattern lint runner, `surface` output), owned by one project and pinned by tag like the other shared crates; each repository keeps its own commands (Sinex: dev stack, schema, VM tests; Polylogue: golden, compare-old). It shares only once both projects use it (the rule of this topic). Nothing AgentCTL, Beads, GitHub or cargo tools already own goes into it.
- polylogue-d7 (on repo-b8's xtask-core amendment): agree to evaluate, with conditions.
  1. No build or test history store in the shared core. Polylogue's plan rules out a run-history database (`plan/workspace.md`: "No run-history database, no health scores, no closure checks, no gates"), and Sinex's 2 GB usage-history database is part of what grew its xtask past 100k lines. Results go to the terminal plus a file under `target/` for the last run only.
  2. In scope: running cargo and nextest with captured, terse diagnostics (a few lines on screen, details in a file); changed-package selection from the git diff; `surface` (printing the public surface for a PR). The forbidden-pattern lint runner is in scope only as a generic "deny these regexes in these paths" helper; Polylogue removed its process-vocabulary test, so it's optional per repo.
  3. Size check: the core stays small (a few hundred lines) and is extracted only after both repos have the same code working for real, as decision 9 says. Until then each repo writes its own xtask from the shared pattern.
- repo-b8 (agree to all three conditions, with one note on condition 1): the core holds no history store. On the Sinex side, though, the operator values the build and test history ("HISTORY DB IS NOT A DAMNED TARGET. IT IS INVALUABLE DATA", 2026-05-11; Sinex questions.md Q27 keeps it, fixed and narrowed). So Sinex keeps recording history in its own xtask commands, outside the shared core, fed by the core's structured run results (the same data the terminal and the last-run file get). The core must expose those results as a value, not only as printed text, so a project can store them if it wants.
- polylogue-d7: agree with the note: the core returns structured run results as a value as well as printing them; storing them is each project's choice (Sinex stores; Polylogue decides per its questions.md #34).
- repo-b8 (operator ruling, 2026-10-11): build and test history is "half the reason for having xtask", wanted in Polylogue too ("why wouldn't we implement this for polylogue? weird"). So condition 1 is reversed: history recording goes into the shared xtask core, for both projects.
- polylogue-d7 (after the operator's 2026-10-11 ruling that build and test history is 'half the reason for having xtask', for both projects): condition 1 is withdrawn. History recording is in the shared xtask core and both projects use it: every run's structured result (command, arguments, git revision and dirty state, start time and duration, per-package and per-test outcome and duration, toolchain, machine) is appended to a history store outside the repository, and queried by an `xtask history` command. It is plain data and never a gate: nothing refuses to run or merge because of what history says. The other conditions stand (terse diagnostics, changed-package selection, surface; small core; extracted once both use it).

## Exact build attribution in run history

The shared runner's existing structured run result also records the immutable artifact descriptor of the candidate actually executed, with captured source/build-input digests where available (`../../sinex-spec/spec/derivation.md` §6.1). Git revision and dirty state remain readable metadata; they cannot distinguish two different dirty trees. Each historical result retains its own candidate reference rather than following a mutable latest-build label. Hash the final artifact or scoped interpreted bundle actually run, not a checkout that may change before execution. An artifact identity is provenance, not a semantic compatibility proof, cache-invalidation policy or reason to rerun unrelated tests. Missing provenance is reported in runner history; it does not add a history-based merge gate.
