# Principles

These rules decide design questions the rest of the spec does not answer
explicitly. Each one comes from the operator's repeated statements; the
evidence is in `evidence/intent-*.md`.

## What Polylogue is

1. **A trustworthy, fast archive of all AI work.** Every chat and agent
   session the operator's tools left behind, from every provider they use;
   nothing silently lost, nothing invented, and every gap (a provider with no
   export, an unreadable location, a period not collected) reported rather
   than hidden. Ordinary questions are answered in interactive time; a
   question too large for one read's time limit is answered in steps: the
   timeout carries a narrower query with the same meaning (`query.md`, four
   bounds).
2. **Shared memory between the operator and their agents.** Coding agents are
   a primary user: they recall past work, decisions and context through the
   CLI and MCP. A human reading the archive is the other primary user.
3. **A substrate, not an analyst.** Polylogue selects, structures, counts and
   measures. It hands models clean context and gives them structure for
   disciplined analysis. It never guesses meaning from prose with code
   (keyword classifiers, regex "intent" detection and similar are out).
4. **A library inside one daemon.** The core is a set of pure Rust
   libraries (model, sources, query, render) that the daemon is built on;
   `polylogue-model` is also shared as types and the export codec. Every
   surface (CLI, MCP, web reader, hooks) is a thin client of the daemon's
   socket; no other program opens the archive files.

## How data flows

5. **Every input is kept.** Provider files can be
   rotated or auto-deleted (Claude Code removes old transcripts by default),
   exports get lost, browser captures exist nowhere else. So every input's
   exact bytes go into Polylogue's content-addressed blob store when first
   seen, and again whenever they change, with Sinex `off`;
   in `primary` the one kept copy lives in Sinex's content store, and
   Polylogue holds only bytes Sinex has not yet confirmed (joint decision
   27). Either way the archive must stay whole even if every original file
   is gone. Provider locations are where Polylogue
   *reads*; the blob store is what it *keeps*. Retention is
   forever; only excision removes content.
   Several routes to the same conversation (live file, hooks, captures,
   exports) are kept as separate witnesses and merged in derivation, never
   avoided at capture (`decisions/merging.md`).
6. **Everything else is derived and rebuildable.** The index, search, profiles
   and statistics are a function of the sources plus the operator's own notes
   and tags. Deleting derived state and starting Polylogue rebuilds it by the
   same code path as any other day. There is one way to bring derived state up
   to date: compare what should exist with what does, and fill the gap.
7. **Nothing is ever broken, at worst not yet up to date.** There are no
   repair commands and no maintenance procedures. If a bad state can appear,
   the code that produces it is wrong and gets fixed.
8. **Strict input.** Polylogue reads each provider's files from their standard
   location and knows exactly what each file is. Input that is not what it
   should be is never guessed at and never dropped: it is stored, counted and
   shown (an unknown field in `attrs`, an unrecognized record or an
   unparseable time counted in `status`), so a provider's change is visible
   the day it happens. A missing value is never filled with an invented
   default.
9. **No "unknown" where the fact can be known.** If a value is derivable from
   the data (a tool call's outcome, a session's start time), derive it. Store
   "unknown" only when the data really does not say, and then say why.
10. **Lineage without duplication.** Forks, resumes and subagents that replay
    earlier messages do not create duplicate copies of those messages in
    counts, search results or costs.

## How it is built

11. **One mechanism per job.** One ingest route, one way to update derived
    data, one query language behind every surface. No special modes, no
    "lightweight" variants, no parallel old and new paths.
12. **Replacement deletes the predecessor**, in the same change. No
    compatibility shims, aliases or legacy paths. Old provider file formats
    are not legacy: the archive contains them and their parsers stay.
13. **Correct by construction.** Prefer types that make a wrong state
    unrepresentable, then plain runtime checks at the boundary. The product
    never gets a checker, census, receipt or proof object beside code that
    could instead be made correct. (The spec's own consistency checks are
    tooling for the documents, not product.)
14. **No pointless knobs.** Provider locations are standard, Polylogue's
    own state lives at fixed XDG paths, and one command imports an export.
    A setting exists only where the operator's world or deployment genuinely
    differs, never as an alternative way of doing the same job ("avoiding
    pointless knobs… but there might be knobs that should be there… nixos
    modules. and runtime, probably in db", operator, 2026-06-18). Static ones
    come from the NixOS module; runtime ones are data: per-machine settings in `archive.db` (`polylogue
    setting`: time zone, subscription plan, embedding provider), a local
    price override at `$XDG_CONFIG_HOME/polylogue/prices.toml`, and the
    Sinex link at `$XDG_CONFIG_HOME/polylogue/sinex.toml` (mode and
    address; kept outside `archive.db` so a rebuild from Sinex can start
    empty), and the commands and targets signals may use at
    `$XDG_CONFIG_HOME/polylogue/signals.toml`, written by the NixOS module. Credentials are state: one 0600 file per service under
    `$XDG_STATE_HOME/polylogue/credentials/`, written by `polylogue auth`, never
    read from the environment and never inside the archive directory.
15. **No arbitrary limits, and no absurd engineering to avoid them.** Valid
    input is never refused or truncated by a made-up cap. Memory use should be
    reasonable for a desktop machine, not minimal; ordinary files (up to
    64 MiB) are read whole, and larger ones are streamed (`behavior.md`,
    memory).
16. **Speed is a requirement, not an optimization.** Targets are set from what
    the hardware can do (parsing JSON and writing SQLite), not from what the
    current code happens to achieve. See `spec/behavior.md`.
17. **Plain words.** Names, docs, logs and messages use ordinary programming
    terms. A coined term needs a reason and a glossary line.

## How it is verified

18. **Tests exercise the real product.** End-to-end runs over provider files,
    through the same code the daemon uses. Behavior is checked by outcomes
    (rows, query results, output), not by internal call sequences.
19. **Few, dense, general tests.** Property laws, golden examples generated
    from real data shapes, and whole-archive equivalence checks replace piles
    of hand-written special cases. A test for a bug that can no longer occur
    is deleted.
20. **Test data comes from the real archive's shapes.** Until schema
    inference exists (slice 8), tests run on the hand-built Morrow world,
    whose files copy real record shapes, and prototypes are checked on the
    operator's real data with counts only. From slice 8, synthetic inputs are
    generated from schemas inferred from the real files, so they match the
    real distribution; fixed fixtures stay only where generation cannot
    work.
21. **The operator can see and steer.** Status, progress and failures are
    stated concretely enough that the operator can tell, without an agent in
    between, whether what should be happening is happening.
22. **Links and identity come only from evidence.** Two records are joined
    (a session to its parent, a prompt to a hook event, a message to a paste)
    only by evidence in the data: a shared identifier, a provider's own
    reference, or the operator's statement. Never by closeness in time or
    similar text alone; a time-window match is recorded as such, never
    presented as provider evidence (rule R76).
23. **Data leaves the machine only to services the operator configured**
    (an embedding provider, Sinex, a model the operator asked to read
    something), and credential-shaped text never leaves in what Polylogue
    assembles for a model, including CLI output an agent reads (joint
    decision 31).
