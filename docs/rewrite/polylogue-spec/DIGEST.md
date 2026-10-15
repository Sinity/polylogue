# Polylogue rewrite: the 20-minute digest

If you read only one file, read this one. Every claim links to where it is
specified in full. For how the parts fit, see
[spec/architecture.md](spec/architecture.md).

## What Polylogue will be

Every AI conversation and agent session your tools left behind, kept on
your machine exactly as they wrote it, which you and your agents can search,
read, measure and build on; where a record is missing, it says so
([purpose](spec/purpose.md)).

- **It keeps everything.** It stores its own copy of every session file,
  export, capture and hook event, and every revision of each, because
  providers delete and rewrite their files ([principles #5](spec/principles.md),
  [raw retention](decisions/raw-retention.md)).
- **It is always current without being asked.** One daemon runs one loop:
  collect, store, parse, derive, wait. It finds its work by comparing what
  exists with what should exist. There are no repair commands, no rebuild
  command and no work queue ([convergence](spec/convergence.md)).
- **One query language serves every surface**: the CLI, MCP for agents, the
  web reader and the Python client ([query](spec/query.md), [surfaces](spec/surfaces/README.md)).
- **Agents are primary users, and history comes to them.** Agents rarely
  look things up on their own, so harness hooks inject briefings: recent
  work, decisions and open questions when a session starts; what changed on
  resume; your own earlier prompts after a compaction; and a full takeover
  briefing when you paste a session ID or link. Automatic injection is on
  probation, per your May "write-passive, read-pull" decision: it stays
  only if the measured comparison shows agents do better with it. What
  each briefing gave and left out is recorded ([MCP and hooks](spec/surfaces/mcp-and-hooks.md), [user-data](spec/user-data.md)).
- **It knows what is running now.** Hook lines carry each harness's process
  identity and job variables, and the daemon records process trees as they
  change, so every session has a live state (working, waiting on you, idle,
  ended, stale), a task when the evidence names one, and peers in the same
  repository, all reproducible from stored facts ([model](spec/model.md)).
- **It counts, but does not guess meaning.** Tokens, cost, time, tool
  outcomes and failures are computed from the records. Questions about
  meaning go to an agent session that reads the selected sessions
  (Polylogue never calls a model itself), and its answers are stored as
  attributed notes ([analysis](spec/analysis.md)).
- **Agent annotation is constrained, not free text.** A claim can carry a
  check, a query the archive answers as holds, fails or unknown; agents
  annotate in passes under a rubric that fixes what they may write; each
  pass is judged on a blind sample and withdrawn or revised as a whole
  ([user-data](spec/user-data.md); Sinex does the same, joint #16). The main
  demonstration is your own archive: the story of each project told from
  stored facts and checked notes ([workflows](spec/workflows.md) §17–21).
- **Later, a shared working surface**: pages and investigations that you and
  your agents edit together, with pings that resume a specific agent
  ([user-data, pages](spec/user-data.md), [slices](plan/slices.md)).

## What changes from the Python version

| Python today | Rust rewrite |
|---|---|
| 640k product lines, 786k test lines | tens of thousands of lines; tests built from laws and examples |
| six SQLite tiers, a reference seal, mutation ceremony | three files, `archive.db`, `index.db`, `embeddings.db`, plus blobs ([storage](spec/storage/README.md)) |
| advisory schema validation using about half the CPU | the typed parse is the validation; unknown fields are counted ([decision](decisions/schema-validation.md)) |
| five routes into the index, two derivation systems | one loop, one way to derive ([convergence](spec/convergence.md)) |
| a 678-file fingerprint that forces rebuilds | a version number per parser and per derived product |
| a child session stores only its new tail | full message lists, copied messages marked, text stored once ([lineage](decisions/lineage.md)) |
| hashed legacy IDs | IDs computed from the provider's raw records, so parser changes never move them ([model](spec/model.md)) |
| verification bureaucracy | examples in the spec run as tests, 141 laws, one comparison against the old parsers ([testing](plan/testing.md)) |
| bugs found and re-found | 87 rules (R1–R92), one per past bug class, each with a concrete enforcement ([rewrite rules](plan/rewrite-rules.md)) |

## How the package is layered

From what Polylogue is for down to code, each layer checked by a flake check
so the layers cannot drift apart:

1. **Vision** (`vision/uses.toml`): 127 uses of the archive, seeded from your
   own chats and notes and expanded well beyond them, each marked intended,
   plausible or speculative, at four levels of rigor: a catalog line, an
   exact query and its output, a query that runs on the example archive, a
   prototype on your real data. Uses the substrate cannot serve are recorded
   as gaps, each fixed in the spec, set aside, or sent to the joint catalog.
2. **Spec** (`spec/`): behavior, in plain language.
3. **Outline** (`outline/components.toml`): the whole finished system as 63
   components, each with what it owns, what it calls, the slice that builds
   it and the vision uses it serves; every feature, law and table placed
   exactly once.
4. **Skeleton** (`skeleton/`): the whole codebase as compiling Rust
   signatures with empty bodies, so "the parts fit" is a compiler fact, with
   the spec's types checked against it.
5. **Slices and prototypes** (`plan/`, `prototype/`): the build order, and
   working code for the hard parts, run on your real data.
6. **What the Python version had** (`plan/today-inventory.toml`): its 350
   commands, tools, tables, file kinds and services, each named by the spec
   or given a disposition.

Where Polylogue meets Sinex, the joint folder holds the same layers:
decisions, a joint vision catalog, a joint outline of the interfaces, a
joint glossary and the query laws both projects keep.

## Where things stand

See the Status paragraph at the top of [README.md](README.md).

## What has been checked by running it

- **Prototypes** (`prototype/`, the flake checks): the per-record parse
  framework; Claude Code and Codex parsers, with the slice-1 counts, append
  equivalence at 3,234 split points, and revision merging; ChatGPT, Gemini
  CLI, Hermes and Grok CLI parsers; Antigravity's steps decoded with the
  schema embedded in its own language server; four search
  plans returning identical pages; the daemon socket (a warm search in about
  0.1 ms, a 0.7 ms client); directory briefings in 4–12 ms, takeovers in 1–9 ms, and resume
  and self briefings up to 45 ms against a 150 ms budget.
- **Your real data, counts only**: every local source parsed or surveyed
  against its spec section, with about 65 rules corrected from what the data
  (and, for Hermes, the harness's own source) showed (`evidence/*-real-*.md`); index-only builds of 20 GB of Claude Code in
  8.3 minutes and 53 GB of Codex in 15.8 (`evidence/ingest-benchmark.md`).
  These exclude collecting and storing bytes. The whole loop has not met
  the full-corpus time and memory targets (`spec/behavior.md`, speed targets;
  `vision/synthesis.md`, what has not been shown).
- **The convergence loop on your real Claude Code sessions** (collector,
  revisions, blob store, parse, index): the corpus stores 23.4 GB as
  3.5 GB; an idle pass does nothing in 0.2 s; a 324 MB live append is parsed
  from its carried state in 3.4 s; an index rebuilt from the archive equals
  the incremental one row for row (`evidence/real-converge.md`).
- **1,439 real Codex rewrites** stored as deltas at 2.2% of a plain copy
  overall, every one read back exactly. Raw record counts changed; they
  do not establish lost messages. Retained revisions support version-aware
  payload conservation checks
  (`evidence/codex-rewrite-deltas.md`).
- **With Sinex**: a backed-mode contract test (publish, suppression,
  excision orders) and the shared blob crate with sealing, both as flake
  checks (`polylogue-sinex-joint/prototype/`).

## The biggest decisions

All are decided; the [decisions index](decisions/README.md) says who made
each. You may overrule any.

1. **Every input and every revision is kept**: appends as new pieces,
   rewrites as deltas, databases as page patches. Codex's rewrites of its
   own files dropped records, so old revisions can be the only copy.
2. **Three database files** by one rule: what cannot be recomputed, what
   can, and what costs money to recompute ([storage files](decisions/storage-files.md)).
3. **Everything goes through the daemon**: the CLI, MCP servers and hooks
   send requests over one local socket, and the daemon serves the web reader
   itself over local HTTP with the same request shape; the daemon answers from warm
   state (vectors, query plans, recent results) and is the only writer. One
   small request shape, no operation registry; no standalone mode
   ([process](decisions/process.md)).
4. **Lineage**: three link kinds (subagent, continuation, reference), each
   saying how it came about; copied messages are marked, never counted twice.
5. **One model response is one message**, which fixes a 2.5x overcount of
   Claude Code costs ([decision](decisions/one-message-per-response.md)).
6. **The old archive's data goes back where it belongs**: captures and hook
   spools are copied back into their spool directories, AI Studio is
   re-collected from Drive, Codex's pre-rewrite files become earlier
   revisions ([migration](plan/migration.md)).
7. **Fresh workspace in the same repository, after pre-rewrite preservation** ([repository](decisions/repository.md)).
8. **Optionally, and for you, Polylogue runs on Sinex.** Polylogue works
   fully with Sinex off, the default. Native `primary` follows finite fixed-basis shadow qualification and a real-corpus
   rebuild proof; the blob store's file layer and the model crate are
   shared ([sinex](decisions/sinex.md)).
9. **Agents can do everything you can, except three acts that cannot be
   undone**: excising, unmasking a credential, and switching on a setting
   that sends archive data to an outside service. Everything else applies
   directly, attributed and revisable: notes, settings, signal rules
   ([user-data](spec/user-data.md)).
10. **Merging one conversation from several routes** follows a fixed order
    of faithfulness, never "latest wins" ([merging](decisions/merging.md)).

## Questions for you

Three remain ([questions.md](questions.md)): whether daily use comes before
the workbench; whether you want to converse with agents or direct them,
which shapes the workbench (your own open question from May); and whether
Polylogue backs an agent's memory tool. The other 44 were settled from the definitions, your rulings and the
evidence; the same file says where each answer lives.

## What to look at first

1. [spec/workflows.md](spec/workflows.md): real uses with commands and
   outputs. If one feels wrong, the design is wrong.
2. [spec/architecture.md](spec/architecture.md): the system on one page and
   one session traced end to end.
3. [plan/slice-01.md](plan/slice-01.md): what you will run first (Claude
   Code end to end) and how it is accepted.
4. [spec/surfaces/cli.md](spec/surfaces/cli.md): the exact CLI output you
   would see every day.

Every example runs against one small invented world, the Morrow archive
([FACTS](examples/archive/FACTS.md)); slice 1 uses its Claude Code part plus
a lineage set of nine sessions (nested and sibling subagents, a fork, a
team, a missing parent, two folders, a sidechain). A checker script recomputes every number.

## How to correct it

Write your correction anywhere: chat, a line in the file, or a comment
starting `> OP:` under the paragraph. A later deliberate ruling overrides an
earlier one ([README, how evidence is weighed](README.md)). Progress is
logged in [LOG.md](LOG.md).
