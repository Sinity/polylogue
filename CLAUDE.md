# Polylogue

Polylogue is a local, single-writer archive for AI coding/chat sessions —
Claude (web + Code), ChatGPT, Codex, Gemini/Drive, Antigravity, Hermes — that
ingests heterogeneous exports and live captures into a split SQLite file set,
derives rich read models, and serves them through a query-first CLI, an MCP
server, a Python API, and an HTTP daemon. Pure Python.

This file carries repository semantics only. Task authority is external
(`bd` redirects outside the checkout); generic workspace/job/publication
mechanics are environment-level concerns, not Polylogue's.

## Public boundary

Treat tracked content, commits, CI logs, and PR/review text as public. Never
commit operator archives, transcripts, private exports, local databases,
receipts, or scratch state; tests use neutral synthetic fixtures. Review the
complete staged diff before publication.

## Orientation

```text
sources/ ─detect→ pipeline/ ─hash+write→ storage/{6 tiers} ─materialize→ insights/
                                              │                              │
                            surfaces: cli/  mcp/  api/  daemon/  ─read-through─┘
                            verification:   devtools/  tests/  schemas/
```

New semantics go into the substrate (`storage`/`insights`) or product layer
first; surfaces adapt through `insights`/`operations`/`api`. Surface→substrate
imports are a ratchet enforced by `devtools gate layering` (baseline may
shrink, never grow); substrate→surface imports are forbidden outright.

## Identity and content model (know this cold)

Identity is computed, never stored redundantly (SQLite generated columns):

- `sessions.session_id = origin || ':' || native_id`
- `messages.message_id = session_id || ':' || COALESCE(native_id, position||'.'||variant_index)`
- `blocks.block_id = message_id || ':' || position`

Sessions → messages → blocks, all `STRICT`. Load-bearing semantics:

- `messages.material_origin` records who or what authored the content, which roles can't express —
  what makes honest cost/user-word accounting possible.
- `blocks.tool_outcome` is the canonical structural outcome. `unknown` means
  the parser retained a deliberate unknown-outcome reason; it is never treated
  as success. Legacy `tool_result_is_error` / `tool_result_exit_code` remain
  nullable compatibility fields.
- `actions` is a VIEW joining tool_use ↔ tool_result blocks by `tool_id`.
- FTS5 is contentless over `blocks.search_text`, trigger-maintained,
  `unicode61` (no porter stemmer in this build).
- Enum/`Literal`-backed CHECK constraints are generated from Python types
  where wired (`check`/`literal_check` in `archive_tiers/common.py`).

**Lineage**: forks/resumes/subagents/compaction physically replay the parent's
prefix; the writer stores only the child's divergent tail +
`branch_point_message_id` + inheritance mode; reads recompose.
`branch_point_message_id` is deliberately not an FK (parent full-replace must
not null it). `session_links` is also the topology-edge table, persisting
parser-asserted parent references resolved on each save through
`write_parsed_session_to_archive` — the single choke point shared by live
ingest and full replay/reindex.

## Six storage tiers (durability is the axis)

| Tier | durability | holds |
| --- | --- | --- |
| `source.db` | durable | raw acquired bytes, artifact taxonomy, blob/GC substrate, hook events, sidecars |
| `index.db` | rebuildable | parsed tree, FTS, links, costs, materialized insights |
| `embeddings.db` | rebuildable | vectors, meta, status |
| `user.db` | durable, irreplaceable | unified `assertions`, settings, annotation schemas/provenance |
| `audit.db` | durable, append-only | previews, authorizations, attempts, continuity |
| `ops.db` | disposable | cursors, attempts, convergence debt, daemon telemetry |

Never use rebuildable state as authority for durable mutation; never strand
existing durable archives with a schema shortcut. Archive writes are
idempotent by content hash (SHA-256 over NFC-normalized payload, excluding
user metadata — tagging never re-imports).

**Schema regimes**: durable tiers evolve by additive numbered migrations under
`storage/sqlite/migrations/{source,user,audit}/` behind a verified backup;
derived tiers declare a lifecycle delta class per version bump
(`storage/sqlite/lifecycle.py`) — only `SEMANTIC_REPARSE` deltas force a full
rebuild, and rebuild runs through the production daemon route (there is no
separate manual rebuild mechanism). Classify every schema change before
editing: metadata-only, index-only, additive-derived, additive-durable, or
semantic-reparse.

## Provider vs Origin vs Source

`Origin` is the public source-origin token on query surfaces and read payloads
(**public filters use `origin`**). `Provider` is the older provider-wire token
— legitimate at raw acquisition/parser/schema boundaries, a leak on public
surfaces. `Source` carries richer acquisition identity. The GEMINI+DRIVE →
AISTUDIO_DRIVE mapping is non-injective: never reverse an Origin into a
guessed Provider. Full table: `docs/provider-origin-identity.md`.

Detection (`sources/dispatch.py`) is shape-based in tightness order; insert new
detectors at the tightness they deserve or an earlier parser claims their
records.

## Runtime

The daemon owns all writes (`polylogued run`); the main process is the sole
SQLite writer. Ingest stages: acquire → parse → materialize → index; the
`DaemonConverger` drives derived-model convergence (FTS, embeddings, insights)
with bounded work, quiet-window deferral for hot files, and `convergence_debt`
for retryable backlog. Derived read models converge from durable evidence —
there is no standing "repair" product concept; a failure state is either
explicit-and-retryable or a typed permanent refusal.

## Surfaces

- **CLI is query-first**: root filters before `find`, verb options after the
  action; verbs `find/read/analyze/mark/select/delete/continue`. Query mode
  needs signalled intent (the `find` keyword, a quoted expression, or field
  syntax) — a bare unquoted word errors with a hint. The grammar
  (`archive/query/expression.py`) is a real DSL lowered to SQL.
- **MCP**: 10 capability-gated operation-dispatcher tools; adding an operation
  updates the dispatcher's verb table (`EXPECTED_TOOL_NAMES` is derived; a
  missing tool contract fails discovery). Tool contracts are currently
  per-tool, not per-operation, and MCP insight projections are a hard-coded
  set that bypasses the registry — per-operation contracts and
  registry-driven MCP are the direction, owned by polylogue-fja2v/4p1, not
  current truth.
- **Insights** are descriptor-driven (`insights/registry.py`); one registry
  drives plaintext and JSON (MCP: see the caveat above).
- New Click params on query verbs go last — a positional shift silently
  reroutes args.

## Verification

`devtools` owns repo readiness. The command surface is generated — consult
`devtools --list-commands` or `docs/devtools.md` (catalog:
`devtools/command_catalog.py`; add a command → add its `CommandSpec` +
`render devtools-reference`).

- `devtools test <sel>` — focused pytest through the managed harness (checkout
  guard, environment, typed result). Never bare `pytest`.
- `devtools verify` — static gates plus pytest, selecting from the checkout's
  one testmon datafile (`.cache/testmon/testmondata`, environment `polylogue`)
  and writing back. No datafile: the run seeds it and runs everything. A
  corrupt or foreign-format datafile stops with `graph_unusable` — delete it
  and rerun. `--all` runs every test and still updates fingerprints;
  `--quick` is the static gates alone.
- Every managed pytest run holds the host's single `pytest` pueue slot: a
  caller outside a queued task (`AGENTCTL_JOB_ID`, or legacy `SINNIXD_JOB_ID`, unset) queues, waits, and
  reads the captured log the run prints, and refuses if pueued is unreachable.
- `devtools why` — explain the last run before reading receipts by hand.
- `devtools gate <name>` — one named invariant check (`gate --list`);
  `verify --quick` is the fast subset. `status`, `render [<surface>|all]
  [--check]`, `scenario`, `smoke`, `archive <sub>`, `schema <sub>`,
  `bench <sub>`, `cache gc` are the other verbs; twelve in all.
- `devtools render all --check` can print per-surface `sync OK` yet exit 1 —
  grep for `out of sync`.

Testmon is an accelerator: a selected green proves the selected scope only,
and the receipt names which selection ran. Every managed run — `devtools test`
included — traces into the same datafile, so the graph is advanced, never
recomputed. The corpus runs as one collection; partitioning it would drop the
edges of every test the last shard did not collect. A test names its anti-vacuity
condition — what mutation or bypass would make it red.
Fixtures are generated and deterministic (`tests/infra/`: SessionBuilder,
seeded archives, pathology composer, corpus programs); timestamp-sensitive
tests use `frozen_clock` (an autouse guard rejects wall-clock reads). Keep
ambient machine data out of tests. Per-PR CI runs only the quick gate — a
green PR check does not mean tests ran; verify locally.

Change cross-checks: parser/detection → origin specs + real fixtures + replay
parity; storage/schema → fresh DDL + declared lifecycle + readers/writers +
restart; query/read → CLI/API/MCP parity + pagination + cancellation; daemon →
lifecycle + cancellation + restart; MCP → registry + shared product route;
fixture/harness → proves a production route.

## Code Review Rules

- A finding names a concrete input at the reviewed head and the wrong
  observable outcome; a scenario that needs the environment corrupted below
  its own integrity contract (lockfile, provision stamp, environment digest,
  tests) is out of scope.
- Verification receipts and caches are keyed on declared inputs; do not ask
  for filesystem enumeration (installed trees, example databases, every
  executable) as a cache key.
- A thread answered by a commit or a stated refutation is closed unless the
  answer is wrong; do not restate an answered finding in a later round.
- Judge a test by the anti-vacuity condition it names, not by whether it
  could be stricter.
- Publication text and lane metadata are not review targets.

## Commit / PR discipline

Product code lands via feature branches + squash-merged PRs to protected
`master`. Conventional subjects; PR title = the squash subject (≤72 chars,
imperative). PR body: Summary, Problem (evidence), Solution, Verification
(exact commands + the line that matters), honest residuals. No resolver
keywords next to issue numbers unless the operator asks. Stage by path.
Release-please owns version/CHANGELOG. Before writing "unified"/"complete",
grep the diff and check both paths.

## Documentation map

`docs/atlas/` — agent-orientation sheets with code-verified anchors: read
`00-core.md` and your area's sheet before exploring (storage, daemon, mcp,
sources/parsers, query/read-path). Run `devtools gate atlas` when changing
anchored code; stale sections are re-verified or deleted.
`docs/architecture.md` (rings, data flow), `docs/internals.md` (invariants,
schema history), `docs/architecture-spine.md` (decisions), `TESTING.md`,
`CONTRIBUTING.md`, `docs/devtools.md` (generated), `docs/daemon.md`,
`docs/search.md`, `docs/cost-model.md`, `docs/provider-origin-identity.md`,
`docs/material-protocol-v1.md`, `docs/sidecars.md`. `AGENTS.md` is a symlink to this file — edit
CLAUDE.md only. This file must not carry campaign state, tracker rosters, or
operational history — Beads and dated scratch notes own those.
