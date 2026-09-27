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
sources/ ─detect→ pipeline/ ─hash+write→ storage/{6 tiers} ─materialize→ analysis/
                                              │                              │
                            surfaces: cli/  mcp/  api/  daemon/  ─read-through─┘
                            verification:   devtools/  tests/  schemas/
```

New semantics go into the substrate (`storage`/`analysis`) or product layer
first; surfaces adapt through `analysis`/`operations`/`api`. Surface→substrate
imports are a ratchet enforced by `devtools gate layering` (baseline may
shrink, never grow); substrate→surface imports are forbidden outright.

## Identity and content model

Sessions → messages → blocks are `STRICT`. IDs are generated from source
identity. A message uses a provider-native ID when present; otherwise its ID
uses a digest of its own declared semantic fields plus an occurrence counter,
never its ordinal position. This prevents an export insertion from silently
reassigning a durable `user.db` reference. `messages.material_origin` is
independent of role. `blocks.tool_outcome` is the structural outcome; a
deliberate `unknown` is not success. See `pipeline/ids.py` and
`docs/atlas/storage.md` for the exact formulas and field partition.

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
| `embeddings.db` | expensive to rebuild | vectors, meta, status; preserve reusable vectors before replacement |
| `user.db` | durable, irreplaceable | unified `assertions`, settings, annotation schemas/provenance |
| `audit.db` | durable, continuity-chained | previews, authorizations, attempts, continuity |
| `ops.db` | disposable | cursors, attempts, convergence debt, daemon telemetry |

A mutable SQLite source is the one material that is not its own bytes: a
declared database member is retained as the canonical logical export of its
declared `logical_tables` (`sources/sqlite_export.py`), and that export's blob
hash is the member's logical revision. A page image cannot be proven against a
live database and re-snapshots on every commit.

Never use rebuildable state as authority for durable mutation. Archive writes are
idempotent by content hash (SHA-256 over NFC-normalized payload, excluding
user metadata — tagging never re-imports). The parser-side hash vocabulary is
declared and exhaustively partitioned in `pipeline/ids.py`: semantic session,
message, and block fields are hashed; parser-only coordinates, provider
signatures, and independently-owned usage/timing/cost measurements are
excluded with reasons.

**Schema regimes**: the fresh archive format starts all six tier files at
`PRAGMA user_version=1`. Move previous Polylogue state aside intact solely as
salvage evidence. Do not migrate, import, carry forward, roll back to, or read
back from it. External source files explicitly declared for intake may be
ingested into the empty archive.

Durable tiers (`source`, `user`, `audit`) may evolve in later formats by
additive numbered migrations under `storage/sqlite/migrations/`, one step at a
time. That future policy does not apply to the
fresh start. Derived tiers (`index`, `embeddings`, `ops`) have no version
migration chain. `storage/sqlite/archive_tiers/schema_identity.py` stamps an
identity over their DDL and the lowering, materializer and replay-routing
fingerprints; each open compares it and reports a typed `SchemaSkew` on
mismatch. Recovery is reconvergence through the production daemon. Classify
schema changes before editing: metadata-only, index-only, additive-derived,
additive-durable, or semantic-reparse.

**Derived identity also moves on ordinary code edits, not just DDL.** The lowering, materializer and replay-routing fingerprints are AST closures over imported source, so a pure-performance change touching no schema can still move the identity. The closure follows the current import graph, not directory boundaries. Determine membership on the candidate with `devtools schema closure <file>`. Comments are absent from the AST. Land every closure change before a rebuild starts; a landing mid-run invalidates it.

Enum membership is not a schema constraint. Durable DDL carries no
enum-generated `CHECK(col IN (...))`; vocabulary membership is validated at the
write boundary (`require_vocabulary` in `archive_tiers/common.py`), so adding a
vocabulary member is not a durable migration.

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

The required live-write owner is `polylogued run`, with one SQLite writer
coordinating mutations. Process-local write-lease enforcement alone does not
exclude external CLI/API writers; inspect the actual route before claiming
sole-writer adoption. Remaining bypasses are completion work, not an alternate
write policy.

Ingest stages are acquire → parse → materialize → index. `DaemonConverger`
drives ordered recovery and derivation stages; `make_default_convergence_stages`
is the current stage list. Hot-file deferral and `convergence_debt` retain
retryable backlog. Derived read models converge from durable evidence —
there is no standing "repair" product concept; a failure state is either
explicit-and-retryable or a typed permanent refusal.

## Surfaces

The CLI is query-first: root filters precede `find`, and verb options follow
the action. A bare unquoted word is not query intent. New Click parameters on
query verbs go last so positional arguments retain their meaning. MCP session
operations have typed contracts in `docs/session-operations.md`; insights are
driven by `analysis/registry.py`. Every row-bearing operation decides one
terminal `outcome` at `surfaces/outcome.py`; a named gap is `degraded`, even
with zero rows. See `docs/atlas/query-read-path.md` and `docs/atlas/mcp.md`
for surface detail.

## Verification

`devtools` owns repo readiness. The command surface is generated — consult
`devtools --list-commands` or `docs/devtools.md` (catalog:
`devtools/command_catalog.py`; add a command → add its `CommandSpec` +
`render devtools-reference`).

Use `devtools test <selection>` for focused behavior through the managed host
pool; never bare `pytest`. Run one combined selection after a coherent source
change, reusing its receipt across related Beads. `devtools verify --quick`
runs static gates only. `devtools verify` makes a bounded affected selection
from a usable testmon graph and refuses when it cannot; it never silently
becomes a corpus run. Broad or complete-corpus verification requires an
explicit request. `devtools why` and the run receipt show exactly
what ran; a zero-test or quick-gate green does not prove behavior. Read
`.agentctl/project.toml` on the candidate for hosted checks and review policy.
Tests exercise the production route and name what would make them fail.
Use `docs/devtools.md` for command details.

Change cross-checks: parser/detection → origin specs + real fixtures + replay
parity; storage/schema → fresh DDL + declared migration or moved identity +
readers/writers + restart; query/read → equivalence between the generic read
operation and the typed session-owner route + pagination + cancellation; daemon →
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

Read the relevant `docs/atlas/` sheet for a storage, daemon, MCP, source, or
query change; confirm its claims against current code. `docs/architecture.md`,
`docs/internals.md`, `docs/devtools.md`, and `CONTRIBUTING.md` hold broader
reference material. Keep campaign state, tracker rosters, and operational
history in Beads and dated evidence, not in this always-loaded file.
