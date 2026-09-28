# Polylogue

Polylogue is a local, single-writer archive of AI coding and chat sessions
(Claude web and Code, ChatGPT, Codex, Gemini and Drive, Antigravity, Hermes).
It ingests heterogeneous exports and live captures into a set of SQLite files,
derives read models, and serves them through a query-first CLI, an MCP server,
a Python API, and an HTTP daemon. Pure Python.

This file holds repository semantics. Tasks live in external Beads (`bd`
redirects outside the checkout); job, worktree, and publication mechanics are
environment-level.

## Public boundary

Tracked content, commits, CI logs, and PR and review text are public. Never
commit operator archives, transcripts, private exports, local databases,
receipts, or scratch state. Tests use neutral synthetic fixtures.

## Orientation

```text
sources/ ─detect→ pipeline/ ─hash+write→ storage/{6 tiers} ─materialize→ analysis/
                                              │                              │
                            surfaces: cli/  mcp/  api/  daemon/  ─read-through─┘
                            verification:   devtools/  tests/  schemas/
```

New semantics go into the substrate (`storage`, `analysis`) or the product
layer first; surfaces adapt through `analysis`, `operations`, or `api`.
Surface-to-substrate imports are a ratchet (`devtools gate layering`: the
baseline may shrink, never grow); substrate-to-surface imports are forbidden.

Before a storage, daemon, MCP, source, or query change, read its
`docs/atlas/` sheet and confirm the claims you rely on against current code.

## Identity and content

- Sessions, messages, and blocks are `STRICT` tables whose IDs are generated
  from source identity. A message uses its provider-native ID when present;
  otherwise a digest of its declared semantic fields plus an occurrence
  counter, never its position, so an export that gains a message cannot
  re-point a durable `user.db` reference. Formulas and the field partition are
  in `pipeline/ids.py` and `docs/atlas/storage.md`.
- `messages.material_origin` records who authored the content, independent of
  role. `blocks.tool_outcome` is the structural outcome; a deliberate
  `unknown` is never success.
- Lineage: forks, resumes, subagents, and compaction replay the parent's
  prefix. The writer stores only the child's divergent tail, its
  `branch_point_message_id` (deliberately not a foreign key, so a parent's
  full replace cannot null it), and the inheritance mode; reads recompose.
  `session_links` also holds parser-asserted topology edges, resolved on every
  save by `write_parsed_session_to_archive`, the one write path shared by live
  ingest and replay.
- Archive writes are idempotent by content hash (SHA-256 over the
  NFC-normalized payload, excluding user metadata, so tagging never
  re-imports). `pipeline/ids.py` declares which parser fields are hashed and
  why each excluded field is excluded.

## Storage tiers

| Tier            | Durability                  | Holds                                                                        |
| --------------- | --------------------------- | ---------------------------------------------------------------------------- |
| `source.db`     | durable                     | raw acquired bytes, artifact taxonomy, blob and GC substrate, hooks, sidecars |
| `index.db`      | rebuildable                 | parsed tree, FTS, links, costs, materialized insights                         |
| `embeddings.db` | expensive to rebuild        | vectors and their status; repurchased, never replayed, so no product route replaces this tier |
| `user.db`       | durable, irreplaceable      | assertions, settings, annotation schemas and provenance                       |
| `audit.db`      | durable, continuity-chained | previews, authorizations, attempts, continuity                                |
| `ops.db`        | disposable                  | cursors, attempts, convergence debt, daemon telemetry                         |

- Rebuildable state is never the authority for a durable mutation.
- A mutable SQLite source is retained as the canonical logical export of its
  declared `logical_tables` (`sources/sqlite_export.py`); that export's blob
  hash is its revision. A page image cannot be proven against a live database.
- The fresh archive format starts every tier at `PRAGMA user_version=1`.
  Earlier Polylogue state is moved aside intact as salvage evidence only: no
  migration, import, readback, or rollback. Explicitly declared external
  source files may be ingested into the empty archive.
- Later durable-tier changes (`source`, `user`, `audit`) are additive numbered
  migrations under `storage/sqlite/migrations/`, one step at a time. Derived
  tiers have no migration chain: `archive_tiers/schema_identity.py` stamps an
  identity over their DDL plus the lowering, materializer, and replay-routing
  fingerprints, every open compares it, and a mismatch is a typed
  `SchemaSkew` resolved by reconvergence through the daemon. Classify a schema
  change before editing: metadata-only, index-only, additive-derived,
  additive-durable, or semantic-reparse.
- Those fingerprints are AST closures over imported source, so an ordinary
  code edit (even a pure performance change) can move the derived identity;
  comments cannot. Check membership on the candidate with
  `devtools schema closure <file>`, and land every closure change before a
  rebuild starts.
- Durable DDL carries no enum-generated `CHECK (col IN …)`; vocabulary
  membership is validated at the write boundary (`require_vocabulary` in
  `archive_tiers/common.py`), so a new vocabulary member is not a migration.

## Correctness over repair

Derived read models converge from durable evidence; there is no repair
product. A failure is either explicit and retryable or a typed permanent
refusal. When a broken state appears, find the code that produces it: remove
a repair or maintenance path whose producer is gone, and enforce the missing
invariant at the write boundary where the producer remains. Compatibility,
legacy, deprecated, and transitional shapes are removal targets: enumerate
their readers and writers and give each a replacement path.

A size or count cap that refuses, truncates, or drops valid input is a defect:
bound memory by streaming or paging instead. Only a real physical limit (such
as SQLite's maximum value length) justifies refusal, and that refusal is typed
and visible, never silent.

## Provider, Origin, Source

`Origin` is the public source token on query surfaces and read payloads;
public filters use `origin`. `Provider` is the provider-wire token, legitimate
at acquisition, parser, and schema boundaries and a leak on public surfaces.
`Source` carries richer acquisition identity. GEMINI+DRIVE maps to
AISTUDIO_DRIVE non-injectively, so never reverse an Origin into a guessed
Provider (`docs/provider-origin-identity.md`). Detection in
`sources/dispatch.py` is shape-based in tightness order; insert a new detector
at its true tightness or an earlier parser claims its records.

## Runtime

`polylogued run` is the required live-write owner, with one SQLite writer
coordinating mutations. A process-local write lease does not exclude external
CLI or API writers; inspect the actual route before claiming a surface writes
only through the daemon. Ingest runs acquire, parse, materialize, index.
`DaemonConverger` drives ordered recovery and derivation stages
(`make_default_convergence_stages`); hot-file deferral and `convergence_debt`
hold retryable backlog.

## Surfaces

- The CLI is query-first: root filters precede `find`, verb options follow
  the action, and a bare unquoted word is not query intent. New Click
  parameters on query verbs go last so positional arguments keep their
  meaning.
- MCP session operations have typed contracts (`docs/session-operations.md`);
  insights are driven by `analysis/registry.py`.
- Every row-bearing operation decides one terminal `outcome` in
  `surfaces/outcome.py`. A named gap is `degraded` even with zero rows.

## Verification

`devtools` owns repository readiness; `devtools --list-commands` and
`docs/devtools.md` describe it (catalog in `devtools/command_catalog.py`; a new
command needs its `CommandSpec` and `devtools render devtools-reference`).

- `devtools test <selection>` runs focused tests through the managed host
  pool; never run bare `pytest`. The pool is shared: run one selection per PR
  head covering every affected file, and batch review-thread fixes into one
  push. Re-running only the failing ids is diagnosis on an unchanged tree;
  after a source change, re-run the complete affected selection. Reuse a receipt across related
  tasks. The hosted quick gate covers static checks, so a local
  `verify --quick` before pushing is optional.
- `devtools verify --quick` runs the static gates (`devtools gate --list`
  enumerates them; `devtools gate <name>` runs one).
  `devtools verify` selects affected tests from a usable testmon graph and
  refuses when it cannot; it never silently becomes a corpus run. Broad or complete-corpus runs need an
  explicit request.
- `devtools why` and the run receipt show what ran. A zero-test or quick-gate
  green does not prove behavior. `.agentctl/project.toml` on the candidate
  declares the hosted checks.
- Tests exercise the production route and name the change that would turn
  them red. Behaviour tests assert typed outcomes, stable event tokens, and
  declared fields, not natural-language wording; rendered text is asserted
  only where that text is itself the declared output contract. Timestamp-sensitive tests use
  `frozen_clock`; fixtures come from `tests/infra/`.
- Cross-check by change type: parser or detection → origin specs, real
  fixtures, replay parity; storage or schema → fresh DDL, the migration or
  moved identity, readers and writers, restart; query or read → equivalence of
  the generic read and the typed session route, pagination, cancellation;
  daemon → lifecycle, cancellation, restart; MCP → registry and the shared
  product route.

## Review guidelines

- A finding names a concrete input at the reviewed head and the wrong
  observable outcome. A scenario that needs the environment corrupted below
  its own integrity contract (lockfile, provision stamp, environment digest)
  is out of scope.
- Receipts and caches are keyed on declared inputs; do not ask for filesystem
  enumeration as a cache key.
- A thread answered by a commit or a stated refutation is closed unless the
  answer is wrong; do not restate it in a later round.
- Judge a test by the anti-vacuity condition it names, not by whether it could
  be stricter.
- Publication text and task metadata are not review targets.

## Commits and PRs

Product code lands through feature branches and squash-merged PRs to
protected `master`. The PR title is the conventional squash subject (72
characters or fewer, imperative). The body has Summary, Problem (with
evidence), Solution, Verification (exact commands and the line that matters),
and honest residuals. Put no resolver keywords beside issue numbers unless the
operator asks. release-please owns the version and changelog. Before writing
"unified" or "complete", grep the diff and check both paths.
