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
  re-point a durable `user.db` reference. Blocks use their exact semantic
  digest and occurrence among identical blocks; position states order only.
  Formulas and the field partition are in `pipeline/ids.py` and
  `docs/atlas/storage.md`.
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
- Archive writes are idempotent by content hash (SHA-256 over the canonical
  JSON payload, excluding user metadata, so tagging never re-imports). Only
  declared prose fields are NFC-folded; identifiers, tool arguments, paths,
  metadata, and mapping keys hash exactly, and absence stays JSON `null`,
  distinct from `""`. `pipeline/ids.py` declares which parser fields are
  hashed, which are prose, and why each excluded field is excluded.

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
  tiers have no migration chain. For `index` and `ops`,
  `archive_tiers/schema_identity.py` stamps an identity over their DDL plus
  the lowering, materializer, and replay-routing fingerprints, every open
  compares it, and a mismatch is a typed `SchemaSkew` resolved by
  reconvergence through the daemon. `embeddings` carries no schema identity;
  its DDL is versioned by `EMBEDDINGS_SCHEMA_VERSION`. Classify a schema
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

An arbitrary limit that changes an outcome is a defect: a size or count cap
that refuses, truncates, or drops valid input, and a timeout that turns slow
but valid work into a failure, a skip, or a partial result. Bound memory by
streaming or paging, and bound waiting by progress or cancellation, instead.
Only a real physical limit (such as SQLite's maximum value length) justifies
refusal, and that refusal is typed and visible, never silent.

## Provider, Origin, Source

`Origin` is the public source token on query surfaces and read payloads;
public filters use `origin`. `Provider` is the provider-wire token, legitimate
at acquisition, parser, and schema boundaries and a leak on public surfaces.
`Source` carries richer acquisition identity. GEMINI+DRIVE maps to
AISTUDIO_DRIVE non-injectively, so never reverse an Origin into a guessed
Provider (`docs/provider-origin-identity.md`). Bound-source shape checks
validate the declared origin; foreign content is a typed refusal. Use
OriginSpec for declared artifact kinds and unbound import classification.

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

- Match verification to the changed behavior and consequence. For an ordinary
  well-understood fix, use a brief delta review, imports/basic static checks,
  or one discriminating behavioral check when useful. Batch broader regression
  at meaningful integration milestones, not after every patch. Record known
  failures and untested areas; unrelated fixture failures need not block a
  product fix. Preserve assertions and report the original run's outcome.
  Destructive changes, durable data-loss boundaries, and purchased-vector
  preservation need stronger checks before execution.
- `devtools test <selection>` runs focused tests through the managed host
  pool; never run bare `pytest`. Workers use an explicit selection:
  `agentctl job start polylogue pytest_focused --workspace <checkout> -- <selection>`.
  A bare worker `verify` wrapper supplies no selector and cannot run the
  required-argument focused profile. The pre-push hook runs
  `devtools verify --quick` on the pushed HEAD; required hooks and hosted
  merge checks remain enforced. Reuse existing results while their relevant
  code is unchanged; task `verification_commands` are examples to reconcile
  with the current descriptor, not an automatic full-suite gate.
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
  `frozen_clock`, except where the reference time comes from outside the process
  (a Git commit, an OS wait, a benchmark), which opt out with `uses_real_clock`
  as `TESTING.md` describes. Static fixture payloads live in
  `tests/fixtures/`; fixture builders and shared harness code live in
  `tests/infra/` and `conftest.py`.
- Cross-check by change type: parser or detection → origin specs, real
  fixtures, replay parity; storage or schema → fresh DDL, the migration or
  moved identity, readers and writers, restart; query or read → equivalence of
  the generic read and the typed session route, pagination, cancellation;
  daemon → lifecycle, cancellation, restart; MCP → registry and the shared
  product route.

## Code Review Rules

Use `docs/review/codex-review-guide.md` for relevant invariants, severity and noise
and the nested `AGENTS.md` beside each changed file.
- A change to an interface, command, config key, schema, route, or file format
  updates every consumer (callers, CLI/MCP, tests, docs, generated references,
  configs, hooks) and deletes the predecessor in the same change; name a
  missed consumer, P1 when a caller breaks.
- A compatibility path in a diff (shim, alias, fallback, dual read/write,
  deprecated wrapper) is a defect. A finding whose remedy keeps the old path,
  or migrates prior archive state into fresh v1, is noise.
- A cap, timeout, or truncation that refuses or cuts valid input is a defect;
  ask for paging or streaming, never a new limit.
- A finding names a concrete input at the reviewed head and the wrong
  observable outcome; environment corruption below its integrity contract is
  out of scope.
- A thread answered by a commit or refutation is closed unless the answer is
  wrong. Publication text and task metadata are not review targets.

## Commits and PRs

Product code lands through feature branches and squash-merged PRs to
protected `master`. The PR title is the conventional squash subject (72
characters or fewer, imperative). The body has Summary, Problem (with
evidence), Solution, Verification (exact commands and the line that matters),
Self-review, and honest residuals. Before pushing, review the changed delta
and its affected contracts; reuse the review of unchanged code. Report
concrete findings and residuals briefly. The guide's checklist is a reference,
not a required item-by-item narration or repeated review ceremony. Required
repository review and merge protections still apply. Put no
resolver keywords beside issue numbers unless the operator asks.
release-please owns the version and changelog. Before writing "unified" or
"complete", grep the diff and check both paths.
