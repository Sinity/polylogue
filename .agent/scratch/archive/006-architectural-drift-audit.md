---
created: "2026-04-12T00:00:00+02:00"
purpose: "Architectural drift audit for Polylogue, including open and recently fixed drift found during the repo cleanup and vetting campaign"
status: "archived"
project: "polylogue"
---

# Architectural Drift Audit

Superseded by `active/009-operator-brief.md` for the live unresolved drift subset.

## Scope

This note captures architectural drift discovered while cleaning and stress-testing
Polylogue as a user, including drift that has already been fixed on the current
branch and drift that remains open.

The focus is not style nitpicks. The threshold here is one of:

- parallel implementations of the same concept
- hidden or fragmented control/config surfaces
- stale or dead subsystems that still look live
- decompositions that were only half-finished, leaving a catch-all facade behind
- user-facing command surfaces that disagree about core conventions

## Findings Summary

### Open Drift

1. Async-first storage coexists with a broad production sync `sqlite3` path.
2. Schema bootstrap/version enforcement is duplicated between sync and async paths.
3. SQLite connection defaults and database path logic are duplicated.
4. `polylogue.paths` still acts as a catch-all facade after the path split.
5. Runtime configuration is spread across ad hoc environment reads instead of one narrow control surface.
6. `polylogue/cli/run_execution_workflow.py` looks like a dead duplicate of `run_workflow.py`.
7. Archive compatibility still relies on fresh-only reset for legacy inline-raw databases; there is no migration path for that older storage shape.
8. Health/status uses the generic name `index` for a narrow readiness check, which may mislead operators.

### Fixed Drift On This Branch

1. Dead `doctor --cached` surface removed.
2. Root query verbs now fail explicitly when root-only flags are placed after the verb.
3. `products` commands now accept `--format json`, aligning them with the rest of the CLI.
4. Schema JSON output now uses the same machine envelope shape as the rest of the CLI.
5. Async read bootstrap no longer races fresh schema creation on first read.
6. Hybrid search again exposes one stable connection boundary for both path-backed and injected sqlite handles.
7. Attribution no longer drops already-normalized repo names during its second normalization pass.
8. Index rebuild no longer duplicates action-event materialization that ingest already performed.
9. Single-record ingest batches no longer pay process-pool overhead.
10. Action-event maintenance debt no longer misreport pure FTS backlog as stale row debt.
11. The transient `v2` archive-schema baseline has already been reset to `v1`.

## Open Drift Details

## 1. Async-first storage still has a broad sync side channel

Polylogue presents an async-first runtime:

- [facade.py](/realm/project/polylogue/polylogue/facade.py)
- [repository.py](/realm/project/polylogue/polylogue/storage/repository.py)
- [async_sqlite.py](/realm/project/polylogue/polylogue/storage/backends/async_sqlite.py)

But the sync `sqlite3` helper in
[connection.py](/realm/project/polylogue/polylogue/storage/backends/connection.py)
is not confined to tests or a tiny compatibility shell. It is used by real
product code in:

- [archive.py](/realm/project/polylogue/polylogue/operations/archive.py)
- [embed_runtime.py](/realm/project/polylogue/polylogue/cli/embed_runtime.py)
- [embed_stats.py](/realm/project/polylogue/polylogue/cli/embed_stats.py)
- [health.py](/realm/project/polylogue/polylogue/health.py)
- [check_workflow.py](/realm/project/polylogue/polylogue/cli/check_workflow.py)
- [check_support.py](/realm/project/polylogue/polylogue/cli/check_support.py)
- [fts5.py](/realm/project/polylogue/polylogue/storage/search_providers/fts5.py)
- [hybrid.py](/realm/project/polylogue/polylogue/storage/search_providers/hybrid.py)
- [search_runtime.py](/realm/project/polylogue/polylogue/storage/search_runtime.py)
- [publication_flow.py](/realm/project/polylogue/polylogue/site/publication_flow.py)
- [repair.py](/realm/project/polylogue/polylogue/storage/repair.py)
- [verification_artifacts.py](/realm/project/polylogue/polylogue/schemas/verification_artifacts.py)
- [verification_corpus.py](/realm/project/polylogue/polylogue/schemas/verification_corpus.py)

That means the repo does not currently have "async core with thin sync edges".
It has an async runtime plus a second raw sync SQL surface that production
modules depend on directly.

This is real architectural drift, not just test convenience.

## 2. Schema/version enforcement exists twice

The schema/version gate is implemented separately in:

- [schema_upgrade.py](/realm/project/polylogue/polylogue/storage/backends/schema_upgrade.py)
- [async_sqlite_schema.py](/realm/project/polylogue/polylogue/storage/backends/async_sqlite_schema.py)

These are not trivial wrappers around one canonical implementation. They
duplicate schema bootstrap and exact-version enforcement logic.

That makes storage evolution riskier:

- one path can gain checks or extensions before the other
- one path can change error text or mismatch semantics independently
- storage behavior depends on which connection family reached the DB first

## 3. Connection tuning and default DB path logic are duplicated

The same SQLite defaults appear in both the async and sync storage helpers:

- `default_db_path()` exists in
  [async_sqlite.py](/realm/project/polylogue/polylogue/storage/backends/async_sqlite.py)
  and
  [connection.py](/realm/project/polylogue/polylogue/storage/backends/connection.py)
- connection tuning pragmas like `cache_size`, `synchronous = NORMAL`,
  `mmap_size`, `temp_store`, and `wal_autocheckpoint` are configured in both
  places

This is a classic "same concept, two homes" problem.

## 4. `polylogue.paths` still undercuts the path split

The repo previously split path logic into:

- `paths_roots`
- `paths_models`
- `paths_config`
- `paths_sanitize`

But [paths.py](/realm/project/polylogue/polylogue/paths.py) is still a large
catch-all facade re-exporting all of that, and many runtime modules still
import from it directly.

Examples:

- [archive.py](/realm/project/polylogue/polylogue/operations/archive.py)
- [validation_flow.py](/realm/project/polylogue/polylogue/pipeline/services/validation_flow.py)
- [ingest_worker.py](/realm/project/polylogue/polylogue/pipeline/services/ingest_worker.py)
- [query_output.py](/realm/project/polylogue/polylogue/cli/query_output.py)
- [renderers/html.py](/realm/project/polylogue/polylogue/rendering/renderers/html.py)
- [renderers/markdown.py](/realm/project/polylogue/polylogue/rendering/renderers/markdown.py)
- [site/models.py](/realm/project/polylogue/polylogue/site/models.py)

So the split happened structurally, but the old catch-all module still acts as
the main import surface. That weakens the architectural benefit of the split.

## 5. Configuration is still fragmented across ad hoc env reads

The docs currently present a small runtime override surface in
[docs/configuration.md](/realm/project/polylogue/docs/configuration.md),
but env reads are still scattered across the codebase:

- path roots: [paths_roots.py](/realm/project/polylogue/polylogue/paths_roots.py)
- vector provider: [search_providers/__init__.py](/realm/project/polylogue/polylogue/storage/search_providers/__init__.py)
- query error messaging and run command checks:
  [query.py](/realm/project/polylogue/polylogue/cli/query.py),
  [commands/run.py](/realm/project/polylogue/polylogue/cli/commands/run.py)
- terminal mode:
  [formatting.py](/realm/project/polylogue/polylogue/cli/formatting.py)
- Drive auth:
  [drive_auth.py](/realm/project/polylogue/polylogue/sources/drive_auth.py)
- schema validation mode:
  [validation.py](/realm/project/polylogue/polylogue/pipeline/services/validation.py),
  [ingest_batch.py](/realm/project/polylogue/polylogue/pipeline/services/ingest_batch.py)

This is much smaller than it was earlier in the cleanup pass, but it is still
more fragmented than the docs imply.

The main architectural concern is not "there are env vars". It is that control
policy is read in many places rather than entering through a tighter config
boundary.

## 6. `run_execution_workflow.py` appears dead and duplicate

The repo still contains
[run_execution_workflow.py](/realm/project/polylogue/polylogue/cli/run_execution_workflow.py),
but a repo-wide search found no references to it.

Its contents substantially overlap with
[run_workflow.py](/realm/project/polylogue/polylogue/cli/run_workflow.py),
especially around:

- `execute_sync_once`
- `run_with_progress`
- `run_sync_once`

That looks like leftover decomposition fallout: a second workflow module that
never became the actual import target.

## 7. Legacy inline-raw archives still require reset instead of migration

The current schema constant is still `1`, but archive compatibility is not
actually "same version means safe". A legacy inline-raw database can still
report the same schema version while remaining incompatible with the current
blob-store-backed raw archive layout.

The enforcement is explicit in:

- [schema_upgrade.py](/realm/project/polylogue/polylogue/storage/backends/schema_upgrade.py)
- [async_sqlite_schema.py](/realm/project/polylogue/polylogue/storage/backends/async_sqlite_schema.py)

The live consequence is that legacy inline-raw archives are not migrated in
place. They are treated as incompatible and must be recreated or moved aside.

This may be an intentional policy, but it is still an architectural sharp edge:
storage evolution is tied to whole-archive recreation rather than explicit
migration.

## Fixed Drift On This Branch

## 1. Dead `doctor --cached` surface removed

The CLI still advertised `--cached`, but the cache layer was already gone and
the flag no longer changed behavior. That dead surface has been removed from the
repo on this branch.

## 2. Root query option ordering clarified

The query-first root command still degraded badly when root-only flags appeared
after a verb. For example:

```bash
polylogue stats --by provider --format json --limit 20
```

That now fails with an explicit message telling the user to move the root flag
before the verb, instead of surfacing a generic subcommand parse error.

## 3. `products` surface now accepts `--format json`

The main query surface teaches `--format json`, but registry-driven product
commands only exposed `--json`. That inconsistency has been fixed on this
branch.

## 4. Schema machine-envelope drift fixed

Schema subcommands had inconsistent JSON success shapes compared with the rest
of the CLI. They now emit the same `{"status":"ok","result":...}` envelope.

## 5. Async read bootstrap race fixed

The async backend's read-oriented connection path tried to avoid schema work
when the DB file already existed. That left a race window where one task could
create the empty SQLite file while another immediately opened it read-only and
hit missing-table errors.

The read path now:

- proves whether the archive schema is already present before trusting the file,
  and
- falls back to the canonical one-time schema initialization path only when it
  is genuinely needed.

This keeps first-read bootstrap safe without regressing already-initialized
read responsiveness.

## 6. Hybrid search connection drift fixed

`HybridSearchProvider` had drifted from the module-level connection boundary
that the rest of the search stack and its tests expected. It was calling the
read-only helper directly, which broke two useful invariants:

- tests could no longer patch the module-level connection helper
- in-memory sqlite connections used by the law tests could no longer be passed
  through the same surface

The module now restores a single `open_connection(...)` boundary that routes
paths to read-only connections and existing sqlite handles to the injected
connection context.

## 8. Health/index terminology may be overstating archive emptiness

The current live `doctor --json` payload can report:

- `index`: `messages indexed: 0`
- `action_event_read_model`: thousands of conversations materialized
- `action_event_fts`: hundreds of thousands of pending rows

If `index` truly means only the lexical or vector message-search index, the
label is underspecified and easy to misread as "the archive has no indexed
messages" in the broad sense. If it is meant to describe overall archive
indexing readiness, it is semantically wrong.

This is still under investigation, but it is already architectural drift at the
status-surface level: one narrow subsystem appears to own the generic term
`index`.

## 9. Single-record ingest batches no longer pay process-pool overhead

When huge provider blobs force one-record ingest batches, the old implementation
still built a `ProcessPoolExecutor(max_workers=1)` and round-tripped each
record through a subprocess. That was real architectural drift: a batch-parallel
mechanism still ran even when there was no parallelism to gain.

The current branch now runs those single-record batches inline and leaves the
process pool for genuine multi-worker batches.

## 10. Action-event maintenance debt now distinguishes row backlog from FTS backlog

The earlier `doctor` debt surface bundled:

- missing action-event rows
- stale action-event rows
- pending action-event FTS rows

under one vague `"pending/stale action-event rows"` detail. That created
semantic drift between the health surface and the maintenance/debt surface.

The current branch now reports those components explicitly, so a pure FTS
backlog is no longer mislabeled as stale row debt.

## 11. The transient `v2` schema bump has already been reversed

An earlier branch state bumped `SCHEMA_VERSION` from `1` to `2`. That baseline
has already been reset in:

- `2d462c15` `fix: reset archive schema baseline to v1`

The active branch no longer expects `v2`. Any `expected version 2` failures
captured in earlier vetting notes belong to the older branch state rather than
the current code.

## 7. Attribution re-normalization no longer drops repo names

`extract_attribution_from_action_events()` already returns normalized repo
names, but `extract_attribution()` was feeding those names back into
`normalize_repo_names(...)` as if they were still raw candidates. That silently
dropped provider-meta repo names like `sinex` during session-profile rebuilds.

The final attribution assembly now preserves already-normalized repo names and
only derives additional names from normalized repo paths.

## 8. Grouped JSONL ingest no longer materializes the full record list for
Codex and Claude Code

The worst live memory spikes were coming from grouped JSONL providers being
decoded into one full Python list before parsing. That is now fixed for the two
dominant record-stream providers:

- `codex`
- `claude-code`

The worker now:

- collects a bounded sample for provider/taxonomy/schema work
- counts malformed JSONL lines across the full file
- reopens the blob and stream-parses the full record stream directly into the
  provider parser

This removes the giant `list[dict]` heap from the hot ingest path for those
providers.

## 9. Index rebuild no longer duplicates action-event materialization

Polylogue already materializes durable `action_events` rows during conversation
save on the ingest path, in
[repository_write_conversations.py](/realm/project/polylogue/polylogue/storage/repository_write_conversations.py).

But the indexing stage in
[indexing.py](/realm/project/polylogue/polylogue/pipeline/services/indexing.py)
was still unconditionally deleting and rebuilding the same read model from
`messages` plus `content_blocks` before rebuilding FTS.

That was real drift:

- action events had two independent materialization paths
- the index phase was doing duplicate semantic work that ingest had already done
- full `run all` rebuilds spent large time and memory on redundant action-event
  reconstruction before the actual FTS work

The branch now fixes that by:

- querying action-event repair candidates first
- skipping the action-event rebuild entirely when rows are already current
- repairing only missing or stale conversations before rebuilding FTS
- reporting progress totals from actual work units instead of assuming the full
  action phase always exists

## 10. Single-record ingest batches were still using a process pool

The unified ingest path already batches raw rows by both:

- record count
- total blob bytes

That is correct in principle, but giant rows (`>128 MiB`) necessarily collapse
to one-record batches. The old implementation still routed those batches through
`ProcessPoolExecutor(max_workers=1)`.

That created a pointless second execution mode inside ingest:

- no parallelism benefit
- extra subprocess startup and teardown
- large pickled `IngestRecordResult` transfers back to the parent
- more memory duplication exactly where raw blobs are largest

The branch now fixes that by running `_iter_ingest_results_sync(...)` inline
when `worker_count <= 1` and keeping the process-pool path only for actual
multi-worker batches.

## 11. Session-product semantics changed without version invalidation

Polylogue already has a durable materializer-version mechanism for derived
session products:

- `SESSION_PRODUCT_MATERIALIZER_VERSION`
- stale-row counting in `session_product_status.py`
- health surfacing in `doctor --json`

But earlier inference/repo-attribution cleanup changed the meaning of produced
session-product rows without bumping that version. The result was subtle drift:

- `products profiles` still returned old inferred repo names and auto-tags
- `doctor` treated those rows as current
- users had no health signal telling them a rematerialize was needed

This is not merely documentation drift. It is semantic drift in durable
read-model provenance.

The branch now fixes that by:

- bumping `SESSION_PRODUCT_MATERIALIZER_VERSION` from `3` to `4`
- aligning session-product DDL defaults to the new version
- adding a regression test proving older versions count as stale

After that change, the live archive correctly reports stale session-product
rows until `run materialize` repairs them.

## 12. Repo attribution still mistakes transcript-store repos for worked-on repos

The live post-rebuild profile surface still showed generic repo labels like
`projects` and `root`. The concrete culprit is that repo normalization was
still willing to treat config-backed transcript stores as real repositories:

- `/home/sinity/.config/claude/projects`
- future analogous Codex config-backed stores

That is the wrong semantic layer. Those locations are agent transcript stores,
not evidence that the archived session worked on a repository called
`projects`.

There was also a secondary normalization leak: once a repo name had been
normalized, later attribution assembly could still reintroduce raw values.

The branch now fixes the first part by extending the non-work repo-root filter
to ignore config-backed Claude/Codex transcript stores, and fixes the second
part by keeping normalized repo names normalized through final attribution
assembly.

One related open concern remains:

- full `run materialize` still peaks around `3.8 GiB RSS` on the live archive,
  which suggests the current session-product rebuild path still over-hydrates
  large conversations even after the page-size and progress fixes

## 13. Pipeline stage defaults still assume throughput when RSS already says otherwise

The live vetting pass exposed another form of architectural drift: pipeline
services were still choosing their default concurrency in isolation.

- ingest batches defaulted to the global worker cap even when the current batch
  already represented `100+ MB` of raw payload
- render stages defaulted to broad concurrency even when earlier stages had
  already pushed the main process RSS into GiB territory

That is the wrong abstraction boundary for a local CLI archive tool. The stage
runner needs memory-aware defaults, not just CPU-aware defaults.

The branch now corrects the obvious part of this by:

- throttling ingest worker count from batch blob sizes
- throttling render worker count when entering the stage at high RSS
- emitting render-stage RSS observations directly in the stage result

This does not solve the deeper `run materialize` rebuild inflation on giant
conversations, but it removes one clear class of self-inflicted memory pressure.

## Recommendations

If this audit turns into actual cleanup work, the likely order is:

1. Collapse schema bootstrap/version enforcement to one canonical implementation.
2. Decide whether sync raw-SQL access is allowed anywhere outside:
   - tests
   - a tiny sync facade
   - strictly justified maintenance code
3. Remove or absorb `run_execution_workflow.py`.
4. Decide whether `polylogue.paths` remains a real public facade or whether
   internal modules must import the split path modules directly.
5. Tighten runtime configuration entry so env reads happen in fewer places.

## Notes

- This note intentionally includes findings that were already fixed on the
  current branch, so future sessions can distinguish "open drift" from "drift
  already removed during the cleanup and vetting pass".
