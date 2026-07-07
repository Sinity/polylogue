---
created: "2026-06-28"
purpose: "Read-only architectural assessment of ingest/convergence pipeline — essential vs accidental complexity"
status: "complete"
project: "polylogue"
---

# Convergence / ingest pipeline — streamlining assessment

Read-only. No code changed. Question: can the convergence STAGES and the
raw/index/insights/FTS/embeddings split be streamlined, or is the complexity
earned?

Bottom line up front: **the hard parts are earned, the soft parts are
residue.** The 5-tier split, convergence-debt machinery, deferred
insights/embed, and inline-FTS+catch-up are essential (crash safety, durability
classes, the #1498 I/O lessons). What is genuinely accidental is *dual-path
residue*: a legacy single-file branch that lives next to the archive branch in
every convergence stage, and a vestigial composite-stage CLI vocabulary
(`run_stages` / `INGEST_STAGE_SEQUENCES`) that runs in parallel to the daemon's
real stages. Removing residue is worth a modest, well-scoped effort. A
"merge the stages / flatten the tiers" rewrite is **not** worth it.

---

## Current shape

### Three distinct "stage" vocabularies (this is the core confusion)

1. **Daemon live ingest path — the real runtime.**
   `polylogue/sources/live/batch.py:LiveBatchProcessor.ingest_files`. Per file
   it decides append-plan vs full-parse vs defer, then writes
   **raw (source.db) + parsed tree + messages + blocks + FTS inline in one
   transaction** via `pipeline/services/ingest_batch/_core.py`
   (`repair_message_fts=True` default; FTS triggers suspend/restore around bulk).
   Then `LiveBatchProcessor._converge_paths` (batch.py:719) runs the convergence
   stages for the post-ingest derived work.

2. **Convergence stages — fts / embed / insights.**
   `polylogue/daemon/convergence_stages.py:make_default_convergence_stages`
   (line 528). Each stage is a `ConvergenceStage`
   (`daemon/convergence.py:61`) carrying **six entry points**:
   `check`/`execute` (single file), `check_many`/`execute_many` (batch),
   `check_sessions`/`execute_sessions` (session-scoped debt retry). Driven by
   `DaemonConverger` (convergence.py:158) via `converge_batch` /
   `converge_sessions`. Failures become durable `live_convergence_debt`,
   drained by `daemon/cli.py:_periodic_convergence_check` →
   `_drain_convergence_debt_once` (cli.py:428).

3. **Legacy CLI/explicit composite stages — acquire/parse/materialize/index/embed/schema.**
   `pipeline/ingest_support.py:INGEST_STAGE_SEQUENCES` + `pipeline/run_stages.py`.
   Used by explicit CLI ingest, `devtools/pipeline_probe/engine.py`,
   `operations/specs.py` (declared operation graph), `demo/workspace.py`.
   **The daemon does not use this path.** `run_stages.execute_ingest_stage`
   (run_stages.py:87) literally `del`s nine of its parameters and just calls
   `parse_sources_archive` — a vestigial wrapper.

### The conceptual "desired state" (convergence.py docstring, lines 1-13)
Per source file: (1) raw blob stored, (2) parsed to records, (3) messages
materialized, (4) FTS indexed, (5) insights refreshed. In the live runtime
steps 1-4 happen **inline** in the write transaction; step 5 (+ embed + FTS
catch-up) is the deferred convergence pass.

### Insight materialization — already unified
`storage/insights/session/rebuild.py:build_session_insight_records` (552)
produces ONE bundle per session covering session_profiles, work_events, phases,
threads, tag_rollups AND the run-projection trio (session_runs,
session_observed_events, session_context_snapshots — all from
`compile_recovery_digest(...).run_projection`, line 617). Eight derived tables,
one atomic materializer, version-stamped (`SESSION_INSIGHT_MATERIALIZER_VERSION`,
`insight_materialization` table). This is **not** over-granular — it is a single
pass keyed on a single version. `insights/registry.py:INSIGHT_REGISTRY` is a
*display/CLI* registry (`fetch_insights`), not a second materialization driver.

### The 5 tiers (storage/sqlite/archive_tiers/)
| Tier | Contents | Durability class |
|---|---|---|
| source.db | raw acquisition rows, source evidence | **durable** (backup-required) |
| index.db | parsed tree, FTS, topology, derived insight read-models | rebuildable from source |
| embeddings.db | vectors, embedding status/catch-up | rebuildable (expensive) |
| user.db | marks, annotations, corrections, tags, saved views | **irreplaceable** |
| ops.db | cursors, attempts, convergence debt, OTLP | disposable |

Split is by **durability class**, not by feature. The fresh-first schema model
(internals.md "Schema Versioning Model") depends on it: bump index schema → nuke
index.db → rebuild from source.db, with user.db untouched.

---

## Essential complexity — keep, do not touch

- **5-tier split by durability class.** This is the backbone of the
  fresh-first / re-ingest schema model. Every index-schema bump (v5→v14 per
  internals.md) rebuilds index.db from source without risking user.db.
  Collapsing tiers would couple irreplaceable user data to rebuildable derived
  data and break backup policy. *Reason: durability + restart/rebuild model.*

- **convergence-debt + live_ingest_attempt + cursor machinery.** This is the
  whole point of the daemon. SIGKILL between blob-write and DB-row, partial
  appends, back-pressure, retry — all encoded here and in the #1498 retro's 26
  PRs. `false_means_pending` (convergence.py:83) + session-scoped debt is what
  lets a bounded successful pass leave the rest as durable retry.
  *Reason: crash safety, idempotency, bounded I/O.*

- **insights/embed as DEFERRED stages, not inline.**
  `_hot_insight_session_ids` (convergence_stages.py:740) defers full insight
  rebuilds while a large source file is actively appending — otherwise every
  small append to a multi-GB agent session triggers a full re-hydrate/re-DELETE
  cycle (this is exactly #1468 / #1607 in the retro). Inlining insights would
  re-introduce the I/O storm the cascade fixed. *Reason: performance / I/O
  amplification.*

- **FTS inline + convergence catch-up (two places, on purpose).** Inline repair
  keeps search atomically fresh with the write; the convergence `fts` stage is
  the safety net for SIGKILL-during-trigger-suspend drift (internals.md FTS5
  model). *Reason: crash safety.* The two are not duplication of intent — one is
  the happy path, one is the repair invariant.

- **session-scoped vs path-scoped convergence (the `_sessions` entry points).**
  After raw compaction or a moved/renamed source file, a failed derived subject
  can no longer be resolved back to a source path; session-scoped debt retry
  (convergence.py:414 `converge_sessions`) exists precisely for that.
  *Reason: correctness under source churn.*

---

## Accidental complexity — streamlining candidates (ranked)

### 1. Legacy single-file branch inside every convergence stage — HIGH payoff / MEDIUM risk
`convergence_stages.py` is ~1,490 lines. Every stage check/execute is
`if (archive_db := _active_archive_index_path(...)) is not None: <archive
branch> else: <legacy single-file db.db branch>`. Since #1787 the split-tier
archive is the **sole runtime** (MEMORY.md), so in production index.db always
exists and the archive branch always wins — the entire `else` half
(`open_connection(db_path)`, `messages_fts` monolith reads, the non-`_archive_*`
helpers) is dead on the live host. Removing the legacy halves would roughly
halve the file and erase the dual-vocabulary that makes the stage logic hard to
follow.
- **Why accidental:** pure post-#1787 residue; two physical code paths for one
  live behavior.
- **Simplification:** delete the non-archive branches and the paired legacy
  helpers (`_session_ids_for_source_path` non-archive variants, the
  `db.db`-oriented `make_*` else arms).
- **Risk:** must confirm nothing constructs a non-split backend for the daemon
  (tests may still exercise the legacy path; `storage/sqlite/schema.py` still
  carries a single-file `SCHEMA_VERSION` notion that should be checked).
  Verify `_active_archive_index_path` can never be None at daemon runtime before
  cutting.
- **Effort:** ~1 focused PR. The retro already flags this: "refactor before
  adding a fourth stage."

### 2. Vestigial composite-stage CLI vocabulary (`run_stages` / `INGEST_STAGE_SEQUENCES`) — MEDIUM / LOW-MEDIUM
The `acquire/parse/materialize/index/embed/schema` composite-stage model
(ingest_support.py) is a *second* pipeline vocabulary describing the same
conceptual flow the live path does inline. `execute_ingest_stage` is a
no-op-parameter shell (run_stages.py:87, `del repository, archive_root, ...`).
It is not fully dead — pipeline_probe, operations specs, and explicit CLI ingest
reference it — but it duplicates concepts and invites drift against the daemon's
real stages.
- **Why accidental:** pre-daemon CLI orchestration kept alongside the daemon
  path; the `del`-everything wrapper is the smoking gun.
- **Simplification:** (a) delete the dead-parameter signature of
  `execute_ingest_stage` (trivial), and (b) reframe `run_stages` explicitly as
  "the explicit/CLI re-ingest entrypoint that delegates to the same primitives
  as the live path," rather than presenting `materialize`/`index` as independent
  stages. Consider folding `materialize` and `index` (they always run together
  in `reprocess`/`all` and the live path treats them as one inline write).
- **Risk:** operations-graph specs and pipeline_probe assert on these stage
  names; update them in lockstep.
- **Effort:** small for (a), medium for (b).

### 3. Split convergence_stages into per-stage modules — MEDIUM / LOW (mechanical)
The retro's standing verdict: pull fts/embed/insights into
`daemon/convergence/<stage>.py`, leaving an orchestrator. Best done *after*
candidate #1 (don't carry the legacy branches into new files).
- **Why accidental:** accretion through ~10 cascade PRs, not design.
- **Risk:** low; pure move + import.

### 4. The single-file `check`/`execute` (non-`many`, non-`sessions`) entry points — MEDIUM / MEDIUM
The plain `check`/`execute` pair is only reached by
`DaemonConverger.converge_file` / `converge_all`. The live path calls
`converge_known_sessions` → `converge_batch` and only falls back to
`converge_file` defensively (batch.py:769). If single-file convergence is
unreachable in production, the per-stage protocol drops from six methods to four.
- **Why possibly accidental:** a third execution shape kept for a fallback that
  the always-present `converge_batch` makes unreachable.
- **Risk:** verify `converge_file`/`converge_all` are truly unused at runtime
  (they have test coverage); medium.

### 5. `embed` `check_*` writes during a "check" — LOW / LOW
`_reconcile_embedding_config_change` runs inside `_archive_embed_check*` on a
writable connection (convergence_stages.py:1123). A check that mutates +commits
is surprising and puts a write-tx on the read/check path. Minor; move
reconciliation into `execute` or a one-shot startup step.

---

## Honest verdict

**Mostly earned.** The pipeline is more complex than a naive importer, but the
complexity tracks real obligations: durability separation, crash/restart safety,
and the empirically-won I/O budget from the #1498 cascade. The insight
materializer is already unified (one bundle, one version, eight tables); the FTS
"two places" is happy-path + repair-invariant, not redundancy; the deferral
logic is load-bearing. **Do not** pursue "merge the convergence stages" or
"flatten the tiers" — those would re-open closed bugs.

What *is* worth doing is removing **residue**, not redesigning structure: the
legacy single-file dual-path that survives post-#1787, the vestigial composite
CLI stage vocabulary, and the dead-parameter wrapper. These add indirection with
zero durability or restart payoff. They make the system *look* more granular and
complex than it functionally is — which is likely what prompted the operator's
question.

## Suggested sequencing (proposals — need operator sign-off)

1. **Verify-then-strip the legacy single-file branches** in
   `convergence_stages.py` (candidate #1). Highest readability payoff; start
   with a verification pass proving `_active_archive_index_path` is never None at
   daemon runtime.
2. **Delete the `execute_ingest_stage` dead-parameter wrapper** and audit
   `run_stages`/`INGEST_STAGE_SEQUENCES` callers (candidate #2a); decide whether
   to fold `materialize`+`index`.
3. **Then** split convergence_stages into per-stage modules (candidate #3) on
   the now-halved file.
4. Optional: prune single-file `check`/`execute` (candidate #4) and move embed
   reconciliation off the check path (candidate #5).

None of these touch the tier split, convergence-debt, deferral, or inline-FTS
invariants — those stay exactly as they are.
