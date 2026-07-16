# B7 — CLI thinning audit: making `cli/` a thin client (bead t46)

Read-only design doc. Prerequisite for the thin-client + UDS-protocol direction.
Grounded in live source at `/realm/project/polylogue` (2026-07-05).

## Method / what the "45" actually is

`rg 'from polylogue.(storage|pipeline|archive)' polylogue/cli/` returns **~90
import lines** across ~40 files. That's the raw coupling surface. I split it two
ways that matter for the thin-client goal:

- **Substrate-execution reaches** — code that *opens the DB, runs SQL, fuses
  vectors, materializes, migrates, bootstraps, GCs*. These are the ones that
  make the CLI fat and that a protocol client **cannot** carry. ~25 distinct
  sites. This is the real "45 vs 18" problem.
- **Type/DTO imports** — `Session`, `SessionSummary`, `Message`, `Action`,
  `SessionQuerySpec`, `ContentProjectionSpec`, `Role`, `MessageType`,
  `SessionSemanticFacts`, neighbor/hit models. ~40 sites, most under
  `if TYPE_CHECKING:`. They don't execute substrate work, but they couple the
  client to Python domain classes buried in `archive.*`. For an in-process CLI
  that's tolerable; for a Go/Rust wire client they must become a **shared
  serialized payload vocabulary**, not deep `archive.*` reaches.

The facade already exists and is **rich**: `polylogue.api.Polylogue`
(`api/archive.py`, 126 async methods). The CLI's non-thinness is not "the API is
missing" — it's that **the CLI grew a parallel substrate path that bypasses the
facade**, plus a genuine gap in **ops/write** contracts.

## The load-bearing finding: `cli/archive_query.py` is a second engine

`polylogue/cli/archive_query.py` (2194 lines) is the main read path
(`query.py:111 → execute_archive_query`). It does substrate work directly:

- `ArchiveStore.open_existing(archive_root)` — opens the DB itself
  (`archive_query.py:138,335,766,1483,1570,1601,1673,1700,1731,1881`).
- `create_vector_provider(config, db_path=...)` + `reciprocal_rank_fusion(...)`
  — runs its **own hybrid RRF search** (`:800,:827`), duplicating
  `Polylogue.search_envelope` / `search_similar`.
- `query_unit_rows` / `fetch_attached_units` (`:1686,:1891`) — the facade
  *re-imports the same helpers* in `query_units` (`api/archive.py:2651-2690`).
  Two callers, same substrate helpers, no shared method.

So the facade and the CLI independently reach `archive.query.unit_results` and
`storage.search_providers`. **The thin-client work is mostly deleting
`archive_query.py`'s engine and routing `execute_archive_query` through the
facade**, not writing new logic.

## Classification by subsystem

Legend: **(a)** already on the facade — just switch; **(b)** needs a new
facade/contract method (named); **(c)** genuine presentation/formatting, stays in
the client; **(T)** type/DTO import — keep, but relocate to a shared payload
package for the wire client.

### 1. Query engine / read execution — `archive_query.py`
| Reach | Sites | Class | Action |
|---|---|---|---|
| `ArchiveStore.open_existing` read/stream/unit exec | `archive_query.py:138…1881` | **b** | route all reads through facade; DB open is facade-owned |
| `create_vector_provider` + `reciprocal_rank_fusion` | `:800,:827` | **a** | use `Polylogue.search_envelope` (already fuses RRF) |
| `query_unit_rows`, `query_unit_session_filters`, `fetch_attached_units` | `:1686,:1891` | **a** | `Polylogue.query_units` already wraps these; extend it to return attached-unit rows |
| `ArchiveSessionEnvelope`, `validate_message_type_filter`, `query_unit_descriptor`, `QueryPredicate`, `SessionQuerySpec`, `ArchiveStats` | imports 21-55 | **T** | envelope/stats are output payloads; spec is input payload |

**Verdict: `archive_query.py` should shrink to a formatter.** Its `_session_payload`
/`_session_text` (`:2102,:2138`) are class **(c)** rendering and stay. Everything
that touches `ArchiveStore` moves behind `Polylogue`.

### 2. Query DSL parse / metadata / completions
| Reach | Sites | Class | Action |
|---|---|---|---|
| `compile_expression_into`, `explain_expression`, `build_session_terminal_pipeline`, `terminal_query_sources` | `root_request.py:26,107`, `query_explain.py:11` | **a/b** | facade has `explain_query_expression`; add `parse_query(expression) -> SessionQuerySpec` and `pipeline_plan(expression)` so the client never imports the Lark grammar |
| `query_completion_payload`, `QUERY_COMPLETION_KINDS` | `completions.py:40`, `shell_completion_values.py:19` | **a** | facade `query_completions` exists — switch |
| `count/date/numeric_query_fields`, `QUERY_FIELD_DESCRIPTORS`, `QUERY_*_LANES/ACTION_TYPES`, `describe_spec_selection_fields`, `parse_query_date`, `split_csv` | `click_app.py:158-180`, `click_option_groups.py`, `shell_completion_values.py`, `query_feedback.py:8`, `insight_command_contracts.py:9` | **b** | these drive Click help/completion at import time. Add a **`query_schema()` contract** (fields, lanes, action types, unit kinds) so completion/help is data from one method, not scattered constants. Critical for the composer's `complete(partial)`. |
| `QueryMissDiagnostics`, `diagnose_query_miss` | `query_feedback.py`, `query_output.py` | **a** | facade `diagnose_query_miss` exists — switch |

### 3. Read views / projection
| Reach | Sites | Class | Action |
|---|---|---|---|
| `read_view_choices`, `READ_VIEW_PROFILE_BY_ID` | `read_view_handlers.py`, `read_view_registry.py`, `read_views/base.py` | **b** | facade `list_read_view_profiles` returns docs; add `read_view_profile(id)` + choices so the client owns no viewport registry |
| `ContentProjectionSpec` | `read_views/standard.py:18`, `query_set_read.py:11` | **T→b** | projection is *input*; becomes a wire payload. Reads should accept a projection spec argument on the facade read methods |
| `ArchiveStore.open_existing` inside standard read view | `read_views/standard.py:264,285,315` | **b** | same as §1 — route through facade |
| `SQLiteBackend` chronicle view | `read_views/chronicle.py:24` | **b** | add `Polylogue.chronicle(...)` (or fold into a projection view) |
| `NeighborDiscoveryError`, `SessionNeighborCandidate` | `read_views/neighbors.py` | **a/T** | facade `neighbor_candidates`/`neighbor_candidate_payloads` exist — switch; keep types as payloads |

### 4. Insights / stats / semantic
| Reach | Sites | Class | Action |
|---|---|---|---|
| `build_session_semantic_facts`, `SessionSemanticFacts`, `build_session_profile` | `query_stats.py:456,643`, `query_semantic.py` | **b** | add `Polylogue.session_semantic_facts(session_id)` + `session_profile_record` (partly exists: `get_session_profile_record` at `api/archive.py:3636`) |
| `session_insight_status_sync` | `status.py:17` | **a** | facade `get_session_insight_status` exists — switch |
| `ArchiveStats`, `SessionFilter` | `query_stats.py`, `verb_cardinality.py`, `archive_query.py:38` | **T** | payload/type; `filter` property exists on facade |

### 5. Embeddings — `commands/embed.py` (811 lines)
| Reach | Sites | Class | Action |
|---|---|---|---|
| `storage.embeddings.preflight`, `materialization` (`_EmbedSessionStore`, `mark_all_archive_sessions_needs_reindex`), `status_payload`, `sqlite_vec_support` | `embed.py:31,510,545,594,598`, `shared/embed_runtime.py`, `shared/embed_stats.py` | **b** | `api/embeddings.py` only has `status`/`preflight`. Add an **`EmbeddingOpsSurface`**: `enable`, `disable`, `backfill(window, max_cost)`, `mark_reindex`. embed.py becomes flag-parse + render. |
| `initialize_archive_database`, `ArchiveTier`, `ops_write.upsert_embedding_catchup_run`, `open_readonly_connection` | `embed.py:550,602,761-763` | **b** | fold into the same ops surface; client never opens connections |

### 6. Maintenance / reset / init / GC — `commands/maintenance.py` (2453), `reset.py`
| Reach | Sites | Class | Action |
|---|---|---|---|
| `ArchiveStore`, `archive_init`, `archive_plan.build_archive_init_plan`, `bootstrap.ARCHIVE_TIER_SPECS`, `ops_write.record_ingest_attempt`, `user_write`, `migration_runner`, `execute_materialize_stage`, `ParsingService`, `SessionRepository`, `SQLiteBackend` | `maintenance.py:15,30-57,1359-1362` | **b** | The `WriteSurface`/`MaintenanceSurface` protocols exist but only cover `ingest/run_maintenance/rebuild/tag/delete`. Extend with **`archive_plan()`, `archive_init(yes)`, `run_migration()`, `materialize()`**. This is the biggest single fat command. |
| `blob_gc.run_blob_gc_report/read_gc_history`, `blob_integrity.scan_blob_integrity`, `repair.run_selected_maintenance/preview_counts_from_archive_debt/raw_materialization_replay_backlog` | `maintenance.py:30-31`, `shared/check_workflow.py:38,123`, `shared/check_maintenance.py:11`, `status.py:1094` | **b** | add `MaintenanceSurface.blob_gc()`, `blob_integrity()`, `repair(targets)`, `archive_debt_preview()` (facade `archive_debt` exists for the read half) |
| `escaped_sql_path_prefix_patterns`, `upsert_suppression`, `initialize_archive_database`, `ArchiveTier`, `source_path_native_id_candidates` | `reset.py:15,27-29`, `maintenance.py:15` | **b** | add `Polylogue.reset(scope, suppress=...)` — reset is currently raw SQL in the CLI |

### 7. Status / diagnostics / paths — `commands/status.py` (2161), `paths.py`, `diagnostics.py`
| Reach | Sites | Class | Action |
|---|---|---|---|
| `ArchiveStore` (diagnostics workload), `open_readonly_connection`, `embedding_status_payload`, `raw_materialization_readiness_snapshot`, `missing_source_raw_session_evidence`, `audit_user_overlay_storage`, `INDEX_SCHEMA_VERSION`, `schema_version_mismatch_message` | `diagnostics.py:503`, `status.py:1094-1999`, `status_diagnostics.py:138` | **b** | most of this is the `ops diagnostics workload`/`ops status` payload builder. Add **`Polylogue.diagnostics_workload()`** and **`readiness_snapshot()`** returning typed payloads; the CLI renders them. |
| `archive_layout`, `archive_readiness` helpers, `ARCHIVE_VERSION_BY_TIER`, `ArchiveTier` | `paths.py:12-19`, `status.py:18-19`, `convergence_feedback.py:13` | **b** | `polylogue config paths` should read a **`paths_report()`** payload, not compute layout in the client |

### 8. Runtime types / formatting — `shared/`
| Reach | Sites | Class | Action |
|---|---|---|---|
| `cursor_state.CursorStatePayload`, `run_state.PlanCounts/RunCounts`, `run_activity.session_activity_counts`, `runtime.MessageRecord/ArtifactObservationRecord`, `artifacts.views.ArtifactCohortSummary`, `repair.RepairResult` | `shared/formatting.py`, `shared/check_models.py`, `query_output.py:61` | **T/c** | pure formatting over already-fetched payloads. Keep the **rendering** in `shared/`; the *types* must move to the shared payload package so the wire client can deserialize them. `session_activity_counts` (`formatting.py:126`) is the one that recomputes — push into the status payload. |
| `open_connection` (check_support), `SQLiteBackend`/`SessionRepository` type hints | `shared/check_support.py:75`, `shared/types.py:16-17` | **b/T** | `check_support` opening a connection is a **b**; the `types.py` hints are **T** |

## Migration worklist (ordered — do top-down)

1. **Extend the read contract to cover projection + query schema.**
   Add to `Polylogue` / `read_surface.py`: `parse_query(expr)`,
   `pipeline_plan(expr)`, `query_schema()` (fields/lanes/units/actions),
   `read_view_profile(id)` + `read_view_choices()`, and make read methods accept
   a `ContentProjectionSpec` payload. This unblocks the composer's `complete()`
   and lets §2/§3 switch. *No behavior change; new methods delegate to existing
   `archive.query.*` helpers.*
2. **Collapse `archive_query.py` onto the facade.** Replace
   `ArchiveStore.open_existing` + RRF + unit-row execution with
   `Polylogue.search_envelope` / `query_units`. Keep only
   `_session_payload`/`_session_text` renderers. Largest single win; removes the
   parallel engine. (~10 substrate sites → 0.)
3. **Author the ops/write contract surface.** New protocols +
   facade methods: `EmbeddingOpsSurface` (enable/disable/backfill/mark_reindex),
   `MaintenanceSurface` extensions (archive_plan/archive_init/run_migration/
   materialize/blob_gc/blob_integrity/repair/reset), and
   `DiagnosticsSurface` (diagnostics_workload/readiness_snapshot/paths_report).
   Then rewrite `embed.py`, `maintenance.py`, `reset.py`, `status.py`,
   `paths.py`, `diagnostics.py`, `shared/check_*` as flag-parse + render over
   those payloads. (~15 substrate sites → 0.)
4. **Relocate the DTO vocabulary.** Move (or re-export) the domain models the
   CLI type-imports (`Session`, `SessionSummary`, `Message`, `Action`,
   `ContentProjectionSpec`, `SessionQuerySpec`, `*Payload`, run/cursor/repair
   records) into a single serializable payload package (extend
   `polylogue/surfaces/payloads.py`). CLI imports only that package. This is what
   makes a Go/Rust client possible — one schema surface to generate stubs from.
5. **Enforce with the contract assertions.** Add `assert_implements` for the new
   surfaces and a lint/test that fails if `polylogue/cli/` imports
   `polylogue.storage`, `polylogue.pipeline`, or `polylogue.archive.query`
   *at runtime* (allow `surfaces.payloads` + `api`). This is the durable guard
   that keeps the CLI thin after t46.

## Contract-surface gaps that must be filled FIRST

These do not exist on the facade today and block the thin client:

- **Query schema/parse contract**: `parse_query`, `pipeline_plan`,
  `query_schema`, `read_view_choices` — client currently imports the Lark
  grammar and field-descriptor constants directly (§2). Highest priority: the
  composer's live `complete(partial)`/`preview(spec)` needs exactly this.
- **Embedding ops surface**: `api/embeddings.py` is read-only
  (status/preflight). No enable/disable/backfill/mark-reindex (§5).
- **Maintenance ops surface**: `WriteSurface` covers ingest/rebuild/tag/delete
  only. No archive_plan/init/migration/materialize/blob_gc/integrity/repair/
  reset (§6). `reset` is raw SQL in the CLI.
- **Diagnostics/status payload surface**: `ops status`, `ops diagnostics
  workload`, `config paths`, readiness all compute in the client from
  `ArchiveStore`/readonly connections (§7). Need typed payload methods.
- **Projection as input**: reads must accept a `ContentProjectionSpec` payload
  arg so the client stops importing `archive.semantic.content_projection` and
  the composable-projection direction (fewer named views) has a wire home.

## What legitimately stays in the client (class c)

Rendering only: `archive_query.py:_session_payload/_session_text`,
`shared/formatting.py` (over payloads), `shared/check_rendering_*.py`,
`shared/cost_rendering.py`, `shared/resume_rendering.py`, `shared/schema_rendering*`,
Click option/group wiring, terminal/plain/json emit. These consume payloads and
format; they never touch `storage`/`pipeline`. Domain-model *types* they render
are class (T) and move to the payload package but the rendering code stays.

## Open questions for the operator

- **Grammar location.** Does the Lark parser run in the daemon only (client sends
  raw expression string, daemon returns spec + completions), or does the thin
  client keep a copy for offline `--no-daemon`? Recommendation: **daemon-only
  parse**; `--no-daemon` imports the facade in-process (accepted floor).
- **DTO transport.** Generate wire schemas from `surfaces/payloads` (JSON) now,
  or keep Python pickling until a non-Python client is real? Recommendation:
  JSON payloads from step 4 regardless, so the protocol boundary is honest.
- **`ops.db`/write reach.** Some ops writes (`record_ingest_attempt`,
  `upsert_embedding_catchup_run`) are daemon-telemetry writes the CLI does
  directly. In the hot-daemon model these should be daemon-side only — confirm
  the CLI never writes `ops.db` once the daemon is required.
