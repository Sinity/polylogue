# Evidence ledger

## Evidence authority and provenance

The question is defined by `05-duplicate-authority-census(1).md`. Evidence was adjudicated in this order: current source and focused execution, repository instructions, current Beads records, then the explicitly slightly stale test-suite context.

Input provenance:

| Input | Size | SHA-256 |
|---|---:|---|
| `00-polylogue-all.tar(14).gz` | 92,457,778 bytes | `3d0c5a2e5024632c05438dbc3a08aae1dcd31529472b1185ad5b80c0cd933e30` |
| `00-slightly-stale-context-testsuite-diet.tar(14).gz` | 934,006 bytes | `d4fa7fc31c70ff30db076e526836ffc81b8124d3f54c7add5e58d3f57c9dcbff` |
| `05-duplicate-authority-census(1).md` | 3,287 bytes | `8b96d48996e1dc34d2b5e482a65243a475fe074f8fb2bf40f5ee19b016fb8774` |

The snapshot manifest names `master` commit `f654480cadb7cc4c194704e24dfd483199547b35`, generated `2026-07-17T043202Z`, and reports a dirty working tree. The supplied working-tree tar contained 2,501 of 3,739 tracked paths; 1,238 omitted paths were predominantly excluded `.agent/` history. All 2,501 included tracked paths matched the Git bundle’s HEAD byte-for-byte.

The stale context’s L27 states the relevant law precisely: a production authority should enumerate and dispatch a capability, real dispatch and schema introspection should prove it, fresh and upgraded state should agree, and a test must not create a third mirror catalog. That context was used as a search seed, not as proof that an old defect still existed.

## Method

The census combined source tracing, AST inventory, runtime introspection, fresh archive initialization, actual CLI and MCP invocation, and focused tests. A repeated string was counted only when separate copies could affect execution, generated output, validation, operator status, compatibility behavior, or contributor edits.

For each candidate, the trace sought:

- writers or declaration owners;
- generators or projections;
- validators and their proof strength;
- real consumers and dispatch routes;
- current tracker decisions and acceptance criteria;
- an exact deletion gate rather than a general preference for deduplication.

## E01 — MCP query recipes contradict executable grammar

**Observed facts**

- `polylogue/mcp/server_prompts.py:456-468` emits a `query_units` recipe whose expression begins `sessions where ... exists action(...)`.
- `polylogue/mcp/server_prompts.py:470-484` emits another sessions-only recipe before a valid `files where ...` recipe.
- `polylogue/mcp/server_tools.py:268-320` calls `parse_unit_source_expression` and returns `invalid_query` when it returns `None`.
- `polylogue/archive/query/unit_results.py:160-178` enforces the same terminal-unit requirement for shared execution.
- `tests/unit/mcp/test_server_surfaces.py:1014-1024` explicitly tests rejection of a nonterminal/session expression.
- `tests/unit/mcp/test_server_surfaces.py:1394-1439` tests only that cookbook prompt text names expected tools; it does not compile or execute embedded expressions.

**Executable witness**

The real read-role MCP server rendered both prompts and each embedded recipe was passed back to its real `query_units` tool on a fresh archive:

- `unacknowledged_failures`: sessions-only expression → `is_error=true`, `code=invalid_query`.
- `sessions_touching_file`: sessions-only expression → `invalid_query`.
- `sessions_touching_file`: `files where path:...` expression → successful envelope.

A direct parser probe likewise returned `None` for both sessions-only forms and a `QueryUnitSource` for files-terminal and sessions-scope-pipeline forms.

**Writer / validator / consumer trace**

- Writer: literal prompt recipes in `server_prompts.py`.
- Executable authority: parser in `archive/query/expression.py` and shared request builder in `archive/query/unit_results.py`.
- Validator: prompt-name test and independent query rejection test; together they currently prove the contradiction rather than prevent it.
- Consumer: a model following the MCP prompt and invoking the registered `query_units` tool.

**Tracker**

`polylogue-z9gh.3` is open P0. Its current notes independently identify the same two product recipes as an active curriculum regression and require recipes to derive from executable declarations.

**Inference**

The parser and shared request builder are the current behavior authority. The prompt literals are stale product curriculum copies, not alternative supported syntax.

**Unresolved evidence**

The Bead says an installed shared skill repeats the expressions. That installed artifact was not present in the supplied evidence, so this report does not claim to have verified it.

## E02 — Maintenance target catalog and executors disagree

**Observed facts**

- `polylogue/maintenance/targets.py:158-234` declares seven `MAINTENANCE_TARGET_SPECS`, including `superseded_raw_snapshots`.
- `polylogue/maintenance/replay.py:101-128` declares six `_REPLAY_DISPATCH` handlers and omits that target.
- `polylogue/maintenance/replay.py:607-618` converts the missing handler into `UnsupportedReplayTargetError`.
- `polylogue/storage/repair.py:6133-6152` has seven preview handlers and seven repair handlers, including the omitted target.
- CLI uses `execute_replay` at `polylogue/cli/commands/maintenance/_run.py:116-124`.
- MCP uses `execute_backfill` at `polylogue/mcp/server_maintenance_tools.py:134-157`.
- HTTP uses `execute_backfill` in `polylogue/daemon/http.py:4482-4499`.
- The existing coverage test at `tests/unit/maintenance/test_idempotency.py:216-224` checks only a three-target subset, so the seventh target can be missing while the test passes.

**Executable witness**

On a freshly initialized five-tier archive:

- `execute_replay(... targets=('superseded_raw_snapshots',), dry_run=True)` returned status `failed` with `UnsupportedReplayTargetError`.
- `execute_backfill` with the same target returned status `completed` and the real handler result, `Would: delete 0 superseded raw snapshots`.
- The actual command `polylogue ops maintenance run --target superseded_raw_snapshots --dry-run --output-format json` emitted a failed operation envelope but exited with code `0`.

**Writer / validator / consumer trace**

- Identity/metadata writer: `MAINTENANCE_TARGET_SPECS`.
- Competing execution writers: `_REPLAY_DISPATCH`, `_PREVIEW_HANDLERS`, `_REPAIR_HANDLERS`.
- Validators: subset coverage test and per-path tests; no equality test through each public surface.
- Consumers: CLI resumable replay, MCP planner, HTTP planner, readiness/doctor projections.

**Tracker**

`polylogue-71ey` is open P1 and explicitly requires the maintenance catalog to own replay semantics.

**Inference**

This is not a target that lacks an implementation. The implementation exists and is reachable through another production orchestrator. The duplicate registry is the defect.

## E03 — Config effective values have multiple authorities

**Observed facts**

- Legacy `Config`, `get_config`, and `IndexConfig.from_env` live at `polylogue/config.py:80-164`.
- Layered `PolylogueConfig` begins at `config.py:172`.
- `_CONFIG_INVENTORY` at `config.py:475-915` contains 50 keys and exactly matches the 50 built-in default keys.
- Type conversion is separately authored in `_INT_CONFIG_KEYS`, `_FLOAT_CONFIG_KEYS`, and `_BOOL_CONFIG_KEYS` at `config.py:917-944`.
- `load_polylogue_config` documents and implements five layers at `config.py:1105-1155`.
- `_merge_toml` assigns nested health tables wholesale at `config.py:1215-1227`.
- AST inventory found 16 direct reads of seven already-inventoried `POLYLOGUE_*` variables outside `config.py`, across 11 files. Eight are in production package files and eight in devtools. It also found 30 reads of 23 non-inventory variables, which require classification rather than automatic migration.
- Direct production examples include `polylogue/ui/theme.py:280`, `polylogue/cli/commands/status.py:76,141`, `polylogue/daemon/http.py:374`, and `polylogue/pipeline/services/ingest_batch/_core.py:1421`.
- A repository search found 37 textual `get_config()` call sites including the definition and service-hook calls; legacy configuration is still live across daemon, MCP, CLI, demo, and devtools paths.

**Executable witness**

A site TOML declared:

- `warning_total = 10`
- `critical_total = 20`
- `families.index = 3`

A user TOML overrode only `warning_total = 12`. `load_polylogue_config` resolved `health_convergence_debt` to `{'warning_total': 12}` with provenance `user`, deleting both inherited siblings.

**Writer / validator / consumer trace**

- Surface metadata writer: `_CONFIG_INVENTORY`.
- Default writer: `_default_config_values`.
- Type writers: three coercion sets and nested downstream decoders.
- Competing effective-value writers: legacy `Config`/paths, layered resolver, and direct environment reads.
- Consumers: CLI defaults, daemon startup and status, theme resolution, ingest schema validation, MCP services, devtools.
- Current validators establish inventory/default equality but do not enforce all consumers through the resolver or nested-key preservation.

**Tracker**

`polylogue-9gh1` is an open P1 epic for the documented-versus-actual five-layer config gap. The stale test-law seed specifically retained the partial-health-override sibling-loss witness; current execution confirms it remains live.

**Inference**

The inventory is a useful surface projection but is not yet a single authority for type, default, merge semantics, or effective runtime value.

## E04 — Two live embeddings DDL and writer contracts are incompatible

**Observed facts**

- `polylogue/storage/search_providers/sqlite_vec_runtime.py:57-105` creates `message_embeddings` with auxiliary columns `source_name` and `session_id`, plus non-STRICT metadata/status tables.
- `polylogue/storage/sqlite/archive_tiers/embeddings.py:5-58` creates the same vec0 table with `session_id` and `origin`, STRICT metadata/status tables, and an additional `embedding_failures` table.
- Fresh archive initialization consumes the tier DDL through `ARCHIVE_DDL_BY_TIER` and `initialize_archive_tier` at `archive_tiers/__init__.py:14-27` and `archive_tiers/bootstrap.py:75-93`.
- Legacy provider writes `source_name` at `search_providers/sqlite_vec_queries.py:62-96`.
- Archive-tier materialization writes `origin` at `archive_tiers/embedding_write.py:107-145`.

**Executable witness**

With the actual sqlite-vec extension and production initializers:

- Tier DDL first, then runtime ensure: schema contains `origin`; a legacy provider-form insert fails `OperationalError: table message_embeddings has no column named source_name`.
- Runtime ensure first, then tier initializer: schema contains `source_name`; an archive-tier insert fails `OperationalError: table message_embeddings has no column named origin`.
- `CREATE ... IF NOT EXISTS` means the second schema writer silently leaves the first shape in place.

**Writer / validator / consumer trace**

- Writers: runtime mixin `_ensure_tables` and archive-tier `EMBEDDINGS_DDL`.
- Competing data writers: `SqliteVecProvider` query mixin and `upsert_message_embeddings`.
- Consumers: similarity/search provider, archive embedding materialization, demo seeding, daemon status/convergence, repair and excision paths.
- No create-order parity test protects the shared table shape.

**Tracker and doctrine**

`polylogue-mhx.7` is open P2 and describes the same two live definitions. Repository instructions state that embeddings are a derived tier whose current schema must come from canonical DDL plus rebuild, not a parallel runtime schema writer.

**Inference**

The archive-tier DDL is the intended canonical current schema. A runtime dimension parameter may be required, but it must be a parameter to the same builder rather than a second schema.

## E05 — MCP tool declaration constellation

**Observed facts**

Runtime introspection of `build_server` found:

- read role: 66 tools;
- write role: 95 tools;
- admin role: 104 tools.

The admin set exactly equals `tests/infra/mcp.py:EXPECTED_TOOL_NAMES`, and the 104-key `TOOL_CONTRACT` exactly covers it. `_KNOWN_MINIMAL` contains 66 entries and exactly covers the read role after its one explicit no-kwargs exception.

The direction of authority is still wrong:

- `devtools/render_mcp_tool_index.py:1-7,23-26` calls the test fixture the registered tool set and generates reader docs from it.
- `devtools/verify_docs_coverage.py:11-21` also imports the test fixture for MCP inventory.
- `CLAUDE.md:209-214` and `CLAUDE.md:478-479` tell contributors there are 103 tools and require manual updates to both expected names and contracts. The live count is 104.

**Writer / generator / validator / consumer trace**

- Execution writers: decorators spread across `polylogue/mcp/server_*.py`.
- Validation copies: `EXPECTED_TOOL_NAMES`, `TOOL_CONTRACT`, `_KNOWN_MINIMAL`.
- Generator: MCP docs index imports `EXPECTED_TOOL_NAMES` from tests.
- Consumers: MCP clients, role-based servers, docs readers, contributor workflow.

**Focused proof**

The parametrized real read-role smoke test invoked all 66 tools with minimal arguments against an empty archive and passed. Current alignment is therefore real, not inferred from set equality alone.

**Tracker**

`polylogue-o21` and `polylogue-t46.8.1` are open P1 declaration/algebra work. This finding should feed those existing designs rather than create another registry.

**Inference**

This is broad edit-cost and authority-direction debt, with one current manual documentation drift. It is not a claim that the live tool sets are presently mismatched.

## E06 — Daemon route contracts do not own POST/DELETE dispatch

**Observed facts**

- `polylogue/daemon/route_contracts.py:1-7` explicitly says the module is descriptive and dispatch remains in `daemon.http`.
- Runtime introspection found 77 contract method/pattern pairs and 77 implemented pairs, with exact equality: 56 GET, 15 POST, 6 DELETE.
- GET route structs carry a `RouteContract` and builders resolve it at `daemon/http.py:115-137,235-258`.
- `_StaticPostRoute` re-declares only pattern, segments, and handler at `daemon/http.py:140-145`.
- POST route groups are authored at `daemon/http.py:353-406`; `implemented_daemon_route_patterns` re-enumerates them at `409-430`.
- `do_POST` applies auth by hard-coded route grouping at `daemon/http.py:1670-1749`; `do_DELETE` applies another hard-coded policy at `1762-1784`.
- Search of production source found no runtime read of daemon `RouteContract.auth_policy`; it is consumed by OpenAPI generation and tests.

**Validators**

`tests/unit/daemon/test_route_contracts.py:31-62` proves exact pattern parity and GET binding. Security tests derive route sets from contracts and exercise actual HTTP auth behavior, so the split is meaningfully guarded today. The residual defect is that changing dispatch and contract requires coordinated edits rather than one executable declaration.

**Tracker**

`polylogue-3utv` is the existing typed route-registry plan. Its current P3 priority is consistent with the absence of a demonstrated live mismatch.

## E07 — Static facade routing status omits live methods

**Observed facts**

- `polylogue/cli/commands/status.py:298-509` hand-authors `_ARCHIVE_FACADE_ROUTES`.
- `_archive_facade_route_status` reports source `static_facade_route_catalog` at `status.py:512-532`.
- `_archive_runtime_path_status` derives `archive_routing_ready` from that map at `status.py:558-600`.
- Runtime introspection found 146 public callable attributes on `Polylogue` and 141 map entries. No stale map entry named a nonexistent method. Five public callables were absent: `embedding_preflight`, `embedding_status`, `filter`, `iter_messages`, and `open`.

**Inference**

Some omissions may be intentional lifecycle/query-builder exclusions, but the current map has no explicit exclusion authority. Because `close` is classified while `open` is absent and the two embedding status methods are user-visible, the status result cannot distinguish an intentional exclusion from drift.

**Tracker**

`polylogue-s1kr` requires semantic-operation parity rather than a fossilized method-name matrix. `polylogue-9e5.31` requires definition-to-production closure but warns against letting authored definitions certify authored registrations.

## E08 — Devtools command catalog has three real bypasses

**Observed facts**

- `devtools/command_catalog.py:881-899` registers `lab policy backlog-hygiene`.
- `.github/workflows/mutation-testing.yml:52` invokes `python -m devtools.verify_mutation_freshness` directly.
- `.github/workflows/nightly-scale.yml:64` invokes `devtools/benchmark_compare_nightly.py` as a raw script.
- `devtools/pre_push_gate.py:99-109` invokes `python -m devtools.verify_backlog_hygiene` instead of its own catalog name.
- `devtools/click_dispatch.py:18-23` builds the executable command surface from `COMMAND_SPECS`.

**Tracker**

`polylogue-l8ee` is open P3 and names these exact three bypasses. No source evidence contradicted it.

**Inference**

The catalog is the execution authority for registered commands, but it is not a complete authority for devtools capabilities while direct production workflow and hook paths remain.

## E09 — Durable user schema copies are legitimate; its validator is incomplete

**Observed facts**

- Repository instructions require additive numbered migrations for durable tiers and canonical DDL plus rebuild for derived tiers (`CLAUDE.md:162-173`).
- `USER_DDL` is the fresh current schema; `polylogue/storage/sqlite/migrations/user/004_user_settings.sql` and later migrations are required upgrade compatibility paths.
- A freshly initialized user tier contains 15 non-SQLite tables.
- `tests/unit/storage/test_archive_tiers_assertions.py:155-189` asserts a hand-written required subset of 13 tables and a legacy-overlay denylist.
- The fresh tables `holdout_access_receipts` and `result_set_holdout_policies` are not in the required subset.

**Inference**

The migrations are not dead duplicates and must remain. The hand list is an incomplete projection of current schema authority and cannot prove exact fresh-versus-upgraded convergence.

**Tracker**

`polylogue-ihp0` is open P2 and calls for durable-tier schema assertions derived from canonical inventory.

## E10 — `user_settings` is declared but has no runtime consumer

**Observed facts**

Repository search found `user_settings` in:

- fresh DDL at `archive_tiers/user.py:288-294`;
- migration `storage/sqlite/migrations/user/004_user_settings.sql`;
- status table probes at `cli/commands/status.py:273-279`;
- schema/internals docs;
- migration/schema/status tests.

No production API, storage helper, reader, writer, or consumer references the table. Tests insert/select directly at the migration layer, which proves the table exists but not that a product operation uses it.

**Tracker decision**

`polylogue-at44` is open P3. Its current decision is explicit: keep the separate table, define typed allowed keys, wire sync and async storage helpers, and make subscription-tier cost computation the first consumer. This report preserves that decision.

## Validated non-defect — Read-view projections

`READ_VIEW_PROFILES`, `READ_VIEW_HANDLER_METADATA`, and `READ_VIEW_HANDLERS` each contain 11 view IDs. `polylogue/cli/read_view_registry.py:78-94` fails fast on ID drift, and `polylogue/cli/read_view_handlers.py:174-209` additionally compares duplicated metadata fields. The split exists to keep static Click option ownership import-light. It is a compatibility/import-layer projection with an executable gate, not a high-value consolidation target.

## Focused test record

Two focused pytest invocations completed successfully:

1. Eleven tests passed in 3.02 seconds: maintenance subset coverage, query-unit rejection, six cookbook prompt-name cases, two route parity tests, and the read-tool minimal-kwargs census.
2. Sixty-eight tests passed in 11.72 seconds: all 66 read-role MCP tools invoked with minimal arguments, plus two fresh user-tier schema tests.

Total: **79 focused tests passed**.

The passing result is important but not exculpatory. The custom production-route probes demonstrated that the maintenance subset test, prompt-name test, and user required-table subset can all pass while the targeted semantic defect remains.

## Missing evidence

- No live deployment, real operator archive, browser session, hosted CI run, or daemon process was inspected.
- No installed out-of-repository shared skill was available.
- No full test suite was run; repository instructions explicitly prefer focused affected tests.
- Fresh-versus-every-supported-version migration parity was not executed in this iteration.
- The direct environment census classifies call sites, not intent. The 23 non-inventory variables must be adjudicated as sanctioned runtime controls or config omissions before migration.
