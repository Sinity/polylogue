# Config Resolution Closure — Engineering Handoff

## Mission and outcome

This package closes the gap between Polylogue's documented five-layer configuration model and the configuration actually consumed by runtime routes. The implemented order is:

1. built-in defaults;
2. site TOML;
3. user/project TOML;
4. environment variables;
5. explicit CLI overrides.

`load_polylogue_config()` is now the sole five-layer value resolver. A single immutable `ResolvedRuntimeConfig` captures the resolved settings, absolute archive/tier paths, source paths, compatibility projections, process bootstrap paths, and the active `ArchiveIdentity`. CLI, daemon, MCP, API, and maintenance composition roots install or receive that projection instead of reconstructing paths or secrets independently.

The patch also deep-merges the two nested health-table settings, preserves array-of-table replacement semantics, migrates actual `POLYLOGUE_*` runtime settings away from direct environment reads, preserves provenance and redaction, and adds a mutation-sensitive inventory-driven regression suite.

## Snapshot identity

- Branch recorded by the supplied snapshot: `master`
- Commit: `536a53efac0cbe4a2473ad379e4db49ef3fce74d`
- Subject: `fix(repair): harden raw authority convergence (#3046)`
- Snapshot generation time: `2026-07-17T18:09:50Z`
- Source archive SHA-256 recorded by the snapshot: `ec3fa193b2f99a6daee0bc620af4f69b5728834e99f8196792a17c3fbae11155`
- Snapshot manifest marked the source dirty. The embedded working-tree layer used for this patch matched the named commit for tracked files; the unrecovered dirty state appears to have been ignored/omitted runtime material such as `.beads`, `.claude`, or other non-patch state. No unknown dirty tracked delta was folded into `PATCH.diff`.

`PATCH.diff` is based directly on the commit above. It has 73 file patches, 2,104+ inserted lines including the new test module, and 849 removed lines. It applies cleanly to a fresh detached worktree at the named commit.

## Evidence inspected

Repository instructions and test conventions:

- `CLAUDE.md`
- `TESTING.md`
- `CONTRIBUTING.md`

Configuration and runtime graph:

- `polylogue/config.py`
- `polylogue/paths/_roots.py` and `polylogue/paths/__init__.py`
- `polylogue/services.py`
- CLI root, shared environment, command routing, and maintenance commands
- daemon root, HTTP/UDS servers, health loops, watcher/capture routes, and background workers
- MCP server construction, server support, call logging, and tool configuration
- public API facade and insight routes
- ingest, validation, embedding, Drive, hook, live-source, parser, logging, readiness, and UI consumers

Tests inspected and exercised include configuration, inventory, CLI, daemon security and health, MCP envelopes and runtime, API facade contracts, pipeline batching/runtime, Drive authentication, logging, and embedding status/provider behavior.

Beads records read in full from `.beads/issues.jsonl`:

- `polylogue-9gh1` — epic requiring closure of split authority, nested-table replacement, bypassing callers, and a future-setting regression guard;
- `polylogue-fd2s` — decided architecture: one resolver plus immutable projections injected at CLI/daemon/MCP/API/maintenance roots;
- `polylogue-cxlk` — deep merge for health tables and nested `families`, while array-of-table values retain replacement semantics;
- `polylogue-uu8r` — classify every direct environment read and migrate actual product settings.

Relevant history inspected includes the configuration/path ownership introduction and later diagnostics work, notably `8b2467629` (`typed PolylogueConfig and env-read consolidation`), `cf1a25ab6` (`split config and path ownership`), `fc70163ad` (`report effective configuration debt`), `d70b0d3fb` (`audit env-provided path diagnostics`), and subsequent archive/path refactors.

## Prior beads-01 adjudication

The earlier external attempt was inspected through:

`.agent/handoffs/external-agent-campaigns/2026-07-16-gpt-pro-wave/beads/results/beads-01/a01/result.json`

Its result was rejected and contained a 35,210,095-byte copied repository package with a zero-byte patch. Its source artifact was not present in the supplied snapshot. No implementation code was reused. The useful evidence retained was limited to its rejection reason and scope signal: a valid handoff must be an apply-ready patch rather than a copied source tree.

## Mechanism

### One resolution boundary

`load_polylogue_config()` captures the bootstrap inputs once: environment mapping, current directory, home directory, and XDG-derived roots. It then applies defaults, site TOML, user TOML, environment values, and explicit CLI values in order. Layer provenance is assigned by key presence, not by value inequality, so an explicit override that happens to equal the lower-layer value is still correctly reported as `env` or `cli`.

Explicitly selected malformed site/user files raise `ConfigError`. Automatically discovered optional files retain fail-open behavior when absent or malformed, preserving the existing optional-discovery contract.

### Immutable runtime projection

`resolve_runtime_config()` produces a frozen `ResolvedRuntimeConfig` containing:

- frozen `PolylogueConfig` settings and per-key provenance;
- `ResolvedArchivePaths` for archive root, all SQLite tiers, render/blob/inbox roots, and daemon socket;
- `ResolvedSourcePaths` for source roots and source-specific paths;
- compatibility `Config`, `IndexConfig`, and `DriveConfig` projections;
- resolved browser-capture, hook-sidecar, backup-temporary, and Antigravity executable paths;
- captured cwd/home/XDG roots;
- the active shipped `ArchiveIdentity`.

Nested mappings and sequences are recursively frozen. Compatibility projections return defensive mutable copies where old APIs require lists, preventing caller mutation from changing the authority snapshot.

`get_config()`, `get_index_config()`, `get_drive_config()`, `get_sources()`, and `polylogue.paths` now project the installed runtime. They do not independently read environment variables, `Path.cwd()`, `Path.home()`, or XDG state.

### Nested TOML merge

For inventory entries whose `toml_kind` is `table`, later layers recursively merge mappings. This preserves top-level siblings and recursively merges `families`, so overriding one threshold or one family member does not delete unrelated site-layer values.

`subscription_plans` remains `toml_kind="array-table"` and therefore replaces the lower-layer array as a whole. No list concatenation or element-level merge was introduced.

### Provenance and redaction

`effective_config_payload()` accepts either settings or a complete runtime. Runtime payloads report all resolved tier paths and archive identity. Inventory metadata, supplying layer, defaults, owner/reload metadata, diagnostics, and secret-presence reporting remain available. Secret values are rendered only as `<set>` or `<unset>` and are not serialized into output.

## Composition-root injection map

| Root | Resolution/install point | Projection passed to production routes | Closure proof |
|---|---|---|---|
| Main CLI | `polylogue/cli/click_app.py` resolves explicit Click values, calls `resolve_runtime_config()`, then `install_runtime_config()` | `AppEnv(runtime=runtime)`; commands receive `AppEnv`, `RuntimeServices`, or the installed projection | Dynamic cross-root test plus daemon/maintenance Click-root test |
| Daemon | `polylogue/daemon/cli.py` distinguishes explicitly supplied Click values from defaults, resolves once, installs once | `run_daemon_services(runtime=runtime)`; watcher, API, UDS, health, maintenance, embedding, and background consumers project that snapshot | Real Click invocation captures the runtime and asserts user-TOML archive root |
| MCP | `polylogue/mcp/server.py` builds/accepts `RuntimeServices`; `server_support` requires installed services | MCP handlers obtain the injected `Config`; API facade is created with `services.runtime` | MCP runtime and envelope suites plus cross-root projection test |
| Public API | `Polylogue(runtime=...)` in `polylogue/api/__init__.py` | Installs the supplied runtime and builds `RuntimeServices(runtime=runtime)`; explicit archive/db arguments remain a library/test seam with conflict checks | Cross-root projection test and API contract suite |
| Maintenance | Main CLI root supplies `AppEnv.runtime`; maintenance code uses that installed authority and projected paths | Archive plan and related commands consume the same archive/tier projection as the invoking CLI | Real `ops maintenance archive-plan` invocation asserts user-TOML archive root |
| Standalone browser capture | `polylogue/daemon/browser_capture.py` resolves only Click values whose source is not a default | Installed runtime drives host, port, spool, auth, origins, and remote-access policy | Existing receiver/security tests and AST bypass guard |
| Shared service scope | `build_runtime_services(runtime=...)` in `polylogue/services.py` | `RuntimeServices` stores the runtime and derives `Config`, database path, source definitions, and facade construction from it | API/CLI/MCP service agreement test |

Click defaults are not treated as layer-five overrides. A TOML value therefore survives unless the operator explicitly supplies the corresponding option.

## Bypass-caller inventory and migration

The table groups call sites by setting family while naming every production file changed for a direct or parallel-authority configuration path.

| File(s) | Setting or bypass shape | Migration |
|---|---|---|
| `polylogue/paths/_roots.py`, `polylogue/paths/__init__.py` | archive root, tier paths, render/blob/inbox roots, runtime/socket roots, hook paths | Removed independent env/home/XDG/cwd resolution; all helpers project `ResolvedRuntimeConfig` |
| `polylogue/services.py`, `polylogue/cli/shared/types.py` | legacy `Config` construction and service db/source reconstruction | `RuntimeServices` and `AppEnv` carry the runtime projection |
| `polylogue/cli/click_app.py` | CLI flags, force-plain/no-color/theme/timing, archive root | Root constructs and installs layer five only from explicit parameter sources |
| `polylogue/cli/archive_query.py`, `commands/dashboard.py`, `facets.py`, `import_command.py`, `status.py`, `tutorial.py` | daemon disablement/mode, daemon URL, Unix socket | Read resolved settings/paths instead of environment or local defaults |
| `polylogue/cli/commands/demo.py` | whether archive root was explicitly configured | Uses provenance from the installed settings |
| `polylogue/cli/commands/embed.py`, `polylogue/pipeline/run_stages.py`, `polylogue/storage/search_providers/__init__.py` | `VOYAGE_API_KEY`/provider construction | Removed Click/env fallback; TOML/env value flows through `IndexConfig` into the real embedding stage and provider factory |
| `polylogue/cli/shared/formatting.py`, `polylogue/logging.py`, `polylogue/readiness/__init__.py`, `polylogue/ui/theme.py` | force plain, `NO_COLOR`, theme, debug/presentation behavior | Presentation settings are resolved once; only terminal capability metadata remains ambient |
| `polylogue/daemon/cli.py` | daemon host/ports, debounce, API/capture policy, embedding enablement, health settings | One daemon runtime resolved before any service/thread starts |
| `polylogue/daemon/browser_capture.py` | capture host/port/spool/auth/remote/origins | Standalone root resolves explicit CLI overrides; server consumers use settings projection |
| `polylogue/daemon/backup.py` | `POLYLOGUE_BACKUP_VERIFY_TMPDIR` | Uses `runtime.backup_verify_tmpdir` |
| `polylogue/daemon/http.py` | archive root, API/capture ports, observability flag, OTLP body cap | Uses installed runtime/settings; effective-config endpoint receives resolved projection |
| `polylogue/daemon/thread_continue.py` | `POLYLOGUE_READER_AGENT_TEMPLATES` | Added inventory/TOML property and reads it from runtime; explicit mapping retained only as deterministic parser-test seam |
| `polylogue/daemon/convergence_debt_alert.py`, `convergence_stages.py`, `cursor_lag_alert.py`, `cursor_lag_anomaly.py`, `cursor_lag_status.py`, `embedding_backlog.py`, `embedding_readiness.py`, `health.py`, `similarity.py` | health tables, intervals/tiers, embedding model/readiness, archive paths | Consume frozen settings/config rather than invoking a second loader or path resolver |
| `polylogue/daemon/uds.py` | Unix socket and archive config | Uses the daemon-installed projection |
| `polylogue/mcp/server.py`, `server_support.py`, `call_log.py` | MCP archive/db paths and call-log config | Server requires injected `RuntimeServices`; call logging uses installed settings |
| `polylogue/api/__init__.py`, `polylogue/api/insights.py` | API archive/db construction and executor propagation | Accepts runtime directly; executor-backed operations retain/bind the same snapshot |
| `polylogue/pipeline/services/archive_ingest.py` | commit batch size and parse worker count | Added inventory/TOML accessors and reads installed settings |
| `polylogue/pipeline/services/ingest_batch/_core.py`, `validation_flow.py` | schema validation and Sinex publication mode | Use installed settings, preserving existing env values as layer four |
| `polylogue/sources/drive/auth.py`, `polylogue/readiness/__init__.py` | Drive credential/token paths | `DriveConfig` projects resolved absolute paths; explicit paths retain historical return semantics even before files exist |
| `polylogue/sources/hooks.py`, `polylogue/hooks/__init__.py` | hook sidecar directory | Added inventory/TOML path and projected helper |
| `polylogue/sources/live/batch_support.py` | full-ingest worker cap | Added inventory/TOML accessor and runtime read |
| `polylogue/sources/parsers/antigravity.py` | Antigravity language-server executable | Added inventory/TOML path and runtime projection |
| `polylogue/storage/embeddings/materialization.py`, `preflight.py`, `status_payload.py` | model, enabled/cost/key settings, database path | Consume settings carried by compatibility config/runtime; legacy explicit test/library objects fall back to the installed runtime without rereading raw env |

### Direct environment reads deliberately retained

A repository-wide AST/grep audit leaves only boundary metadata or configuration-file discovery outside `config.py`:

| File | Variable(s) | Classification |
|---|---|---|
| CLI completion files | `COMP_WORDS` | shell protocol input, not product configuration |
| `cli/commands/embed.py` | `POLYLOGUE_CONFIG` | explicit config-file selection/writer boundary, not an effective runtime value |
| `coordination/envelope.py` | Polylogue/Codex/Claude session and correlation identifiers | per-request/process correlation metadata |
| daemon CLI/HTTP | `POLYLOGUE_DEV_LOOP_RUN_ID`, `POLYLOGUE_DEV_LOOP_LOG_DIR` | development-loop invocation metadata, not an operator setting |
| `demo/workspace.py` | temporary arbitrary environment mutation | subprocess/demo isolation mechanism |
| `hooks/__init__.py` | `CLAUDE_CONFIG_DIR`, `CODEX_HOME`, `POLYLOGUE_HOOK_PROVIDER` | external tool installation/provider boundary; not five-layer Polylogue runtime state |
| `readiness/__init__.py` | `TERM` | terminal capability metadata |
| `storage/archive_identity.py` | `INVOCATION_ID` | systemd invocation identity |
| `ui/theme.py` | `COLORFGBG` | terminal auto-detection signal used only when resolved theme is `auto` |

The regression test fails if any inventoried environment variable is read directly by a runtime module outside `config.py`.

## Behavior-change census

Environment-backed behavior is preserved because environment remains layer four and still outranks both TOML layers. The repository cloud-shell `POLYLOGUE_ARCHIVE_ROOT` case has an explicit regression test proving that it wins over user TOML.

Intentional corrections that may change effective behavior for users relying on the previous bugs:

1. `archive.root` in site/user TOML now changes the archive used by real CLI, daemon, MCP, API, and maintenance routes. Previously many routes ignored it.
2. `embedding.voyage_api_key` in TOML now reaches embedding execution. An environment variable is no longer required when TOML supplies the key.
3. Newly inventoried settings—daemon/client routing, browser capture, batching/workers, Drive paths, hook sidecar, backup temp directory, Antigravity executable, UI timing/theme/plain mode, and reader templates—now honor site/user TOML before environment overrides.
4. Health tables retain lower-layer sibling thresholds and family overrides instead of deleting them on a partial later-layer table.
5. Click defaults no longer overwrite TOML merely because Click supplied a default value internally.
6. Equal-valued explicit overrides now report their true supplying layer.
7. Runtime values are snapshots. Mutating `os.environ`, cwd, home, or XDG variables after construction does not change paths or secrets until a new runtime is explicitly resolved/installed.
8. Explicitly selected malformed TOML fails with `ConfigError` instead of being silently ignored.
9. Explicit source roots now reach the runtime source projection.

## Regression-guard design

`tests/unit/core/test_config_runtime_closure.py` is inventory-driven rather than a hand-maintained list.

- Every scalar inventory entry with a `toml_path` receives a generated TOML fixture and is asserted through its typed property and provenance. Adding an inventory entry without loader/property plumbing makes the test fail automatically.
- Both nested health tables are tested with site siblings and multiple families plus partial user overrides. Replacing either table or shallow-merging `families` fails.
- `subscription_plans` proves array-of-table replacement remains intact.
- Five-layer combinations prove site/user/env/CLI order and equal-valued provenance.
- The cloud-shell archive environment override is explicit.
- Runtime mutation tests change environment, cwd, home, and runtime directory after construction and assert all compatibility/path projections remain stable and frozen.
- A real `execute_embed_stage()` route captures the key passed to `create_vector_provider`; restoring a raw environment lookup makes it fail.
- Real daemon and maintenance Click invocations prove user-TOML archive selection.
- API/CLI/MCP/service objects prove one archive and index tier projection.
- Effective payload assertions cover tier paths, active identity, provenance, and secret redaction.
- An AST audit detects direct reads of every inventoried environment variable outside `config.py`.

Each test documents the representative production mutation that should turn it red.

## Changed files

The patch changes 52 production modules and 21 test modules. The complete authoritative list is encoded in the 73 `diff --git` headers in `PATCH.diff`.

Production groups:

- core authority: `polylogue/config.py`, `polylogue/paths/*`, `polylogue/services.py`;
- CLI: root, archive routing, formatting, shared environment, config/dashboard/demo/embed/facets/import/status/tutorial commands;
- daemon: root, HTTP/UDS, browser capture, backup, health/convergence/cursor/embedding/similarity/thread routes;
- MCP/API: server, support, call log, facade, insights;
- pipeline/sources/storage/UI: ingest/validation, Drive/hooks/live/Antigravity, embedding materialization/preflight/status/provider, logging/readiness/theme.

Tests update old ambient-loader expectations to install or patch the runtime projection and add the new closure module. No existing test or helper was deleted.

## Acceptance matrix

| Acceptance condition | Status | Evidence |
|---|---|---|
| Sole five-layer resolver | Implemented | Compatibility objects/path helpers project `ResolvedRuntimeConfig`; AST/import audits and composition tests |
| Site/user TOML archive affects CLI, daemon, MCP, API, maintenance | Implemented | Cross-root object test plus real daemon and maintenance Click invocations |
| Same resolved tier paths and active identity | Implemented | Shared frozen runtime, tier/identity payload assertions, service/root equality tests |
| TOML Voyage key reaches embedding execution | Implemented | Real embedding-stage/provider capture test with environment key absent |
| Deep merge both health tables and `families` | Implemented | Parameterized two-layer fixtures |
| Array-of-table replacement retained | Implemented | Subscription-plan replacement test |
| Env remains layer four/cloud-shell override wins | Implemented | Dedicated cloud-shell case and generated precedence tests |
| Provenance and redaction preserved | Implemented | Equal-value provenance and effective-payload tests |
| No post-construction env/cwd/home drift | Implemented | Immutable-snapshot mutation test |
| Explicit malformed selected config fails typed | Implemented | Existing/new config tests; theme no longer swallows resolver failure |
| Bypassing product settings migrated/classified | Implemented | Inventory expansion, caller table, AST guard, retained-boundary census |
| Future-setting guard | Implemented | Dynamic scalar inventory test plus direct-read AST audit |
| Full unit suite | Partially verified | All 15,456 tests collect; full execution exceeded the available practical window. See `TESTS.md` |

## Apply order

From a clean checkout at `536a53efac0cbe4a2473ad379e4db49ef3fce74d`:

```bash
git apply --check PATCH.diff
git apply PATCH.diff
```

Recommended local certification order:

```bash
ruff check <changed-python-files>
ruff format --check <changed-python-files>
git diff --check
pytest -q tests/unit/core/test_config.py \
  tests/unit/core/test_config_inventory.py \
  tests/unit/core/test_config_runtime_closure.py
pytest -q <all changed test modules>
pytest -q tests/unit
```

The patch has already passed `git apply --check`, application, `git diff --check`, and the 172-test core closure suite from an independently patched clean worktree.

## Risks and limitations

- The complete 15,456-test unit suite was collected successfully but not executed to completion. Serial and xdist whole-suite attempts exceeded the available practical execution window. All 21 changed test modules pass together, and additional focused unchanged consumer suites pass; the remaining value of a further pass is broad repository certification, not an architectural redesign.
- Two API facade tests fail on both the patched tree and the untouched snapshot with `sqlite3.OperationalError: database source_debt is locked`. One CLI JSON snapshot fails on both trees because its stored schema version is 37 while the current source produces 38. These baseline failures were not altered by this patch.
- `tests/unit/pipeline/test_validation_parallelism_contracts.py` hung before producing a result in this environment during a grouped follow-up run. It is not changed by this patch and is marked unverified rather than passed.
- No operator live daemon, browser extension, archive, secrets, NixOS deployment, systemd unit, or external Drive/Voyage service was used. Those checks remain unverified.
- A process may install only one compatibility runtime at a time. Library embedders that intentionally switch archive authorities inside one process should pass a `ResolvedRuntimeConfig` explicitly and install it at their own composition boundary rather than mutating ambient state.
- Some legacy explicit `Config`/`SimpleNamespace` test and library seams remain supported. They do not become a second five-layer resolver; where they lack the settings projection, consumers fall back to the already-installed runtime.

A further iteration would most plausibly add low-to-moderate value through complete-suite execution in the repository's intended CI/Nix environment and cleanup of the three pre-existing baseline failures. It is not expected to require a substantial second architecture pass.
