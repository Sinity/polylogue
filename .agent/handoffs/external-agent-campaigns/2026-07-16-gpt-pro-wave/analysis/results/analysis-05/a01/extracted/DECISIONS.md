# Consolidation decisions and deletion gates

## Decision rules

A production authority is the declaration or implementation that the real consumer executes. Generated projections may remain separate files when their arrow points from that authority and a drift check proves regeneration. Compatibility copies remain when they preserve supported historical state. A validation list is not allowed to become a third product model.

The following decisions are scoped to existing owning areas and tracker work. They do not introduce a parallel architecture.

## D01 — Generate and execute query recipes from query declarations

**Decision:** Adopt the executable query parser/declaration model as canonical for grammar, terminal result semantics, fields, examples, completions, errors, prompt recipes, and docs. Preserve the six cookbook intents, but remove handwritten executable expressions from prompt implementations.

**Immediate repair:** Replace the two sessions-only recipes with valid terminal forms. The precise form should be selected from the existing supported grammar, such as a sessions scope followed by a terminal action/file source, without inventing a second parser rule merely to accept the stale text.

**Delete only when:**

1. Every shipped `query_units(expression=...)` recipe is represented as structured declaration data or generated text.
2. Registration/build validates every recipe with `parse_unit_source_expression`.
3. The generated prompt text is byte-stable or intentionally versioned.
4. The literal executable recipes in `server_prompts.py` and any available installed skill source are removed.

**Real-consumer proof:** Build the real MCP server, render every prompt, extract each `query_units` call, invoke the registered tool against a fresh bounded archive, and assert the declared result semantic class, continuation metadata, and non-error envelope. Mutate one recipe to sessions-only syntax and prove the gate fails.

**Falsification evidence:** A recipe that compiles but returns a different unit, silently truncates an exhaustive intent, or only passes a parser mock means the declaration is insufficient.

## D02 — Make maintenance target declarations own all execution paths

**Decision:** Extend the maintenance target declaration—or a single adjacent execution registry keyed by its spec—to own preview handler, execute handler, destructive policy, resumability capability, invalidation keys, and surface visibility. Both `execute_replay` and `execute_backfill` consume that same dispatch.

A target may be explicitly unsupported for resumable execution only if the declaration says so and every surface presents that capability state before invocation. The current implicit omission is not acceptable because the same target already executes through another path.

**Delete only when:**

1. Catalog names, preview handlers, execute handlers, and replay-capable handlers are equality-checked from one declaration.
2. `superseded_raw_snapshots` runs through the unified path.
3. CLI, MCP, and HTTP choose the same target set and targetless semantics.
4. Failed CLI operations exit nonzero in both plain and JSON modes while preserving the operation envelope.
5. `_REPLAY_DISPATCH`, `_PREVIEW_HANDLERS`, and `_REPAIR_HANDLERS` no longer independently enumerate names.

**Real-consumer proof:** On a freshly initialized archive, iterate every declared target through CLI, MCP, and HTTP dry-run routes, then execute safe fixtures for destructive targets. Assert identical target identity, status, failure shape, and resumability claim. Kill a run after a checkpoint and resume through the same declaration.

**Falsification evidence:** A target can be added to help text without executing, a surface still chooses a different orchestrator, or JSON failure still exits zero.

## D03 — One typed config declaration and one effective-value resolver

**Decision:** Evolve the current config inventory into a typed `ConfigSpec`-style declaration per key. It must own default, value type/coercer, TOML path, environment variable, CLI override, secrecy, reload behavior, merge strategy, and provenance behavior. `load_polylogue_config` is the effective-value authority. Legacy `Config` remains temporarily as a derived runtime adapter, not an independent resolver.

Nested objects require explicit merge semantics. Health threshold maps should deep-merge by declared key/family unless a declaration explicitly marks full replacement.

**Delete only when:**

1. The partial nested override preserves inherited sibling keys and records winning provenance at the leaf level or an equally precise explain form.
2. Every direct read of an inventoried environment variable is migrated to an injected resolved config, or is placed in a reviewed runtime-only bypass manifest with a reason.
3. The legacy `get_config` path derives from the layered resolver and no longer calls separate path/env authorities.
4. Type coercion sets and default maps are generated from the typed declaration.
5. All existing CLI, daemon, MCP, ingest, and theme consumers pass composition tests.

**Real-consumer proof:** For each key class, invoke at least one actual consumer under default, site TOML, user TOML, env, and CLI layers. For nested thresholds, run daemon alert evaluation, not only loader assertions. For secrets, prove inspection output redacts values while the consumer receives them.

**Falsification evidence:** A consumer reads `os.environ` directly for an inventoried key, a nested override deletes an unrelated value, or an explain command reports a layer different from the value actually used.

## D04 — Archive-tier embeddings DDL is the only table-creation authority

**Decision:** `polylogue.storage.sqlite.archive_tiers.embeddings` owns the current embeddings schema. If runtime-selectable dimension remains required, expose a canonical DDL builder parameterized by dimension and make both fresh initialization and provider setup consume it. Choose one public metadata name—current archive-tier `origin` is consistent with repository vocabulary—and migrate all writers/readers to it.

**Delete only when:**

1. `SqliteVecRuntimeMixin._ensure_tables` no longer contains an independent `CREATE TABLE` or `CREATE VIRTUAL TABLE` definition.
2. Legacy provider writes and archive-tier writes use one column contract and one metadata/status shape.
3. Fresh initialization in either call order produces identical `sqlite_master`, PRAGMA column metadata, indexes, and strictness.
4. Existing derived-tier data receives an explicit rebuild gate rather than an in-place hidden shape mutation.

**Real-consumer proof:** Create databases in both historical orders, run both writer APIs, run similarity lookup and materialization/status consumers, then compare schema fingerprints and row semantics. Mutate the canonical auxiliary column and prove both writers’ tests fail until regenerated/migrated together.

**Falsification evidence:** `IF NOT EXISTS` still lets a first writer win, either writer needs a compatibility-only column not represented in the declaration, or a dimension change silently drops only one table without rebuilding dependent metadata.

## D05 — Declare MCP tools once in production code

**Decision:** Pilot the existing declaration work with an `MCPToolSpec` that owns tool name, role visibility, callable, input schema, result contract, minimal smoke invocation, destructive/confirmation semantics, and docs text. Runtime registration, expected sets, envelope tests, role matrices, and generated docs derive from it.

**Delete only when:**

1. `build_server(role=...)` registers from the specs.
2. `EXPECTED_TOOL_NAMES`, `TOOL_CONTRACT`, and `_KNOWN_MINIMAL` are generated views or eliminated.
3. Devtools docs no longer import from `tests/`.
4. The manual 103-tool count in repo instructions is generated or removed.
5. Every declared tool is invoked through the real server in the roles where it is visible.

**Real-consumer proof:** Iterate specs through read/write/admin servers, invoke minimal arguments on a fresh archive, validate runtime envelopes, and test role rejection and destructive confirmation. A docs generator must read production specs, not a test mirror.

**Falsification evidence:** A decorator registers a tool outside the specs, a spec exists without a callable, or generated docs name a tool absent from runtime.

## D06 — Route contracts must drive dispatch, not only describe it

**Decision:** Use the existing `RouteContract` data as the migration seed for the planned typed `RouteSpec`; do not create another route table. Before ASGI migration, bind `_StaticPostRoute` and DELETE routes to contracts and dispatch authentication from the declared policy or a policy strategy keyed by it.

**Delete only when:**

1. GET, POST, and DELETE route entries all carry the same production spec.
2. Router construction and OpenAPI consume the spec.
3. Hard-coded auth grouping in `do_POST`/`do_DELETE` is removed or generated.
4. Pattern census remains exact without a separately authored implemented-pattern list.

**Real-consumer proof:** For every route spec, issue unauthenticated, wrong-origin, valid read, valid write/admin, and feature-disabled requests as applicable against the actual handler. Assert response schema and streaming mode from the spec.

**Falsification evidence:** Route path parity passes while auth behavior diverges, or a test derives both expected route and mock handler from the same spec without making an HTTP request.

## D07 — Status reports derive from semantic operations, with explicit exclusions

**Decision:** Key facade/CLI parity by stable semantic operation IDs, in line with `polylogue-s1kr`. Status may project method names, but it must classify intentional exclusions such as lifecycle/builders explicitly. Do not make Python reflection the product model.

**Delete only when:**

1. Every public callable maps to an operation or an explicit exclusion reason.
2. `embedding_preflight` and `embedding_status` receive correct routing classifications.
3. `_ARCHIVE_FACADE_ROUTES` is generated or removed.
4. Status readiness is based on executable route checks or representative operations, not map completeness alone.

**Real-consumer proof:** Invoke one real method per route/tier class against a temporary archive and compare the observed store/path effect with the operation declaration. Mutation of a method’s store route must fail the status proof.

**Falsification evidence:** A generated matrix is complete by name but reports a route that the method no longer executes.

## D08 — Close or declare devtools catalog bypasses

**Decision:** CI and hooks should invoke `devtools` catalog paths. A bypass is acceptable only when a machine-readable declaration records why catalog dispatch is impossible or undesirable.

**Delete only when:**

1. The three known sites route through catalog commands or a sanctioned bypass manifest.
2. A lint scans workflow YAML, shell, and Python subprocess calls for direct `devtools.*` module/script execution.
3. The lint has a mutation witness that introduces a fourth bypass and fails.

**Real-consumer proof:** Run the actual workflow command lines and pre-push branch through the catalog dispatcher; compare exit code, arguments, and artifacts with the former direct call.

## D09 — Retain migrations; derive exact schema parity

**Decision:** Keep numbered durable migrations as compatibility copies. Build a canonical schema manifest by initializing fresh DDL and introspecting tables, columns, indexes, triggers, constraints, and `user_version`. Compare every supported migrated starting version to that manifest.

**Delete only when:**

1. The hand-written required-table subset is replaced by exact manifest comparison.
2. Fresh and upgraded schemas match at the supported contract boundary.
3. Explicit forbidden legacy overlays remain a separate semantic denylist where useful.
4. Migration tests use the production migration runner and verified backup manifest.

**Real-consumer proof:** Open each upgraded database through the real archive API and execute representative reads/writes for every durable capability, including holdout policies and receipts.

**Falsification evidence:** Schema names match but constraints/indexes differ, or direct SQL tests pass while archive API operations fail.

## D10 — Keep and wire `user_settings`; do not delete by census alone

**Decision:** Follow `polylogue-at44`: retain the separate state table, add a typed allowed-key registry, sync and async storage twins, facade/CLI access, and a first subscription-tier cost consumer. Secrets remain outside user storage.

**Delete only if the product decision is reversed:** A forward migration or copy-forward design must preserve supported data, fresh DDL and status/docs must be updated, and old archives must receive an explicit compatibility plan. Mere absence of a current consumer is not enough to erase durable user data design.

**Real-consumer proof:** Set and get a typed setting through CLI/API and both storage paths, then run cost computation and show the setting changes the result while an unknown or secret key is rejected.
