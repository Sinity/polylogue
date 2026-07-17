# Ordered next actions

## Gate zero — Land semantic regression tests before consolidation

These tests should be small, focused, and attached to the owning production routes. They are the deletion gates for the first implementation wave.

1. **Query recipe execution law — owner: MCP/query.** Render every registered cookbook prompt, extract declared `query_units` recipes, compile them, and invoke the real tool on a fresh archive. Preserve the two exact failing source recipes as regression fixtures until fixed. Acceptance: no generated recipe returns `invalid_query`; valid result-unit and continuation semantics are asserted. Falsification mutation: replace one recipe with a sessions-only expression and observe failure.

2. **Maintenance all-target surface law — owner: maintenance/storage.** Iterate every `MAINTENANCE_TARGET_SPECS` entry through the actual CLI command, MCP tool, and HTTP handler in dry-run mode. Acceptance: every advertised target is executable or explicitly declared unsupported before invocation; statuses and target identities agree; failed CLI operations exit nonzero. Falsification mutation: remove one handler registration.

3. **Config composition law — owner: config/daemon.** Apply default, site, user, env, and CLI layers while invoking real consumers. Acceptance: a partial health-table override preserves unrelated site keys and the daemon alert decoder observes the composed value. Falsification mutation: restore wholesale nested assignment or add a direct inventoried env read.

4. **Embeddings create-order law — owner: embeddings/storage.** Initialize in tier-first and provider-first orders, run both production writer APIs, and compare schema fingerprints. Acceptance: both orders are identical and both writers work. Falsification mutation: rename one auxiliary column in only one writer.

## Wave one — Repair the active contradictions

### A1. Fix and generate query recipes (`polylogue-z9gh.3`)

Owning files: `polylogue/archive/query/expression.py`, query metadata/declaration modules, `polylogue/mcp/server_prompts.py`, `polylogue/mcp/server_tools.py`, query docs/completions.

Implementation order:

1. Express the six cookbook intents as structured recipe data tied to terminal result semantics.
2. Correct the two invalid sessions-only expressions using existing grammar.
3. Generate prompt text from the recipes and make registration validate compilation.
4. Add recovery guidance to `invalid_query` from the same declaration data.
5. Remove literal executable expressions only after prompt output and actual tool behavior pass.

Acceptance: the real MCP probe that currently produces two `invalid_query` results produces successful, correctly typed envelopes; DSL and any structured-plan form lower to the same canonical plan.

### A2. Unify maintenance dispatch (`polylogue-71ey`)

Owning files: `polylogue/maintenance/targets.py`, `maintenance/replay.py`, `maintenance/planner.py`, `storage/repair.py`, CLI maintenance command, MCP and HTTP maintenance surfaces.

Implementation order:

1. Add handlers and resumability capability to the target declaration or one adjacent registry.
2. Make replay and planner consume it.
3. Normalize targetless semantics across CLI, MCP, and HTTP.
4. Route `superseded_raw_snapshots` through the unified executor.
5. Correct CLI nonzero exit behavior for failed operation envelopes.
6. Delete independent handler-name maps after the all-target surface law passes.

Acceptance: fresh-archive dry run for all seven targets succeeds through all supported surfaces; checkpoint/resume behavior is proved for resumable targets; explicitly non-resumable targets report that capability consistently.

### A3. Repair config composition and authority (`polylogue-9gh1`)

Owning files: `polylogue/config.py`, `polylogue/services.py`, direct consumer files from the AST census, daemon alert decoders, CLI config/status surfaces.

Implementation order:

1. Add type and merge strategy to each config declaration.
2. Deep-merge nested health maps with precise provenance.
3. Make legacy `Config` derive from the resolved layered config.
4. Migrate the eight production-package direct reads of inventoried vars first, then the eight devtools reads.
5. Classify the 23 non-inventory variables as typed config, sanctioned runtime control, test-only control, or obsolete.
6. Generate defaults and coercion from the declarations and remove independent sets.

Acceptance: the recorded site/user sibling-loss probe preserves `critical_total` and `families.index`; actual daemon, CLI, MCP, ingest, and theme consumers agree with `polylogue config` explain output.

### A4. Canonicalize embeddings schema (`polylogue-mhx.7`)

Owning files: `archive_tiers/embeddings.py`, `archive_tiers/embedding_write.py`, `search_providers/sqlite_vec_runtime.py`, `search_providers/sqlite_vec_queries.py`, materialization and rebuild commands.

Implementation order:

1. Ratify `origin` versus `source_name` and the canonical metadata/status shape.
2. Add a dimension-parameterized canonical DDL builder if required.
3. Move provider setup to the canonical initializer.
4. Migrate the legacy provider writer to the archive-tier writer contract.
5. Add an explicit derived-tier rebuild/version gate.
6. Delete runtime inline DDL after both-order proof passes.

Acceptance: both historical initialization orders produce one schema and both writer APIs, similarity reads, status, and materialization work.

## Wave two — Reduce broad edit cost without creating shadow models

### B1. MCP declaration pilot (`polylogue-o21`, `polylogue-t46.8.1`)

Start with a small but representative family containing one read tool, one write tool, one admin maintenance tool, one resource, and one prompt. Generate runtime registration, role visibility, input schema, envelope contract, minimal invocation, and docs. Then migrate all 104 tools. Remove the stale manual count from `CLAUDE.md`.

Acceptance: no generator imports `tests.infra.mcp`; adding one tool changes one production declaration and all projections update or fail with one actionable error.

### B2. Bind daemon POST/DELETE routes to contracts (`polylogue-3utv`)

Before the ASGI rewrite, add the contract reference to POST/DELETE route entries and dispatch policy through it. During the rewrite, make the registry construct the router and OpenAPI directly.

Acceptance: 77-route behavior remains byte/semantics compatible; a policy mutation fails an actual HTTP auth test; there is no separate implemented-pattern enumeration.

### B3. Replace static facade status with semantic operations (`polylogue-s1kr`, `polylogue-9e5.31`)

Classify all five currently missing callables. Add explicit operation IDs or exclusion reasons, especially for `embedding_preflight`, `embedding_status`, `open`, `filter`, and `iter_messages`. Generate status projections and preserve representative real-route proof.

Acceptance: every public callable is classified, but readiness does not become a name-count claim; changing an operation’s actual store route fails a consumer test.

### B4. Close devtools bypasses (`polylogue-l8ee`)

Route the mutation-freshness workflow, nightly benchmark comparison, and pre-push backlog hygiene through catalog commands or declare narrow sanctioned bypasses. Add a workflow/subprocess scanner.

Acceptance: the catalog inventory is complete modulo an explicit machine-readable allowlist; a fourth direct call fails the lint.

## Wave three — Compatibility and dead-capability cleanup

### C1. Derive durable schema parity (`polylogue-ihp0`)

Generate an exact current user-schema manifest from fresh initialization, then run the production migrator from every supported version with a backup manifest and compare. Include the two currently unasserted holdout tables.

Acceptance: fresh and upgraded schemas and real API behavior match; numbered migrations remain intact as compatibility evidence.

### C2. Wire typed user settings (`polylogue-at44`)

Implement typed allowed keys, sync/async helpers, facade/CLI access, and subscription-tier cost usage. Keep deployment secrets excluded.

Acceptance: a setting changes the real cost result through both storage paths; unknown/secret keys are rejected; the table is no longer only DDL/status/docs.

## Verification package for each pull request

Each change should attach:

- the focused real-consumer test command and result;
- the declaration/projection arrow changed;
- the exact old duplicate deleted or retained as compatibility data;
- a sensitivity mutation that proves the gate notices a reintroduced split;
- remaining tracker scope, with existing Bead IDs rather than a new parallel epic.

## Expected value of a follow-up census

A follow-up pass should concentrate on lower-ranked breadth: all table-creating helpers, generated docs sourcing, CLI operation maps, route/resource/prompt declarations, and the 23 non-inventory environment variables. Its expected value is medium: it can discover more consumers and sharpen ranks five through ten, but the executable evidence already makes the top four implementation decisions stable.
