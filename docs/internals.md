# Internals Reference

Working map of the live codebase: invariants, hot files, extension points, and
debugging landmarks. For the conceptual system shape, see
[architecture.md](architecture.md).

## Key Invariants

| Invariant | Enforced in |
| --- | --- |
| Archive writes are idempotent by content hash | `pipeline/ids.py`, `pipeline/prepare_enrichment.py` |
| Content hash excludes user metadata (tags, summaries) | `pipeline/ids.py:conversation_content_hash()` |
| Content hash uses NFC normalization | `lib/hashing.py:hash_text()` |
| Async SQLite is the primary runtime; sync SQLite exists for CLI, schema tooling, and batch-ingest write paths | `storage/backends/async_sqlite.py`, `storage/backends/connection.py`, `pipeline/services/ingest_batch.py` |
| SQLite read/write tuning is profile-driven, not backend-local | `storage/backends/connection_profile.py` |
| FTS tokenizer is `unicode61` (no porter stemmer) | `storage/backends/schema_ddl_archive.py` |
| Schema bootstrap branching is shared across sync and async backends | `storage/backends/schema_bootstrap.py:decide_schema_bootstrap()` |

## Hot Files

### Entry Points

| File | Purpose |
| --- | --- |
| `polylogue/facade.py` | Async library API |
| `polylogue/config.py` | Runtime configuration and XDG resolution |
| `polylogue/cli/click_app.py` | Root query-first CLI dispatch |
| `polylogue/cli/command_inventory.py` | CLI command inventory |
| `polylogue/operations/archive.py` | High-level archive operations |

### Storage

| File | Purpose |
| --- | --- |
| `storage/backends/schema_ddl.py` | Schema definition and `SCHEMA_VERSION` |
| `storage/backends/schema.py` | Shared sync/async fresh-init, version guard, and extension application |
| `storage/backends/schema_bootstrap.py` | Shared schema snapshot, bootstrap branching, and extension planning |
| `storage/backends/connection_profile.py` | Canonical read/write SQLite timeouts, cache, mmap, and PRAGMA profiles |
| `storage/repository.py` | Repository facade (10 mixin composition) |
| `storage/search_providers/fts5.py` | Lexical search |
| `storage/search_providers/hybrid.py` | Hybrid retrieval (RRF fusion) |

### Sources and Pipeline

| File | Purpose |
| --- | --- |
| `sources/dispatch.py` | Provider detection and parser routing |
| `sources/parsers/*.py` | Per-provider parsing |
| `pipeline/run_support.py` | Stage definitions and sequences |
| `pipeline/ids.py` | Content hashing and ID generation |
| `pipeline/services/ingest_batch.py` | Batch ingest (largest pipeline file) |

## Source Layer Map

The source layer has three distinct responsibilities:

- acquisition and walking: `source_acquisition*.py`, `source_walk.py`,
  decoders, token storage, and Drive source clients;
- provider detection and parsing: `dispatch.py`, `parsers/`, and assembly
  helpers;
- provider payload models: `providers/`.

Keep provider-specific implementation behind one of those boundaries. New Drive
source code should not add more `drive_*` modules at the package root unless it
is genuinely shared source-layer infrastructure. New parser/provider model code
should use provider names consistently and avoid historical catch-all files.

The larger import-aware cleanup of Drive, Claude, provider models, and oversized
source tests is tracked in GitHub issue #403.

## Source Test Map

Source tests should be named by the contract they protect:

- acquisition, walking, decoding, and token-store tests stay close to the source
  acquisition boundary;
- parser tests stay provider-specific unless they assert a shared parser law;
- provider model tests should identify the provider or schema family under test;
- law/property tests should stay small enough that a failing assertion points to
  one semantic contract.

The current large files under `tests/unit/sources/` are accepted legacy shape,
not a model for new tests. Split new coverage by behavior rather than appending
to `test_source_laws.py`, `test_models.py`, or `test_parsers_base.py`.

## Extension Points

**Adding a provider**: Start at `sources/dispatch.py:detect_provider()`. Add a
`looks_like()` function in a new parser under `sources/parsers/`. Add a
`Provider` enum variant in `types.py`. Add a provider schema bundle under
`schemas/providers/`.

**Adding a filter**: Filter chain: `lib/filters.py`. If the filter
needs a stats-table join, update `_needs_stats_join()` in
`storage/backends/connection.py`. Add the corresponding CLI flag in
`cli/query.py` and MCP parameter in `mcp/`.

**Adding a CLI command**: Register in `cli/command_inventory.py`. Implementation
goes in `cli/commands/`. The CLI is query-first — bare `polylogue` is search,
not help.

**Adding a session product**: Define the product model in `products/`. Add
storage in `storage/session_product_*.py`. Wire rebuild logic and register in
`products/registry.py`.

**Adding a devtools command**: Add a `CommandSpec` to
`devtools/command_catalog.py`. Implementation goes in `devtools/<name>.py`.
Run `devtools render-all` to update the generated catalog in
`docs/devtools.md`.

## Debugging Landmarks

Cross-check adjacent surfaces after changes:

- query: `cli/query*.py` ↔ `lib/filters.py` ↔ `storage/search*.py`
- pipeline: `cli/commands/run.py` ↔ `pipeline/` ↔ `storage/` ↔ `products/`
- maintenance: `cli/commands/check.py` ↔ `storage/repair.py` ↔ `health.py`
- publication: `rendering/` ↔ `site/` ↔ `showcase/`
- schema: `schemas/` ↔ `sources/providers/` ↔ `pipeline/services/validation_*`

Drift check:

```bash
devtools render-all --check
devtools lab-scenario verify-baselines
```

## Local State

- `.cache/`: disposable caches (hypothesis, pytest, mypy, ruff)
- `.local/`: untracked outputs (campaigns, showcases, build artifacts)
- `.local/result`: out-link for `devtools build-package`
