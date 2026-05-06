# Internals Reference

Working map of the live codebase: invariants, hot files, extension points, and
debugging landmarks. For the conceptual system shape, see
[architecture.md](architecture.md).

## Key Invariants

| Invariant | Enforced in |
| --- | --- |
| Archive writes are idempotent by content hash | `pipeline/ids.py`, `pipeline/prepare_enrichment.py` |
| Content hash excludes user metadata (tags, summaries) | `pipeline/ids.py:conversation_content_hash()` |
| Content hash uses NFC normalization | `core/hashing.py:hash_text()` |
| Async SQLite is the primary runtime; sync SQLite exists for CLI, schema tooling, and batch-ingest write paths | `storage/sqlite/async_sqlite.py`, `storage/sqlite/connection.py`, `pipeline/services/ingest_batch.py` |
| SQLite read/write tuning is profile-driven, not backend-local | `storage/sqlite/connection_profile.py` |
| FTS tokenizer is `unicode61` (no porter stemmer) | `storage/sqlite/schema_ddl_archive.py` |
| Schema bootstrap branching is shared across sync and async backends | `storage/sqlite/schema_bootstrap.py:decide_schema_bootstrap()` |

## Hot Files

### Entry Points

| File | Purpose |
| --- | --- |
| `polylogue/api/__init__.py` | Async library API |
| `polylogue/config.py` | Runtime configuration and XDG resolution |
| `polylogue/cli/click_app.py` | Root query-first CLI dispatch |
| `polylogue/cli/command_inventory.py` | CLI command inventory |
| `polylogue/operations/archive.py` | High-level archive operations |

### Storage

| File | Purpose |
| --- | --- |
| `storage/sqlite/schema_ddl.py` | Schema definition and `SCHEMA_VERSION` |
| `storage/sqlite/schema.py` | Shared sync/async fresh-init, version guard, and extension application |
| `storage/sqlite/schema_bootstrap.py` | Shared schema snapshot, bootstrap branching, and extension planning |
| `storage/sqlite/connection_profile.py` | Canonical read/write SQLite timeouts, cache, mmap, and PRAGMA profiles |
| `storage/repository/__init__.py` | Repository facade (9-mixin composition: archive reads/writes, action reads, four insight readers, raw, vectors) |
| `storage/search_providers/fts5.py` | Lexical search |
| `storage/search_providers/hybrid.py` | Hybrid retrieval (RRF fusion) |

### Sources and Pipeline

| File | Purpose |
| --- | --- |
| `sources/dispatch.py` | Provider detection and parser routing |
| `sources/parsers/*.py` | Per-provider parsing |
| `pipeline/ingest_support.py` | Ingest stage definitions and source selection helpers |
| `pipeline/ids.py` | Content hashing and ID generation |
| `pipeline/services/ingest_batch.py` | Batch ingest (largest pipeline file) |

## Extension Points

**Adding a provider**: Start at `sources/dispatch.py:detect_provider()`. Add a
`looks_like()` function in a new parser under `sources/parsers/`. Add a
`Provider` enum variant in `types.py`. Add a provider schema bundle under
`schemas/providers/`. Dispatch is generally strict-before-loose for record-path
and Pydantic-validated checks, but `sources/dispatch.py` runs some structural
sequence detectors before loose code/dict probes. Insert the new check at the
tightness level it deserves or an earlier parser will claim its records first.

**Adding a filter**: Filter chain: `archive/filter/filters.py`. If the filter
needs a stats-table join, update `_needs_stats_join()` in
`storage/sqlite/connection.py`. Add the corresponding CLI flag in
`cli/query.py` and MCP parameter in `mcp/`.

**Adding a CLI command**: Register in `cli/command_inventory.py`. Implementation
goes in `cli/commands/`. The CLI is query-first — bare `polylogue` is search,
not help.

**Adding a session insight**: Define the insight model in `insights/`. Add
storage in `storage/insights/session/`. Wire rebuild logic and register in
`insights/registry.py`.

**Adding a devtools command**: Add a `CommandSpec` to
`devtools/command_catalog.py`. Implementation goes in `devtools/<name>.py`.
Run `devtools render-all` to update the generated catalog in
`docs/devtools.md`.

## Schema Versioning Model

Polylogue uses fresh-first schema versioning, not migration chains:

- `SCHEMA_VERSION` constant in `storage/sqlite/schema_ddl.py` is the authority
- On startup, the version in the database is compared against the constant
- **Version match**: normal operation
- **Version mismatch**: the database is rejected. There is no automatic
  migration. The operator must explicitly run a reviewed in-place upgrade
  script for that exact version transition.
- Schema is regenerated fresh for provider schemas via `devtools schema-generate`
  and promoted via `devtools schema-promote`

This design avoids migration-chain complexity (no Alembic, no forward/reverse
migrations, no partially-applied migration states) at the cost of requiring
explicit version-transition scripts.

## Content Hash Model

Archive writes are idempotent by content hash:

- SHA-256 over NFC-normalized (Unicode Normalization Form C) conversation
  payload
- Hashed fields: title, timestamps, messages, attachments, content blocks
- Excluded from hash: user metadata (tags, summaries, notes) — editing these
  does not trigger re-import
- Hash is computed in `pipeline/ids.py:conversation_content_hash()` and stored
  as `content_hash` on conversations
- On re-ingest, if the content hash matches, the conversation is skipped
  (idempotency). If it differs, the conversation is updated and dependent
  insights are rebuilt.

## FTS5 Model

Full-text search uses SQLite FTS5:

- **Tokenizer**: `unicode61` (no porter stemmer — this SQLite build doesn't
  include it). Case-insensitive for ASCII. Unicode-aware tokenization.
- **Content sync**: FTS5 indexes use `content='messages'` to stay in sync with
  the source table. Triggers handle INSERT/UPDATE/DELETE.
- **Trigger suspension**: During bulk operations, FTS triggers are suspended
  for performance, then re-enabled and the index rebuilt via
  `INSERT INTO messages_fts(messages_fts) VALUES('rebuild')`.
- **Risk**: SIGKILL during trigger suspension leaves FTS out of sync.
  Mitigation: daemon/CLI startup checks and restores FTS triggers.
- **Query syntax**: FTS5 boolean operators (AND/OR/NOT), phrase search
  (`"exact phrase"`), prefix search (`prefix*`). Column filters are not
  directly exposed; use CLI/MCP filters instead.

## Blob Store Model

Content-addressed blob storage for large binary data:

- **Content addressing**: SHA-256 hash over raw bytes. The hash IS the address.
  Identical content → identical hash → automatic deduplication.
- **Prefix sharding**: 256 subdirectories (`blob/00/` through `blob/ff/`),
  each containing blobs keyed by the remaining 62 hex characters of the hash.
- **Linking**: `link_group_key` groups related blobs (e.g., all blobs belonging
  to one session). `blob_links` table maps conversations to their blobs.
- **Operations**: Blobs are write-once, read-many. No in-place modification.
  GC identifies unreferenced blobs via link counting.
- **Known issues**: GC has bugs with orphan detection and integrity
  verification ([#818](https://github.com/Sinity/polylogue/issues/818)).

## Debugging Landmarks

Cross-check adjacent surfaces after changes:

- query: `cli/query*.py` ↔ `archive/filter/filters.py` ↔ `storage/search*.py`
- pipeline: `daemon/` ↔ `pipeline/` ↔ `storage/` ↔ `insights/`
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
