# Polylogue

AI chat export archiver — ingests Claude, ChatGPT, Codex, Gemini exports into a SQLite archive with full-text search, session analytics, and derived products.

## Development

```bash
direnv allow                  # One-time setup; afterward entering the repo loads the devshell
# or: nix develop

pytest -q --ignore=tests/integration
polylogue --help
POLYLOGUE_FORCE_PLAIN=1 polylogue run
```

Flake-based: `pyproject.toml` is authoritative for deps. No imperative installs.

## Code Rules

- No backwards compatibility shims — delete and replace
- `extra="ignore"` on Pydantic models reading from DB (not `extra="forbid"`)
- SQLite performance pragmas are mandatory: `cache_size=-524288`, `synchronous=NORMAL`, `mmap_size=1073741824`, `temp_store=MEMORY`, `wal_autocheckpoint=10000`
- FTS5 tokenizer is `unicode61` — porter stemmer is NOT compiled in this SQLite build
- Schema v4: fresh-only, wipe-on-mismatch, no migration chain
- Run `pytest --ignore=tests/integration` before claiming work is done

## Architecture

```
polylogue/
+-- lib/               # Core domain (models, filters, query plans, hashing, dates)
+-- storage/           # SQLite backends (async + sync), session products, search
|   +-- backends/      # Connection management, schema DDL, upgrades
|   +-- session_product_*.py  # Derived data: profiles, work events, phases, threads
+-- sources/           # Provider detection, parsers, acquisition
+-- pipeline/          # Ingestion orchestration (acquisition, parsing, validation, indexing)
+-- schemas/           # Schema inference, pinning, verification
+-- showcase/          # QA exercise catalog and runner
+-- cli/               # 11 commands: run, doctor, audit, schema, products, tags, etc.
+-- mcp/               # Model Context Protocol server
+-- operations/        # High-level archive operations
+-- rendering/         # Markdown/HTML output
+-- site/              # Static site generation
+-- archive_products.py  # Pydantic models for session-derived products
+-- facade.py          # Top-level library API
+-- config.py          # Configuration
```

### Database: 32.5 GB SQLite

- 6,650 conversations, 2.1M messages, 5 providers
- Session products: 6,650 profiles, 17,888 work events, 13,936 phases
- WAL mode with deferred auto-checkpoint

### Provider Detection

ChatGPT (`mapping` field) -> Claude web (`chat_messages`) -> Claude Code (`parentUuid`) -> Codex (session envelope) -> Gemini (Drive API)

## Testing

Large test suite. Protected files (never delete): `tests/integration/`, `tests/unit/security/`, `test_parsers_props.py`, `test_null_guard_properties.py`, `test_properties.py`, `test_crud.py`.

Mutation testing: `mutmut run` (8 modules).
QA exercises: `POLYLOGUE_FORCE_PLAIN=1 polylogue audit --only exercises --tier 0`

@CONTRIBUTING.md
@.claude/includes/architecture.md
@TESTING.md
