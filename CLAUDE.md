# Polylogue

AI chat export archiver — ingests Claude, ChatGPT, Codex, Gemini exports into a SQLite archive with full-text search, session analytics, and derived products.

## Development

```bash
nix develop                    # Enter devshell (or direnv allow)
nix develop -c pytest -q --ignore=tests/integration   # Run tests (~4300 tests, ~3 min)
nix develop -c polylogue --help                        # CLI
POLYLOGUE_FORCE_PLAIN=1 nix develop -c polylogue run   # Full pipeline run
```

Flake-based: `pyproject.toml` is authoritative for deps. No imperative installs.

## Git Workflow

### Branch Model

All work goes through **feature branches** that are **squash-merged** onto `master` via GitHub PRs.

```
master                   <- linear history of squash commits
  |-- feature/X/Y       <- working branch (individual commits preserved)
        | squash-merge via PR
master                   <- gains one clean narrative commit
```

### Rules

- **Always work on a feature branch** — `feature/<category>/<description>`
- **Branch from `origin/master`**
- **Never commit directly to `master`** — enforced by pre-commit hook
- **Squash-merge via PR** targeting `master` — one narrative commit per branch
- **Link PRs to era issues** — `Ref #NNN` in PR body
- **No force pushes** — enforced by pre-push hook
- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, `perf:`

### Branch Naming

`feature/<category>/<description>` where category is one of:
`feat`, `refactor`, `perf`, `bugfix`, `testing`, `schema`, `docs`

### Era Issues

Each major body of work gets a GitHub issue: `Era NN: Description`.
Chain: Era Issue -> PR -> Feature Branch -> Squash commit on `master`.

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

4,300+ tests. Protected files (never delete): `tests/integration/`, `tests/unit/security/`, `test_parsers_props.py`, `test_null_guard_properties.py`, `test_properties.py`, `test_crud.py`.

Mutation testing: `nix develop -c mutmut run` (8 modules).
QA exercises: `POLYLOGUE_FORCE_PLAIN=1 nix develop -c polylogue audit --only exercises --tier 0`

@.claude/includes/git-workflow.md
@.claude/includes/architecture.md
@.claude/includes/testing.md
