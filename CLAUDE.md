# Polylogue

AI chat export archiver — ingests Claude, ChatGPT, Codex, Gemini exports into a SQLite archive with full-text search, session analytics, and derived products.

## Working Rules

- New semantics go into the substrate or product layer first, then surfaces
  adapt.
- Archive writes are idempotent by content hash. User metadata (tags, summaries)
  is excluded from the hash — changing it does not trigger re-import.

@CONTRIBUTING.md
@TESTING.md
@docs/architecture.md
@docs/internals.md
@docs/devtools.md
