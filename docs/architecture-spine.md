# Architecture Spine

Target shape, guardrails, and major decisions for the Polylogue codebase.
This is the canonical architectural anchor — `docs/architecture.md` evolves
with the implementation; this document records the *why* and the *rules*.

## Four Rings

| Ring | Role | Primary modules |
|------|------|-----------------|
| **Archive Substrate** | Owns stored meaning: acquisition, parsing, persistence, query | `sources/`, `pipeline/`, `storage/`, `archive/`, `operations/` |
| **Derived Read Models** | Stored insights computed over the archive | `insights/`, `storage/insights/session/` |
| **Surfaces** | Expose the archive to users and machines | `cli/`, `mcp/`, `api/`, `rendering/`, `ui/`, `daemon/` |
| **Verification** | Schema, demos, devtools, tests | `schemas/`, `demo/`, `devtools/`, `tests/` |

**Rules:**
- New semantics go into substrate or insights first, then surfaces adapt.
- Surfaces may not import substrate internals directly (enforced by `devtools verify layering`).
- Archive writes are idempotent by content hash.

## Guardrails

Every `docs/plans/*.yaml` manifest is enforced by a lint in `devtools verify`.

| Manifest | Lint | What it prevents |
|----------|------|-----------------|
| `layering.yaml` | `verify layering` | Surface-to-substrate coupling |
| `topology-target.yaml` | `verify topology` | Orphaned/unclassified modules |
| `campaign-coverage.yaml` | `verify manifests` | Missing campaign declarations |
| `coverage-manifest.yaml` | `verify manifests` | Stale gap/coverage declarations |

## Major Decisions

### Schema versioning: fresh-first, no in-place upgrade chains
- **Chosen**: `SCHEMA_VERSION` constant is the authority. Mismatch = rejected. Schema bumps define the new canonical DDL and document the re-ingest/rebuild expectation.
- **Rejected**: Alembic/in-place upgrade chains — complexity, partial state risk, forward/reverse upgrade burden.
- **Constraint**: Archive SQLite file set, WAL mode, operator owns the rebuild trigger when a tier is rejected.

### Content hash: idempotent by SHA-256 over NFC-normalized payload
- **Chosen**: Hash over title, timestamps, messages, attachments, content blocks. Excludes user metadata (tags, summaries, notes).
- **Rejected**: UUID-based identity — breaks idempotency on re-ingest.
- **Constraint**: User metadata edits must not trigger re-import.

### FTS5: unicode61 tokenizer, no porter stemmer
- **Chosen**: `unicode61` tokenizer. Triggers on messages table with suspend/resume during bulk ops.
- **Rejected**: Porter stemmer — not compiled in this SQLite build.
- **Constraint**: FTS triggers must survive SIGKILL (startup repair check).

### Blob store: content-addressed with SHA-256 prefix sharding
- **Chosen**: 256 subdirectories (`blob/00/` through `blob/ff/`), write-once, read-many.
- **Rejected**: Path-based storage — no automatic dedup.
- **Constraint**: Link counting for GC, unreferenced blobs need explicit cleanup.

### Daemon convergence: explicit stages, no implicit storage upgrade
- **Chosen**: Daemon runs named convergence stages (FTS repair, insight refresh, embedding) on a schedule.
- **Rejected**: Event-driven cascade — harder to reason about, harder to test.
- **Constraint**: Local HTTP API is read-only by default; write/mutation requires explicit role.

### MCP server: read/write/admin role split
- **Chosen**: Three MCP roles. Read tools always available. Write tools require `--role write`. Admin tools require `--role admin`.
- **Rejected**: Single unrestricted MCP server — unsafe for automated agent use.
- **Constraint**: `polylogue-mcp` CLI entry point with `--role` flag.

### Browser capture: unpacked extension + local receiver
- **Chosen**: MV3 browser extension captures provider-native page/app evidence
  where available, falls back to DOM snapshots when no structured source is
  reachable, posts to the local Python receiver, and the receiver writes source
  artifacts for archive ingestion.
- **Rejected**: Cloud-based capture, headless automation — Cloudflare friction on Claude.ai, privacy concerns.
- **Constraint**: Extension is opt-in per-site; no background capture. DOM text
  is a compatibility fallback, not the fidelity target for providers such as
  ChatGPT that expose a full conversation payload to the authenticated page.
