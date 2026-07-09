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

### Schema versioning: two regimes keyed by tier durability
- **Chosen**: per-tier version constants are the authority; mismatch is rejected.
  Two evolution regimes (see `docs/internals.md` § Schema Versioning Model):
  - **Durable tiers** (`source.db`, `user.db`) use explicit *additive* numbered SQL
    migrations under `storage/sqlite/migrations/{source,user}/NNN_*.sql`, advancing
    `PRAGMA user_version` one step at a time behind a **verified backup manifest**.
    Additive = `CREATE TABLE`/`CREATE INDEX`/`ADD COLUMN`/bounded backfill;
    destructive durable changes need copy-forward + explicit operator consent.
  - **Derived tiers** (`index.db`, `embeddings.db`) have no in-place chains — a
    schema mismatch rebuilds/blue-green-replaces the tier from durable evidence.
  - **Disposable tier** (`ops.db`) may keep narrow bootstrap `ALTER TABLE` helpers.
- **Rejected**: a *single* fresh-first-only policy that forces re-ingest for any
  durable-tier change (loses irreplaceable `user.db` assertions); and full
  Alembic-style forward/reverse upgrade chains for derived tiers (unnecessary —
  they rebuild). The `devtools lab policy schema-versioning` lint enforces the
  boundary: numbered durable migrations allowed, derived-tier upgrade helpers forbidden.
- **Constraint**: Archive SQLite file set, WAL mode. Durable-tier migration
  requires a backup manifest; derived-tier rebuild is operator-triggered on reject.

### Content hash: idempotent by SHA-256 over NFC-normalized payload
- **Chosen**: Hash over title, timestamps, messages, attachments, content blocks. Excludes user metadata (tags, summaries, notes).
- **Rejected**: UUID-based identity — breaks idempotency on re-ingest.
- **Constraint**: User metadata edits must not trigger re-import.

### FTS5: unicode61 tokenizer, no porter stemmer
- **Chosen**: `unicode61` tokenizer. Triggers on messages table with suspend/resume during bulk ops.
- **Rejected**: Porter stemmer — not compiled in this SQLite build.
- **Constraint**: FTS freshness is an invariant; startup may perform bounded
  recovery, but global drift is represented as explicit convergence debt.

### Blob store: content-addressed with SHA-256 prefix sharding
- **Chosen**: 256 subdirectories (`blob/00/` through `blob/ff/`), write-once, read-many.
- **Rejected**: Path-based storage — no automatic dedup.
- **Constraint**: GC combines a DB snapshot reference check with a
  generation-age floor (`gc_generations`, `MIN_AGE_S`) as its sole defense
  against a blob write racing a concurrent GC pass; a lease-based second
  invariant was removed as unreachable dead code (polylogue-v7e0). Unreferenced
  blobs still need explicit cleanup.

### Daemon convergence: explicit stages, no implicit storage upgrade
- **Chosen**: Daemon runs named convergence stages (FTS freshness, insight refresh, embedding) on a schedule.
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
