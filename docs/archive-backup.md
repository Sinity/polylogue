# Archive Backup and Restore Boundaries

Polylogue stores one archive root as a split SQLite file set plus a
content-addressed blob store. Backups must preserve the tiers by durability
class instead of treating the archive root as one anonymous cache directory.

## Archive Root Layout

The configured archive root contains these durable paths:

| Path | Durability | Backup policy |
| --- | --- | --- |
| `source.db` | Raw acquisition evidence and source observations. | Back up. This is the rebuild root for parsed/indexed data. |
| `index.db` | Parsed sessions, messages, FTS/search indexes, graph rows, and derived read models. | Rebuildable from `source.db`; include in full evidence backups for faster restore, but cache-exclude profiles may omit it. |
| `embeddings.db` | Vector rows, embedding status, and catch-up metadata. | Back up when present. It is rebuildable, but expensive and may require provider cost. |
| `user.db` | Human/user/agent overlays stored as assertions: marks, annotations, corrections, tags, metadata, notes, saved views, recall packs, workspaces, candidates, and judgments. | Always back up. This tier is irreplaceable user state. |
| `ops.db` | Daemon cursors, attempts, convergence debt, stage events, and operational telemetry. | Disposable. Include only in diagnostics bundles or incident snapshots. |
| `blob/` | Content-addressed binary payloads keyed by SHA-256. | Back up referenced blobs with `source.db`/`user.db`; do not prune by age alone. |

`polylogue ops maintenance archive-plan --output-format json` is the machine-readable
inventory for tier filenames, expected versions, backup-required tiers, and
missing blockers. Run it before backup automation rather than hard-coding only
the files that happen to exist locally.

## Backup Profiles

Use these profiles when choosing what to copy:

| Profile | Include | Exclude | Use case |
| --- | --- | --- | --- |
| Full evidence | `source.db`, `index.db`, `embeddings.db`, `user.db`, referenced `blob/`, and optional `ops.db` snapshot. | Temporary SQLite `*-wal`/`*-shm` only after a clean checkpoint. | Fastest complete restore with raw evidence, read models, vectors, and overlays. |
| User overlays | `user.db` and any assertion/note evidence blobs referenced by user-owned rows. | `index.db`, `ops.db`, rebuildable search/derived models. | Protect irreplaceable human/agent state before resets or schema rebuilds. |
| Rebuildable-cache exclude | `source.db`, `user.db`, referenced `blob/`, optionally `embeddings.db`. | `index.db`, `ops.db`, derived/cache artifacts. | Small backup that can rebuild parsed/indexed data locally. |
| Diagnostics bundle | `ops.db`, `archive-plan` JSON, `daemon-workload-probe` JSON, logs, and readonly status outputs. | Private raw blobs unless explicitly needed for the incident. | Bug reports and incident triage without over-sharing archive contents. |

When SQLite WAL files are present, either stop the daemon or run an explicit
checkpoint before copying. Copying only `*.db` while an uncheckpointed `*-wal`
contains recent writes creates an incomplete backup.

## Restore Rules

Restore into an isolated archive root first:

```bash
export POLYLOGUE_ARCHIVE_ROOT=/tmp/polylogue-restore-check
polylogue ops maintenance archive-plan --output-format json
polylogue ops status --format json
```

Then verify the restored root before pointing the daemon at it:

```bash
polylogue ops diagnostics workload --json
polylogue ops doctor --format json
polylogue find pytest then read --view summary
```

Restore expectations:

- `user.db` must survive ordinary `polylogue ops reset --database` and
  `polylogue ops reset --all`; deleting it requires the explicit
  `--include-user-db` opt-in.
- Assertion candidates, accepted/rejected/deferred judgments, and promoted
  active assertions all live in `user.db`. Rebuilding `index.db` from
  `source.db` must not turn rejected or deferred inference candidates back into
  actionable user assertions, and editing assertion metadata is outside the raw
  session content-hash boundary.
- `index.db` may be rebuilt from `source.db` when schema versions change.
- `embeddings.db` may be rebuilt, but restore it when possible to avoid
  provider cost and delay.
- `ops.db` does not decide archive correctness; restore it only when preserving
  daemon history matters.
- A restored blob store is valid only when referenced blobs still match their
  SHA-256 paths and `source.db`/`user.db` references.

## Blob GC Safety Boundary

Blob garbage collection is dry-run-first work. A safe GC report must prove:

- the candidate blob is not referenced by `source.db.raw_sessions`;
- the candidate is older than the generation/age gate (`MIN_AGE_S`, the sole
  defense against a blob write racing a concurrent GC pass — see
  `docs/internals.md` "GC concurrency model");
- the report names exact candidate counts and references before deletion.

Do not delete blobs based only on filesystem age, directory mtime, or absence
from `index.db`. `source.db` is the authority for raw evidence reachability.
