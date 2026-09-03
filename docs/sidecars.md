# Sidecars

Every non-primary file the system produces or consumes alongside a primary
artifact. Mishandling a sidecar has caused phantom sessions, invalid
verification graphs, and failed backups; each kind below names its contract.

| Kind | Producer | Consumer | Lifecycle | Contract | Mishandling failure |
| --- | --- | --- | --- | --- | --- |
| Provider metadata (`*.md.metadata.json`, Antigravity) | vendor app | artifact taxonomy only | durably acquired as raw evidence | `parse_policy="raw-only"` with no parser (`sources/origin_specs.py`, rule `agent_sidecar_meta`): admitted to the raw tier, never session-parsed | phantom one-message sessions |
| Claude tool-result payloads (`.json`, `.txt`, `.html`, …) | Claude Code | artifact acquisition joined to owning tool_use | lives with the transcript tree | declared opaque payloads; admission completeness owned by polylogue-omsw (only `.json` admitted today) | silent acquisition gaps; independent-session risk |
| `history_sidecars` rows (source.db) | acquisition | replay/verification | durable | `content_hash` is a 32-byte blob hash addressing the sidecar bytes, unique with `(origin, source_path)`; the row carries no raw-revision column, so `blob_refs` `ref_type='sidecar'` keyed by `sidecar_id` is its only ownership edge | a sidecar row read as evidence of a primary revision it never names |
| SQLite `-wal`/`-shm` | SQLite runtime | SQLite runtime only | transient beside any live DB | never copy, hash, or ingest independently; snapshots go through the backup API (`sources/sqlite_snapshot.py::snapshot_sqlite_database`); staging discards them (`_SQLITE_SIDECAR_SUFFIXES`) | `mode=ro` opens fail under read-only mounts — use `immutable=1` only for genuinely frozen files |
| SQLite staging provenance (`*.polylogue-import`) | `stage_sqlite_snapshot` | `original_sqlite_source_path` | with the staged snapshot | records the original source path; non-ingestible | snapshot loses acquisition identity |
| Verify receipts/graph (`.cache/verify/**`) | devtools | devtools, harvest evidence | per checkout, gitignored | a testmon datafile without its sidecars, and sidecars without the datafile, both read as an unusable graph | an unusable graph falls back to the complete corpus |

Hook spool entries (`hooks/pending/<day>/<event_id>.json`) are primary
acquisition sources, not sidecars; their envelope contract lives in
`sources/hooks.py`.
