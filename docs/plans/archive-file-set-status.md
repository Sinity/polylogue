# Split-Tier Archive Status

Issue: #1743

This matrix records the current claim boundary for the archive branch.
It is intentionally stricter than the implementation history: a row is
`done` only when the active code path, operator flow, and verification
evidence all support the claim.

## Current Claim Boundary

The branch is a archive replacement phase. It is not yet a
literal replacement of every storage, repository, daemon, embedding, and
blob-GC path.

## Matrix

| Area | Status | Evidence |
| --- | --- | --- |
| Durability split | done | `source.db`, `index.db`, `embeddings.db`, `user.db`, and `ops.db` DDL/bootstrap exist under `polylogue/storage/sqlite/archive_tiers/`; `archive-init` creates the file set. |
| Closed vocabularies | done | `polylogue/core/enums.py` provides archive `StrEnum` values and SQL `CHECK` generation. |
| Deterministic identities | done | `polylogue/core/identity_law.py` and archive DDL use deterministic session/message/block ids. |
| Archive file-set ingest/rederive | done for public primary paths | Live ingest and public facade ingest bootstraps and writes tiers directly. The static route catalog reports zero unsupported facade or CLI primary routes; remaining cleanup is hidden compatibility helpers and duplicate read-model internals. |
| User-owned overlays | done | Runtime user state writes and reads for tags, metadata, marks, annotations, saved views, recall packs, workspaces, blackboard notes, and corrections use `user.db` with `index.db` existence checks rather than copying from `polylogue.db`. |
| Query read path | done for root CLI/API/MCP reads | Root queries read the active `index.db`; public query, facets, context-pack, neighbors, maintenance, and MCP filters expose `origin` rather than `provider`. |
| Runtime selection switches | removed | Archive initialization is file-presence based. The `archive-activate` command and `[archive] enabled` switch are no longer runtime boundaries. |
| Upgrade-state evidence | removed | The archive runtime no longer records activation or upgrade steps from the maintenance CLI; route/status/metrics evidence comes from archive state and daemon stage events. |
| Message prose contract | done with revised target | The current archive stores prose in `blocks`; older `messages.text` denormalization is not part of the realized target. |
| Rebuild/rederive framework | partial | Archive rederive comes from source artifacts and archive materializers on public paths, but hidden compatibility helpers still need retirement. |
| Insight slimming | partial | Archive insight tables and copied read models exist, but not every duplicate insight column or view/materialization boundary has been retired from active code. |
| Action-events retirement | partial | The archive exposes action-like data from blocks, while `action_events` machinery still exists in active paths. |
| Daemon split adoption | partial | `ops.db` tables and some status projections exist; the daemon still has monolithic-runtime table assumptions to remove. |
| Embeddings/blob/GC rewrite | partial | Embedding/blob-reference DDL exists; full production embedding and blob-GC behavior has not been rewritten around the split. |
| Provider vocabulary purge | partial | Public query/read surfaces now expose `origin` tokens and omit `provider` from archive payloads; internal provider-wire/schema metadata and compatibility bridges remain until their owning boundaries are removed. |
| Broad diagnostic gate | blocked by separate issue | `devtools verify` is green; `devtools verify --all` has xdist/load instability tracked by #1775. |

## PR Shape

A PR from this branch should claim the archive read/write surface,
origin-facing public vocabulary, and user-overlay audit.
It should reference #1743 but not close it unless the remaining partial rows
above are completed or split into accepted follow-up issues.
