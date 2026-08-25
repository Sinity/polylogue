# Storage

## Area boundary

Six SQLite tiers plus a content-addressed filesystem blob store. Durability, not subject matter, determines tier placement (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:24-34`).

## Tier map

| Tier | Runtime durability | Backup | Primary contents |
| --- | --- | --- | --- |
| `source.db` | `irreplaceable` | required | Raw acquisition records, blob references and publication reservations, GC generations, hook events, sidecars (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:45-51`; `polylogue/storage/sqlite/archive_tiers/source.py:28-62`; `polylogue/storage/sqlite/archive_tiers/source.py:535-587`) |
| `index.db` | `rebuildable` | no | Parsed sessions, messages, blocks, action pairs/views, lineage links, FTS state, materialized insights (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:52-57`; `polylogue/storage/sqlite/archive_tiers/index.py:493-590`; `polylogue/storage/sqlite/archive_tiers/index.py:766-857`; `polylogue/storage/sqlite/archive_tiers/index.py:1085-1099`) |
| `embeddings.db` | `expensive_rebuild` | required | Vector table, metadata, references, status, derivation state, failures (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:58-63`; `polylogue/storage/sqlite/archive_tiers/embeddings.py:24-87`) |
| `user.db` | `human` | required | Assertions, saved queries/results, annotation schemas and batches, settings, context-delivery provenance (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:64-69`; `polylogue/storage/sqlite/archive_tiers/user.py:19-40`; `polylogue/storage/sqlite/archive_tiers/user.py:236-318`) |
| `audit.db` | `irreplaceable` | required | Operation previews, authorization, runs, attempts, events, continuity head (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:76-81`; `polylogue/storage/sqlite/archive_tiers/audit.py:20-35`; `polylogue/storage/sqlite/archive_tiers/audit.py:80-115`; `polylogue/storage/sqlite/archive_tiers/audit.py:181-229`) |
| `ops.db` | `disposable` | no | Ingest cursors, convergence debt, daemon events/lifecycle, MCP telemetry (`polylogue/storage/sqlite/archive_tiers/bootstrap.py:70-75`; `polylogue/storage/sqlite/archive_tiers/ops.py:101-110`; `polylogue/storage/sqlite/archive_tiers/ops.py:153-170`; `polylogue/storage/sqlite/archive_tiers/ops.py:193-213`; `polylogue/storage/sqlite/archive_tiers/ops.py:242-250`) |

## Identity and generated columns

- `sessions.session_id` is stored-generated as `origin || ':' || native_id` (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:520-530`).
- `messages.message_id` is stored-generated with explicit namespace tags: native identity becomes `session_id || ':n:' || native_id`; positional identity becomes `session_id || ':p:' || position || '.' || variant_index` (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:101-105`).
- `blocks.block_id` is stored-generated as `message_id || ':' || position`; tool command/path/search projections are virtual generated columns (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:266-313`).
- Sessions, messages, and blocks are `STRICT`; message and block ownership is enforced by cascading FKs (`polylogue/storage/sqlite/archive_tiers/index.py:493-495`; `polylogue/storage/sqlite/archive_tiers/index.py:512-514`; `polylogue/storage/sqlite/archive_tiers/index.py:570-572`; `polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:106-113`; `polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:279-283`).
- `material_origin` is independently constrained from role, preserving authoredness as a separate axis (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:115-125`).
- Tool outcomes come from structured result columns; the `actions` view joins paired blocks and derives `result_state` without prose matching (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:293-299`; `polylogue/storage/sqlite/archive_tiers/index.py:801-839`).

## Parsed-session write choke point

- `write_parsed_session_to_archive` computes public origin, stored native identity, session identity, parser fingerprint, and lowering fingerprint before lowering one parsed session (`polylogue/storage/sqlite/archive_tiers/write.py:363-382`; `polylogue/storage/sqlite/archive_tiers/write.py:446-451`).
- It owns its transaction by default; bulk callers may explicitly own a surrounding transaction to amortize commits (`polylogue/storage/sqlite/archive_tiers/write.py:404-408`).
- This is the parsed-session lowering choke point for normal API writes, batch ingest, revision replay, and reindex paths. It is not the only mutation function in the six-tier substrate (`polylogue/storage/sqlite/archive_tiers/archive.py:1034-1083`; `polylogue/pipeline/services/ingest_batch/_core.py:1077-1131`; `polylogue/storage/revision_governance.py:381-414`).

## Blob publication, liveness, and GC

- Blob paths are SHA-256-addressed as `<root>/<first-two-hex>/<remaining-hex>` (`polylogue/storage/blob_store.py:196-200`).
- Preparation hashes while writing a private staging file, then fsyncs its bytes before publication (`polylogue/storage/blob_store.py:210-240`; `polylogue/storage/blob_store.py:248-277`; `polylogue/storage/blob_store.py:285-298`).
- PR #4152, commit `a15421981`, landed the current durability fix: all preparation routes fsync bytes, and publication fsyncs the shard directory after atomic `os.replace` (`polylogue/storage/blob_store.py:236-240`; `polylogue/storage/blob_store.py:273-277`; `polylogue/storage/blob_store.py:294-298`; `polylogue/storage/blob_store.py:306-319`).
- Archive publication commits durable reservation receipts before exposing final paths; the exact receipt is consumed in the durable-reference transaction (`polylogue/storage/blob_publication.py:146-170`; `polylogue/storage/blob_publication.py:196-207`; `polylogue/storage/blob_publication.py:254-265`).
- Liveness is descriptor-owned. Ordinary `blob_refs.ref_type` values must map unambiguously to one referent relation (`polylogue/storage/blob_liveness.py:65-89`).
- A destructive liveness check returns `LIVE`, `UNREFERENCED`, or typed `BLOCKED`; unavailable or unreadable required tiers block deletion (`polylogue/storage/blob_liveness.py:294-329`).
- GC safety requires no live DB reference, no publication reservation, a final locked recheck across source/index, an age floor, and bounded deletion batches (`polylogue/storage/blob_gc.py:7-25`).

### Two-phase `gc_generations`

1. Commit one generation and every exact member intent as `pending` before any unlink (`polylogue/storage/sqlite/archive_tiers/source.py:559-587`; `polylogue/storage/blob_gc.py:489-532`).
2. Under `BEGIN IMMEDIATE` on source and index, recheck liveness/reservations, unlink or reconcile each member, commit outcomes, then finalize only when no pending members remain (`polylogue/storage/blob_gc.py:535-589`; `polylogue/storage/blob_gc.py:705-839`).

Pending generations are restartable; a restart resumes their exact member set instead of rediscovering intent from the filesystem (`polylogue/storage/blob_gc.py:592-600`; `polylogue/storage/blob_gc.py:856-890`).

## Lineage storage model

- A prefix-sharing child stores only its divergent tail. The writer resolves the parent, compares composed signatures, records the last inherited message as the branch point, and lowers only the remaining messages (`polylogue/storage/sqlite/archive_tiers/write.py:505-563`; `polylogue/storage/sqlite/archive_tiers/write.py:6854-6886`).
- `session_links` stores destination identity, resolved parent, branch point, inheritance mode, status, method, confidence, and evidence (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:851-885`).
- Reads recursively compose the parent through the branch point, with explicit depth-limit and dangling-branch-point status instead of silently claiming completeness (`polylogue/storage/sqlite/archive_tiers/write.py:1488-1555`).
- Link writes refuse to let parser inference overwrite an existing hook-authoritative edge (`polylogue/storage/sqlite/archive_tiers/write.py:3890-3949`).

## Invariants and gotchas

- `branch_point_message_id` is deliberately not an FK. Parent full replacement deletes before reinserting deterministic message IDs; `ON DELETE SET NULL` would permanently sever the child (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:866-877`).
- A failed or unavailable liveness surface is not equivalent to zero references (`polylogue/storage/blob_liveness.py:294-310`; `polylogue/storage/blob_gc.py:9-20`).
- A published blob may legitimately have no durable ref yet; its reservation protects that publication window (`polylogue/storage/blob_publication.py:196-207`; `polylogue/storage/sqlite/archive_tiers/source.py:548-557`).
- GC history counters are summaries derived only after all member outcomes close; member rows are the crash-recovery authority (`polylogue/storage/sqlite/archive_tiers/source.py:568-583`; `polylogue/storage/blob_gc.py:559-589`).
- Rebuildable `index.db` must not become authority for an irreversible durable mutation; blob GC therefore requires source-ledger and active-index checks to agree (`polylogue/storage/blob_gc.py:7-20`; `polylogue/storage/blob_liveness.py:294-329`).

## DISCREPANCIES

- `docs/architecture.md` draws only source, index, embeddings, user, and ops; code has six tiers and includes `audit.db` (`docs/architecture.md:24-28`; `polylogue/storage/sqlite/archive_tiers/bootstrap.py:45-82`).
- `CLAUDE.md` omits the `n:` and `p:` namespaces from `messages.message_id`; the generated-column expression includes them (`CLAUDE.md:36-40`; `polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:101-105`).
- `CLAUDE.md` and `docs/architecture.md` call embeddings simply rebuildable; runtime metadata classifies them as `expensive_rebuild` with backup required (`CLAUDE.md:63-72`; `docs/architecture.md:54-56`; `polylogue/storage/sqlite/archive_tiers/bootstrap.py:58-63`).

verified: 4abb7a80bca2160d27fdc799891305cf02b680ff 2026-08-25
