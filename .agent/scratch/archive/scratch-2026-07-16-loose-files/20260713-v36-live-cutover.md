---
created: 2026-07-13
purpose: v35-to-v36 live archive cutover recovery
status: blocked-on-two-P0-code-fixes
project: polylogue
---

## Current state

- Service is intentionally stopped. Active archive versions remain source=7,
  user=6, index=35, embeddings=1, ops=1.
- Verified backup manifest:
  `/realm/staging/polylogue-sqlite/recovery/master-v36-20260713/polylogue-archive-20260713T164600Z/manifest.json`.
- Prepared receipt:
  `.../schema-forward/v36-live-reuse-20260713T180500Z.json`.
  It proves index=36, embeddings=2, fresh ops=1, with no raw reparse, FTS
  rebuild, or vector re-embed.
- Durable source/user bytes remain equal to the receipt's pre-activation
  fingerprints after both failed attempts.

## Defects found by real activation

1. `source/009_expand_origin_vocabulary.sql` copies `raw_sessions_v7` with
   `INSERT INTO raw_sessions SELECT *`. v7 has
   `predecessor_source_revision` after `revision_authority`; v9 inserts it
   before `predecessor_raw_id`. Positional copy maps NULL into the new non-null
   `revision_authority`, causing `NOT NULL constraint failed`. P0
   `polylogue-25vy`; worker `/root/migration_repair` owns fix.
2. `activate_prepared_forward` records pre-migration rollback snapshots in
   `promoted`, then tries to restore them after a migration failure. Its
   `os.replace(active, rollback/failed-*)` fails `EXDEV` across Btrfs
   subvolumes and masks the root migration error. P0 `polylogue-b08j`; worker
   `/root/rollback_repair` owns fix.
3. After the first repair, live activation correctly rolled back but
   `_promote_index_generation` still used a cross-subvolume `os.replace` from
   staging into the active index generation directory. The same P0 now owns
   local-copy + local-rename generation publication and an EXDEV test.

## Performance follow-ups

- `polylogue-qg6x`: persist resumable clone proofs; existing reuse performs
  repeated index hash/count/quick-check scans.
- `polylogue-pf8s`: avoid rehashing the full verified backup blob inventory
  twice per durable tier in one activation.

## Retry requirements

Merge both P0 fixes, build a new package, checkpoint/truncate zero durable
SQLite sidecars while daemon is stopped, then run exactly one managed
`sinnix-scope background` activation against the existing prepared receipt.
Use a durable scope log; do not run foreground activation through the exec
harness. Confirm receipt becomes `activated`, all versions are 9/8/36/2/1,
then point the systemd drop-in to the new package and start the daemon.
