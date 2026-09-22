# Durability by tier

What each of the six storage tiers promises when the process dies, when the
host loses power, what reconstructs it afterwards, and where a caller is
allowed to say "this is retained". `polylogue/storage/sqlite/connection_profile.py`
owns the pragma half of that policy; this page states the contract those
pragmas are chosen to meet.

Two distinctions carry the whole page.

**A process crash and a power loss are different failures.** A process crash
leaves the filesystem intact: SQLite's WAL and the rollback journal recover the
last committed transaction because the bytes reached the page cache and the
kernel keeps running. A power loss discards everything the kernel had not
forced to stable storage. `synchronous=NORMAL` survives the first and is
explicitly *not* a claim about the second. Neither an application-crash test nor
a WAL-persistence test is evidence of power-loss durability.

**Durability and recoverability are different properties.** A tier that can be
rebuilt from material still present elsewhere does not need to be crash-proof;
it needs to be *detectably* stale. The tiers below are ordered by how much
their loss costs, which is the axis the six-tier split exists to express.

## The table

| Tier | Write profile | Process crash | Power loss | Reconstructed from | Acknowledgment boundary |
| --- | --- | --- | --- | --- | --- |
| `source.db` | WAL, `synchronous=NORMAL` | last committed transaction survives | the last committed transactions may be lost | retained input bytes, re-acquired or re-read from the source path | the ingest cursor, advanced only after the index and source commits both returned |
| `index.db` (live) | WAL, `synchronous=NORMAL` | last committed transaction survives | may be lost or corrupt | full replay from `source.db` | none: a read model, never an acknowledgment |
| `index.db` (owned inactive generation) | `journal_mode=MEMORY`, `synchronous=OFF`, `locking_mode=EXCLUSIVE` | the generation is discarded, never promoted | the generation is discarded, never promoted | full replay from `source.db` | promotion, which is a symlink swap after the build closes |
| `index.db` (proven-empty active cold build) | WAL, `synchronous=OFF`, raised autocheckpoint | last committed transaction survives | may be lost or corrupt | full replay from `source.db` | none |
| `embeddings.db` | WAL, `synchronous=NORMAL` | last committed transaction survives | the last committed transactions may be lost | recomputation, at the provider cost the vectors originally paid | per-vector status rows; a lost vector is recomputed, not silently absent |
| `user.db` | WAL, `synchronous=NORMAL` | last committed transaction survives | the last committed assertions may be lost | **nothing** | the write gateway's commit |
| `audit.db` | WAL, `synchronous=NORMAL` | last committed transaction survives | the last committed receipts may be lost | **nothing** | the continuity chain head |
| `ops.db` | WAL, `synchronous=NORMAL` | last committed transaction survives | may be lost or corrupt | discarded and recreated | none: no cursor advance or retention certificate may rest on it alone |

`source.db`, `user.db` and `audit.db` are the durable tiers. `index.db` and
`ops.db` are rebuildable. `embeddings.db` is neither: it is reconstructible in
principle and expensive enough in practice that existing vectors are preserved
across a replacement rather than recomputed.

## `source.db` under WAL + NORMAL: the acceptable-loss contract

`source.db` holds authoritative acquired bytes, and it is written with
`synchronous=NORMAL`, which does not fsync on every commit. That is a
deliberate choice with a stated replacement guarantee, not an oversight.

The guarantee is **replay from retained input, bounded by the cursor**:

* The acquisition cursor is not advanced until both the index commit and the
  source commit have returned. A power loss between them, or after either one
  but before the cursor advance, leaves the cursor pointing at material that is
  re-ingested on restart.
* Re-ingest is idempotent by content hash. Re-reading a file whose bytes are
  already recorded produces the same session, message and block identities and
  writes nothing new, so replay after a loss is safe to run unconditionally and
  is exactly what the daemon's ordered recovery does.
* The retained input is therefore the recovery authority, not `source.db`'s own
  last few transactions. The acceptable loss is "transactions committed after
  the last durable cursor position", and the replay that covers it is the
  ordinary convergence route.

Two consequences worth stating outright:

* **A power-loss window exists and is named.** Between a `source.db` commit and
  the kernel's writeback, a host power loss loses that commit. Nothing in the
  connection policy claims otherwise; `initialize_source_tier_database_mode`
  says so in its own docstring, and the cursor boundary above is what makes the
  window survivable rather than what makes it empty.
* **This contract depends on the input still existing.** A source whose bytes
  were deleted after ingest has no replay authority, and for that material the
  power-loss window is a real loss window. Retention policy, not connection
  policy, is what closes it.

`user.db` and `audit.db` have no replay authority at all. Their acceptable-loss
statement is simply the `synchronous=NORMAL` window, and the mitigation is
backup (see [Archive backup and rollback](archive-backup.md)), not replay.

## Blob publication

Blob bytes are not in any SQLite tier, so they carry their own boundary.

1. The staged file is written into the store's private staging directory and
   `fsync`ed before its descriptor is closed. The bytes are durable before any
   name points at them.
2. `os.replace` moves the staged file to its content-addressed path. The rename
   is atomic; a crash here leaves either the staging entry or the final entry,
   never a partial file under the final name.
3. The containing shard directory is `fsync`ed so the new name itself is
   durable, and when the batch created the shard, the blob root is `fsync`ed so
   the shard's own entry in the root is durable too. Persisting the shard
   without persisting the root would let a power loss take a whole new shard
   while publication had already reported success.

`publish_prepared` does steps 1-3 per blob. `publish_many` does step 1-2 for
every member and then step 3 once per distinct shard, so the durability
boundary is the **batch**: it returns only after every directory it touched is
persisted. That is the boundary its caller already draws --
`ArchiveBlobPublisher.flush` publishes the whole pending page under one
source-db publisher slot -- and no caller may advance a cursor or certify a
source as retained on a partial return.

A batch that raises leaves exactly the state the per-blob loop left: bytes in
place, the directory entry not yet persisted, and the retained source still the
recovery authority.

## The no-leak rule

**An unsafe profile never reaches an active durable tier.** The two profiles
that relax durability -- `BULK_BUILD_WRITE_CONNECTION_PROFILE` and
`COLD_BUILD_ACTIVE_WRITE_CONNECTION_PROFILE` -- are selected in exactly one
place, `ArchiveStore`'s index-connection setup, and are applied only to
`index.db`. The source-tier writable open is a different factory,
`open_source_tier_write_connection`, which applies the normal
`WRITE_CONNECTION_PROFILE` connection-local pragmas and deliberately omits
`journal_mode`, leaving the one-time database-mode transition to bootstrap.
`user.db`, `audit.db` and `ops.db` open through `open_connection`, whose default
is also `WRITE_CONNECTION_PROFILE`.

Both relaxed profiles keep `foreign_keys=ON`. A cold shape that could change
what a pass writes, defers or refuses would not be a durability choice; keeping
foreign keys on is what makes the relaxation safe to select automatically.

## What this page does not establish

* No power-loss test exists. The claims above about power loss are derived from
  the declared pragmas and the documented cursor boundary, not from an observed
  host-level power cut. Treat them as the contract the code is written to meet,
  and any test that claims to verify them as suspect until it names how it
  removed the page cache from the path.
* Per-tier fsync/commit counters do not exist yet, so "how often does
  `source.db` actually reach stable storage" is currently answerable only by
  `strace` against a specific driver, not by the product's own telemetry.
