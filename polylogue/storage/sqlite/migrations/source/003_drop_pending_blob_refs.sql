-- Drop the never-reachable blob-GC lease table (polylogue-v7e0).
--
-- pending_blob_refs and its acquire_blob_leases/release_operation_leases
-- read/write helpers existed to bridge the acquire-blob -> write-DB-row
-- commit window more tightly than a timing heuristic. A race-window audit
-- (docs/audits/2026-07-09-race-window-audit.md, rows 1a/1b) found that no
-- production ingest caller ever populated the payload keys
-- (_blob_hashes/_operation_id) that would have triggered a lease acquire, so
-- the table was permanently empty in production and the mechanism never
-- engaged. GC's defense against a blob write racing a concurrent GC pass is
-- the MIN_AGE_S age floor plus the gc_generations high-water mark (see
-- docs/internals.md "GC concurrency model"), unaffected by this drop.
DROP INDEX IF EXISTS idx_pending_blob_refs_operation;

DROP TABLE IF EXISTS pending_blob_refs;
