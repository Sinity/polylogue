# 04. polylogue-8jg9.4 + polylogue-8jg9.2 — Make ops-doctor orphan cleanup use the same lease/generation invariants as blob GC

Priority: **P1/P2**  
Lane: **blob-integrity**  
Readiness: **ready-now / code-local**

Depends on packet(s): polylogue-s7ae.6

## Why this is urgent / critical-path

A cleanup tool that can delete an in-flight blob is a data-loss bug. It must be fixed before any backup/blob claim is trustworthy.

## Static diagnosis / likely mechanism

Root cause: there are two deletion paths. `run_blob_gc_report` checks reference surfaces, pending leases, minimum age, and generations (`polylogue/storage/blob_gc.py:163`, `:355+`, `MIN_AGE_S`). But `BlobStore.detect_orphans` and `cleanup_orphans` only compare disk hashes to a caller-supplied set and then unlink files (`polylogue/storage/blob_store.py:352-409`). `repair_orphaned_blobs_data` uses that unsafe path (`polylogue/storage/blob_repair.py:92+`). A blob written to disk with an active pending lease but not yet a durable reference can be seen as orphaned and deleted.

## Implementation plan

Implementation shape:
1. Treat `BlobStore.detect_orphans` as a preview/helper, not an apply-time safety authority.
2. Change ops-doctor / `repair_orphaned_blobs_data(... dry_run=False)` to delegate deletion to `run_blob_gc_report`, passing the archive DB path and blob root so the GC can see leases and reference surfaces.
3. For dry-run, either call `run_blob_gc_report(... dry_run=True)` or mark the generic orphan preview as unsafe/advisory.
4. If a public `cleanup_orphans` apply method remains, document it as low-level and ensure no ops command calls it without GC invariants.
5. Add a dedicated race fixture from `polylogue-8jg9.2`: disk blob exists, `pending_blob_refs` lease exists, no final reference row yet, doctor cleanup apply runs; file must survive.

## Test plan

Tests:
- pending-lease blob survives ops-doctor cleanup apply.
- old unleased orphan is deleted by safe GC after minimum-age/generation conditions.
- dry-run reports lease-skipped or not-deletable status.
- a stale caller-supplied orphan set cannot delete a blob that gained a reference between detection and apply.
- direct low-level `BlobStore.cleanup_orphans` tests may remain, but ops/doctor coverage must exercise the safe wrapper.

## Verification command / proof

`devtools test tests/unit/storage/test_blob_store.py tests/unit/storage/test_blob_gc.py tests/unit/storage/test_blob_repair.py -k 'orphan or lease or gc or cleanup'`

## Pitfalls

Do not reimplement half of GC inside blob_store. The store cannot know pending leases without DB context. The fix is route consolidation, not local unlink cleverness.

## Files/functions to inspect or touch

- `polylogue/storage/blob_store.py:352-409`
- `polylogue/storage/blob_repair.py:92+`
- `polylogue/storage/blob_gc.py:163`
- `polylogue/storage/blob_gc.py:355+`
- `polylogue/operations/*doctor*`
