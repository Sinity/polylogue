# 20. polylogue-83u.6 — Run read-only attachment acquisition census by origin/status/bytes

Priority: **P2**  
Lane: **attachment-integrity**  
Readiness: **ready-now / read-only artifact**

Depends on packet(s): polylogue-83u.4

## Why this is urgent / critical-path

Before claiming attachments are preserved or deciding which acquisition work matters, the project needs a byte-backed census grouped by origin and acquisition class.

## Static diagnosis / likely mechanism

The bead already gives a strong design: read source.db/index.db/blob store read-only; group by origin and acquisition_status; reconcile against blob-reference-debt diagnostics. Baseline numbers in the bead: 7,226 attachment rows, 958 acquired with non-null hash and 0 missing acquired blobs, 6,268 unfetched with NULL hash.

## Implementation plan

Implementation shape:
1. Add a devtools or ops diagnostic command that opens archive DBs using SQLite `mode=ro`.
2. Resolve blob paths through `BlobStore` without mutating.
3. Group rows by `(origin, acquisition_status)` and emit counts, declared byte sum, acquired blob count, bytes on disk, unfetched, unavailable, missing acquired blob refs, top source_ref classes, and bounded samples.
4. Write JSON + Markdown under `.agent/scratch/research/` or a demo shelf.
5. Store baseline now, then rerun after 83u.2/83u.3 to measure delta.

## Test plan

Tests:
- command refuses or is proven not to open write connections against a fixture/live path.
- fixture totals reconcile with blob-reference-debt primitive.
- missing acquired blob and unfetched NULL hash are separated.
- sample list is bounded.

## Verification command / proof

`devtools test tests/unit/operations/test_attachment_census*.py -k 'read_only or acquisition or census'` and one read-only run against a copy/live archive as operator evidence.

## Pitfalls

This packet should not mutate the archive. It is evidence acquisition about evidence acquisition.

## Files/functions to inspect or touch

- `new diagnostics command`
- `polylogue/storage/blob_store.py`
- `polylogue/storage/blob_integrity.py`
- `polylogue/operations/archive_debt.py`
- `index.db attachments read path`
